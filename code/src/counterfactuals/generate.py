"""MinCostCFE search: cost-ordered Dijkstra over feasible edits to a context graph.

Loads the KG and the HNSW node/edge indexes once at import time, then drives a
priority-queue search whose extracted-state order matches non-decreasing edit
cost. Per local.tex sec. 1.5, six edit ops are supported (delete/replace/add
for nodes and edges), filtered by F1-F3.

CLI knobs (see parse_args): cost & LLM-call budgets, allowed ops, case filter,
F1 mode (type-only / strict-label / off), add mode (expand / retrieve / both),
replace mode (atomic / decomposed = del+add), Pivotal-Star Probe toggle,
output-suffix, and judge-against (original / ground_truth).
"""

from datetime import datetime
from src.query import *
from src.retrieve import *
from src.parser import *
from src.llm_judge import judge_response
from src.counterfactuals.edit_costs import (
    delete_edge_cost, delete_node_cost,
    delete_edge_uc, delete_node_uc,
    replace_edge_cost, replace_node_cost,
    replace_edge_uc, replace_node_uc,
    add_edge_cost, add_node_cost,
    add_edge_cost_for, add_node_cost_for,
    add_edge_uc, add_node_uc,
)
from src.counterfactuals.perturbations import (
    delete_node, delete_edge,
    replace_node, replace_edge,
    add_node, add_edge,
)
from src.counterfactuals.feasibility_constraints import check_f1, check_f2, check_f3
from src.counterfactuals.utils import compute_answer_similarity, cosine_similarity_norm
from src.embeddings.utils import load_index
from collections import defaultdict
from src.embeddings.query import (
    find_most_similar_node, find_most_distant_node,
    find_most_similar_edge, find_most_distant_edge,
    DIM, build_lookup, get_embedding, build_edge_lookup,
)
from src.embeddings.query import query as embedding_query

### Explanation Stability/Consistency
from src.counterfactuals.robustness import graph_to_context_shuffled

import argparse
import heapq
import networkx as nx
import asyncio
import itertools
import os


### Setup ###

def create_type_index(G: nx.Graph):
    """I_T: type t -> list of nodes of type t in G."""
    type_index = defaultdict(list)
    for node, data in G.nodes(data=True):
        node_type = data.get("entity_type")
        type_index[node_type].append(node)
    return type_index


def create_schema_index(G: nx.Graph):
    """I_S: (src_type, tgt_type) -> set of labels seen in G."""
    schema = defaultdict(set)
    for u, v, data in G.edges(data=True):
        src_type = G.nodes[u].get("entity_type")
        tgt_type = G.nodes[v].get("entity_type")
        label = data.get("description", "") or data.get("keywords", "")
        if src_type is None or tgt_type is None:
            continue
        schema[(src_type, tgt_type)].add(label)
    return dict(schema)


def create_adjacency_index(G: nx.Graph):
    """I_A: v -> list of (label, neighbor_type, neighbor) for incident edges in G."""
    adj = defaultdict(list)
    for u, v, data in G.edges(data=True):
        label = data.get("description", "") or data.get("keywords", "")
        u_type = G.nodes[u].get("entity_type")
        v_type = G.nodes[v].get("entity_type")
        adj[u].append((label, v_type, v))
        adj[v].append((label, u_type, u))
    return dict(adj)


counter = itertools.count()

dataset = "synthetic"  ### "hotpotqa" or "synthetic"

G = nx.read_graphml(f"KGs/lightrag/{dataset}/graph_chunk_entity_relation.graphml")

type_index = create_type_index(G)
schema_index = create_schema_index(G)
adjacency_index = create_adjacency_index(G)

# Node setup
node_index_prefix = f"src/embeddings/{dataset}/node_index"
node_index, node_records, node_embeddings = load_index(node_index_prefix, DIM, 2000)
node_lookup = build_lookup(node_records)

# Edge setup
edge_index_prefix = f"src/embeddings/{dataset}/edge_index"
edge_index, edge_records, edge_embeddings = load_index(edge_index_prefix, DIM, 2000)
edge_lookup = build_edge_lookup(edge_records)

################################################


### Replacement Index (Node/Edge)
## T -> F (we aim at worsening the response)
## F -> T (we aim at fixing the response)

def create_node_replacement_index(nodes, context_graph, flip_direction="tf"):
    node_replacement_index = {}
    for node in nodes:
        data = G.nodes[node] if node in G.nodes else context_graph.nodes[node]
        node_type = data.get("entity_type")

        if not node_type:
            print(f"Skipping {node}: no available entity_type found!")
            continue

        if flip_direction == "tf":
            most_distant = find_most_distant_node(node, node_type, node_embeddings, node_lookup, type_index)
            if most_distant is None:
                print(f"Skipping {node}: no dissimilar node found!")
                continue
            node_replacement_index[node] = most_distant

        elif flip_direction == "ft":
            most_similar = find_most_similar_node(node, node_type, node_embeddings, node_lookup, type_index)
            if most_similar is None:
                print(f"Skipping {node}: no similar node found!")
                continue
            node_replacement_index[node] = most_similar

    return node_replacement_index


def create_edge_replacement_index(edges, context_graph, flip_direction="tf"):
    edge_replacement_index = {}
    for edge in edges:
        data = G.edges[edge] if edge in G.edges else context_graph.edges[edge]

        if flip_direction == "tf":
            most_distant = find_most_distant_edge(edge, edge_embeddings, edge_lookup)
            if most_distant is None:
                print(f"Skipping {edge}: no dissimilar edge found!")
                continue
            edge_replacement_index[edge] = most_distant

        elif flip_direction == "ft":
            most_similar = find_most_similar_edge(edge, edge_embeddings, edge_lookup)
            if most_similar is None:
                print(f"Skipping {edge}: no similar edge found!")
                continue
            edge_replacement_index[edge] = most_similar

    return edge_replacement_index

################################################

### Similarity Index (Node/Edge)

def create_node_similarity_index(nodes, query_embedding):
    node_similarity_index = {}
    for node in nodes:
        try:
            node_embedding = get_embedding(node_embeddings, node_lookup, node)
        except Exception:
            node_embedding = None
        if node_embedding is not None:
            similarity = cosine_similarity_norm(query_embedding, node_embedding)
        else:
            similarity = 0.0
        node_similarity_index[node] = similarity
    return node_similarity_index


async def create_edge_similarity_index(edge_labels, query_embedding):
    edge_similarity_index = {}
    for (u, v), label in edge_labels.items():
        if not label:
            edge_similarity_index[(u, v)] = 0.0
            continue
        edge_embedding = (await sentence_transformer_embed([label]))[0]
        similarity = cosine_similarity_norm(query_embedding, edge_embedding)
        edge_similarity_index[(u, v)] = similarity
    return edge_similarity_index

################################################


def _state_key(cg: nx.Graph):
    """Cache key spans both nodes and edges so isolated-node states don't collide."""
    return (
        frozenset(cg.nodes()),
        frozenset(
            (u, v, cg.edges[u, v].get("description", ""))
            for u, v in cg.edges()
        ),
    )


def _node_emb(node):
    try:
        return get_embedding(node_embeddings, node_lookup, node)
    except Exception:
        return None


def _edge_emb_by_key(edge):
    try:
        return get_embedding(edge_embeddings, edge_lookup, edge)
    except Exception:
        return None


################################################
### Pivotal-Star Probe (heuristic)

async def pivotal_star_probe(rag, question, context_graph, original_answer,
                             node_similarity_index, unit_cost: bool,
                             max_pivots: int, state_cache: set,
                             judge_against: str = "original",
                             ground_truth: str = ""):
    """See local.tex sec. 1.6. Probes top-K query-similar pivots eagerly,
    returns (best_sigma, best_cost, blacklist, llm_calls).

    Under the relaxed F3 (`check_f3` is a no-op), every pivot is eligible —
    bridges and articulation points included — and the singleton-sweep charge
    is already absorbed by `delete_node_cost`/`delete_edge_cost`. No
    cut-vertex / bridge precomputation is needed here."""
    cost_fn = delete_node_uc if unit_cost else delete_node_cost
    edge_cost_fn = delete_edge_uc if unit_cost else delete_edge_cost

    undirected = context_graph.to_undirected()

    pivots = sorted(
        [n for n in context_graph.nodes],
        key=lambda n: -node_similarity_index.get(n, 0.0),
    )[:max_pivots]

    best_sigma = None
    best_cost = float("inf")
    blacklist = set()
    llm_calls = 0

    async def _probe_state(cg, ops, cost):
        nonlocal llm_calls, best_sigma, best_cost
        state = _state_key(cg)
        if state in state_cache:
            return False
        state_cache.add(state)
        cg_context = graph_to_context(cg)
        new_response = await query(rag, cg_context, question)
        target = ground_truth if judge_against == "ground_truth" else original_answer
        score = await judge_response(question, new_response, target)
        llm_calls += 1
        # For "original": flip means score == 0 (differs from original)
        # For "ground_truth": flip means score == 1 (matches ground truth)
        flipped = (score == 1) if judge_against == "ground_truth" else (score == 0)
        print(f"[PSP] Cost {cost:.2f} | flipped={flipped} | ops={ops}")
        if flipped and cost < best_cost:
            best_sigma = ops
            best_cost = cost
        return flipped

    for v in pivots:
        incident_edges = list(context_graph.in_edges(v)) + list(context_graph.out_edges(v)) if context_graph.is_directed() else list(context_graph.edges(v))
        singletons = [n for n in undirected.neighbors(v) if undirected.degree(n) == 1]

        cluster_cg = delete_node(context_graph, v)
        cluster_cost = cost_fn(context_graph, v)

        flipped = await _probe_state(cluster_cg, [("delete_node", v)], cluster_cost)

        if flipped:
            for e in incident_edges:
                sub_cost = edge_cost_fn(context_graph, e)
                if sub_cost >= best_cost:
                    continue
                sub_cg = delete_edge(context_graph, e)
                await _probe_state(sub_cg, [("delete_edge", e)], sub_cost)

            for n in singletons:
                sub_cost = cost_fn(context_graph, n)
                if sub_cost >= best_cost:
                    continue
                sub_cg = delete_node(context_graph, n)
                await _probe_state(sub_cg, [("delete_node", n)], sub_cost)
        else:
            blacklist.add(("delete_node", v))
            for e in incident_edges:
                blacklist.add(("delete_edge", e))
            for n in singletons:
                blacklist.add(("delete_node", n))

    return best_sigma, best_cost, blacklist, llm_calls


################################################
### Pivotal-Frontier Probe (heuristic for F→T additions)

async def pivotal_frontier_probe(
    rag, question, context_graph, original_answer,
    node_similarity_index, query_embedding,
    add_cost_mode: str, mix_alpha: float,
    max_frontiers: int, state_cache: set,
    judge_against: str = "original", ground_truth: str = "",
):
    """Addition-side analog of pivotal_star_probe. For F→T flips we eagerly
    probe the top-K candidates ranked by

        pfp_score(v') = α · rel_q(v') + (1 − α) · prox_C(v')

    where α = mix_alpha. Each candidate v' is attached via the cheapest grounded
    edge e' to V_C; we probe add_n(v', e'). On flip we keep the cheapest
    attaching edge as the candidate σ*. NO pruning — addition monotonicity
    goes the wrong way (a sub-edit may flip even when the full add does not),
    so we cannot safely blacklist anything. Returns
    (best_sigma, best_cost, llm_calls).
    """
    existing_nodes = set(context_graph.nodes)
    existing_edges = set(context_graph.edges())

    # Candidate pool = neighbors of V_C (1-hop frontier in G), capped to a manageable size.
    frontier = set()
    for v in existing_nodes:
        if v in G.nodes:
            frontier.update(G.neighbors(v))
    frontier -= existing_nodes
    if not frontier:
        return None, float("inf"), 0

    def _prox_c(v_prime):
        v_emb = get_embedding(node_embeddings, node_lookup, v_prime)
        if v_emb is None:
            return 0.0
        best = 0.0
        for v in existing_nodes:
            v_in_cg_emb = get_embedding(node_embeddings, node_lookup, v)
            if v_in_cg_emb is None:
                continue
            sim = cosine_similarity_norm(v_in_cg_emb, v_emb)
            if sim > best:
                best = sim
        return best  # ∈ [-1, 1]; high = close to some V_C node

    scored = []
    for v_prime in frontier:
        rel_q = node_similarity_index.get(v_prime, 0.0)
        prox_c = _prox_c(v_prime)
        score = mix_alpha * rel_q + (1.0 - mix_alpha) * prox_c
        scored.append((score, v_prime))
    scored.sort(reverse=True)
    top_k = [v for _, v in scored[:max_frontiers]]

    best_sigma = None
    best_cost = float("inf")
    llm_calls = 0

    async def _probe_state(cg_perturbed, ops, op_cost):
        nonlocal llm_calls, best_sigma, best_cost
        state = _state_key(cg_perturbed)
        if state in state_cache:
            return False
        state_cache.add(state)
        cg_context = graph_to_context(cg_perturbed)
        new_response = await query(rag, cg_context, question)
        target = ground_truth if judge_against == "ground_truth" else original_answer
        score = await judge_response(question, new_response, target)
        llm_calls += 1
        flipped = (score == 1) if judge_against == "ground_truth" else (score == 0)
        print(f"[PFP] Cost {op_cost:.2f} | flipped={flipped} | ops={ops}")
        if flipped and op_cost < best_cost:
            best_sigma = ops
            best_cost = op_cost
        return flipped

    for v_prime in top_k:
        # All grounded attaching edges between v' and V_C in G.
        attach_edges = []
        for v in existing_nodes:
            if (v, v_prime) in edge_lookup and (v, v_prime) not in existing_edges:
                attach_edges.append((v, v_prime))
            if (v_prime, v) in edge_lookup and (v_prime, v) not in existing_edges:
                attach_edges.append((v_prime, v))
        if not attach_edges:
            continue

        # Rank attaching edges by current add-cost-mode; probe cheapest first.
        def _attach_cost(e):
            return add_node_cost_for(
                add_cost_mode, context_graph,
                node_embeddings, node_lookup, edge_embeddings, edge_lookup,
                v_prime, e, query_embedding=query_embedding, alpha=mix_alpha,
            )
        attach_edges.sort(key=_attach_cost)

        # Cheapest attach: probe it; if flips, also try a couple of alternates
        # to refine the incumbent (no pruning either way).
        for idx, e in enumerate(attach_edges):
            cg_perturbed = add_node(context_graph, v_prime, **G.nodes[v_prime])
            cg_perturbed = add_edge(cg_perturbed, e, **G.edges[e])
            ops = [("add_node", v_prime), ("add_edge", e)]
            op_cost = _attach_cost(e)
            if op_cost >= best_cost:
                # No room to improve the incumbent — stop refining this candidate.
                break
            flipped = await _probe_state(cg_perturbed, ops, op_cost)
            if flipped and idx == 0:
                # Probe the next 1–2 alternate attachments only when the cheapest flipped;
                # this is the refinement step (still no pruning).
                continue
            if not flipped and idx == 0:
                # Even the cheapest attach didn't flip; don't waste calls on dearer ones.
                break

    return best_sigma, best_cost, llm_calls


################################################

async def find_counterfactuals(
    rag,
    question: str,
    context,
    max_cost=3,
    max_llm_calls=100,
    max_sparsity=None,
    unit_cost: bool = False,
    current_ops: list = None,
    use_pivotal_probe: bool = False,
    max_pivots: int = 3,
    suffix: str = "",
    f1_mode: str = "type-only",
    add_mode: str = "both",
    add_cost_mode: str = "context",
    mix_alpha: float = 0.5,
    add_rank: str = "sim",
    use_pivotal_frontier: bool = False,
    max_frontiers: int = 5,
    replace_mode: str = "atomic",
    judge_against: str = "original",
    ground_truth: str = "",
):
    if current_ops is None:
        current_ops = ["delete_node", "delete_edge", "replace_node",
                       "replace_edge", "add_node", "add_edge"]

    query_embedding = (await sentence_transformer_embed([question]))[0]
    original_answer = await query(rag, context, question)

    ### Lightrag specific
    parsed_subgraph = parse_context(context)
    context_graph = parse_graph(parsed_subgraph)
    #####################

    context_graph_nodes = set(context_graph.nodes)
    context_graph_edges = set(context_graph.edges())

    # Replacement / similarity indexes built over G (main's choice) so that
    # add_node/add_edge candidates outside V_C also have similarity scores.
    edge_labels = {(u, v): data.get("description", "") for u, v, data in G.edges(data=True)}

    node_replacement_index = create_node_replacement_index(context_graph_nodes, G, flip_direction="tf")
    edge_replacement_index = create_edge_replacement_index(context_graph_edges, G, flip_direction="tf")

    node_similarity_index = create_node_similarity_index(set(G.nodes), query_embedding)
    edge_similarity_index = await create_edge_similarity_index(edge_labels, query_embedding)

    llm_calls = 0
    state_cache = set()

    ### Pivotal-Star Probe (deletions, T→F) ###
    best_probe_sigma = None
    best_probe_cost = float("inf")
    blacklist = set()
    if use_pivotal_probe:
        best_probe_sigma, best_probe_cost, blacklist, probe_calls = await pivotal_star_probe(
            rag=rag,
            question=question,
            context_graph=context_graph,
            original_answer=original_answer,
            node_similarity_index=node_similarity_index,
            unit_cost=unit_cost,
            max_pivots=max_pivots,
            state_cache=state_cache,
            judge_against=judge_against,
            ground_truth=ground_truth,
        )
        llm_calls += probe_calls
        if best_probe_cost < max_cost:
            max_cost = best_probe_cost

    ### Pivotal-Frontier Probe (additions, F→T) ###
    if use_pivotal_frontier and ("add_node" in current_ops or "add_edge" in current_ops):
        pfp_sigma, pfp_cost, pfp_calls = await pivotal_frontier_probe(
            rag=rag,
            question=question,
            context_graph=context_graph,
            original_answer=original_answer,
            node_similarity_index=node_similarity_index,
            query_embedding=query_embedding,
            add_cost_mode=add_cost_mode,
            mix_alpha=mix_alpha,
            max_frontiers=max_frontiers,
            state_cache=state_cache,
            judge_against=judge_against,
            ground_truth=ground_truth,
        )
        llm_calls += pfp_calls
        if pfp_sigma is not None and pfp_cost < best_probe_cost:
            best_probe_sigma = pfp_sigma
            best_probe_cost = pfp_cost
        if pfp_sigma is not None and pfp_cost < max_cost:
            max_cost = pfp_cost

    Q = []
    heapq.heappush(Q, (0, 0.0, next(counter), (context_graph, [])))

    explored_nodes = set()

    while Q:
        cost, _, _, (cg, ops) = heapq.heappop(Q)

        if cost > max_cost:
            print(f"Max cost {max_cost} exceeded (current cost: {cost:.4f}). Stopping search.")
            break
        elif llm_calls > max_llm_calls:
            print(f"Max LLM calls {max_llm_calls} exceeded. Stopping search.")
            break

        state = _state_key(cg)
        if state in state_cache:
            continue
        state_cache.add(state)

        if len(ops) > 0:
            cg_context = graph_to_context(cg)
            new_response = await query(rag, cg_context, question)

            target = ground_truth if judge_against == "ground_truth" else original_answer
            score = await judge_response(question, new_response, target)
            llm_calls += 1

            flipped = (score == 1) if judge_against == "ground_truth" else (score == 0)

            print(f"Cost: {cost:.4f} | New response: {new_response} | Target: {target}")

            if flipped:
                print(f"Counterfactual Operations: {ops}")

                answer_similarity = await compute_answer_similarity(
                    target, new_response
                )
                print(f"Answer similarity (target vs perturbed): {answer_similarity:.4f}")

                save_operations_to_json(
                    ops=ops,
                    question=question,
                    original_answer=original_answer,
                    perturbed_answer=new_response,
                    answer_similarity=answer_similarity,
                    original_subgraph=parsed_subgraph,
                    perturbed_subgraph=graph_to_subgraph(cg),
                    found=True,
                    cost=cost,
                    llm_calls=llm_calls,
                    current_ops=current_ops,
                    suffix=suffix,
                )
                return ops

        expand(
            Q, (cost, cg, ops),
            node_replacement_index=node_replacement_index,
            edge_replacement_index=edge_replacement_index,
            node_similarity_index=node_similarity_index,
            edge_similarity_index=edge_similarity_index,
            unit_cost=unit_cost,
            current_ops=current_ops,
            original_nodes=context_graph_nodes,
            original_edges=context_graph_edges,
            explored_nodes=explored_nodes,
            query_embedding=query_embedding,
            blacklist=blacklist,
            f1_mode=f1_mode,
            add_mode=add_mode,
            add_cost_mode=add_cost_mode,
            mix_alpha=mix_alpha,
            add_rank=add_rank,
            replace_mode=replace_mode,
        )

    # Dijkstra ended without finding a flip — fall back to probe winner if present
    # (Pivotal-Star Probe for deletions or Pivotal-Frontier Probe for additions).
    if best_probe_sigma is not None:
        print(f"Probe winner: {best_probe_sigma} (cost {best_probe_cost})")
        save_operations_to_json(
            ops=best_probe_sigma,
            question=question,
            original_answer=original_answer,
            perturbed_answer=None,
            answer_similarity=0.0,
            original_subgraph=parsed_subgraph,
            perturbed_subgraph=None,
            found=True,
            cost=best_probe_cost,
            llm_calls=llm_calls,
            current_ops=current_ops,
            suffix=suffix,
        )
        return best_probe_sigma

    print("Could not find feasible counterfactual explanations.")

    save_operations_to_json(
        ops=[],
        question=question,
        original_answer=original_answer,
        perturbed_answer=None,
        answer_similarity=0.0,
        original_subgraph=parsed_subgraph,
        perturbed_subgraph=None,
        found=False,
        cost=0.0,
        llm_calls=llm_calls,
        current_ops=current_ops,
        suffix=suffix,
    )
    return None


def expand(
    Q,
    heap_element,
    node_replacement_index,
    edge_replacement_index,
    node_similarity_index,
    edge_similarity_index,
    unit_cost: bool = False,
    current_ops: list = None,
    original_nodes: set = None,
    original_edges: set = None,
    explored_nodes: set = None,
    query_embedding=None,
    blacklist: set = None,
    f1_mode: str = "type-only",
    add_mode: str = "both",
    add_cost_mode: str = "context",
    mix_alpha: float = 0.5,
    add_rank: str = "sim",
    replace_mode: str = "atomic",
):
    # When --unit-cost is set, force add_cost_mode to "unit" for consistency.
    # The CLI flags are otherwise orthogonal — unit overrides any other add-cost.
    if unit_cost:
        add_cost_mode = "unit"
    if current_ops is None:
        current_ops = ["delete_node", "delete_edge", "replace_node",
                       "replace_edge", "add_node", "add_edge"]
    if original_nodes is None:
        original_nodes = set()
    if original_edges is None:
        original_edges = set()
    if explored_nodes is None:
        explored_nodes = set()
    if blacklist is None:
        blacklist = set()

    cg: nx.DiGraph
    cost, cg, ops = heap_element

    # Under relaxed F3 (`check_f3` is a no-op), articulation_points / bridges
    # are no longer consulted to gate deletions. The cost functions absorb the
    # singleton sweep via their +1-per-stranded-neighbour term.

    ##### Delete Node #####
    if "delete_node" in current_ops:
        for node in list(cg.nodes):
            if ("delete_node", node) in blacklist:
                continue
            # Only consider nodes that were in the original context (mirrors main's guard
            # so we don't repeatedly delete nodes the search just added).
            if node not in original_nodes:
                continue

            perturbed_cg = delete_node(cg, node)
            perturbation_cost = delete_node_uc(cg, node) if unit_cost else delete_node_cost(cg, node)
            new_ops = ops + [("delete_node", node)]
            similarity = node_similarity_index.get(node, 0.0)
            heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

    ##### Delete Edge #####
    if "delete_edge" in current_ops:
        for edge in list(cg.edges):
            if ("delete_edge", edge) in blacklist or ("delete_edge", (edge[1], edge[0])) in blacklist:
                continue
            if edge not in original_edges:
                continue

            perturbed_cg = delete_edge(cg, edge)
            perturbation_cost = delete_edge_uc(cg, edge) if unit_cost else delete_edge_cost(cg, edge)
            new_ops = ops + [("delete_edge", edge)]
            similarity = edge_similarity_index.get(edge, 0.0)
            heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

    ##### Add Node #####
    if "add_node" in current_ops:
        existing_nodes = set(cg.nodes)
        existing_edges = set(cg.edges())
        candidate_nodes_for_expansion = existing_nodes - explored_nodes

        do_retrieve = add_mode in ("retrieve", "both")
        do_expand = add_mode in ("expand", "both")

        # Retrieve mode: when no expand candidates left (or retrieve-only), pull
        # top-k by query similarity and attach them as a new component anchor.
        retrieve_triggered = do_retrieve and (
            add_mode == "retrieve" or not candidate_nodes_for_expansion
        )
        if retrieve_triggered:
            relevant_nodes = embedding_query(node_index, node_records, query_embedding, k=10)
            for nrec in relevant_nodes:
                node_name = nrec.get("name")
                if not node_name or node_name in existing_nodes:
                    continue
                perturbed_cg = add_node(cg, node_name, **G.nodes[node_name])
                similarity = node_similarity_index.get(node_name, 0.0)
                new_ops = ops + [("add_node", node_name)]

                # Track the first attaching edge added this iteration; pass it as
                # the canonical e' to the cost dispatcher (single-edge semantics).
                # Bug fix (port of 270c84e): every implicit neighbour node addition
                # also records an ("add_node", neighbor) op so downstream analysis
                # sees the full perturbation set.
                first_attach_edge = None
                neighbors = list(G.neighbors(node_name))
                for neighbor in neighbors:
                    if neighbor not in existing_nodes:
                        perturbed_cg = add_node(perturbed_cg, neighbor, **G.nodes[neighbor])
                        new_ops = new_ops + [("add_node", neighbor)]
                        if (node_name, neighbor) in edge_lookup and (node_name, neighbor) not in existing_edges:
                            perturbed_cg = add_edge(perturbed_cg, (node_name, neighbor), **G.edges[node_name, neighbor])
                            new_ops = new_ops + [("add_edge", (node_name, neighbor))]
                            if first_attach_edge is None:
                                first_attach_edge = (node_name, neighbor)
                        if (neighbor, node_name) in edge_lookup and (neighbor, node_name) not in existing_edges:
                            perturbed_cg = add_edge(perturbed_cg, (neighbor, node_name), **G.edges[neighbor, node_name])
                            new_ops = new_ops + [("add_edge", (neighbor, node_name))]
                            if first_attach_edge is None:
                                first_attach_edge = (neighbor, node_name)

                perturbation_cost = add_node_cost_for(
                    add_cost_mode, perturbed_cg,
                    node_embeddings, node_lookup, edge_embeddings, edge_lookup,
                    node_name, first_attach_edge,
                    query_embedding=query_embedding, alpha=mix_alpha,
                )

                heap_secondary = len(new_ops) if add_rank == "num-ops" else -similarity
                heapq.heappush(Q, (cost + perturbation_cost, heap_secondary, next(counter), (perturbed_cg, new_ops)))
                break  # one retrieve seed per expansion is enough

        if do_expand:
            for node in candidate_nodes_for_expansion:
                neighbors = list(G.neighbors(node))
                similarity = node_similarity_index.get(node, 0.0)

                for neighbor in neighbors:
                    if neighbor in existing_nodes:
                        continue

                    # F1 type-pair check on the edge that will attach `neighbor`.
                    n_type = G.nodes[node].get("entity_type")
                    nb_type = G.nodes[neighbor].get("entity_type")
                    if (node, neighbor) in edge_lookup:
                        e_label = G.edges[node, neighbor].get("description", "") or G.edges[node, neighbor].get("keywords", "")
                        if not check_f1(schema_index, n_type, e_label, nb_type, mode=f1_mode):
                            continue
                    elif (neighbor, node) in edge_lookup:
                        e_label = G.edges[neighbor, node].get("description", "") or G.edges[neighbor, node].get("keywords", "")
                        if not check_f1(schema_index, nb_type, e_label, n_type, mode=f1_mode):
                            continue
                    else:
                        continue  # no edge in G connects them, skip

                    perturbed_cg = add_node(cg, neighbor, **G.nodes[neighbor])
                    new_ops = ops + [("add_node", neighbor)]
                    first_attach_edge = None
                    if (node, neighbor) in edge_lookup and (node, neighbor) not in existing_edges:
                        perturbed_cg = add_edge(perturbed_cg, (node, neighbor), **G.edges[node, neighbor])
                        new_ops = new_ops + [("add_edge", (node, neighbor))]
                        first_attach_edge = (node, neighbor)
                    if (neighbor, node) in edge_lookup and (neighbor, node) not in existing_edges:
                        perturbed_cg = add_edge(perturbed_cg, (neighbor, node), **G.edges[neighbor, node])
                        new_ops = new_ops + [("add_edge", (neighbor, node))]
                        if first_attach_edge is None:
                            first_attach_edge = (neighbor, node)

                    perturbation_cost = add_node_cost_for(
                        add_cost_mode, perturbed_cg,
                        node_embeddings, node_lookup, edge_embeddings, edge_lookup,
                        neighbor, first_attach_edge,
                        query_embedding=query_embedding, alpha=mix_alpha,
                    )

                    heap_secondary = len(new_ops) if add_rank == "num-ops" else -similarity
                    heapq.heappush(Q, (cost + perturbation_cost, heap_secondary, next(counter), (perturbed_cg, new_ops)))

                explored_nodes.add(node)

    ##### Add Edge #####
    if "add_edge" in current_ops:
        existing_edges = set(cg.edges())
        existing_nodes = set(cg.nodes)

        for node in existing_nodes:
            available_edges = set(G.edges(node))
            for edge in available_edges:
                node1, node2 = edge
                similarity = edge_similarity_index.get(edge, 0.0)

                # Only proceed if at least one endpoint is in V_C (main's "expand" semantics).
                if node1 not in existing_nodes and node2 not in existing_nodes:
                    continue

                # Both directions are checked separately; F1 applied per direction.
                if (node1, node2) in edge_lookup and (node1, node2) not in existing_edges:
                    n1_type = G.nodes[node1].get("entity_type")
                    n2_type = G.nodes[node2].get("entity_type")
                    label = G.edges[node1, node2].get("description", "") or G.edges[node1, node2].get("keywords", "")
                    if not check_f1(schema_index, n1_type, label, n2_type, mode=f1_mode):
                        continue
                    perturbed_cg = cg.copy()
                    new_ops = ops[:]
                    implicit_node_cost = 0
                    if node2 not in existing_nodes:
                        perturbed_cg.add_node(node2, **G.nodes[node2])
                        new_ops = new_ops + [("add_node", node2)]
                        implicit_node_cost = 1
                    perturbed_cg = add_edge(perturbed_cg, (node1, node2), **G.edges[node1, node2])
                    new_ops = new_ops + [("add_edge", (node1, node2))]

                    perturbation_cost = implicit_node_cost + add_edge_cost_for(
                        add_cost_mode, perturbed_cg, edge_embeddings, edge_lookup,
                        (node1, node2), query_embedding=query_embedding, alpha=mix_alpha,
                    )
                    heap_secondary = len(new_ops) if add_rank == "num-ops" else -similarity
                    heapq.heappush(Q, (cost + perturbation_cost, heap_secondary, next(counter), (perturbed_cg, new_ops)))

                elif (node2, node1) in edge_lookup and (node2, node1) not in existing_edges:
                    n1_type = G.nodes[node1].get("entity_type")
                    n2_type = G.nodes[node2].get("entity_type")
                    label = G.edges[node2, node1].get("description", "") or G.edges[node2, node1].get("keywords", "")
                    if not check_f1(schema_index, n2_type, label, n1_type, mode=f1_mode):
                        continue
                    perturbed_cg = cg.copy()
                    new_ops = ops[:]
                    implicit_node_cost = 0
                    if node1 not in existing_nodes:
                        perturbed_cg.add_node(node1, **G.nodes[node1])
                        new_ops = new_ops + [("add_node", node1)]
                        implicit_node_cost = 1
                    perturbed_cg = add_edge(perturbed_cg, (node2, node1), **G.edges[node2, node1])
                    new_ops = new_ops + [("add_edge", (node2, node1))]

                    perturbation_cost = implicit_node_cost + add_edge_cost_for(
                        add_cost_mode, perturbed_cg, edge_embeddings, edge_lookup,
                        (node2, node1), query_embedding=query_embedding, alpha=mix_alpha,
                    )
                    heap_secondary = len(new_ops) if add_rank == "num-ops" else -similarity
                    heapq.heappush(Q, (cost + perturbation_cost, heap_secondary, next(counter), (perturbed_cg, new_ops)))

    ##### Replace Node #####
    if "replace_node" in current_ops:
        if replace_mode == "decomposed":
            # Decomposed: enqueue a del_n followed by an add_n of the same target.
            # We push them as a single composite op so Dijkstra sees one heap entry
            # per replacement attempt, but the recorded ops list still contains
            # the del + add pair for downstream analysis.
            for node, _ in list(cg.nodes(data=True)):
                if node not in original_nodes:
                    continue
                node_replacement = node_replacement_index.get(node)
                if node_replacement is None:
                    continue
                current_replacement = node_replacement.get("name")
                if current_replacement is None or current_replacement not in G.nodes:
                    continue

                # Apply del then add
                step1 = delete_node(cg, node)
                step2 = add_node(step1, current_replacement, **G.nodes[current_replacement])
                # Restore the incident edges from G if they exist
                for nb in G.neighbors(current_replacement):
                    if nb in step2.nodes:
                        if (current_replacement, nb) in G.edges and (current_replacement, nb) not in step2.edges:
                            step2 = add_edge(step2, (current_replacement, nb), **G.edges[current_replacement, nb])
                        if (nb, current_replacement) in G.edges and (nb, current_replacement) not in step2.edges:
                            step2 = add_edge(step2, (nb, current_replacement), **G.edges[nb, current_replacement])

                del_cost = delete_node_uc(cg, node) if unit_cost else delete_node_cost(cg, node)
                add_cost = (
                    add_node_uc(step2)
                    if unit_cost
                    else add_node_cost(step2, node_embeddings, node_lookup, edge_embeddings, edge_lookup, current_replacement)
                )
                total_cost = del_cost + add_cost
                new_ops = ops + [("delete_node", node), ("add_node", current_replacement)]
                similarity = node_similarity_index.get(node, 0.0)
                heapq.heappush(Q, (cost + total_cost, -similarity, next(counter), (step2, new_ops)))

        else:  # atomic
            for node, _ in list(cg.nodes(data=True)):
                node_replacement = node_replacement_index.get(node)
                if node_replacement is None:
                    continue
                current_replacement = node_replacement.get("name")
                if current_replacement is None:
                    continue
                replacement_attr = G.nodes[current_replacement]

                # F1: enforce τ(v') = τ(v)
                if cg.nodes[node].get("entity_type") != replacement_attr.get("entity_type"):
                    continue

                perturbed_cg = replace_node(cg, node, current_replacement, **replacement_attr)

                if unit_cost:
                    perturbation_cost = replace_node_uc(cg, node)
                else:
                    old_emb = _node_emb(node)
                    new_emb = _node_emb(current_replacement)
                    if old_emb is None or new_emb is None:
                        sim = node_replacement.get("similarity", 0.0)
                        perturbation_cost = 1 + (1 - sim)
                        if cg.is_directed():
                            perturbation_cost += len(list(cg.in_edges(node))) + len(list(cg.out_edges(node)))
                        else:
                            perturbation_cost += len(list(cg.edges(node)))
                    else:
                        perturbation_cost = replace_node_cost(old_emb, new_emb, cg, node)

                new_ops = ops + [("replace_node", (node, current_replacement))]
                similarity = node_similarity_index.get(node, 0.0)
                heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

    ##### Replace Edge #####
    if "replace_edge" in current_ops:
        if replace_mode == "decomposed":
            for edge in list(cg.edges):
                if edge not in original_edges:
                    continue
                edge_replacement = edge_replacement_index.get(edge)
                if edge_replacement is None:
                    continue
                current_replacement = edge_replacement.get("edge")
                if current_replacement is None or current_replacement not in G.edges:
                    continue

                step1 = delete_edge(cg, edge)
                # Add replacement edge — keep both endpoints if they exist; otherwise add them.
                u2, v2 = current_replacement
                step2 = step1.copy()
                if u2 not in step2.nodes:
                    step2.add_node(u2, **G.nodes[u2])
                if v2 not in step2.nodes:
                    step2.add_node(v2, **G.nodes[v2])
                step2 = add_edge(step2, (u2, v2), **G.edges[u2, v2])

                del_cost = delete_edge_uc(cg, edge) if unit_cost else delete_edge_cost(cg, edge)
                add_cost = add_edge_uc() if unit_cost else add_edge_cost(step2, edge_embeddings, edge_lookup, (u2, v2))
                total_cost = del_cost + add_cost
                new_ops = ops + [("delete_edge", edge), ("add_edge", (u2, v2))]
                similarity = edge_similarity_index.get(edge, 0.0)
                heapq.heappush(Q, (cost + total_cost, -similarity, next(counter), (step2, new_ops)))

        else:  # atomic
            for edge in list(cg.edges):
                edge_replacement = edge_replacement_index.get(edge)
                if edge_replacement is None:
                    continue
                current_replacement = edge_replacement.get("edge")
                if current_replacement is None:
                    continue
                replacement_attr = G.edges[current_replacement]

                # F1 on the new label given the existing endpoints.
                u, v = edge
                src_type = cg.nodes[u].get("entity_type")
                tgt_type = cg.nodes[v].get("entity_type")
                new_label = replacement_attr.get("description", "") or replacement_attr.get("keywords", "")
                if not check_f1(schema_index, src_type, new_label, tgt_type, mode=f1_mode):
                    continue

                perturbed_cg = replace_edge(cg, edge, current_replacement, **replacement_attr)

                if unit_cost:
                    perturbation_cost = replace_edge_uc()
                else:
                    old_emb = _edge_emb_by_key(edge)
                    new_emb = _edge_emb_by_key(current_replacement)
                    if old_emb is None or new_emb is None:
                        sim = edge_replacement.get("similarity", 0.0)
                        perturbation_cost = 1 + (1 - sim)
                    else:
                        perturbation_cost = replace_edge_cost(old_emb, new_emb)

                new_ops = ops + [("replace_edge", (edge, current_replacement))]
                similarity = edge_similarity_index.get(edge, 0.0)
                heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))


def save_operations_to_json(ops: list, question: str, original_answer: str, perturbed_answer: str,
                            answer_similarity: float, original_subgraph, perturbed_subgraph,
                            output_dir: str = "src/counterfactuals/robustness/stability",
                            filename: str = None, found: bool = True, cost: float = 0.0,
                            llm_calls: int = 0, current_ops: list = None, suffix: str = ""):
    if current_ops is None:
        current_ops = []

    if current_ops == ["delete_node", "delete_edge", "replace_node", "replace_edge"]:
        output_dir = f"{output_dir}_sem_all"
    elif current_ops == ["delete_node"]:
        output_dir = f"{output_dir}_delete_node"
    elif current_ops == ["delete_edge"]:
        output_dir = f"{output_dir}_delete_edge"
    elif current_ops == ["replace_node"]:
        output_dir = f"{output_dir}_replace_node"
    elif current_ops == ["replace_edge"]:
        output_dir = f"{output_dir}_replace_edge"
    elif current_ops == ["delete_node", "delete_edge"]:
        output_dir = f"{output_dir}/sem_delete_s_neither"
    elif current_ops == ["add_node"]:
        output_dir = f"{output_dir}_add_node"
    elif current_ops == ["add_edge"]:
        output_dir = f"{output_dir}_add_edge"
    elif current_ops == ["add_node", "add_edge"]:
        output_dir = f"{output_dir}_add_only"
    elif current_ops == ["add_node", "add_edge", "delete_node", "delete_edge"]:
        output_dir = f"{output_dir}_add_delete"
    else:
        output_dir = f"{output_dir}_other"

    if suffix:
        output_dir = f"{output_dir}{suffix}"

    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"counterfactual_{timestamp}.json"

    filepath = os.path.join(output_dir, filename)

    serialisable_ops = []
    for op in ops or []:
        if isinstance(op, tuple):
            serialisable_ops.append(list(op))
        else:
            serialisable_ops.append(op)

    try:
        cost_f = float(cost)
    except Exception:
        cost_f = None
    try:
        sim_f = round(float(answer_similarity), 6)
    except Exception:
        sim_f = None

    payload = {
        "question": question,
        "found": found,
        "num_operations": len(serialisable_ops),
        "operations": serialisable_ops,
        "cost": cost_f,
        "answers": {
            "original": original_answer,
            "perturbed": perturbed_answer,
            "similarity": sim_f,
        },
        "original_subgraph": subgraph_to_dict(original_subgraph),
        "perturbed_subgraph": subgraph_to_dict(perturbed_subgraph),
        "timestamp": datetime.now().isoformat(),
        "llm_calls": int(llm_calls) if llm_calls is not None else None,
    }

    def _json_default(o):
        try:
            import numpy as _np
            if isinstance(o, (_np.floating, _np.integer)):
                return o.item()
            if isinstance(o, _np.ndarray):
                return o.tolist()
        except Exception:
            pass
        if isinstance(o, (set, frozenset)):
            return list(o)
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

    print(f"Operations saved to: {filepath}")
    return filepath


def parse_args():
    p = argparse.ArgumentParser(description="Counterfactual search driver.")
    p.add_argument("--input", default="benchmark/results/comparison.json",
                   help="Path to comparison.json")
    p.add_argument("--case", choices=["all", "tf", "ft", "ff", "tt"], default="all",
                   help="Which entries to process by flip direction")
    p.add_argument("--max-cost", type=float, default=10.0)
    p.add_argument("--max-llm-calls", type=int, default=100)
    p.add_argument("--unit-cost", action="store_true",
                   help="Use unit costs instead of semantic 1+d_sem costs")
    p.add_argument("--ops", nargs="+",
                   default=["delete_node", "delete_edge", "replace_node",
                            "replace_edge", "add_node", "add_edge"])
    p.add_argument("--use-psp", action="store_true",
                   help="Enable Pivotal-Star Probe heuristic")
    p.add_argument("--max-pivots", type=int, default=3)
    p.add_argument("--suffix", default="",
                   help="Appended to output directory")
    p.add_argument("--retrieve-mode", default="hybrid")
    p.add_argument("--retrieve-top-k", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--f1-mode", choices=["type-only", "strict-label", "off"], default="type-only",
                   help="F1 schema check stringency")
    p.add_argument("--add-mode", choices=["expand", "retrieve", "both"], default="both",
                   help="Node-addition candidate pool: neighbors of V_C (expand), top-K query-similar nodes in G (retrieve), or both.")
    p.add_argument("--add-cost-mode", choices=["unit", "query", "context", "mix"], default="context",
                   help="Cost formula for additions: unit (paper-flat), query (distance to query), context (paper Eq. cost-add-n, distance to nearest CG element), or mix (convex blend).")
    p.add_argument("--mix-alpha", type=float, default=0.5,
                   help="α for --add-cost-mode mix: w_mix = α · w_query + (1−α) · w_context. Default 0.5.")
    p.add_argument("--add-rank", choices=["sim", "num-ops"], default="sim",
                   help="Heap secondary key for add candidates: similarity (default, kaoukis behaviour) or operation-count.")
    p.add_argument("--use-pfp", action="store_true",
                   help="Enable Pivotal-Frontier Probe heuristic (additions; analog of PSP for F→T flips)")
    p.add_argument("--max-frontiers", type=int, default=5,
                   help="Top-K candidates to probe before Dijkstra under --use-pfp")
    p.add_argument("--replace-mode", choices=["atomic", "decomposed"], default="atomic",
                   help="Run replacement as one atomic op (atomic) or as del+add (decomposed)")
    p.add_argument("--judge-against", choices=["original", "ground_truth"], default="original",
                   help="Compare new answer to original (flip = differ) or to ground truth (flip = match)")
    return p.parse_args()


async def main():
    args = parse_args()
    rag = await initialize_lightrag()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = 0
    for idx, r in data["results"].items():
        if args.limit is not None and processed >= args.limit:
            break
        case = r.get("case", "all")
        if args.case != "all" and case != args.case:
            continue

        question = r["question"]
        ground_truth = r.get("ground_truth", "")

        print(f"\n=== [{idx}] case={case} | {question} ===")

        context = await retrieve_subgraph(rag, query=question,
                                          mode=args.retrieve_mode,
                                          top_k=args.retrieve_top_k)
        await find_counterfactuals(
            rag, question, context=context,
            max_cost=args.max_cost,
            max_llm_calls=args.max_llm_calls,
            unit_cost=args.unit_cost,
            current_ops=args.ops,
            use_pivotal_probe=args.use_psp,
            max_pivots=args.max_pivots,
            suffix=args.suffix,
            f1_mode=args.f1_mode,
            add_mode=args.add_mode,
            add_cost_mode=args.add_cost_mode,
            mix_alpha=args.mix_alpha,
            add_rank=args.add_rank,
            use_pivotal_frontier=args.use_pfp,
            max_frontiers=args.max_frontiers,
            replace_mode=args.replace_mode,
            judge_against=args.judge_against,
            ground_truth=ground_truth,
        )
        processed += 1

    print(f"\nProcessed {processed} entries.")


if __name__ == "__main__":
    asyncio.run(main())
