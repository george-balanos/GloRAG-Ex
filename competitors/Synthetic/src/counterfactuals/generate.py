"""MinCostCFE search: cost-ordered Dijkstra over feasible edits to a context graph.

Loads the KG and the HNSW node/edge indexes once at import time, then drives a
priority-queue search whose extracted-state order matches non-decreasing edit
cost. Per local.tex sec. 1.5, six edit ops are supported (delete/replace/add
for nodes and edges), filtered by the feasibility constraints F1-F3.
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
from src.embeddings.query import find_most_similar_node, find_most_distant_node, find_most_similar_edge, find_most_distant_edge, DIM, build_lookup, get_embedding, build_edge_lookup

### Explanation Stability/Consistency
from src.counterfactuals.robustness import graph_to_context_shuffled

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
    """I_S: (src_type, tgt_type) -> set of valid labels in G."""
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

G = nx.read_graphml("KGs/synthetic/graph_chunk_entity_relation.graphml")

type_index = create_type_index(G)
schema_index = create_schema_index(G)
adjacency_index = create_adjacency_index(G)

# Node setup
node_index_prefix = "src/embeddings/node_index"
node_index, node_records, node_embeddings = load_index(node_index_prefix, DIM, 2000)
node_lookup = build_lookup(node_records)

# Edge setup
edge_index_prefix = "src/embeddings/edge_index"
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

async def find_counterfactuals(rag, question: str, context, max_cost=3, max_llm_calls=100, max_sparsity=None, unit_cost: bool=False, current_ops: list=["delete_node", "delete_edge", "replace_node", "replace_edge", "add_node", "add_edge"], use_pivotal_probe: bool=False, max_pivots: int=3, suffix: str=""):
    query_embedding = (await sentence_transformer_embed([question]))[0]
    original_answer = await query(rag, context, question)

    ### Lightrag specific
    parsed_subgraph = parse_context(context)
    context_graph = parse_graph(parsed_subgraph)
    #####################

    context_graph_nodes = set(context_graph.nodes)
    context_graph_edges = set(context_graph.edges())

    edge_labels = {(u, v): data.get("description", "") for u, v, data in context_graph.edges(data=True)}

    node_replacement_index = create_node_replacement_index(context_graph_nodes, context_graph, flip_direction="tf")
    edge_replacement_index = create_edge_replacement_index(context_graph_edges, context_graph, flip_direction="tf")

    node_similarity_index = create_node_similarity_index(context_graph_nodes, query_embedding)
    edge_similarity_index = await create_edge_similarity_index(edge_labels, query_embedding)

    llm_calls = 0
    state_cache = set()

    ### Pivotal-Star Probe (heuristic) ###
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
        )
        llm_calls += probe_calls
        if best_probe_cost < max_cost:
            max_cost = best_probe_cost  # tighten Dijkstra cutoff

    Q = []
    heapq.heappush(Q, (0, 0.0, next(counter), (context_graph, [])))

    while Q:
        cost, _, _, (cg, ops) = heapq.heappop(Q)

        if cost > max_cost:
            print(f"Max cost {max_cost} exceeded (current cost: {cost:.4f}). Stopping search.")
            break
        elif llm_calls > max_llm_calls:
            print(f"Max LLM calls {max_llm_calls} exceeded. Stopping search.")
            break

        ### Check state cache
        state = frozenset(
            (u, v, cg.edges[u, v].get("description", ""))
            for u, v in cg.edges
        )
        if state in state_cache:
            continue
        state_cache.add(state)

        if len(ops) > 0:
            cg_context = graph_to_context_shuffled(cg, shuffle_entities=False, shuffle_relations=False)

            new_response = await query(rag, cg_context, question)

            print(f"Cost: {cost} | New response: {new_response} | Original: {original_answer}")

            score = await judge_response(question, new_response, original_answer)

            llm_calls += 1

            if score == 0:
                print(f"Counterfactual Operations: {ops}")

                answer_similarity = await compute_answer_similarity(original_answer, new_response)
                print(f"Answer similarity (original vs perturbed): {answer_similarity:.4f}")

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
                    suffix=suffix
                )
                return ops

        expand(Q, (cost, cg, ops), node_replacement_index=node_replacement_index, edge_replacement_index=edge_replacement_index, node_similarity_index=node_similarity_index, edge_similarity_index=edge_similarity_index, unit_cost=unit_cost, current_ops=current_ops, query_embedding=query_embedding, blacklist=blacklist)

        print()

    if best_probe_sigma is not None:
        print(f"Pivotal-Star Probe winner: {best_probe_sigma} (cost {best_probe_cost})")
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
            suffix=suffix
        )
        return best_probe_sigma

    print(f"Could not find feasible counterfactual explanations.")

    save_operations_to_json(
        ops=[],
        question=question,
        original_answer=original_answer,
        perturbed_answer=None,
        answer_similarity=0.0,
        original_subgraph=parsed_subgraph,
        perturbed_subgraph=None,
        found=False,
        llm_calls=llm_calls,
        current_ops=current_ops,
        suffix=suffix
    )


def _node_emb(node):
    try:
        return get_embedding(node_embeddings, node_lookup, node)
    except Exception:
        return None


def _edge_emb(edge):
    try:
        return get_embedding(edge_embeddings, edge_lookup, edge)
    except Exception:
        return None


def expand(Q, heap_element, node_replacement_index, edge_replacement_index, node_similarity_index, edge_similarity_index, unit_cost: bool = False, current_ops: list=None, query_embedding=None, add_topk: int = 5, blacklist: set=None):
    if current_ops is None:
        current_ops = ["delete_node", "delete_edge", "replace_node", "replace_edge", "add_node", "add_edge"]
    if blacklist is None:
        blacklist = set()

    cost, cg, ops = heap_element

    undirected: nx.Graph = cg.to_undirected()
    cut_vertices = set(nx.articulation_points(undirected))
    cut_edges = set(nx.bridges(undirected))

    if "delete_node" in current_ops:
        for node in list(cg.nodes):
            if ("delete_node", node) in blacklist:
                continue
            if not check_f3(cand_node=node, cut_vertices=cut_vertices, undirected=undirected):
                continue

            perturbed_cg = delete_node(cg, node)

            if unit_cost:
                perturbation_cost = delete_node_uc(cg, node)
            else:
                perturbation_cost = delete_node_cost(cg, node)

            new_ops = ops + [("delete_node", node)]
            similarity = node_similarity_index.get(node, 0.0)
            heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

    if "delete_edge" in current_ops:
        for edge in list(cg.edges):
            if ("delete_edge", edge) in blacklist or ("delete_edge", (edge[1], edge[0])) in blacklist:
                continue
            if not check_f3(cand_edge=edge, bridges=cut_edges, undirected=undirected):
                continue

            perturbed_cg = delete_edge(cg, edge)

            if unit_cost:
                perturbation_cost = delete_edge_uc(cg, edge)
            else:
                perturbation_cost = delete_edge_cost(cg, edge)

            new_ops = ops + [("delete_edge", edge)]
            similarity = edge_similarity_index.get(edge, 0.0)
            heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

    if "replace_node" in current_ops:
        for node, _ in list(cg.nodes(data=True)):
            node_replacement = node_replacement_index.get(node)
            if node_replacement is None:
                continue

            current_replacement = node_replacement.get("name")
            if current_replacement is None:
                continue

            replacement_attr = G.nodes[current_replacement]

            old_type = cg.nodes[node].get("entity_type") or G.nodes[node].get("entity_type") if node in G.nodes else cg.nodes[node].get("entity_type")
            new_type = replacement_attr.get("entity_type")
            if old_type != new_type:
                continue  # F1: type must match for node replacement

            perturbed_cg = replace_node(cg, node, current_replacement, **replacement_attr)

            if unit_cost:
                perturbation_cost = replace_node_uc(cg, node)
            else:
                old_emb = _node_emb(node)
                new_emb = _node_emb(current_replacement)
                if old_emb is None or new_emb is None:
                    sim = node_replacement.get("similarity", 0.0)
                    d_sem = 1 - sim
                    perturbation_cost = 1 + len(list(cg.edges(node))) + d_sem
                else:
                    perturbation_cost = replace_node_cost(old_emb, new_emb, cg, node)

            new_ops = ops + [("replace_node", (node, current_replacement))]
            similarity = node_similarity_index.get(node, 0.0)
            heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

    if "replace_edge" in current_ops:
        for edge in list(cg.edges):
            edge_replacement = edge_replacement_index.get(edge)
            if edge_replacement is None:
                continue

            current_replacement = edge_replacement.get("edge")
            if current_replacement is None:
                continue

            replacement_attr = G.edges[current_replacement]

            u, v = edge
            src_type = cg.nodes[u].get("entity_type")
            tgt_type = cg.nodes[v].get("entity_type")
            new_label = replacement_attr.get("description", "") or replacement_attr.get("keywords", "")
            if not check_f1(schema_index, src_type, new_label, tgt_type):
                continue

            perturbed_cg = replace_edge(cg, edge, current_replacement, **replacement_attr)

            if unit_cost:
                perturbation_cost = replace_edge_uc()
            else:
                old_emb = _edge_emb(edge)
                new_emb = _edge_emb(current_replacement)
                if old_emb is None or new_emb is None:
                    sim = edge_replacement.get("similarity", 0.0)
                    d_sem = 1 - sim
                    perturbation_cost = 1 + d_sem
                else:
                    perturbation_cost = replace_edge_cost(old_emb, new_emb)

            new_ops = ops + [("replace_edge", (edge, current_replacement))]
            similarity = edge_similarity_index.get(edge, 0.0)
            heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

    if "add_edge" in current_ops:
        cg_nodes = set(cg.nodes)
        cg_edges = set(cg.edges)
        for u in cg_nodes:
            for (label, neigh_type, neigh) in adjacency_index.get(u, []):
                if neigh not in cg_nodes:
                    continue
                cand_edge = (u, neigh)
                if cand_edge in cg_edges or (neigh, u) in cg_edges:
                    continue
                if not check_f2(G, new_edge=cand_edge):
                    continue
                src_type = cg.nodes[u].get("entity_type")
                tgt_type = cg.nodes[neigh].get("entity_type")
                if not check_f1(schema_index, src_type, label, tgt_type):
                    continue

                attrs = G.edges[cand_edge] if cand_edge in G.edges else {}
                perturbed_cg = add_edge(cg, cand_edge, **attrs)

                if unit_cost:
                    perturbation_cost = add_edge_uc()
                else:
                    new_emb = _edge_emb(cand_edge)
                    if new_emb is None:
                        perturbation_cost = 2.0
                    else:
                        min_d = float("inf")
                        for e in cg.edges:
                            cur_emb = _edge_emb(e)
                            if cur_emb is None:
                                continue
                            d = 1 - cosine_similarity_norm(cur_emb, new_emb)
                            if d < min_d:
                                min_d = d
                        if min_d == float("inf"):
                            min_d = 1.0
                        perturbation_cost = 1 + min_d

                new_ops = ops + [("add_edge", cand_edge)]
                heapq.heappush(Q, (cost + perturbation_cost, 0.0, next(counter), (perturbed_cg, new_ops)))

    if "add_node" in current_ops and query_embedding is not None:
        cg_nodes = set(cg.nodes)
        scored = []
        for v_prime in G.nodes:
            if v_prime in cg_nodes:
                continue
            adj = adjacency_index.get(v_prime, [])
            edges_to_cg = [(label, nt, n) for (label, nt, n) in adj if n in cg_nodes]
            if not edges_to_cg:
                continue
            new_emb = _node_emb(v_prime)
            if new_emb is None:
                continue
            sim = cosine_similarity_norm(query_embedding, new_emb)
            scored.append((sim, v_prime, edges_to_cg, new_emb))

        scored.sort(key=lambda x: -x[0])
        for sim, v_prime, edges_to_cg, new_emb in scored[:add_topk]:
            attrs = G.nodes[v_prime]
            perturbed_cg = add_node(cg, v_prime, **attrs)
            connecting_ops = []
            for label, neigh_type, neigh in edges_to_cg:
                cand_edge = (v_prime, neigh) if (v_prime, neigh) in G.edges else (neigh, v_prime)
                src_type = G.nodes[cand_edge[0]].get("entity_type")
                tgt_type = G.nodes[cand_edge[1]].get("entity_type")
                if not check_f1(schema_index, src_type, label, tgt_type):
                    continue
                edge_attrs = G.edges[cand_edge] if cand_edge in G.edges else {}
                perturbed_cg = add_edge(perturbed_cg, cand_edge, **edge_attrs)
                connecting_ops.append(cand_edge)

            if not connecting_ops:
                continue

            if unit_cost:
                perturbation_cost = add_node_uc(cg, connecting_ops)
            else:
                min_d_node = float("inf")
                for u in cg.nodes:
                    cur_emb = _node_emb(u)
                    if cur_emb is None:
                        continue
                    d = 1 - cosine_similarity_norm(cur_emb, new_emb)
                    if d < min_d_node:
                        min_d_node = d
                if min_d_node == float("inf"):
                    min_d_node = 1.0
                total = 1 + min_d_node
                for ce in connecting_ops:
                    ce_emb = _edge_emb(ce)
                    if ce_emb is None:
                        total += 2.0
                        continue
                    min_d_e = float("inf")
                    for e in cg.edges:
                        cur_emb = _edge_emb(e)
                        if cur_emb is None:
                            continue
                        d = 1 - cosine_similarity_norm(cur_emb, ce_emb)
                        if d < min_d_e:
                            min_d_e = d
                    if min_d_e == float("inf"):
                        min_d_e = 1.0
                    total += 1 + min_d_e
                perturbation_cost = total

            new_ops = ops + [("add_node", (v_prime, connecting_ops))]
            heapq.heappush(Q, (cost + perturbation_cost, -sim, next(counter), (perturbed_cg, new_ops)))


async def pivotal_star_probe(rag, question, context_graph, original_answer,
                             node_similarity_index, unit_cost: bool,
                             max_pivots: int, state_cache: set):
    """Pivotal-Star Probe heuristic (see local.tex sec. 1.6).

    Eagerly probes the top-K query-similar nodes by deleting each one
    together with its singleton-stranded neighbours. On flip, refines
    inside the deleted star for a cheaper sub-deletion. On no-flip,
    hard-prunes every deletion fully inside the star from later search.

    Returns (best_sigma, best_cost, blacklist, llm_calls).
    """
    cost_fn = delete_node_uc if unit_cost else delete_node_cost
    edge_cost_fn = delete_edge_uc if unit_cost else delete_edge_cost

    undirected = context_graph.to_undirected()
    cut_vertices = set(nx.articulation_points(undirected))

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
        state = frozenset(
            (u, v, cg.edges[u, v].get("description", ""))
            for u, v in cg.edges
        )
        if state in state_cache:
            return False
        state_cache.add(state)
        cg_context = graph_to_context_shuffled(cg, shuffle_entities=False, shuffle_relations=False)
        new_response = await query(rag, cg_context, question)
        score = await judge_response(question, new_response, original_answer)
        llm_calls += 1
        flipped = score == 0
        print(f"[PSP] Cost {cost:.2f} | flipped={flipped} | ops={ops}")
        if flipped and cost < best_cost:
            best_sigma = ops
            best_cost = cost
        return flipped

    for v in pivots:
        if not check_f3(cand_node=v, cut_vertices=cut_vertices, undirected=undirected):
            continue

        incident_edges = list(context_graph.edges(v))
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
                if not check_f3(cand_node=n, cut_vertices=cut_vertices, undirected=undirected):
                    continue
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


def save_operations_to_json(ops: list, question: str, original_answer: str, perturbed_answer: str, answer_similarity: float, original_subgraph, perturbed_subgraph, output_dir: str = "src/counterfactuals/robustness/stability", filename: str = None, found: bool = True, cost: float = 0.0, llm_calls: int = 0, current_ops: list=[], suffix: str = ""):
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
    else:
        output_dir = f"{output_dir}_uc_all"

    if suffix:
        output_dir = f"{output_dir}{suffix}"

    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"counterfactual_{timestamp}.json"

    filepath = os.path.join(output_dir, filename)

    serialisable_ops = []
    for op in ops:
        if isinstance(op, tuple):
            serialisable_ops.append(list(op))
        else:
            serialisable_ops.append(op)

    payload = {
        "question": question,
        "found": found,
        "num_operations": len(serialisable_ops),
        "operations": serialisable_ops,
        "cost": cost,
        "answers": {
            "original": original_answer,
            "perturbed": perturbed_answer,
            "similarity": round(answer_similarity, 6)
        },
        "original_subgraph": subgraph_to_dict(original_subgraph),
        "perturbed_subgraph": subgraph_to_dict(perturbed_subgraph),
        "timestamp": datetime.now().isoformat(),
        "llm_calls": llm_calls
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Operations saved to: {filepath}")
    return filepath

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Counterfactual search driver.")
    p.add_argument("--input", default="benchmark/results/comparison.json",
                   help="Path to comparison.json (with {results: {idx: {question, case, ...}}})")
    p.add_argument("--case", choices=["all", "tf", "ft"], default="all",
                   help="Which entries to process by flip direction (default: all)")
    p.add_argument("--max-cost", type=float, default=10.0,
                   help="Dijkstra cost cutoff")
    p.add_argument("--max-llm-calls", type=int, default=100,
                   help="LLM call budget per query")
    p.add_argument("--unit-cost", action="store_true",
                   help="Use unit costs instead of semantic 1+d_sem costs")
    p.add_argument("--ops", nargs="+",
                   default=["delete_node", "delete_edge", "replace_node",
                            "replace_edge", "add_node", "add_edge"],
                   help="Which edit ops to enable")
    p.add_argument("--use-psp", action="store_true",
                   help="Enable the Pivotal-Star Probe heuristic")
    p.add_argument("--max-pivots", type=int, default=3,
                   help="Top-K query-similar pivots for PSP")
    p.add_argument("--suffix", default="",
                   help="Suffix appended to output directory (keeps runs separate)")
    p.add_argument("--retrieve-mode", default="hybrid",
                   help="LightRAG retrieval mode for context graph")
    p.add_argument("--retrieve-top-k", type=int, default=2,
                   help="Retrieval top-k for the context graph")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most this many entries (for smoke tests)")
    return p.parse_args()


async def main():
    args = parse_args()
    rag = await initialize_lightrag()

    with open(args.input, "r", encoding="utf-8") as results:
        data = json.load(results)

    op_set = args.ops
    processed = 0

    for idx, r in data["results"].items():
        question = r["question"]
        case = r.get("case", "all")

        if args.case != "all" and case != args.case:
            continue
        if args.limit is not None and processed >= args.limit:
            break

        print(f"\n=== [{idx}] case={case} | {question} ===")

        context = await retrieve_subgraph(rag, query=question,
                                          mode=args.retrieve_mode,
                                          top_k=args.retrieve_top_k)
        await find_counterfactuals(
            rag, question, context=context,
            max_cost=args.max_cost,
            max_llm_calls=args.max_llm_calls,
            unit_cost=args.unit_cost,
            current_ops=op_set,
            use_pivotal_probe=args.use_psp,
            max_pivots=args.max_pivots,
            suffix=args.suffix,
        )
        processed += 1

    print(f"\nProcessed {processed} entries.")


if __name__ == "__main__":
    asyncio.run(main())
