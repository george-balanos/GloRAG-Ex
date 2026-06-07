from datetime import datetime
from src.query import *
from src.retrieve import *
from src.parser import *
from src.llm_judge import judge_response
from src.counterfactuals.edit_costs import *
from src.counterfactuals.perturbations import *
from src.counterfactuals.utils import compute_answer_similarity, cosine_similarity_norm
from src.embeddings.query import get_embedding
from src.embeddings.query import query as embedding_query

import argparse
import heapq
import json
import math
import networkx as nx
import asyncio
import itertools
import os
import time

from src.dataset_setup import (
    WORKING_DIRS,
    DATASETS,
    setup_dataset as _shared_setup_dataset,
)

### Setup ###

counter = itertools.count()


def _heap_push(Q, cost, similarity, len_ops, payload, *, relevance_sign=-1.0, add_params=None):
    """Push to Q as a 6-tuple (priority, len_ops, similarity_key, raw_cost, counter, payload).

    add_params=None         => priority = cost (deletions, defaults, blend off)
    add_params={'mode':'tier','tier_width':W}  => priority = W * floor(cost / W)
    add_params={'mode':'blend','alpha':a}      => priority = cost - a * similarity

    raw_cost is preserved in slot 4 so the cost-budget check and child cost
    accumulation never see the modified priority. Cost-optimality:
      - tier mode preserves it up to tier_width.
      - blend mode loses it by up to alpha (intentional tradeoff).
    """
    if add_params is None or add_params.get("mode") in (None, "none"):
        priority = cost
    elif add_params["mode"] == "tier":
        w = add_params["tier_width"]
        priority = w * math.floor(cost / w) if w > 0 else cost
    elif add_params["mode"] == "blend":
        a = add_params["alpha"]
        priority = cost - a * similarity
    else:
        priority = cost
    heapq.heappush(Q, (priority, len_ops, relevance_sign * similarity, cost, next(counter), payload))


dataset: str = "synthetic"
adm: int = 2  # default add-mode; overridden by main() when invoked via CLI.
G = None
type_index = None
node_index = node_records = node_embeddings = node_lookup = None
edge_index = edge_records = edge_embeddings = edge_lookup = None


def setup_dataset(name: str):
    """(Re)bind module-level dataset globals via the shared loader in src.dataset_setup."""
    global dataset, G, type_index
    global node_index, node_records, node_embeddings, node_lookup
    global edge_index, edge_records, edge_embeddings, edge_lookup

    bundle = _shared_setup_dataset(name)
    dataset = bundle["dataset"]
    G = bundle["G"]
    type_index = bundle["type_index"]
    node_index = bundle["node_index"]
    node_records = bundle["node_records"]
    node_embeddings = bundle["node_embeddings"]
    node_lookup = bundle["node_lookup"]
    edge_index = bundle["edge_index"]
    edge_records = bundle["edge_records"]
    edge_embeddings = bundle["edge_embeddings"]
    edge_lookup = bundle["edge_lookup"]


### Setup - Timer (Start):
_setup_start = time.perf_counter()
setup_dataset("synthetic")
### Setup - Timer (End)
setup_time = time.perf_counter() - _setup_start
print(f"Setup time: {setup_time:.3f}s")

################################################

### Similarity Index (Node/Edge)

def create_node_similarity_index(nodes, query_embedding):
    node_similarity_index = {}
    for node in nodes:
        node_embedding = get_embedding(node_embeddings, node_lookup, node)
        if node_embedding is not None:
            similarity = cosine_similarity_norm(query_embedding, node_embedding)
        else:
            similarity = 0.0

        node_similarity_index[node] = similarity

    return node_similarity_index

async def create_edge_similarity_index(edge_labels, query_embedding):
    edge_similarity_index = {}

    edges_with_labels = [(e, l) for e, l in edge_labels.items() if l]
    edges_without_labels = [e for e, l in edge_labels.items() if not l]

    for edge in edges_without_labels:
        edge_similarity_index[edge] = 0.0

    if edges_with_labels:
        edges, labels = zip(*edges_with_labels)
        embeddings = await sentence_transformer_embed(list(labels))
        for edge, embedding in zip(edges, embeddings):
            edge_similarity_index[edge] = cosine_similarity_norm(query_embedding, embedding)

    return edge_similarity_index

################################################

def _closed_star(cg, pivot):
    """Return (star_nodes, star_edges) = {pivot} ∪ degree-1 neighbors, and incident edges."""
    preds = list(cg.predecessors(pivot))
    succs = list(cg.successors(pivot))
    neighbors = preds + succs
    singleton_neighbors = {
        u for u in neighbors
        if cg.in_degree(u) + cg.out_degree(u) == 1
    }
    incident_edges = set(cg.in_edges(pivot)) | set(cg.out_edges(pivot))
    return {pivot} | singleton_neighbors, incident_edges


def _state_key(cg):
    """Canonical state key matching the main loop's state_cache schema."""
    return (
        frozenset(cg.nodes()),
        frozenset(
            (u, v, cg.edges[u, v].get("description", ""))
            for u, v in cg.edges()
        ),
    )


async def psp_probe(
    rag,
    question: str,
    original_answer: str,
    cg,
    node_similarity_index: dict,
    K: int,
    unit_cost: bool,
):
    """
    Pivotal-Star Probe. Picks top-K pivots from cg.nodes by query relevance,
    evaluates del_n(v) for each as a single LLM call, and returns:
      psp_hits           : list of (pivot, cost, perturbed_cg, response) for pivots that flipped
      psp_pruned_nodes   : union of S_C(v).nodes for non-flipping pivots
      psp_pruned_edges   : union of S_C(v).edges for non-flipping pivots
      psp_probed_pivots  : set of all probed pivots (skip re-eval in expand)
      psp_response_cache : {state_key: response} so the main loop can skip the LLM
                           re-call when a PSP-probed state is popped
      llm_calls          : number of LLM calls consumed
      psp_llm_time       : wall-clock seconds spent inside LLM/judge during the probe
    """
    pivots = sorted(
        cg.nodes,
        key=lambda v: node_similarity_index.get(v, 0.0),
        reverse=True,
    )[:K]

    psp_hits = []
    psp_pruned_nodes: set = set()
    psp_pruned_edges: set = set()
    psp_probed_pivots: set = set(pivots)
    psp_response_cache: dict = {}
    llm_calls = 0
    psp_llm_time = 0.0

    for v in pivots:
        star_nodes, star_edges = _closed_star(cg, v)

        perturbed_cg = delete_node(cg, v)

        if unit_cost:
            cost = delete_node_uc(cg, v)
        else:
            cost = delete_node_cost(cg, v)

        cg_context = graph_to_context(perturbed_cg)

        _t0 = time.perf_counter()
        new_response = await query(rag, cg_context, question)
        score = await judge_response(question, new_response, original_answer)
        psp_llm_time += time.perf_counter() - _t0

        llm_calls += 1
        flipped = (score == 0)

        print(f"[PSP] pivot={v} cost={cost} flipped={flipped}")

        # Cache the LLM response keyed by perturbed-graph state so the main loop
        # does not re-evaluate this graph when the pivot is popped from Q.
        psp_response_cache[_state_key(perturbed_cg)] = new_response

        if flipped:
            psp_hits.append((v, cost, perturbed_cg, new_response))
        else:
            psp_pruned_nodes |= star_nodes
            psp_pruned_edges |= star_edges

    return (psp_hits, psp_pruned_nodes, psp_pruned_edges,
            psp_probed_pivots, psp_response_cache, llm_calls, psp_llm_time)


async def psp_refine_star(
    rag,
    question: str,
    original_answer: str,
    cg,
    pivot,
    star_nodes: set,
    star_edges: set,
    unit_cost: bool,
    seed_cost: float,
    seed_perturbed_cg,
    seed_response: str,
    response_cache: dict,
    llm_budget: int,
):
    """Focused Dijkstra restricted to deletions within S_C(pivot).

    Returns the minimum-cost flipping subset:
      (best_ops, best_perturbed_cg, best_cost, best_response, llm_calls, llm_time)

    The full-star deletion is treated as the seed (guaranteed to flip by
    psp_probe). The sub-search tries cheaper subsets first; if none flip
    within `llm_budget` LLM calls, returns the seed.
    """
    # Atomic moves allowed inside the sub-search:
    #   - delete_node(u) for u ∈ star_nodes
    #   - delete_edge(e) for e ∈ star_edges
    allowed_nodes = set(star_nodes)
    allowed_edges = {tuple(e) for e in star_edges}

    Q = []
    state_cache: set = set()

    # Push the empty-ops root (= original cg).
    _heap_push(Q, cost=0, similarity=0.0, len_ops=0, payload=(cg, []))

    best_ops = [("delete_node", pivot)]
    best_perturbed_cg = seed_perturbed_cg
    best_cost = float(seed_cost)
    best_response = seed_response

    llm_calls = 0
    llm_time = 0.0

    while Q:
        _, _, _, cost, _, (current_cg, ops) = heapq.heappop(Q)

        # Hard-bound: anything ≥ seed_cost cannot improve.
        if cost >= best_cost:
            continue
        if llm_calls >= llm_budget:
            break

        state = _state_key(current_cg)
        if state in state_cache:
            continue
        state_cache.add(state)

        # Eval if we've actually taken at least one step (root has empty ops).
        if len(ops) > 0:
            cached = response_cache.pop(state, None)
            _t0 = time.perf_counter()
            if cached is not None:
                new_response = cached
            else:
                llm_calls += 1
                cg_context = graph_to_context(current_cg)
                new_response = await query(rag, cg_context, question)
            score = await judge_response(question, new_response, original_answer)
            llm_time += time.perf_counter() - _t0

            print(f"[PSP-refine pivot={pivot}] cost={cost} ops={ops} score={score}")

            if score == 0 and cost < best_cost:
                best_ops = list(ops)
                best_perturbed_cg = current_cg
                best_cost = cost
                best_response = new_response
                # Don't return immediately — Dijkstra's first goal-pop with
                # strict cost priority IS the minimum; but we already had a
                # known-flip seed, so we just record and let the heap drain
                # the remaining strictly-cheaper candidates (cost < best_cost
                # filter prunes them naturally).
                continue

        # Expand: enumerate every allowed deletion that stays within S_C(pivot).
        for node in list(current_cg.nodes):
            if node not in allowed_nodes:
                continue
            perturbed_cg = delete_node(current_cg, node)
            if unit_cost:
                step_cost = delete_node_uc(current_cg, node)
            else:
                step_cost = delete_node_cost(current_cg, node)
            new_ops = ops + [("delete_node", node)]
            _heap_push(Q, cost=cost + step_cost, similarity=0.0,
                       len_ops=len(new_ops), payload=(perturbed_cg, new_ops))

        for edge in list(current_cg.edges):
            if tuple(edge) not in allowed_edges:
                continue
            perturbed_cg = delete_edge(current_cg, edge)
            if unit_cost:
                step_cost = delete_edge_uc(current_cg, edge)
            else:
                step_cost = delete_edge_cost(current_cg, edge)
            new_ops = ops + [("delete_edge", tuple(edge))]
            _heap_push(Q, cost=cost + step_cost, similarity=0.0,
                       len_ops=len(new_ops), payload=(perturbed_cg, new_ops))

    return best_ops, best_perturbed_cg, best_cost, best_response, llm_calls, llm_time


async def find_breaking_counterfactuals(
    rag,
    question: str,
    original_answer: str,
    ground_truth: str,
    query_embedding,
    context: str,
    max_cost: int = 3,
    max_llm_calls: int = 100,
    unit_cost: bool = False,
    current_ops: list=["delete_node", "delete_edge"],
    mode: str = "ft",
    use_psp: bool = False,
    psp_k: int = 5,
    output_dir: str = "src/counterfactuals/results",
    add_params: dict = None,
    total_start: float = None,
    setup_time: float = 0.0,
    pre_llm_time: float = 0.0,
):
    if total_start is None:
        total_start = time.perf_counter()

    ## LLM timer (Start):
    llm_time = pre_llm_time

    llm_calls = 0

    ### Lightrag specific
    parsed_subgraph = parse_context(context)
    context_graph = parse_graph(parsed_subgraph)
    #####################

    context_graph_nodes = set(context_graph.nodes)
    context_graph_edges = set(context_graph.edges())

    ## Similarity index timer (Start):
    _index_start = time.perf_counter()

    edge_labels = {(u, v): data.get("description", "") for u, v, data in G.edges(data=True)}
    node_similarity_index = create_node_similarity_index(set(G.nodes), query_embedding)
    edge_similarity_index = await create_edge_similarity_index(edge_labels, query_embedding)

    ## Similarity index timer (End):
    index_time = time.perf_counter() - _index_start
    print(f"Index creation time: {index_time:.3f}s")

    ### PSP (T→F deletion-only heuristic)
    psp_pruned_nodes: set = set()
    psp_pruned_edges: set = set()
    psp_response_cache: dict = {}
    if use_psp and "delete_node" in current_ops:
        # _probed_pivots returned but unused by callers; kept for return shape.
        (psp_hits, psp_pruned_nodes, psp_pruned_edges,
         _probed_pivots, psp_response_cache, psp_calls, psp_llm_time) = await psp_probe(
            rag=rag,
            question=question,
            original_answer=original_answer,
            cg=context_graph,
            node_similarity_index=node_similarity_index,
            K=psp_k,
            unit_cost=unit_cost,
        )
        llm_calls += psp_calls
        llm_time  += psp_llm_time

        # If any pivot flipped: refine each hit by sub-Dijkstra over its star,
        # take the global min, save, and SKIP the main Dijkstra entirely
        # (per spec: "if you do [flip], you check only perturbations in the
        # subgraph deleted").
        if psp_hits:
            refine_budget_per_pivot = max(8, 2 * psp_k)

            best_ops, best_cg, best_cost, best_response = None, None, float("inf"), None
            for v, hit_cost, perturbed_cg, response in psp_hits:
                star_nodes, star_edges = _closed_star(context_graph, v)
                r_ops, r_cg, r_cost, r_response, r_calls, r_time = await psp_refine_star(
                    rag=rag,
                    question=question,
                    original_answer=original_answer,
                    cg=context_graph,
                    pivot=v,
                    star_nodes=star_nodes,
                    star_edges=star_edges,
                    unit_cost=unit_cost,
                    seed_cost=hit_cost,
                    seed_perturbed_cg=perturbed_cg,
                    seed_response=response,
                    response_cache=psp_response_cache,
                    llm_budget=refine_budget_per_pivot,
                )
                llm_calls += r_calls
                llm_time  += r_time
                if r_cost < best_cost:
                    best_ops, best_cg, best_cost, best_response = r_ops, r_cg, r_cost, r_response

            print(f"[PSP] hits={len(psp_hits)} → refined CFE cost={best_cost} ops={best_ops}")

            _t0 = time.perf_counter()
            answer_similarity = await compute_answer_similarity(original_answer, best_response)
            llm_time += time.perf_counter() - _t0

            total_time = time.perf_counter() - total_start
            algo_time = total_time - llm_time - index_time

            parsed_subgraph = parse_context(context)
            save_operations_to_json(
                ops=best_ops,
                question=question,
                ground_truth=ground_truth,
                original_answer=original_answer,
                perturbed_answer=best_response,
                answer_similarity=answer_similarity,
                original_subgraph=parsed_subgraph,
                perturbed_subgraph=graph_to_subgraph(best_cg),
                output_dir=output_dir,
                found=True,
                cost=best_cost,
                llm_calls=llm_calls,
                current_ops=current_ops,
                mode=mode,
                total_time=total_time,
                algo_time=algo_time,
                setup_time=setup_time,
                index_time=index_time,
                pre_llm_time=pre_llm_time,
            )
            return best_ops

    ### Main Dijkstra (only reached if PSP off OR no pivot flipped).
    Q = []
    state_cache = set()
    _heap_push(Q, cost=0, similarity=0.0, len_ops=0, payload=(context_graph, []))

    explored_nodes = set()  ## For addition

    while Q:
        # 6-tuple: (priority, len_ops, similarity_key, raw_cost, counter, payload)
        _, _, _, cost, _, (cg, ops) = heapq.heappop(Q)

        if cost > max_cost:
            print(f"Max cost {max_cost} exceeded (current cost: {cost:.4f}). Stopping search.")
            break
        elif llm_calls > max_llm_calls:
            print(f"Max LLM calls {max_llm_calls} exceeded. Stopping search.")
            break

        state = (
            frozenset(cg.nodes()),
            frozenset(
                (u, v, cg.edges[u, v].get("description", ""))
                for u, v in cg.edges()
            )
        )

        if state in state_cache:
            continue

        state_cache.add(state)

        if len(ops) > 0:
            # PSP eval-time short-circuit removed: with expand-time hard-pruning
            # (see expand()), no pruned-star deletion ever reaches the heap.
            llm_calls += 1
            cg_context = graph_to_context(cg)

            _t0 = time.perf_counter()
            new_response = await query(rag, cg_context, question)
            print(f"Cost: {cost} | New response: {new_response} | Original: {original_answer}")
            print(f"Ground Truth: {ground_truth}")

            score = await judge_response(question, new_response, original_answer)
            llm_time += time.perf_counter() - _t0

            if score == 0:
                print(f"Counterfactual Operations: {ops}")

                _t0 = time.perf_counter()
                answer_similarity = await compute_answer_similarity(original_answer, new_response)
                llm_time += time.perf_counter() - _t0

                ### Pure Algorithm timer
                total_time = time.perf_counter() - total_start
                algo_time = total_time - llm_time - index_time
                ##############################################

                print(f"Answer similarity (original vs perturbed): {answer_similarity:.4f}")

                parsed_subgraph = parse_context(context)

                save_operations_to_json(
                    ops=ops,
                    question=question,
                    ground_truth=ground_truth,
                    original_answer=original_answer,
                    perturbed_answer=new_response,
                    answer_similarity=answer_similarity,
                    original_subgraph=parsed_subgraph,
                    perturbed_subgraph=graph_to_subgraph(cg),
                    output_dir=output_dir,
                    found=True,
                    cost=cost,
                    llm_calls=llm_calls,
                    current_ops=current_ops,
                    mode=mode,
                    total_time=total_time,
                    algo_time=algo_time,
                    setup_time=setup_time,
                    index_time=index_time,
                    pre_llm_time=pre_llm_time,
                )

                return ops

        await expand(
            Q,
            (cost, cg, ops),
            node_similarity_index=node_similarity_index,
            edge_similarity_index=edge_similarity_index,
            unit_cost=unit_cost,
            current_ops=current_ops,
            original_nodes=context_graph_nodes,
            original_edges=context_graph_edges,
            explored_nodes=explored_nodes,
            query_embedding=query_embedding,
            mode=mode,
            add_params=add_params,
            psp_pruned_nodes=psp_pruned_nodes,
            psp_pruned_edges=psp_pruned_edges,
        )

    ### Exhausted budget timer
    total_time = time.perf_counter() - total_start
    algo_time = total_time - llm_time - index_time

    print(f"Could not find feasible counterfactual explanations.")

    save_operations_to_json(
        ops=[],
        question=question,
        ground_truth=ground_truth,
        original_answer=original_answer,
        perturbed_answer=None,
        answer_similarity=0.0,
        original_subgraph=parsed_subgraph,
        perturbed_subgraph=None,
        output_dir=output_dir,
        found=False,
        llm_calls=llm_calls,
        cost=cost,
        current_ops=current_ops,
        mode=mode,
        total_time=total_time,
        algo_time=algo_time,
        setup_time=setup_time,
        index_time=index_time,
        pre_llm_time=pre_llm_time,
    )

async def find_corrective_counterfactuals(
    rag,
    question: str,
    original_answer: str,
    ground_truth: str,
    query_embedding,
    context: str,
    max_cost: int = 3,
    max_llm_calls: int = 100,
    unit_cost: bool = False,
    current_ops: list=["delete_node", "delete_edge", "add_node", "add_edge"],
    mode: str = "ff",
    output_dir: str = "src/counterfactuals/results",
    add_params: dict = None,
    total_start: float = None,
    setup_time: float = 0.0,
    pre_llm_time: float = 0.0,
):
    if total_start is None:
        total_start = time.perf_counter()

    llm_time = pre_llm_time
    llm_calls = 0

    ### Lightrag specific
    parsed_subgraph = parse_context(context)
    context_graph = parse_graph(parsed_subgraph)
    #####################

    context_graph_nodes = set(context_graph.nodes)
    context_graph_edges = set(context_graph.edges())

    ## Similarity index timer (Start):
    _index_start = time.perf_counter()

    edge_labels = {(u, v): data.get("description", "") for u, v, data in G.edges(data=True)}
    node_similarity_index = create_node_similarity_index(set(G.nodes), query_embedding)
    edge_similarity_index = await create_edge_similarity_index(edge_labels, query_embedding)

    ## Similarity index timer (End):
    index_time = time.perf_counter() - _index_start
    print(f"Index creation time: {index_time:.3f}s")

    ### Min-heap
    Q = []
    state_cache = set()
    _heap_push(Q, cost=0, similarity=0.0, len_ops=0, payload=(context_graph, []))

    explored_nodes = set()  ## For addition
    edge_embedding_cache = {}

    while Q:
        # 6-tuple: (priority, len_ops, similarity_key, raw_cost, counter, payload)
        _, _, _, cost, _, (cg, ops) = heapq.heappop(Q)

        if cost > max_cost:
            print(f"Max cost {max_cost} exceeded (current cost: {cost:.4f}). Stopping search.")
            break
        elif llm_calls > max_llm_calls:
            print(f"Max LLM calls {max_llm_calls} exceeded. Stopping search.")
            break

        state = (
            frozenset(cg.nodes()),
            frozenset(
                (u, v, cg.edges[u, v].get("description", ""))
                for u, v in cg.edges()
            )
        )

        if state in state_cache:
            continue

        state_cache.add(state)

        if len(ops) > 0:
            llm_calls += 1

            cg_context = graph_to_context(cg)

            ## LLM interval (start):
            _t0 = time.perf_counter()

            new_response = await query(rag, cg_context, question)

            score = await judge_response(question, new_response, ground_truth)

            ## LLM interval (end):
            llm_time += time.perf_counter() - _t0

            print(f"Cost: {cost} | New response: {new_response} | Original: {original_answer}")
            print(f"Ground Truth: {ground_truth}")

            if score == 1:
                print(f"Counterfactual Operations: {ops}")

                _t0 = time.perf_counter()

                answer_similarity = await compute_answer_similarity(ground_truth, new_response)
                
                llm_time += time.perf_counter() - _t0

                ### Pure Algorithm timer
                total_time = time.perf_counter() - total_start
                algo_time = total_time - llm_time - index_time
                ###########################

                print(f"Answer similarity (ground truth vs perturbed): {answer_similarity:.4f}")

                save_operations_to_json(
                    ops=ops,
                    question=question,
                    ground_truth=ground_truth,
                    original_answer=original_answer,
                    perturbed_answer=new_response,
                    answer_similarity=answer_similarity,
                    original_subgraph=parsed_subgraph,
                    perturbed_subgraph=graph_to_subgraph(cg),
                    output_dir=output_dir,
                    found=True,
                    cost=cost,
                    llm_calls=llm_calls,
                    current_ops=current_ops,
                    mode=mode,
                    total_time=total_time,
                    algo_time=algo_time,
                    setup_time=setup_time,
                    index_time=index_time,
                    pre_llm_time=pre_llm_time,
                )

                return ops

        await expand(
            Q,
            (cost, cg, ops),
            node_similarity_index=node_similarity_index,
            edge_similarity_index=edge_similarity_index,
            unit_cost=unit_cost,
            current_ops=current_ops,
            original_nodes=context_graph_nodes,
            original_edges=context_graph_edges,
            explored_nodes=explored_nodes,
            query_embedding=query_embedding,
            edge_labels=edge_labels,
            edge_embedding_cache=edge_embedding_cache,
            mode=mode,
            add_params=add_params,
        )

    ### Exhausted budget timer
    total_time = time.perf_counter() - total_start
    algo_time = total_time - llm_time - index_time

    print(f"Could not find feasible counterfactual explanations.")

    save_operations_to_json(
        ops=[],
        question=question,
        ground_truth=ground_truth,
        original_answer=original_answer,
        perturbed_answer=None,
        answer_similarity=0.0,
        original_subgraph=parsed_subgraph,
        perturbed_subgraph=None,
        output_dir=output_dir,
        found=False,
        llm_calls=llm_calls,
        cost=cost,
        current_ops=current_ops,
        mode=mode,
        total_time=total_time,
        algo_time=algo_time,
        setup_time=setup_time,
        index_time=index_time,
        pre_llm_time=pre_llm_time,
    )

async def find_counterfactuals(
    rag,
    question: str,
    context,
    max_cost=3,
    max_llm_calls=100,
    unit_cost: bool=False,
    current_ops: list=["delete_node", "delete_edge"],
    ground_truth: str = "",
    mode: str = "ft",
    use_psp: bool = False,
    psp_k: int = 5,
    output_dir: str = "src/counterfactuals/results",
    add_params: dict = None,
    total_start: float = None,
    setup_time: float = 0.0,
    pre_llm_time: float = 0.0,
):
    if total_start is None:
        total_start = time.perf_counter()

    _t0 = time.perf_counter()
    query_embedding = (await sentence_transformer_embed([question]))[0]
    original_answer = await query(rag, context, question)
    pre_llm_time += time.perf_counter() - _t0

    if mode == "ft":
        await find_breaking_counterfactuals(
            rag=rag,
            question=question,
            original_answer=original_answer,
            ground_truth=ground_truth,
            query_embedding=query_embedding,
            context=context,
            max_cost=max_cost,
            max_llm_calls=max_llm_calls,
            unit_cost=unit_cost,
            current_ops=current_ops,
            mode=mode,
            use_psp=use_psp,
            psp_k=psp_k,
            output_dir=output_dir,
            add_params=add_params,
            total_start=total_start,
            setup_time=setup_time,
            pre_llm_time=pre_llm_time,
        )
    elif mode in ("ff", "tf"):
        await find_corrective_counterfactuals(
            rag=rag,
            question=question,
            original_answer=original_answer,
            ground_truth=ground_truth,
            query_embedding=query_embedding,
            context=context,
            max_cost=max_cost,
            max_llm_calls=max_llm_calls,
            unit_cost=unit_cost,
            current_ops=current_ops,
            mode=mode,
            output_dir=output_dir,
            add_params=add_params,
            total_start=total_start,
            setup_time=setup_time,
            pre_llm_time=pre_llm_time,
        )

async def expand(
    Q,
    heap_element,
    node_similarity_index,
    edge_similarity_index,
    edge_labels: dict = None,
    unit_cost: bool = False,
    current_ops: list=["delete_node", "delete_edge"],
    original_nodes: set = {},
    original_edges: set = {},
    explored_nodes: set = {},
    query_embedding: list = [],
    edge_embedding_cache: dict = None,
    mode: str = "ft",
    add_params: dict = None,
    psp_pruned_nodes: set = None,
    psp_pruned_edges: set = None,
):
    cg: nx.DiGraph
    cost, cg, ops = heap_element

    psp_pruned_nodes = psp_pruned_nodes or set()
    psp_pruned_edges = psp_pruned_edges or set()

    # Within-tier ordering by flip direction (see tab:flip-heuristics):
    # T→F deletions: most query-relevant first (popped first ⇒ use -similarity).
    # F→T deletions ("ff"/"tf"): most-distant first (least-relevant first ⇒ use +similarity).
    relevance_sign = -1.0 if mode == "ft" else 1.0

    if "delete_node" in current_ops:
        for node in list(cg.nodes):
            if node in original_nodes:
                # PSP hard-prune: failed-star members are excluded entirely
                # from the search space (per spec: "if no flip, skip all of
                # these changes from your search space").
                if node in psp_pruned_nodes:
                    continue

                perturbed_cg = delete_node(cg, node)

                if unit_cost == False:
                    perturbation_cost = delete_node_cost(cg, node)
                elif unit_cost == True:
                    perturbation_cost = delete_node_uc(cg, node)

                new_ops = ops + [("delete_node", node)]

                similarity = node_similarity_index.get(node, 0.0)

                _heap_push(Q, cost=cost + perturbation_cost, similarity=similarity, len_ops=len(new_ops),
                           payload=(perturbed_cg, new_ops), relevance_sign=relevance_sign)

    if "delete_edge" in current_ops:
        for edge in list(cg.edges):
            if edge in original_edges:
                # PSP hard-prune: edges in/incident-to a failed star are
                # excluded entirely.
                if (edge in psp_pruned_edges
                        or edge[0] in psp_pruned_nodes
                        or edge[1] in psp_pruned_nodes):
                    continue

                perturbed_cg = delete_edge(cg, edge)

                if unit_cost == False:
                    perturbation_cost = delete_edge_cost(cg, edge)
                elif unit_cost == True:
                    perturbation_cost = delete_edge_uc(cg, edge)

                new_ops = ops + [("delete_edge", edge)]

                similarity = edge_similarity_index.get(edge, 0.0)

                _heap_push(Q, cost=cost + perturbation_cost, similarity=similarity, len_ops=len(new_ops),
                           payload=(perturbed_cg, new_ops), relevance_sign=relevance_sign)

    ############################# Distance-based with query relevance for retrieval #############################

    if adm == 1:
        if "add_node" in current_ops:
            existing_nodes = set(cg.nodes)
            existing_edges = set(cg.edges())
            candidate_nodes_for_expansion = existing_nodes - explored_nodes

            candidates_pushed = 0

            for node in candidate_nodes_for_expansion:
                neighbors = list(G.neighbors(node))

                similarity = node_similarity_index.get(node, 0.0)

                for neighbor in neighbors:

                    if neighbor not in existing_nodes:
                        perturbed_cg = add_node(cg, neighbor, **G.nodes[neighbor])
                        new_ops = ops + [("add_node", neighbor)]

                        if (node, neighbor) in edge_lookup and (node, neighbor) not in existing_edges:
                            perturbed_cg = add_edge(perturbed_cg, (node, neighbor), **G.edges[node, neighbor])
                            new_ops = new_ops + [("add_edge", (node, neighbor))]

                            if unit_cost == False:
                                perturbation_cost = add_node_cost(cg, node_embeddings, node_lookup, edge_embeddings, edge_lookup, neighbor, (node, neighbor))
                            elif unit_cost == True:
                                perturbation_cost = 2

                            _heap_push(Q, cost=cost + perturbation_cost, similarity=similarity, len_ops=len(new_ops), payload=(perturbed_cg, new_ops), add_params=add_params)

                        if (neighbor, node) in edge_lookup and (neighbor, node) not in existing_edges:
                            perturbed_cg = add_edge(perturbed_cg, (neighbor, node), **G.edges[neighbor, node])
                            new_ops = new_ops + [("add_edge", (neighbor, node))]

                            if unit_cost == False:
                                perturbation_cost = add_node_cost(cg, node_embeddings, node_lookup, edge_embeddings, edge_lookup, neighbor, (neighbor, node))
                            elif unit_cost == True:
                                perturbation_cost = 2

                            _heap_push(Q, cost=cost + perturbation_cost, similarity=similarity, len_ops=len(new_ops), payload=(perturbed_cg, new_ops), add_params=add_params)

                        candidates_pushed += 1

                explored_nodes.add(node)

            if candidates_pushed == 0: ### ===> Do not have anyone to explore/expand
                existing_edges = set(cg.edges())
                candidate_edges = [e for e in edge_lookup if e not in existing_edges]

                uncached = [e for e in candidate_edges if e not in edge_embedding_cache]

                if uncached:
                    labels = [edge_labels.get(e, "") for e in uncached]
                    embeddings = await sentence_transformer_embed(list(labels))
                    for edge, embedding in zip(uncached, embeddings):
                        edge_embedding_cache[edge] = embedding if embedding is not None else None

                best_edge, best_cost, best_similarity = None, float("inf"), 0.0
                for edge in candidate_edges:
                    embedding = edge_embedding_cache.get(edge)
                    if embedding is None:
                        continue
                    similarity = cosine_similarity_norm(query_embedding, embedding)
                    perturbation_cost = add_edge_cost(cg, edge_embeddings, edge_lookup, edge)
                    if perturbation_cost < best_cost:
                        best_cost, best_similarity, best_edge = perturbation_cost, similarity, edge

                if best_edge is not None:
                    src, tgt = best_edge
                    perturbed_cg = add_edge(cg, (src, tgt), **G.edges[src, tgt])
                    new_ops = ops + [("add_edge", (src, tgt)), ("add_node", src), ("add_node", tgt)]
                    _heap_push(Q, cost=cost + best_cost, similarity=best_similarity, len_ops=len(new_ops), payload=(perturbed_cg, new_ops), add_params=add_params)

        if "add_edge" in current_ops:
            existing_edges = set(cg.edges())
            existing_nodes = set(cg.nodes)

            for node in existing_nodes:
                available_edges = set(G.edges(node))

                for edge in available_edges:
                    node1, node2 = edge

                    similarity = edge_similarity_index.get(edge, 0.0)

                    if node1 in existing_nodes and node2 in existing_nodes:
                        if (node1, node2) in edge_lookup and (node1, node2) not in existing_edges:
                            perturbed_cg = cg.copy()
                            new_ops = ops + [("add_edge", (node1, node2))]

                            perturbed_cg = add_edge(perturbed_cg, (node1, node2), **G.edges[node1, node2])

                            if unit_cost == False:
                                perturbation_cost = add_edge_cost(cg, edge_embeddings, edge_lookup, (node1, node2))
                            elif unit_cost == True:
                                perturbation_cost = add_edge_uc()

                            _heap_push(Q, cost=cost + perturbation_cost, similarity=similarity, len_ops=len(new_ops), payload=(perturbed_cg, new_ops), add_params=add_params)

                        if (node2, node1) in edge_lookup and (node2, node1) not in existing_edges:
                            perturbed_cg = cg.copy()
                            new_ops = ops + [("add_edge", (node2, node1))]

                            perturbed_cg = add_edge(perturbed_cg, (node2, node1), **G.edges[node2, node1])

                            if unit_cost == False:
                                perturbation_cost = add_edge_cost(cg, edge_embeddings, edge_lookup, (node2, node1))
                            elif unit_cost == True:
                                perturbation_cost = add_edge_uc()

                            _heap_push(Q, cost=cost + perturbation_cost, similarity=similarity, len_ops=len(new_ops), payload=(perturbed_cg, new_ops), add_params=add_params)

    ############################################################################################################
    ############################# Query-Relevance-based with query relevance for retrieval #####################

    elif adm == 2:
        if "add_node" in current_ops:
            existing_nodes = set(cg.nodes)
            existing_edges = set(cg.edges())
            candidate_nodes_for_expansion = existing_nodes - explored_nodes

            candidates_pushed = 0

            for node in candidate_nodes_for_expansion:
                neighbors = list(G.neighbors(node))

                similarity = node_similarity_index.get(node, 0.0)

                for neighbor in neighbors:
                    if neighbor not in existing_nodes:
                        perturbed_cg = add_node(cg, neighbor, **G.nodes[neighbor])
                        new_ops = ops + [("add_node", neighbor)]

                        if (node, neighbor) in edge_lookup and (node, neighbor) not in existing_edges:
                            perturbed_cg = add_edge(perturbed_cg, (node, neighbor), **G.edges[node, neighbor])
                            new_ops = new_ops + [("add_edge", (node, neighbor))]

                            perturbation_cost = (1 + (1 - node_similarity_index.get(neighbor))) + (1 + (1 - edge_similarity_index.get((node, neighbor), 0.0)))

                            _heap_push(Q, cost=cost + perturbation_cost, similarity=similarity, len_ops=len(new_ops), payload=(perturbed_cg, new_ops), add_params=add_params)

                        if (neighbor, node) in edge_lookup and (neighbor, node) not in existing_edges:
                            perturbed_cg = add_edge(perturbed_cg, (neighbor, node), **G.edges[neighbor, node])
                            new_ops = new_ops + [("add_edge", (neighbor, node))]

                            perturbation_cost = (1 + (1 - node_similarity_index.get(neighbor))) + (1 + (1 - edge_similarity_index.get((neighbor, node), 0.0)))

                            _heap_push(Q, cost=cost + perturbation_cost, similarity=similarity, len_ops=len(new_ops), payload=(perturbed_cg, new_ops), add_params=add_params)

                        candidates_pushed += 1

                explored_nodes.add(node)

            if candidates_pushed == 0:
                relevant_edges = embedding_query(edge_index, edge_records, query_embedding, k=20)

                best_edge, best_cost, best_similarity = None, float("inf"), 0.0
                for edge in relevant_edges:
                    src = edge["src"]
                    tgt = edge["tgt"]

                    if (src, tgt) not in existing_edges and (src, tgt) in edge_lookup:
                        perturbed_cg = add_edge(cg, (src, tgt), **G.edges[src, tgt])
                        new_ops = ops + [("add_edge", (src, tgt))]

                        similarity = edge_similarity_index.get((src, tgt), 0.0)

                        perturbation_cost = (1 + (1 - node_similarity_index.get(src))) + (1 + (1 - node_similarity_index.get(tgt))) + (1 + (1 - edge_similarity_index.get((src, tgt), 0.0)))

                        if perturbation_cost < best_cost:
                            best_cost, best_similarity, best_edge = perturbation_cost, similarity, (src, tgt)

                if best_edge is not None:
                    src, tgt = best_edge
                    perturbed_cg = add_edge(cg, (src, tgt), **G.edges[src, tgt])
                    new_ops = ops + [("add_edge", (src, tgt)), ("add_node", src), ("add_node", tgt)]
                    _heap_push(Q, cost=cost + best_cost, similarity=best_similarity, len_ops=len(new_ops), payload=(perturbed_cg, new_ops), add_params=add_params)

        if "add_edge" in current_ops:
            existing_edges = set(cg.edges())
            existing_nodes = set(cg.nodes)

            for node in existing_nodes:
                available_edges = set(G.edges(node))

                for edge in available_edges:
                    node1, node2 = edge

                    similarity = edge_similarity_index.get(edge, 0.0)

                    if node1 in existing_nodes and node2 in existing_nodes:
                        if (node1, node2) in edge_lookup and (node1, node2) not in existing_edges:
                            perturbed_cg = cg.copy()
                            perturbed_cg = add_edge(perturbed_cg, (node1, node2), **G.edges[node1, node2])

                            perturbation_cost = 1 + 1 - edge_similarity_index.get((node1, node2), 0.0)
                            new_ops = ops + [("add_edge", (node1, node2))]

                            _heap_push(Q, cost=cost + perturbation_cost, similarity=similarity, len_ops=len(new_ops), payload=(perturbed_cg, new_ops), add_params=add_params)

                        if (node2, node1) in edge_lookup and (node2, node1) not in existing_edges:
                            perturbed_cg = cg.copy()
                            perturbed_cg = add_edge(perturbed_cg, (node2, node1), **G.edges[node2, node1])

                            perturbation_cost = 1 + 1 - edge_similarity_index.get((node2, node1), 0.0)
                            new_ops = ops + [("add_edge", (node2, node1))]

                            _heap_push(Q, cost=cost + perturbation_cost, similarity=similarity, len_ops=len(new_ops), payload=(perturbed_cg, new_ops), add_params=add_params)

    ############################# Number of operations / Query Relevance priority #############################

    elif adm == 3:
        if "add_node" in current_ops:
            existing_nodes = set(cg.nodes)
            existing_edges = set(cg.edges())
            candidate_nodes_for_expansion = existing_nodes - explored_nodes

            candidates_pushed = 0

            for node in candidate_nodes_for_expansion:
                neighbors = list(G.neighbors(node))

                similarity = node_similarity_index.get(node, 0.0)

                for neighbor in neighbors:
                    if neighbor not in existing_nodes:
                        perturbed_cg = add_node(cg, neighbor, **G.nodes[neighbor])
                        new_ops = ops + [("add_node", neighbor)]

                        if (node, neighbor) in edge_lookup and (node, neighbor) not in existing_edges:
                            perturbed_cg = add_edge(perturbed_cg, (node, neighbor), **G.edges[node, neighbor])
                            new_ops = new_ops + [("add_edge", (node, neighbor))]

                            perturbation_cost = 2

                            _heap_push(Q, cost=cost + perturbation_cost, similarity=similarity, len_ops=len(new_ops), payload=(perturbed_cg, new_ops), add_params=add_params)

                        if (neighbor, node) in edge_lookup and (neighbor, node) not in existing_edges:
                            perturbed_cg = add_edge(perturbed_cg, (neighbor, node), **G.edges[neighbor, node])
                            new_ops = new_ops + [("add_edge", (neighbor, node))]

                            perturbation_cost = 2

                            _heap_push(Q, cost=cost + perturbation_cost, similarity=similarity, len_ops=len(new_ops), payload=(perturbed_cg, new_ops), add_params=add_params)

                        candidates_pushed += 1

                explored_nodes.add(node)

            if candidates_pushed == 0:
                relevant_edges = embedding_query(edge_index, edge_records, query_embedding, k=20)

                best_edge, best_similarity = None, -1
                for edge in relevant_edges:
                    src = edge["src"]
                    tgt = edge["tgt"]

                    if (src, tgt) not in existing_edges and (src, tgt) in edge_lookup:
                        similarity = edge_similarity_index.get((src, tgt), 0.0)
                        if similarity > best_similarity:
                            best_similarity, best_edge = similarity, (src, tgt)

                if best_edge is not None:
                    src, tgt = best_edge
                    perturbed_cg = add_edge(cg, (src, tgt), **G.edges[src, tgt])
                    new_ops = ops + [("add_edge", (src, tgt))]
                    _heap_push(Q, cost=cost + 3, similarity=best_similarity, len_ops=len(new_ops), payload=(perturbed_cg, new_ops), add_params=add_params)

        if "add_edge" in current_ops:
            existing_edges = set(cg.edges())
            existing_nodes = set(cg.nodes)

            for node in existing_nodes:
                available_edges = set(G.edges(node))

                for edge in available_edges:
                    node1, node2 = edge

                    similarity = edge_similarity_index.get(edge, 0.0)

                    if node1 in existing_nodes and node2 in existing_nodes:
                        if (node1, node2) in edge_lookup and (node1, node2) not in existing_edges:
                            perturbed_cg = cg.copy()
                            perturbed_cg = add_edge(perturbed_cg, (node1, node2), **G.edges[node1, node2])

                            perturbation_cost = 1
                            new_ops = ops + [("add_edge", (node1, node2))]

                            _heap_push(Q, cost=cost + perturbation_cost, similarity=similarity, len_ops=len(new_ops), payload=(perturbed_cg, new_ops), add_params=add_params)

                        if (node2, node1) in edge_lookup and (node2, node1) not in existing_edges:
                            perturbed_cg = cg.copy()
                            perturbed_cg = add_edge(perturbed_cg, (node2, node1), **G.edges[node2, node1])

                            perturbation_cost = 1
                            new_ops = ops + [("add_edge", (node2, node1))]

                            _heap_push(Q, cost=cost + perturbation_cost, similarity=similarity, len_ops=len(new_ops), payload=(perturbed_cg, new_ops), add_params=add_params)

    ############################################################################################################


def save_operations_to_json(
    ops: list,
    question: str,
    ground_truth: str,
    original_answer: str,
    perturbed_answer: str,
    answer_similarity: float,
    original_subgraph,
    perturbed_subgraph,
    output_dir: str = "src/counterfactuals/results",
    filename: str = None,
    found: bool = True,
    cost: float = 0.0,
    llm_calls: int = 0,
    current_ops: list=[],
    mode: str = "ff",
    total_time: float = 0.0,
    algo_time: float = 0.0,
    setup_time: float = 0.0,
    index_time: float = 0.0,
    pre_llm_time: float = 0.0,
):

    if current_ops == ["add_node", "add_edge", "delete_node", "delete_edge"]:
        output_dir = f"{output_dir}/{dataset}/all_ops_{mode}"
    elif current_ops == ["delete_node", "delete_edge"]:
        output_dir = f"{output_dir}/{dataset}/delete_ops_{mode}"
    elif current_ops == ["add_node", "add_edge"]:
        output_dir = f"{output_dir}/{dataset}/add_ops_{mode}"

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
            "ground_truth": ground_truth,
            "original": original_answer,
            "perturbed": perturbed_answer,
            "similarity": round(answer_similarity, 6)
        },
        "original_subgraph": subgraph_to_dict(original_subgraph),
        "perturbed_subgraph": subgraph_to_dict(perturbed_subgraph),
        "timestamp": datetime.now().isoformat(),
        "llm_calls": llm_calls,
        "mode": mode,
        "timings": {
            "setup_seconds":     round(setup_time, 5),
            "total_seconds":     round(total_time, 5),
            "retrieval_seconds": round(pre_llm_time, 5),
            "index_seconds":     round(index_time, 5),
            "algorithm_seconds": round(algo_time, 5),
            "llm_seconds":       round(total_time - algo_time - index_time - pre_llm_time, 5),
        },
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Operations saved to: {filepath}")
    return filepath


_ALL_OPS = ["delete_node", "delete_edge", "add_node", "add_edge"]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_2",
        description="Run counterfactual search over a LightRAG-backed KG (timer-instrumented).",
    )
    p.add_argument("--dataset", choices=DATASETS, default="synthetic",
                   help="Dataset name; selects working_dir and embedding indices.")
    p.add_argument("--input", default=None,
                   help="Path to comparison JSON (defaults to benchmark/results/comparison_<dataset>_<top-k>.json).")
    p.add_argument("--mode", choices=["ff", "ft", "tf"], default="ff",
                   help="CFE flip direction: ff (corrective F→F), ft (breaking T→F), tf (corrective T→F).")
    p.add_argument("--rag-mode", choices=["hybrid", "local", "global", "naive"], default="hybrid",
                   help="LightRAG retrieval mode used by retrieve_subgraph.")
    p.add_argument("--top-k", type=int, default=2, help="LightRAG retrieval top_k.")
    p.add_argument("--ops", default=",".join(_ALL_OPS),
                   help="Comma-separated operations to enable. Subset of: " + ",".join(_ALL_OPS))
    p.add_argument("--max-cost", type=int, default=20, help="Cost budget c_max.")
    p.add_argument("--max-llm-calls", type=int, default=200, help="LLM-call budget.")
    p.add_argument("--unit-cost", action="store_true", help="Use unit-cost variant of edit costs.")
    p.add_argument("--adm", type=int, choices=[1, 2, 3], default=2,
                   help="Add-mode variant (1=distance-based, 2/3=alternative add expansions).")
    p.add_argument("--psp", action="store_true",
                   help="Enable Pivotal-Star Probe (T→F only, requires delete_node in --ops).")
    p.add_argument("--psp-k", type=int, default=5, help="Top-K pivots for PSP.")
    p.add_argument("--output-dir", default="src/counterfactuals/results",
                   help="Directory for saved counterfactual JSON results.")
    p.add_argument("--add-heuristic", choices=["none", "tier", "blend"], default="none",
                   help="Within-tier ordering heuristic for additions.")
    p.add_argument("--tier-width", type=float, default=1.0,
                   help="Cost-tier width for --add-heuristic=tier (>0). Default 1.0.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Blend weight for --add-heuristic=blend (priority = cost - alpha*similarity). Default 0.5.")
    return p


def _parse_ops(spec: str) -> list:
    ops = [o.strip() for o in spec.split(",") if o.strip()]
    bad = [o for o in ops if o not in _ALL_OPS]
    if bad:
        raise SystemExit(f"Unknown ops: {bad}. Allowed: {_ALL_OPS}")
    return ops


async def main(args: argparse.Namespace):
    global adm

    if args.dataset != dataset:
        setup_dataset(args.dataset)

    adm = args.adm
    current_ops = _parse_ops(args.ops)

    add_params = None
    if args.add_heuristic == "tier":
        if args.tier_width <= 0:
            raise SystemExit("--tier-width must be > 0 when --add-heuristic=tier")
        add_params = {"mode": "tier", "tier_width": args.tier_width}
    elif args.add_heuristic == "blend":
        if args.alpha < 0:
            raise SystemExit("--alpha must be >= 0 when --add-heuristic=blend")
        add_params = {"mode": "blend", "alpha": args.alpha}

    input_path = args.input or f"benchmark/results/comparison_{dataset}_{args.top_k}.json"
    with open(input_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    rag = await initialize_lightrag(working_dir=WORKING_DIRS[dataset])

    for idx, r in data["results"].items():
        if r.get("case") != args.mode:
            continue

        question = r["question"]
        ground_truth = r["ground_truth"]

        print(f"\n=== [{idx}] {question} ===")

        ### Total timer (Start):
        total_start = time.perf_counter()

        context = await retrieve_subgraph(
            rag,
            query=question,
            mode=args.rag_mode,
            top_k=args.top_k,
        )

        ### Retrieval time:
        pre_llm_time = time.perf_counter() - total_start

        await find_counterfactuals(
            rag=rag,
            question=question,
            context=context,
            max_cost=args.max_cost,
            max_llm_calls=args.max_llm_calls,
            unit_cost=args.unit_cost,
            current_ops=current_ops,
            ground_truth=ground_truth,
            mode=args.mode,
            use_psp=args.psp,
            psp_k=args.psp_k,
            output_dir=args.output_dir,
            add_params=add_params,
            total_start=total_start,
            setup_time=setup_time,
            pre_llm_time=pre_llm_time,
        )


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    asyncio.run(main(args))
