from datetime import datetime
from src.medical.query import query_rag
from src.medical.retriever import bfs_subgraph, shortest_paths_subgraph
from src.medical.parser import graph_to_context, graph_to_subgraph
from src.parser import subgraph_to_dict
from src.llm_judge import judge_response
from src.counterfactuals.edit_costs import *
from src.counterfactuals.perturbations import *
from src.counterfactuals.utils import compute_answer_similarity, cosine_similarity_norm
from src.embeddings.query import get_embedding, DIM, build_edge_lookup, build_lookup
from src.embeddings.utils import load_index
from src.embeddings.query import query as embedding_query
from benchmark.run_medical import load_partition
from src.llm.utils import sentence_transformer_embed
from src.medical.extract_entities import *
from src.medical.retriever import *
from collections import defaultdict

import heapq
import json
import math
import networkx as nx
import asyncio
import itertools
import os
import time
import random

random.seed(42)

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

### Similarity Index (Node/Edge)

def create_type_index(G: nx.Graph):
    type_index = defaultdict(list)
    for node, data in G.nodes(data=True):
        node_type = data.get("entity_type")
        type_index[node_type].append(node)

    return type_index

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
    preds = set(cg.predecessors(pivot))
    succs = set(cg.successors(pivot))
    neighbors = preds | succs
    singleton_neighbors = {
        u for u in neighbors
        if (set(cg.predecessors(u)) | set(cg.successors(u))) <= {pivot}
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
    question: str,
    options: str,
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
        new_response = await query_rag(input_question=question, options=options, context=cg_context)
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
    question: str,
    options: str, 
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
                new_response = await query_rag(input_question=question, options=options, context=cg_context)
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
    question: str,
    options: str,
    original_answer: str,
    ground_truth: str,
    query_embedding,
    context_graph: nx.Graph,
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
            question=question,
            options=options,
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
                    question=question,
                    options=options,
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


            save_operations_to_json(
                ops=best_ops,
                question=question,
                ground_truth=ground_truth,
                original_answer=original_answer,
                perturbed_answer=best_response,
                answer_similarity=answer_similarity,
                original_subgraph=graph_to_subgraph(context_graph),
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
            new_response = await query_rag(input_question=question, options=options, context=cg_context)
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

                save_operations_to_json(
                    ops=ops,
                    question=question,
                    ground_truth=ground_truth,
                    original_answer=original_answer,
                    perturbed_answer=new_response,
                    answer_similarity=answer_similarity,
                    original_subgraph=graph_to_subgraph(context_graph),
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
        original_subgraph=graph_to_subgraph(context_graph),
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
    question: str,
    options: str,
    original_answer: str,
    ground_truth: str,
    query_embedding,
    context_graph: nx.Graph,
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

            new_response = await query_rag(input_question=question, options=options, context=cg_context)

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
                    original_subgraph=graph_to_subgraph(context_graph),
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
        original_subgraph=graph_to_subgraph(context_graph),
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
    question: str,
    options: str, 
    context_graph: nx.Graph,
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
    original_answer = await query_rag(input_question=question, options=options, context=graph_to_context(context_graph))
    pre_llm_time += time.perf_counter() - _t0

    if mode == "ft":
        await find_breaking_counterfactuals(
            question=question,
            options=options,
            original_answer=original_answer,
            ground_truth=ground_truth,
            query_embedding=query_embedding,
            context_graph=context_graph,
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
            question=question,
            options=options,
            original_answer=original_answer,
            ground_truth=ground_truth,
            query_embedding=query_embedding,
            context_graph=context_graph,
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

    ############################# Query-Relevance-based with query relevance for retrieval #####################

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


async def main():
    global mode

    mode = "ft"

    benchmark_data = load_partition(f"datasets/medical/{partition}_ff.json", partition)
    benchmark_lookup = benchmark_data.set_index("id").to_dict(orient="index")

    with open(f"benchmark/bfs/comparison_{dataset}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    operation_sets = [
        ["delete_node", "delete_edge"],
        # ["add_node", "add_edge", "delete_node", "delete_edge"]
    ]

    for op_set in operation_sets:
        results = data["results"]
        for idx, r in results.items():
            case = r["case"]

            if case != mode:
                continue

            bench = benchmark_lookup.get(idx)
            if bench is None:
                print(f"[{idx}] Not found in benchmark, skipping.")
                continue

            question     = bench["questions"]
            options      = bench["options"]
            ground_truth = bench["answers"]

            print(f"\n=== [{idx}] {question} ===")

            total_start = time.perf_counter()

            entities = await extract_entities(input_text=question)

            validated_entities = validate_entity(G, entities)
            found_entities = validated_entities["found"]
            not_found_entities = validated_entities["not_found"]
            
            seed_nodes = found_entities
            for ent in not_found_entities:
                most_similar_node_id = find_similar_node_id(node_index, node_records, ent)
                if most_similar_node_id:
                    seed_nodes.append(most_similar_node_id)

            context_graph = bfs_subgraph(G=G, seed_nodes=seed_nodes, depth=1)
            context_graph = prune_subgraph(context_graph, query_text=question, top_k_nodes=5, top_k_edges=5, lookup=node_lookup, embeddings=node_embeddings)  # ← prune before context

            pre_llm_time = time.perf_counter() - total_start

            await find_counterfactuals(
                question=question,
                options=options,
                context_graph=context_graph,
                max_cost=20,
                max_llm_calls=200,
                unit_cost=False,
                current_ops=op_set,
                ground_truth=ground_truth,
                mode=mode,
                total_start=total_start,
                setup_time=setup_time,
                pre_llm_time=pre_llm_time,

                use_psp=True,
                psp_k=5,
                output_dir=f"src/counterfactuals/results/psp/medical_{partition}_1/ft_delete_psp_k_5"
            )

if __name__ == "__main__":
    counter = itertools.count()

    partition = "bioasq"

    ### Setup - Timer (Start):
    _setup_start = time.perf_counter()

    dataset = f"medical_{partition}_1"
    G = nx.read_graphml(f"KGs/medical/graph_chunk_entity_relation_digraph.graphml")
    type_index = create_type_index(G)

    # Node setup
    node_index_prefix = f"src/embeddings/medical/node_index"
    node_index, node_records, node_embeddings = load_index(node_index_prefix, DIM, 2000)
    node_lookup = build_lookup(node_records)

    # Edge setup
    edge_index_prefix = f"src/embeddings/medical/edge_index"
    edge_index, edge_records, edge_embeddings = load_index(edge_index_prefix, DIM, 2000)
    edge_lookup = build_edge_lookup(edge_records)

    ### Setup - Timer (End)
    setup_time = time.perf_counter() - _setup_start
    print(f"Setup time: {setup_time:.3f}s")

    asyncio.run(main())