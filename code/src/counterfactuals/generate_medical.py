from datetime import datetime
#TODO: RAG query
from src.medical.query import query_rag
#TODO: RAG retrieval
from src.medical.retriever import bfs_subgraph, shortest_paths_subgraph
#TODO: Parser
from src.medical.parser import graph_to_context, graph_to_subgraph
from src.parser import subgraph_to_dict
from src.counterfactuals.edit_costs import *
from src.counterfactuals.perturbations import *
from src.embeddings.utils import load_index
from collections import defaultdict
from src.embeddings.query import DIM, build_lookup, build_edge_lookup, get_embedding
from src.embeddings.query import query as embedding_query
from src.llm.utils import sentence_transformer_embed
from src.medical.extract_entities import *
from src.medical.retriever import *
from benchmark.run_medical import load_partition

import heapq
import networkx as nx
import asyncio
import itertools
import os
import time
import json

### Setup ##

import random

random.seed(42)

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


    ### Min-heap
    Q = []
    state_cache = set()
    # heapq.heappush(Q, (0, 0.0, next(counter), (context_graph, [])))
    heapq.heappush(Q, (0, 0, 0.0, next(counter), (context_graph, []))) ## Added operation sequence lenght

    explored_nodes = set()  ## For addition

    while Q:
        # cost, _, _, (cg, ops) = heapq.heappop(Q)
        cost, _, _, _, (cg, ops) = heapq.heappop(Q) ## Added operation sequence lenght

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

            print(f"Current operations: {ops}")

            cg_context = graph_to_context(cg)

            ## LLM interval (start):
            _t0 = time.perf_counter()

            new_response = await query_rag(input_question=question, options=options, context=cg_context)
            score = 1 if new_response == original_answer else 0

            ## LLM interval (end):
            llm_time += time.perf_counter() - _t0

            print(f"Cost: {cost} | New response: {new_response} | Original: {original_answer}")
            print(f"Ground Truth: {ground_truth}")

            if score == 0:
                print(f"Counterfactual Operations: {ops}")

                ### Pure Algorithm timer
                total_time = time.perf_counter() - total_start
                algo_time = total_time - llm_time - index_time
                ##############################################

                parsed_subgraph = graph_to_subgraph(cg)

                save_operations_to_json(
                    ops=ops,
                    question=question,
                    ground_truth=ground_truth,
                    original_answer=original_answer,
                    perturbed_answer=new_response,
                    answer_similarity=0.0,
                    original_subgraph=graph_to_subgraph(context_graph),
                    perturbed_subgraph=graph_to_subgraph(cg),
                    found=True,
                    cost=cost,
                    llm_calls=llm_calls,
                    current_ops=current_ops,
                    mode=mode,

                    total_time=total_time, 
                    algo_time=algo_time,
                    setup_time=setup_time,
                    index_time=index_time,
                    pre_llm_time=pre_llm_time
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
            query_embedding=query_embedding
        )

    ### Exhausted budget timer
    total_time = time.perf_counter() - total_start
    algo_time = total_time - llm_time - index_time

    print(f"Could not find feasible counterfactual explanations.")

    parsed_subgraph = graph_to_subgraph(context_graph)

    save_operations_to_json(
        ops=[],
        question=question,
        ground_truth=ground_truth,
        original_answer=original_answer,
        perturbed_answer=None,
        answer_similarity=0.0,
        original_subgraph=parsed_subgraph,
        perturbed_subgraph=None,
        found=False,
        llm_calls=llm_calls,
        cost=cost,
        current_ops=current_ops,
        mode=mode,

        total_time=total_time,
        algo_time=algo_time,
        setup_time=setup_time,
        index_time=index_time,
        pre_llm_time=pre_llm_time
    )

async def find_corrective_counterfactuals(
    question: str,
    options: str,
    original_answer: str,
    ground_truth: str,
    query_embedding,
    context_graph: nx.DiGraph, 
    max_cost: int = 3,
    max_llm_calls: int = 100,
    unit_cost: bool = False,
    current_ops: list=["delete_node", "delete_edge", "add_node", "add_edge"],
    mode: str = "ff",

    total_start: float = None,
    setup_time: float = 0.0,
    pre_llm_time: float = 0.0
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

    ### Min-heap
    Q = []
    state_cache = set()
    # heapq.heappush(Q, (0, 0.0, next(counter), (context_graph, [])))
    heapq.heappush(Q, (0, 0, 0.0, next(counter), (context_graph, []))) ## Added operation sequence lenght

    explored_nodes = set()  ## For addition

    while Q:
        # cost, _, _, (cg, ops) = heapq.heappop(Q)
        cost, _, _, _, (cg, ops) = heapq.heappop(Q) ## Added operation sequence lenght

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
            score = 1 if new_response == ground_truth else 0

            ## LLM interval (end):
            llm_time += time.perf_counter() - _t0

            print(f"Cost: {cost} | New response: {new_response} | Original: {original_answer}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Operations: {ops}")

            if score == 1:
                print(f"Counterfactual Operations: {ops}")

                ### Pure Algorithm timer
                total_time = time.perf_counter() - total_start
                algo_time = total_time - llm_time - index_time
                ###########################

                parsed_subgraph = graph_to_subgraph(cg)

                save_operations_to_json(
                    ops=ops,
                    question=question,
                    ground_truth=ground_truth,
                    original_answer=original_answer,
                    perturbed_answer=new_response,
                    answer_similarity=0.0,
                    original_subgraph=parsed_subgraph,
                    perturbed_subgraph=graph_to_subgraph(cg),
                    found=True,
                    cost=cost,
                    llm_calls=llm_calls,
                    current_ops=current_ops,
                    mode=mode,

                    total_time=total_time,
                    algo_time=algo_time,
                    setup_time=setup_time,
                    index_time=index_time,
                    pre_llm_time=pre_llm_time
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
        )

    ### Exhausted budget timer
    total_time = time.perf_counter() - total_start
    algo_time = total_time - llm_time - index_time

    print(f"Could not find feasible counterfactual explanations.")

    parsed_subgraph = graph_to_subgraph(context_graph)

    save_operations_to_json(
        ops=[],
        question=question,
        ground_truth=ground_truth,
        original_answer=original_answer,
        perturbed_answer=None,
        answer_similarity=0.0,
        original_subgraph=parsed_subgraph,
        perturbed_subgraph=None,
        found=False,
        llm_calls=llm_calls,
        cost=cost,
        current_ops=current_ops,
        mode=mode,

        total_time=total_time,
        algo_time=algo_time,
        setup_time=setup_time,
        index_time=index_time,
        pre_llm_time=pre_llm_time
    )

async def find_counterfactuals(
    question: str, 
    options: str,
    context_graph: nx.Graph, 
    max_cost=3, 
    max_llm_calls=100, 
    unit_cost: bool=False, 
    current_ops: list=["delete_node", "delete_edge", "replace_node", "replace_edge"], 
    ground_truth: str = "",
    mode: str = "ft",

    total_start: float = None,
    setup_time: float = 0.0,
    pre_llm_time: float = 0.0
):
    
    if total_start is None:
        total_start = time.perf_counter()

    _t0 = time.perf_counter()

    query_embedding = (await sentence_transformer_embed([question]))[0]
    original_answer = await query_rag(input_question=question, options=options, context=graph_to_context(context_graph))

    pre_llm_time += time.perf_counter() - _t0 ## Add query embed and initial answer time

    if mode == "ft":
        await find_breaking_counterfactuals(
            question=question,
            original_answer=original_answer,
            options=options,
            ground_truth=ground_truth,
            query_embedding=query_embedding,
            context_graph=context_graph,
            max_cost=max_cost,
            max_llm_calls=max_llm_calls,
            unit_cost=unit_cost,
            current_ops=current_ops,
            mode=mode,

            total_start=total_start,
            setup_time=setup_time,
            pre_llm_time=pre_llm_time
        )
    elif mode == "ff":
        await find_corrective_counterfactuals(
            question=question,
            original_answer=original_answer,
            options=options,
            ground_truth=ground_truth,
            query_embedding=query_embedding,
            context_graph=context_graph,
            max_cost=max_cost,
            max_llm_calls=max_llm_calls,
            unit_cost=unit_cost,
            current_ops=current_ops,
            mode=mode,

            total_start=total_start,
            setup_time=setup_time,
            pre_llm_time=pre_llm_time
        )
    elif mode == "tf":
        await find_corrective_counterfactuals(
            question=question,
            original_answer=original_answer,
            options=options,
            ground_truth=ground_truth,
            query_embedding=query_embedding,
            context_graph=context_graph,
            max_cost=max_cost,
            max_llm_calls=max_llm_calls,
            unit_cost=unit_cost,
            current_ops=current_ops,
            mode=mode,

            total_start=total_start,
            setup_time=setup_time,
            pre_llm_time=pre_llm_time
        )

async def expand(
    Q, 
    heap_element, 
    node_similarity_index, 
    edge_similarity_index, 
    edge_labels: dict = None,
    unit_cost: bool = False, 
    current_ops: list=["delete_node", "delete_edge", "replace_node", "replace_edge"],
    original_nodes: set = {},
    original_edges: set = {},
    explored_nodes: set = {},
    query_embedding: list = [],
):
    cg: nx.DiGraph
    cost, cg, ops = heap_element

    cg = nx.DiGraph(cg)

    if "delete_node" in current_ops:
        for node in list(cg.nodes):
            ### Feasibility Constraint
            if node in original_nodes:
                # Snapshot before deletion
                nodes_before = set(cg.nodes)
                edges_before = set(cg.edges)

                perturbed_cg = delete_node(cg, node)

                # Exact diff
                deleted_nodes = nodes_before - set(perturbed_cg.nodes)
                deleted_edges = edges_before - set(perturbed_cg.edges)
                
                if unit_cost == False:
                    perturbation_cost = delete_node_cost(cg, node) 
                elif unit_cost == True:
                    perturbation_cost = delete_node_uc(cg, node)

                # new_ops = ops + [("delete_node", node)]
                new_ops = ops + [{"op": "delete_node", "target": node, "type": "action"}] \
                          + [{"op": "delete_node", "target": n, "type": "side_effect"} for n in deleted_nodes - {node}] \
                          + [{"op": "delete_edge", "target": e, "type": "side_effect"} for e in deleted_edges]

                similarity = node_similarity_index.get(node, 0.0)

                num_actions = sum(1 for o in new_ops if o.get("type") == "action")

                if mode == "ff":
                    heapq.heappush(Q, (cost + perturbation_cost, num_actions, similarity, next(counter), (perturbed_cg, new_ops)))
                elif mode == "ft":
                    heapq.heappush(Q, (cost + perturbation_cost, num_actions, -similarity, next(counter), (perturbed_cg, new_ops)))

    if "delete_edge" in current_ops:
        for edge in list(cg.edges):
            ### Feasibility Constraint
            if edge in original_edges:
                nodes_before = set(cg.nodes)
                edges_before = set(cg.edges)

                perturbed_cg = delete_edge(cg, edge)

                deleted_nodes = nodes_before - set(perturbed_cg.nodes)
                deleted_edges = edges_before - set(perturbed_cg.edges)
                
                if unit_cost == False:
                    perturbation_cost = delete_edge_cost(cg, edge)
                elif unit_cost == True:
                    perturbation_cost = delete_edge_uc(cg, edge)

                new_ops = ops + [{"op": "delete_edge", "target": edge, "type": "action"}] \
                          + [{"op": "delete_edge", "target": e, "type": "side_effect"} for e in deleted_edges - {edge}] \
                          + [{"op": "delete_node", "target": n, "type": "side_effect"} for n in deleted_nodes]

                similarity = edge_similarity_index.get(edge, 0.0)

                num_actions = sum(1 for o in new_ops if o.get("type") == "action")

                if mode == "ff":
                    heapq.heappush(Q, (cost + perturbation_cost, num_actions, similarity, next(counter), (perturbed_cg, new_ops)))
                elif mode == "ft":
                    heapq.heappush(Q, (cost + perturbation_cost, num_actions, -similarity, next(counter), (perturbed_cg, new_ops)))

    ############################################################################################################
    ############################# Query-Relevance-based with query relevance for retrieval #####################

    if adm == 2:
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
                        # new_ops = ops + [("add_node", neighbor)]
                        new_ops = ops + [{"op": "add_node", "target": neighbor, "type": "action"}]


                        if (node, neighbor) in edge_lookup and (node, neighbor) not in existing_edges:
                            perturbed_cg = add_edge(perturbed_cg, (node, neighbor), **G.edges[node, neighbor])
                            # new_ops = new_ops + [("add_edge", (node, neighbor))]
                            new_ops = new_ops + [{"op": "add_edge", "target": (node, neighbor), "type": "action"}]


                            perturbation_cost = (1 + (1 - node_similarity_index.get(neighbor))) + (1 + (1 - edge_similarity_index.get((node, neighbor), 0.0)))
                            
                            num_actions = sum(1 for o in new_ops if o.get("type") == "action")
                    
                            # heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))
                            heapq.heappush(Q, (cost + perturbation_cost, num_actions, -similarity, next(counter), (perturbed_cg, new_ops)))
                        
                        if (neighbor, node) in edge_lookup and (neighbor, node) not in existing_edges:
                            perturbed_cg = add_edge(perturbed_cg, (neighbor, node), **G.edges[neighbor, node])
                            # new_ops = new_ops + [("add_edge", (neighbor, node))]
                            new_ops = new_ops + [{"op": "add_edge", "target": (neighbor, node), "type": "action"}]


                            perturbation_cost = (1 + (1 - node_similarity_index.get(neighbor))) + (1 + (1 - edge_similarity_index.get((neighbor, node), 0.0)))

                            num_actions = sum(1 for o in new_ops if o.get("type") == "action")

                            # heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))
                            heapq.heappush(Q, (cost + perturbation_cost, num_actions, -similarity, next(counter), (perturbed_cg, new_ops)))

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
                    # new_ops = ops + [("add_edge", (src, tgt)), ("add_node", src), ("add_node", tgt)]
                    new_ops = ops + [
                        {"op": "add_edge", "target": (src, tgt), "type": "action"},
                        {"op": "add_node", "target": src, "type": "action"},
                        {"op": "add_node", "target": tgt, "type": "action"},
                    ]
                    num_actions = sum(1 for o in new_ops if o.get("type") == "action")
                    # heapq.heappush(Q, (cost + best_cost, len(new_ops), -best_similarity, next(counter), (perturbed_cg, new_ops)))
                    heapq.heappush(Q, (cost + best_cost, num_actions, -best_similarity, next(counter), (perturbed_cg, new_ops)))

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

                            perturbation_cost = (1 + (1 - edge_similarity_index.get((node1, node2), 0.0)))

                            # new_ops = ops + [("add_edge", (node1, node2))]
                            new_ops = ops + [{"op": "add_edge", "target": (node1, node2), "type": "action"}]

                            num_actions = sum(1 for o in new_ops if o.get("type") == "action")

                            # heapq.heappush(Q, (cost+perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))
                            heapq.heappush(Q, (cost+perturbation_cost, num_actions, -similarity, next(counter), (perturbed_cg, new_ops)))

                        if (node2, node1) in edge_lookup and (node2, node1) not in existing_edges:
                            perturbed_cg = cg.copy()
                            perturbed_cg = add_edge(perturbed_cg, (node2, node1), **G.edges[node2, node1])

                            perturbation_cost = (1 + (1 - edge_similarity_index.get((node2, node1), 0.0)))

                            # new_ops = ops + [("add_edge", (node2, node1))]

                            new_ops = ops + [{"op": "add_edge", "target": (node2, node1), "type": "action"}]
                            num_actions = sum(1 for o in new_ops if o.get("type") == "action")

                            # heapq.heappush(Q, (cost+perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))
                            heapq.heappush(Q, (cost+perturbation_cost, num_actions, -similarity, next(counter), (perturbed_cg, new_ops)))

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
    pre_llm_time: float = 0.0
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

    # serialisable_ops = []
    # for op in ops:
    #     if isinstance(op, tuple):
    #         serialisable_ops.append(list(op))
    #     else:
    #         serialisable_ops.append(op)

    serialisable_ops = []
    for op in ops:
        if isinstance(op, dict):
            entry = dict(op)
            if isinstance(entry.get("target"), tuple):
                entry["target"] = list(entry["target"])
            serialisable_ops.append(entry)
        elif isinstance(op, tuple):
            serialisable_ops.append(list(op))
        else:
            serialisable_ops.append(op)

    action_ops = [o for o in serialisable_ops if isinstance(o, dict) and o.get("type") == "action"]


    payload = {
        "question": question,
        "found": found,
        
        # "num_operations": len(serialisable_ops),
        # "operations": serialisable_ops,
        "num_operations": len(action_ops),
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
            "setup_seconds": round(setup_time, 5),
            "total_seconds": round(total_time, 5),
            "retrieval_seconds": round(pre_llm_time, 5),
            "index_seconds": round(index_time, 5),
            "algorithm_seconds": round(algo_time, 5),
            "llm_seconds": round(total_time - algo_time - index_time - pre_llm_time, 5)
        }
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Operations saved to: {filepath}")
    return filepath


async def main():
    global adm
    global mode

    mode = "ff"

    benchmark_data = load_partition(f"datasets/medical/{partition}_ff.json", partition)
    benchmark_lookup = benchmark_data.set_index("id").to_dict(orient="index")

    with open(f"benchmark/bfs/comparison_{dataset}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    operation_sets = [
        # ["delete_node", "delete_edge"],
        ["add_node", "add_edge", "delete_node", "delete_edge"]
    ]

    add_modes = [2]

    for adm in add_modes:
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
                    pre_llm_time=pre_llm_time
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