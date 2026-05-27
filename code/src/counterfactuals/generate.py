from datetime import datetime
from src.query import *
from src.retrieve import *
from src.parser import *
from src.llm_judge import judge_response
from src.counterfactuals.edit_costs import *
from src.counterfactuals.perturbations import *
from src.counterfactuals.utils import compute_answer_similarity, cosine_similarity_norm
from src.embeddings.utils import load_index
from collections import defaultdict
from src.embeddings.query import DIM, build_lookup, get_embedding, build_edge_lookup
from src.embeddings.query import query as embedding_query

import heapq
import networkx as nx
import asyncio
import itertools
import os

### Setup ###

def create_type_index(G: nx.Graph):
    type_index = defaultdict(list)
    for node, data in G.nodes(data=True):
        node_type = data.get("entity_type")
        type_index[node_type].append(node)

    return type_index

counter = itertools.count()

dataset = "hotpotqa"  ### "hotpotqa" or "synthetic"

G = nx.read_graphml(f"KGs/lightrag/{dataset}/graph_chunk_entity_relation.graphml")

type_index = create_type_index(G)

# Node setup
node_index_prefix = f"src/embeddings/{dataset}/node_index"
node_index, node_records, node_embeddings = load_index(node_index_prefix, DIM, 2000)
node_lookup = build_lookup(node_records)

# Edge setup
edge_index_prefix = f"src/embeddings/{dataset}/edge_index"
edge_index, edge_records, edge_embeddings = load_index(edge_index_prefix, DIM, 2000)
edge_lookup = build_edge_lookup(edge_records)

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
    mode: str = "ft"
):
    llm_calls = 0

    ### Lightrag specific
    parsed_subgraph = parse_context(context)
    context_graph = parse_graph(parsed_subgraph)
    #####################
    
    context_graph_nodes = set(context_graph.nodes)
    context_graph_edges = set(context_graph.edges())

    edge_labels = {(u, v): data.get("description", "") for u, v, data in G.edges(data=True)}
    node_similarity_index = create_node_similarity_index(set(G.nodes), query_embedding)
    edge_similarity_index = await create_edge_similarity_index(edge_labels, query_embedding)

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

            new_response = await query(rag, cg_context, question)

            print(f"Cost: {cost} | New response: {new_response} | Original: {original_answer}")
            print(f"Ground Truth: {ground_truth}")

            score = await judge_response(question, new_response, original_answer)

            if score == 0:
                print(f"Counterfactual Operations: {ops}")

                answer_similarity = await compute_answer_similarity(original_answer, new_response)
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
                    found=True,
                    cost=cost,
                    llm_calls=llm_calls,
                    current_ops=current_ops,
                    mode=mode
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
        found=False,
        llm_calls=llm_calls,
        cost=cost,
        current_ops=current_ops,
        mode=mode
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
    mode: str = "ff"
):
    llm_calls = 0

    ### Lightrag specific
    parsed_subgraph = parse_context(context)
    context_graph = parse_graph(parsed_subgraph)
    #####################
    
    context_graph_nodes = set(context_graph.nodes)
    context_graph_edges = set(context_graph.edges())

    edge_labels = {(u, v): data.get("description", "") for u, v, data in G.edges(data=True)}
    node_similarity_index = create_node_similarity_index(set(G.nodes), query_embedding)
    edge_similarity_index = await create_edge_similarity_index(edge_labels, query_embedding)

    ### Min-heap
    Q = []
    state_cache = set()
    # heapq.heappush(Q, (0, 0.0, next(counter), (context_graph, [])))
    heapq.heappush(Q, (0, 0, 0.0, next(counter), (context_graph, []))) ## Added operation sequence lenght

    explored_nodes = set()  ## For addition
    edge_embedding_cache = {}

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

            new_response = await query(rag, cg_context, question)

            print(f"Cost: {cost} | New response: {new_response} | Original: {original_answer}")
            print(f"Ground Truth: {ground_truth}")

            score = await judge_response(question, new_response, ground_truth)

            if score == 1:
                print(f"Counterfactual Operations: {ops}")

                answer_similarity = await compute_answer_similarity(ground_truth, new_response)
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
                    found=True,
                    cost=cost,
                    llm_calls=llm_calls,
                    current_ops=current_ops,
                    mode=mode
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
            edge_embedding_cache=edge_embedding_cache
        )

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
        found=False,
        llm_calls=llm_calls,
        cost=cost,
        current_ops=current_ops,
        mode=mode
    )

async def find_counterfactuals(
    rag, 
    question: str, 
    context, 
    max_cost=3, 
    max_llm_calls=100, 
    unit_cost: bool=False, 
    current_ops: list=["delete_node", "delete_edge", "replace_node", "replace_edge"], 
    ground_truth: str = "",
    mode: str = "ft"
):
    query_embedding = (await sentence_transformer_embed([question]))[0]
    original_answer = await query(rag, context, question)

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
            mode=mode
        )
    elif mode == "ff":
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
            mode=mode
        )
    elif mode == "tf":
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
            mode=mode
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
    edge_embedding_cache: dict = None,
):
    cg: nx.DiGraph
    cost, cg, ops = heap_element

    undirected: nx.Graph = cg.to_undirected()
    cut_vertices = set(nx.articulation_points(cg.to_undirected()))
    cut_edges = set(nx.bridges(cg.to_undirected()))

    if "delete_node" in current_ops:
        # Allow if not a cut vertex, OR if it is a cut vertex but all neighbors
        # would become isolated (meaning no real split, just singleton cleanup)
        for node in list(cg.nodes):
            ### Feasibility Constraint
            if node in original_nodes:
                # if node in cut_vertices:
                #     neighbors = list(undirected.neighbors(node))
                    
                #     would_isolate = {n for n in neighbors if undirected.degree(n) == 1}
                #     nodes_to_remove = {node} | would_isolate
                #     residual = undirected.copy()
                #     residual.remove_nodes_from(nodes_to_remove)

                #     components_before = nx.number_connected_components(undirected)
                #     components_after = nx.number_connected_components(residual)

                #     if components_after > components_before:
                #         continue

                perturbed_cg = delete_node(cg, node)
                
                if unit_cost == False:
                    perturbation_cost = delete_node_cost(cg, node) 
                elif unit_cost == True:
                    perturbation_cost = delete_node_uc(cg, node)

                new_ops = ops + [("delete_node", node)]

                similarity = node_similarity_index.get(node, 0.0)

                # heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))
                # heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))

                if mode == "ff":
                    heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), similarity, next(counter), (perturbed_cg, new_ops)))
                elif mode == "ft":
                    heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))

    if "delete_edge" in current_ops:
        ### Updated Delete Edge
        # Allow if not a cut edge, OR if it is a cut edge but both endpoints
        # would become isolated (meaning no real split, just singleton cleanup)
        for edge in list(cg.edges):
            ### Feasibility Constraint
            if edge in original_edges:
                # if edge in cut_edges:
                #     u, v = edge[0], edge[1]
                #     would_split = undirected.degree(u) > 1 and undirected.degree(v) > 1
                #     if would_split:
                #         continue

                perturbed_cg = delete_edge(cg, edge)
                
                if unit_cost == False:
                    perturbation_cost = delete_edge_cost(cg, edge)
                elif unit_cost == True:
                    perturbation_cost = delete_edge_uc(cg, edge)

                new_ops = ops + [("delete_edge", edge)]

                similarity = edge_similarity_index.get(edge, 0.0)

                # heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))
                # heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))

                if mode == "ff":
                    heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), similarity, next(counter), (perturbed_cg, new_ops)))
                elif mode == "ft":
                    heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))

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

                            heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))
                        
                        if (neighbor, node) in edge_lookup and (neighbor, node) not in existing_edges:
                            perturbed_cg = add_edge(perturbed_cg, (neighbor, node), **G.edges[neighbor, node])
                            new_ops = new_ops + [("add_edge", (neighbor, node))]

                            if unit_cost == False:
                                perturbation_cost = add_node_cost(cg, node_embeddings, node_lookup, edge_embeddings, edge_lookup, neighbor, (neighbor, node))
                            elif unit_cost == True:
                                perturbation_cost = 2

                            heapq.heappush(Q, (cost + perturbation_cost,  len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))


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
                    heapq.heappush(Q, (cost + best_cost, len(new_ops), -best_similarity, next(counter), (perturbed_cg, new_ops)))

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

                            heapq.heappush(Q, (cost+perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))

                        if (node2, node1) in edge_lookup and (node2, node1) not in existing_edges:
                            perturbed_cg = cg.copy()
                            new_ops = ops + [("add_edge", (node2, node1))]

                            perturbed_cg = add_edge(perturbed_cg, (node2, node1), **G.edges[node2, node1])

                            if unit_cost == False:
                                perturbation_cost = add_edge_cost(cg, edge_embeddings, edge_lookup, (node2, node1))
                            elif unit_cost == True:
                                perturbation_cost = add_edge_uc()

                            heapq.heappush(Q, (cost+perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))

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

                            heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))
                        
                        if (neighbor, node) in edge_lookup and (neighbor, node) not in existing_edges:
                            perturbed_cg = add_edge(perturbed_cg, (neighbor, node), **G.edges[neighbor, node])
                            new_ops = new_ops + [("add_edge", (neighbor, node))]

                            perturbation_cost = (1 + (1 - node_similarity_index.get(neighbor))) + (1 + (1 - edge_similarity_index.get((node, neighbor), 0.0)))

                            heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))

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
                    new_ops = ops + [("add_edge", (src, tgt))]
                    heapq.heappush(Q, (cost + best_cost, len(new_ops), -best_similarity, next(counter), (perturbed_cg, new_ops)))

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

                            new_ops = ops + [("add_edge", (node1, node2))]

                            heapq.heappush(Q, (cost+perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))

                        if (node2, node1) in edge_lookup and (node2, node1) not in existing_edges:
                            perturbed_cg = cg.copy()
                            perturbed_cg = add_edge(perturbed_cg, (node2, node1), **G.edges[node2, node1])

                            perturbation_cost = (1 + (1 - edge_similarity_index.get((node2, node1), 0.0)))

                            new_ops = ops + [("add_edge", (node2, node1))]

                            heapq.heappush(Q, (cost+perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))

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

                            heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))
                        
                        if (neighbor, node) in edge_lookup and (neighbor, node) not in existing_edges:
                            perturbed_cg = add_edge(perturbed_cg, (neighbor, node), **G.edges[neighbor, node])
                            new_ops = new_ops + [("add_edge", (neighbor, node))]

                            perturbation_cost = 2

                            heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))

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
                    heapq.heappush(Q, (cost + 3, len(new_ops), -best_similarity, next(counter), (perturbed_cg, new_ops)))

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

                            heapq.heappush(Q, (cost+perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))

                        if (node2, node1) in edge_lookup and (node2, node1) not in existing_edges:
                            perturbed_cg = cg.copy()
                            perturbed_cg = add_edge(perturbed_cg, (node2, node1), **G.edges[node2, node1])

                            perturbation_cost = 1
                            new_ops = ops + [("add_edge", (node2, node1))]

                            heapq.heappush(Q, (cost+perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))

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
    mode: str = "ff"
):

    if current_ops == ["add_node", "add_edge", "delete_node", "delete_edge"]:
        output_dir = f"{output_dir}/{dataset}/ff-case-{adm}/all_ops_{mode}"
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
        "mode": mode
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Operations saved to: {filepath}")
    return filepath


async def main():
    global adm
    global mode

    rag = await initialize_lightrag(working_dir=WORKING_DIR_HOTPOTQA)
    
    with open(f"benchmark/results/comparison_{dataset}_2.json", "r", encoding="utf-8") as results:
        data = json.load(results)

    operation_sets = [
        ["add_node", "add_edge", "delete_node", "delete_edge"]
        # ["add_node", "add_edge"]
        # ["delete_node", "delete_edge"]
    ]

    mode = "ff"
    # add_modes = [1, 2 ,3]
    add_modes = [2]

    for adm in add_modes:
        for op_set in operation_sets:
            results = data["results"]
            for idx, r in results.items():
                question = r["question"]
                case = r["case"]

                ground_truth = r["ground_truth"]

                if case != mode:
                    continue

                print(f"\n=== [{idx}] {question} ===")

                context = await retrieve_subgraph(
                    rag, 
                    query=question, 
                    mode="hybrid", 
                    top_k=2
                )

                await find_counterfactuals(
                    rag=rag, 
                    question=question, 
                    context=context, 
                    max_cost=20, 
                    max_llm_calls=200, 
                    unit_cost=False, 
                    current_ops=op_set, 
                    ground_truth=ground_truth,
                    mode=mode
                )

if __name__ == "__main__":
    asyncio.run(main())






# if "add_node" in current_ops:
#         existing_nodes = set(cg.nodes)
#         existing_edges = set(cg.edges())
#         candidate_nodes_for_expansion = existing_nodes - explored_nodes



#         candidates_pushed = 0

#         for node in candidate_nodes_for_expansion:
#             neighbors = list(G.neighbors(node))
            
#             similarity = node_similarity_index.get(node, 0.0)

#             for neighbor in neighbors:

#                 if neighbor not in existing_nodes:
#                     perturbed_cg = add_node(cg, neighbor, **G.nodes[neighbor])
#                     new_ops = ops + [("add_node", neighbor)]

#                     if (node, neighbor) in edge_lookup and (node, neighbor) not in existing_edges:
#                         perturbed_cg = add_edge(perturbed_cg, (node, neighbor), **G.edges[node, neighbor])
#                         new_ops = new_ops + [("add_edge", (node, neighbor))]
                    
#                     if (neighbor, node) in edge_lookup and (neighbor, node) not in existing_edges:
#                         perturbed_cg = add_edge(perturbed_cg, (neighbor, node), **G.edges[neighbor, node])
#                         new_ops = new_ops + [("add_edge", (neighbor, node))]

#                     if unit_cost == False:
#                         perturbation_cost = add_node_cost(perturbed_cg, node_embeddings, node_lookup, edge_embeddings, edge_lookup, neighbor)
#                     elif unit_cost == True:
#                         perturbation_cost = add_node_uc(perturbed_cg, neighbor)

#                     heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

#                     candidates_pushed += 1

#             explored_nodes.add(node)
        
#         if candidates_pushed == 0: ### ===> Do not have anyone to explore/expand
#             relevant_nodes = embedding_query(node_index, node_records, query_embedding, k=10)

#             ### Relevant nodes are sorted in descending similarity order (most relevant first).
#             for node in relevant_nodes: 
#                 node_name = node["name"]
#                 if node_name in existing_nodes:
#                     continue

#                 ### Add New Component Anchor:
#                 perturbed_cg = add_node(cg, node_name, **G.nodes[node_name])

#                 new_ops = ops + [("add_node", node_name)]

#                 similarity = node_similarity_index.get(node_name, 0.0) ### ???

#                 neighbors = list(G.neighbors(node_name))
                
#                 for neighbor in neighbors:

#                     if neighbor not in existing_nodes:
#                         perturbed_cg = add_node(perturbed_cg, neighbor, **G.nodes[neighbor])

#                         if (node_name, neighbor) in edge_lookup and (node_name, neighbor) not in existing_edges:
#                             perturbed_cg = add_edge(perturbed_cg, (node_name, neighbor), **G.edges[node_name, neighbor])
#                             new_ops = new_ops + [("add_edge", (node_name, neighbor))]
                        
#                         if (neighbor, node_name) in edge_lookup and (neighbor, node_name) not in existing_edges:
#                             perturbed_cg = add_edge(perturbed_cg, (neighbor, node_name), **G.edges[neighbor, node_name])
#                             new_ops = new_ops + [("add_edge", (neighbor, node_name))]


#                 ### Calculate cost of adding new anchor
#                 if unit_cost == False:
#                     perturbation_cost = add_node_cost(perturbed_cg, node_embeddings, node_lookup, edge_embeddings, edge_lookup, node_name) ### Not sure if I have to use cg or perturbed_cg
#                 elif unit_cost == True:
#                     perturbation_cost = add_node_uc(perturbed_cg, node_name)                                                               ### Not sure if I have to use cg or perturbed_cg

#                 heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

#                 explored_nodes.add(node_name)

#                 break

#     if "add_edge" in current_ops:
#         existing_edges = set(cg.edges())
#         existing_nodes = set(cg.nodes)

#         for node in existing_nodes:
#             available_edges = set(G.edges(node))

#             for edge in available_edges:
#                 node1, node2 = edge

#                 similarity = edge_similarity_index.get(edge, 0.0) ### ???

#                 # if node1 in existing_nodes and node2 in existing_nodes:
#                 if node1 in existing_nodes:
#                     if (node1, node2) in edge_lookup and (node1, node2) not in existing_edges:
#                         perturbed_cg = cg.copy()
#                         perturbation_cost = 0
#                         new_ops = ops + [("add_edge", (node1, node2))]

#                         if node2 not in set(perturbed_cg.nodes):
#                             perturbed_cg.add_node(node2, **G.nodes[node2])
#                             perturbation_cost = 1 ### ???

#                             new_ops = ops + [("add_node", node2), ("add_edge", (node1, node2))]

#                         perturbed_cg = add_edge(perturbed_cg, (node1, node2), **G.edges[node1, node2])

#                         if unit_cost == False:
#                             perturbation_cost += add_edge_cost(perturbed_cg, edge_embeddings, edge_lookup, (node1, node2))
#                         elif unit_cost == True:
#                             perturbation_cost += add_edge_uc()

#                         heapq.heappush(Q, (cost+perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

#                     if (node2, node1) in edge_lookup and (node2, node1) not in existing_edges:
#                         perturbed_cg = cg.copy()
#                         perturbation_cost = 0
#                         new_ops = ops + [("add_edge", (node2, node1))]

#                         if node1 not in set(perturbed_cg.nodes):
#                             perturbed_cg.add_node(node1, **G.nodes[node1])
#                             perturbation_cost = 1 ### ???

#                             new_ops = ops + [("add_node", node1), ("add_edge", (node2, node1))]

#                         perturbed_cg = add_edge(perturbed_cg, (node2, node1), **G.edges[node2, node1])

#                         if unit_cost == False:
#                             perturbation_cost += add_edge_cost(perturbed_cg, edge_embeddings, edge_lookup, (node2, node1))
#                         elif unit_cost == True:
#                             perturbation_cost += add_edge_uc()

#                         heapq.heappush(Q, (cost+perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

#     #########################################################################################