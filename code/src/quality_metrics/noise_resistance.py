from datetime import datetime
from src.query import *
from src.retrieve import *
from src.parser import *
from src.llm_judge import judge_response
from src.counterfactuals.edit_costs import *
from src.counterfactuals.perturbations import *
from src.counterfactuals.utils import compute_answer_similarity, cosine_similarity_norm
from src.embeddings.query import query as embedding_query

from src.dataset_setup import (
    WORKING_DIRS,
    DATASETS,
    setup_dataset as _shared_setup_dataset,
)

import argparse
import heapq
import json
import networkx as nx
import asyncio
import itertools
import os
import random


### Setup ###

counter = itertools.count()

# dataset: str = "synthetic"
G = None
type_index = None
node_index = node_records = node_embeddings = node_lookup = None
edge_index = edge_records = edge_embeddings = edge_lookup = None


def setup_dataset(name: str):
    """(Re)bind module-level dataset globals via the shared loader."""
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


# setup_dataset("synthetic")

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


def add_random_noise_nodes(cg: nx.Graph, G: nx.Graph, n: int = None, noise_pct: float = None, seed: int = None):
    if seed is not None:
        random.seed(seed)

    if noise_pct is not None:
        n = max(1, round(len(cg.nodes()) * noise_pct))
    elif n is None:
        raise ValueError("Either n or noise_pct must be provided.")

    cg = cg.copy()
    ops_applied = []

    candidate_nodes = [node for node in G.nodes() if node not in cg.nodes()]

    if not candidate_nodes:
        print("No candidate nodes available in G outside of cg.")
        return cg, ops_applied

    # eligible_anchors = [node for node in cg.nodes() if cg.degree(node) >= 2]
    eligible_anchors = [node for node in cg.nodes()]

    if not eligible_anchors:
        print("No anchor nodes with degree >= 2 found. Skipping noise.")
        return cg, ops_applied

    all_G_edges = list(G.edges(data=True))
    sampled_nodes = random.sample(candidate_nodes, min(n, len(candidate_nodes)))

    for new_node in sampled_nodes:
        anchor = random.choice(eligible_anchors)
        node_attr = G.nodes[new_node]

        _, _, random_edge_attr = random.choice(all_G_edges)

        cg.add_node(new_node, **node_attr)
        cg.add_edge(new_node, anchor, **random_edge_attr)

        ops_applied.append(("add_node", new_node))
        ops_applied.append(("add_edge", (new_node, anchor), random_edge_attr))

    print(f"Added {len(ops_applied)} noise node(s) with random edge attributes")
    return cg, ops_applied


async def find_breaking_counterfactuals(
    rag,
    question: str,
    original_answer: str,
    ground_truth: str,
    query_embedding,
    context: str,
    noise_pct: float,
    max_cost: int = 3,
    max_llm_calls: int = 100,
    unit_cost: bool = False,
    current_ops: list=["delete_node", "delete_edge"],
    mode: str = "ft",
    seed: int = 1,
    output_dir: str = "src/counterfactuals/robustness",
):
    llm_calls = 0

    ### Lightrag specific
    parsed_subgraph = parse_context(context)
    context_graph = parse_graph(parsed_subgraph)
    #####################

    ### TEST Noisy Graph against original
    noisy_cg, noise_ops = add_random_noise_nodes(context_graph, G, noise_pct=noise_pct, seed=seed)
    noisy_context = graph_to_context(noisy_cg)
    noisy_response = await query(rag, noisy_context, question)

    # Judge noisy response against the original unperturbed answer
    noise_score = await judge_response(question, noisy_response, original_answer)
    noise_robust = noise_score != 0  # True => noise didn't break the original answer, so proceed with CFE

    noise_metadata = {
        "ops": [list(op) if isinstance(op, tuple) else op for op in noise_ops],
        "num_ops": len(noise_ops),
        "score_after_noise": noise_score,
        "noise_robust": noise_robust,
    }

    print(f"Score (Original vs Noisy): {noise_score} | Robust: {noise_robust}")

    if not noise_robust:
        # Noise alone broke the answer — system is fragile, skip CF search
        print("Answer not robust to noise. Skipping counterfactual search.")
        save_operations_to_json(
            ops=[],
            question=question,
            original_answer=original_answer,
            ground_truth=ground_truth,
            perturbed_answer=noisy_response,
            answer_similarity=0.0,
            original_subgraph=parsed_subgraph,
            perturbed_subgraph=graph_to_subgraph(noisy_cg),
            noisy_subgraph=parse_context(noisy_context),
            noise_metadata=noise_metadata,
            noise_p=noise_pct,
            output_dir=output_dir,
            found=False,
            cost=0.0,
            llm_calls=1,
            mode=mode,
        )
        return

    # Noise didn't affect the answer — proceed with noisy graph as new baseline
    context_graph = noisy_cg
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
                    noisy_subgraph=parse_context(noisy_context),
                    noise_metadata=noise_metadata,
                    noise_p=noise_pct,
                    output_dir=output_dir,
                    found=True,
                    cost=cost,
                    llm_calls=llm_calls,
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
            mode=mode,
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
        noisy_subgraph=parse_context(noisy_context),
        noise_metadata=noise_metadata,
        noise_p=noise_pct,
        output_dir=output_dir,
        found=False,
        llm_calls=llm_calls,
        cost=cost,
        mode=mode
    )

async def find_corrective_counterfactuals(
    rag,
    question: str,
    original_answer: str,
    ground_truth: str,
    query_embedding,
    context: str,
    noise_pct: float,
    max_cost: int = 3,
    max_llm_calls: int = 100,
    unit_cost: bool = False,
    current_ops: list=["delete_node", "delete_edge", "add_node", "add_edge"],
    mode: str = "ff",
    seed: int = 1,
    output_dir: str = "src/counterfactuals/robustness",
):
    llm_calls = 0

    ### Lightrag specific
    parsed_subgraph = parse_context(context)
    context_graph = parse_graph(parsed_subgraph)
    #####################

    ### TEST Noisy Graph against original
    noisy_cg, noise_ops = add_random_noise_nodes(context_graph, G, noise_pct=noise_pct, seed=seed)
    noisy_context = graph_to_context(noisy_cg)
    noisy_response = await query(rag, noisy_context, question)

    # Judge noisy response against the original unperturbed answer
    noise_score = await judge_response(question, noisy_response, original_answer)
    noise_robust = noise_score != 0  # True => noise didn't break the original answer, so proceed with CFE

    noise_metadata = {
        "ops": [list(op) if isinstance(op, tuple) else op for op in noise_ops],
        "num_ops": len(noise_ops),
        "score_after_noise": noise_score,
        "noise_robust": noise_robust,
    }

    print(f"Score (Original vs Noisy): {noise_score} | Robust: {noise_robust}")

    if not noise_robust:
        # Noise alone broke the answer — system is fragile, skip CF search
        print("Answer not robust to noise. Skipping counterfactual search.")
        save_operations_to_json(
            ops=[],
            question=question,
            ground_truth=ground_truth,
            original_answer=original_answer,
            perturbed_answer=noisy_response,
            answer_similarity=0.0,
            original_subgraph=parsed_subgraph,
            perturbed_subgraph=graph_to_subgraph(noisy_cg),
            noisy_subgraph=parse_context(noisy_context),
            found=False,
            cost=0.0,
            llm_calls=1,
            noise_metadata=noise_metadata,
            noise_p=noise_pct,
            output_dir=output_dir,
            mode=mode,
        )
        return

    # Noise didn't affect the answer — proceed with noisy graph as new baseline
    context_graph = noisy_cg
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
                    noisy_subgraph=parse_context(noisy_context),
                    noise_metadata=noise_metadata,
                    noise_p=noise_pct,
                    output_dir=output_dir,
                    found=True,
                    cost=cost,
                    llm_calls=llm_calls,
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
            edge_embedding_cache=edge_embedding_cache,
            mode=mode,
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
        noisy_subgraph=parse_context(noisy_context),
        noise_metadata=noise_metadata,
        noise_p=noise_pct,
        output_dir=output_dir,
        found=False,
        llm_calls=llm_calls,
        cost=cost,
        mode=mode
    )

async def find_counterfactuals(
    rag,
    question: str,
    context,
    noise_pct: float,
    max_cost=3,
    max_llm_calls=100,
    unit_cost: bool=False,
    current_ops: list=["delete_node", "delete_edge", "replace_node", "replace_edge"],
    ground_truth: str = "",
    mode: str = "ft",
    seed: int = 1,
    output_dir: str = "src/counterfactuals/robustness",
):
    query_embedding = (await sentence_transformer_embed([question]))[0]
    original_answer = await query(rag, context, question)

    common = dict(
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
        noise_pct=noise_pct,
        seed=seed,
        output_dir=output_dir,
    )

    if mode == "ft":
        await find_breaking_counterfactuals(**common)
    elif mode in ("ff", "tf"):
        await find_corrective_counterfactuals(**common)
    else:
        raise SystemExit(f"Unknown mode: {mode!r}. Expected one of ft/ff/tf.")

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
    mode: str = "ff",
):
    cg: nx.DiGraph
    cost, cg, ops = heap_element

    if "delete_node" in current_ops:
        for node in list(cg.nodes):
            if node in original_nodes:
                perturbed_cg = delete_node(cg, node)
                
                if unit_cost == False:
                    perturbation_cost = delete_node_cost(cg, node) 
                elif unit_cost == True:
                    perturbation_cost = delete_node_uc(cg, node)

                new_ops = ops + [("delete_node", node)]

                similarity = node_similarity_index.get(node, 0.0)

                if mode == "ft":
                    heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))
                else:
                    heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), similarity, next(counter), (perturbed_cg, new_ops)))

    if "delete_edge" in current_ops:
        for edge in list(cg.edges):
            if edge in original_edges:

                perturbed_cg = delete_edge(cg, edge)
                
                if unit_cost == False:
                    perturbation_cost = delete_edge_cost(cg, edge)
                elif unit_cost == True:
                    perturbation_cost = delete_edge_uc(cg, edge)

                new_ops = ops + [("delete_edge", edge)]

                similarity = edge_similarity_index.get(edge, 0.0)


                if mode == "ft":
                    heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))
                else:
                    heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), similarity, next(counter), (perturbed_cg, new_ops)))

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

                        perturbation_cost = (1 + (1 - node_similarity_index.get(neighbor, 0.0))) + (1 + (1 - edge_similarity_index.get((node, neighbor), 0.0)))

                        heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))

                    if (neighbor, node) in edge_lookup and (neighbor, node) not in existing_edges:
                        perturbed_cg = add_edge(perturbed_cg, (neighbor, node), **G.edges[neighbor, node])
                        new_ops = new_ops + [("add_edge", (neighbor, node))]

                        perturbation_cost = (1 + (1 - node_similarity_index.get(neighbor, 0.0))) + (1 + (1 - edge_similarity_index.get((neighbor, node), 0.0)))

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

                    perturbation_cost = (1 + (1 - node_similarity_index.get(src, 0.0))) + (1 + (1 - node_similarity_index.get(tgt, 0.0))) + (1 + (1 - edge_similarity_index.get((src, tgt), 0.0)))

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

def save_operations_to_json(
    ops: list, 
    question: str, 
    ground_truth: str,
    original_answer: str, 
    perturbed_answer: str, 
    answer_similarity: float, 
    original_subgraph, 
    perturbed_subgraph, 
    noisy_subgraph,
    output_dir: str = "src/counterfactuals/robustness",
    filename: str = None, 
    found: bool = True, 
    cost: float = 0.0, 
    llm_calls: int = 0, 
    mode: str = "ff",
    noise_metadata: dict = None,
    noise_p=0.1
):

    noise = noise_p*100
    output_dir = f"{output_dir}/{mode}/noise_level_{int(noise)}"

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
        "noise": {
            "ops": noise_metadata.get("ops", []),
            "num_ops": noise_metadata.get("num_ops", 0),
            "score_after_noise": noise_metadata.get("score_after_noise", None),
            "noise_robust": noise_metadata.get("noise_robust", None),
            "noise_nodes_in_counterfactual": noise_metadata.get("noise_nodes_in_counterfactual", []),
            "noise_in_explanation": noise_metadata.get("noise_in_explanation", None),
        } if noise_metadata else {},
        "original_subgraph": subgraph_to_dict(original_subgraph),
        "perturbed_subgraph": subgraph_to_dict(perturbed_subgraph),
        "noisy_subgraph": subgraph_to_dict(noisy_subgraph),
        "timestamp": datetime.now().isoformat(),
        "llm_calls": llm_calls,
        "mode": mode
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Operations saved to: {filepath}")
    return filepath


_ALL_OPS = ("delete_node", "delete_edge", "add_node", "add_edge",
            "replace_node", "replace_edge")


def _parse_ops(spec: str) -> list:
    ops = [o.strip() for o in spec.split(",") if o.strip()]
    bad = [o for o in ops if o not in _ALL_OPS]
    if bad:
        raise SystemExit(f"Unknown ops: {bad}. Allowed: {_ALL_OPS}")
    return ops


def _parse_noise_percentages(spec: str) -> list:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError:
            raise SystemExit(f"--noise-percentages: '{tok}' is not a float.")
        if not (0.0 < v < 1.0):
            raise SystemExit(f"--noise-percentages: {v} must be in (0, 1).")
        out.append(v)
    if not out:
        raise SystemExit("--noise-percentages: at least one value required.")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="noise_resistance",
        description="Noise-resistance CFE search: inject random noise into the CG, "
                    "then run a bounded Dijkstra counterfactual search if the "
                    "noisy graph didn't already break the original answer.",
    )
    p.add_argument("--dataset", choices=DATASETS, default="synthetic",
                   help="Dataset name; selects working_dir and embedding indices.")
    p.add_argument("--rag-mode", choices=["hybrid", "local", "global", "naive"], default="hybrid",
                   help="LightRAG retrieval mode used by retrieve_subgraph.")
    p.add_argument("--top-k", type=int, default=2, help="LightRAG retrieval top_k.")
    p.add_argument("--input", default=None,
                   help="Directory of prior CFE JSONs to re-evaluate under noise. "
                        "Default: src/counterfactuals/results/<dataset>/all_ops_<mode>.")
    p.add_argument("--mode", choices=["ff", "ft", "tf"], default="ff",
                   help="CFE flip direction: ff (corrective F→F), ft (breaking T→F), tf (corrective T→F).")
    p.add_argument("--ops", default="delete_node,delete_edge,add_node,add_edge",
                   help="Comma-separated operations to enable. Subset of: " + ",".join(_ALL_OPS))
    p.add_argument("--max-cost", type=int, default=20, help="Cost budget c_max.")
    p.add_argument("--max-llm-calls", type=int, default=200, help="LLM-call budget.")
    p.add_argument("--unit-cost", action="store_true", help="Use unit-cost variant of edit costs.")
    p.add_argument("--noise-percentages", default="0.1,0.3,0.5,0.8",
                   help="Comma-separated noise fractions in (0, 1). One run per fraction × input file.")
    p.add_argument("--output-dir", default=None,
                   help="Base directory for saved JSON results. "
                        "Default: src/counterfactuals/robustness/<dataset>/noise_resistance.")
    p.add_argument("--seed_num", type=int)
    
    return p


async def main(args: argparse.Namespace):
    # if args.dataset != dataset:
    
    setup_dataset(args.dataset)

    current_ops = _parse_ops(args.ops)
    noise_percentages = _parse_noise_percentages(args.noise_percentages)

    input_dir = args.input or f"src/counterfactuals/results/{dataset}/all_ops_{args.mode}"
    if not os.path.isdir(input_dir):
        raise SystemExit(f"--input: directory not found: {input_dir}")

    output_dir = args.output_dir or f"src/counterfactuals/robustness/{dataset}/noise_resistance"

    rag = await initialize_lightrag(working_dir=WORKING_DIRS[dataset])

    json_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".json"))
    if not json_files:
        raise SystemExit(f"--input: no .json files under {input_dir}")

    for noise_p in noise_percentages:
        for i, json_file in enumerate(json_files):
            filepath = os.path.join(input_dir, json_file)
            print(f"\n=== Loading: {json_file} ===")

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            ground_truth = data["answers"]["ground_truth"]
            question = data["question"]

            print(f"\n=== {question} ===")

            context = await retrieve_subgraph(
                rag, query=question, mode=args.rag_mode, top_k=args.top_k,
            )

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
                noise_pct=noise_p,
                seed=i+args.seed_num,
                output_dir=output_dir,
            )


if __name__ == "__main__":
    asyncio.run(main(build_arg_parser().parse_args()))