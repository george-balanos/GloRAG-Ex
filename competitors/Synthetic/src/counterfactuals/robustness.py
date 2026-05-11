"""Noise-resistance evaluation harness and noise-injection helpers.

Wraps `find_counterfactuals` with a step that adds k random noise nodes/edges
to the context graph drawn from the KG, so we can measure whether explanations
remain stable. `inject_noise` adds a single semantically-distant
node+edge pair, used by validate_optimality to confirm the search ignores
irrelevant additions.
"""

from datetime import datetime
from src.query import *
from src.retrieve import *
from src.parser import *
from src.llm_judge import judge_response
from src.counterfactuals.edit_costs import *
from src.counterfactuals.perturbations import *
from src.counterfactuals.feasibility_constraints import *
from src.counterfactuals.utils import compute_answer_similarity, cosine_similarity_norm
from src.embeddings.utils import load_index
from collections import defaultdict
from src.embeddings.query import find_most_similar_node, find_most_distant_node, find_most_similar_edge, find_most_distant_edge, DIM, build_lookup, get_embedding, build_edge_lookup

import heapq
import networkx as nx
import asyncio
import itertools
import os
import random
import numpy as np


def graph_to_context_shuffled(cg: nx.Graph, shuffle_entities: bool = False, shuffle_relations: bool = False):
    """Serialize a context graph for the RAG prompt, optionally shuffling order.

    With both flags False, identical to `graph_to_context` (imported from
    src.parser). Shuffling tests whether the LLM's answer depends on the
    surface order of entities/relations in the prompt.
    """
    if not shuffle_entities and not shuffle_relations:
        return graph_to_context(cg)

    cg = cg.copy()
    if shuffle_entities:
        nodes = list(cg.nodes(data=True))
        random.shuffle(nodes)
        H = type(cg)()
        for n, attrs in nodes:
            H.add_node(n, **attrs)
        for u, v, attrs in cg.edges(data=True):
            H.add_edge(u, v, **attrs)
        cg = H
    if shuffle_relations:
        edges = list(cg.edges(data=True))
        random.shuffle(edges)
        H = type(cg)()
        for n, attrs in cg.nodes(data=True):
            H.add_node(n, **attrs)
        for u, v, attrs in edges:
            H.add_edge(u, v, **attrs)
        cg = H
    return graph_to_context(cg)


### Setup ###

def create_type_index(G: nx.Graph):
    type_index = defaultdict(list)
    for node, data in G.nodes(data=True):
        node_type = data.get("entity_type")
        type_index[node_type].append(node)

    return type_index

counter = itertools.count()

G = nx.read_graphml("synthetic/graph_chunk_entity_relation.graphml")

type_index = create_type_index(G)

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

async def create_edge_replacement_index(edges, context_graph, flip_direction="tf"):
    global edge_embeddings
    global edge_lookup

    edge_replacement_index = {}
    for edge in edges:
        data = G.edges[edge] if edge in G.edges else context_graph.edges[edge]
        
        if edge not in edge_lookup:
            description = data.get("description", "")
            if not description:
                print(f"Skipping {edge}: not in index and no description to embed.")
                continue

            print(f"Computing ad hoc embedding for noise edge {edge}...")
            embedding = (await sentence_transformer_embed([description]))[0]

            # Update lookup and embeddings in place
            idx = len(edge_lookup)
            edge_lookup[edge] = idx
            edge_embeddings = np.vstack([edge_embeddings, embedding.reshape(1, -1)])

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
        node_embedding = get_embedding(node_embeddings, node_lookup, node)
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
### Add Noise

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

    eligible_anchors = [node for node in cg.nodes() if cg.degree(node) >= 2]

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

        ops_applied.append(("add_node", new_node, anchor, random_edge_attr))

    print(f"Added {len(ops_applied)} noise node(s) attached to degree>=2 anchors with random edge attributes")
    return cg, ops_applied


def inject_noise(cg: nx.Graph, G: nx.Graph, query_embedding: np.ndarray):
    """Add one semantically-distant noise node + edge to cg.

    Picks v_noise in V_G \\ V_C with the lowest cos-sim to query_embedding,
    attaches it to the cg node with the lowest cos-sim to v_noise's embedding
    via a real edge in G if one exists, else by constructing one in cg only.
    Returns (noisy_cg, v_noise, edge).
    """
    cg = cg.copy()
    candidate_nodes = [n for n in G.nodes() if n not in cg.nodes()]
    if not candidate_nodes:
        return cg, None, None

    def _emb(node):
        try:
            return get_embedding(node_embeddings, node_lookup, node)
        except Exception:
            return None

    # most distant from query
    best_node, best_sim = None, float("inf")
    for n in candidate_nodes:
        emb = _emb(n)
        if emb is None:
            continue
        sim = cosine_similarity_norm(query_embedding, emb)
        if sim < best_sim:
            best_sim = sim
            best_node = n

    if best_node is None:
        return cg, None, None

    # pick anchor in cg most distant from best_node
    best_emb = _emb(best_node)
    best_anchor, anchor_sim = None, float("inf")
    for u in cg.nodes():
        emb = _emb(u)
        if emb is None:
            continue
        sim = cosine_similarity_norm(best_emb, emb)
        if sim < anchor_sim:
            anchor_sim = sim
            best_anchor = u

    if best_anchor is None:
        best_anchor = next(iter(cg.nodes()))

    cg.add_node(best_node, **G.nodes[best_node])
    edge = (best_node, best_anchor)
    if edge in G.edges:
        cg.add_edge(*edge, **G.edges[edge])
    elif (best_anchor, best_node) in G.edges:
        edge = (best_anchor, best_node)
        cg.add_edge(*edge, **G.edges[edge])
    else:
        cg.add_edge(*edge, description="noise", keywords="noise")

    return cg, best_node, edge

################################################

async def find_counterfactuals(rag, question: str, context, example, max_cost=3, max_llm_calls=100, max_sparsity=None, unit_cost: bool=False, seed=None, noise_pct=0.1):
    found = example["found"]
    if not found:
        return

    query_embedding = (await sentence_transformer_embed([question]))[0]
    original_answer = example["answers"]["original"]

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
    noise_robust = noise_score != 0  # True = noise didn't break the answer

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
            perturbed_answer=noisy_response,
            answer_similarity=0.0,
            original_subgraph=parsed_subgraph,
            perturbed_subgraph=graph_to_subgraph(noisy_cg),
            found=False,
            cost=0.0,
            llm_calls=1,
            noise_metadata=noise_metadata,
            noise_p=noise_pct
        )
        return

    # Noise didn't affect the answer — proceed with noisy graph as new baseline
    context_graph = noisy_cg
    #####################

    context_graph_nodes = set(context_graph.nodes)
    context_graph_edges = set(context_graph.edges())

    edge_labels = {(u, v): data.get("description", "") for u, v, data in context_graph.edges(data=True)}

    node_replacement_index = create_node_replacement_index(context_graph_nodes, context_graph, flip_direction="tf")
    edge_replacement_index = await create_edge_replacement_index(context_graph_edges, context_graph, flip_direction="tf")

    node_similarity_index = create_node_similarity_index(context_graph_nodes, query_embedding)
    edge_similarity_index = await create_edge_similarity_index(edge_labels, query_embedding)

    llm_calls = 0

    Q = []

    ### Prune seen context graph.
    state_cache = set()

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
            cg_context = graph_to_context(cg)
            new_response = await query(rag, cg_context, question)

            print(f"Cost: {cost} | New response: {new_response} | Original: {original_answer}")

            score = await judge_response(question, new_response, original_answer)
            llm_calls += 1

            if score == 0:
                print(f"Counterfactual Operations: {ops}")

                answer_similarity = await compute_answer_similarity(original_answer, new_response)
                print(f"Answer similarity (original vs perturbed): {answer_similarity:.4f}")

                noise_nodes_added = {op[1] for op in noise_ops if op[0] == "add_node"}
                cf_nodes_deleted = {op[1] for op in ops if op[0] == "delete_node"}
                noise_overlap = noise_nodes_added & cf_nodes_deleted
                noise_in_explanation = len(noise_overlap) > 0

                noise_metadata["noise_nodes_in_counterfactual"] = list(noise_overlap)
                noise_metadata["noise_in_explanation"] = noise_in_explanation

                print(f"Noise contributed to explanation: {noise_in_explanation} (overlap: {noise_overlap})")

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
                    noise_metadata=noise_metadata,
                    noise_p=noise_pct
                )
                return ops
            
        expand(Q, (cost, cg, ops), node_replacement_index=node_replacement_index, edge_replacement_index=edge_replacement_index, node_similarity_index=node_similarity_index, edge_similarity_index=edge_similarity_index, unit_cost=unit_cost)

        print()

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
        noise_p=noise_pct
    )

def expand(Q, heap_element, node_replacement_index, edge_replacement_index, node_similarity_index, edge_similarity_index, unit_cost: bool = False):
    cost, cg, ops = heap_element

    undirected: nx.Graph = cg.to_undirected()
    cut_vertices = set(nx.articulation_points(cg.to_undirected()))
    cut_edges = set(nx.bridges(cg.to_undirected()))

    ### Delete Node
    # for node in list(cg.nodes):
    #     if node not in cut_vertices:
    #         perturbed_cg = delete_node(cg, node)
            
    #         if unit_cost == False:
    #             perturbation_cost = delete_node_cost(cg, node) 
    #         elif unit_cost == True:
    #             perturbation_cost = delete_node_uc(cg, node)

    #         new_ops = ops + [("delete_node", node)]

    #         similarity = node_similarity_index.get(node, 0.0)

    #         heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

    ### Updated Delete Node
    # Allow if not a cut vertex, OR if it is a cut vertex but all neighbors
    # would become isolated (meaning no real split, just singleton cleanup)
    for node in list(cg.nodes):
        if node in cut_vertices:
            neighbors = list(undirected.neighbors(node))
            would_split = any(undirected.degree(n) > 1 for n in neighbors)
            if would_split:
                continue

        perturbed_cg = delete_node(cg, node)
        
        if unit_cost == False:
            perturbation_cost = delete_node_cost(cg, node) 
        elif unit_cost == True:
            perturbation_cost = delete_node_uc(cg, node)

        new_ops = ops + [("delete_node", node)]

        similarity = node_similarity_index.get(node, 0.0)

        heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))


    # ### Delete Edge
    # for edge in list(cg.edges):
    #     if edge not in cut_edges:
    #         perturbed_cg = delete_edge(cg, edge)
            
    #         if unit_cost == False:
    #             perturbation_cost = delete_edge_cost(cg, edge)
    #         elif unit_cost == True:
    #             perturbation_cost = delete_edge_uc(cg, edge)

    #         new_ops = ops + [("delete_edge", edge)]

    #         similarity = edge_similarity_index.get(edge, 0.0)

    #         heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

    ### Updated Delete Edge
    # Allow if not a cut edge, OR if it is a cut edge but both endpoints
    # would become isolated (meaning no real split, just singleton cleanup)
    for edge in list(cg.edges):
        if edge in cut_edges:
            u, v = edge[0], edge[1]
            would_split = undirected.degree(u) > 1 and undirected.degree(v) > 1
            if would_split:
                continue

        perturbed_cg = delete_edge(cg, edge)
        
        if unit_cost == False:
            perturbation_cost = delete_edge_cost(cg, edge)
        elif unit_cost == True:
            perturbation_cost = delete_edge_uc(cg, edge)

        new_ops = ops + [("delete_edge", edge)]

        similarity = edge_similarity_index.get(edge, 0.0)

        heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

    # ### Replace Node
    # for node, _ in list(cg.nodes(data=True)):
    #     node_replacement = node_replacement_index.get(node)
    #     if node_replacement is None:
    #         continue

    #     current_replacement = node_replacement.get("name")
    #     if current_replacement is None:
    #         continue

    #     if current_replacement in G.nodes:
    #         replacement_attr = G.nodes[current_replacement]
    #     elif current_replacement in cg.nodes:
    #         replacement_attr = cg.nodes[current_replacement]
    #     else:
    #         print(f"Skipping replacement: node {current_replacement} not found in G or cg.")
    #         continue
        
    #     sim = node_replacement.get("similarity")

    #     perturbed_cg = replace_node(cg, node, current_replacement, **replacement_attr)
        
    #     if unit_cost == False:
    #         perturbation_cost = 1 - sim
    #     elif unit_cost == True:
    #         perturbation_cost = replace_node_uc()

    #     new_ops = ops + [("replace_node", (node, current_replacement))]

    #     similarity = node_similarity_index.get(node, 0.0)

    #     heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

    # ### Replace Edge
    # for edge in list(cg.edges):
    #     edge_replacement = edge_replacement_index.get(edge)
    #     if edge_replacement is None:
    #         continue

    #     current_replacement = edge_replacement.get("edge")
    #     if current_replacement is None:
    #         continue

    #     if current_replacement in G.edges:
    #         replacement_attr = G.edges[current_replacement]
    #     elif current_replacement in cg.edges:
    #         replacement_attr = cg.edges[current_replacement]
    #     else:
    #         print(f"Skipping replacement: edge {current_replacement} not found in G or cg.")
    #         continue
        
    #     sim = edge_replacement.get("similarity")

    #     perturbed_cg = replace_edge(cg, edge, current_replacement, **replacement_attr)
        
    #     if unit_cost == False:
    #         perturbation_cost = 1 - sim
    #     elif unit_cost == True:
    #         perturbation_cost = replace_edge_uc()

    #     new_ops = ops + [("replace_edge", (edge, current_replacement))]

    #     similarity = edge_similarity_index.get(edge, 0.0)

    #     heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

def save_operations_to_json(ops: list,question: str, original_answer: str, perturbed_answer: str, answer_similarity: float, original_subgraph, perturbed_subgraph, output_dir: str = "src/counterfactuals/robustness/delete_only_results",filename: str = None, found: bool = True, cost: float = 0.0, llm_calls: int = 0, noise_metadata: dict = None, noise_p=0.1):

    noise = noise_p*100

    output_dir = f"{output_dir}_{noise}"
        
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
        "timestamp": datetime.now().isoformat(),
        "llm_calls": llm_calls
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Operations saved to: {filepath}")
    return filepath


async def main():
    rag = await initialize_lightrag()
    
    results_folder = "src/counterfactuals/counterfactual_results_sem_delete_c_10"

    json_files = [f for f in os.listdir(results_folder) if f.endswith(".json")]

    noise_percentages = [0.1, 0.2, 0.3, 0.5, 0.8]

    for noise_p in noise_percentages:
        for i, json_file in enumerate(json_files):
            filepath = os.path.join(results_folder, json_file)
            print(f"\n=== Loading: {json_file} ===")

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            question = data["question"]

            print(f"\n=== {question} ===")

            context = await retrieve_subgraph(rag, query=question, mode="hybrid", top_k=2)
            await find_counterfactuals(rag, question, context=context, max_cost=10, max_llm_calls=200, unit_cost=False, example=data, seed=i, noise_pct=noise_p)

if __name__ == "__main__":
    asyncio.run(main())