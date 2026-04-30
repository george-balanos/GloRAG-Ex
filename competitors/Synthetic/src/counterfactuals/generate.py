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
# edge_lookup = build_lookup(edge_records)
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

async def find_counterfactuals(rag, question: str, context, max_cost=3, max_llm_calls=100, max_sparsity=None):
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
                    llm_calls=llm_calls
                )
                return ops
            
        expand(Q, (cost, cg, ops), node_replacement_index=node_replacement_index, edge_replacement_index=edge_replacement_index, node_similarity_index=node_similarity_index, edge_similarity_index=edge_similarity_index)

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
        llm_calls=llm_calls
    )



def expand(Q, heap_element, node_replacement_index, edge_replacement_index, node_similarity_index, edge_similarity_index):
    cost, cg, ops = heap_element

    cut_vertices = set(nx.articulation_points(cg.to_undirected()))
    cut_edges = set(nx.bridges(cg.to_undirected()))

    ### Delete Node
    for node in list(cg.nodes):
        if node not in cut_vertices:
            perturbed_cg = delete_node(cg, node)
            perturbation_cost = delete_node_cost(cg, node)
            new_ops = ops + [("delete_node", node)]

            similarity = node_similarity_index.get(node, 0.0)

            heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

    ### Delete Edge
    for edge in list(cg.edges):
        if edge not in cut_edges:
            perturbed_cg = delete_edge(cg, edge)
            perturbation_cost = delete_edge_cost()
            new_ops = ops + [("delete_edge", edge)]

            similarity = edge_similarity_index.get(edge, 0.0)

            heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

    ### Replace Node
    for node, _ in list(cg.nodes(data=True)):
        node_replacement = node_replacement_index.get(node)
        if node_replacement is None:
            continue

        current_replacement = node_replacement.get("name")
        if current_replacement is None:
            continue

        replacement_attr = G.nodes[current_replacement]
        sim = node_replacement.get("similarity")

        perturbed_cg = replace_node(cg, node, current_replacement, **replacement_attr)
        perturbation_cost = 1 - sim
        new_ops = ops + [("replace_node", (node, current_replacement))]

        similarity = node_similarity_index.get(node, 0.0)

        heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

    ### Replace Edge
    for edge in list(cg.edges):
        edge_replacement = edge_replacement_index.get(edge)
        if edge_replacement is None:
            continue

        current_replacement = edge_replacement.get("edge")
        if current_replacement is None:
            continue

        replacement_attr = G.edges[current_replacement]
        sim = edge_replacement.get("similarity")

        perturbed_cg = replace_edge(cg, edge, current_replacement, **replacement_attr)
        perturbation_cost = 1 - sim
        new_ops = ops + [("replace_edge", (edge, current_replacement))]

        similarity = edge_similarity_index.get(edge, 0.0)

        heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))



def save_operations_to_json(ops: list, question: str, original_answer: str, perturbed_answer: str, answer_similarity: float, original_subgraph, perturbed_subgraph, output_dir: str = "src/counterfactuals/counterfactual_results", filename: str = None, found: bool = True, cost: float = 0.0, llm_calls: int = 0):
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

async def main():
    rag = await initialize_lightrag()
    
    with open(f"benchmark/results/comparison.json", "r", encoding="utf-8") as results:
        data = json.load(results)

    results = data["results"]
    for idx, r in results.items():
        question = r["question"]
        case = r["case"]

        if case != "ft":
            continue

        print(f"\n=== [{idx}] {question} ===")

        context = await retrieve_subgraph(rag, query=question, mode="hybrid", top_k=2)
        await find_counterfactuals(rag, question, context=context, max_cost=10, max_llm_calls=200)

if __name__ == "__main__":
    asyncio.run(main())