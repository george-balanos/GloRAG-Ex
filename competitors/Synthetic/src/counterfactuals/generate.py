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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

counter = itertools.count()

G = nx.read_graphml("synthetic/graph_chunk_entity_relation.graphml")

type_index = defaultdict(list)
for node, data in G.nodes(data=True):
    node_type = data.get("entity_type")
    type_index[node_type].append(node)

node_index_prefix = "src/embeddings/node_index"
node_index, node_records, node_embeddings = load_index(node_index_prefix, DIM, 2000)
node_lookup = build_lookup(node_records)

edge_index_prefix = "src/embeddings/edge_index"
edge_index, edge_records, edge_embeddings = load_index(edge_index_prefix, DIM, 2000)
# edge_lookup = build_lookup(edge_records)
edge_lookup = build_edge_lookup(edge_records)

print(edge_lookup)

async def find_counterfactuals(rag, question: str, context, operation="delete_node", max_cost=3):
    query_embedding = (await sentence_transformer_embed([question]))[0]

    parsed_subgraph = parse_context(context)
    context_graph = parse_graph(parsed_subgraph)

    print_subgraph(parsed_subgraph)

    original_answer = await query(rag, context, question)

    context_graph_nodes = set(context_graph.nodes)

    context_graph_edges = set(context_graph.edges())
    edge_labels = {(u, v): data.get("description", "") 
                for u, v, data in context_graph.edges(data=True)}

    print(context_graph_edges)    

    replacement_index = {}
    if operation == "replace_node":
        for node in context_graph_nodes:
            # Get attributes from G, fallback to context_graph
            data = G.nodes[node] if node in G.nodes else context_graph.nodes[node]
            node_type = data.get("entity_type")

            if not node_type:
                print(f"Skipping {node}: no entity_type found")
                continue

            # Use for F -> T
            # most_similar = find_most_similar_node(node, node_type, node_embeddings, node_lookup, type_index)
            # if most_similar is None:
            #     print(f"Skipping {node}: no similar node found")
            #     continue

            # replacement_index[node] = most_similar

            # Use for T -> F
            most_distant = find_most_distant_node(node, node_type, node_embeddings, node_lookup, type_index)
            if most_distant is None:
                print(f"Skipping {node}: no similar node found")
                continue

            replacement_index[node] = most_distant

    elif operation == "replace_edge":
        for edge in context_graph_edges:
            data = G.edges[edge] if edge in G.edges else context_graph.edges[edge]
            most_distant = find_most_distant_edge(edge, edge_embeddings, edge_lookup)

            if most_distant is None:
                print(f"Skipping {edge}: no similar edge found")
                continue

            replacement_index[edge] = most_distant

    node_similarity_index = {}
    for node in context_graph_nodes:
        node_embedding = get_embedding(node_embeddings, node_lookup, node)
        if node_embedding is not None:
            similarity = cosine_similarity_norm(query_embedding, node_embedding)
        else:
            similarity = 0.0
        node_similarity_index[node] = similarity

    edge_similarity_index = {}
    for (u, v), label in edge_labels.items():
        if not label:
            edge_similarity_index[(u, v)] = 0.0
            continue
        edge_embedding = (await sentence_transformer_embed([label]))[0]
        similarity = cosine_similarity_norm(query_embedding, edge_embedding)
        edge_similarity_index[(u, v)] = similarity

    # print(edge_similarity_index)


    # Min-Heap
    Q = []

    cut_vertices = set(nx.articulation_points(context_graph.to_undirected()))
    bridges = set(nx.bridges(context_graph.to_undirected()))

    heapq.heappush(Q, (0, 0.0, next(counter), (context_graph, [], cut_vertices, bridges)))

    seen = set()

    llm_calls = 0

    while Q:
        c, _, _, (cg, ops, cv, bg) = heapq.heappop(Q)

        if c > max_cost:
            print(f"Max cost {max_cost} exceeded (current cost: {c:.4f}). Stopping search.")
            break

        # print(f"Processing: CG {cg} | Ops {ops}")

        if operation == "delete_node" or operation == "replace_node" or operation == "add_node":
            state = frozenset(cg.nodes)
            if state in seen:
                continue
            seen.add(state)

        elif operation == "delete_edge":
            state = frozenset(cg.edges)
            if state in seen:
                continue
            seen.add(state)

        elif operation == "replace_edge":
            state = frozenset(
                (u, v, cg.edges[u, v].get("description", ""))
                for u, v in cg.edges
            )
            if state in seen:
                continue
            seen.add(state)

        elif operation == "add_edge":
            pass #TODO

        if len(ops) > 0:
            cg_context = graph_to_context(cg)
            new_response = await query(rag, cg_context, question)

            # print(f"Perturbed CG: {cg_context}")

            print(f"Cost: {c} | New response: {new_response} | Original: {original_answer}")

            score = await judge_response(question, new_response, original_answer)

            llm_calls += 1

            if score == 0:
                print(f"Counterfactual Operations: {ops}")

                answer_similarity = await compute_answer_similarity(original_answer, new_response)
                print(f"Answer similarity (original vs perturbed): {answer_similarity:.4f}")

                save_operations_to_json(
                    ops=ops,
                    question=question,
                    operation=operation,
                    original_answer=original_answer,
                    perturbed_answer=new_response,
                    answer_similarity=answer_similarity,
                    original_subgraph=parsed_subgraph,
                    perturbed_subgraph=graph_to_subgraph(cg),
                    found=True,
                    cost=c,
                    llm_calls=llm_calls
                )
                return ops

        if operation == "delete_node" or operation == "replace_node" or operation == "add_node":
            expand(Q, (c, cg, ops, cv, bg), operation=operation, replacement_index=replacement_index, similarity_index=node_similarity_index)
        else:
            expand(Q, (c, cg, ops, cv, bg), operation=operation, replacement_index=replacement_index, similarity_index=edge_similarity_index)

        print()

    print(f"Could not find feasible counterfactual explanations.")

    save_operations_to_json(
        ops=[],
        question=question,
        operation=operation,
        original_answer=original_answer,
        perturbed_answer=None,
        answer_similarity=0.0,
        original_subgraph=parsed_subgraph,
        perturbed_subgraph=None,
        found=False,
        llm_calls=llm_calls
    )

def expand(Q, heap_element, operation, replacement_index=None, similarity_index=None):
    c, cg, ops, cv, bg = heap_element

    if operation == "delete_node":
        for node in list(cg.nodes):
            if node not in cv: # Feasibility Constraint (ACTIVE)
            # if node:
                perturbed_cg = delete_node(cg, node)
                perturbation_cost = delete_node_cost(cg, node)
                new_ops = ops + [node]

                similarity = similarity_index.get(node, 0.0)

                cut_vertices = set(nx.articulation_points(perturbed_cg.to_undirected()))
                cut_vertices.update(cv)
                bridges = set(nx.bridges(perturbed_cg.to_undirected()))
                bridges.update(bg)
                # heapq.heappush(Q, (c + perturbation_cost, next(counter), (perturbed_cg, new_ops, cut_vertices, bridges)))
                
                heapq.heappush(Q, (c + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops, cut_vertices, bridges)))
            else:
                # print(f"Not feasible perturbation: {node}")
                pass

    elif operation == "delete_edge":
        for edge in list(cg.edges):
            if edge not in bg: # Feasibility Constraint (ACTIVE)
            # if edge:
                perturbed_cg = delete_edge(cg, edge)
                perturbation_cost = delete_edge_cost()
                new_ops = ops + [edge]

                similarity = similarity_index.get(edge, 0.0)
                # print(f"Similarity: {similarity}")

                cut_vertices = set(nx.articulation_points(perturbed_cg.to_undirected()))
                cut_vertices.update(cv)
                bridges = set(nx.bridges(perturbed_cg.to_undirected()))
                bridges.update(bg)
                # heapq.heappush(Q, (c + perturbation_cost, next(counter), (perturbed_cg, new_ops, cut_vertices, bridges)))

                heapq.heappush(Q, (c + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops, cut_vertices, bridges)))
            else:
                # print(f"Not feasible perturbation: {edge}")
                pass

    elif operation == "replace_node":
        for node, data in list(cg.nodes(data=True)):
            most_similar_node = replacement_index.get(node)
            if most_similar_node is None:
                continue

            replacement_node = most_similar_node.get("name")
            if replacement_node is None:
                continue

            replacement_attrs = G.nodes[replacement_node]
            sim = most_similar_node.get("similarity")

            perturbed_cg = replace_node(cg, node, replacement_node, **replacement_attrs)
            perturbation_cost = 1 - sim
            new_ops = ops + [(node, replacement_node)]

            similarity = similarity_index.get(node, 0.0)

            cut_vertices = set(nx.articulation_points(perturbed_cg.to_undirected()))
            cut_vertices.update(cv)
            bridges = set(nx.bridges(perturbed_cg.to_undirected()))
            bridges.update(bg)
            # heapq.heappush(Q, (c + perturbation_cost, next(counter), (perturbed_cg, new_ops, cv, bg)))

            heapq.heappush(Q, (c + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops, cut_vertices, bridges)))

    elif operation == "replace_edge":
        for edge in list(cg.edges):
            most_similar_edge = replacement_index.get(edge)
            if most_similar_edge is None:
                continue

            replacement_edge = most_similar_edge.get("edge")
            if replacement_edge is None:
                continue

            replacement_attrs = G.edges[replacement_edge]

            sim = most_similar_edge.get("similarity")

            perturbed_cg = replace_edge(cg, edge, replacement_edge, **replacement_attrs)
            perturbation_cost = 1 - sim
            new_ops = ops + [(edge, replacement_edge)]

            similarity = similarity_index.get(edge, 0.0)

            cut_vertices = set(nx.articulation_points(perturbed_cg.to_undirected()))
            cut_vertices.update(cv)
            bridges = set(nx.bridges(perturbed_cg.to_undirected()))
            bridges.update(bg)

            heapq.heappush(Q, (c + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops, cut_vertices, bridges)))



def save_operations_to_json(ops: list, question: str, operation: str, original_answer: str, perturbed_answer: str, answer_similarity: float, original_subgraph, perturbed_subgraph, output_dir: str = "src/counterfactuals/counterfactual_results", filename: str = None, found: bool = True, cost: float = 0.0, llm_calls: int = 0):
    os.makedirs(f"{output_dir}/{operation}", exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"counterfactual_{operation}_{timestamp}.json"

    # filepath = os.path.join(output_dir, filename)
    filepath = os.path.join(f"{output_dir}/{operation}", filename)

    serialisable_ops = []
    for op in ops:
        if isinstance(op, tuple):
            serialisable_ops.append(list(op))
        else:
            serialisable_ops.append(op)

    payload = {
        "question": question,
        "found": found,
        "operation_type": operation,
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

    # start_from = 51


    # operations = ["delete_node", "delete_edge", "replace_node"]
    operations = ["delete_node", "delete_edge", "replace_node", "replace_edge"]

    for o in operations:
        results = data["results"]
        for idx, r in results.items():
            # if int(idx) < start_from:
            #     continue

            question = r["question"]
            case = r["case"]

            if case != "ft":
                continue

            print(f"\n=== [{idx}] {question} ===")

            context = await retrieve_subgraph(rag, query=question, mode="hybrid", top_k=2)
            await find_counterfactuals(rag, question, context=context, operation=o, max_cost=10)

    # QUERY = "How do you activate a Nebulon communicator?"
    # context = await retrieve_subgraph(rag, query=QUERY, mode="hybrid", top_k=2)

    # await find_counterfactuals(rag, QUERY, context=context, operation="delete_node", max_cost=10)


if __name__ == "__main__":
    asyncio.run(main())