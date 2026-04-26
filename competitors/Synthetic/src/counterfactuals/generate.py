from src.query import *
from src.retrieve import *
from src.parser import *
from src.llm_judge import judge_response
from src.counterfactuals.edit_costs import *
from src.counterfactuals.perturbations import *
from src.counterfactuals.feasibility_constraints import *
from src.counterfactuals.utils import cosine_similarity_norm
from src.embeddings.utils import load_index
from collections import defaultdict
from src.embeddings.query import find_most_similar_node, DIM, build_lookup

import heapq
import networkx as nx
import asyncio
import itertools

counter = itertools.count()

G = nx.read_graphml("synthetic/graph_chunk_entity_relation.graphml")

type_index = defaultdict(list)
for node, data in G.nodes(data=True):
    node_type = data.get("entity_type")
    type_index[node_type].append(node)

index_prefix = "src/embeddings/node_index"
index, records, embeddings = load_index(index_prefix, DIM, 2000)
lookup = build_lookup(records)

async def find_counterfactuals(rag, question: str, context, operation="delete_node"):
    parsed_subgraph = parse_context(context)
    context_graph = parse_graph(parsed_subgraph)

    print_subgraph(parsed_subgraph)

    original_answer = await query(rag, context, question)

    # Ie: Embedding index
    # node_embedding_index = load_index("src/embeddings/node_index.bin", dim=384, max_elements=200)

    context_graph_nodes = set(context_graph.nodes)
    replacement_index = {}
    if operation == "replace_node":
        for node in context_graph_nodes:
            # Get attributes from G, fallback to context_graph
            data = G.nodes[node] if node in G.nodes else context_graph.nodes[node]
            node_type = data.get("entity_type")

            if not node_type:
                print(f"Skipping {node}: no entity_type found")
                continue

            most_similar = find_most_similar_node(node, node_type, embeddings, lookup, type_index)
            if most_similar is None:
                print(f"Skipping {node}: no similar node found")
                continue

            replacement_index[node] = most_similar

    # Min-Heap
    Q = []

    cut_vertices = set(nx.articulation_points(context_graph.to_undirected()))
    bridges = set(nx.bridges(context_graph.to_undirected()))

    heapq.heappush(Q, (0, next(counter), (context_graph, [], cut_vertices, bridges)))

    seen = set()

    while Q:
        c, _, (cg, ops, cv, bg) = heapq.heappop(Q)

        print(f"Processing: CG {cg} | Ops {ops}")

        state = frozenset(cg.nodes)
        if state in seen:
            continue
        seen.add(state)

        if len(ops) > 0:
            if ops[0][0] == "Xylotian Sky-Skiff":
                s = graph_to_subgraph(cg, parsed_subgraph)
                print_subgraph(s)

                exit()

            cg_context = graph_to_context(cg, parsed_subgraph)
            new_response = await query(rag, cg_context, question)

            print(f"Cost: {c} | New response: {new_response} | Original: {original_answer}")

            score = await judge_response(question, new_response, original_answer)

            # print(f"Score: {score}")

            if score == 0:
                print(f"Counterfactual Operations: {ops}")
                return ops

        expand(Q, (c, cg, ops, cv, bg), operation=operation, replacement_index=replacement_index)
        print()

    print(f"Could not find feasible counterfactual explanations.")

def expand(Q, heap_element, operation, replacement_index=None):
    c, cg, ops, cv, bg = heap_element

    if operation == "delete_node":
        for node in list(cg.nodes):
            if node not in cv:
                perturbed_cg = delete_node(cg, node)
                perturbation_cost = delete_node_cost(cg, node)
                new_ops = ops + [node]
                cut_vertices = set(nx.articulation_points(perturbed_cg.to_undirected()))
                cut_vertices.update(cv)
                bridges = set(nx.bridges(perturbed_cg.to_undirected()))
                bridges.update(bg)
                heapq.heappush(Q, (c + perturbation_cost, next(counter), (perturbed_cg, new_ops, cut_vertices, bridges)))
            else:
                print(f"Not feasible perturbation: {node}")
                pass

    elif operation == "delete_edge":
        for edge in list(cg.edges):
            if edge not in bg:
                perturbed_cg = delete_edge(cg, edge)
                perturbation_cost = delete_edge_cost()
                new_ops = ops + [edge]
                cut_vertices = set(nx.articulation_points(perturbed_cg.to_undirected()))
                cut_vertices.update(cv)
                bridges = set(nx.bridges(perturbed_cg.to_undirected()))
                bridges.update(bg)
                heapq.heappush(Q, (c + perturbation_cost, next(counter), (perturbed_cg, new_ops, cut_vertices, bridges)))
            else:
                print(f"Not feasible perturbation: {edge}")
                pass

    elif operation == "replace_node":
        for node, data in list(cg.nodes(data=True)):
            most_similar_node = replacement_index.get(node)
            if most_similar_node is None:
                continue

            replacement_node = most_similar_node.get("name")
            replacement_attrs = G.nodes[replacement_node]
            sim = most_similar_node.get("similarity")

            perturbed_cg = replace_node(cg, node, replacement_node, **replacement_attrs)
            perturbation_cost = 1 - sim
            new_ops = ops + [(node, replacement_node)]
            heapq.heappush(Q, (c + perturbation_cost, next(counter), (perturbed_cg, new_ops, cv, bg)))


async def main():
    QUERY = "What are the two primary materials used to construct a Xylotian 'Sky-Skiff' hull?"

    rag = await initialize_lightrag()

    context = await retrieve_subgraph(rag, query=QUERY, mode="local", top_k=10)

    await find_counterfactuals(rag, QUERY, context=context, operation="replace_node")


if __name__ == "__main__":
    asyncio.run(main())