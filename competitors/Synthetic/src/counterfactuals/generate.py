from src.query import *
from src.retrieve import *
from src.parser import *
from src.llm_judge import judge_response
from src.counterfactuals.edit_costs import *
from src.counterfactuals.perturbations import *
from src.counterfactuals.feasibility_constraints import *
from src.counterfactuals.utils import cosine_similarity_norm
from src.embeddings.utils import load_index

import heapq
import networkx as nx
import asyncio
import itertools

counter = itertools.count()

async def find_counterfactuals(rag, question: str, context):
    parsed_subgraph = parse_context(context)
    context_graph = parse_graph(parsed_subgraph)

    original_answer = await query(rag, context, question)

    # Ie: Embedding index
    # node_embedding_index = load_index("src/embeddings/node_index.bin", dim=384, max_elements=200)

    context_graph_nodes = list(context_graph.nodes)

    # Min-Heap
    Q = []

    print(f"Min-heap: {Q}")
    print(f"Context Graph nodes: {context_graph_nodes}")

    heapq.heappush(Q, (0, next(counter), (context_graph, [])))

    print(Q)

    seen = set()

    while Q:
        c, _, (cg, ops) = heapq.heappop(Q)

        print(f"Processing: CG {cg} | Ops {ops}")

        state = frozenset(cg.nodes)
        if state in seen:
            continue
        seen.add(state)

        if len(ops) > 0:
            cg_context = graph_to_context(cg, parsed_subgraph)
            new_response = await query(rag, cg_context, question)

            print(f"Cost: {c} | New response: {new_response} | Original: {original_answer}")

            score = await judge_response(question, new_response, original_answer)

            # print(f"Score: {score}")

            if score == 0:
                print(f"Counterfactual Operations: {ops}")
                break

        expand(Q, (c, cg, ops), operation="delete_node")
        print()

def expand(Q, heap_element, operation):
    c = heap_element[0]
    cg: nx.Graph = heap_element[1]
    ops = heap_element[2]
    
    cut_vertices = set(nx.articulation_points(cg.to_undirected()))

    if operation == "delete_node":
        for node in list(cg.nodes):
            if node not in cut_vertices:
                perturbed_cg = delete_node(cg, node)
                perturbation_cost = delete_node_cost(cg, node)
                new_ops = ops + [node]
                heapq.heappush(Q, (c + perturbation_cost, next(counter), (perturbed_cg, new_ops)))
            else:
                # print(f"Not feasible perturbation: {node}")
                pass

async def main():
    QUERY = "What are the two primary materials used to construct a Xylotian 'Sky-Skiff' hull?"

    rag = await initialize_lightrag()

    context = await retrieve_subgraph(rag, query=QUERY)

    await find_counterfactuals(rag, QUERY, context=context)


if __name__ == "__main__":
    asyncio.run(main())