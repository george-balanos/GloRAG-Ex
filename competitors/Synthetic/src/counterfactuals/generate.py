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

    cut_vertices = set(nx.articulation_points(context_graph.to_undirected()))

    heapq.heappush(Q, (0, next(counter), (context_graph, [], cut_vertices)))

    seen = set()

    while Q:
        c, _, (cg, ops, cv) = heapq.heappop(Q)

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
                return ops

        expand(Q, (c, cg, ops, cv), operation="delete_node")
        print()

    print(f"Could not find feasible counterfactual explanations.")

def expand(Q, heap_element, operation):
    c, cg, ops, cv = heap_element
    
    # cut_vertices = set(nx.articulation_points(cg.to_undirected()))

    if operation == "delete_node":
        for node in list(cg.nodes):
            if node not in cv:
                perturbed_cg = delete_node(cg, node)
                perturbation_cost = delete_node_cost(cg, node)
                new_ops = ops + [node]
                cut_vertices = set(nx.articulation_points(perturbed_cg.to_undirected()))
                cut_vertices.update(cv)
                heapq.heappush(Q, (c + perturbation_cost, next(counter), (perturbed_cg, new_ops, cut_vertices)))
            else:
                print(f"Not feasible perturbation: {node}")
                pass

async def main():
    QUERY = "What two distinct abilities does a Xylotian 'Chrono-Weaver' possess?"

    rag = await initialize_lightrag()

    context = await retrieve_subgraph(rag, query=QUERY)

    await find_counterfactuals(rag, QUERY, context=context)


if __name__ == "__main__":
    asyncio.run(main())