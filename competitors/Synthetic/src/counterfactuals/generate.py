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

### Explanation Stability/Consistency
from src.counterfactuals.robustness import graph_to_context_shuffled

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

G = nx.read_graphml("KGs/synthetic/graph_chunk_entity_relation.graphml")

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

async def find_counterfactuals(
    rag, 
    question: str, 
    context, 
    max_cost=3, 
    max_llm_calls=100, 
    max_sparsity=None, 
    unit_cost: bool=False, 
    current_ops: list=["delete_node", "delete_edge", "replace_node", "replace_edge"], 
    ground_truth: str = ""
):
    query_embedding = (await sentence_transformer_embed([question]))[0]
    original_answer = await query(rag, context, question)

    ### Lightrag specific
    parsed_subgraph = parse_context(context)
    context_graph = parse_graph(parsed_subgraph)
    #####################

    context_graph_nodes = set(context_graph.nodes)
    context_graph_edges = set(context_graph.edges())

    # edge_labels = {(u, v): data.get("description", "") for u, v, data in context_graph.edges(data=True)}

    # node_replacement_index = create_node_replacement_index(context_graph_nodes, context_graph, flip_direction="tf")
    # edge_replacement_index = create_edge_replacement_index(context_graph_edges, context_graph, flip_direction="tf")

    # node_similarity_index = create_node_similarity_index(context_graph_nodes, query_embedding)
    # edge_similarity_index = await create_edge_similarity_index(edge_labels, query_embedding)


    ##################### For Addition #####################
    edge_labels = {(u, v): data.get("description", "") for u, v, data in G.edges(data=True)}

    node_replacement_index = create_node_replacement_index(context_graph_nodes, G, flip_direction="tf")
    edge_replacement_index = create_edge_replacement_index(context_graph_edges, G, flip_direction="tf")

    node_similarity_index = create_node_similarity_index(set(G.nodes), query_embedding)
    edge_similarity_index = await create_edge_similarity_index(edge_labels, query_embedding)
    ##################### For Addition #####################


    llm_calls = 0

    Q = []

    print(f"Context Graph Nodes: {context_graph_nodes}")
    print(f"Context Graph Edges: {context_graph_edges}")

    ### Prune seen context graph.
    state_cache = set()

    heapq.heappush(Q, (0, 0.0, next(counter), (context_graph, [])))

    explored_nodes = set()  ## For addition

    while Q:
        # print(f"\nMin-Heap: {Q}")

        cost, _, _, (cg, ops) = heapq.heappop(Q)

        if cost > max_cost:
            print(f"Max cost {max_cost} exceeded (current cost: {cost:.4f}). Stopping search.")
            break
        elif llm_calls > max_llm_calls:
            print(f"Max LLM calls {max_llm_calls} exceeded. Stopping search.")
            break

        ### Check state cache
        # state = frozenset(
        #     (u, v, cg.edges[u, v].get("description", ""))
        #     for u, v in cg.edges
        # )

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

        print(f"Operations: {ops}")
        if len(ops) > 0:
            # print(f"\nCurrent Operation set: {ops}")

            cg_context = graph_to_context(cg)
            # print(f"Context: {cg_context}\n")
            
            ### Explanation Stability/Consistency
            # cg_context = graph_to_context_shuffled(cg, shuffle_entities=True, shuffle_relations=True)

            new_response = await query(rag, cg_context, question)

            print(f"Cost: {cost} | New response: {new_response} | Original: {original_answer}")

            ### For addition:
            print(f"Ground Truth: {ground_truth}")

            # score = await judge_response(question, new_response, original_answer)

            ### For addition:
            score = await judge_response(question, new_response, ground_truth)

            llm_calls += 1

            # if score == 0:

            ### For addition:
            if score == 1:
                print(f"Counterfactual Operations: {ops}")

                # answer_similarity = await compute_answer_similarity(original_answer, new_response)

                ### For addition:
                answer_similarity = await compute_answer_similarity(ground_truth, new_response)
                
                # print(f"Answer similarity (original vs perturbed): {answer_similarity:.4f}")

                print(f"Answer similarity (ground truth vs perturbed): {answer_similarity:.4f}")

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
                    current_ops=current_ops
                )
                return ops
            
        expand(Q, (cost, cg, ops), node_replacement_index=node_replacement_index, edge_replacement_index=edge_replacement_index, node_similarity_index=node_similarity_index, edge_similarity_index=edge_similarity_index, unit_cost=unit_cost, current_ops=current_ops, original_nodes=context_graph_nodes, original_edges=context_graph_edges, explored_nodes=explored_nodes)

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
        cost=cost,
        current_ops=current_ops
    )



def expand(
        Q, 
        heap_element, 
        node_replacement_index, 
        edge_replacement_index, 
        node_similarity_index, 
        edge_similarity_index, 
        unit_cost: bool = False, 
        current_ops: list=["delete_node", "delete_edge", "replace_node", "replace_edge"],
        original_nodes: set = {},
        original_edges: set = {},
        explored_nodes: set = {},
    ):
    cg: nx.DiGraph
    cost, cg, ops = heap_element

    undirected: nx.Graph = cg.to_undirected()
    cut_vertices = set(nx.articulation_points(cg.to_undirected()))
    cut_edges = set(nx.bridges(cg.to_undirected()))

    if "delete_node" in current_ops:
        ### Updated Delete Node
        # Allow if not a cut vertex, OR if it is a cut vertex but all neighbors
        # would become isolated (meaning no real split, just singleton cleanup)
        for node in list(cg.nodes):
            ### Feasibility Constraint
            if node in cut_vertices:
                neighbors = list(undirected.neighbors(node))
                
                would_isolate = {n for n in neighbors if undirected.degree(n) == 1}
                nodes_to_remove = {node} | would_isolate
                residual = undirected.copy()
                residual.remove_nodes_from(nodes_to_remove)

                components_before = nx.number_connected_components(undirected)
                components_after = nx.number_connected_components(residual)

                if components_after > components_before:
                    continue

            perturbed_cg = delete_node(cg, node)
            
            if unit_cost == False:
                perturbation_cost = delete_node_cost(cg, node) 
            elif unit_cost == True:
                perturbation_cost = delete_node_uc(cg, node)

            new_ops = ops + [("delete_node", node)]

            similarity = node_similarity_index.get(node, 0.0)

            heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

    if "delete_edge" in current_ops:
        ### Updated Delete Edge
        # Allow if not a cut edge, OR if it is a cut edge but both endpoints
        # would become isolated (meaning no real split, just singleton cleanup)
        for edge in list(cg.edges):
            ### Feasibility Constraint
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

    if "add_node" in current_ops:
        existing_nodes = set(cg.nodes)
        existing_edges = set(cg.edges())
        candidate_nodes_for_expansion = existing_nodes - explored_nodes

        print(f"Existing Nodes: {existing_nodes}")
        print(f"Candidate Nodes: {candidate_nodes_for_expansion}")
        print(f"Explored Nodes: {explored_nodes}")

        for node in candidate_nodes_for_expansion:
            # neighbors = list(G.successors(node)) + list(G.predecessors(node))
            neighbors = list(G.neighbors(node))

            print(f"Current neighbors: {neighbors}")
            
            similarity = node_similarity_index.get(node, 0.0)

            for neighbor in neighbors:
                
                print(f"Neighbor: {neighbor}")

                if neighbor not in existing_nodes:
                    perturbed_cg = add_node(cg, neighbor, **G.nodes[neighbor])

                    if (node, neighbor) in edge_lookup and (node, neighbor) not in existing_edges:
                        perturbed_cg = add_edge(cg, (node, neighbor), **G.edges[node, neighbor])
                    
                    if (neighbor, node) in edge_lookup and (neighbor, node) not in existing_edges:
                        perturbed_cg = add_edge(cg, (neighbor, node), **G.edges[neighbor, node])

                    print(f"New nodes: {perturbed_cg.nodes}")
                    print(f"New edges: {perturbed_cg.edges()}")

                    if unit_cost == False:
                        perturbation_cost = add_node_cost(cg, node_embeddings, node_lookup, edge_embeddings, edge_lookup, neighbor)
                    elif unit_cost == True:
                        perturbation_cost = add_node_uc(cg, neighbor)

                    new_ops = ops + [("add_node", neighbor)]

                    heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

            explored_nodes.add(node)

    #########################################################################################
    ##################################TBD####################################################

    if "replace_node" in current_ops:
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
            
            if unit_cost == False:
                perturbation_cost = 1 - sim
            elif unit_cost == True:
                perturbation_cost = replace_node_uc()

            new_ops = ops + [("replace_node", (node, current_replacement))]

            similarity = node_similarity_index.get(node, 0.0)

            heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))

    if "replace_edge" in current_ops:
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
            
            if unit_cost == False:
                perturbation_cost = 1 - sim
            elif unit_cost == True:
                perturbation_cost = replace_edge_uc()

            new_ops = ops + [("replace_edge", (edge, current_replacement))]

            similarity = edge_similarity_index.get(edge, 0.0)

            heapq.heappush(Q, (cost + perturbation_cost, -similarity, next(counter), (perturbed_cg, new_ops)))



def save_operations_to_json(ops: list, question: str, original_answer: str, perturbed_answer: str, answer_similarity: float, original_subgraph, perturbed_subgraph, output_dir: str = "src/counterfactuals/results", filename: str = None, found: bool = True, cost: float = 0.0, llm_calls: int = 0, current_ops: list=[]):
    # os.makedirs(output_dir, exist_ok=True)

    if current_ops == ["delete_node", "delete_edge", "replace_node", "replace_edge"]:
        output_dir = f"{output_dir}_sem_all"
    elif current_ops == ["delete_node"]:
        output_dir = f"{output_dir}_delete_node"
    elif current_ops == ["delete_edge"]:
        output_dir = f"{output_dir}_delete_edge"
    elif current_ops == ["replace_node"]:
        output_dir = f"{output_dir}_replace_node"
    elif current_ops == ["replace_edge"]:
        output_dir = f"{output_dir}_replace_edge"
    elif current_ops == ["delete_node", "delete_edge"]:
        output_dir = f"{output_dir}/delete_only_20_check"
    elif current_ops == ["add_node"]:
        output_dir = f"{output_dir}/add_node_only"
    else:
        output_dir = f"{output_dir}_uc_all"
        
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
    
    with open(f"benchmark/results/degraded_results.json", "r", encoding="utf-8") as results:
        data = json.load(results)

    operation_sets = [
        # ["delete_node", "delete_edge", "replace_node", "replace_edge"],
        # ["delete_node"],
        # ["delete_edge"],
        # ["replace_node"],
        # ["replace_edge"],
        ["add_node"]
        # ["delete_node", "delete_edge"],
    ]

    for op_set in operation_sets:

        results = data["results"]
        for idx, r in results.items():

            question = r["question"]
            case = r["case"]

            ground_truth = r["ground_truth"]

            if case != "ff":
                continue

            print(f"\n=== [{idx}] {question} ===")

            context = await retrieve_subgraph(rag, query=question, mode="hybrid", top_k=1)
            await find_counterfactuals(rag, question, context=context, max_cost=20, max_llm_calls=200, unit_cost=False, current_ops=op_set, ground_truth=ground_truth)

if __name__ == "__main__":
    asyncio.run(main())