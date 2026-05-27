from datetime import datetime
from src.query import *
from src.retrieve import *
from src.parser import *
from src.llm_judge import judge_response
from src.counterfactuals.edit_costs import *
from src.counterfactuals.perturbations import *
from src.counterfactuals.utils import compute_answer_similarity, cosine_similarity_norm
from src.embeddings.query import get_embedding
from src.dataset_setup import (
    WORKING_DIRS,
    DATASETS,
    setup_dataset as _shared_setup_dataset,
)

import argparse
import heapq
import networkx as nx
import asyncio
import itertools
import os
import random

### Setup ###

counter = itertools.count()

dataset: str = "synthetic"
G = None
type_index = None
node_index = node_records = node_embeddings = node_lookup = None
edge_index = edge_records = edge_embeddings = edge_lookup = None


def setup_dataset(name: str):
    """(Re)bind module-level dataset globals via the shared loader in src.dataset_setup."""
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


setup_dataset("synthetic")

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

async def find_counterfactuals(
    rag, 
    question: str, 
    context, 
    example, 
    max_cost=3, 
    max_llm_calls=100, 
    unit_cost: bool=False, 
    seed=None, 
    noise_pct=0.1
):
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
            perturbed_answer=noisy_response,
            answer_similarity=0.0,
            original_subgraph=parsed_subgraph,
            perturbed_subgraph=graph_to_subgraph(noisy_cg),
            noisy_subgraph=parse_context(noisy_context),
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

    edge_labels = {(u, v): data.get("description", "") for u, v, data in G.edges(data=True)}

    node_similarity_index = create_node_similarity_index(set(G.nodes), query_embedding)
    edge_similarity_index = await create_edge_similarity_index(edge_labels, query_embedding)

    llm_calls = 0

    Q = []

    ### Prune seen context graph.
    state_cache = set()

    # heapq.heappush(Q, (0, 0.0, next(counter), (context_graph, [])))
    heapq.heappush(Q, (0, 0, 0.0, next(counter), (context_graph, [])))

    while Q:
        # cost, _, _, (cg, ops) = heapq.heappop(Q)
        cost, _, _, _, (cg, ops) = heapq.heappop(Q)

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
                    noisy_subgraph=parse_context(noisy_context),
                    found=True,
                    cost=cost,
                    llm_calls=llm_calls,
                    noise_metadata=noise_metadata,
                    noise_p=noise_pct
                )
                return ops
            
        expand(Q, (cost, cg, ops), node_similarity_index=node_similarity_index, edge_similarity_index=edge_similarity_index, unit_cost=unit_cost)

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
        noisy_subgraph=parse_context(noisy_context),
        found=False,
        cost=cost,
        llm_calls=llm_calls,
        noise_p=noise_pct
    )

def expand(
    Q, 
    heap_element,
    node_similarity_index, 
    edge_similarity_index, 
    unit_cost: bool = False
    ):
    cg: nx.DiGraph
    cost, cg, ops = heap_element

    undirected: nx.Graph = cg.to_undirected()
    cut_vertices = set(nx.articulation_points(cg.to_undirected()))
    cut_edges = set(nx.bridges(cg.to_undirected()))

    for node in list(cg.nodes):
        ### Feasibility Constraint
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
        heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))

    for edge in list(cg.edges):
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
        heapq.heappush(Q, (cost + perturbation_cost, len(new_ops), -similarity, next(counter), (perturbed_cg, new_ops)))


def save_operations_to_json(ops: list,question: str, original_answer: str, perturbed_answer: str, answer_similarity: float, original_subgraph, perturbed_subgraph, noisy_subgraph, output_dir: str = f"src/counterfactuals/robustness/{dataset}/noise_resistance",filename: str = None, found: bool = True, cost: float = 0.0, llm_calls: int = 0, noise_metadata: dict = None, noise_p=0.1):

    noise = noise_p*100
    output_dir = f"{output_dir}/noise_level_{int(noise)}"
        
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
        "noisy_subgraph": subgraph_to_dict(noisy_subgraph),
        "timestamp": datetime.now().isoformat(),
        "llm_calls": llm_calls
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Operations saved to: {filepath}")
    return filepath


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="noise_resistance",
        description="Noise-resistance quality metric over an existing CFE result set.",
    )
    p.add_argument("--dataset", choices=DATASETS, default="synthetic")
    p.add_argument("--rag-mode", choices=["hybrid", "local", "global", "naive"], default="hybrid")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--noise-percentages", default="0.1,0.3,0.5,0.8",
                   help="Comma-separated noise fractions, e.g. '0.1,0.3,0.5,0.8'.")
    p.add_argument("--results-folder", default=None,
                   help="Folder of CFE result JSONs to load (default: src/counterfactuals/results/<dataset>/with_f3/delete_ops_ft).")
    p.add_argument("--max-cost", type=int, default=20)
    p.add_argument("--max-llm-calls", type=int, default=200)
    p.add_argument("--unit-cost", action="store_true")
    return p


async def main(args: argparse.Namespace):
    if args.dataset != dataset:
        setup_dataset(args.dataset)

    noise_percentages = [float(x) for x in args.noise_percentages.split(",") if x.strip()]
    results_folder = args.results_folder or f"src/counterfactuals/results/{dataset}/with_f3/delete_ops_ft"

    rag = await initialize_lightrag(working_dir=WORKING_DIRS[dataset])

    json_files = [f for f in os.listdir(results_folder) if f.endswith(".json")]

    for noise_p in noise_percentages:
        for i, json_file in enumerate(json_files):
            filepath = os.path.join(results_folder, json_file)
            print(f"\n=== Loading: {json_file} ===")

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            question = data["question"]

            print(f"\n=== {question} ===")

            context = await retrieve_subgraph(rag, query=question, mode=args.rag_mode, top_k=args.top_k)
            await find_counterfactuals(
                rag, question, context=context,
                max_cost=args.max_cost, max_llm_calls=args.max_llm_calls,
                unit_cost=args.unit_cost,
                example=data, seed=i, noise_pct=noise_p,
            )


if __name__ == "__main__":
    asyncio.run(main(build_arg_parser().parse_args()))