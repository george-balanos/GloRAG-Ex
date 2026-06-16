from src.base import Subgraph, Entity, Relation
from src.query import query
from src.retrieve import initialize_lightrag
from src.parser import parse_graph, graph_to_context
from src.llm_judge import judge_response

import json
import os
import asyncio
import argparse

def load_explanation(filepath: str, mode="ft"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    found = data["found"]
    if not found:
        return

    question = data["question"]
    operations = data["operations"]
    ground_truth = data["answers"]["ground_truth"]

    original_entities = data["original_subgraph"]["entities"] if data.get("original_subgraph") else []
    original_relations = data["original_subgraph"]["relations"] if data.get("original_subgraph") else []

    perturbed_entities = data["perturbed_subgraph"]["entities"] if data.get("perturbed_subgraph") else []
    perturbed_relations = data["perturbed_subgraph"]["relations"] if data.get("perturbed_subgraph") else []

    if mode == "ft":
        important_nodes = []
        important_edges = []
        for op in operations:
            if op[0] == "delete_edge":
                important_edges.append(op[1])
            elif op[0] == "delete_node":
                important_nodes.append(op[1])

        entities = []
        relations = []

        for node in important_nodes:
            metadata = get_node_metadata(node, original_entities)
            if metadata is not None:
                entities.append(
                    Entity(
                        name=metadata["name"],
                        type=metadata["type"],
                        description=metadata["description"],
                        rank=metadata["rank"]
                    )
                )

        for rel in important_edges:
            metadata = get_edge_metadata(rel, original_relations)
            if metadata is not None:
                relations.append(
                    Relation(
                        src=metadata["src"],
                        tgt=metadata["tgt"],
                        keywords=metadata["keywords"],
                        description=metadata["description"],
                        weight=metadata["weight"]
                    )
                )

        subgraph_obj = Subgraph(entities=entities, relations=relations)
    
    elif mode in ("ff", "tf"):
        ## Ignore deletions in this mode.
        important_nodes = []
        important_edges = []
        for op in operations:
            if op[0] == "add_edge":
                important_edges.append(op[1])
            elif op[0] == "add_node":
                important_nodes.append(op[1])

        entities = []
        relations = []

        for node in important_nodes:
            metadata = get_node_metadata(node, perturbed_entities)
            if metadata is not None:
                entities.append(
                    Entity(
                        name=metadata["name"],
                        type=metadata["type"],
                        description=metadata["description"],
                        rank=metadata["rank"]
                    )
                )

        for rel in important_edges:
            metadata = get_edge_metadata(rel, perturbed_relations)
            if metadata is not None:
                relations.append(
                    Relation(
                        src=metadata["src"],
                        tgt=metadata["tgt"],
                        keywords=metadata["keywords"],
                        description=metadata["description"],
                        weight=metadata["weight"]
                    )
                )

        subgraph_obj = Subgraph(entities=entities, relations=relations)

    return {
        "question": question,
        "ground_truth": ground_truth,
        "subgraph_obj": subgraph_obj
    }

def get_node_metadata(node: str, entities: list[dict]):
    for entry in entities:
        if entry["name"] == node:
            return entry
    
    return None

def get_edge_metadata(edge: tuple, edges: list[dict]):
    for entry in edges:
        if entry["src"] == edge[0] and entry["tgt"] == edge[1]:
            return entry
        
    return None

async def evaluate_explanation(rag, filepath: str, mode="ft"):
    explanation_dict = load_explanation(filepath, mode=mode)

    if explanation_dict is None:
        return None

    graph = parse_graph(explanation_dict["subgraph_obj"])
    context = graph_to_context(graph)

    response = await query(rag, context, explanation_dict["question"])

    score = await judge_response(
        question=explanation_dict["question"],
        generated_answer=response,
        ground_truth=explanation_dict["ground_truth"]
    )

    if score == 1:
        return True
    return False

WORKING_DIRS = {
    "hotpotqa": "KGs/lightrag/hotpotqa",
    "synthetic": "KGs/lightrag/synthetic",
    "musique": "KGs/lightrag/musique"
}

async def main():
    parser = argparse.ArgumentParser(description="Evaluate counterfactual explanations.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--partition", type=str, required=True)
    parser.add_argument("--mode", type=str, default="ft", choices=["ft", "ff", "tf"])
    args = parser.parse_args()

    explanation_dir = f"src/counterfactuals/results/{args.dataset}/{args.partition}"

    print(f"Dataset: {args.dataset} | Partition: {args.partition} | Mode: {args.mode}")

    rag = await initialize_lightrag(working_dir=WORKING_DIRS[args.dataset])

    json_files = [
        os.path.join(explanation_dir, f)
        for f in os.listdir(explanation_dir)
        if f.endswith(".json")
    ]

    print(f"Found {len(json_files)} files in {explanation_dir}\n")

    results = []
    file_results = []
    for i, filepath in enumerate(json_files):
        filename = os.path.basename(filepath)
        print(f"[{i+1}/{len(json_files)}] Evaluating {filename}...", end=" ")
        try:
            result = await evaluate_explanation(rag, filepath, mode=args.mode)
            if result is None:
                print("SKIPPED (not found)")
                file_results.append({"file": filename, "correct": None, "skipped": True})
            else:
                results.append(result)
                file_results.append({"file": filename, "correct": result})
                print("✓" if result else "✗")
        except Exception as e:
            print(f"SKIPPED ({type(e).__name__}: {e})")
            file_results.append({"file": filename, "correct": None, "error": str(e)})


    total = len(results)
    correct = sum(results)
    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.2%})")

    output = {
        "dataset": args.dataset,
        "partition": args.partition,
        "mode": args.mode,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "files": file_results
    }

    output_path = f"src/quality_metrics/results_suff/{args.dataset}/{args.partition}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())