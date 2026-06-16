from src.base import Subgraph, Entity, Relation
from src.query import query
from src.retrieve import initialize_lightrag
from src.parser import parse_graph, graph_to_context
from src.llm_judge import judge_response

import json
import os
import asyncio
import argparse
import networkx as nx


def load_explanation(data: dict, G):
    if data.get("degenerate", False):
        return None

    question     = data["question"]
    ground_truth = data["ground_truth"]

    # New format: flat node_attributions / edge_attributions lists
    top_nodes = data.get("node_attributions", [])
    top_edges = data.get("edge_attributions", [])

    selected_nodes = [n for n in top_nodes if n["attribution"] > 0]
    selected_edges = [e for e in top_edges if e["attribution"] > 0]

    entities = []
    for n in selected_nodes:
        name      = n["node"]
        node_data = G.nodes.get(name, {})
        entities.append(Entity(
            name=name,
            type=node_data.get("entity_type", ""),
            description=node_data.get("description", ""),
            rank=0,
        ))

    relations = []
    for e in selected_edges:
        src, tgt  = e["source"], e["target"]
        edge_data = G.edges.get((src, tgt), {})
        relations.append(Relation(
            src=src,
            tgt=tgt,
            keywords=edge_data.get("keywords", ""),
            description=edge_data.get("description", ""),
            weight=0.0,
        ))

    return {
        "question":     question,
        "ground_truth": ground_truth,
        "subgraph_obj": Subgraph(entities=entities, relations=relations),
    }


async def evaluate_explanation(rag, entry: dict, G):
    explanation_dict = load_explanation(entry, G)

    if explanation_dict is None:
        return None

    try:
        graph = parse_graph(explanation_dict["subgraph_obj"])
    except Exception as e:
        print(f"\nDEBUG subgraph_obj: {explanation_dict['subgraph_obj']}")
        raise e

    context  = graph_to_context(graph)
    response = await query(rag, context, explanation_dict["question"])

    score = await judge_response(
        question=explanation_dict["question"],
        generated_answer=response,
        ground_truth=explanation_dict["ground_truth"],
    )

    return score == 1


WORKING_DIRS = {
    "hotpotqa":  "KGs/lightrag/hotpotqa",
    "synthetic": "KGs/lightrag/synthetic",
    "musique": "KGs/lightrag/musique"
}


async def main():
    parser = argparse.ArgumentParser(description="Evaluate attribution explanations.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", type=str, default="ft")
    args = parser.parse_args()

    input_path = f"kg_smile/results/kg_smile_results_{args.dataset}_{args.mode}.json"
    graph_path = f"/home/gbalanos/GloRAG-Ex/code/KGs/lightrag/{args.dataset}/graph_chunk_entity_relation.graphml"

    print(f"Dataset:  {args.dataset}")
    print(f"Loading entries from {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # New format is a dict {"0": {...}, "1": {...}, ...}; sort by numeric key
    entries = [raw[k] for k in sorted(raw, key=int)]
    print(f"Found {len(entries)} entries\n")

    rag = await initialize_lightrag(working_dir=WORKING_DIRS[args.dataset])
    G   = nx.read_graphml(graph_path)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    results      = []
    entry_results = []

    for i, entry in enumerate(entries):
        # Prefer the stored id; fall back to the dict key order index
        entry_id = entry.get("id", i)
        print(f"[{i+1}/{len(entries)}] Evaluating entry {entry_id}...", end=" ")

        # Skip entries that failed during the run pipeline
        if "error" in entry:
            print(f"SKIPPED (runner error: {entry['error']})")
            entry_results.append({"id": entry_id, "correct": None, "skipped": True,
                                   "error": entry["error"]})
            continue

        try:
            result = await evaluate_explanation(rag, entry, G)
            if result is None:
                print("SKIPPED (degenerate)")
                entry_results.append({"id": entry_id, "correct": None, "skipped": True})
            else:
                results.append(result)
                entry_results.append({"id": entry_id, "correct": result})
                print("✓" if result else "✗")
        except Exception as e:
            print(f"SKIPPED ({type(e).__name__}: {e})")
            entry_results.append({"id": entry_id, "correct": None, "error": str(e)})

    total    = len(results)
    correct  = sum(results)
    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.2%})")

    output = {
        "dataset":  args.dataset,
        "accuracy": accuracy,
        "correct":  correct,
        "total":    total,
        "entries":  entry_results,
    }

    output_path = f"kg_smile/quality_metrics/results_suff/{args.dataset}_{args.mode}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())