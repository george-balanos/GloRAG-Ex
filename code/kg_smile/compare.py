"""
compare.py
==========
Compares KG-SMILE attributions against your method's outputs using
graph centrality metrics (degree, betweenness) on the selected nodes/edges.

Reads two result folders and prints a side-by-side comparison table.
"""

import json
from pathlib import Path

import numpy as np

from .graph_utils import build_graph_from_result, compute_centrality, centrality_scores
from .io_utils    import load_folder


# ─────────────────────────────────────────────────────────────
# Component extraction
# ─────────────────────────────────────────────────────────────

def extract_your_method_components(
    result: dict,
) -> tuple[set[str], set[tuple[str, str]]] | None:
    """
    Extract the nodes and edges touched by your method's operations.
    Returns None if the method did not find a solution (found=False).
    """
    if not result.get("found"):
        return None

    nodes: set[str]              = set()
    edges: set[tuple[str, str]]  = set()

    for op_type, target in result.get("operations", []):
        if op_type in ("delete_node", "add_node"):
            nodes.add(target)
        elif op_type in ("delete_edge", "add_edge"):
            edges.add(tuple(target))

    return nodes, edges


def extract_kgsmile_components(
    result: dict,
    top_k: int,
) -> tuple[set[str], set[tuple[str, str]]]:
    """Extract the top-k nodes and edges from a KG-SMILE attribution result."""
    edges = {(e["source"], e["target"]) for e in result["edge_attributions"][:top_k]}
    nodes = {n["node"]                  for n in result["node_attributions"][:top_k]}
    return nodes, edges


# ─────────────────────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────────────────────

METRICS = [
    "mean_node_degree",
    "mean_node_betweenness",
    "mean_edge_betweenness",
    "max_node_degree",
    "max_node_betweenness",
    "max_edge_betweenness",
]


def run_comparison(
    your_folder:    str,
    kgsmile_folder: str,
    top_k:          int = 5,
    output_path:    str = "results/centrality_comparison.json",
) -> None:
    your_results    = load_folder(your_folder)
    kgsmile_results = load_folder(kgsmile_folder)

    matched_questions = set(your_results) & set(kgsmile_results)
    print(f"[compare] {len(matched_questions)} matched questions")

    per_question:      list[dict] = []
    skipped_not_found: int        = 0

    for question in sorted(matched_questions):
        your_result    = your_results[question]
        kgsmile_result = kgsmile_results[question]

        your_components = extract_your_method_components(your_result)
        if your_components is None:
            skipped_not_found += 1
            continue

        your_nodes,    your_edges    = your_components
        kgsmile_nodes, kgsmile_edges = extract_kgsmile_components(kgsmile_result, top_k)

        G          = build_graph_from_result(your_result)
        centrality = compute_centrality(G)

        per_question.append({
            "question":    question,
            "cost":        your_result["cost"],
            "found":       your_result["found"],
            "your_method": centrality_scores(your_nodes, your_edges, centrality),
            "kg_smile":    centrality_scores(kgsmile_nodes, kgsmile_edges, centrality),
        })

    print(f"[compare] Skipped {skipped_not_found} not-found questions")
    print(f"[compare] Evaluating over {len(per_question)} questions\n")

    # Aggregate per-metric means
    aggregated = {
        metric: {
            "your_method": float(np.mean([r["your_method"][metric] for r in per_question])),
            "kg_smile":    float(np.mean([r["kg_smile"][metric]    for r in per_question])),
        }
        for metric in METRICS
    }

    # Print comparison table
    sep = "=" * 68
    print(sep)
    print(f"CENTRALITY COMPARISON  (top_k={top_k}, n={len(per_question)} questions)")
    print(sep)
    print(f"{'Metric':<30}  {'Your Method':>12}  {'KG-SMILE':>12}  {'Winner':>10}")
    print("-" * 68)
    for metric, vals in aggregated.items():
        yours   = vals["your_method"]
        kgsmile = vals["kg_smile"]
        winner  = (
            "Yours ✓"  if yours   < kgsmile - 1e-6 else
            "KG-SMILE" if kgsmile < yours   - 1e-6 else
            "Tie"
        )
        print(f"{metric:<30}  {yours:>12.4f}  {kgsmile:>12.4f}  {winner:>10}")
    print(sep)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "top_k":             top_k,
            "n_questions":       len(per_question),
            "skipped_not_found": skipped_not_found,
            "aggregated":        aggregated,
            "per_question":      per_question,
        }, f, indent=2)

    print(f"\n[✓] Saved to {output_path}")


if __name__ == "__main__":
    run_comparison(
        your_folder="results/your_method/",
        kgsmile_folder="results/kg_smile_results/",
        top_k=5,
        output_path="results/centrality_comparison.json",
    )
