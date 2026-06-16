"""
evaluation.py
=============
Per-question Kendall τ robustness scoring (used programmatically by runner.py).

Computes stability of KG-SMILE attributions under graph noise by comparing
the noise=0 baseline ranking against each noisy run's ranking.

Imports ranking utilities from ranking.py; does not re-implement them.
"""

from __future__ import annotations

import json
from pathlib import Path

from .ranking import (
    edge_rank_list,
    node_rank_list,
    kendall_tau_edges,
    kendall_tau_nodes,
    robustness_auc_from_scores,
)


# ─────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_kg_smile_robustness(baseline: dict, benchmarks: list[dict]) -> dict:
    """
    Compute per-noise-level Kendall τ for a single question.

    Args:
        baseline:   The noise=0 result dict (contains node_attributions, edge_attributions).
        benchmarks: List of {"noise_pct": float, "result": dict} for noise > 0.

    Returns a dict with edge_scores, node_scores, edge_auc, node_auc.
    """
    base_edges = edge_rank_list(baseline["edge_attributions"])
    base_nodes = node_rank_list(baseline["node_attributions"])

    edge_scores: list[dict] = []
    node_scores: list[dict] = []

    for run in benchmarks:
        result = run.get("result")
        if result is None:      # error run — skip
            continue

        noise = run["noise_pct"]
        noisy_edges = edge_rank_list(result["edge_attributions"])
        noisy_nodes = node_rank_list(result["node_attributions"])

        edge_scores.append({"noise": noise, "tau": kendall_tau_edges(base_edges, noisy_edges)})
        node_scores.append({"noise": noise, "tau": kendall_tau_nodes(base_nodes, noisy_nodes)})

    return {
        "edge_scores": edge_scores,
        "node_scores": node_scores,
        "edge_auc":    robustness_auc_from_scores(edge_scores),
        "node_auc":    robustness_auc_from_scores(node_scores),
    }


# ─────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────

def print_robustness_report(report: dict) -> None:
    print("\n" + "=" * 60)
    print("KG-SMILE ROBUSTNESS REPORT")
    print("=" * 60)

    print("\nEdge Stability:")
    for r in report["edge_scores"]:
        tau_str = f"{r['tau']:.4f}" if r["tau"] is not None else "N/A"
        print(f"  Noise {r['noise']*100:>3.0f}% -> tau = {tau_str}")
    print(f"\nEdge AUC: {report['edge_auc']:.4f}")

    print("\nNode Stability:")
    for r in report["node_scores"]:
        tau_str = f"{r['tau']:.4f}" if r["tau"] is not None else "N/A"
        print(f"  Noise {r['noise']*100:>3.0f}% -> tau = {tau_str}")
    print(f"\nNode AUC: {report['node_auc']:.4f}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, required=True,
                        help="Benchmark JSON produced by: python -m kg_smile.runner robustness")
    parser.add_argument("--output", type=str, default="robustness_report.json")
    parser.add_argument("--print",  action="store_true")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(args.input)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_reports = []

    for item in data:
        if "benchmark" not in item or "error" in item:
            continue

        runs           = sorted(item["benchmark"], key=lambda x: x["noise_pct"])
        baseline_runs  = [r for r in runs if r["noise_pct"] == 0.0]
        noisy_runs     = [r for r in runs if r["noise_pct"] >  0.0]

        if not baseline_runs:
            print(f"[SKIP] No noise=0 baseline for: {item.get('question', '?')[:60]}")
            continue

        report = evaluate_kg_smile_robustness(
            baseline   = baseline_runs[0]["result"],
            benchmarks = noisy_runs,
        )

        all_reports.append({
            "id":       item.get("id"),
            "question": item.get("question"),
            "report":   report,
        })

        if args.print:
            print_robustness_report(report)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)

    print(f"\nSaved -> {args.output}")
