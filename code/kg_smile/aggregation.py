#!/usr/bin/env python3
"""
aggregation.py
==============
Evaluates KG-SMILE attributions against ground-truth graph-edit sequences
by simulating the attributed operations sequentially and comparing costs.

Reads:
  --attributions  flat attribution JSON produced by runner.py (normal mode)
  --results-dir   directory of per-question result JSONs from the main pipeline
"""

import argparse
import json
from statistics import mean, median

from .graph_utils import execute_and_cost_node_deletion, execute_and_cost_edge_deletion
from .io_utils    import load_attributions, load_results


# ─────────────────────────────────────────────────────────────
# Degeneracy detection
# ─────────────────────────────────────────────────────────────

def is_degenerate(entry: dict) -> bool:
    """Return True if node and edge attributions are all identical (flat/uninformative)."""
    def all_same(vals: list) -> bool:
        return len(set(vals)) <= 1

    node_vals = [x.get("attribution", 0.0) for x in entry.get("node_attributions", [])]
    edge_vals = [x.get("attribution", 0.0) for x in entry.get("edge_attributions", [])]
    return all_same(node_vals) and all_same(edge_vals)


# ─────────────────────────────────────────────────────────────
# Attribution extraction
# ─────────────────────────────────────────────────────────────

def extract_top_nodes(entry: dict, k: int) -> list[str]:
    node_attrs = entry.get("node_attributions") or entry.get("answers", {}).get("top_nodes", [])
    node_attrs = sorted(node_attrs, key=lambda x: x["attribution"], reverse=True)
    return [x["node"] for x in node_attrs[:k]]


def extract_top_edges(entry: dict, k: int) -> list[tuple[str, str]]:
    edge_attrs = entry.get("edge_attributions") or entry.get("answers", {}).get("top_edges", [])
    edge_attrs = sorted(edge_attrs, key=lambda x: x["attribution"], reverse=True)
    return [
        (x.get("source") or x.get("src"), x.get("target") or x.get("tgt"))
        for x in edge_attrs[:k]
    ]


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(
    attributions: list[dict],
    graphs_by_question: dict,
    ops_by_question:    dict,
    costs_by_question:  dict,
    found_by_question:  dict,
) -> tuple[dict, list[dict]]:
    """
    For each attribution entry, simulate the attributed operations on the graph
    and compare the incurred cost against the ground-truth cost.

    Returns (summary, details) where details is one dict per evaluated question.
    """
    details: list[dict] = []

    attributed_costs: list[float] = []
    actual_costs:     list[float] = []
    gaps:             list[float] = []
    abs_gaps:         list[float] = []
    ratios:           list[float] = []

    skipped_not_found  = 0
    skipped_degenerate = 0
    missing_questions  = 0

    for entry in attributions:
        q = entry["question"]

        if q not in graphs_by_question:
            missing_questions += 1
            continue

        if not found_by_question.get(q, False):
            skipped_not_found += 1
            continue

        if entry.get("degenerate", False) or is_degenerate(entry):
            skipped_degenerate += 1
            continue

        actual_cost = costs_by_question[q]
        if actual_cost is None:
            continue

        ops = ops_by_question[q]
        n_node_ops = sum(1 for op in ops if op[0] == "delete_node")
        n_edge_ops = sum(1 for op in ops if op[0] == "delete_edge")

        top_nodes = extract_top_nodes(entry, n_node_ops)
        top_edges = extract_top_edges(entry, n_edge_ops)

        # Simulate the attributed operations sequentially
        G = graphs_by_question[q].copy()
        node_i = edge_i = attributed_cost = 0

        for op in ops:
            if op[0] == "delete_node" and node_i < len(top_nodes):
                G, cost = execute_and_cost_node_deletion(G, top_nodes[node_i])
                attributed_cost += cost
                node_i += 1

            elif op[0] == "delete_edge" and edge_i < len(top_edges):
                G, cost = execute_and_cost_edge_deletion(G, top_edges[edge_i])
                attributed_cost += cost
                edge_i += 1

        gap   = attributed_cost - actual_cost
        ratio = attributed_cost / actual_cost if actual_cost > 0 else None

        attributed_costs.append(attributed_cost)
        actual_costs.append(actual_cost)
        gaps.append(gap)
        abs_gaps.append(abs(gap))
        if ratio is not None:
            ratios.append(ratio)

        details.append({
            "question":        q,
            "actual_cost":     actual_cost,
            "attributed_cost": attributed_cost,
            "gap":             gap,
            "abs_gap":         abs(gap),
            "ratio":           ratio,
        })

    summary = {
        "questions_evaluated":        len(details),
        "questions_missing":          missing_questions,
        "questions_skipped_not_found":    skipped_not_found,
        "questions_skipped_degenerate":   skipped_degenerate,
        "mean_actual_cost":           mean(actual_costs)     if actual_costs else 0,
        "mean_attributed_cost":       mean(attributed_costs) if attributed_costs else 0,
        "mean_gap":                   mean(gaps)             if gaps else 0,
        "mean_abs_gap":               mean(abs_gaps)         if abs_gaps else 0,
        "mean_ratio":                 mean(ratios)           if ratios else 0,
        "median_ratio":               median(ratios)         if ratios else 0,
    }

    return summary, details


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attributions", required=True)
    parser.add_argument("--results-dir",  required=True)
    parser.add_argument("--output",       default=None)
    args = parser.parse_args()

    attributions = load_attributions(args.attributions)
    graphs, ops, costs, found = load_results(args.results_dir)
    summary, details = evaluate(attributions, graphs, ops, costs, found)

    print("\n============== SUMMARY ==============")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"summary": summary, "details": details}, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
