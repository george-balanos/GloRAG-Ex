"""Aggregate analysis over saved CFE result directories.

Loads JSON outputs from `find_counterfactuals` and reports success rate,
operation-count and cost distributions, answer-similarity histograms, graph
size statistics, star-structure and cut-vertex/bridge analyses, and how
operation counts scale with graph size. Read-only.
"""

import json
import os
from collections import Counter
from pathlib import Path
import networkx as nx
from src.counterfactuals.feasibility_check import build_graph_from_subgraph


def load_results(results_dir: str) -> list[dict]:
    results = []
    for path in Path(results_dir).glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["_filename"] = path.name
            results.append(data)
    return results


def classify_failure_reason(r: dict, max_llm_calls: int = 200, max_cost: float = 10.0) -> str:
    if r.get("llm_calls", 0) >= max_llm_calls:
        return "llm_budget_exceeded"
    if r.get("cost", 0.0) >= max_cost:
        return "cost_budget_exceeded"
    return "unknown"


def get_op_type(op: list) -> str:
    """Extract operation type from an operation list like ['delete_node', 'X'] or ['replace_edge', [...]]."""
    if isinstance(op, list) and len(op) >= 1:
        return op[0]
    return "unknown"


def evaluate_success_rate(results: list[dict]):
    total = len(results)
    successful = [r for r in results if r.get("found") and len(r.get("operations", [])) > 0]
    failed = [r for r in results if not r.get("found") or len(r.get("operations", [])) == 0]

    print("=" * 50)
    print("SUCCESS RATE")
    print("=" * 50)
    print(f"  Total cases:       {total}")
    print(f"  Found:             {len(successful)} ({100 * len(successful) / total:.1f}%)")
    print(f"  Not found:         {len(failed)} ({100 * len(failed) / total:.1f}%)")

    if failed:
        reasons = Counter(classify_failure_reason(r) for r in failed)
        print(f"\n  Failure reasons:")
        for reason, count in reasons.most_common():
            label = {
                "llm_budget_exceeded": f"LLM call budget exceeded (>= 200 calls)",
                "cost_budget_exceeded": f"Cost budget exceeded (>= 10.0)",
                "unknown": "Unknown / terminated early",
            }.get(reason, reason)
            print(f"    {label:<45} {count}x")


def evaluate_operation_stats(results: list[dict]):
    successful = [r for r in results if len(r.get("operations", [])) > 0]

    if not successful:
        print("\nNo successful counterfactuals to compute operation stats.")
        return

    num_ops = [r["num_operations"] for r in successful]
    costs = [r["cost"] for r in successful if "cost" in r]

    # Count op types and most-affected nodes/edges
    op_type_counter = Counter()
    entity_counter = Counter()

    for r in successful:
        for op in r["operations"]:
            op_type = get_op_type(op)
            op_type_counter[op_type] += 1

            if op_type == "delete_node":
                # op = ["delete_node", "NodeName"]
                entity_counter[op[1]] += 1
            elif op_type == "delete_edge":
                # op = ["delete_edge", ["u", "v"]]
                entity_counter[tuple(op[1])] += 1
            elif op_type == "replace_node":
                # op = ["replace_node", ["old", "new"]]
                entity_counter[op[1][0]] += 1
            elif op_type == "replace_edge":
                # op = ["replace_edge", [["u", "v"], ["u2", "v2"]]]
                entity_counter[tuple(op[1][0])] += 1

    print("\n" + "=" * 50)
    print("OPERATION STATS")
    print("=" * 50)
    print(f"  Avg operations:    {sum(num_ops) / len(num_ops):.2f}")
    print(f"  Min operations:    {min(num_ops)}")
    print(f"  Max operations:    {max(num_ops)}")

    if costs:
        print(f"\n  Edit distance (cost):")
        print(f"    Avg:             {sum(costs) / len(costs):.4f}")
        print(f"    Min:             {min(costs):.4f}")
        print(f"    Max:             {max(costs):.4f}")

    print(f"\n  Operation type breakdown:")
    for op_type, count in op_type_counter.most_common():
        print(f"    {op_type:<30} {count}x")

    print(f"\n  Top 10 most affected nodes/edges:")
    for entity, count in entity_counter.most_common(10):
        label = str(entity) if isinstance(entity, tuple) else entity
        print(f"    {label:<40} {count}x")


def evaluate_answer_similarity(results: list[dict]):
    successful = [r for r in results if len(r.get("operations", [])) > 0]

    if not successful:
        print("\nNo successful counterfactuals to compute similarity stats.")
        return

    similarities = [r["answers"]["similarity"] for r in successful if "similarity" in r["answers"]]
    if not similarities:
        return

    avg = sum(similarities) / len(similarities)
    sorted_sims = sorted(similarities)
    median = sorted_sims[len(sorted_sims) // 2]

    buckets = {"0.0–0.2": 0, "0.2–0.4": 0, "0.4–0.6": 0, "0.6–0.8": 0, "0.8–1.0": 0}
    for s in similarities:
        if s < 0.2:
            buckets["0.0–0.2"] += 1
        elif s < 0.4:
            buckets["0.2–0.4"] += 1
        elif s < 0.6:
            buckets["0.4–0.6"] += 1
        elif s < 0.8:
            buckets["0.6–0.8"] += 1
        else:
            buckets["0.8–1.0"] += 1

    print("\n" + "=" * 50)
    print("ANSWER SIMILARITY (original vs perturbed)")
    print("=" * 50)
    print(f"  Avg:               {avg:.4f}")
    print(f"  Median:            {median:.4f}")
    print(f"  Min:               {min(similarities):.4f}")
    print(f"  Max:               {max(similarities):.4f}")
    print(f"\n  Distribution:")
    for bucket, count in buckets.items():
        bar = "█" * count
        print(f"    {bucket}  {bar} {count}")


def evaluate_llm_calls(results: list[dict]):
    successful = [r for r in results if len(r.get("operations", [])) > 0]
    unsuccessful = [r for r in results if len(r.get("operations", [])) == 0]

    def print_stats(label, subset):
        calls = [r["llm_calls"] for r in subset if "llm_calls" in r]
        if not calls:
            print(f"\n  No data for {label}.")
            return
        print(f"\n  {label} ({len(calls)} cases):")
        print(f"    Avg:             {sum(calls) / len(calls):.2f}")
        print(f"    Min:             {min(calls)}")
        print(f"    Max:             {max(calls)}")
        print(f"    Total:           {sum(calls)}")

    print("\n" + "=" * 50)
    print("LLM CALLS")
    print("=" * 50)
    print_stats("Successful", successful)
    print_stats("Unsuccessful", unsuccessful)


def evaluate_graph_size(results: list[dict]):
    node_counts, edge_counts = [], []
    for r in results:
        subgraph = r.get("original_subgraph", {})
        node_counts.append(len(subgraph.get("entities", [])))
        edge_counts.append(len(subgraph.get("relations", [])))

    if not node_counts:
        return

    print("\n" + "=" * 50)
    print("GRAPH SIZE (original subgraph)")
    print("=" * 50)
    print(f"  Nodes — Avg: {sum(node_counts) / len(node_counts):.2f}  Min: {min(node_counts)}  Max: {max(node_counts)}")
    print(f"  Edges — Avg: {sum(edge_counts) / len(edge_counts):.2f}  Min: {min(edge_counts)}  Max: {max(edge_counts)}")


def is_star(G: nx.DiGraph) -> tuple[bool, str]:
    U = G.to_undirected()
    degrees = dict(U.degree())

    if not degrees:
        return False, "empty graph"

    hubs = [n for n, d in degrees.items() if d > 1]
    leaves = [n for n, d in degrees.items() if d == 1]

    if len(hubs) == 1 and len(leaves) == len(degrees) - 1:
        return True, f"single star (hub: '{hubs[0]}', leaves: {len(leaves)})"

    components = list(nx.connected_components(U))
    if len(components) > 1:
        all_stars = True
        component_descs = []
        for comp in components:
            subgraph = U.subgraph(comp)
            sub_degrees = dict(subgraph.degree())
            sub_hubs = [n for n, d in sub_degrees.items() if d > 1]
            sub_leaves = [n for n, d in sub_degrees.items() if d == 1]

            if len(comp) == 1:
                component_descs.append(f"isolated node ('{list(comp)[0]}')")
            elif len(sub_hubs) == 1 and len(sub_leaves) == len(comp) - 1:
                component_descs.append(f"star (hub: '{sub_hubs[0]}', leaves: {len(sub_leaves)})")
            else:
                all_stars = False
                break

        if all_stars:
            return True, f"star forest ({len(components)} components: {', '.join(component_descs)})"

    return False, "not a star structure"


def evaluate_star_structure(results: list[dict]):
    star_cases, non_star_cases = [], []

    for r in results:
        subgraph = r.get("original_subgraph")
        if not subgraph:
            continue

        G = build_graph_from_subgraph(subgraph)
        is_star_graph, desc = is_star(G)

        entry = {
            "question": r["question"],
            "found": r.get("found", False),
            "description": desc,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
        }

        (star_cases if is_star_graph else non_star_cases).append(entry)

    print("\n" + "=" * 50)
    print("STAR STRUCTURE ANALYSIS")
    print("=" * 50)
    print(f"  Total:             {len(star_cases) + len(non_star_cases)}")
    print(f"  Star-like:         {len(star_cases)}")
    print(f"  Non-star:          {len(non_star_cases)}")

    return star_cases, non_star_cases


def evaluate_jaccard_similarity(results1: list[dict], results2: list[dict]):
    def successful_questions(results):
        return {r["question"] for r in results if len(r.get("operations", [])) > 0}

    successful1 = successful_questions(results1)
    successful2 = successful_questions(results2)
    intersection = successful1 & successful2
    union = successful1 | successful2
    jaccard = len(intersection) / len(union) if union else 0.0

    print("\n" + "=" * 50)
    print("JACCARD SIMILARITY (successful cases)")
    print("=" * 50)
    print(f"  Successful in dir1:        {len(successful1)}")
    print(f"  Successful in dir2:        {len(successful2)}")
    print(f"  Common successful:         {len(intersection)}")
    print(f"  Union:                     {len(union)}")
    print(f"  Jaccard similarity:        {jaccard:.4f}")

    only1 = successful1 - successful2
    only2 = successful2 - successful1
    if only1:
        print(f"\n  Only successful in dir1 ({len(only1)}):")
        for q in sorted(only1):
            print(f"    - {q}")
    if only2:
        print(f"\n  Only successful in dir2 ({len(only2)}):")
        for q in sorted(only2):
            print(f"    - {q}")


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def evaluate_perturbation_impact(results: list[dict], output_path: str = "perturbation_impact.png"):
    successful = [r for r in results if r.get("found") and r.get("operations")]

    if not successful:
        print("\nNo successful counterfactuals to compute perturbation impact.")
        return

    total_pct_modified = []

    for r in successful:
        orig = r.get("original_subgraph", {})
        orig_nodes = {e["name"] for e in orig.get("entities", [])}
        orig_edges = [(e["src"], e["tgt"]) for e in orig.get("relations", [])]

        changed_nodes = set()
        changed_edges = set()

        for op in r.get("operations", []):
            op_type = get_op_type(op)

            if op_type == "delete_node":
                node = op[1]
                changed_nodes.add(node)
                for src, tgt in orig_edges:
                    if src == node or tgt == node:
                        changed_edges.add((src, tgt))
            elif op_type == "replace_node":
                node = op[1][0]
                changed_nodes.add(node)
                for src, tgt in orig_edges:
                    if src == node or tgt == node:
                        changed_edges.add((src, tgt))
            elif op_type == "delete_edge":
                changed_edges.add(tuple(op[1]))
            elif op_type == "replace_edge":
                changed_edges.add(tuple(op[1][0]))

        total = len(orig_nodes) + len(orig_edges)
        if total > 0:
            total_pct_modified.append(
                100.0 * (len(changed_nodes) + len(changed_edges)) / total
            )

    avg = sum(total_pct_modified) / len(total_pct_modified)
    mn  = min(total_pct_modified)
    mx  = max(total_pct_modified)

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle("Perturbation impact on graph", fontsize=14, fontweight="bold")

    bins = np.linspace(0, 100, 21)
    ax.hist(total_pct_modified, bins=bins, color="#4C6EF5", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(avg, color="black", linewidth=1.2, linestyle="--", label=f"avg {avg:.1f}%")

    ax.set_xlabel("% of KG modified (nodes + edges)")
    ax.set_ylabel("# cases")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.set_xlim(0, 100)
    ax.legend(fontsize=9)
    ax.text(
        0.98, 0.95,
        f"min {mn:.1f}%\nmax {mx:.1f}%\nn={len(total_pct_modified)}",
        transform=ax.transAxes,
        fontsize=8,
        va="top", ha="right",
        color="gray",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved perturbation impact plot to '{output_path}'")


def evaluate_operations_vs_graph_size(results: list[dict], output_path: str = "operations_vs_graph_size.png"):
    successful = [r for r in results if r.get("found") and r.get("operations")]

    if not successful:
        print("\nNo successful counterfactuals to compute operations vs graph size.")
        return

    graph_sizes = []
    num_ops     = []

    for r in successful:
        orig  = r.get("original_subgraph", {})
        nodes = len(orig.get("entities", []))
        edges = len(orig.get("relations", []))
        graph_sizes.append(nodes + edges)
        num_ops.append(len(r["operations"]))

    graph_sizes = np.array(graph_sizes)
    num_ops     = np.array(num_ops)

    # bin graph sizes into equal-width buckets
    n_bins    = 10
    bin_edges = np.linspace(graph_sizes.min(), graph_sizes.max() + 1, n_bins + 1)
    bin_labels, mean_ops, counts = [], [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (graph_sizes >= lo) & (graph_sizes < hi)
        if mask.sum() == 0:
            continue
        bin_labels.append(f"{int(lo)}–{int(hi)}")
        mean_ops.append(num_ops[mask].mean())
        counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Mean operations by graph size", fontsize=14, fontweight="bold")

    x    = np.arange(len(bin_labels))
    bars = ax.bar(x, mean_ops, color="#4C6EF5", alpha=0.85, edgecolor="white", linewidth=0.5)

    # annotate each bar with case count
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"n={count}",
            ha="center", va="bottom",
            fontsize=8, color="gray",
        )

    overall_mean = num_ops.mean()
    ax.axhline(overall_mean, color="black", linewidth=1.2, linestyle="--",
               label=f"overall mean {overall_mean:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Graph size (nodes + edges)")
    ax.set_ylabel("Mean number of operations")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved operations vs graph size plot to '{output_path}'")

def evaluate_cut_vertex_edge(results: list[dict]):
    successful = [r for r in results if r.get("found") and r.get("operations")]

    if not successful:
        print("\nNo successful counterfactuals to compute cut vertex/edge analysis.")
        return

    total_ops       = 0
    cut_vertex_ops  = 0
    cut_edge_ops    = 0
    neither_ops     = 0

    # break down by op type
    op_type_totals   = Counter()
    op_type_critical = Counter()

    per_result_pct  = []

    for r in successful:
        orig       = r.get("original_subgraph", {})
        orig_nodes = [e["name"] for e in orig.get("entities", [])]
        orig_edges = [(e["src"], e["tgt"]) for e in orig.get("relations", [])]

        G = nx.Graph()
        G.add_nodes_from(orig_nodes)
        G.add_edges_from(orig_edges)

        cut_vertices = set(nx.articulation_points(G))
        cut_edges    = set(map(tuple, nx.bridges(G)))
        cut_edges   |= {(v, u) for u, v in cut_edges}

        result_critical = 0

        for op in r.get("operations", []):
            op_type = get_op_type(op)
            total_ops          += 1
            op_type_totals[op_type] += 1
            is_critical = False

            if op_type == "delete_node":
                node = op[1]
                if node in cut_vertices:
                    cut_vertex_ops  += 1
                    is_critical      = True
                else:
                    neither_ops += 1

            elif op_type == "replace_node":
                # check if the old node (op[1][0]) was a cut vertex
                node = op[1][0]
                if node in cut_vertices:
                    cut_vertex_ops  += 1
                    is_critical      = True
                else:
                    neither_ops += 1

            elif op_type == "delete_edge":
                edge = tuple(op[1])
                if edge in cut_edges:
                    cut_edge_ops += 1
                    is_critical   = True
                else:
                    neither_ops += 1

            elif op_type == "replace_edge":
                # check if the old edge (op[1][0]) was a bridge
                edge = tuple(op[1][0])
                if edge in cut_edges:
                    cut_edge_ops += 1
                    is_critical   = True
                else:
                    neither_ops += 1

            if is_critical:
                op_type_critical[op_type] += 1
                result_critical           += 1

        per_result_pct.append(100.0 * result_critical / len(r["operations"]))

    print("\n" + "=" * 50)
    print("CUT VERTEX / CUT EDGE ANALYSIS")
    print("=" * 50)
    print(f"  Total ops:               {total_ops}")
    print(f"\n  Targeted a cut vertex:   {cut_vertex_ops} ({100*cut_vertex_ops/total_ops:.1f}%)")
    print(f"  Targeted a cut edge:     {cut_edge_ops} ({100*cut_edge_ops/total_ops:.1f}%)")
    print(f"  Targeted neither:        {neither_ops} ({100*neither_ops/total_ops:.1f}%)")

    print(f"\n  Breakdown by op type:")
    for op_type in sorted(op_type_totals):
        total    = op_type_totals[op_type]
        critical = op_type_critical[op_type]
        pct      = 100.0 * critical / total
        node_or_edge = "cut vertex" if "node" in op_type else "cut edge"
        print(f"    {op_type:<20} {critical}/{total} hit a {node_or_edge} ({pct:.1f}%)")

    if per_result_pct:
        avg = sum(per_result_pct) / len(per_result_pct)
        mn  = min(per_result_pct)
        mx  = max(per_result_pct)
        print(f"\n  % of ops hitting a cut vertex/edge (per result):")
        print(f"    Avg:               {avg:.2f}%")
        print(f"    Min:               {mn:.2f}%")
        print(f"    Max:               {mx:.2f}%")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="src/counterfactuals/counterfactual_results")
    parser.add_argument("--dir2", type=str, default=None, help="Optional second directory for Jaccard comparison")
    args = parser.parse_args()

    results = load_results(args.dir)
    if not results:
        print(f"No JSON files found in '{args.dir}'")
        return

    print(f"\nLoaded {len(results)} result files from '{args.dir}'")

    evaluate_success_rate(results)
    evaluate_graph_size(results)
    evaluate_operation_stats(results)
    evaluate_cut_vertex_edge(results)
    evaluate_perturbation_impact(results)
    evaluate_operations_vs_graph_size(results)
    evaluate_answer_similarity(results)
    evaluate_llm_calls(results)
    evaluate_star_structure(results)

    if args.dir2:
        results2 = load_results(args.dir2)
        if results2:
            print(f"\nLoaded {len(results2)} result files from '{args.dir2}'")
            evaluate_jaccard_similarity(results, results2)
        else:
            print(f"No JSON files found in '{args.dir2}'")


if __name__ == "__main__":
    main()