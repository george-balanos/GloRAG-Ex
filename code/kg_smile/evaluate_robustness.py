#!/usr/bin/env python3
"""
evaluate_robustness.py
======================
Aggregate Kendall τ stability and noise infiltration analysis across all
questions in a robustness benchmark file.

Reads:  benchmark JSON produced by: python -m kg_smile.runner robustness
Writes: summary JSON + optional publication-quality PDF plots

Metrics computed:
  - Mean / std Kendall τ per noise level (nodes and edges)
  - Trapezoidal AUC over noise levels
  - Noise infiltration rate: fraction of questions where a noise-injected
    node/edge appears in the top-k of the noisy attribution ranking

Optional --plot flag produces:
  - robustness_combined.pdf
  - robustness_node_tau.pdf
  - robustness_edge_tau.pdf
  - robustness_noise_infiltration.pdf
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .ranking import (
    edge_rank_list,
    node_rank_list,
    kendall_tau_on_overlap,
    robustness_auc_from_summary,
)


# ─────────────────────────────────────────────────────────────
# Noise infiltration helpers
# ─────────────────────────────────────────────────────────────

def noise_nodes_in_topk(base_rank: list, noisy_rank: list, k: int) -> bool:
    """True if any node absent from the baseline appears in the noisy top-k."""
    base_set  = set(base_rank)
    new_nodes = [n for n in noisy_rank if n not in base_set]
    return bool(set(new_nodes) & set(noisy_rank[:k]))


def noise_edges_in_topk(base_rank: list, noisy_rank: list, k: int) -> bool:
    """True if any edge absent from the baseline appears in the noisy top-k."""
    base_set  = set(base_rank)
    new_edges = [e for e in noisy_rank if e not in base_set]
    return bool(set(new_edges) & set(noisy_rank[:k]))


# ─────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_noise_robustness(
    benchmark_path: str,
    top_ks: list[int] = (3, 5),
) -> tuple[dict, dict, dict, dict, dict, dict]:
    """
    Aggregate robustness metrics across all questions.

    Returns:
        node_results      — {noise_label: {mean_tau, std_tau, n}}
        edge_results      — same shape
        node_taus_raw     — {noise_pct_float: [tau, ...]}  (for plotting)
        edge_taus_raw     — same shape
        infiltration      — {k: {"node": {label: rate}, "edge": {label: rate}}}
        infiltration_raw  — {k: {"node": {noise_pct: [bool]}, "edge": ...}}
    """
    with open(benchmark_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Support both dict-of-dicts {"0": {...}, ...} and legacy list format
    data = list(raw.values()) if isinstance(raw, dict) else raw

    node_taus: dict[float, list[float]] = defaultdict(list)
    edge_taus: dict[float, list[float]] = defaultdict(list)

    infil: dict = {
        k: {"node": defaultdict(list), "edge": defaultdict(list)}
        for k in top_ks
    }

    skipped_no_baseline = 0

    for item in data:
        if "error" in item or "benchmark" not in item:
            continue

        runs          = sorted(item["benchmark"], key=lambda r: r["noise_pct"])
        baseline_runs = [r for r in runs if r["noise_pct"] == 0.0]

        if not baseline_runs:
            skipped_no_baseline += 1
            continue

        base_result = baseline_runs[0]["result"]
        base_nodes  = node_rank_list(base_result["node_attributions"])
        base_edges  = edge_rank_list(base_result["edge_attributions"])

        for run in runs:
            if run["noise_pct"] == 0.0:
                continue

            noisy_result = run["result"]
            noise_pct    = run["noise_pct"]
            noisy_nodes  = node_rank_list(noisy_result["node_attributions"])
            noisy_edges  = edge_rank_list(noisy_result["edge_attributions"])

            tau_nodes = kendall_tau_on_overlap(base_nodes, noisy_nodes)
            if tau_nodes is not None:
                node_taus[noise_pct].append(tau_nodes)

            tau_edges = kendall_tau_on_overlap(base_edges, noisy_edges)
            if tau_edges is not None:
                edge_taus[noise_pct].append(tau_edges)

            for k in top_ks:
                infil[k]["node"][noise_pct].append(noise_nodes_in_topk(base_nodes, noisy_nodes, k))
                infil[k]["edge"][noise_pct].append(noise_edges_in_topk(base_edges, noisy_edges, k))

    if skipped_no_baseline:
        print(f"[evaluate_robustness] Skipped {skipped_no_baseline} items with no noise=0 baseline")

    def summarise_tau(taus_by_level: dict) -> dict:
        return {
            f"{int(pct * 100)}%": {
                "mean_tau": float(np.mean(v)) if v else None,
                "std_tau":  float(np.std(v))  if v else None,
                "n":        len(v),
            }
            for pct, v in sorted(taus_by_level.items())
        }

    def summarise_infiltration(infil_raw: dict) -> dict:
        return {
            f"{int(pct * 100)}%": float(np.mean(flags)) if flags else 0.0
            for pct, flags in sorted(infil_raw.items())
        }

    infiltration_summary = {
        k: {
            "node": summarise_infiltration(infil[k]["node"]),
            "edge": summarise_infiltration(infil[k]["edge"]),
        }
        for k in top_ks
    }

    return (
        summarise_tau(node_taus),
        summarise_tau(edge_taus),
        dict(node_taus),
        dict(edge_taus),
        infiltration_summary,
        {k: {"node": dict(infil[k]["node"]), "edge": dict(infil[k]["edge"])} for k in top_ks},
    )


# ─────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────

def print_report(
    node_results: dict,
    edge_results: dict,
    infiltration: dict,
) -> None:
    print("\n========== NODE KENDALL TAU ==========")
    for noise_level, stats in node_results.items():
        print(f"  Noise {noise_level:>4} -> tau = {stats['mean_tau']:.4f} "
              f"+/- {stats['std_tau']:.4f}  (n={stats['n']})")
    print(f"  AUC = {robustness_auc_from_summary(node_results):.4f}")

    print("\n========== EDGE KENDALL TAU ==========")
    for noise_level, stats in edge_results.items():
        print(f"  Noise {noise_level:>4} -> tau = {stats['mean_tau']:.4f} "
              f"+/- {stats['std_tau']:.4f}  (n={stats['n']})")
    print(f"  AUC = {robustness_auc_from_summary(edge_results):.4f}")

    print("\n========== NOISE RESISTANCE (fraction noise-free in top-k) ==========")
    for k, summary in infiltration.items():
        print(f"\n  Top-{k}:")
        print(f"  {'Noise':>6}  {'Node resist':>12}  {'Edge resist':>12}")
        for nl in sorted(summary["node"], key=lambda x: int(x.rstrip("%"))):
            nr = summary["node"].get(nl, 0.0)
            er = summary["edge"].get(nl, 0.0)
            print(f"  {nl:>6}  {1-nr:>11.1%}  {1-er:>11.1%}")


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def _setup_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          10,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })


def _plot_tau_panel(ax, taus_raw: dict, color: str, label: str) -> None:
    noise_levels = sorted(taus_raw.keys())
    xs    = [int(n * 100) for n in noise_levels]
    means = [float(np.mean(taus_raw[n])) for n in noise_levels]
    stds  = [float(np.std(taus_raw[n]))  for n in noise_levels]

    ax.plot(xs, means, marker="o", markersize=5, linewidth=1.5, color=color, label=label)
    ax.fill_between(
        xs,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.15, color=color,
    )
    for x, m in zip(xs, means):
        ax.annotate(f"{m:.2f}", (x, m),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.5, color=color)


def _style_tau_ax(ax, taus_raw: dict, ylabel: str) -> None:
    ax.set_xlabel("Noise Level (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(int(n * 100) for n in taus_raw))
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="grey", linewidth=0.7, linestyle=":")
    ax.legend(frameon=False)


def save_plots(
    node_taus_raw:    dict,
    edge_taus_raw:    dict,
    infiltration_raw: dict,
    out_dir:          str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    NODE_COLOR = "#2980B9"
    EDGE_COLOR = "#E74C3C"

    # Combined two-panel τ plot
    fig, (ax_node, ax_edge) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, taus_raw, color, label in [
        (ax_node, node_taus_raw, NODE_COLOR, "Node Kendall τ"),
        (ax_edge, edge_taus_raw, EDGE_COLOR, "Edge Kendall τ"),
    ]:
        _plot_tau_panel(ax, taus_raw, color, label)
        _style_tau_ax(ax, taus_raw, "Kendall τ")
    fig.tight_layout()
    p = out / "robustness_combined.pdf"
    fig.savefig(p, format="pdf")
    plt.close(fig)
    print(f"  ✓  {p}")

    # Individual τ panels
    for taus_raw, color, label, fname in [
        (node_taus_raw, NODE_COLOR, "Node Kendall τ", "robustness_node_tau.pdf"),
        (edge_taus_raw, EDGE_COLOR, "Edge Kendall τ", "robustness_edge_tau.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        _plot_tau_panel(ax, taus_raw, color, label)
        _style_tau_ax(ax, taus_raw, "Kendall τ")
        fig.tight_layout()
        p = out / fname
        fig.savefig(p, format="pdf")
        plt.close(fig)
        print(f"  ✓  {p}")

    # Noise resistance panel
    top_ks = sorted(infiltration_raw.keys())
    INFIL_COLORS = {"node": NODE_COLOR, "edge": EDGE_COLOR}

    fig, axes = plt.subplots(1, len(top_ks), figsize=(3.5 * len(top_ks), 3.2), sharey=True)
    if len(top_ks) == 1:
        axes = [axes]

    for i, (ax, k) in enumerate(zip(axes, top_ks)):
        for entity_type, color in INFIL_COLORS.items():
            raw          = infiltration_raw[k][entity_type]
            noise_levels = sorted(raw.keys())
            xs    = [int(n * 100) for n in noise_levels]
            rates = [(1.0 - float(np.mean(raw[n]))) * 100 for n in noise_levels]

            ax.plot(xs, rates, marker="o", markersize=5, linewidth=1.5,
                    color=color, label=f"{entity_type.capitalize()} (top-{k})")
            for x, r in zip(xs, rates):
                ax.annotate(f"{r:.0f}%", (x, r),
                            textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=7.5, color=color)

        ax.set_xlabel("Noise Level (%)")
        ax.set_ylabel("Noise-resistant questions (%)" if i == 0 else "")
        if i > 0:
            ax.tick_params(labelleft=False)
        ax.set_xticks(xs)
        ax.set_ylim(-5, 105)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.legend(frameon=False)

    fig.tight_layout()
    p = out / "robustness_noise_infiltration.pdf"
    fig.savefig(p, format="pdf")
    plt.close(fig)
    print(f"  ✓  {p}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KG-SMILE aggregate robustness evaluation")
    parser.add_argument("--input",  default="results/robustness_results.json")
    parser.add_argument("--output", default=None,  help="Save summary as JSON")
    parser.add_argument("--plot",   default=None,  metavar="DIR",
                        help="Directory to save PDF plots")
    parser.add_argument("--top_k",  nargs="+", type=int, default=[3, 5],
                        help="Top-k cutoffs for noise infiltration (default: 3 5)")
    args = parser.parse_args()

    (node_res, edge_res,
     node_taus_raw, edge_taus_raw,
     infiltration, infiltration_raw) = evaluate_noise_robustness(args.input, top_ks=args.top_k)

    print_report(node_res, edge_res, infiltration)

    if args.plot:
        print(f"\nSaving plots to: {args.plot}")
        save_plots(node_taus_raw, edge_taus_raw, infiltration_raw, args.plot)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "node_results": node_res,
                "edge_results": edge_res,
                "node_auc":     robustness_auc_from_summary(node_res),
                "edge_auc":     robustness_auc_from_summary(edge_res),
                "infiltration": infiltration,
            }, f, indent=2)
        print(f"\n[OK] Saved -> {args.output}")