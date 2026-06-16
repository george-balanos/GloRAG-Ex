"""
ranking.py
==========
Shared ranking and stability utilities used by both robustness pipelines
(evaluation.py and evaluate_robustness.py).

Provides:
  - Sorted ranking extraction from raw attribution dicts
  - Kendall τ computation on the overlapping subset of two rankings
  - Trapezoidal AUC over (noise_level, tau) pairs
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau


# ─────────────────────────────────────────────────────────────
# Ranking extraction
# ─────────────────────────────────────────────────────────────

def edge_rank_list(edge_attributions: list[dict]) -> list[tuple[str, str]]:
    """
    Sort edges by |attribution| descending and return (source, target) pairs.

    Note: attribution values are intentionally dropped from the identity key
    so that the same edge is recognised as the same item across noise levels
    when computing set overlap for Kendall τ.
    """
    sorted_edges = sorted(
        edge_attributions,
        key=lambda e: abs(e["attribution"]),
        reverse=True,
    )
    return [(e["source"], e["target"]) for e in sorted_edges]


def node_rank_list(node_attributions: list[dict]) -> list[str]:
    """Sort nodes by |attribution| descending and return node name list."""
    sorted_nodes = sorted(
        node_attributions,
        key=lambda n: abs(n["attribution"]),
        reverse=True,
    )
    return [n["node"] for n in sorted_nodes]


# ─────────────────────────────────────────────────────────────
# Kendall τ
# ─────────────────────────────────────────────────────────────

def ranking_to_index(rank_list: list) -> dict:
    """Map each item to its 0-based rank position."""
    return {item: i for i, item in enumerate(rank_list)}


def kendall_tau_on_overlap(base_rank: list, noisy_rank: list) -> float | None:
    """
    Compute Kendall τ restricted to items present in both rankings.
    Returns None when fewer than 2 items are shared.
    """
    base_map  = ranking_to_index(base_rank)
    noisy_map = ranking_to_index(noisy_rank)
    common    = list(set(base_map.keys()) & set(noisy_map.keys()))

    if len(common) < 2:
        return None

    tau, _ = kendalltau(
        [base_map[x]  for x in common],
        [noisy_map[x] for x in common],
    )
    return float(tau)


# Convenience wrappers kept for call-site clarity
def kendall_tau_edges(base_edges: list, noisy_edges: list) -> float | None:
    return kendall_tau_on_overlap(base_edges, noisy_edges)


def kendall_tau_nodes(base_nodes: list, noisy_nodes: list) -> float | None:
    return kendall_tau_on_overlap(base_nodes, noisy_nodes)


# ─────────────────────────────────────────────────────────────
# AUC
# ─────────────────────────────────────────────────────────────

def robustness_auc_from_scores(scores: list[dict]) -> float:
    """
    Trapezoidal AUC over a list of {"noise": float, "tau": float | None} dicts.
    Used by evaluation.py.
    """
    valid = [(s["noise"], s["tau"]) for s in scores if s["tau"] is not None]
    if len(valid) < 2:
        return 0.0
    xs, ys = zip(*sorted(valid))
    return float(np.trapz(ys, xs))


def robustness_auc_from_summary(results: dict) -> float:
    """
    Trapezoidal AUC over a summary dict of the form:
        {"10%": {"mean_tau": float, ...}, "20%": {...}, ...}
    Used by evaluate_robustness.py.
    """
    pairs = sorted(
        [
            (int(k.rstrip("%")) / 100, v["mean_tau"])
            for k, v in results.items()
            if v.get("mean_tau") is not None
        ]
    )
    if len(pairs) < 2:
        return 0.0
    xs, ys = zip(*pairs)
    return float(np.trapezoid(ys, xs))
