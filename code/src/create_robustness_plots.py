"""
Noise Resistance plotting script for robustness experiments.

Usage:
    python plot_robustness.py \
        --glorag /path/to/GloRAG-Ex/robustness \
        --shapley /path/to/Shapley/robustness \
        --ragex /path/to/RAG-Ex/robustness \
        --kgsmile_base /path/to/KG-SMILE/results \
        --kgsmile_robust /path/to/KG-SMILE/robustness/results \
        --k 3 \
        --output ./plots

Produces 4 PDF files (one per dataset), each with 2 subplots (ff, ft) side by side.
5 lines per subplot: GloRAG-Ex, Shapley, RAG-Ex (sent), RAG-Ex (para), KG-SMILE.

Filtering:
    Only questions where GloRAG-Ex has found == True AND noise_robust == True,
    matched across methods by question text.
"""

import os
import json
import argparse
import warnings
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams

# ── VLDB Style ───────────────────────────────────────────────────────────────

rcParams['font.family']      = 'serif'
rcParams['font.serif']       = ['Times New Roman', 'Times', 'DejaVu Serif']
rcParams['font.size']        = 14
rcParams['axes.titlesize']   = 15
rcParams['axes.labelsize']   = 17
rcParams['xtick.labelsize']  = 15
rcParams['ytick.labelsize']  = 15
rcParams['legend.fontsize']  = 12
rcParams['lines.linewidth']  = 2.2
rcParams['lines.markersize'] = 8
rcParams['pdf.fonttype']     = 42
rcParams['ps.fonttype']      = 42

# ── Constants ────────────────────────────────────────────────────────────────

DATASETS     = ["synthetic", "hotpotqa", "musique", "2wiki"]
CASES        = ["ff", "ft"]
NOISE_LEVELS = [10, 20, 30, 50]

METHOD_STYLES = {
    "GLoRAG-Ex":     {"color": "#1f77b4", "marker": "o", "linestyle": "-",  "zorder": 5},
    "TMC-Shapley-RAG":       {"color": "#d62728", "marker": "s", "linestyle": "--", "zorder": 4},
    "RAG-Ex (sentence)": {"color": "#2ca02c", "marker": "^", "linestyle": "-.", "zorder": 3},
    "RAG-Ex (paragraph)": {"color": "#ff7f0e", "marker": "D", "linestyle": ":",  "zorder": 3},
    "KG-SMILE":      {"color": "#9467bd", "marker": "P", "linestyle": "--", "zorder": 4},
}

DATASET_DISPLAY = {
    "synthetic": "Synthetic",
    "hotpotqa":  "HotpotQA",
    "musique":   "MuSiQue",
    "2wiki":     "2WikiMultiHopQA",
}

CASE_DISPLAY = {
    "ff": "(a) FF",
    "ft": "(b) FT",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_load_json(path):
    if not os.path.exists(path):
        warnings.warn(f"File not found: {path}")
        return None
    return load_json(path)


def noise_resistance_rate(resistant, total):
    if total == 0:
        return float("nan")
    return resistant / total


# ── GloRAG-Ex whitelist ───────────────────────────────────────────────────────

def build_glorag_whitelist(glorag_root, dataset, case, noise_level):
    """
    Returns a set of question strings where GloRAG-Ex has:
        found == True AND noise_robust == True
    """
    folder = os.path.join(
        glorag_root, dataset, "noise_resistance", case, f"noise_level_{noise_level}"
    )
    if not os.path.isdir(folder):
        warnings.warn(f"GloRAG-Ex folder not found: {folder}")
        return set()

    whitelist = set()
    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        data = safe_load_json(os.path.join(folder, fname))
        if data is None:
            continue
        if not data.get("noise", {}).get("noise_robust", False):
            continue
        if not data.get("found", False):
            continue
        question = data.get("question", "").strip()
        if question:
            whitelist.add(question)

    return whitelist


def build_glorag_whitelists(glorag_root):
    whitelists = defaultdict(lambda: defaultdict(dict))
    for dataset in DATASETS:
        for case in CASES:
            for nl in NOISE_LEVELS:
                whitelists[dataset][case][nl] = build_glorag_whitelist(
                    glorag_root, dataset, case, nl
                )
    return whitelists


# ── GloRAG-Ex ────────────────────────────────────────────────────────────────

def compute_glorag(glorag_root, dataset, case, noise_level, whitelist):
    """
    Noise Resistance for GloRAG-Ex:
      - Filter: question in whitelist (found==True AND noise_robust==True)
      - Noise entities: only from add_node ops
      - Resistant: none of those noise entities appear in 'operations'
    """
    folder = os.path.join(
        glorag_root, dataset, "noise_resistance", case, f"noise_level_{noise_level}"
    )
    if not os.path.isdir(folder):
        return float("nan")

    total, resistant = 0, 0
    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        data = safe_load_json(os.path.join(folder, fname))
        if data is None:
            continue

        question = data.get("question", "").strip()
        if question not in whitelist:
            continue

        total += 1

        # Collect noise node names ONLY from add_node ops
        noise_ops = data.get("noise", {}).get("ops", [])
        noise_entities = set()
        for op in noise_ops:
            if len(op) >= 2 and op[0] == "add_node" and isinstance(op[1], str):
                noise_entities.add(op[1])

        # Collect entities mentioned in explanation operations
        explanation_ops = data.get("operations", [])
        explanation_entities = set()
        for op in explanation_ops:
            if len(op) >= 2:
                if isinstance(op[1], str):
                    explanation_entities.add(op[1])
                elif isinstance(op[1], list):
                    explanation_entities.update(op[1])

        if noise_entities.isdisjoint(explanation_entities):
            resistant += 1

    return noise_resistance_rate(resistant, total)


def compute_glorag_all(glorag_root, whitelists):
    results = defaultdict(lambda: defaultdict(dict))
    for dataset in DATASETS:
        for case in CASES:
            for nl in NOISE_LEVELS:
                results[dataset][case][nl] = compute_glorag(
                    glorag_root, dataset, case, nl,
                    whitelists[dataset][case][nl]
                )
    return results


# ── Shapley ───────────────────────────────────────────────────────────────────

def compute_shapley_all(shapley_root, k, whitelists):
    results = defaultdict(lambda: defaultdict(dict))

    for dataset in DATASETS:
        fpath = os.path.join(shapley_root, f"{dataset}_shapley_noise.json")
        data = safe_load_json(fpath)
        if data is None:
            for case in CASES:
                for nl in NOISE_LEVELS:
                    results[dataset][case][nl] = float("nan")
            continue

        for nl in NOISE_LEVELS:
            nl_key = f"noise_level_{nl}"
            nl_data = data.get(nl_key, {})

            has_case_field = any(
                "mode" in q or "case" in q for q in nl_data.values()
            )

            for case in CASES:
                whitelist = whitelists[dataset][case][nl]
                total, resistant = 0, 0

                for qid, qdata in nl_data.items():
                    if not qdata.get("noise_robust", False):
                        continue
                    if has_case_field:
                        q_case = qdata.get("mode", qdata.get("case", None))
                        if q_case is not None and q_case != case:
                            continue

                    question = qdata.get("question", "").strip()
                    if question not in whitelist:
                        continue

                    total += 1
                    in_topk = (
                        qdata.get("metrics", {})
                             .get("topk", {})
                             .get(str(k), {})
                             .get("in_topk", True)
                    )
                    if not in_topk:
                        resistant += 1

                results[dataset][case][nl] = noise_resistance_rate(resistant, total)

    return results


# ── RAG-Ex ────────────────────────────────────────────────────────────────────

def compute_ragex_all(ragex_root, k, whitelists):
    results = {
        "sent": defaultdict(lambda: defaultdict(dict)),
        "para": defaultdict(lambda: defaultdict(dict)),
    }

    for gran in ["sent", "para"]:
        for dataset in DATASETS:
            fpath = os.path.join(
                ragex_root,
                f"robustness_rag_ex_{dataset}_{gran}.json"
            )
            data = safe_load_json(fpath)
            if data is None:
                for case in CASES:
                    for nl in NOISE_LEVELS:
                        results[gran][dataset][case][nl] = float("nan")
                continue

            for nl in NOISE_LEVELS:
                nl_key = f"noise_level_{nl}"
                nl_data = data.get(nl_key, {})

                has_case_field = any(
                    "mode" in q or "case" in q for q in nl_data.values()
                )

                for case in CASES:
                    whitelist = whitelists[dataset][case][nl]
                    total, resistant = 0, 0

                    for qid, qdata in nl_data.items():
                        if not qdata.get("noise_robust", False):
                            continue
                        if has_case_field:
                            q_case = qdata.get("mode", qdata.get("case", None))
                            if q_case is not None and q_case != case:
                                continue

                        question = qdata.get("question", "").strip()
                        if question not in whitelist:
                            continue

                        total += 1
                        in_topk = (
                            qdata.get("metrics", {})
                                 .get("topk", {})
                                 .get(str(k), {})
                                 .get("in_topk", True)
                        )
                        if not in_topk:
                            resistant += 1

                    results[gran][dataset][case][nl] = noise_resistance_rate(
                        resistant, total
                    )

    return results


# ── KG-SMILE ─────────────────────────────────────────────────────────────────

def compute_kgsmile_all(kgsmile_base_root, kgsmile_robust_root, k, whitelists):
    results = defaultdict(lambda: defaultdict(dict))

    for dataset in DATASETS:
        for case in CASES:
            base_path = os.path.join(
                kgsmile_base_root, f"kg_smile_{dataset}_{case}.json"
            )
            base_data = safe_load_json(base_path)

            robust_path = os.path.join(
                kgsmile_robust_root,
                f"robustness_results_{dataset}_{case}.json"
            )
            robust_data = safe_load_json(robust_path)

            if base_data is None or robust_data is None:
                for nl in NOISE_LEVELS:
                    results[dataset][case][nl] = float("nan")
                continue

            # Build base keys index by question text
            base_keys_by_question = {}
            for qid, qdata in base_data.items():
                question = qdata.get("question", "").strip()
                if not question:
                    continue
                scores = qdata.get("scores", {})
                if scores:
                    base_keys_by_question[question] = set(scores.keys())
                else:
                    keys = set()
                    for na in qdata.get("node_attributions", []):
                        keys.add(f"E::{na['node']}")
                    for ea in qdata.get("edge_attributions", []):
                        keys.add(f"R::{ea['source']}->{ea['target']}")
                    base_keys_by_question[question] = keys

            nl_counts = {nl: {"total": 0, "resistant": 0} for nl in NOISE_LEVELS}

            for qid, qdata in robust_data.items():
                question = qdata.get("question", "").strip()
                base_keys = base_keys_by_question.get(question, set())

                for entry in qdata.get("benchmark", []):
                    noise_pct = entry.get("noise_pct", None)
                    if noise_pct is None:
                        continue

                    nl = int(round(noise_pct * 100))
                    if nl not in NOISE_LEVELS:
                        continue

                    if not entry.get("noise_robust", False):
                        continue

                    whitelist = whitelists[dataset][case][nl]
                    if question not in whitelist:
                        continue

                    nl_counts[nl]["total"] += 1

                    result = entry.get("result", {})
                    noisy_scores = {}
                    scores = result.get("scores", {})
                    if scores:
                        noisy_scores = scores
                    else:
                        for na in result.get("node_attributions", []):
                            noisy_scores[f"E::{na['node']}"] = na["attribution"]
                        for ea in result.get("edge_attributions", []):
                            noisy_scores[f"R::{ea['source']}->{ea['target']}"] = ea["attribution"]

                    noise_keys = set(noisy_scores.keys()) - base_keys

                    if not noise_keys:
                        nl_counts[nl]["resistant"] += 1
                        continue

                    ranked = sorted(
                        noisy_scores.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )
                    top_k_keys = {key for key, _ in ranked[:k]}

                    if noise_keys.isdisjoint(top_k_keys):
                        nl_counts[nl]["resistant"] += 1

            for nl in NOISE_LEVELS:
                results[dataset][case][nl] = noise_resistance_rate(
                    nl_counts[nl]["resistant"], nl_counts[nl]["total"]
                )

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_all(glorag_results, shapley_results, ragex_results, kgsmile_results,
             output_dir, k):
    """
    Generate 4 PDF files, one per dataset.
    Each PDF has 2 subplots side by side: ff (left) and ft (right).
    Y-label only on the left subplot.

    When k == 2:
      - Y-axis label text is hidden (tick marks and values still shown)
      - Legend is suppressed
    When k != 2:
      - Y-axis label reads "Noise-free explanations (%)"
      - Legend is shown
    """
    os.makedirs(output_dir, exist_ok=True)

    x = NOISE_LEVELS
    show_ylabel  = (k != 2)
    show_legend  = False
    ylabel_text  = "Noise-free explanations (%)"

    for dataset in DATASETS:
        for case in CASES:
            fig, ax = plt.subplots(figsize=(5.0, 4.2))

            lines = [
                ("GloRAG-Ex",
                 [glorag_results[dataset][case].get(nl, float("nan")) for nl in x]),
                ("TMC-Shapley-RAG (KG)",
                 [shapley_results[dataset][case].get(nl, float("nan")) for nl in x]),
                ("RAG-Ex (sent)",
                 [ragex_results["sent"][dataset][case].get(nl, float("nan")) for nl in x]),
                ("RAG-Ex (para)",
                 [ragex_results["para"][dataset][case].get(nl, float("nan")) for nl in x]),
                ("KG-SMILE",
                 [kgsmile_results[dataset][case].get(nl, float("nan")) for nl in x]),
            ]

            for label, y in lines:
                style = METHOD_STYLES[label]
                y_pct = [v * 100 if not np.isnan(v) else float("nan") for v in y]
                ax.plot(
                    x, y_pct,
                    label=label,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=rcParams['lines.linewidth'],
                    markersize=rcParams['lines.markersize'],
                    zorder=style["zorder"],
                )

            ax.set_xlabel("Noise Level (%)")
            if show_ylabel:
                ax.set_ylabel(ylabel_text)
            else:
                ax.set_ylabel("")

            ax.set_xticks(x)
            ax.set_xlim(7, 53)
            ax.set_ylim(-5, 105)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
            ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if show_legend:
                ax.legend(loc="lower left", framealpha=0.9, edgecolor="gray")

            fname = f"noise_resistance_{dataset}_{case}_top{k}.pdf"
            fpath = os.path.join(output_dir, fname)
            fig.tight_layout()
            fig.savefig(fpath, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {fpath}")

def save_legend(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots()

    for label, style in METHOD_STYLES.items():
        ax.plot(
            [], [],
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=rcParams['lines.linewidth'],
            markersize=rcParams['lines.markersize'],
        )

    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles, labels,
        loc="center",
        ncol=len(METHOD_STYLES),
        framealpha=0.9,
        edgecolor="gray",
    )

    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.set_size_inches(bbox.width + 0.2, bbox.height + 0.2)
    ax.set_visible(False)

    fpath = os.path.join(output_dir, "legend.pdf")
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")

# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Noise Resistance across methods and datasets."
    )
    parser.add_argument(
        "--legend_only", action="store_true",
        help="Save a standalone legend PDF and exit"
    )
    parser.add_argument(
        "--glorag", required=True,
        help="Path to GloRAG-Ex robustness folder ({dataset}/noise_resistance/{case}/noise_level_{x}/)"
    )
    parser.add_argument(
        "--shapley", required=True,
        help="Path to Shapley folder containing {dataset}_shapley_noise.json files"
    )
    parser.add_argument(
        "--ragex", required=True,
        help="Path to RAG-Ex folder containing robustness_rag_ex_{dataset}_{sent|para}.json files"
    )
    parser.add_argument(
        "--kgsmile_base", required=True,
        help="Path to KG-SMILE base run folder (kg_smile_{dataset}_{case}.json)"
    )
    parser.add_argument(
        "--kgsmile_robust", required=True,
        help="Path to KG-SMILE robustness/results folder (robustness_results_{dataset}_{case}.json)"
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Top-k threshold for Noise Resistance (default: 3)"
    )
    parser.add_argument(
        "--output", default="./plots",
        help="Output directory for plots (default: ./plots)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.legend_only:
        save_legend(args.output)
        return

    print(f"\n{'='*60}")
    print(f"  Noise Resistance Plot  |  top-k = {args.k}")
    print(f"{'='*60}\n")

    print("► Building GloRAG-Ex whitelists (found=True & noise_robust=True)...")
    whitelists = build_glorag_whitelists(args.glorag)

    for dataset in DATASETS:
        for case in CASES:
            sizes = [len(whitelists[dataset][case][nl]) for nl in NOISE_LEVELS]
            print(f"  {dataset}/{case}: {dict(zip(NOISE_LEVELS, sizes))}")

    print("\n► Computing GloRAG-Ex results...")
    glorag_results = compute_glorag_all(args.glorag, whitelists)

    print("► Computing Shapley results...")
    shapley_results = compute_shapley_all(args.shapley, args.k, whitelists)

    print("► Computing RAG-Ex results (sent + para)...")
    ragex_results = compute_ragex_all(args.ragex, args.k, whitelists)

    print("► Computing KG-SMILE results...")
    kgsmile_results = compute_kgsmile_all(
        args.kgsmile_base, args.kgsmile_robust, args.k, whitelists
    )

    print(f"\n► Generating plots → {args.output}/\n")
    plot_all(
        glorag_results, shapley_results, ragex_results, kgsmile_results,
        args.output, args.k
    )

    print("\nDone! ✓")


if __name__ == "__main__":
    main()