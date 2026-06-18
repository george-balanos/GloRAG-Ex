"""
Global-explanation figures for the paper, from a folder of local counterfactual
JSONs (generate.py :: save_operations_to_json), stratified by run mode (ft vs ff).
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# --- Constants ---
NODE_FEATURES = [
    ("local_degree", "Local degree (Cᵢ)"),
    ("global_degree", "Global degree (KG)"),
    ("local_closeness", "Closeness (Cᵢ)"),
    ("pagerank", "PageRank (Cᵢ)"),
]

EDGE_FEATURES = [
    ("local_betweenness", "Edge betweenness (Cᵢ)"),
]

OP_TYPES = ["delete_node", "delete_edge", "add_node", "add_edge"]

# Updated to a more cohesive, modern qualitative palette
OP_COLORS = {
    "delete_node": "#E53E3E",  # Deep Red
    "delete_edge": "#FC8181",  # Soft Red
    "add_node": "#38A169",     # Deep Green
    "add_edge": "#68D391"      # Soft Green
}

MODE_LABEL = {
    "ft": "ft (breaking T→F)",
    "ff": "ff (corrective F→T)"
}

# --- Styling ---
def set_plot_style():
    """Apply modern, clean typography and styling to all plots."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.color": "#CBD5E0",
        "grid.linestyle": "--",
        "axes.axisbelow": True,  # Draw gridlines behind plot elements
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.dpi": 200,       # High-res by default
    })


# --- Data Loading & Processing ---
def load_explanations(results_root: str, dataset: Optional[str]) -> List[Dict[str, Any]]:
    search_pattern = os.path.join(results_root, "**", "counterfactual_*.json")
    files = sorted(glob.glob(search_pattern, recursive=True))
    
    out = []
    for fp in files:
        if dataset and f"/{dataset}/" not in fp.replace(os.sep, "/"):
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"  [skip] {fp}: {e}")
            continue
            
        if payload.get("found"):
            payload["_filepath"] = fp
            out.append(payload)
    return out

def dedupe_by_mode_question(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for payload in payloads:
        key = (payload.get("mode", ""), payload.get("question", ""))
        ts = payload.get("timestamp", "")
        if key not in best or ts > best[key].get("timestamp", ""):
            best[key] = payload
    return list(best.values())

def group_by_mode(payloads: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        groups[payload.get("mode", "?")].append(payload)
    return dict(groups)

# --- Graph & Feature Extraction ---
def assemble_context_graph(payload: Dict[str, Any]) -> nx.DiGraph:
    original_subgraph = payload.get("original_subgraph") or {}
    G = nx.DiGraph()
    for entity in original_subgraph.get("entities") or []:
        G.add_node(entity.get("name", ""), entity_type=entity.get("type", ""))
    for relation in original_subgraph.get("relations") or []:
        src, tgt = relation.get("src", ""), relation.get("tgt", "")
        if src and tgt:
            G.add_edge(src, tgt)
    return G

def extract_modified(payload: Dict[str, Any]) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    nodes, edges = set(), set()
    for op in payload.get("operations") or []:
        kind = op[0]
        if kind in ("delete_node", "add_node"):
            nodes.add(op[1])
        elif kind in ("delete_edge", "add_edge"):
            edges.add((op[1][0], op[1][1]))
    return nodes, edges

def node_type_map(payload: Dict[str, Any]) -> Dict[str, str]:
    original_subgraph = payload.get("original_subgraph") or {}
    return {
        e.get("name", ""): (e.get("type", "") or "untyped") 
        for e in (original_subgraph.get("entities") or [])
    }

def collect_features(
    payloads: List[Dict[str, Any]], 
    kg_degree: Optional[Dict[str, Any]]
) -> Dict[str, Dict[str, List[float]]]:
    acc = {k: {"mod": [], "bg": []} for k, _ in NODE_FEATURES + EDGE_FEATURES}
    for payload in payloads:
        G = assemble_context_graph(payload)
        if G.number_of_nodes() == 0:
            continue
            
        mod_nodes, mod_edges = extract_modified(payload)
        deg = dict(G.degree())
        clo = nx.closeness_centrality(G)
        
        try:
            pr = nx.pagerank(G)
        except Exception:
            pr = {n: float("nan") for n in G.nodes}
            
        per_node = {
            "local_degree": deg,
            "local_closeness": clo,
            "pagerank": pr,
        }
        
        if kg_degree is not None:
            per_node["global_degree"] = {n: kg_degree.get(n) for n in G.nodes}

        for feat, table in per_node.items():
            for n, val in table.items():
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                group_key = "mod" if n in mod_nodes else "bg"
                acc[feat][group_key].append(val)

        if G.number_of_edges() > 0:
            try:
                ebc = nx.edge_betweenness_centrality(G)
            except Exception:
                ebc = {}
            for e, val in ebc.items():
                group_key = "mod" if e in mod_edges else "bg"
                acc["local_betweenness"][group_key].append(val)
    return acc

# --- Statistical & Plotting Functions ---
def auc_ks(mod: List[float], bg: List[float]) -> Tuple[Optional[float], Optional[float]]:
    mod_arr = np.asarray(mod, float)
    bg_arr = np.asarray(bg, float)
    if len(mod_arr) == 0 or len(bg_arr) == 0:
        return None, None
        
    bg_sorted = np.sort(bg_arr)
    less = np.searchsorted(bg_sorted, mod_arr, side="left")
    lesseq = np.searchsorted(bg_sorted, mod_arr, side="right")
    auc = (less.sum() + 0.5 * (lesseq - less).sum()) / (len(mod_arr) * len(bg_arr))
    
    grid = np.sort(np.concatenate([mod_arr, bg_arr]))
    cdf_m = np.searchsorted(np.sort(mod_arr), grid, side="right") / len(mod_arr)
    cdf_b = np.searchsorted(bg_sorted, grid, side="right") / len(bg_arr)
    ks = float(np.max(np.abs(cdf_m - cdf_b)))
    return float(auc), ks


def plot_feature_separation(acc: Dict[str, Dict[str, List[float]]], mode: str, out_dir: str):
    feats = [(k, lbl) for k, lbl in NODE_FEATURES + EDGE_FEATURES if acc[k]["mod"] and acc[k]["bg"]]
    if not feats:
        print(f"[F1/{mode}] no feature data; skipping.")
        return

    # --- (a) Overlaid distributions ---
    ncol = min(3, len(feats))
    nrow = (len(feats) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.8 * nrow), squeeze=False)
    
    for ax in axes.flat:
        ax.axis("off")
        
    for ax, (key, lbl) in zip(axes.flat, feats):
        ax.axis("on")
        mod, bg = acc[key]["mod"], acc[key]["bg"]
        lo, hi = min(min(mod), min(bg)), max(max(mod), max(bg))
        bins = np.linspace(lo, hi, 20) if hi > lo else 10
        
        # Added edge colors and refined opacities for a cleaner look
        ax.hist(bg, bins=bins, density=True, color="#A0AEC0", alpha=0.5, edgecolor="white", linewidth=1.2, label="rest of context")
        ax.hist(mod, bins=bins, density=True, color="#3182CE", alpha=0.75, edgecolor="white", linewidth=1.2, label="flip-driving")
        
        auc, ks = auc_ks(mod, bg)
        ax.set_title(f"{lbl}\nAUC={auc:.2f}  |  KS={ks:.2f}", fontsize=10, pad=10)
        ax.set_ylabel("Density")
        
    axes.flat[0].legend(loc="best", fontsize=9, frameon=True, facecolor="white", edgecolor="#E2E8F0")
    fig.suptitle(f"Flip-driving vs. rest of context — {MODE_LABEL.get(mode, mode)}", y=1.02)
    fig.tight_layout()
    
    p1 = os.path.join(out_dir, f"feature_separation_{mode}.png")
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)

    # --- (b) Importance bar ---
    rows = []
    for key, lbl in feats:
        auc, ks = auc_ks(acc[key]["mod"], acc[key]["bg"])
        if auc is not None:
            rows.append((lbl, auc, ks, len(acc[key]["mod"])))
            
    rows.sort(key=lambda r: abs(r[1] - 0.5))
    
    fig, ax = plt.subplots(figsize=(7, 0.7 * len(rows) + 1.5))
    ylab = [r[0] for r in rows]
    aucs = [r[1] for r in rows]
    
    # Modernized bar colors
    colors = ["#3182CE" if a >= 0.5 else "#DD6B20" for a in aucs]
    
    bars = ax.barh(ylab, [a - 0.5 for a in aucs], left=0.5, color=colors, edgecolor="white", linewidth=1.5, height=0.6)
    ax.axvline(0.5, color="#718096", linestyle="--", linewidth=1.2, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC: P(flip-driving > rest)  —  0.5 = no separation")
    
    for bar, r in zip(bars, rows):
        val_auc, val_ks, n_samples = r[1], r[2], r[3]
        x_offset = 0.015 if val_auc >= 0.5 else -0.015
        ha = "left" if val_auc >= 0.5 else "right"
        
        ax.text(
            val_auc + x_offset, 
            bar.get_y() + bar.get_height() / 2,
            f"KS={val_ks:.2f} (n={n_samples})", 
            va="center", ha=ha, fontsize=9, color="#2D3748"
        )
        
    ax.set_title(f"Feature importance — {MODE_LABEL.get(mode, mode)}", y=1.05)
    fig.tight_layout()
    
    p2 = os.path.join(out_dir, f"feature_importance_{mode}.png")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)


def plot_op_cost_size(groups: Dict[str, List[Dict[str, Any]]], out_dir: str):
    modes = sorted(groups)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Operation-type composition ---
    ax = axes[0]
    comp = {m: Counter() for m in modes}
    
    for m in modes:
        for payload in groups[m]:
            for op in payload.get("operations") or []:
                if op[0] in OP_TYPES:
                    comp[m][op[0]] += 1
                    
    bottoms = np.zeros(len(modes))
    for op in OP_TYPES:
        vals = []
        for m in modes:
            tot = sum(comp[m].values()) or 1
            vals.append(comp[m][op] / tot)
            
        ax.bar(
            range(len(modes)), vals, bottom=bottoms, 
            label=op.replace("_", " ").title(), 
            color=OP_COLORS[op], edgecolor="white", linewidth=1.5, width=0.6
        )
        bottoms += np.array(vals)
        
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([MODE_LABEL.get(m, m) for m in modes], fontsize=9)
    ax.set_ylabel("Fraction of Operations")
    ax.set_title("Operation-Type Composition")
    
    # Place legend outside plot to avoid crowding
    ax.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="#E2E8F0", 
              bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)

    # --- Cost + Size Violins ---
    violin_configs = [
        (axes[1], "cost", "Edit Cost"),
        (axes[2], "num_operations", "Explanation Size |M|")
    ]
    
    for ax, key, title in violin_configs:
        data = [[payload.get(key, 0) for payload in groups[m]] for m in modes]
        data = [d if d else [0] for d in data]
        
        # Enhanced violin aesthetics
        parts = ax.violinplot(data, showmeans=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4299E1")
            pc.set_edgecolor("#2B6CB0")
            pc.set_alpha(0.6)
            
        parts['cmeans'].set_color('#2B6CB0')
        parts['cmins'].set_color('#A0AEC0')
        parts['cmaxes'].set_color('#A0AEC0')
        parts['cbars'].set_color('#A0AEC0')
            
        ax.set_xticks(range(1, len(modes) + 1))
        ax.set_xticklabels([MODE_LABEL.get(m, m) for m in modes], fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(key.replace("_", " ").title())

    fig.suptitle("Operation Type, Cost, and Explanation Size by Mode", y=1.02)
    fig.tight_layout()
    
    p = os.path.join(out_dir, "op_cost_size_by_mode.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)


def plot_node_type_enrichment(payloads: List[Dict[str, Any]], mode: str, out_dir: str):
    mod_types, bg_types = Counter(), Counter()
    
    for payload in payloads:
        tmap = node_type_map(payload)
        mod_nodes, _ = extract_modified(payload)
        for name, typ in tmap.items():
            if name in mod_nodes:
                mod_types[typ] += 1
            else:
                bg_types[typ] += 1
                
    n_mod, n_bg = sum(mod_types.values()), sum(bg_types.values())
    if n_mod == 0 or n_bg == 0:
        print(f"[F3/{mode}] no node-type data; skipping.")
        return
        
    types = sorted(set(mod_types) | set(bg_types))
    rows = []
    
    for t in types:
        p_mod = mod_types[t] / n_mod
        p_bg = (bg_types[t] / n_bg) or 1e-9
        lift = np.log2((p_mod or 1e-9) / p_bg)
        rows.append((t, lift, mod_types[t]))
        
    rows.sort(key=lambda r: r[1])
    
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(rows) + 1.5))
    
    # Diverging color palette: Green for positive lift, Red for negative
    colors = ["#38A169" if r[1] >= 0 else "#E53E3E" for r in rows]
    
    bars = ax.barh([r[0] for r in rows], [r[1] for r in rows], color=colors, edgecolor="white", linewidth=1.2, height=0.6)
    ax.axvline(0, color="#718096", linestyle="--", linewidth=1.2, zorder=0)
    ax.set_xlabel("log₂ lift: enrichment among flip-driving nodes (>0 over-represented)")
    
    for i, r in enumerate(rows):
        ha = "left" if r[1] >= 0 else "right"
        x_offset = 0.1 if r[1] >= 0 else -0.1
        ax.text(r[1] + x_offset, i, f"n={r[2]}", va="center", ha=ha, fontsize=9, color="#2D3748")
        
    ax.set_title(f"Node-Type Enrichment — {MODE_LABEL.get(mode, mode)}", y=1.05)
    fig.tight_layout()
    
    p = os.path.join(out_dir, f"node_type_enrichment_{mode}.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)


def plot_saliency_zipf(groups: Dict[str, List[Dict[str, Any]]], out_dir: str):
    modes = sorted(groups)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    any_data = False
    
    configs = [
        (axes[0], "nodes", "Node Saliency"), 
        (axes[1], "edges", "Edge Saliency")
    ]
    
    # Modern palette for the lines
    line_colors = ["#3182CE", "#DD6B20"]
    
    for ax, kind, title in configs:
        for idx, m in enumerate(modes):
            counter = Counter()
            for payload in groups[m]:
                mn, me = extract_modified(payload)
                counter.update(mn if kind == "nodes" else me)
                
            freqs = sorted(counter.values(), reverse=True)
            if not freqs:
                continue
                
            any_data = True
            
            # Thicker lines, prominent markers with white edges
            ax.loglog(
                range(1, len(freqs) + 1), freqs, 
                marker="o", markersize=6, linewidth=2, 
                markeredgecolor="white", markeredgewidth=1.2,
                color=line_colors[idx % len(line_colors)],
                label=MODE_LABEL.get(m, m)
            )
            
        ax.set_xlabel("Rank")
        ax.set_ylabel("Times Modified")
        ax.set_title(title)
        ax.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="#E2E8F0")
        
    if not any_data:
        plt.close(fig)
        print("[F4] no modified elements; skipping.")
        return
        
    fig.suptitle("Saliency Concentration (Rank–Frequency of Flip-Driving Elements)", y=1.02)
    fig.tight_layout()
    
    p = os.path.join(out_dir, "saliency_zipf.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

def load_kg_degree(graphml: Optional[str]) -> Optional[Dict[str, int]]:
    if not graphml or not os.path.isfile(graphml):
        if graphml:
            print(f"[warn] graphml not found ({graphml}); skipping global-degree feature.")
        return None
        
    G = nx.read_graphml(graphml)
    return dict(G.degree())

# --- Main Entry Point ---
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_figures",
        description="Global-explanation paper figures from local counterfactual JSONs (by mode)."
    )
    p.add_argument("--results-root", required=True,
                   help="Directory of counterfactual_*.json (searched recursively).")
    p.add_argument("--dataset", default=None,
                   help="Optional: filter paths containing /<dataset>/ and pick default graphml.")
    p.add_argument("--graphml", default=None,
                   help="Full KG GraphML for global-degree feature (default: KGs/lightrag/<dataset>/...).")
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default: src/global_explanations/figures/<dataset>).")
    p.add_argument("--figures", default="f1,f2,f3", 
                   help="Subset of f1,f2,f3,f4. Default: f1, f2, f3.")
    return p


def main(args: argparse.Namespace):
    # INJECT NEW STYLE CONFIGURATION HERE
    set_plot_style()
    
    payloads = dedupe_by_mode_question(load_explanations(args.results_root, args.dataset))
    
    if not payloads:
        err_msg = f"No found counterfactual JSONs under {args.results_root}"
        if args.dataset:
            err_msg += f" for dataset={args.dataset}"
        raise SystemExit(err_msg)
        
    groups = group_by_mode(payloads)
    mode_counts = ", ".join(f"{m}={len(v)}" for m, v in sorted(groups.items()))
    print(f"Loaded {len(payloads)} found explanation(s); modes: {mode_counts}")

    out_dir = args.out_dir or f"src/global_explanations/figures/{args.dataset or 'all'}"
    os.makedirs(out_dir, exist_ok=True)
    
    figs = {f.strip().lower() for f in args.figures.split(",") if f.strip()}

    kg_degree = None
    if "f1" in figs:
        graphml = args.graphml
        if not graphml and args.dataset:
            graphml = f"KGs/lightrag/{args.dataset}/graph_chunk_entity_relation.graphml"
        kg_degree = load_kg_degree(graphml)

    for mode, ps in sorted(groups.items()):
        if "f1" in figs:
            plot_feature_separation(collect_features(ps, kg_degree), mode, out_dir)
        if "f3" in figs:
            plot_node_type_enrichment(ps, mode, out_dir)
            
    if "f2" in figs:
        plot_op_cost_size(groups, out_dir)
    if "f4" in figs:
        plot_saliency_zipf(groups, out_dir)
        
    print(f"Done. Figures under {out_dir}/")

if __name__ == "__main__":
    main(build_arg_parser().parse_args())