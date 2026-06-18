"""Global-explanation figures for the paper, from a folder of local counterfactual
JSONs (generate.py :: save_operations_to_json), stratified by run mode (ft vs ff).

One parameterized driver that supersedes the ad-hoc per-aggregator __main__'s. It
recursively loads `counterfactual_*.json`, keeps the found flips, dedupes per
(mode, question), and emits:

  F1  feature_separation_<mode>.png  + feature_importance_<mode>.png
        per-feature distributions of FLIP-DRIVING elements vs the REST of the
        context graph, plus a sorted AUC / KS importance bar. This is the
        "which features are systematically associated with flips" figure.
  F2  op_cost_size_by_mode.png
        operation-type composition + edit-cost + explanation-size (|M|) per mode.
  F3  node_type_enrichment_<mode>.png
        lift of each entity type among modified nodes vs background context nodes.
  F4  saliency_zipf.png
        log-log rank-frequency of how often each node/edge is modified (saliency
        concentration).

Run from code/ (CWD=code), e.g.:
  ../.venv/bin/python -m src.global_explanations.make_figures \
      --results-root src/counterfactuals/results/<RUN_TS>/synthetic \
      --dataset synthetic
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

NODE_FEATURES = [
    ("local_degree",   "Local degree (Cᵢ)"),
    ("global_degree",  "Global degree (KG)"),
    ("local_closeness", "Closeness (Cᵢ)"),
    ("pagerank",       "PageRank (Cᵢ)"),
]
EDGE_FEATURES = [
    ("local_betweenness", "Edge betweenness (Cᵢ)"),
]
OP_TYPES = ["delete_node", "delete_edge", "add_node", "add_edge"]
OP_COLORS = {"delete_node": "#E74C3C", "delete_edge": "#C0392B",
             "add_node": "#2ECC71", "add_edge": "#27AE60"}
MODE_LABEL = {"ft": "ft (breaking T→F)", "ff": "ff (corrective F→T)"}


def load_explanations(results_root: str, dataset: str | None) -> list[dict]:
    """Recursively load found counterfactual JSONs (optionally filtered by dataset path)."""
    files = sorted(glob.glob(os.path.join(results_root, "**", "counterfactual_*.json"), recursive=True))
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


def dedupe_by_mode_question(payloads: list[dict]) -> list[dict]:
    """Keep the latest-timestamp payload per (mode, question) — collapses no-PSP/PSP dupes."""
    best: dict[tuple, dict] = {}
    for p in payloads:
        key = (p.get("mode"), p.get("question", ""))
        ts = p.get("timestamp", "")
        if key not in best or ts > best[key].get("timestamp", ""):
            best[key] = p
    return list(best.values())


def group_by_mode(payloads: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in payloads:
        groups[p.get("mode", "?")].append(p)
    return dict(groups)


def assemble_context_graph(payload: dict) -> nx.DiGraph:
    """Build the context graph Cᵢ from original_subgraph (entities + relations)."""
    og = payload.get("original_subgraph") or {}
    G = nx.DiGraph()
    for e in og.get("entities") or []:
        G.add_node(e.get("name", ""), entity_type=e.get("type", ""))
    for r in og.get("relations") or []:
        src, tgt = r.get("src", ""), r.get("tgt", "")
        if src and tgt:
            G.add_edge(src, tgt)
    return G


def extract_modified(payload: dict) -> tuple[set, set]:
    """(modified node names, modified (src,tgt) edges) from the operations list."""
    nodes, edges = set(), set()
    for op in payload.get("operations") or []:
        kind = op[0]
        if kind in ("delete_node", "add_node"):
            nodes.add(op[1])
        elif kind in ("delete_edge", "add_edge"):
            edges.add((op[1][0], op[1][1]))
    return nodes, edges


def node_type_map(payload: dict) -> dict[str, str]:
    og = payload.get("original_subgraph") or {}
    return {e.get("name", ""): (e.get("type", "") or "untyped") for e in (og.get("entities") or [])}


# Feature collection
def collect_features(payloads: list[dict], kg_degree: dict | None):
    """Return {feat: {'mod': [...], 'bg': [...]}} for node + edge features.

    'mod' = flip-driving elements; 'bg' = the REST of the context graph (complement),
    so the comparison isolates what distinguishes modified elements from the rest.
    """
    acc = {k: {"mod": [], "bg": []} for k, _ in NODE_FEATURES + EDGE_FEATURES}
    for p in payloads:
        G = assemble_context_graph(p)
        if G.number_of_nodes() == 0:
            continue
        mod_nodes, mod_edges = extract_modified(p)

        deg = dict(G.degree())
        clo = nx.closeness_centrality(G)
        try:
            pr = nx.pagerank(G)
        except Exception:
            pr = {n: float("nan") for n in G.nodes}
        per_node = {
            "local_degree":    deg,
            "local_closeness": clo,
            "pagerank":        pr,
        }
        if kg_degree is not None:
            per_node["global_degree"] = {n: kg_degree.get(n) for n in G.nodes}

        for feat, table in per_node.items():
            for n, val in table.items():
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                acc[feat]["mod" if n in mod_nodes else "bg"].append(val)

        if G.number_of_edges() > 0:
            try:
                ebc = nx.edge_betweenness_centrality(G)
            except Exception:
                ebc = {}
            for e, val in ebc.items():
                acc["local_betweenness"]["mod" if e in mod_edges else "bg"].append(val)
    return acc


def auc_ks(mod: list, bg: list):
    """Common-language effect size AUC = P(mod > bg) (tie-aware) and the KS statistic."""
    mod = np.asarray(mod, float)
    bg = np.asarray(bg, float)
    if len(mod) == 0 or len(bg) == 0:
        return None, None
    # AUC via rank counting against sorted background.
    bg_sorted = np.sort(bg)
    less = np.searchsorted(bg_sorted, mod, side="left")    # #bg strictly < m
    lesseq = np.searchsorted(bg_sorted, mod, side="right")  # #bg <= m
    auc = (less.sum() + 0.5 * (lesseq - less).sum()) / (len(mod) * len(bg))
    # KS = max gap between the two empirical CDFs over the pooled support.
    grid = np.sort(np.concatenate([mod, bg]))
    cdf_m = np.searchsorted(np.sort(mod), grid, side="right") / len(mod)
    cdf_b = np.searchsorted(bg_sorted, grid, side="right") / len(bg)
    ks = float(np.max(np.abs(cdf_m - cdf_b)))
    return float(auc), ks


def plot_feature_separation(acc: dict, mode: str, out_dir: str):
    feats = [(k, lbl) for k, lbl in NODE_FEATURES + EDGE_FEATURES
             if acc[k]["mod"] and acc[k]["bg"]]
    if not feats:
        print(f"[F1/{mode}] no feature data; skipping.")
        return

    # (a) overlaid distributions
    ncol = min(3, len(feats))
    nrow = (len(feats) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (key, lbl) in zip(axes.flat, feats):
        ax.axis("on")
        mod, bg = acc[key]["mod"], acc[key]["bg"]
        lo = min(min(mod), min(bg))
        hi = max(max(mod), max(bg))
        bins = np.linspace(lo, hi, 20) if hi > lo else 10
        ax.hist(bg, bins=bins, density=True, color="#BBBBBB", alpha=0.7, label="rest of context")
        ax.hist(mod, bins=bins, density=True, color="#4C72B0", alpha=0.6, label="flip-driving")
        auc, ks = auc_ks(mod, bg)
        ax.set_title(f"{lbl}\nAUC={auc:.2f}  KS={ks:.2f}", fontsize=9)
        ax.set_ylabel("density")
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle(f"Flip-driving vs. rest of context — {MODE_LABEL.get(mode, mode)}", fontweight="bold")
    fig.tight_layout()
    p1 = os.path.join(out_dir, f"feature_separation_{mode}.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p1}")

    # (b) importance bar (AUC, sorted; KS annotated)
    rows = []
    for key, lbl in feats:
        auc, ks = auc_ks(acc[key]["mod"], acc[key]["bg"])
        if auc is not None:
            rows.append((lbl, auc, ks, len(acc[key]["mod"])))
    rows.sort(key=lambda r: abs(r[1] - 0.5))
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(rows) + 1.5))
    ylab = [r[0] for r in rows]
    aucs = [r[1] for r in rows]
    colors = ["#4C72B0" if a >= 0.5 else "#DD8452" for a in aucs]
    bars = ax.barh(ylab, [a - 0.5 for a in aucs], left=0.5, color=colors, edgecolor="black")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC: P(flip-driving > rest)  —  0.5 = no separation")
    for bar, r in zip(bars, rows):
        ax.text(r[1] + (0.01 if r[1] >= 0.5 else -0.01), bar.get_y() + bar.get_height() / 2,
                f"KS={r[2]:.2f} (n={r[3]})", va="center",
                ha="left" if r[1] >= 0.5 else "right", fontsize=8)
    ax.set_title(f"Feature importance — {MODE_LABEL.get(mode, mode)}", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p2 = os.path.join(out_dir, f"feature_importance_{mode}.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p2}")


# F2: op-type + cost + size
def plot_op_cost_size(groups: dict[str, list[dict]], out_dir: str):
    modes = sorted(groups)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # op-type composition (stacked fraction per mode)
    ax = axes[0]
    comp = {m: Counter() for m in modes}
    for m in modes:
        for p in groups[m]:
            for op in p.get("operations") or []:
                if op[0] in OP_TYPES:
                    comp[m][op[0]] += 1
    bottoms = np.zeros(len(modes))
    for op in OP_TYPES:
        vals = []
        for m in modes:
            tot = sum(comp[m].values()) or 1
            vals.append(comp[m][op] / tot)
        ax.bar(range(len(modes)), vals, bottom=bottoms, label=op, color=OP_COLORS[op], edgecolor="white")
        bottoms += np.array(vals)
    ax.set_xticks(range(len(modes))); ax.set_xticklabels([MODE_LABEL.get(m, m) for m in modes], fontsize=8)
    ax.set_ylabel("fraction of operations"); ax.set_title("Operation-type composition")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    # cost + size violins
    for ax, key, title in ((axes[1], "cost", "Edit cost"),
                           (axes[2], "num_operations", "Explanation size |M|")):
        data = [[p.get(key, 0) for p in groups[m]] for m in modes]
        data = [d if d else [0] for d in data]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0"); pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(modes) + 1)); ax.set_xticklabels([MODE_LABEL.get(m, m) for m in modes], fontsize=8)
        ax.set_title(title); ax.set_ylabel(key)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Operation type, cost, and explanation size by mode", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "op_cost_size_by_mode.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


# F3: node-type enrichment
def plot_node_type_enrichment(payloads: list[dict], mode: str, out_dir: str):
    mod_types, bg_types = Counter(), Counter()
    for p in payloads:
        tmap = node_type_map(p)
        mod_nodes, _ = extract_modified(p)
        for name, typ in tmap.items():
            (mod_types if name in mod_nodes else bg_types)[typ] += 1
    n_mod, n_bg = sum(mod_types.values()), sum(bg_types.values())
    if n_mod == 0 or n_bg == 0:
        print(f"[F3/{mode}] no node-type data; skipping.")
        return
    types = sorted(set(mod_types) | set(bg_types))
    rows = []
    for t in types:
        p_mod = mod_types[t] / n_mod
        p_bg = (bg_types[t] / n_bg) or 1e-9
        rows.append((t, np.log2((p_mod or 1e-9) / p_bg), mod_types[t]))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 1.5))
    colors = ["#4C72B0" if r[1] >= 0 else "#DD8452" for r in rows]
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=colors, edgecolor="black")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("log₂ lift: enrichment among flip-driving nodes (>0 over-represented)")
    for i, r in enumerate(rows):
        ax.text(r[1], i, f"  n={r[2]}", va="center", ha="left" if r[1] >= 0 else "right", fontsize=8)
    ax.set_title(f"Node-type enrichment — {MODE_LABEL.get(mode, mode)}", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(out_dir, f"node_type_enrichment_{mode}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


# F4: saliency Zipf
def plot_saliency_zipf(groups: dict[str, list[dict]], out_dir: str):
    modes = sorted(groups)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    any_data = False
    for ax, kind, title in ((axes[0], "nodes", "Node saliency"), (axes[1], "edges", "Edge saliency")):
        for m in modes:
            counter = Counter()
            for p in groups[m]:
                mn, me = extract_modified(p)
                counter.update(mn if kind == "nodes" else me)
            freqs = sorted(counter.values(), reverse=True)
            if not freqs:
                continue
            any_data = True
            ax.loglog(range(1, len(freqs) + 1), freqs, marker="o", markersize=3,
                      linestyle="-", label=MODE_LABEL.get(m, m))
        ax.set_xlabel("rank"); ax.set_ylabel("times modified"); ax.set_title(title)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    if not any_data:
        plt.close(fig); print("[F4] no modified elements; skipping."); return
    fig.suptitle("Saliency concentration (rank–frequency of flip-driving elements)", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "saliency_zipf.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


def load_kg_degree(graphml: str | None):
    if not graphml or not os.path.isfile(graphml):
        if graphml:
            print(f"[warn] graphml not found ({graphml}); skipping global-degree feature.")
        return None
    G = nx.read_graphml(graphml)
    return dict(G.degree())


def main(args):
    payloads = dedupe_by_mode_question(load_explanations(args.results_root, args.dataset))
    if not payloads:
        raise SystemExit(f"No found counterfactual JSONs under {args.results_root}"
                         + (f" for dataset={args.dataset}" if args.dataset else ""))
    groups = group_by_mode(payloads)
    print(f"Loaded {len(payloads)} found explanation(s); modes: "
          + ", ".join(f"{m}={len(v)}" for m, v in sorted(groups.items())))

    out_dir = args.out_dir or f"src/global_explanations/figures/{args.dataset or 'all'}"
    os.makedirs(out_dir, exist_ok=True)
    figs = {f.strip() for f in args.figures.split(",") if f.strip()}

    kg_degree = None
    if "f1" in figs:
        graphml = args.graphml or (f"KGs/lightrag/{args.dataset}/graph_chunk_entity_relation.graphml"
                                   if args.dataset else None)
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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_figures",
        description="Global-explanation paper figures from local counterfactual JSONs (by mode).")
    p.add_argument("--results-root", required=True,
                   help="Directory of counterfactual_*.json (searched recursively).")
    p.add_argument("--dataset", default=None,
                   help="Optional: filter paths containing /<dataset>/ and pick default graphml.")
    p.add_argument("--graphml", default=None,
                   help="Full KG GraphML for the global-degree feature (default: KGs/lightrag/<dataset>/...).")
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default: src/global_explanations/figures/<dataset>).")
    p.add_argument("--figures", default="f1,f2,f3", help="Subset of f1,f2,f3,f4. Default: f1, f2, f3.")
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
