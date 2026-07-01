"""
Usage:
    python analyze_kg_nodes.py --graph global_kg.graphml --ops_dir ./ops_folder/
    python analyze_kg_nodes.py --graph global_kg.graphml --ops_dir ./ops_folder/all_ops_ff/

For each node referenced in the 'operations' fields of the JSON files,
looks it up in the global KG and reports:
  - degree (in, out, total)
  - PageRank
  - node type/label distribution across all such nodes

If ops_dir ends with 'all_ops_ff', deletions are treated as irrelevant/noise
and additions are treated as flip nodes — analyzed separately.
"""

import argparse
import json
import os
import networkx as nx
import pandas as pd
from pathlib import Path
from collections import Counter
import warnings


# ──────────────────────────────────────────────
# 1. Argument parsing
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Analyze KG nodes referenced in ops JSON files.")
    p.add_argument("--graph",          required=True,        help="Path to the global KG (.graphml)")
    p.add_argument("--ops_dir",        required=True,        help="Directory containing the JSON operation files")
    p.add_argument("--output",         default=None,         help="Optional CSV output path for per-node stats")
    p.add_argument("--plots_dir",      default="./plots",    help="Directory to save plots (default: ./plots)")
    p.add_argument("--pagerank_alpha", type=float, default=0.85, help="Damping factor for PageRank (default 0.85)")
    return p.parse_args()


# ──────────────────────────────────────────────
# 2. Extract node names from operations
# ──────────────────────────────────────────────

def extract_nodes_from_ops(ops_dir: str) -> dict[str, list[str]]:
    """
    Returns a dict: { node_name -> [list of source json filenames] }
    """
    node_sources: dict[str, list[str]] = {}

    json_files = list(Path(ops_dir).glob("**/*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {ops_dir}")

    for jf in json_files:
        with open(jf) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                warnings.warn(f"Skipping {jf.name}: {e}")
                continue

        operations = data.get("operations", [])
        for op in operations:
            if not op or len(op) < 2:
                continue
            op_type, operand = op[0], op[1]
            if op_type in ("delete_node", "add_node") and isinstance(operand, str):
                node_sources.setdefault(operand, []).append(jf.name)
            elif op_type in ("delete_edge", "add_edge") and isinstance(operand, (list, tuple)):
                for node in operand:
                    node_sources.setdefault(node, []).append(jf.name)

    return node_sources


def extract_nodes_split(ops_dir: str) -> tuple[dict, dict]:
    """
    For all_ops_ff directories: split into
      - additions (flip nodes, meaningful)
      - deletions (irrelevant/noise)
    Returns (addition_sources, deletion_sources)
    """
    addition_sources: dict[str, list[str]] = {}
    deletion_sources: dict[str, list[str]] = {}

    json_files = list(Path(ops_dir).glob("**/*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {ops_dir}")

    for jf in json_files:
        with open(jf) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                warnings.warn(f"Skipping {jf.name}: {e}")
                continue

        operations = data.get("operations", [])
        for op in operations:
            if not op or len(op) < 2:
                continue
            op_type, operand = op[0], op[1]

            is_addition = op_type in ("add_node", "add_edge")
            is_deletion = op_type in ("delete_node", "delete_edge")
            target = addition_sources if is_addition else deletion_sources if is_deletion else None
            if target is None:
                continue

            if op_type in ("delete_node", "add_node") and isinstance(operand, str):
                target.setdefault(operand, []).append(jf.name)
            elif op_type in ("delete_edge", "add_edge") and isinstance(operand, (list, tuple)):
                for node in operand:
                    target.setdefault(node, []).append(jf.name)

    return addition_sources, deletion_sources


# ──────────────────────────────────────────────
# 3. Load graph & compute metrics
# ──────────────────────────────────────────────

def load_graph(graph_path: str) -> nx.Graph | nx.DiGraph:
    G = nx.read_graphml(graph_path)
    print(f"Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges  "
          f"({'directed' if G.is_directed() else 'undirected'})")
    return G


def compute_pagerank(G: nx.Graph, alpha: float) -> dict:
    try:
        return nx.pagerank(G, alpha=alpha)
    except nx.PowerIterationFailedConvergence:
        warnings.warn("PageRank did not converge; returning zeros.")
        return {n: 0.0 for n in G.nodes()}


def get_node_type(G: nx.Graph, node: str) -> str:
    attrs = G.nodes.get(node, {})
    return attrs.get("type") or attrs.get("label") or attrs.get("entity_type") or "-"


def analyze_nodes(G: nx.Graph, node_sources: dict, alpha: float) -> pd.DataFrame:
    is_directed = G.is_directed()
    pr = compute_pagerank(G, alpha)

    rows = []
    missing = []

    for node, sources in sorted(node_sources.items()):
        if node not in G:
            missing.append(node)
            rows.append({
                "node": node,
                "in_graph": False,
                "degree": None,
                "in_degree": None,
                "out_degree": None,
                "pagerank": None,
                "node_type": "-",
                "num_json_files": len(sources),
                "source_files": "; ".join(sources),
            })
            continue

        rows.append({
            "node": node,
            "in_graph": True,
            "degree": G.degree(node),
            "in_degree":  G.in_degree(node)  if is_directed else None,
            "out_degree": G.out_degree(node) if is_directed else None,
            "pagerank": pr.get(node, 0.0),
            "node_type": get_node_type(G, node),
            "num_json_files": len(sources),
            "source_files": "; ".join(sources),
        })

    if missing:
        print(f"\n⚠  {len(missing)} node(s) from ops not found in graph:")
        for m in missing:
            print(f"   • {m}")

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 4. Summary + plots
# ──────────────────────────────────────────────

def print_summary(df: pd.DataFrame, label: str = ""):
    found = df[df["in_graph"]]
    tag   = f" [{label}]" if label else ""
    print(f"\n{'─'*55}")
    print(f"  {tag} Nodes referenced in ops :  {len(df)}")
    print(f"  {tag} Found in global KG      :  {len(found)}  ({len(df)-len(found)} missing)")

    if found.empty:
        print("  No matched nodes to summarise.")
        return

    print(f"\n  DEGREE STATS")
    print(found["degree"].describe().to_string())

    if found["in_degree"].notna().any():
        print(f"\n  IN-DEGREE")
        print(found["in_degree"].describe().to_string())
        print(f"\n  OUT-DEGREE")
        print(found["out_degree"].describe().to_string())

    print(f"\n{'─'*55}")
    print(f"  PAGERANK STATS{tag}")
    print(found["pagerank"].describe().to_string())

    print(f"\n{'─'*55}")
    print(f"  NODE TYPE DISTRIBUTION{tag}")
    type_counts = Counter(found["node_type"])
    total = sum(type_counts.values())
    for t, c in type_counts.most_common():
        bar = "█" * int(30 * c / total)
        print(f"   {t:<30s}  {c:4d}  {c/total*100:5.1f}%  {bar}")

    print(f"\n  TOP 20 NODES BY PAGERANK{tag}")
    print(found.nlargest(20, "pagerank")[["node", "degree", "pagerank", "node_type"]].to_string(index=False))

    print(f"\n  TOP 20 NODES BY DEGREE{tag}")
    print(found.nlargest(20, "degree")[["node", "degree", "pagerank", "node_type"]].to_string(index=False))


# def save_plots(df: pd.DataFrame, plots_dir: str, label: str = ""):
#     import matplotlib.pyplot as plt

#     os.makedirs(plots_dir, exist_ok=True)
#     found = df[df["in_graph"]]
#     if found.empty:
#         return

#     slug = f"_{label}" if label else ""

#     # # degree distribution
#     # fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
#     # found["degree"].hist(bins=30, ax=ax, color="#0072B2", edgecolor="white")
#     # ax.set_xlabel("Degree")
#     # ax.set_ylabel("Count")
#     # ax.set_title(f"Degree Distribution{' — ' + label if label else ''}")
#     # plt.tight_layout()
#     # plt.savefig(os.path.join(plots_dir, f"degree_dist{slug}.pdf"), format="pdf")
#     # plt.close()

#     # # pagerank distribution
#     # fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
#     # found["pagerank"].hist(bins=30, ax=ax, color="#D55E00", edgecolor="white")
#     # ax.set_xlabel("PageRank")
#     # ax.set_ylabel("Count")
#     # ax.set_title(f"PageRank Distribution{' — ' + label if label else ''}")
#     # plt.tight_layout()
#     # plt.savefig(os.path.join(plots_dir, f"pagerank_dist{slug}.pdf"), format="pdf")
#     # plt.close()

#     # node type bar chart
#     type_counts = Counter(found["node_type"])
#     total = sum(type_counts.values())
#     labels = list(type_counts.keys())
#     values = [c / total * 100 for c in type_counts.values()]

#     fig, ax = plt.subplots(figsize=(6, max(3, len(labels) * 0.4)), dpi=150)
#     ax.barh(labels, values, color="#009E73", edgecolor="white")
#     ax.set_ylabel("Node Type")
#     ax.set_xlabel("% of nodes")
#     # ax.set_title(f"Node Type Distribution{' — ' + label if label else ''}")
#     ax.invert_yaxis()
#     ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
#     plt.tight_layout()
#     plt.savefig(os.path.join(plots_dir, f"node_type_dist{slug}.pdf"), format="pdf")
#     plt.close()

#     print(f"✓ Plots saved to: {plots_dir}  (suffix: '{slug}')")


def save_plots(df: pd.DataFrame, plots_dir: str, label: str = "", G: nx.Graph = None):
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)
    found = df[df["in_graph"]]
    if found.empty:
        return

    slug = f"_{label}" if label else ""

    # node type bar chart — normalized by global KG type frequency
    type_counts = Counter(found["node_type"])

    if G is not None:
        # global type distribution from full KG
        global_type_counts = Counter(
            (attrs.get("type") or attrs.get("label") or attrs.get("entity_type") or "-")
            for _, attrs in G.nodes(data=True)
        )
        global_total = sum(global_type_counts.values())

        # enrichment = (ops % of type) / (global % of type)
        ops_total = sum(type_counts.values())
        enrichment = {}
        for t, c in type_counts.items():
            ops_pct    = c / ops_total
            global_pct = global_type_counts.get(t, 0) / global_total if global_total > 0 else 0
            enrichment[t] = ops_pct / global_pct if global_pct > 0 else float("inf")

        # sort by enrichment descending
        sorted_types = sorted(enrichment, key=enrichment.get, reverse=True)
        values = [enrichment[t] for t in sorted_types]
        colors = ["#D55E00" if v >= 1 else "#0072B2" for v in values]  # red=enriched, blue=depleted
        xlabel = "Enrichment (ops % / global %)"
        vline  = 1.0  # reference line at 1 = no enrichment
    else:
        # fallback: plain percentage
        ops_total = sum(type_counts.values())
        sorted_types = [t for t, _ in type_counts.most_common()]
        values = [type_counts[t] / ops_total * 100 for t in sorted_types]
        colors = ["#009E73"] * len(sorted_types)
        xlabel = "% of nodes"
        vline  = None

    fig, ax = plt.subplots(figsize=(6, max(3, len(sorted_types) * 0.4)), dpi=150)
    bars = ax.barh(sorted_types, values, color=colors, edgecolor="white")
    ax.set_ylabel("Node Type")
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()

    if vline is not None:
        ax.axvline(vline, color="black", linewidth=0.8, linestyle="--", label="No enrichment")
        ax.legend(fontsize=8)

    if vline is None:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}×"))

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"node_type_dist{slug}.pdf"), format="pdf")
    plt.close()

    print(f"✓ Plots saved to: {plots_dir}  (suffix: '{slug}')")


# ──────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"Loading graph from: {args.graph}")
    G = load_graph(args.graph)

    ops_dir      = args.ops_dir.rstrip("/")
    is_all_ops_ff = Path(ops_dir).name == "all_ops_ff"

    if is_all_ops_ff:
        print(f"\nDetected 'all_ops_ff' directory — splitting additions (flips) vs deletions (noise).")
        addition_sources, deletion_sources = extract_nodes_split(ops_dir)

        print(f"  Additions (flip nodes): {len(addition_sources)} unique nodes")
        print(f"  Deletions (noise):      {len(deletion_sources)} unique nodes")

        df_add = analyze_nodes(G, addition_sources, alpha=args.pagerank_alpha)
        df_del = analyze_nodes(G, deletion_sources, alpha=args.pagerank_alpha)

        print_summary(df_add, label="F→T flips (additions)")
        print_summary(df_del, label="Irrelevant (deletions)")

        # save_plots(df_add, args.plots_dir, label="additions")
        # save_plots(df_del, args.plots_dir, label="deletions")

        save_plots(df_add, args.plots_dir, label="additions", G=G)
        save_plots(df_del, args.plots_dir, label="deletions", G=G)

        out_add = args.output or os.path.join(ops_dir, "ops_node_stats_additions.csv")
        out_del = (Path(out_add).parent / (Path(out_add).stem + "_deletions" + Path(out_add).suffix)).as_posix()
        df_add.to_csv(out_add, index=False)
        df_del.to_csv(out_del, index=False)
        print(f"\n✓ Addition stats saved to: {out_add}")
        print(f"✓ Deletion stats saved to:  {out_del}")

    else:
        print(f"Scanning JSON ops files in: {ops_dir}")
        node_sources = extract_nodes_from_ops(ops_dir)
        print(f"Found {len(node_sources)} unique node(s) across all operation files.")

        df = analyze_nodes(G, node_sources, alpha=args.pagerank_alpha)
        print_summary(df)
        # save_plots(df, args.plots_dir)
        save_plots(df, args.plots_dir, G=G)

        out_path = args.output or os.path.join(ops_dir, "ops_node_stats.csv")
        df.to_csv(out_path, index=False)
        print(f"\n✓ Per-node stats saved to: {out_path}")


if __name__ == "__main__":
    main()