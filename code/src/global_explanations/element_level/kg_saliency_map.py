"""
kg_saliency_map.py
──────────────────
Builds a KG saliency / frequency heatmap from a folder of counterfactual JSON
files, drawn on top of the FULL KG loaded from a GraphML file.

Node colour  : degree in the FULL KG (global structural importance).
Edge colour  : frequency across examples (how often the edge appears).
Node size    : frequency across examples (how often the node appears).

Only the top-N nodes by frequency are visualised (no neighbour expansion).

Output: kg_saliency_map.pdf
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────

JSON_DIR     = Path("src/counterfactuals/results/synthetic")
GRAPHML_PATH = Path("/home/gbalanos/GloRAG-Ex/code/KGs/lightrag/synthetic/graph_chunk_entity_relation.graphml")
OUTPUT_PDF   = "/home/gbalanos/GloRAG-Ex/code/src/global_explanations/element_level/plots/kg_saliency_map.pdf"

TOP_N        = 50

OP_EDGE_STYLE = {
    "add_edge":    "#2ECC71",
    "delete_edge": "#E74C3C",
}

NODE_CMAP       = "Reds"    # degree → seed node colour
EDGE_CMAP    = "YlOrRd"  # freq    → edge colour
LAYOUT_SEED  = 42
LAYOUT_K     = None
LAYOUT_ITERS = 100


# ── 1. Load full KG from GraphML ──────────────────────────────────────────────

def load_kg(graphml_path: Path) -> nx.DiGraph:
    G = nx.read_graphml(str(graphml_path))
    if not G.is_directed():
        G = G.to_directed()
    for n in G.nodes:
        G.nodes[n]["freq"] = 0
    for u, v in G.edges:
        G[u][v]["freq"] = 0
    print(f"KG loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ── 2. Load all JSON example files ───────────────────────────────────────────

def load_examples(directory: Path) -> list[dict]:
    examples = []
    for path in sorted(directory.rglob("*.json")):
        with open(path) as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    examples.extend(data)
                else:
                    examples.append(data)
            except json.JSONDecodeError:
                print(f"  [skip] Could not parse {path}")
    print(f"Loaded {len(examples)} examples from {directory}")
    return examples


# ── 3. Count frequencies ──────────────────────────────────────────────────────

def build_frequency_counts(examples: list[dict]):
    node_freq: Counter = Counter()
    edge_freq: Counter = Counter()

    for ex in examples:
        subgraph = ex.get("original_subgraph", {})
        nodes_in_ex = {e["name"] for e in subgraph.get("entities", [])}
        edges_in_ex = {
            (r["src"], r["tgt"]) for r in subgraph.get("relations", [])
        }
        node_freq.update(nodes_in_ex)
        edge_freq.update(edges_in_ex)

    return node_freq, edge_freq


# ── 4. Collect operations ─────────────────────────────────────────────────────

def build_operation_sets(examples: list[dict]):
    op_nodes: dict[str, set] = {}
    op_edges: dict[tuple, set] = {}

    for ex in examples:
        for op in ex.get("operations", []):
            kind = op[0]
            if kind in ("add_node", "delete_node"):
                op_nodes.setdefault(op[1], set()).add(kind)
            elif kind in ("add_edge", "delete_edge"):
                src, tgt = op[1][0], op[1][1]
                op_edges.setdefault((src, tgt), set()).add(kind)

    return op_nodes, op_edges


# ── 5. Overlay frequencies onto G ────────────────────────────────────────────

def overlay_frequencies(G: nx.DiGraph, node_freq, edge_freq, op_nodes, op_edges):
    for node, freq in node_freq.items():
        if node in G.nodes:
            G.nodes[node]["freq"] = freq

    for (src, tgt), freq in edge_freq.items():
        if G.has_edge(src, tgt):
            G[src][tgt]["freq"] = freq

    for node in op_nodes:
        if node not in G:
            G.add_node(node, freq=0)
    for (src, tgt) in op_edges:
        if src not in G:
            G.add_node(src, freq=0)
        if tgt not in G:
            G.add_node(tgt, freq=0)
        if not G.has_edge(src, tgt):
            G.add_edge(src, tgt, freq=0)

    return G


# ── 6. Extract top-N subgraph ─────────────────────────────────────────────────

def extract_top_subgraph(
    G: nx.DiGraph,
    node_freq: Counter,
    op_nodes: dict,
    top_n: int = TOP_N,
) -> nx.DiGraph:
    op_only = {n for n in op_nodes if n in G}
    ranked  = [n for n, _ in node_freq.most_common() if n in G and n not in op_only]

    # Frequency-ranked seeds — strictly top_n
    freq_seeds: list[str] = ranked[:top_n]
    freq_seed_set = set(freq_seeds)

    # 1-hop neighbours of freq seeds only
    neighbours: set[str] = set()
    for s in freq_seed_set:
        neighbours |= set(G.predecessors(s))
        neighbours |= set(G.successors(s))
    neighbours -= freq_seed_set

    # Op-only nodes are added only if they fall within the neighbourhood
    # (don't expand from them — they'd pull in unrelated parts of the KG)
    op_in_neighbourhood = op_only & (freq_seed_set | neighbours)

    keep = freq_seed_set | neighbours | op_in_neighbourhood

    sub = G.subgraph(keep).copy()
    for nd in sub.nodes:
        sub.nodes[nd]["is_seed"] = nd in freq_seed_set
    print(
        f"\nSubgraph: {len(freq_seed_set)} seed(s) + {len(neighbours)} neighbours "
        f"= {sub.number_of_nodes()} total nodes, {sub.number_of_edges()} edges"
    )
    return sub


# ── 7. Layout ─────────────────────────────────────────────────────────────────

def _layout(G: nx.DiGraph) -> dict:
    MIN_DIST = 0.18
    GRAVITY  = 0.15   # ← strength of pull toward centre (0 = none, 1 = strong)
    n = G.number_of_nodes()

    if n <= 30:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=LAYOUT_SEED, k=1.5 / max(np.sqrt(n), 1),
                                   iterations=LAYOUT_ITERS)
    else:
        k = LAYOUT_K if LAYOUT_K is not None else 1.8 / max(np.sqrt(n), 1)
        pos = nx.spring_layout(G, seed=LAYOUT_SEED, k=k, iterations=LAYOUT_ITERS)

    nodes  = list(pos.keys())
    coords = np.array([pos[nd] for nd in nodes], dtype=float)

    # ── repulsion: push overlapping nodes apart
    for _ in range(200):
        moved = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                delta = coords[i] - coords[j]
                dist  = np.linalg.norm(delta)
                if dist < MIN_DIST and dist > 1e-9:
                    push       = (MIN_DIST - dist) / 2 * (delta / dist)
                    coords[i] += push
                    coords[j] -= push
                    moved = True
        if not moved:
            break

    # ── gravity: pull every node toward the centroid
    centroid = coords.mean(axis=0)
    coords   = coords + GRAVITY * (centroid - coords)

    return {nd: coords[i] for i, nd in enumerate(nodes)}


# ── 8. Plot ───────────────────────────────────────────────────────────────────

def plot_saliency(G_sub, G_full, node_freq, edge_freq, op_nodes, op_edges):
    """
    G_sub  : trimmed subgraph to visualise
    G_full : full KG — used to read global degree for node colouring
    """
    pos = _layout(G_sub)

    # ── global degree from the full KG (in-degree + out-degree)
    global_degree = {n: G_full.in_degree(n) + G_full.out_degree(n)
                     for n in G_sub.nodes}
    max_deg = max(global_degree.values(), default=1)
    max_ef  = max(edge_freq.values(), default=1)
    max_nf  = max(node_freq.values(), default=1)

    n_cmap = plt.cm.get_cmap(NODE_CMAP)
    e_cmap = plt.cm.get_cmap(EDGE_CMAP)

    # ── node colour: seeds → Blues heatmap by degree; neighbours → flat grey
    def node_colour(n):
        is_seed = G_sub.nodes[n].get("is_seed", False)
        if not is_seed:
            return "#CCCCCC"
        deg = global_degree.get(n, 0)
        return "#DDDDDD" if deg == 0 else n_cmap(0.25 + 0.75 * deg / max_deg)

    def node_size(n):
        freq = G_sub.nodes[n].get("freq", 0)
        return 80 + 320 * (freq / max(max_nf, 1))

    node_colours = [node_colour(n) for n in G_sub.nodes]
    node_sizes   = [node_size(n)   for n in G_sub.nodes]

    # ── edge colour = frequency (YlOrRd); uniform slim width
    EDGE_WIDTH_FG = 0.9
    EDGE_WIDTH_BG = 0.4

    def edge_colour(u, v):
        ops = op_edges.get((u, v), set())
        if "add_edge"    in ops: return OP_EDGE_STYLE["add_edge"]
        if "delete_edge" in ops: return OP_EDGE_STYLE["delete_edge"]
        freq = G_sub[u][v].get("freq", 0)
        return "#DDDDDD" if freq == 0 else e_cmap(freq / max_ef)

    # ── figure
    fig, ax = plt.subplots(figsize=(18, 14))
    fig.subplots_adjust(bottom=0.14)

    nx.draw_networkx_nodes(
        G_sub, pos, ax=ax,
        node_color=node_colours,
        node_size=node_sizes,
        linewidths=0,
    )

    salient_nodes = {n for n in G_sub.nodes
                     if G_sub.nodes[n].get("freq", 0) > 0 or n in op_nodes}
    nx.draw_networkx_labels(
        G_sub, pos, ax=ax,
        labels={n: n for n in salient_nodes},
        font_size=8,
        font_color="#111111",
    )

    bg_edges = [(u, v) for u, v in G_sub.edges()
                if G_sub[u][v].get("freq", 0) == 0 and (u, v) not in op_edges]
    fg_edges = [(u, v) for u, v in G_sub.edges()
                if G_sub[u][v].get("freq", 0) > 0 or (u, v) in op_edges]

    nx.draw_networkx_edges(
        G_sub, pos, ax=ax,
        edgelist=bg_edges,
        edge_color="#AAAAAA",
        width=EDGE_WIDTH_BG,
        arrows=True,
        arrowsize=8,
        connectionstyle="arc3,rad=0.1",
        min_source_margin=20,
        min_target_margin=20,
        alpha=0.5,
    )
    nx.draw_networkx_edges(
        G_sub, pos, ax=ax,
        edgelist=fg_edges,
        edge_color=[edge_colour(u, v) for u, v in fg_edges],
        width=EDGE_WIDTH_FG,
        arrows=True,
        arrowsize=12,
        connectionstyle="arc3,rad=0.1",
        min_source_margin=20,
        min_target_margin=20,
        alpha=0.85,
    )

    # ── colorbars: horizontal at the bottom
    #    left  → node degree (Blues)
    #    right → edge frequency (YlOrRd)
    # ── colorbars: node degree (seeds) | node degree (neighbours) | bottom centre
    cax1 = fig.add_axes([0.30, 0.05, 0.40, 0.022])
    sm_seed = plt.cm.ScalarMappable(
        cmap=NODE_CMAP, norm=plt.Normalize(vmin=0, vmax=max_deg)
    )
    sm_seed.set_array([])
    cb1 = fig.colorbar(sm_seed, cax=cax1, orientation="horizontal")
    cb1.set_label("Important node degree (full KG)", fontsize=10, labelpad=4)
    cb1.ax.tick_params(labelsize=8)

    ax.axis("off")

    # ── crop very tightly around node positions
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        pad_x = (max(xs) - min(xs)) * 0.04 or 0.2
        pad_y = (max(ys) - min(ys)) * 0.04 or 0.2
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.show()
    print(f"Saved to {OUTPUT_PDF}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    G_full = load_kg(GRAPHML_PATH)

    examples = load_examples(JSON_DIR)

    node_freq, edge_freq = build_frequency_counts(examples)
    op_nodes,  op_edges  = build_operation_sets(examples)

    print(f"\nUnique nodes seen : {len(node_freq)}")
    print(f"Unique edges seen : {len(edge_freq)}")
    print(f"Op nodes          : {len(op_nodes)}")
    print(f"Op edges          : {len(op_edges)}")

    G_full = overlay_frequencies(G_full, node_freq, edge_freq, op_nodes, op_edges)
    G_sub  = extract_top_subgraph(G_full, node_freq, op_nodes, top_n=TOP_N)

    # Pass both: sub for layout/drawing, full for global degree lookup
    plot_saliency(G_sub, G_full, node_freq, edge_freq, op_nodes, op_edges)