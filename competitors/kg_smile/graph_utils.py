from __future__ import annotations

import networkx as nx
import numpy as np


# ─────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────

def build_graph(subgraph: dict) -> nx.DiGraph:
    """
    Build a DiGraph from a subgraph dict with "entities" and "relations" keys.

    Accepts both shapes found across the codebase:
      - aggregation.py shape:  {"entities": [{"name": ...}], "relations": [{"src", "tgt"}]}
      - compare.py shape:      same, with an optional "description" field on relations
    """
    G = nx.DiGraph()

    for ent in subgraph.get("entities", []):
        G.add_node(ent["name"])

    for rel in subgraph.get("relations", []):
        kwargs = {}
        if "description" in rel:
            kwargs["description"] = rel["description"]
        G.add_edge(rel["src"], rel["tgt"], **kwargs)

    return G


def build_graph_from_result(result: dict) -> nx.DiGraph:
    """Convenience wrapper: build a graph from a full result JSON object."""
    return build_graph(result["original_subgraph"])


# ─────────────────────────────────────────────────────────────
# Sequential graph-edit cost functions
# ─────────────────────────────────────────────────────────────

def execute_and_cost_node_deletion(
    G: nx.DiGraph, node_to_delete: str
) -> tuple[nx.DiGraph, int]:
    """
    Remove a node and its incident edges; also prune any neighbours that
    become singletons.  Returns the updated graph and the edit cost.

    Cost = 1 (node) + |incident edges| + |singleton neighbours created|
    """
    if node_to_delete not in G:
        return G, 0

    predecessors = list(G.predecessors(node_to_delete))
    successors   = list(G.successors(node_to_delete))
    neighbors    = set(predecessors + successors)

    incident_edges = G.in_degree(node_to_delete) + G.out_degree(node_to_delete)

    new_G = G.copy()
    new_G.remove_node(node_to_delete)

    singleton_neighbors = [
        n for n in neighbors
        if new_G.has_node(n) and (new_G.in_degree(n) + new_G.out_degree(n) == 0)
    ]

    cost = 1 + incident_edges + len(singleton_neighbors)
    new_G.remove_nodes_from(singleton_neighbors)

    return new_G, cost


def execute_and_cost_edge_deletion(
    G: nx.DiGraph, edge_to_delete: tuple[str, str]
) -> tuple[nx.DiGraph, int]:
    """
    Remove an edge; also prune any endpoint nodes that become singletons.
    Returns the updated graph and the edit cost.

    Cost = 1 (edge) + |singleton endpoints created|
    """
    src, tgt = edge_to_delete

    if not G.has_edge(src, tgt):
        return G, 0

    new_G = G.copy()
    new_G.remove_edge(src, tgt)

    nodes_to_remove = [
        node for node in (src, tgt)
        if new_G.has_node(node)
        and (new_G.in_degree(node) + new_G.out_degree(node) == 0)
    ]

    new_G.remove_nodes_from(nodes_to_remove)
    cost = 1 + len(nodes_to_remove)

    return new_G, cost


# ─────────────────────────────────────────────────────────────
# Centrality helpers (used by compare.py)
# ─────────────────────────────────────────────────────────────

def compute_centrality(G: nx.DiGraph) -> dict:
    return {
        "degree":            dict(G.degree()),
        "betweenness_nodes": nx.betweenness_centrality(G),
        "betweenness_edges": nx.edge_betweenness_centrality(G),
    }


def centrality_scores(
    nodes: set[str],
    edges: set[tuple[str, str]],
    centrality: dict,
) -> dict:
    node_deg = [centrality["degree"].get(n, 0)            for n in nodes]
    node_btw = [centrality["betweenness_nodes"].get(n, 0) for n in nodes]
    edge_btw = [centrality["betweenness_edges"].get(e, 0) for e in edges]

    return {
        "mean_node_degree":      np.mean(node_deg) if node_deg else 0.0,
        "max_node_degree":       np.max(node_deg)  if node_deg else 0.0,
        "mean_node_betweenness": np.mean(node_btw) if node_btw else 0.0,
        "max_node_betweenness":  np.max(node_btw)  if node_btw else 0.0,
        "mean_edge_betweenness": np.mean(edge_btw) if edge_btw else 0.0,
        "max_edge_betweenness":  np.max(edge_btw)  if edge_btw else 0.0,
    }
