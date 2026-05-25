"""Edit-operation cost functions used by the counterfactual search.

Implements the semantic costs from local.tex sec. 1.3 (deletion, replacement,
addition) and the unit-cost variant. All single-element edits cost at least 1,
so Dijkstra never extracts a free degenerate replacement.
"""

from src.counterfactuals.utils import cosine_similarity, cosine_similarity_norm
from src.embeddings.query import get_embedding

import networkx as nx

##################################### Semantic Costs ####################################

#### Delete ####

def delete_edge_cost(context_graph: nx.Graph, edge_to_delete: tuple):
    src = edge_to_delete[0]
    tgt = edge_to_delete[1]

    singletons = sum(
        1 for node in [src, tgt]
        if context_graph.in_degree(node) + context_graph.out_degree(node) == 1
    )

    return 1 + singletons


def delete_node_cost(context_graph: nx.Graph, node_to_remove):
    predecessors = list(context_graph.predecessors(node_to_remove))
    successors = list(context_graph.successors(node_to_remove))
    neighbors = predecessors + successors

    incident_edges = list(context_graph.in_edges(node_to_remove)) + list(context_graph.out_edges(node_to_remove))

    singleton_neighbors = [
        n for n in neighbors
        if context_graph.in_degree(n) + context_graph.out_degree(n) == 1
    ]

    return 1 + len(incident_edges) + len(singleton_neighbors)


#### Replace ####

def replace_edge_cost(edge_to_replace_emb, edge_replacement_emb):
    d_sem = 1 - cosine_similarity(edge_to_replace_emb, edge_replacement_emb)
    return 1 + d_sem


def replace_node_cost(node_to_replace_emb, node_replacement_emb, C: nx.Graph = None, node_to_replace=None):
    d_sem = 1 - cosine_similarity(node_to_replace_emb, node_replacement_emb)
    if C is None or node_to_replace is None:
        return 1 + d_sem
    incident_edges = list(C.in_edges(node_to_replace)) + list(C.out_edges(node_to_replace)) if C.is_directed() else list(C.edges(node_to_replace))
    return 1 + len(incident_edges) + d_sem


#### Add ####

def add_edge_cost(C: nx.DiGraph, edge_embeddings, edge_lookup, edge_to_add):
    """Cost = 1 + min_{e in E_C} d_sem(e_to_add, e). Falls back to 1.0 if
    embeddings can't be located for any side."""
    if edge_to_add is None:
        return 1.0
    src, tgt = edge_to_add
    edge_key = (src, tgt) if (src, tgt) in edge_lookup else (tgt, src) if (tgt, src) in edge_lookup else None
    if edge_key is None:
        return 1.0

    edge_to_add_emb = get_embedding(edge_embeddings, edge_lookup, edge_to_add)
    if edge_to_add_emb is None:
        return 1.0

    min_dist = float("inf")
    
    for edge in C.edges:
        if edge == edge_to_add:
            continue

        current_emb = get_embedding(edge_embeddings, edge_lookup, edge)
        if current_emb is None:
            continue

        dist = 1 - cosine_similarity_norm(current_emb, edge_to_add_emb)
        if dist < min_dist:
            min_dist = dist

    if min_dist == float("inf"):
        min_dist = 1.0
    return 1 + min_dist


def add_node_cost(C: nx.DiGraph, node_embeddings, node_lookup, edge_embeddings, edge_lookup, node_to_add, connecting_edges=None):
    """Cost = 1 + min_{v in V_C} d_sem(node_to_add, v) + Σ_{e' in E_{v'}} w(add_e(e')).
    The unit floors compose: this returns ≥ 1 for the node alone, plus the
    add_edge_cost (which is itself ≥ 1) for each connecting edge."""
    node_to_add_emb = get_embedding(node_embeddings, node_lookup, node_to_add)

    if node_to_add_emb is None:
        # No embedding -> fallback unit cost
        node_dist = 1.0
    else:
        min_dist = float("inf")
        for node in C.nodes:
            if node == node_to_add:
                continue
            current_emb = get_embedding(node_embeddings, node_lookup, node)
            if current_emb is None:
                continue
            dist = 1 - cosine_similarity_norm(current_emb, node_to_add_emb)
            if dist < min_dist:
                min_dist = dist
        node_dist = min_dist if min_dist != float("inf") else 1.0

    total = 1 + node_dist
    # If caller didn't supply connecting edges, derive them from C around node_to_add.
    if connecting_edges is None:
        connecting_edges = list(C.in_edges(node_to_add)) + list(C.out_edges(node_to_add))
    for edge in connecting_edges:
        total += add_edge_cost(C, edge_embeddings, edge_lookup, edge)
    return total


##################################### Unit Costs #####################################

#### Delete ####

def delete_edge_uc(context_graph: nx.Graph, edge_to_delete: tuple):
    src = edge_to_delete[0]
    tgt = edge_to_delete[1]

    singletons = sum(
        1 for node in [src, tgt]
        if context_graph.in_degree(node) + context_graph.out_degree(node) == 1
    )

    return 1 + singletons


def delete_node_uc(context_graph: nx.Graph, node_to_remove):
    predecessors = list(context_graph.predecessors(node_to_remove))
    successors = list(context_graph.successors(node_to_remove))
    neighbors = predecessors + successors

    incident_edges = list(context_graph.in_edges(node_to_remove)) + list(context_graph.out_edges(node_to_remove))

    singleton_neighbors = [
        n for n in neighbors
        if context_graph.in_degree(n) + context_graph.out_degree(n) == 1
    ]

    return 1 + len(incident_edges) + len(singleton_neighbors)


#### Replace #####

def replace_edge_uc():
    return 1


def replace_node_uc(C: nx.Graph = None, node_to_replace=None):
    if C is None or node_to_replace is None:
        return 1
    if C.is_directed():
        incident_edges = list(C.in_edges(node_to_replace)) + list(C.out_edges(node_to_replace))
    else:
        incident_edges = list(C.edges(node_to_replace))
    return 1 + len(incident_edges)


#### Add ####

def add_edge_uc():
    return 1


def add_node_uc(C: nx.Graph = None, connecting_edges=None):
    n_edges = len(connecting_edges) if connecting_edges else 0
    return 1 + n_edges
