"""Edit-operation cost functions used by the counterfactual search.

Implements the semantic costs from local.tex sec. 1.3 (deletion, replacement,
addition) and the unit-cost variant. All single-element edits cost at least 1,
so Dijkstra never extracts a free degenerate replacement.
"""

from src.counterfactuals.utils import cosine_similarity

import networkx as nx

##################################### Semantic Costs ####################################

def _undirected_neighbors(C: nx.Graph, node):
    if C.is_directed():
        return list(C.to_undirected().neighbors(node))
    return list(C.neighbors(node))

#### Delete ####

def delete_edge_cost(context_graph: nx.Graph, edge_to_delete: tuple):
    src = edge_to_delete[0]
    tgt = edge_to_delete[1]

    singletons = sum(1 for node in [src, tgt] if context_graph.degree(node) == 1)

    return 1 + singletons

def delete_node_cost(C: nx.Graph, node_to_remove):
    neighbors = _undirected_neighbors(C, node_to_remove)
    incident_edges = list(C.edges(node_to_remove))

    singleton_neighbors = [n for n in neighbors if C.degree(n) == 1]

    return 1 + len(incident_edges) + len(singleton_neighbors)

#### Replace ####

def replace_edge_cost(edge_to_replace_emb, edge_replacement_emb):
    d_sem = 1 - cosine_similarity(edge_to_replace_emb, edge_replacement_emb)
    return 1 + d_sem

def replace_node_cost(node_to_replace_emb, node_replacement_emb, C: nx.Graph = None, node_to_replace=None):
    d_sem = 1 - cosine_similarity(node_to_replace_emb, node_replacement_emb)
    if C is None or node_to_replace is None:
        return 1 + d_sem
    incident_edges = list(C.edges(node_to_replace))
    return 1 + len(incident_edges) + d_sem

#### Add ####

def add_edge_cost(C: nx.Graph, edge_index, edge_to_add_emb):
    if not C.edges:
        return 1.0
    min_dist = float("inf")

    for edge in C.edges:
        current_emb = edge_index.get_items([edge])
        dist = 1 - cosine_similarity(current_emb, edge_to_add_emb)
        if dist < min_dist:
            min_dist = dist

    return 1 + min_dist

def add_node_cost(C: nx.Graph, node_index, edge_index, node_to_add_emb, connecting_edge_embs=None):
    min_dist = float("inf")

    for node in C.nodes:
        current_emb = node_index.get_items([node])
        dist = 1 - cosine_similarity(current_emb, node_to_add_emb)
        if dist < min_dist:
            min_dist = dist

    total = 1 + min_dist

    if connecting_edge_embs:
        for emb in connecting_edge_embs:
            total += add_edge_cost(C, edge_index, emb)

    return total


##################################### Unit Costs #####################################

#### Delete ####

def delete_edge_uc(context_graph: nx.Graph, edge_to_delete: tuple):
    src = edge_to_delete[0]
    tgt = edge_to_delete[1]

    singletons = sum(1 for node in [src, tgt] if context_graph.degree(node) == 1)

    return 1 + singletons

def delete_node_uc(C: nx.Graph, node_to_remove):
    neighbors = _undirected_neighbors(C, node_to_remove)
    incident_edges = list(C.edges(node_to_remove))

    singleton_neighbors = [n for n in neighbors if C.degree(n) == 1]

    return 1 + len(incident_edges) + len(singleton_neighbors)

#### Replace #####

def replace_edge_uc():
    return 1

def replace_node_uc(C: nx.Graph = None, node_to_replace=None):
    if C is None or node_to_replace is None:
        return 1
    incident_edges = list(C.edges(node_to_replace))
    return 1 + len(incident_edges)

#### Add ####

def add_edge_uc():
    return 1

def add_node_uc(C: nx.Graph, connecting_edges=None):
    n_edges = len(connecting_edges) if connecting_edges else 0
    return 1 + n_edges
