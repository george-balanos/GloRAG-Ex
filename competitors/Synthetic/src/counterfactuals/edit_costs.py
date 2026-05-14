from src.counterfactuals.utils import cosine_similarity, cosine_similarity_norm
from src.embeddings.query import get_embedding

import networkx as nx

##################################### Semantic Costs ####################################

#### Delete ####

# def delete_edge_cost(context_graph: nx.Graph, edge_to_delete: tuple):
#     src = edge_to_delete[0]
#     tgt = edge_to_delete[1]

#     singletons = sum(1 for node in [src, tgt] if context_graph.degree(node) == 1)

#     return 1 + singletons

# # def delete_node_cost(C: nx.Graph, node_to_remove):
# #     incident_edges = list(C.edges(node_to_remove))

# #     return 1 + len(incident_edges)

# def delete_node_cost(C: nx.Graph, node_to_remove):
#     print(f"Node to remove: {node_to_remove}")

#     neighbors = list(C.neighbors(node_to_remove))
#     incident_edges = list(C.edges(node_to_remove))

#     print(f"Neighbors: {neighbors}")
#     print(f"Incident edges: {incident_edges}")

#     singleton_neighbors = [n for n in neighbors if C.degree(n) == 1]

#     return 1 + len(incident_edges) + len(singleton_neighbors)

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

    singletons_neighbors = [
        n for n in neighbors
        if context_graph.in_degree(n) + context_graph.out_degree(n) == 1
    ]

    return 1 + len(incident_edges) + len(singletons_neighbors)

#### Replace ####

def replace_edge_cost(edge_to_replace_emb, edge_replacement_emb):
    return 1 - cosine_similarity(edge_to_replace_emb, edge_replacement_emb)

def replace_node_cost(node_to_replace_emb, node_replacement_emb):
    return 1 - cosine_similarity(node_to_replace_emb, node_replacement_emb)

#### Add ####

def add_edge_cost(C: nx.DiGraph, edge_embeddings, edge_lookup, edge_to_add):
    src, tgt = edge_to_add

    edge_key = (src, tgt) if (src, tgt) in edge_lookup else (tgt, src) if (tgt, src) in edge_lookup else None
    
    if edge_key is None:
        return 1

    edge_to_add_emb = get_embedding(edge_embeddings, edge_lookup, edge_key)
    if edge_to_add_emb is None:
        return 1

    min_dist = float("inf")
    for u, v in C.edges:
        current_key = (u, v) if (u, v) in edge_lookup else (v, u) if (v, u) in edge_lookup else None
        if current_key is None:
            continue
        current_emb = get_embedding(edge_embeddings, edge_lookup, current_key)
        if current_emb is None:
            continue
        dist = 1 - cosine_similarity_norm(current_emb, edge_to_add_emb)
        if dist < min_dist:
            min_dist = dist

    return min_dist if min_dist != float("inf") else 1


def add_node_cost(C: nx.DiGraph, node_embeddings, node_lookup, edge_embeddings, edge_lookup, node_to_add):
    node_to_add_emb = get_embedding(node_embeddings, node_lookup, node_to_add)
    if node_to_add_emb is None:
        return 1  # default cost

    min_dist = float("inf")
    selected_node = list(C.nodes)[0]

    for node in C.nodes:
        current_emb = get_embedding(node_embeddings, node_lookup, node)
        if current_emb is None:
            continue
        dist = 1 - cosine_similarity_norm(current_emb, node_to_add_emb)
        if dist < min_dist:
            min_dist = dist
            selected_node = node

    for edge in C.edges(selected_node):
        min_dist += add_edge_cost(C, edge_embeddings, edge_lookup, edge)

    return min_dist


##################################### Unit Costs #####################################

#### Delete ####

def delete_edge_uc(context_graph: nx.Graph, edge_to_delete: tuple):
    src = edge_to_delete[0]
    tgt = edge_to_delete[1]

    singletons = sum(1 for node in [src, tgt] if context_graph.degree(node) == 1)

    return 1 + singletons

# def delete_node_uc(C: nx.Graph, node_to_remove):
#     incident_edges = list(C.edges(node_to_remove))

#     return 1 + len(incident_edges)

def delete_node_uc(C: nx.Graph, node_to_remove):
    neighbors = list(C.neighbors(node_to_remove))
    incident_edges = list(C.edges(node_to_remove))

    singleton_neighbors = [n for n in neighbors if C.degree(n) == 1]

    return 1 + len(incident_edges) + len(singleton_neighbors)

#### Replace #####

def replace_edge_uc():
    return 1

def replace_node_uc():
    return 1

#### Add ####

def add_edge_uc():
    return 1

def add_node_uc(C: nx.Graph, node_to_add):
    incident_edges = list(C.edges(node_to_add))

    return 1 + len(incident_edges)