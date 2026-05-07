from src.counterfactuals.utils import cosine_similarity

import networkx as nx

##################################### Semantic Costs ####################################

#### Delete ####

def delete_edge_cost(context_graph: nx.Graph, edge_to_delete: tuple):
    src = edge_to_delete[0]
    tgt = edge_to_delete[1]

    singletons = sum(1 for node in [src, tgt] if context_graph.degree(node) == 1)

    return 1 + singletons

# def delete_node_cost(C: nx.Graph, node_to_remove):
#     incident_edges = list(C.edges(node_to_remove))

#     return 1 + len(incident_edges)

def delete_node_cost(C: nx.Graph, node_to_remove):
    neighbors = list(C.neighbors(node_to_remove))
    incident_edges = list(C.edges(node_to_remove))

    singleton_neighbors = [n for n in neighbors if C.degree(n) == 1]

    return 1 + len(incident_edges) + len(singleton_neighbors)

#### Replace ####

def replace_edge_cost(edge_to_replace_emb, edge_replacement_emb):
    return 1 - cosine_similarity(edge_to_replace_emb, edge_replacement_emb)

def replace_node_cost(node_to_replace_emb, node_replacement_emb):
    return 1 - cosine_similarity(node_to_replace_emb, node_replacement_emb)

#### Add ####

def add_edge_cost(C: nx.Graph, edge_index, edge_to_add_emb): 
    min_dist = float("inf")

    for edge in C.edges:
        current_emb = edge_index.get_items([edge])
        dist = 1 - cosine_similarity(current_emb, edge_to_add_emb)
        if dist < min_dist:
            min_dist = dist

    return min_dist

def add_node_cost(C: nx.Graph, node_index, edge_index, node_to_add_emb):
    min_dist = float("inf")
    selected_node = list(C.nodes)[0]

    for node in C.nodes:
        current_emb = node_index.get_items([node])
        dist = 1 - cosine_similarity(current_emb, node_to_add_emb)
        if dist < min_dist:
            min_dist = dist
            selected_node = node

    incident_edges = list(C.edges(selected_node))

    for edge in incident_edges:
        edge_emb = edge_index.get_items([edge])
        min_dist += add_edge_cost(C, edge_index, edge_emb)

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