import networkx as nx

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