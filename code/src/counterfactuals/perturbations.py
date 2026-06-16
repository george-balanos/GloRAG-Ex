import networkx as nx

def delete_node(context_graph: nx.Graph, node_to_delete: str):
    G = context_graph.copy()

    predecessors = list(G.predecessors(node_to_delete))
    successors = list(G.successors(node_to_delete))
    neighbors = predecessors + successors

    G.remove_node(node_to_delete)

    singletons = [n for n in neighbors if G.in_degree(n) + G.out_degree(n) == 0]
    G.remove_nodes_from(singletons)

    return G

def delete_edge(context_graph: nx.Graph, edge_to_delete: tuple):
    G = context_graph.copy()

    src = edge_to_delete[0]
    tgt = edge_to_delete[1]
    
    G.remove_edge(src, tgt)

    if G.in_degree(src) + G.out_degree(src) == 0:
        G.remove_node(src)
    if G.in_degree(tgt) + G.out_degree(tgt) == 0:
        G.remove_node(tgt)

    return G

def add_node(context_graph: nx.Graph, node: str, **attrs) -> nx.Graph:
    G = context_graph.copy()
    G.add_node(node, **attrs)
    return G
 
def add_edge(context_graph: nx.Graph, edge: tuple, **attrs) -> nx.Graph:
    G = context_graph.copy()
    src = edge[0]
    tgt = edge[1]
    G.add_edge(src, tgt, **attrs)
    return G