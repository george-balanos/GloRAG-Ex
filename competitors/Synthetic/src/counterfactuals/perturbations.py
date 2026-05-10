"""Pure graph-mutation primitives for the six edit operations.

Each function returns a fresh copy of the input graph; callers do not need to
copy. Singleton cleanup after delete_node / delete_edge mirrors the cleanup
rule in local.tex sec. 1.2.
"""

import networkx as nx

# def delete_node(context_graph: nx.Graph, node_to_delete: str):
#     G = context_graph.copy()

#     G.remove_node(node_to_delete)
#     return G

def delete_node(context_graph: nx.Graph, node_to_delete: str):
    G = context_graph.copy()
    
    neighbors = list(G.neighbors(node_to_delete))
    G.remove_node(node_to_delete)
    
    singletons = [n for n in neighbors if G.degree(n) == 0]
    G.remove_nodes_from(singletons)
    
    return G

def delete_edge(context_graph: nx.Graph, edge_to_delete: tuple):
    G = context_graph.copy()

    src = edge_to_delete[0]
    tgt = edge_to_delete[1]

    G.remove_edge(src, tgt)

    if G.degree(src) == 0:
        G.remove_node(src)
    if G.degree(tgt) == 0:
        G.remove_node(tgt)

    return G
    
def replace_node(context_graph: nx.Graph, old_name: str, new_name: str, **new_attrs) -> nx.Graph:
    G = context_graph.copy()
    G = nx.relabel_nodes(G, {old_name: new_name})
    
    G.nodes[new_name].clear()
    if new_attrs:
        G.nodes[new_name].update(new_attrs)
    return G

def replace_edge(context_graph: nx.Graph, edge_to_replace: tuple, edge_replacement: tuple, **new_attrs) -> nx.Graph:
    G = context_graph.copy()
    u, v = edge_to_replace
    
    G.edges[u, v].clear()
    if new_attrs:
        G.edges[u, v].update(new_attrs)
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
    
if __name__ == "__main__":
    G = nx.path_graph(4)

    print(f"G: {G}")
    
    edges = nx.edges(G)
    print(f"Edges: {edges}")

    nodes = nx.nodes(G)
    print(f"Nodes: {nodes}")

    ########### Delete node
    print(f"-"*40)

    node_to_delete = 2
    print(f"Node to delete {node_to_delete}")

    new_G = delete_node(G, node_to_delete)
    print(f"New G: {new_G}")

    edges = nx.edges(new_G)
    print(f"Edges: {edges}")

    nodes = nx.nodes(new_G)
    print(f"Nodes: {nodes}")

    ########### Delete edge
    print(f"-"*40)
    
    edge_to_delete = (0,1)
    print(f"Edge to delete {edge_to_delete}")

    new_G = delete_edge(G, edge_to_delete)
    print(f"New G: {new_G}")

    edges = nx.edges(new_G)
    print(f"Edges: {edges}")

    nodes = nx.nodes(new_G)
    print(f"Nodes: {nodes}")

    ########### Replace node
    print(f"-"*40)
    
    node_to_replace = 1
    node_replacement = 10
    print(f"Node to replace {node_to_replace}")

    new_G = replace_node(G, node_to_replace, node_replacement)
    print(f"New G: {new_G}")

    edges = nx.edges(new_G)
    print(f"Edges: {edges}")

    nodes = nx.nodes(new_G)
    print(f"Nodes: {nodes}")

    ########### Replace edge
    print(f"-"*40)
    
    edge_to_replace = (1,2)
    edge_replacement = "new description here"
    print(f"Edge to replace {edge_to_replace} with {edge_replacement}")

    new_G = replace_edge(G, edge_to_replace, edge_replacement)
    print(f"New G: {new_G}")

    edges = nx.edges(new_G)
    print(f"Edges: {edges}")

    nodes = nx.nodes(new_G)
    print(f"Nodes: {nodes}")

    for u, v, data in new_G.edges(data=True):
        print(f"  ({u}, {v}): {data.get('description', 'N/A')}")

    ########### Add node
    print("-" * 40)
 
    node_to_add = 99
    print(f"Node to add: {node_to_add}")
 
    new_G = add_node(G, node_to_add, description="a brand new node")
    print(f"New G: {new_G}")
    print(f"Edges: {nx.edges(new_G)}")
    print(f"Nodes: {nx.nodes(new_G)}")
    print(f"  Node {node_to_add} attrs: {new_G.nodes[node_to_add]}")
 
    ########### Add edge
    print("-" * 40)
 
    edge_to_add = (0, 3)
    print(f"Edge to add: {edge_to_add}")
 
    new_G = add_edge(G, edge_to_add, description="a brand new edge")
    print(f"New G: {new_G}")
    print(f"Edges: {nx.edges(new_G)}")
    print(f"Nodes: {nx.nodes(new_G)}")
    for u, v, data in new_G.edges(data=True):
        print(f"  ({u}, {v}): {data.get('description', 'N/A')}")