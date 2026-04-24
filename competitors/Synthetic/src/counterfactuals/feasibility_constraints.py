from src.counterfactuals.perturbations import delete_node

import networkx as nx

### Feasibility Constraints
# F1: Schema Compatibility
# F2: Graph Grounding
# F3: Connectivity Preservation

def check_f1(schema: dict, src, l, tgt):
    if l not in schema:
        return False
    src_type, tgt_type = schema[l]
    return src == src_type and tgt == tgt_type

def check_f2(G: nx.Graph, new_node: str=None, new_edge: tuple=None):
    if new_node is not None:
        return new_node in G.nodes 
    if new_edge is not None:
        return new_edge in G.edges

def check_f3(cand_node: str=None, cand_edge: tuple=None, cut_vertices: set=None, bridges: set=None):
    if cand_node is not None and cut_vertices:
        return cand_node in cut_vertices
    if cand_edge is not None and bridges:
        return cand_edge in bridges


# def check_isolates_constraint(context_graph: nx.Graph, perturbed_graph: nx.Graph):
#     '''Check if perturbation introduces new isolated nodes.'''

#     original_isolated = set(nx.isolates(context_graph))
#     perturbed_isolated = set(nx.isolates(perturbed_graph))

#     return perturbed_isolated.issubset(original_isolated)

# def check_connectivity_constraint(context_graph: nx.Graph, perturbed_graph: nx.Graph):
#     '''Check if perturbation further splits the graph into more components.'''

#     original_components = nx.number_connected_components(context_graph)
#     perturbed_components = nx.number_connected_components(perturbed_graph)

#     return perturbed_components == original_components

# if __name__ == "__main__":
#     import networkx as nx

#     G = nx.path_graph(4)
#     G.add_node(10)  

#     print("ORIGINAL GRAPH")
#     print(f"Nodes: {list(G.nodes)}")
#     print(f"Edges: {list(G.edges)}")
#     print(f"Isolates: {list(nx.isolates(G))}")
#     print(f"Connected components: {nx.number_connected_components(G)}")

#     print("-" * 50)

#     perturbed_G = delete_node(G, 0)

#     print("PERTURBED GRAPH")
#     print(f"Nodes: {list(perturbed_G.nodes)}")
#     print(f"Edges: {list(perturbed_G.edges)}")
#     print(f"Isolates: {list(nx.isolates(perturbed_G))}")
#     print(f"Connected components: {nx.number_connected_components(perturbed_G)}")

#     print("-" * 50)

#     print(f"Isolates constraint OK? {check_isolates_constraint(G, perturbed_G)}")
#     print(f"Connectivity constraint OK? {check_connectivity_constraint(G, perturbed_G)}")