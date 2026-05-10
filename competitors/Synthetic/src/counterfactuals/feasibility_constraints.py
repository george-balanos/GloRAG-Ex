"""Feasibility constraints F1-F3 from local.tex sec. 1.4.

Pure predicates that the search calls to filter candidate edits before
applying them. F1 = schema, F2 = grounding, F3 = connectivity.
"""

import networkx as nx


def check_f1(schema_index: dict, src_type: str, label: str, tgt_type: str) -> bool:
    """F1: schema-compatible edge.

    schema_index maps (src_type, tgt_type) -> set of valid labels in G.
    Returns True iff (src_type, label, tgt_type) is in the schema.
    """
    valid_labels = schema_index.get((src_type, tgt_type))
    if not valid_labels:
        return False
    return label in valid_labels


def check_f2(G: nx.Graph, new_node: str = None, new_edge: tuple = None) -> bool:
    """F2: new nodes/edges must already exist in the knowledge graph G."""
    if new_node is not None and new_node not in G.nodes:
        return False
    if new_edge is not None and new_edge not in G.edges:
        return False
    return True


def check_f3(cand_node: str = None, cand_edge: tuple = None,
             cut_vertices: set = None, bridges: set = None,
             undirected: nx.Graph = None) -> bool:
    """F3: deletion does not split the graph into multiple non-trivial components.

    A cut vertex / bridge whose removal only strands singletons (cleaned up)
    is feasible; one that splits C into >=2 non-trivial components is not.
    """
    if cand_node is not None:
        if cut_vertices is None or cand_node not in cut_vertices:
            return True
        if undirected is None:
            return False
        neighbors = list(undirected.neighbors(cand_node))
        return not any(undirected.degree(n) > 1 for n in neighbors)

    if cand_edge is not None:
        if bridges is None or cand_edge not in bridges:
            return True
        if undirected is None:
            return False
        u, v = cand_edge[0], cand_edge[1]
        return not (undirected.degree(u) > 1 and undirected.degree(v) > 1)

    return True
