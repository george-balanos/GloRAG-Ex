"""Feasibility constraints F1-F3 from local.tex sec. 1.4.

Pure predicates that the search calls to filter candidate edits before
applying them. F1 = schema, F2 = grounding, F3 = connectivity.
"""

import networkx as nx


def check_f1(schema_index: dict, src_type: str, label: str, tgt_type: str,
             mode: str = "type-only") -> bool:
    """F1: schema-compatible edge.

    schema_index maps (src_type, tgt_type) -> set of labels seen in G for that
    type pair.

    mode:
      * "off"          — always pass (skip F1)
      * "type-only"    — pass iff (src_type, tgt_type) appears in the schema;
                         label is ignored. Right default for synthetic datasets
                         where edge labels are free-form metadata.
      * "strict-label" — original closed-vocabulary check: pass iff the exact
                         label is in schema_index[(src_type, tgt_type)].
    """
    if mode == "off":
        return True

    valid_labels = schema_index.get((src_type, tgt_type))
    if not valid_labels:
        return False

    if mode == "type-only":
        return True
    if mode == "strict-label":
        return label in valid_labels

    raise ValueError(f"Unknown F1 mode: {mode!r}")


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
    """F3 (relaxed): C' may become disconnected.

    The connectivity requirement from local.tex sec. 1.4 has been dropped:
    deletions are no longer rejected for splitting C into multiple non-trivial
    components. The singleton sweep in `perturbations.delete_node` /
    `delete_edge` removes any degree-0 isolates left behind, and the cost
    formulas in `edit_costs` charge +1 per swept singleton, so the cost stays
    honest about the cleanup work.

    The signature is preserved (parameters `cut_vertices`, `bridges`,
    `undirected` are now unused) so existing call sites in `generate.py` and
    `pivotal_star_probe` keep their imports intact.
    """
    return True
