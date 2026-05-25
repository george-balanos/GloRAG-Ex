"""Edit-operation cost functions used by the counterfactual search.

Implements the semantic costs from local.tex sec. 1.3 (deletion, replacement,
addition) and the unit-cost variant. All single-element edits cost at least 1
(unit floor), so Dijkstra never extracts a free degenerate replacement.

Addition costs ship in four flavours, switched by ``mode`` on the dispatchers
``add_node_cost_for`` and ``add_edge_cost_for``:

* ``unit``    — paper-exact unit costs: ``w(add_n) = 2``, ``w(add_e) = 1``.
* ``query``   — semantic distance to the **query** embedding: rewards adding
                content the user is asking about.
* ``context`` — semantic distance to the nearest CG element: rewards
                additions that are coherent with the current context.
                Matches local.tex Eq. cost-add-n.
* ``mix``     — convex blend ``α · query + (1−α) · context`` for ``α ∈ [0, 1]``,
                lets the search trade off query-relevance vs CG-proximity.

``add_n`` is single-edge per the paper: each call is associated with the
specific grounded edge ``e' = edge_to_add`` that attaches the new node to V_C.
The historical Σ-form is no longer used.
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


#### Add (context-distance variant — paper Eq. cost-add-e / cost-add-n) ####

def add_edge_cost(C: nx.DiGraph, edge_embeddings, edge_lookup, edge_to_add):
    """w(add_e) = 1 + min_{e ∈ E_C} d_sem(e', e). Falls back to 1.0 when no
    embedding is available for the candidate or no E_C edge has one."""
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


def add_node_cost(C: nx.DiGraph, node_embeddings, node_lookup, edge_embeddings, edge_lookup, node_to_add, edge_to_add):
    """w(add_n(v', e')) = 1 + min_{v ∈ V_C} d_sem(v', v) + w(add_e(e')).

    Single-edge form per local.tex Eq. cost-add-n / commit fbd8b86. The caller
    is responsible for choosing the specific grounded edge ``e' = edge_to_add``
    that will attach v' to V_C this step."""
    node_to_add_emb = get_embedding(node_embeddings, node_lookup, node_to_add)
    if node_to_add_emb is None:
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

    return 1 + node_dist + add_edge_cost(C, edge_embeddings, edge_lookup, edge_to_add)


#### Add (query-distance variant) ####

def add_edge_cost_query(edge_embeddings, edge_lookup, edge_to_add, query_embedding):
    """w_q(add_e) = 1 + d_sem(φ(e'), φ(q)). Distance to the query embedding."""
    if edge_to_add is None or query_embedding is None:
        return 1.0
    edge_to_add_emb = get_embedding(edge_embeddings, edge_lookup, edge_to_add)
    if edge_to_add_emb is None:
        return 1.0
    d_sem = 1 - cosine_similarity_norm(edge_to_add_emb, query_embedding)
    return 1 + d_sem


def add_node_cost_query(node_embeddings, node_lookup, edge_embeddings, edge_lookup,
                        node_to_add, edge_to_add, query_embedding):
    """w_q(add_n(v', e')) = 1 + d_sem(φ(v'), φ(q)) + w_q(add_e(e'))."""
    node_to_add_emb = get_embedding(node_embeddings, node_lookup, node_to_add)
    if node_to_add_emb is None or query_embedding is None:
        node_dist = 1.0
    else:
        node_dist = 1 - cosine_similarity_norm(node_to_add_emb, query_embedding)
    return 1 + node_dist + add_edge_cost_query(edge_embeddings, edge_lookup, edge_to_add, query_embedding)


#### Add (dispatcher) ####

def add_edge_cost_for(mode, C, edge_embeddings, edge_lookup, edge_to_add,
                     query_embedding=None, alpha=0.5):
    """Dispatcher for w(add_e) under {unit, query, context, mix}."""
    if mode == "unit":
        return add_edge_uc()
    if mode == "context":
        return add_edge_cost(C, edge_embeddings, edge_lookup, edge_to_add)
    if mode == "query":
        return add_edge_cost_query(edge_embeddings, edge_lookup, edge_to_add, query_embedding)
    if mode == "mix":
        w_q = add_edge_cost_query(edge_embeddings, edge_lookup, edge_to_add, query_embedding)
        w_c = add_edge_cost(C, edge_embeddings, edge_lookup, edge_to_add)
        return alpha * w_q + (1.0 - alpha) * w_c
    raise ValueError(f"Unknown add_cost mode: {mode!r}")


def add_node_cost_for(mode, C, node_embeddings, node_lookup, edge_embeddings, edge_lookup,
                     node_to_add, edge_to_add, query_embedding=None, alpha=0.5):
    """Dispatcher for w(add_n) under {unit, query, context, mix}."""
    if mode == "unit":
        return add_node_uc()
    if mode == "context":
        return add_node_cost(C, node_embeddings, node_lookup, edge_embeddings, edge_lookup,
                             node_to_add, edge_to_add)
    if mode == "query":
        return add_node_cost_query(node_embeddings, node_lookup, edge_embeddings, edge_lookup,
                                   node_to_add, edge_to_add, query_embedding)
    if mode == "mix":
        w_q = add_node_cost_query(node_embeddings, node_lookup, edge_embeddings, edge_lookup,
                                  node_to_add, edge_to_add, query_embedding)
        w_c = add_node_cost(C, node_embeddings, node_lookup, edge_embeddings, edge_lookup,
                            node_to_add, edge_to_add)
        return alpha * w_q + (1.0 - alpha) * w_c
    raise ValueError(f"Unknown add_cost mode: {mode!r}")


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


def add_node_uc(*_args, **_kwargs):
    """Paper-exact: w(add_n) = 2 (one for the node + one for its attaching edge,
    accounting separately in the ops list). Args ignored for back-compat."""
    return 2
