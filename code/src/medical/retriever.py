from src.embeddings.query import build_lookup, load_index, DIM, model, query

import networkx as nx
import numpy as np

def find_similar_node_id(index, records, entity: str) -> dict:
    vec = model.encode([entity], normalize_embeddings=True)[0].astype("float32")
    most_similar = query(index, records, vec, k=1)
    if most_similar[0]["similarity"] > 0.7:
        return most_similar[0]["id"]
    return {}

def validate_entity(G: nx.DiGraph, entities: list) -> dict:
    validated_entities = {
        "found": [],
        "not_found": []
    } 

    entities = [f"{e["entity_name"].lower()}|{e["entity_category"]}" for e in entities]
    for ent in entities:
        if ent in G:
            validated_entities["found"].append(ent)
        else:
            validated_entities["not_found"].append(ent)

    return validated_entities

def bfs_subgraph(G: nx.DiGraph, seed_nodes: list, depth: int = 2) -> nx.DiGraph:
    visited = set()
    for seed in seed_nodes:
        if seed not in G:
            continue
        bfs_nodes = nx.bfs_tree(G, seed, depth_limit=depth).nodes()
        visited.update(bfs_nodes)

    return nx.DiGraph(G.subgraph(visited))  # <-- force DiGraph here too

def shortest_paths_subgraph(G: nx.DiGraph, seed_nodes: list) -> nx.DiGraph:
    visited = set(seed_nodes)

    pairs = [(u, v) for i, u in enumerate(seed_nodes) for v in seed_nodes[i+1:] if u in G and v in G]

    for u, v in pairs:
        for source, target in [(u, v), (v, u)]: 
            try:
                path = nx.shortest_path(G, source=source, target=target)
                visited.update(path)
            except nx.NetworkXNoPath:
                pass

    return nx.DiGraph(G.subgraph(visited))

def prune_subgraph(
    subgraph: nx.DiGraph,
    query_text: str,
    lookup,
    embeddings,
    top_k_nodes: int = 20,
    top_k_edges: int = 30,
) -> nx.DiGraph:
    query_vec = model.encode([query_text], normalize_embeddings=True)[0].astype("float32")

    # --- score nodes in one matrix op ---
    nodes = [n for n in subgraph.nodes() if n in lookup]
    if nodes:
        node_matrix = np.stack([embeddings[lookup[n]].astype("float32") for n in nodes])
        node_sims   = node_matrix @ query_vec  # (N,) — already normalized
        top_idx     = np.argpartition(node_sims, -min(top_k_nodes, len(nodes)))[-top_k_nodes:]
        top_nodes   = {nodes[i] for i in top_idx}
    else:
        top_nodes = set()

    # --- score edges in one batched encode call ---
    edge_list  = [(src, tgt) for src, tgt, _ in subgraph.edges(data=True)]
    edge_data  = [data for _, _, data in subgraph.edges(data=True)]
    labels = [d.get("relation", f"{s} {t}") for (s, t), d in zip(edge_list, edge_data)]

    if labels:
        edge_matrix = model.encode(labels, normalize_embeddings=True,
                                   batch_size=256, show_progress_bar=False).astype("float32")
        edge_sims   = edge_matrix @ query_vec  # (E,)
        top_idx     = np.argpartition(edge_sims, -min(top_k_edges, len(labels)))[-top_k_edges:]
        top_edges   = {edge_list[i] for i in top_idx}
    else:
        top_edges = set()

    # --- union ---
    edge_nodes  = {n for src, tgt in top_edges for n in (src, tgt)}
    final_nodes = top_nodes | edge_nodes

    pruned = nx.DiGraph(subgraph.subgraph(final_nodes))
    pruned.remove_edges_from([(u, v) for u, v in pruned.edges() if (u, v) not in top_edges])

    return pruned

if __name__ == "__main__":
    query_text = "Bedside Assessment"
    
    print(find_similar_node_id(query_text))

