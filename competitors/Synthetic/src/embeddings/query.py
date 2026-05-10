"""Query helpers over the prebuilt HNSW node/edge indexes.

Provides KNN queries, ID/name -> embedding lookups, and the
most-similar / most-distant scans used to generate replacement candidates
for `find_counterfactuals` (within-type for nodes, across all edges for
edges).
"""

import hnswlib
import numpy as np

from sentence_transformers import SentenceTransformer
from src.embeddings.utils import load_index
from src.counterfactuals.utils import cosine_similarity_norm

EMB_MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384

model = SentenceTransformer(EMB_MODEL_NAME)


def query(index: hnswlib.Index, records: list[dict], vec: np.ndarray, k: int=5):
    vec = vec.astype("float16").reshape(1, -1)
    vec /= np.linalg.norm(vec)

    labels, distances = index.knn_query(vec, k=k)
    return [
        {**records[i], "similarity": round(1-float(d), 4)}
        for i, d in zip(labels[0], distances[0])
    ]

def build_lookup(records: list[dict]) -> dict[str, int]:
    lookup = {}
    for i, r in enumerate(records):
        lookup[r["id"]] = i
        if "name" in r:
            lookup[r["name"]] = i
        if "label" in r:
            # lookup[r["label"]] = i
            lookup[f'({r["src"]}, {r["tgt"]})'] = i
    return lookup

def build_edge_lookup(records: list[dict]) -> dict[str, int]:
    lookup = {}
    for i, r in enumerate(records):
        lookup[(r["src"], r["tgt"])] = i
    return lookup

def get_embedding(embeddings: np.ndarray, lookup: dict[str, int], key: str) -> np.ndarray:
    if key not in lookup:
        raise ValueError(f"Key not found in lookup: '{key}'")
    return embeddings[lookup[key]]

def find_most_similar_node(name, node_type, embeddings, lookup, type_index):
    vec = get_embedding(embeddings, lookup, name).astype("float32")

    same_type_nodes = [n for n in type_index.get(node_type, []) if n != name]

    best_node, best_sim = None, -1
    for node in same_type_nodes:
        if node not in lookup:
            continue
        candidate_vec = get_embedding(embeddings, lookup, node).astype("float32")
        sim = cosine_similarity_norm(vec, candidate_vec)
        if sim > best_sim:
            best_sim = sim
            best_node = node

    return {"name": best_node, "similarity": round(best_sim, 4)}

def find_most_distant_node(name, node_type, embeddings, lookup, type_index):
    vec = get_embedding(embeddings, lookup, name).astype("float32")

    same_type_nodes = [n for n in type_index.get(node_type, []) if n != name]

    worst_node, worst_sim = None, float("inf")
    for node in same_type_nodes:
        if node not in lookup:
            continue
        candidate_vec = get_embedding(embeddings, lookup, node).astype("float32")
        sim = cosine_similarity_norm(vec, candidate_vec)
        if sim < worst_sim:
            worst_sim = sim
            worst_node = node

    return {"name": worst_node, "similarity": round(worst_sim, 4)}

def find_most_similar_edge(edge, embeddings, lookup):
    vec = get_embedding(embeddings, lookup, edge)
    if vec is None:
        return None
    vec = vec.astype("float32")

    best_edge, best_sim = None, -1
    for candidate_edge in lookup:
        if candidate_edge == edge:
            continue
        candidate_vec = get_embedding(embeddings, lookup, candidate_edge).astype("float32")
        sim = cosine_similarity_norm(vec, candidate_vec)
        if sim > best_sim:
            best_sim  = sim
            best_edge = candidate_edge

    if best_edge is None:
        return None
    
    return {"edge": best_edge, "similarity": round(best_sim, 4)}

def find_most_distant_edge(edge, embeddings, lookup):
    vec = get_embedding(embeddings, lookup, edge)
    if vec is None:
        return None
    vec = vec.astype("float32")

    worst_edge, worst_sim = None, float("inf")
    for candidate_edge in lookup:
        if candidate_edge == edge:
            continue
        candidate_vec = get_embedding(embeddings, lookup, candidate_edge).astype("float32")
        sim = cosine_similarity_norm(vec, candidate_vec)
        if sim < worst_sim:
            worst_sim = sim
            worst_edge = candidate_edge

    if worst_edge is None:
        return None

    return {"edge": worst_edge, "similarity": round(worst_sim, 4)}


if __name__ == "__main__":
    index_prefix = "src/embeddings/node_index"
    index, records, embeddings = load_index(index_prefix, DIM, 2000)
    lookup = build_lookup(records)

    # # --- Run query ---

    # query_text = "Xylos offers exotic goods."
    # query_vec = model.encode([query_text], normalize_embeddings=True)[0].astype("float32")

    # results = query(index, records, query_vec, k=3)

    # print(f"Query: '{query_text}'\n")
    # for r in results:
    #     print(f"  [{r['similarity']:.4f}] {r}")

    # vec1 = get_embedding(embeddings, lookup, "exotic goods")
    # vec2 = get_embedding(embeddings, lookup, "Markets of Xylos")
    # print(f"\nEmbedding for 'exotic goods': shape={vec1.shape}, first 5 dims={vec1[:5]}")
    # print(f"\nEmbedding for 'Markets of Xylos': shape={vec2.shape}, first 5 dims={vec2[:5]}")

    # from src.counterfactuals.utils import cosine_similarity_norm

    # sim = cosine_similarity_norm(vec1, vec2)
    # print(f"Similarity: {sim}")

    import networkx as nx
    from collections import defaultdict

    G = nx.read_graphml("synthetic/graph_chunk_entity_relation.graphml")

    type_index = defaultdict(list)
    for node, data in G.nodes(data=True):
        node_type = data.get("entity_type")
        type_index[node_type].append(node)

    print(find_most_similar_node('Xylotian Sky-Skiff', 'artifact', embeddings, lookup, type_index))