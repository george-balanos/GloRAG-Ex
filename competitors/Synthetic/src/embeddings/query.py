import hnswlib
import numpy as np

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
            lookup[r["label"]] = i
    return lookup

def get_embedding(embeddings: np.ndarray, lookup: dict[str, int], key: str) -> np.ndarray:
    if key not in lookup:
        raise ValueError(f"Key not found in lookup: '{key}'")
    return embeddings[lookup[key]]

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from src.embeddings.utils import load_index

    # --- Test setup ---

    MODEL_NAME = "all-MiniLM-L6-v2"
    DIM = 384

    model = SentenceTransformer(MODEL_NAME)

    index_prefix = "src/embeddings/node_index"
    index, records, embeddings = load_index(index_prefix, DIM, 2000)
    lookup = build_lookup(records)

    # --- Run query ---

    query_text = "Xylos offers exotic goods."
    query_vec = model.encode([query_text], normalize_embeddings=True)[0].astype("float32")

    results = query(index, records, query_vec, k=3)

    print(f"Query: '{query_text}'\n")
    for r in results:
        print(f"  [{r['similarity']:.4f}] {r}")

    vec1 = get_embedding(embeddings, lookup, "exotic goods")
    vec2 = get_embedding(embeddings, lookup, "Markets of Xylos")
    print(f"\nEmbedding for 'exotic goods': shape={vec1.shape}, first 5 dims={vec1[:5]}")
    print(f"\nEmbedding for 'Markets of Xylos': shape={vec2.shape}, first 5 dims={vec2[:5]}")

    from src.counterfactuals.utils import cosine_similarity_norm

    sim = cosine_similarity_norm(vec1, vec2)
    print(f"Similarity: {sim}")