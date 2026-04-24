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

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from src.embeddings.utils import load_index

    # --- Test setup ---

    MODEL_NAME = "all-MiniLM-L6-v2"
    DIM = 384

    model = SentenceTransformer(MODEL_NAME)

    index_prefix = "src/embeddings/node_index"
    index, records = load_index(index_prefix, DIM, 2000)

    # --- Run query ---

    query_text = "Xylos offers exotic goods."
    query_vec = model.encode([query_text], normalize_embeddings=True)[0].astype("float32")

    results = query(index, records, query_vec, k=3)

    print(f"Query: '{query_text}'\n")
    for r in results:
        print(f"  [{r['similarity']:.4f}] {r}")