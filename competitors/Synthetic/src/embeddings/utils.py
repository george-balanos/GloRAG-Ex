import hnswlib
import json
import numpy as np

# def save_index(index: hnswlib.Index, records: list, path_prefix: str):
#     index.save_index(f"{path_prefix}.bin")
#     with open(f"{path_prefix}_records.json", "w") as f:
#         json.dump(records, f)
#     print(f"Saved: {path_prefix}.bin + {path_prefix}_records.json")

def save_index(index: hnswlib.Index, records: list, embeddings: np.ndarray, path_prefix: str):
    index.save_index(f"{path_prefix}.bin")
    with open(f"{path_prefix}_records.json", "w") as f:
        json.dump(records, f)
    np.save(f"{path_prefix}_embeddings.npy", embeddings)
    print(f"Saved: {path_prefix}.bin + {path_prefix}_records.json + {path_prefix}_embeddings.npy")


# def load_index(path_prefix: str, dim: int, max_elements: int):
#     index = hnswlib.Index(space="cosine", dim=dim)
#     index.load_index(f"{path_prefix}.bin", max_elements=max_elements)
#     index.set_ef(50)
#     with open(f"{path_prefix}_records.json") as f:
#         records = json.load(f)
#     return index, records

def load_index(path_prefix: str, dim: int, max_elements: int) -> tuple[hnswlib.Index, list, np.ndarray]:
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(f"{path_prefix}.bin", max_elements=max_elements)
    index.set_ef(50)
    with open(f"{path_prefix}_records.json") as f:
        records = json.load(f)
    embeddings = np.load(f"{path_prefix}_embeddings.npy")
    return index, records, embeddings

if __name__ == "__main__":
    pass