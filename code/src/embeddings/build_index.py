from src.embeddings.decoder import *
from src.embeddings.utils import save_index

import json
import os
import numpy as np
import hnswlib

def build_index(json_path: str):
    with open(json_path) as f:
        data = json.load(f)

    dim = data["embedding_dim"]
    items = data["data"]

    embeddings = np.stack([decode_vector(item["vector"], dim) for item in items])

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-10, None)

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(items), ef_construction=200, M=16)
    index.add_items(embeddings, ids=np.arange(len(items)))
    index.set_ef(50)

    records = []
    for item in items:
        record = {"id": item["__id__"], "content": item.get("content", "")}

        if "entity_name" in item:
            record["name"] = item["entity_name"]
        else:
            record["src"] = item.get("src_id", "")
            record["tgt"] = item.get("tgt_id", "")
            record["label"] = item.get("content", "")
        
        records.append(record)

    print(f"Loaded {len(items)} |  dim={dim}")
    return index, records, dim, embeddings

if __name__ == "__main__":
    # `dataset` can be overridden via env var; vdb_*.json live in the LightRAG storage dir
    # for this dataset, alongside the graphml.
    dataset = os.environ.get("DATASET", "synthetic")  # "synthetic" or "hotpotqa"

    node_json = f"KGs/lightrag/{dataset}/vdb_entities.json"
    edge_json = f"KGs/lightrag/{dataset}/vdb_relationships.json"

    out_dir = f"src/embeddings/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    node_index, node_records, dim, node_embeddings = build_index(node_json)
    save_index(node_index, node_records, node_embeddings, f"{out_dir}/node_index")

    edge_index, edge_records, dim, edge_embeddings = build_index(edge_json)
    save_index(edge_index, edge_records, edge_embeddings, f"{out_dir}/edge_index")