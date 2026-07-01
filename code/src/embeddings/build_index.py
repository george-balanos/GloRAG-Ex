from src.embeddings.decoder import *
from src.embeddings.utils import save_index
from src.dataset_setup import DATASETS

import argparse
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

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_index",
        description="Build HNSW node/edge indices from a KG working directory.",
    )
    p.add_argument("--dataset", choices=DATASETS, default="synthetic",
                   help="Dataset name; reads KGs/<dataset>/vdb_*.json, writes src/embeddings/<dataset>/{node,edge}_index.*")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    dataset = args.dataset

    # node_json = f"KGs/{dataset}/vdb_entities.json"
    # edge_json = f"KGs/{dataset}/vdb_relationships.json"
    node_json = f"KGs/lightrag/{dataset}/vdb_entities.json"
    edge_json = f"KGs/lightrag/{dataset}/vdb_relationships.json"

    out_dir = f"src/embeddings/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    node_index, node_records, dim, node_embeddings = build_index(node_json)
    save_index(node_index, node_records, node_embeddings, f"{out_dir}/node_index")

    edge_index, edge_records, dim, edge_embeddings = build_index(edge_json)
    save_index(edge_index, edge_records, edge_embeddings, f"{out_dir}/edge_index")