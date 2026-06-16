import json
import time
import hashlib
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
DIM = 384

import base64, zlib

def encode_vector(vec: np.ndarray) -> str:
    compressed = zlib.compress(vec.astype(np.float16).tobytes())
    return base64.b64encode(compressed).decode()

GRAPH_PATH    = "KGs/medical/graph_chunk_entity_relation_digraph.graphml"
VDB_ENTITIES  = "KGs/medical/vdb_entities.json"
VDB_RELATIONS = "KGs/medical/vdb_relationships.json"


def build_entity_vdb(G: nx.Graph):
    records = []
    texts   = []

    for node, data in tqdm(G.nodes(data=True), total=G.number_of_nodes(), desc="Collecting entities"):
        text = f"{node} {data.get('entity_type', '')} {data.get('description', '')}"
        texts.append(text)
        records.append({"node": node, "data": data})

    print(f"Encoding {len(texts)} entities...")
    vectors = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)

    output = []
    for rec, vec in tqdm(zip(records, vectors), total=len(records), desc="Building entity records"):
        vec = vec.astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-10
        output.append({
            "__id__":          rec["node"],
            "entity_name":     rec["node"],
            "entity_category": rec["data"].get("entity_type", ""),
            "content":         rec["data"].get("description", ""),
            "vector":          encode_vector(vec),
        })

    with open(VDB_ENTITIES, "w") as f:
        json.dump({"embedding_dim": DIM, "data": output}, f)
    print(f"Wrote {len(output)} entities → {VDB_ENTITIES}")


def build_relationship_vdb(G: nx.Graph):
    records = []
    texts   = []

    for src, tgt, data in tqdm(G.edges(data=True), total=G.number_of_edges(), desc="Collecting relationships"):
        rel = data.get("relation_name", data.get("label", "related_to"))
        texts.append(f"{src} {rel} {tgt}")
        records.append({"src": src, "tgt": tgt, "rel": rel, "data": data})

    print(f"Encoding {len(texts)} relationships...")
    vectors = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)

    output = []
    for rec, vec in tqdm(zip(records, vectors), total=len(records), desc="Building relationship records"):
        vec = vec.astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-10
        key = f"{rec['src']}|{rec['tgt']}|{rec['rel']}"
        output.append({
            "__id__":         "rel-" + hashlib.md5(key.encode()).hexdigest(),
            "__created_at__": int(time.time()),
            "src_id":         rec["src"],
            "tgt_id":         rec["tgt"],
            "source_id":      rec["data"].get("source_id", ""),
            "content":        f"{rec['rel']}\t{rec['src']}\n{rec['tgt']}\n{rec['data'].get('description', '')}",
            "file_path":      rec["data"].get("file_path", "unknown_source"),
            "vector":         encode_vector(vec),
        })

    with open(VDB_RELATIONS, "w") as f:
        json.dump({"embedding_dim": DIM, "data": output}, f)
    print(f"Wrote {len(output)} relationships → {VDB_RELATIONS}")


if __name__ == "__main__":
    print("Loading graph...")
    G = nx.read_graphml(GRAPH_PATH)
    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

    build_entity_vdb(G)
    build_relationship_vdb(G)