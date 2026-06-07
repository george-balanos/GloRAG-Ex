"""Shared dataset binding for entrypoint scripts.

Centralizes WORKING_DIRS, QA_CSV_PATHS and setup_dataset(name) so every CLI uses
the same path convention.
"""
from collections import defaultdict

import networkx as nx

from src.embeddings.utils import load_index
from src.embeddings.query import DIM, build_lookup, build_edge_lookup


WORKING_DIRS = {
    "synthetic": "KGs/lightrag/synthetic",
    "hotpotqa":  "KGs/lightrag/hotpotqa",
}

QA_CSV_PATHS = {
    "synthetic": "qa/qa_data_synthetic.csv",
    "hotpotqa":  "qa/qa_data_hotpotqa.csv",
}

DATASETS = list(WORKING_DIRS.keys())


def create_type_index(G):
    idx = defaultdict(list)
    for node, data in G.nodes(data=True):
        idx[data.get("entity_type")].append(node)
    return idx


def setup_dataset(name: str, max_elements: int = 2000) -> dict:
    """Load graph + HNSW indices for `name`. Returns a dict the caller stashes module-side.

    Keys: dataset, G, type_index, node_index, node_records, node_embeddings,
          node_lookup, edge_index, edge_records, edge_embeddings, edge_lookup.
    """
    if name not in WORKING_DIRS:
        raise ValueError(f"Unknown dataset {name!r}; choices: {DATASETS}")

    G = nx.read_graphml(f"{WORKING_DIRS[name]}/graph_chunk_entity_relation.graphml")
    type_index = create_type_index(G)

    node_index, node_records, node_embeddings = load_index(
        f"src/embeddings/{name}/node_index", DIM, max_elements)
    edge_index, edge_records, edge_embeddings = load_index(
        f"src/embeddings/{name}/edge_index", DIM, max_elements)

    return {
        "dataset": name,
        "G": G,
        "type_index": type_index,
        "node_index": node_index,
        "node_records": node_records,
        "node_embeddings": node_embeddings,
        "node_lookup": build_lookup(node_records),
        "edge_index": edge_index,
        "edge_records": edge_records,
        "edge_embeddings": edge_embeddings,
        "edge_lookup": build_edge_lookup(edge_records),
    }
