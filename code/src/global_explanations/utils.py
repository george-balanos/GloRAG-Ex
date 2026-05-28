import json
import networkx as nx

def load_local_explanation(filepath: str):
    with open(filepath, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def load_graph(graph_filepath: str, backend_system: str = "lightrag"):
    if backend_system == "lightrag":
        return nx.read_graphml(graph_filepath)

    return

if __name__ == "__main__":
    filepath = "/home/gbalanos/GloRAG-Ex/code/src/counterfactuals/results/synthetic/delete_ops_ft/counterfactual_20260527_152659.json"
    graph_filepath = "/home/gbalanos/GloRAG-Ex/code/KGs/lightrag/synthetic/graph_chunk_entity_relation.graphml"

    print(load_local_explanation(filepath))
    print(load_graph(graph_filepath))