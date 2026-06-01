from src.global_explanations.utils import load_local_explanation
import networkx as nx
import matplotlib.pyplot as plt
import os

class ElementLevelAggregator:
    def __init__(self):
        pass

    def collect_elements(self, local_explanation: str):
        self.local_explanation_data = load_local_explanation(local_explanation)
        operations = self.local_explanation_data["operations"]
        elements_dict = self._extract_explanation_elements(operations)
        return elements_dict["nodes"], elements_dict["edges"]

    def _extract_explanation_elements(self, operations):
        elements_dict = {"nodes": [], "edges": []}

        for op in operations:
            op_type = op[0]
            if op_type == "delete_node":
                elements_dict["nodes"].append(op[1])
            elif op_type == "delete_edge":
                elements_dict["edges"].append((op[1][0], op[1][1]))
            elif op_type == "add_node":
                elements_dict["nodes"].append(op[1])
            elif op_type == "add_edge":
                elements_dict["edges"].append((op[1][0], op[1][1]))

        return elements_dict


def create_unique_element_dict(element_list) -> dict[str | tuple, int]:
    current_dict = {}
    for element in element_list:
        current_dict[element] = current_dict.get(element, 0) + 1
    return current_dict


def build_graph(node_dict, edge_dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for node, count in node_dict.items():
        G.add_node(node, count=count)
    for (src, tgt), count in edge_dict.items():
        G.add_edge(src, tgt, count=count)
    return G


def visualize_graph(G: nx.DiGraph, output_path: str = "plots/element_graph.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    node_counts = [max(G.nodes[n].get("count", 0), 1) for n in G.nodes]
    edge_counts = [G.edges[e]["count"] for e in G.edges]

    pos = nx.spring_layout(G, seed=42, k=3.0, iterations=100)  # k controls spacing

    fig, ax = plt.subplots(figsize=(20, 16))  # larger canvas

    nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_size=600,
                       node_color="steelblue")

    nx.draw_networkx_edges(G, pos, ax=ax,
                       width=[1.0 + c * 0.5 for c in edge_counts],
                       edge_color="crimson",
                       arrows=True, arrowsize=15,
                       min_source_margin=15, min_target_margin=15,
                       connectionstyle="arc3,rad=0.1")

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

    ax.set_title("Element-Level Explanation Graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Graph saved to {output_path}")

def print_frequent_elements(node_dict: dict, edge_dict: dict, min_count: int = 2):
    frequent_nodes = {n: c for n, c in node_dict.items() if c >= min_count}
    frequent_edges = {e: c for e, c in edge_dict.items() if c >= min_count}

    print(f"\nNodes appearing in {min_count}+ cases ({len(frequent_nodes)} total):")
    for node, count in sorted(frequent_nodes.items(), key=lambda x: -x[1]):
        print(f"  {node}: {count}")

    print(f"\nEdges appearing in {min_count}+ cases ({len(frequent_edges)} total):")
    for (src, tgt), count in sorted(frequent_edges.items(), key=lambda x: -x[1]):
        print(f"  {src} -> {tgt}: {count}")

def print_graph_stats(G: nx.DiGraph):
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    weakly_connected = nx.number_weakly_connected_components(G)
    strongly_connected = nx.number_strongly_connected_components(G)
    print(f"Weakly connected components:  {weakly_connected}")
    print(f"Strongly connected components: {strongly_connected}")

    if nx.is_weakly_connected(G):
        print("Graph is weakly connected (single component)")
    else:
        sizes = sorted([len(c) for c in nx.weakly_connected_components(G)], reverse=True)
        print(f"Weakly connected component sizes: {sizes}")

if __name__ == "__main__":
    element_aggregator = ElementLevelAggregator()

    local_explanations_folder_name = "src/counterfactuals/results/synthetic/all_ops_ff"

    nodes, edges = [], []

    for filename in os.listdir(local_explanations_folder_name):
        if filename.endswith(".json"):
            filepath = os.path.join(local_explanations_folder_name, filename)
            explanation = load_local_explanation(filepath)
            if explanation["found"]:
                curr_nodes, curr_edges = element_aggregator.collect_elements(filepath)
                nodes.extend(curr_nodes)
                edges.extend(curr_edges)

    node_dict = create_unique_element_dict(nodes)
    edge_dict = create_unique_element_dict(edges)

    G = build_graph(node_dict, edge_dict)
    print_frequent_elements(node_dict, edge_dict, min_count=2)
    print_graph_stats(G)
    visualize_graph(G, output_path="src/global_explanations/element_level/plots/element_graph.png")