from src.global_explanations.utils import load_graph, load_local_explanation
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

########## Setup ##########

backend_system = "lightrag"
dataset = "hotpotqa"

knowledge_graph_filepath = f"KGs/{backend_system}/{dataset}/graph_chunk_entity_relation.graphml"
knowledge_graph = load_graph(knowledge_graph_filepath)

###########################

class FeatureVectorGenerator:
    def __init__(self):
        pass

    def create_feature_vectors(self, local_explanation: str):
        self.local_explanation_data = load_local_explanation(local_explanation)

        operations = self.local_explanation_data["operations"]

        entities = self.local_explanation_data["original_subgraph"]["entities"]
        relations = self.local_explanation_data["original_subgraph"]["relations"]
        subgraph = self._assemble_local_graph(entities, relations)

        operation_dict = self._extract_explanation_elements(operations)

        #### Delete-Node-Features ####

        local_delete_node_degrees = list(subgraph.degree(operation_dict["delete_node"]))
        global_delete_node_degrees = list(knowledge_graph.degree(operation_dict["delete_node"]))

        #### Delete-Edge-Features ####

        local_betweenness_centrality = nx.edge_betweenness_centrality(subgraph)

        local_delete_edge_betweennesses = []
        for edge in operation_dict["delete_edge"]:
            local_delete_edge_betweennesses.append(round(local_betweenness_centrality.get(edge), 3))

        feature_vector = {
            "delete": {
                "local_node_degrees": local_delete_node_degrees,
                "global_node_degrees": global_delete_node_degrees,
                "local_edge_betweennesses": local_delete_edge_betweennesses
            },
            "add": {

            }
        }

        return feature_vector
    
    def _extract_explanation_elements(self, operations):
        operation_dict = {
            "delete_node": [],
            "delete_edge": [],
            "add_node": [],
            "add_edge": []
        }

        for op in operations:
            op_type = op[0]

            if op_type == "delete_node":
                operation_dict[op_type].append(op[1])
            elif op_type == "delete_edge":
                operation_dict[op_type].append((op[1][0], op[1][1]))
            elif op_type == "add_node":
                operation_dict[op_type].append(op[1])
            elif op_type == "add_edge":
                operation_dict[op_type].append((op[1][0], op[1][1]))

        return operation_dict

    def _assemble_local_graph(self, ents, rels) -> nx.DiGraph:
        subgraph = nx.DiGraph()
        for ent in ents:
            name = ent["name"]
            type = ent["type"]
            desc = ent["description"]

            subgraph.add_node(name, entity_type=type, description=desc)

        for rel in rels:
            src = rel["src"]
            tgt = rel["tgt"]
            keywords = rel["keywords"]
            desc = rel["description"]

            subgraph.add_edge(src, tgt, keywords=keywords, description=desc)

        return subgraph


class FeatureVectorAggregator:
    def __init__(self, feature_vectors, output_dir: str = "plots"):
        self.vecs = feature_vectors
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_distributions(self):
        self._plot_local_degrees()
        self._plot_global_degrees()
        self._plot_edge_betweennesses()
        plt.show()

    def _plot_local_degrees(self):
        values = [deg for vec in self.vecs for _, deg in vec["delete"]["local_node_degrees"]]
        self._plot_hist(values, title="Local Node Degrees", xlabel="Degree", color="steelblue", filename="local_node_degrees")

    def _plot_global_degrees(self):
        values = [deg for vec in self.vecs for _, deg in vec["delete"]["global_node_degrees"]]
        self._plot_hist(values, title="Global Node Degrees", xlabel="Degree", color="darkorange", filename="global_node_degrees")

    def _plot_edge_betweennesses(self):
        values = [b for vec in self.vecs for b in vec["delete"]["local_edge_betweennesses"]]
        self._plot_hist(values, title="Local Edge Betweenness", xlabel="Betweenness Score", color="seagreen", filename="local_edge_betweennesses")

    def _plot_hist(self, values, title, xlabel, color, filename, max_int_bins=20):
        fig, ax = plt.subplots()

        if all(isinstance(v, int) for v in values):
            spread = max(values) - min(values)
            if spread <= max_int_bins:
                bins = range(min(values), max(values) + 2)
                ax.hist(values, bins=bins, color=color, edgecolor="white", align="left")
                ax.set_xticks(range(min(values), max(values) + 1))
            else:
                log_bins = np.logspace(np.log10(max(1, min(values))), np.log10(max(values)), 20)
                ax.hist(values, bins=log_bins, color=color, edgecolor="white")
                ax.set_xscale("log")
        else:
            ax.hist(values, bins=10, color=color, edgecolor="white")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"{filename}.png"), dpi=150)
        plt.close(fig)

if __name__ == "__main__":
    featureVector = FeatureVectorGenerator()

    local_explanations_folder_name = "src/counterfactuals/results/hotpotqa/delete_ops_ft"

    feature_vector_list: list[dict] = []

    for filename in os.listdir(local_explanations_folder_name):
        if filename.endswith(".json"):
            filepath = os.path.join(local_explanations_folder_name, filename)
            vecs = featureVector.create_feature_vectors(filepath)
            feature_vector_list.append(vecs)

    featureAggregator = FeatureVectorAggregator(feature_vector_list, output_dir="src/global_explanations/feature_level/plots")
    featureAggregator.plot_distributions()