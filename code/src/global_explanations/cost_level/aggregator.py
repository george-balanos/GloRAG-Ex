from src.global_explanations.utils import load_local_explanation

import os
import matplotlib.pyplot as plt

class CostLevelAggregator:
    def __init__(self):
        self.costs = []

    def collect_costs(self, local_explanation: str):
        self.local_explanation_data = load_local_explanation(local_explanation)
        self.costs.append(self.local_explanation_data["cost"])

    def visualize_costs(self, output_path: str = "src/global_explanations/cost_level/plots/cost_distribution.png"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.hist(self.costs, bins=10, color='skyblue', edgecolor='black')
        plt.title('Distribution of Explanation Costs')
        plt.xlabel('Cost')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    aggregator = CostLevelAggregator()

    local_explanations_folder_name = "src/counterfactuals/results/hotpotqa/delete_ops_ft"

    for filename in os.listdir(local_explanations_folder_name):
        if filename.endswith(".json"):
            filepath = os.path.join(local_explanations_folder_name, filename)
            if load_local_explanation(filepath)["found"]:
                aggregator.collect_costs(filepath)

    aggregator.visualize_costs()