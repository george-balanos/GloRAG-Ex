from src.global_explanations.utils import load_local_explanation
import os
import matplotlib.pyplot as plt

class OperationTypeLevelAggregator:
    def __init__(self):
        self.operation_types = []

    def collect_operation_types(self, local_explanation: str):
        self.local_explanation_data = load_local_explanation(local_explanation)
        operations = self.local_explanation_data["operations"]
        operation_types = self._extract_operation_types(operations)
        self.operation_types.extend(operation_types)
        return operation_types

    def _extract_operation_types(self, operations):
        operation_types = []
        for op in operations:
            operation_types.append(op[0])
        return operation_types
    
    def create_unique_operation_type_dict(self,operation_type_list) -> dict[str, int]:
        current_dict = {}
        for operation_type in operation_type_list:
            current_dict[operation_type] = current_dict.get(operation_type, 0) + 1
        return current_dict

    def visualize_operation_type_distribution(self, operation_type_dict, output_path: str = "src/global_explanations/operation_type_level/plots/operation_type_distribution.png"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        operation_types = list(operation_type_dict.keys())
        counts = list(operation_type_dict.values())

        plt.figure(figsize=(12, 8))
        plt.bar(operation_types, counts, color='skyblue')
        plt.xlabel('Operation Types')
        plt.ylabel('Frequency')
        plt.title('Distribution of Operation Types in Local Explanations')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":

    aggregator = OperationTypeLevelAggregator()
    local_explanations_folder_name = "src/counterfactuals/results/hotpotqa/all_ops_ff"

    for filename in os.listdir(local_explanations_folder_name):
        if filename.endswith(".json"):
            filepath = os.path.join(local_explanations_folder_name, filename)
            explanation = load_local_explanation(filepath)
            if explanation["found"]:
                aggregator.collect_operation_types(filepath)

    operation_type_dict = aggregator.create_unique_operation_type_dict(aggregator.operation_types)
    aggregator.visualize_operation_type_distribution(
        operation_type_dict,
        output_path="src/global_explanations/operation_type_level/plots/operation_type_distribution.png"
    )