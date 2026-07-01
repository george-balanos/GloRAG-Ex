from src.global_explanations.utils import load_local_explanation
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family']      = 'serif'
rcParams['font.serif']       = ['Times New Roman', 'Times', 'DejaVu Serif']
rcParams['font.size']        = 12
rcParams['axes.titlesize']   = 15
rcParams['axes.labelsize']   = 13
rcParams['xtick.labelsize']  = 10
rcParams['ytick.labelsize']  = 10
rcParams['lines.linewidth']  = 2.2
rcParams['pdf.fonttype']     = 42
rcParams['ps.fonttype']      = 42

DATASETS = ["synthetic", "musique", "2wiki", "hotpotqa"]

DATASET_DISPLAY = {
    "synthetic": "Synthetic",
    "hotpotqa":  "Hotpot",
    "musique":   "MuSiQue",
    "2wiki":     "2Wiki",
}


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

    def create_unique_operation_type_dict(self, operation_type_list) -> dict[str, int]:
        current_dict = {}
        for operation_type in operation_type_list:
            current_dict[operation_type] = current_dict.get(operation_type, 0) + 1
        return current_dict

    def addition_percentage(self, operation_type_dict: dict) -> float:
        total = sum(operation_type_dict.values())
        if total == 0:
            return 0.0
        additions = sum(
            v for k, v in operation_type_dict.items()
            if k in ("add_node", "add_edge")
        )
        return additions / total * 100


def visualize_addition_percentages(
    dataset_percentages: dict[str, float],
    output_path: str = "src/global_explanations/operation_type_level/plots/addition_percentage_by_dataset.pdf",
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    labels = [DATASET_DISPLAY.get(d, d) for d in dataset_percentages]
    values = list(dataset_percentages.values())

    fig, ax = plt.subplots(figsize=(6, max(2, len(labels) * 0.4)), dpi=150)
    ax.barh(labels, values, height=0.75, color="#b3d4e8", edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Additions in explanations (%)")
    ax.set_ylabel("Dataset")
    ax.set_xlim(left=0)
    ax.margins(y=0.1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35, linewidth=0.7)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"✓ Plot saved to: {output_path}")


if __name__ == "__main__":
    dataset_percentages = {}

    for dataset in DATASETS:
        aggregator = OperationTypeLevelAggregator()
        folder = f"src/counterfactuals/results/{dataset}/all_ops_ff"

        for filename in os.listdir(folder):
            if filename.endswith(".json"):
                filepath = os.path.join(folder, filename)
                explanation = load_local_explanation(filepath)
                if explanation["found"]:
                    aggregator.collect_operation_types(filepath)

        op_dict = aggregator.create_unique_operation_type_dict(aggregator.operation_types)
        dataset_percentages[dataset] = aggregator.addition_percentage(op_dict)
        print(f"  {DATASET_DISPLAY.get(dataset, dataset)}: {dataset_percentages[dataset]:.1f}% additions")

    visualize_addition_percentages(dataset_percentages)