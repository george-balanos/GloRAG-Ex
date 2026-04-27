import json
import os
from collections import Counter
from pathlib import Path


def load_results(results_dir: str = "src/counterfactuals/counterfactual_results") -> list[dict]:
    results = []
    for path in Path(results_dir).glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["_filename"] = path.name
            results.append(data)
    return results


def evaluate_success_rate(results: list[dict]):
    total = len(results)
    found = sum(1 for r in results if r.get("found", True) and len(r.get("operations", [])) > 0)
    not_found = total - found

    print("=" * 50)
    print("SUCCESS RATE")
    print("=" * 50)
    print(f"  Total cases:       {total}")
    print(f"  Found:             {found} ({100 * found / total:.1f}%)")
    print(f"  Not found:         {not_found} ({100 * not_found / total:.1f}%)")


def evaluate_operation_stats(results: list[dict]):
    successful = [r for r in results if len(r.get("operations", [])) > 0]

    if not successful:
        print("\nNo successful counterfactuals to compute operation stats.")
        return

    num_ops = [r["num_operations"] for r in successful]
    costs = [r["answers"].get("similarity", 0.0) for r in successful]

    # Flatten all operations — handle both node (str) and edge (list) ops
    node_counter = Counter()
    for r in successful:
        for op in r["operations"]:
            if isinstance(op, str):
                node_counter[op] += 1
            elif isinstance(op, list) and len(op) == 2:
                # replace_node: [original, replacement]
                node_counter[op[0]] += 1

    print("\n" + "=" * 50)
    print("OPERATION STATS")
    print("=" * 50)
    print(f"  Avg operations:    {sum(num_ops) / len(num_ops):.2f}")
    print(f"  Min operations:    {min(num_ops)}")
    print(f"  Max operations:    {max(num_ops)}")

    print(f"\n  Top 10 most deleted/replaced nodes:")
    for node, count in node_counter.most_common(10):
        print(f"    {node:<40} {count}x")

    operation_types = Counter(r.get("operation_type", "unknown") for r in successful)
    print(f"\n  Operation types:")
    for op_type, count in operation_types.items():
        print(f"    {op_type:<30} {count}")


def evaluate_answer_similarity(results: list[dict]):
    successful = [r for r in results if len(r.get("operations", [])) > 0]

    if not successful:
        print("\nNo successful counterfactuals to compute similarity stats.")
        return

    similarities = [r["answers"]["similarity"] for r in successful if "similarity" in r["answers"]]

    if not similarities:
        return

    avg = sum(similarities) / len(similarities)
    sorted_sims = sorted(similarities)
    median = sorted_sims[len(sorted_sims) // 2]

    buckets = {"0.0–0.2": 0, "0.2–0.4": 0, "0.4–0.6": 0, "0.6–0.8": 0, "0.8–1.0": 0}
    for s in similarities:
        if s < 0.2:
            buckets["0.0–0.2"] += 1
        elif s < 0.4:
            buckets["0.2–0.4"] += 1
        elif s < 0.6:
            buckets["0.4–0.6"] += 1
        elif s < 0.8:
            buckets["0.6–0.8"] += 1
        else:
            buckets["0.8–1.0"] += 1

    print("\n" + "=" * 50)
    print("ANSWER SIMILARITY (original vs perturbed)")
    print("=" * 50)
    print(f"  Avg similarity:    {avg:.4f}")
    print(f"  Median similarity: {median:.4f}")
    print(f"  Min similarity:    {min(similarities):.4f}")
    print(f"  Max similarity:    {max(similarities):.4f}")
    print(f"\n  Distribution:")
    for bucket, count in buckets.items():
        bar = "█" * count
        print(f"    {bucket}  {bar} {count}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="src/counterfactuals/counterfactual_results",
                        help="Directory containing counterfactual JSON files")
    args = parser.parse_args()

    results = load_results(args.dir)

    if not results:
        print(f"No JSON files found in '{args.dir}'")
        return

    print(f"\nLoaded {len(results)} result files from '{args.dir}'")

    evaluate_success_rate(results)
    evaluate_operation_stats(results)
    evaluate_answer_similarity(results)


if __name__ == "__main__":
    main()