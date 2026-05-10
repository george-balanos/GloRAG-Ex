"""Compare counterfactual stability across noisy and clean result folders.

Loads two folders of saved CFE JSON files keyed by question, computes Jaccard
overlap of returned operation sets per question, and reports aggregate
agreement so we can quantify how much injected noise perturbs explanations.
"""

from pathlib import Path

import json

def load_examples(folder: Path):
    examples = {}
    for path in folder.rglob("*.json"):
        with open(path) as f:
            data = json.load(f)
        
        question = data.get("question", "")
        if question:
            examples[question] = data

    return examples

def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0

def evaluate_robustness(original_folder, noisy_folder):
    original_examples = load_examples(original_folder)
    noisy_examples = load_examples(noisy_folder)

    common_questions = sorted(set(original_examples) & set(noisy_examples))

    print(f"\nFolder A : {original_folder}")
    print(f"Folder B : {noisy_folder}")
    print(f"Found    : {len(original_examples)} successes in A, {len(noisy_examples)} in B")
    print(f"Common   : {len(common_questions)} shared questions\n")

    if not common_questions:
        print(f"No common successful examples found.")
        return
    
    noise_unsuccessful_cases = set()
    for question in common_questions:
        if noisy_examples[question]["noise"] == {}:
            noise_unsuccessful_cases.add(question)
    
    print(f"Unsuccessful cases due to noise: {len(noise_unsuccessful_cases)}")
    print(f"Noise affected Counterfactual Generation: {100*len(noise_unsuccessful_cases)/len(common_questions):.2f}")
    print()

    noise_robust_cases = set()
    for question in common_questions:
        if question not in noise_unsuccessful_cases:
            if len(noisy_examples[question]["noise"]["noise_nodes_in_counterfactual"]) == 0:
                noise_robust_cases.add(question)

    print(f"Noise did not affect CFEs generation: {len(noise_robust_cases)}")
    print(f"Noise robust Counterfactual Generation (successful): {100*len(noise_robust_cases)/(len(common_questions)-len(noise_unsuccessful_cases)):.2f}")
    print()
    print(f"Noise robust Counterfactual Generation (total): {100*len(noise_robust_cases)/(len(common_questions)):.2f}")

if __name__ == "__main__":
    noise = 0.1

    original_folder = Path("src/counterfactuals/counterfactual_results_sem_all")
    noisy_folder = Path(f"src/counterfactuals/robustness/delete_only_results_{noise*100}")

    evaluate_robustness(original_folder, noisy_folder)