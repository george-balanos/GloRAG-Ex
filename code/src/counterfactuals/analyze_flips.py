import json
import os
import argparse

def analyze_counterfactuals(input_path):
    # 1. Load the original JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_entries = len(data)
    empty_perms_count = 0
    total_perms = 0
    total_flipped = 0
    
    # Stability buckets
    perfect_stability = 0
    partial_stability = 0
    zero_stability = 0
    all_flipped_count = 0
    
    # List to track specific 0.0 stability cases
    zero_stability_cases = []
    
    # 2. Iterate through the JSON to calculate stats
    for key, entry in data.items():
        perms = entry.get("permutations", {})
        
        # Count empty permutations
        if not perms:
            empty_perms_count += 1
        else:
            # Aggregate stats for entries that actually have permutations
            total_perms += entry.get("num_permutations", 0)
            total_flipped += entry.get("num_flipped", 0)
            
            # Using -1.0 as a default to safely catch missing keys without falsely flagging as 0.0
            stab = entry.get("flip_stability", -1.0)
            if stab == 1.0:
                perfect_stability += 1
            elif stab > 0.0:
                partial_stability += 1
            elif stab == 0.0:
                zero_stability += 1
                zero_stability_cases.append((key, entry.get("question", "Unknown Question")))
                
            if entry.get("flip_under_all_permutations", False):
                all_flipped_count += 1

    # 3. Create a copy of the JSON in the same folder
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_copy{ext}"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    # 4. Print the generated statistics
    print("=" * 80)
    print("Counterfactual Analysis Results")
    print(f"Input:  {input_path}")
    print(f"Copied: {output_path}")
    print("=" * 80)
    
    print(f"Total Entries:                     {total_entries}")
    print(f"Entries with Empty Permutations:   {empty_perms_count}")
    print(f"Entries with Valid Permutations:   {total_entries - empty_perms_count}")
    print("-" * 80)
    
    print("Permutation Stats (Valid Entries Only):")
    print(f"Total Permutations Run:            {total_perms}")
    print(f"Total Permutations Flipped:        {total_flipped}")
    if total_perms > 0:
        print(f"Overall Flip Rate:                 {(total_flipped / total_perms) * 100:.2f}%")
    else:
        print("Overall Flip Rate:                 N/A")
    print("-" * 80)
    
    print("Flip Stability Breakdown:")
    print(f"Perfect Stability (1.0):           {perfect_stability}")
    print(f"Partial Stability (0.0 < x < 1.0): {partial_stability}")
    print(f"Zero Stability (0.0):              {zero_stability}")
    print(f"Flipped under ALL permutations:    {all_flipped_count}")
    print("=" * 80)

    # 5. Print the specific 0.0 stability cases
    print(f"\nCases with 0.0 Flip Stability (Total: {len(zero_stability_cases)})")
    print("-" * 80)
    
    if not zero_stability_cases:
        print("No cases with 0 stability found.")
    else:
        for idx, (key, question) in enumerate(zero_stability_cases, 1):
            print(f"{idx}. {key}")
            print(f"   Question: {question}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze counterfactual permutation JSON.")
    parser.add_argument("input_json", help="Path to the input JSON file")
    args = parser.parse_args()
    
    analyze_counterfactuals(args.input_json)