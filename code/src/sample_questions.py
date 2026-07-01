"""
Collect all questions where found=True across a folder of JSON files,
then keep a random percentage of them.

Usage:
    python sample_questions.py --input_dir ./data --output_file sampled.json --pct 60

Arguments:
    --input_dir   : Folder containing JSON files (one record per file)
    --output_file : Output JSON file (a JSON array of sampled records)
    --pct         : Percentage of found=True records to keep (0–100)
    --seed        : Random seed for reproducibility (default: 42)
"""

import os
import json
import random
import argparse


def main(input_dir, output_file, pct, seed):
    random.seed(seed)

    found_true = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(input_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                record = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [skip] {fname}: {e}")
            continue
        if record.get("found") is True:
            found_true.append(record)

    n_keep = round(len(found_true) * pct / 100)
    sampled = [r["question"] for r in random.sample(found_true, n_keep)]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    print(f"Found=True total : {len(found_true)}")
    print(f"Kept ({pct}%)     : {n_keep}")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",   required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--pct",         required=True, type=float, help="Percentage of found=True to keep (0–100)")
    parser.add_argument("--seed",        default=3,    type=int)
    args = parser.parse_args()

    if not (0.0 <= args.pct <= 100.0):
        parser.error("--pct must be between 0 and 100")

    main(args.input_dir, args.output_file, args.pct, args.seed)