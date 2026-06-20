"""Combine correctness JSONs into one table (CSV + summary).

Reads one or more ``*_correctness.json`` files written by
``src.correctness.evaluate`` and emits a tidy CSV with one row per
(method, dataset, direction) plus a combined summary JSON -- the numbers that go
into the paper's correctness table (tab:correctness-agreement).

Run from code/ (PYTHONPATH=code), e.g.:
  ../.venv/bin/python -m src.correctness.analyze_correctness \
      benchmark/results/hotpotqa_glorag_correctness.json \
      benchmark/results/hotpotqa_shapley_correctness.json \
      --csv benchmark/results/correctness_summary.csv
"""
import argparse
import csv
import json
import os

KS = (1, 2, 3, 5)
# Unified schema: GloRAG-Ex fills precision/precision_micro/tp/fp; attribution/ragex fill P@k.
# All methods fill the graph fact-coverage columns; only GloRAG-Ex fills the perturbed-graph ones.
COLS = ["source", "method", "dataset", "direction", "n",
        "precision", "hit_rate", "precision_micro", "tp", "fp",
        "P@1", "P@3", "P@5", "Hit@1", "Hit@3", "Hit@5", "P@2", "Hit@2",
        "orig_facts_cov", "orig_cov_ratio", "facts_total", "pert_facts_cov", "pert_cov_ratio"]


def rows_from_file(path):
    with open(path, encoding="utf-8") as f:
        summary = json.load(f).get("__summary__", {})
    meta = summary.get("_meta", {})
    method = meta.get("method", "?")
    dataset = meta.get("dataset", "?")
    source = os.path.basename(path).replace("_correctness.json", "").replace(".json", "")
    rows = []
    for direction in ("overall", "T->F", "F->T"):
        b = summary.get(direction)
        if not b or not b.get("n"):
            continue
        rows.append({
            "source": source, "method": method, "dataset": dataset, "direction": direction, "n": b["n"],
            "precision": b.get("precision"), "hit_rate": b.get("hit_rate"),
            "precision_micro": b.get("precision_micro"),
            "tp": b.get("tp_total"), "fp": b.get("fp_total"),
            **{f"P@{k}": b.get(f"P@{k}") for k in KS},
            **{f"Hit@{k}": b.get(f"Hit@{k}") for k in KS},
            "orig_facts_cov": b.get("orig_facts_cov"), "orig_cov_ratio": b.get("orig_cov_ratio"),
            "facts_total": b.get("facts_total"),
            "pert_facts_cov": b.get("pert_facts_cov"), "pert_cov_ratio": b.get("pert_cov_ratio"),
        })
    return rows


def main():
    p = argparse.ArgumentParser(description="Combine correctness JSONs into a CSV + summary.")
    p.add_argument("inputs", nargs="+", help="One or more *_correctness.json files.")
    p.add_argument("--csv", default="benchmark/results/correctness_summary.csv")
    p.add_argument("--json", default="benchmark/results/correctness_summary.json")
    args = p.parse_args()

    rows = []
    for path in args.inputs:
        rows.extend(rows_from_file(path))

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else ("-" if v is None else str(v))
    show = ("precision", "P@1", "P@2", "P@5", "orig_facts_cov", "pert_facts_cov", "facts_total")
    print(f"{'source':<28}{'dir':<8}{'n':>5}  " + "".join(f"{c:>10}" for c in show))
    for r in rows:
        print(f"{r['source']:<28}{r['direction']:<8}{r['n']:>5}  " +
              "".join(f"{fmt(r.get(c)):>10}" for c in show))
    print(f"\n-> {args.csv}\n-> {args.json}")


if __name__ == "__main__":
    main()
