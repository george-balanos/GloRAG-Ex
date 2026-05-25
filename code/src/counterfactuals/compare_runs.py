"""Compare two counterfactual run directories (e.g. baseline vs PSP).

Reads every JSON output in each directory, joins by question text, prints a
per-question table and aggregate stats, and writes a CSV summary.
"""

import argparse
import csv
import json
import os
from pathlib import Path


def load_dir(path: str) -> dict:
    """Load all counterfactual_*.json files in `path`, key by question text."""
    out = {}
    for p in sorted(Path(path).glob("counterfactual_*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"[warn] could not read {p}: {e}")
            continue
        q = payload.get("question")
        if not q:
            continue
        # Keep the latest file per question (sorted = lexicographic, timestamps
        # in filenames preserve order)
        out[q] = payload
    return out


def ops_signature(payload):
    """Canonical hashable signature of the ops list, for equality comparison."""
    ops = payload.get("operations") or []
    sig = []
    for op in ops:
        if isinstance(op, list):
            sig.append(tuple(_freeze(x) for x in op))
        else:
            sig.append(op)
    return tuple(sig)


def _freeze(x):
    if isinstance(x, list):
        return tuple(_freeze(y) for y in x)
    return x


def parse_args():
    p = argparse.ArgumentParser(description="Compare baseline vs PSP CFE runs.")
    p.add_argument("--baseline-dir", required=True,
                   help="Output directory of the baseline run")
    p.add_argument("--psp-dir", required=True,
                   help="Output directory of the PSP run")
    p.add_argument("--out", default="benchmark/results/psp_compare.csv",
                   help="CSV output path")
    return p.parse_args()


def main():
    args = parse_args()
    base = load_dir(args.baseline_dir)
    psp = load_dir(args.psp_dir)

    print(f"Loaded {len(base)} baseline entries from {args.baseline_dir}")
    print(f"Loaded {len(psp)} PSP      entries from {args.psp_dir}")

    all_questions = sorted(set(base.keys()) | set(psp.keys()))
    rows = []
    for q in all_questions:
        b = base.get(q)
        p = psp.get(q)

        def fld(payload, key, default=None):
            return payload.get(key, default) if payload else default

        b_found = bool(fld(b, "found", False))
        p_found = bool(fld(p, "found", False))
        b_cost = fld(b, "cost", float("nan"))
        p_cost = fld(p, "cost", float("nan"))
        b_calls = fld(b, "llm_calls", None)
        p_calls = fld(p, "llm_calls", None)
        b_n = fld(b, "num_operations", 0)
        p_n = fld(p, "num_operations", 0)
        try:
            d_cost = (p_cost - b_cost) if (b and p and b_found and p_found) else None
        except TypeError:
            d_cost = None
        try:
            d_calls = (p_calls - b_calls) if (b_calls is not None and p_calls is not None) else None
        except TypeError:
            d_calls = None
        ops_match = (b_found and p_found and ops_signature(b) == ops_signature(p))

        rows.append({
            "question": q,
            "base_found": b_found,
            "psp_found": p_found,
            "base_cost": b_cost,
            "psp_cost": p_cost,
            "delta_cost": d_cost,
            "base_llm_calls": b_calls,
            "psp_llm_calls": p_calls,
            "delta_llm_calls": d_calls,
            "base_n_ops": b_n,
            "psp_n_ops": p_n,
            "ops_match": ops_match,
        })

    # Per-question table
    header = f"{'idx':<4} {'q[:60]':<62} {'b.f':<4} {'p.f':<4} {'b.cost':<8} {'p.cost':<8} {'Δcost':<8} {'b.llm':<6} {'p.llm':<6} {'Δllm':<6} {'match':<6}"
    print("\n" + header)
    print("-" * len(header))
    for i, r in enumerate(rows):
        q_short = (r["question"] or "")[:60]
        dc = f"{r['delta_cost']:+.2f}" if r["delta_cost"] is not None else "-"
        dl = f"{r['delta_llm_calls']:+d}" if r["delta_llm_calls"] is not None else "-"
        bc = f"{r['base_cost']:.2f}" if r["base_found"] else "-"
        pc = f"{r['psp_cost']:.2f}" if r["psp_found"] else "-"
        bl = str(r["base_llm_calls"]) if r["base_llm_calls"] is not None else "-"
        pl = str(r["psp_llm_calls"]) if r["psp_llm_calls"] is not None else "-"
        print(f"{i:<4} {q_short:<62} {str(r['base_found']):<4} {str(r['psp_found']):<4} "
              f"{bc:<8} {pc:<8} {dc:<8} {bl:<6} {pl:<6} {dl:<6} {str(r['ops_match']):<6}")

    # Aggregates
    total = len(rows)
    both = sum(1 for r in rows if r["base_found"] and r["psp_found"])
    base_only = sum(1 for r in rows if r["base_found"] and not r["psp_found"])
    psp_only = sum(1 for r in rows if r["psp_found"] and not r["base_found"])
    neither = sum(1 for r in rows if not r["base_found"] and not r["psp_found"])
    match = sum(1 for r in rows if r["ops_match"])

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    mean_dcost = _mean([r["delta_cost"] for r in rows if r["delta_cost"] is not None])
    mean_dllm = _mean([r["delta_llm_calls"] for r in rows if r["delta_llm_calls"] is not None])

    print("\n=== Aggregate ===")
    print(f"  Total questions     : {total}")
    print(f"  Both found          : {both}")
    print(f"  Baseline only       : {base_only}")
    print(f"  PSP only            : {psp_only}")
    print(f"  Neither             : {neither}")
    if both:
        print(f"  Exact-op-match (both): {match} / {both} ({100*match/both:.1f}%)")
    if mean_dcost is not None:
        print(f"  Mean Δcost  (psp - base, where both found): {mean_dcost:+.3f}")
    if mean_dllm is not None:
        print(f"  Mean Δllm   (psp - base):                   {mean_dllm:+.2f}")

    # CSV
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nCSV written to: {args.out}")


if __name__ == "__main__":
    main()
