"""Analyze a run_shapley.py --permute output JSON.

For each question it reports how many of the permutations AGREE on:
  - top-1 object,
  - top-2 set,
  - top-5 set,
  - the exact full ranking ("exactly the same entities" in the same order),
where "how many agree" = the size of the largest group of permutations sharing
that key (the modal/plurality count, out of num_permutations). It also surfaces
the per-question ranking stats already in the file (Kendall-tau, positional
top-k matches) and prints + writes an aggregate.

Top-k uses SET membership (the top-k objects, any order); the exact column adds
order. With <=k retrieved objects the top-k set is the whole set, so that column
trivially equals num_permutations — read `exact` for small contexts.

Usage (from code/):
  ../.venv/bin/python benchmark/analyze_shapley_permutation.py \
      --input benchmark/results/<RUN_TS>/synthetic_shapley_permutation.json
"""
import argparse
import csv
import json
import os
from collections import Counter

KS = (1, 2, 5)


def get_ranking(rec) -> list:
    """Ranking (ids, best first) for one permutation record; derive if absent."""
    if rec.get("ranking"):
        return list(rec["ranking"])
    sc = rec.get("shapley_scores", {})
    return sorted(sc, key=lambda o: sc[o], reverse=True)


def modal_agreement(keys) -> tuple[int, int]:
    """Return (largest #perms sharing one key, #distinct keys)."""
    c = Counter(keys)
    _, top = c.most_common(1)[0]
    return top, len(c)


def analyze_question(rec) -> dict:
    perms = rec.get("permutations", [])
    rankings = [get_ranking(p) for p in perms]
    n_perms = len(rankings)
    n_obj = len(rec.get("object_ids") or (rankings[0] if rankings else []))
    stats = rec.get("stats", {})

    row = {
        "n_objects": n_obj,
        "n_permutations": n_perms,
        "mean_kendall_tau": stats.get("mean_kendall_tau"),
        "min_kendall_tau": stats.get("min_kendall_tau"),
        "topk_position_matches": stats.get("topk_position_matches"),
        "topk_positions_checked": stats.get("topk_positions_checked"),
    }

    # top-k SET agreement (largest group of perms sharing the same top-k members)
    for k in KS:
        k_eff = min(k, n_obj) if n_obj else 0
        keys = [frozenset(r[:k_eff]) for r in rankings]
        agree, distinct = modal_agreement(keys) if keys else (0, 0)
        row[f"top{k}_agree"] = agree
        row[f"top{k}_distinct"] = distinct
        row[f"top{k}_all_agree"] = (agree == n_perms and n_perms > 0)

    # exact full-ranking agreement (order matters)
    exact_keys = [tuple(r) for r in rankings]
    agree, distinct = modal_agreement(exact_keys) if exact_keys else (0, 0)
    row["exact_agree"] = agree
    row["exact_distinct"] = distinct
    row["exact_all_agree"] = (agree == n_perms and n_perms > 0)
    return row


def main(args):
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    rows = {}
    for rid, rec in data.items():
        if rid == "__summary__" or not isinstance(rec, dict) or "permutations" not in rec:
            continue
        rows[rid] = analyze_question(rec)

    n = len(rows) or 1

    def avg(key):
        vals = [r[key] for r in rows.values() if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    def pct_all(key):
        return round(100 * sum(1 for r in rows.values() if r.get(key)) / n, 2)

    aggregate = {
        "questions": len(rows),
        "avg_mean_kendall_tau": avg("mean_kendall_tau"),
        "avg_min_kendall_tau": avg("min_kendall_tau"),
        "avg_topk_position_matches": avg("topk_position_matches"),
    }
    for k in KS:
        aggregate[f"avg_top{k}_agree"] = avg(f"top{k}_agree")          # avg # perms sharing modal top-k
        aggregate[f"pct_top{k}_all_agree"] = pct_all(f"top{k}_all_agree")  # % questions where ALL perms agree
    aggregate["avg_exact_agree"] = avg("exact_agree")
    aggregate["pct_exact_all_agree"] = pct_all("exact_all_agree")

    # ── write CSV ────────────────────────────────────────────────────────────
    out_csv = args.output or (os.path.splitext(args.input)[0] + "_agreement.csv")
    cols = (["question_id", "n_objects", "n_permutations"]
            + [c for k in KS for c in (f"top{k}_agree", f"top{k}_distinct", f"top{k}_all_agree")]
            + ["exact_agree", "exact_distinct", "exact_all_agree",
               "mean_kendall_tau", "min_kendall_tau",
               "topk_position_matches", "topk_positions_checked"])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rid, r in rows.items():
            w.writerow({"question_id": rid, **{c: r.get(c) for c in cols if c != "question_id"}})

    out_json = os.path.splitext(out_csv)[0] + "_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "per_question": rows}, f, indent=2)

    # ── print ────────────────────────────────────────────────────────────────
    print(f"\nPermutation agreement  ({len(rows)} questions)  input={args.input}")
    print("=" * 92)
    hdr = f"{'qid':<10}{'n':>3}{'P':>3}  " + "".join(f"{'t'+str(k):>6}" for k in KS) + f"{'exact':>7}{'meanτ':>8}{'minτ':>8}{'pos':>6}"
    print(hdr)
    print("-" * 92)
    for rid, r in rows.items():
        line = f"{rid[:10]:<10}{r['n_objects']:>3}{r['n_permutations']:>3}  "
        line += "".join(f"{str(r[f'top{k}_agree'])+'/'+str(r['n_permutations']):>6}" for k in KS)
        line += f"{str(r['exact_agree'])+'/'+str(r['n_permutations']):>7}"
        mt = r["mean_kendall_tau"]; nt = r["min_kendall_tau"]
        line += f"{(mt if mt is not None else float('nan')):>8.3f}{(nt if nt is not None else float('nan')):>8.3f}"
        pm, pc = r["topk_position_matches"], r["topk_positions_checked"]
        line += f"{(str(pm)+'/'+str(pc)) if pm is not None else '-':>6}"
        print(line)
    print("=" * 92)
    a = aggregate
    print(f"questions                : {a['questions']}")
    print(f"avg mean / min Kendall-τ : {a['avg_mean_kendall_tau']} / {a['avg_min_kendall_tau']}")
    for k in KS:
        print(f"top-{k}: avg perms agreeing = {a[f'avg_top{k}_agree']}   "
              f"| all-perms-agree in {a[f'pct_top{k}_all_agree']}% of questions")
    print(f"exact ranking: avg perms agreeing = {a['avg_exact_agree']}   "
          f"| all-perms-agree in {a['pct_exact_all_agree']}% of questions")
    print("=" * 92)
    print(f"Per-question CSV -> {out_csv}\nSummary          -> {out_json}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze_shapley_permutation",
        description="Per-question top-1/2/5 + exact-ranking agreement across Shapley permutations.")
    p.add_argument("--input", required=True, help="A run_shapley.py --permute output JSON.")
    p.add_argument("--output", default=None, help="Per-question CSV path (summary JSON derived from it).")
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
