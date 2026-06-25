"""RAGE noise-resistance benchmark over retrieved SENTENCES.

RAGE is a SET method (its explanation is the minimal counterfactual combination, not
a graded importance vector), so its noise metric is the SAME set-based one the method
uses — `noise_in_explanation` (code/src/quality_metrics/noise_resistance.py) — NOT the
Shapley graded "mass on noise" (which that file's docstring calls merely "the Shapley
analog of CFE's noise_in_explanation").

For each comparison ft/ff case and each noise level we:
  1. Retrieve the clean sentences D_q and the clean answer (the judge reference).
  2. Inject foreign sentences drawn from the WHOLE KG chunk store (load_all_graph_chunks
     -> kv_store_text_chunks.json), excluding the question's own sentences.
  3. Regenerate on the noisy context and flag noise_robust = judge(noisy, clean) != 0.
  4. Run RAGE's combination counterfactual over the noisy source set:
       ft (top-down): remove a minimal set R to break the clean answer  -> explanation = R.
       ff (bottom-up): retain a minimal set K to reach ground_truth      -> explanation = K.
  5. The headline metric is DIRECTIONAL set membership — does the explanation contain
     injected noise?
       T->F: is noise in the REMOVED set R?     F->T: is noise in the RETAINED set K?
     A faithful explainer's R/K is genuine supporting content, so noise_in_explanation
     should be False (cf_noise_frac ~ 0), especially on robust rows.

  cd code && ../.venv/bin/python ../competitors/RAGE/run_rage_noise.py \
      --dataset hotpotqa --comparison <path>/comparison_hotpotqa.json \
      --noise-percentages 0.1,0.2,0.3,0.5 --max-llm-calls 200
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_CODE_DIR = os.path.join(_REPO_ROOT, "code")
_SHAPLEY_DIR = os.path.join(_REPO_ROOT, "competitors", "Shapley")
for _p in (_CODE_DIR, _SHAPLEY_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_shapley import RagCounter  #
from run_shapley_text import split_into_players  #
from run_shapley_noise_text import load_all_graph_chunks, add_random_noise_chunks  #

from src.retrieve import initialize_lightrag  #
from src.query import query  #
from src.dataset_setup import WORKING_DIRS, DATASETS  #

from chunk_utils import retrieve_chunks, render_context_from_chunks  #
from run_rage import (  #
    GRANULARITY, JudgedEvaluator, relevance_scores, rage_combination,
    _records_from_comparison, _records_from_csv,
)

from tqdm import tqdm  #
import argparse  #
import asyncio  #
import json  #
import logging  #
import time  #

logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("nano-vectordb").setLevel(logging.WARNING)
logging.getLogger("lightrag").setLevel(logging.WARNING)


def noise_in_explanation_metrics(cf_sentences, noise_set) -> dict:
    """Directional set membership of noise in RAGE's counterfactual explanation
    (R for ft, K for ff). The native `noise_in_explanation` metric the method uses."""
    cf = set(cf_sentences)
    noise = set(noise_set)
    leaked = cf & noise
    return {
        "cf_size": len(cf),
        "n_noise": len(noise),
        "n_noise_in_explanation": len(leaked),
        "noise_in_explanation": len(leaked) > 0,                 # the headline boolean
        "cf_noise_frac": (len(leaked) / len(cf)) if cf else 0.0,  # contamination of R/K
        "noise_recall": (len(leaked) / len(noise)) if noise else 0.0,
        "noise_sentences_in_explanation": sorted(leaked),
    }


def _summarize_level(level_records: dict) -> dict:
    rows = list(level_records.values())
    flipped = [r for r in rows if r["flipped"]]            # cases with a counterfactual
    robust = [r for r in flipped if r["noise_robust"]]     # noise didn't change the answer
    def frac_in(rs, key):
        return round(sum(int(r[key]) for r in rs) / len(rs), 4) if rs else None
    def mean(rs, key):
        return round(sum(r[key] for r in rs) / len(rs), 4) if rs else None
    return {
        "rows": len(rows),
        "n_with_counterfactual": len(flipped),
        "n_robust": len(robust),
        "n_fragile": len(flipped) - len(robust),
        # Headline: among robust rows with a counterfactual, how often does noise leak
        # into the explanation, and what fraction of R/K is noise?
        "pct_noise_in_explanation_robust": (round(100 * frac_in(robust, "noise_in_explanation"), 2)
                                            if robust else None),
        "avg_cf_noise_frac_robust": mean(robust, "cf_noise_frac"),
        "pct_noise_in_explanation_all": (round(100 * frac_in(flipped, "noise_in_explanation"), 2)
                                         if flipped else None),
        "avg_cf_size": mean(flipped, "cf_size"),
    }


async def run_noise(args, rag, rag_counter, records):
    noise_percentages = [float(x) for x in args.noise_percentages.split(",") if x.strip()]
    results = {f"noise_level_{int(p * 100)}": {} for p in noise_percentages}

    all_chunks = load_all_graph_chunks(args.dataset)
    pool = list(dict.fromkeys(split_into_players(all_chunks, GRANULARITY)))
    print(f"Foreign-sentence pool: {len(pool)} unique sentences from {len(all_chunks)} KG chunks "
          f"(noise drawn from the WHOLE graph, minus each question's own sentences).")

    for row_idx, (qid, case_type, question, ground_truth) in enumerate(
            tqdm(records, desc="RAGE noise", total=len(records))):
        bottom_up = (case_type == "ff")
        rag_counter.reset()
        _, chunks = await retrieve_chunks(rag, query=question, mode=args.rag_mode, top_k=args.top_k)
        players = split_into_players(chunks, GRANULARITY)
        if len(players) == 0:
            print(f"[{qid}] no retrieved sentences; skipping.")
            continue
        clean_answer = await query(rag, render_context_from_chunks(players), question)

        for p in noise_percentages:
            level_key = f"noise_level_{int(p * 100)}"
            t0 = time.perf_counter()
            noisy_units, noise_set = add_random_noise_chunks(players, pool, p, seed=args.seed + row_idx)

            ev = JudgedEvaluator(rag, noisy_units, question)
            noisy_full_answer = await ev.answer(range(len(noisy_units)))
            noise_robust = (await ev.judge(noisy_full_answer, clean_answer)) != 0

            rel = await relevance_scores(question, noisy_units)
            # ft breaks the CLEAN answer (reference = clean_answer); ff reaches ground_truth.
            # Only the counterfactual SET is needed (RAGE is a set method), not the graded scores.
            _, meta = await rage_combination(
                ev, rel, clean_answer, ground_truth, case_type,
                max_llm_calls=args.max_llm_calls, max_size=args.max_size)

            m = noise_in_explanation_metrics(meta["counterfactual_sentences"], noise_set)
            results[level_key][qid] = {
                "question": question,
                "case_type": case_type,
                "mapped_label": "F->T" if bottom_up else "T->F",
                "ground_truth": ground_truth,
                "clean_answer": clean_answer,
                "noisy_full_answer": noisy_full_answer,
                "noise_pct": p,
                "noise_robust": noise_robust,
                "flipped": meta["flipped"],
                "min_comb_size": meta["min_comb_size"],
                "explanation": "removed_set" if not bottom_up else "retained_set",
                "counterfactual_sentences": meta["counterfactual_sentences"],
                "num_noise_units": len(noise_set),
                "noise_sentences": sorted(noise_set),
                **m,
                "n_items": len(noisy_units),
                "n_tests": meta["n_tests"],
                "query_calls": ev.query_calls,
                "judge_calls": ev.judge_calls,
                "elapsed_time": round(time.perf_counter() - t0, 4),
            }
            print(f"[{qid} | {results[level_key][qid]['mapped_label']} | noise={int(p*100)}%] "
                  f"sents={len(noisy_units)}(+{len(noise_set)}) robust={noise_robust} "
                  f"flipped={meta['flipped']} cf={meta['min_comb_size']} "
                  f"noise_in_expl={m['noise_in_explanation']} cf_noise_frac={m['cf_noise_frac']:.2f}")

    summary = {lvl: _summarize_level(rec) for lvl, rec in results.items() if rec}
    results["__summary__"] = summary

    out = args.output or f"benchmark/results/{args.dataset}_rage_noise.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 84)
    print(f"  RAGE noise resistance — noise_in_explanation (set-based)  (dataset={args.dataset})")
    print("=" * 84)
    print(f"{'noise':<7}{'rows':>6}{'cf':>6}{'robust':>8}{'%noise_in_expl(robust)':>24}{'avg_cf_noise_frac':>20}")
    for p in noise_percentages:
        s = summary.get(f"noise_level_{int(p * 100)}")
        if not s:
            continue
        print(f"{int(p*100):>3}% {'':<2}{s['rows']:>6}{s['n_with_counterfactual']:>6}{s['n_robust']:>8}"
              f"{str(s['pct_noise_in_explanation_robust']):>24}{str(s['avg_cf_noise_frac_robust']):>20}")
    print("=" * 84)
    print(f"Results -> {out}")


async def run_benchmark(args):
    rag_counter = RagCounter()
    import src.retrieve as _retr
    _retr.vllm_model_complete = rag_counter.make_wrapper()
    rag = await initialize_lightrag(working_dir=WORKING_DIRS[args.dataset])

    if args.comparison:
        records = _records_from_comparison(args.comparison, args.num_rows)
    else:
        records = _records_from_csv(args.dataset, args.num_rows)
    await run_noise(args, rag, rag_counter, records)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_rage_noise",
        description="RAGE noise resistance over retrieved sentences: inject foreign sentences "
                    "from the whole KG, run RAGE's combination counterfactual, and check whether "
                    "the noise leaks into the explanation set (ft -> removed set, ff -> retained set) "
                    "— the method's set-based noise_in_explanation metric.")
    p.add_argument("--dataset", choices=DATASETS, default="hotpotqa")
    p.add_argument("--rag-mode", choices=["hybrid", "local", "global", "naive"], default="hybrid")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--num-rows", type=int, default=None, help="Cap on cases (default: all).")
    p.add_argument("--comparison", default=None,
                   help="FF/FT/TF/TT comparison JSON; ft cases -> top-down (check removed set), "
                        "ff cases -> bottom-up (check retained set). Without it, the QA CSV is used "
                        "(all rows as ft / top-down).")
    p.add_argument("--noise-percentages", default="0.1,0.2,0.3,0.5",
                   help="Comma-separated noise fractions in (0, 1).")
    p.add_argument("--max-llm-calls", type=int, default=200,
                   help="Per-(case,noise) RAGE perturbation budget (each = one query + one judge).")
    p.add_argument("--max-size", type=int, default=None,
                   help="Largest RAGE combination subset size searched (default: #sentences).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    asyncio.run(run_benchmark(args))
