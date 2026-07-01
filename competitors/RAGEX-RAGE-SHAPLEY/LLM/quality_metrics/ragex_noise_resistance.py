"""ragex_noise_resistance.py

For each RAG-Ex case (sentence or paragraph granularity):
  1. Loads original_context and importance weights from the stored analysis.
  2. Injects k% noise units (sentences or paragraphs) from the pool.
  3. Judges noisy answer vs original_answer  →  noise_robust check.
  4. If robust: re-runs RAG-Ex remove-one on the NOISY context,
     computes new importance ranking, checks whether noise units
     appear in the top-k of that ranking.
  5. Writes results in the target format.

Usage
-----
  python ragex_noise_resistance.py \
      --input  /path/to/synthetic_remove_sentence_analysis.json \
      --pool-sent  kg_noise_sentences.csv \
      --working-dir /path/to/lightrag_kg/synthetic \
      --noise-levels 0.1,0.3,0.5 \
      --granularity sentence \
      --dataset synthetic \
      --output /path/to/ragex_noise_resistance.json
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))   # .../LLM/quality_metrics
_ROOT = os.path.dirname(os.path.dirname(_HERE))       # .../RAGEX-RAGE-SHAPLEY
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import asyncio
import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from tqdm import tqdm

from retrieval.retrieve import initialize_lightrag
from lightrag.prompt import PROMPTS
from LLM.llm_judge import judge_response
from LLM.quality_metrics.PoolNoiseSelector import PoolNoiseSelector


# ═══════════════════════════════════════════════════════════════════
# Minimal LLM wrapper
# ═══════════════════════════════════════════════════════════════════

class _Evaluator:
    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.rag = None

    async def setup(self) -> "_Evaluator":
        self.rag = await initialize_lightrag(self.working_dir)
        return self

    async def generate(self, question: str, context: str) -> str:
        system_prompt = PROMPTS["rag_response"].format(
            context_data=context,
            response_type="Single Sentence, without references and extra explanations.",
            user_prompt="",
        )
        return await self.rag.llm_model_func(question, system_prompt=system_prompt)

    async def similarity(self, a: str, b: str) -> float:
        vecs = await self.rag.embedding_func([str(a), str(b)])
        return float(cos_sim([vecs[0]], [vecs[1]])[0][0])


# ═══════════════════════════════════════════════════════════════════
# Context helpers
# ═══════════════════════════════════════════════════════════════════

def split_units(context: str, granularity: str) -> list[str]:
    """Split context into sentences or paragraphs."""
    if granularity == "sentence":
        return [s.strip() for s in context.split(".") if s.strip()]
    else:
        return [p.strip() for p in context.split("\n\n") if p.strip()]


def render_context(units: list[str], granularity: str) -> str:
    if granularity == "sentence":
        return ". ".join(units) + "." if units else ""
    else:
        return "\n\n".join(units) if units else ""


# ═══════════════════════════════════════════════════════════════════
# Remove-one loop on a (possibly noisy) context
# ═══════════════════════════════════════════════════════════════════

async def run_remove_one(
    ev: _Evaluator,
    units: list[str],
    unit_ids: list[str],
    question: str,
    baseline_answer: str,
    ground_truth: str,
    granularity: str,
) -> list[dict]:
    records = []
    for i, (uid, unit) in enumerate(zip(unit_ids, units)):
        remaining  = units[:i] + units[i + 1:]
        new_ctx    = render_context(remaining, granularity)
        new_answer = await ev.generate(question, new_ctx)
        is_correct = await judge_response(
            question=question,
            generated_answer=new_answer,
            ground_truth=ground_truth,
        )
        sim               = await ev.similarity(baseline_answer, new_answer)
        importance_weight = 1.0 - sim
        records.append({
            "unit_id":           uid,
            "unit_text":         unit,
            "new_answer":        new_answer,
            "judge_result":      is_correct,
            "is_flip":           is_correct == 0,
            "similarity":        float(sim),
            "importance_weight": float(importance_weight),
        })
        print(f"    [remove-one] {uid} iw={importance_weight:.4f} "
              f"-> '{new_answer[:80]}'")
    return records


def importance_ranking(records: list[dict]) -> list[str]:
    return [
        r["unit_id"]
        for r in sorted(records, key=lambda r: r["importance_weight"], reverse=True)
    ]


# ═══════════════════════════════════════════════════════════════════
# Top-k metrics  (mirrors KG output format)
# ═══════════════════════════════════════════════════════════════════

def topk_metrics(
    ranking: list[str],
    noise_ids: set[str],
    original_ids: set[str],
    importance_by_id: dict[str, float],
    max_k: int = 5,
) -> dict:
    n_objects  = len(ranking)
    n_noise    = len(noise_ids)
    n_original = len(original_ids)

    noise_weights    = [importance_by_id.get(nid, 0.0) for nid in noise_ids]
    original_weights = [importance_by_id.get(oid, 0.0) for oid in original_ids]
    all_weights      = list(importance_by_id.values())

    total_abs       = sum(abs(w) for w in all_weights) or 1.0
    noise_abs       = sum(abs(w) for w in noise_weights)
    noise_ranks     = [ranking.index(nid) for nid in noise_ids if nid in ranking]
    best_noise_rank = min(noise_ranks) if noise_ranks else n_objects

    topk = {}
    for k in range(1, max_k + 1):
        k_eff       = min(k, n_objects)
        topk_set    = set(ranking[:k_eff])
        in_topk     = bool(topk_set & noise_ids)
        num_in_topk = len(topk_set & noise_ids)
        topk[str(k)] = {
            "k_eff":        k_eff,
            "in_topk":      in_topk,
            "num_in_topk":  num_in_topk,
            "frac_in_topk": num_in_topk / k_eff if k_eff else 0.0,
        }

    return {
        "n_objects":       n_objects,
        "n_noise":         n_noise,
        "n_original":      n_original,
        "noise_abs_frac":  noise_abs / total_abs,
        "noise_abs_mass":  noise_abs,
        "total_abs_mass":  total_abs,
        "mean_noise_iw":   float(np.mean(noise_weights)) if noise_weights else 0.0,
        "max_noise_iw":    float(max(noise_weights))     if noise_weights else 0.0,
        "mean_orig_iw":    float(np.mean(original_weights)) if original_weights else 0.0,
        "max_orig_iw":     float(max(original_weights))     if original_weights else 0.0,
        "best_noise_rank": best_noise_rank,
        "topk":            topk,
        "ranking":         ranking,
    }


def compute_summary(level_results: dict, noise_pct: float, max_k: int) -> dict:
    """Aggregate per-case metrics into a summary block for one noise level."""
    cases       = {k: v for k, v in level_results.items() if k != "__summary__"}
    total       = len(cases)
    robust      = [v for v in cases.values() if v.get("noise_robust")]
    n_robust    = len(robust)

    best_ranks  = [
        v["metrics"]["best_noise_rank"]
        for v in robust
        if v.get("metrics") is not None
    ]
    topk_counts: dict[str, int] = {}
    for k in range(1, max_k + 1):
        topk_counts[str(k)] = sum(
            1 for v in robust
            if v.get("metrics") is not None
            and v["metrics"]["topk"][str(k)]["in_topk"]
        )

    return {
        "total_cases":              total,
        "robust_cases":             n_robust,
        "non_robust_cases":         total - n_robust,
        "pct_robust":               round(100 * n_robust / total, 2) if total else 0.0,
        "noise_pct":                noise_pct,
        "mean_best_noise_rank":     round(float(np.mean(best_ranks)), 4)
                                    if best_ranks else None,
        "pct_noise_in_topk":        {
            str(k): round(100 * topk_counts[str(k)] / n_robust, 2)
                    if n_robust else 0.0
            for k in range(1, max_k + 1)
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

async def run_noise_resistance(args: argparse.Namespace) -> None:

    # ── load pool ────────────────────────────────────────────────────
    selector = PoolNoiseSelector()
    selector.load_csv_pools(
        para_csv=args.pool_para,
        sent_csv=args.pool_sent,
    )

    # ── load cases ───────────────────────────────────────────────────
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases", [])
    print(f"Loaded {len(cases)} cases from {args.input}.")

    if not cases:
        print("No cases found. Check --input.")
        return

    # ── boot LLM ─────────────────────────────────────────────────────
    ev = await _Evaluator(args.working_dir).setup()

    # granularity-aware unit ID prefix
    uid_prefix   = "sent" if args.granularity == "sentence" else "para"
    noise_levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]
    output: dict = {}

    for noise_pct in noise_levels:
        level_key = f"noise_level_{int(noise_pct * 100)}"
        output[level_key] = {}
        print(f"\n{'='*60}\nNoise level: {noise_pct*100:.0f}%\n{'='*60}")

        for case in tqdm(cases, desc=level_key):
            cid      = str(case.get("case_id", "?"))
            question = case["question"]
            gt       = case.get("ground_truth", "")
            orig_ans = case.get("original_answer", "")
            ctx      = case.get("original_context", "")

            orig_units = split_units(ctx, args.granularity)

            if len(orig_units) < 1:
                print(f"[{cid}] empty context; skipping.")
                continue

            # ── inject noise, get back positions ──────────────────────
            seed = hash((cid, noise_pct)) % (2 ** 32)
            noisy_ctx, noise_positions = selector.inject_noise(
                ctx,
                noise_percent=noise_pct,
                mode=args.granularity,
                seed=seed,
            )

            if not noise_positions:
                print(f"[{cid}] no noise injected (pool exhausted); skipping.")
                continue

            noisy_units    = split_units(noisy_ctx, args.granularity)
            # granularity-aware IDs: sent_0, sent_1 ... or para_0, para_1 ...
            noisy_unit_ids = [f"{uid_prefix}_{i}" for i in range(len(noisy_units))]

            # position-based tracking — exact regardless of duplicate text
            noise_ids     = {noisy_unit_ids[i] for i in noise_positions
                             if i < len(noisy_unit_ids)}
            original_ids  = set(noisy_unit_ids) - noise_ids
            num_noise_ops = len(noise_ids)

            print(f"\n[{cid}] orig_units={len(orig_units)} "
                  f"noisy_units={len(noisy_units)} "
                  f"injected={num_noise_ops}")

            # ── noise robust check ────────────────────────────────────
            noisy_answer = await ev.generate(question, noisy_ctx)
            noise_score  = await judge_response(
                question=question,
                generated_answer=noisy_answer,
                ground_truth=orig_ans,
            )
            noise_robust = noise_score != 0

            print(f"[{cid}] noise_score={noise_score} "
                  f"noise_robust={noise_robust} "
                  f"noisy_answer='{noisy_answer[:80]}'")

            result: dict = {
                "question":        question,
                "ground_truth":    gt,
                "original_answer": orig_ans,
                "noisy_answer":    noisy_answer,
                "noise_pct":       noise_pct,
                "noise_score":     noise_score,
                "noise_robust":    noise_robust,
                "num_noise_ops":   num_noise_ops,
                "noise_ids":       sorted(noise_ids),
                "noise_texts":     [
                    noisy_units[int(nid.split("_")[1])]
                    for nid in sorted(noise_ids)
                    if int(nid.split("_")[1]) < len(noisy_units)
                ],
            }

            if not noise_robust:
                print(f"[{cid}] not robust to noise; skipping remove-one.")
                result["importance_weights"] = {}
                result["metrics"]            = None
                output[level_key][cid]       = result
                continue

            # ── remove-one on noisy context ───────────────────────────
            records = await run_remove_one(
                ev, noisy_units, noisy_unit_ids,
                question, noisy_answer, gt, args.granularity,
            )

            iw_by_id = {r["unit_id"]: r["importance_weight"] for r in records}
            ranking  = importance_ranking(records)

            result["importance_weights"] = iw_by_id
            result["metrics"]            = topk_metrics(
                ranking          = ranking,
                noise_ids        = noise_ids,
                original_ids     = original_ids,
                importance_by_id = iw_by_id,
                max_k            = args.topk,
            )

            output[level_key][cid] = result
            print(f"[{cid}] done | best_noise_rank="
                  f"{result['metrics']['best_noise_rank']} "
                  f"/ {result['metrics']['n_objects']}")

        # ── per-noise-level summary ───────────────────────────────────
        output[level_key]["__summary__"] = compute_summary(
            output[level_key], noise_pct, args.topk
        )

    # ── write output ──────────────────────────────────────────────────
    out_path = args.output or (
        f"benchmark/results/{args.dataset}_ragex_noise_resistance.json"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults -> {out_path}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ragex_noise_resistance",
        description="Noise-resistance evaluation for RAG-Ex (text-based).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", required=True,
        help="RAG-Ex analysis JSON (sentence or paragraph granularity).",
    )
    p.add_argument(
        "--pool-para", default=None, dest="pool_para",
        help="CSV of paragraph noise pool. Required for --granularity paragraph.",
    )
    p.add_argument(
        "--pool-sent", default=None, dest="pool_sent",
        help="CSV of sentence noise pool. Required for --granularity sentence.",
    )
    p.add_argument(
        "--working-dir", required=True,
        help="LightRAG working directory for LLM + embeddings.",
    )
    p.add_argument(
        "--granularity", choices=["sentence", "paragraph"], default="sentence",
        help="Granularity matching the input file's method.",
    )
    p.add_argument(
        "--noise-levels", default="0.1,0.3,0.5",
        help="Comma-separated noise fractions in (0, 1).",
    )
    p.add_argument(
        "--topk", type=int, default=5,
        help="Max k for top-k noise-in-ranking metrics.",
    )
    p.add_argument(
        "--dataset", default="synthetic",
        help="Tag for the default output filename.",
    )
    p.add_argument(
        "--output", default=None,
        help="Output JSON path.",
    )
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    asyncio.run(run_noise_resistance(args))