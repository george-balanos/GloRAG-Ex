"""Permutation-robustness of RAGE's combination counterfactual — the RAGE analog of
code/src/counterfactuals/permutation_robustness.py.

GloRAG-Ex's permutation_robustness takes each saved counterfactual flip (a perturbed
graph) and tests whether the flip survives when the context serialization is
reordered. This does the same for RAGE: it reads the combination output
(<ds>_combination_analysis.json, written by run_rage.py --mode combination), and for
each FLIPPED case reconstructs the perturbed/reduced source set, samples random
permutations of it, regenerates the answer, and checks (via the SAME LLM judge)
whether the flip persists:

  ft (top-down): perturbed context = the surviving sentences (D_q minus the removed
      counterfactual set). Flip persists iff judge(answer, original_answer) == 0
      (still differs from the full-context answer).
  ff (bottom-up): perturbed context = the retained counterfactual set. Flip persists
      iff judge(answer, ground_truth) == 1 (still reaches the corrective target).

Per case we report num_permutations / num_flipped / flip_stability /
flip_under_all_permutations, and a summary avg_flip_stability /
pct_flip_under_all_permutations — the same fields and semantics as
src/counterfactuals/permutation_robustness.py.

Post-hoc: uses the vLLM generation model + the judge directly (no LightRAG, no
retrieval — the perturbed source set already lives in the combination JSON).

  cd code && ../.venv/bin/python ../competitors/RAGE/run_rage_perm_robustness.py \
      --dataset hotpotqa \
      --input ../all_results/results_rage/hotpotqa/hotpotqa_combination_analysis.json
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

from src.query import build_rag_system_prompt
from src.llm.utils import vllm_model_complete
from src.llm_judge import judge_response
from src.dataset_setup import DATASETS

from chunk_utils import render_context_from_chunks

from tqdm import tqdm
import argparse
import asyncio
import json
import logging
import random

logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("lightrag").setLevel(logging.WARNING)


async def _generate_answer(context: str, question: str) -> str:
    """RAG answer for `context` WITHOUT LightRAG/retrieval — byte-identical to
    query()'s generation (same system prompt, same vLLM model)."""
    return await vllm_model_complete(question, system_prompt=build_rag_system_prompt(context))


def perturbed_sources(case) -> list[str]:
    """The source set whose flip we re-test, reconstructed from the combination case:
    ft -> the surviving sentences (original minus the removed counterfactual set);
    ff -> the retained counterfactual set."""
    cf = list(case.get("counterfactual_sentences") or [])
    if case.get("case_type") == "ff":
        return cf
    # Prefer the explicit sentence list; a sentence may embed a newline, so the legacy
    # original_context.split("\n") fallback can mis-reconstruct D_q (kept for old files).
    sentences = case.get("sentences")
    if sentences is None:
        sentences = [s for s in (case.get("original_context") or "").split("\n") if s.strip()]
    cf_set = set(cf)
    return [s for s in sentences if s not in cf_set]


def random_permutations(items: list[str], count: int, seed: int):
    """Up to `count` distinct random orderings of `items` (the identity included as
    perm 0, like permutation_robustness.py's first sample)."""
    n = len(items)
    rng = random.Random(seed)
    seen, perms = set(), []
    identity = tuple(range(n))
    for attempt in range(count * 10):
        perm = identity if not perms else tuple(rng.sample(range(n), n))
        if perm in seen:
            continue
        seen.add(perm)
        perms.append(list(perm))
        if len(perms) >= count:
            break
    return perms


def load_flip_cases(input_path: str):
    with open(input_path, encoding="utf-8") as f:
        analysis = json.load(f)
    cases = analysis.get("cases", [])
    flips, counts = [], {"total_cases": len(cases), "n_flipped": 0, "n_permutable": 0}
    for c in cases:
        if not c.get("flipped"):
            continue
        counts["n_flipped"] += 1
        if len(perturbed_sources(c)) >= 2:        # need ≥2 sources to reorder
            counts["n_permutable"] += 1
            flips.append(c)
    return flips, counts


async def run(args):
    cases, counts = load_flip_cases(args.input)
    if args.num_cases is not None:
        cases = cases[:args.num_cases]
    print(f"Loaded {counts['total_cases']} combination cases from {args.input}")
    print(f"  flipped={counts['n_flipped']} | permutable (≥2 sources)={counts['n_permutable']}")

    results = {}
    all_flip_count = 0
    stability_sum = 0.0

    for c in tqdm(cases, desc="RAGE comb. permutation robustness", total=len(cases)):
        question = c["question"]
        case_type = c.get("case_type", "ft")
        original_answer = c.get("original_answer", "")
        ground_truth = c.get("ground_truth", "")
        sources = perturbed_sources(c)
        reference = ground_truth if case_type == "ff" else original_answer

        perms = random_permutations(sources, args.count, args.seed)
        per_perm = {}
        n_flipped = 0
        for pi, perm in enumerate(perms):
            ctx = render_context_from_chunks([sources[i] for i in perm])
            new_response = await _generate_answer(ctx, question)
            score = await judge_response(question, new_response, reference)
            persists = (score == 1) if case_type == "ff" else (score == 0)
            n_flipped += int(persists)
            per_perm[f"perm_{pi}"] = {
                "perm": perm,
                "identity": perm == list(range(len(sources))),
                "response": new_response,
                "judge_score": score,
                "flipped": persists,
            }

        n_perms = len(perms)
        stability = n_flipped / n_perms if n_perms else 0.0
        all_flip = (n_perms > 0 and n_flipped == n_perms)
        all_flip_count += int(all_flip)
        stability_sum += stability

        results[str(c.get("case_id"))] = {
            "case_id": c.get("case_id"),
            "case_type": case_type,
            "mapped_label": c.get("mapped_label"),
            "question": question,
            "original_answer": original_answer,
            "ground_truth": ground_truth,
            "counterfactual_sentences": c.get("counterfactual_sentences"),
            "min_comb_size": c.get("min_comb_size"),
            "n_sources_permuted": len(sources),
            "num_permutations": n_perms,
            "num_flipped": n_flipped,
            "flip_stability": round(stability, 4),
            "flip_under_all_permutations": all_flip,
            "permutations": per_perm,
        }
        print(f"[{c.get('case_id')} | {c.get('mapped_label')}] sources={len(sources)} "
              f"flipped={n_flipped}/{n_perms} stability={stability:.2f} all={all_flip}")

    n = len(results) or 1
    summary = {
        "cases": len(results),
        "avg_flip_stability": round(stability_sum / n, 4),
        "pct_flip_under_all_permutations": round(100 * all_flip_count / n, 2),
        **counts,
    }
    results["__summary__"] = summary

    out = args.output or os.path.join(
        os.path.dirname(args.input),
        f"{args.dataset}_combination_permutation_robustness.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print(f"  RAGE combination permutation robustness  ({summary['cases']} flip cases, dataset={args.dataset})")
    print("=" * 64)
    print(f"flipped / permutable cases                        : {counts['n_flipped']} / {counts['n_permutable']}")
    print(f"avg flip-stability (frac of perms still flipping) : {summary['avg_flip_stability']}")
    print(f"cases flipping under ALL permutations             : {summary['pct_flip_under_all_permutations']}%")
    print("=" * 64)
    print(f"Results -> {out}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_rage_perm_robustness",
        description="Permutation robustness of RAGE's combination counterfactual: does the "
                    "judge-verified flip survive reordering of the perturbed source set? "
                    "(RAGE analog of src/counterfactuals/permutation_robustness.py).")
    p.add_argument("--dataset", choices=DATASETS, default="hotpotqa")
    p.add_argument("--input", required=True,
                   help="Path to <ds>_combination_analysis.json (run_rage.py --mode combination).")
    p.add_argument("--count", type=int, default=5, help="Random permutations per case (incl. identity).")
    p.add_argument("--num-cases", type=int, default=None, help="Cap on number of flip cases.")
    p.add_argument("--seed", type=int, default=42, help="Seed for the random permutations.")
    p.add_argument("--output", default=None,
                   help="Output path (default: alongside --input as <ds>_combination_permutation_robustness.json).")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    asyncio.run(run(args))
