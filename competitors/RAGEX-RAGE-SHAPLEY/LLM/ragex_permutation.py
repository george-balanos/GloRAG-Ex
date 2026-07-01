"""ragex_permutation.py – sentence-level permutation robustness for RAG-Ex.

Reads original_context + original_answer from RAG-Ex output JSONs (same format
produced by KGCasePerturbationEvaluator). No LightRAG retrieval is performed.

For each case:
  1. Splits original_context into sentences (same "." split as perturb()).
  2. Keeps the original ordering + samples --num-perms random permutations.
  3. For each ordering:
       - Original order  : reuses the stored original_answer (no generation).
       - Permuted orders : generates a fresh answer from the rearranged context,
                          then records whether the answer changed.
       For EVERY ordering, runs the full remove-one-sentence perturbation loop:
         * removes each sentence in turn
         * generates a new answer, judges it vs ground_truth
         * computes similarity to that ordering's baseline answer
         * derives importance_weight = 1 − similarity
  4. Ranks sentences by importance_weight and computes Kendall-tau ranking
     stability + answer-change rate across all orderings.

Usage
-----
  python ragex_permutation.py \\
      --input-dir  /path/to/ragex_output/ \\
      --questions-file /path/to/questions.json \\
      --working-dir /path/to/lightrag_kg/   # needed for LLM + embedding \\
      --dataset musique \\
      --num-perms 5 \\
      --output benchmark/results/musique_ragex_permutation.json

  # Filter from a directory of question-list JSONs instead of a single file:
  python ragex_permutation.py \\
      --input-dir  /path/to/ragex_output/ \\
      --questions-dir /path/to/question_lists/ \\
      --working-dir /path/to/lightrag_kg/ \\
      --dataset musique
"""
from __future__ import annotations

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import asyncio
import glob
import itertools
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau
from tqdm import tqdm

# ── project-local imports (same layout as the existing evaluator) ─────────────
# Adjust the import path if KGCasePerturbationEvaluator lives elsewhere.
# try:
#     from retrieval.retrieve import initialize_lightrag          # type: ignore
#     from lightrag.prompt import PROMPTS                         # type: ignore
#     from LLM.llm_judge import judge_response                    # type: ignore
# except ImportError:
#     # Allow the file to be parsed/linted even outside the project tree.
#     initialize_lightrag = None  # type: ignore
#     PROMPTS = {}                # type: ignore
#     judge_response = None       # type: ignore

from retrieval.retrieve import initialize_lightrag
from lightrag.prompt import PROMPTS                         # type: ignore
from LLM.llm_judge import judge_response                    # type: ignore

# ═════════════════════════════════════════════════════════════════════════════
# Sentence helpers
# ═════════════════════════════════════════════════════════════════════════════

def split_sentences(context: str) -> list[str]:
    """Split a context string into non-empty sentences.

    Mirrors the logic in KGCasePerturbationEvaluator.perturb(method='remove_sentence'):
      full_text.split(".")  →  strip  →  drop empty
    """
    return [s.strip() for s in context.split(".") if s.strip()]


def sentence_id(idx: int) -> str:
    """Stable identifier tied to the sentence's position in the original context."""
    return f"sent_{idx}"


def render_context(sentences: list[str]) -> str:
    """Rejoin a sentence list into a context string (mirrors perturb() output)."""
    return ". ".join(sentences) + "." if sentences else ""


def random_sentence_permutations(
    sentences: list[str],
    count: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Return up to `count` distinct random permutations of `sentences`.

    Each entry:
        perm_id  : str  – "perm_0", "perm_1", …
        perm     : list[int]  – new index order (into original sentences)
        sentences: list[str]  – reordered sentence texts
        context  : str        – joined context string
    """
    rng = random.Random(seed)
    n = len(sentences)
    seen: set[tuple[int, ...]] = set()
    perms: list[dict] = []
    max_tries = count * 30
    tries = 0
    while len(perms) < count and tries < max_tries:
        tries += 1
        idx = list(range(n))
        rng.shuffle(idx)
        key = tuple(idx)
        if key in seen:
            continue
        seen.add(key)
        reordered = [sentences[i] for i in idx]
        perms.append({
            "perm_id":   f"perm_{len(perms)}",
            "perm":      idx,
            "sentences": reordered,
            "context":   render_context(reordered),
        })
    return perms


# ═════════════════════════════════════════════════════════════════════════════
# Ranking / stability metrics  (ported from Shapley permutation_stats)
# ═════════════════════════════════════════════════════════════════════════════

def permutation_stats(
    rankings_by_perm: dict[str, list[str]],
    sent_ids: list[str],
    top_k: int,
) -> dict:
    """Compute Kendall-tau + top-k stability across all permutation rankings.

    Parameters
    ----------
    rankings_by_perm : {perm_id -> ordered list of sent_ids, most important first}
    sent_ids         : canonical list of all sentence ids (original order)
    top_k            : k for the set / positional stability checks
    """
    perm_ids = list(rankings_by_perm.keys())
    rank_index = {
        p: {sid: i for i, sid in enumerate(rankings_by_perm[p])}
        for p in perm_ids
    }

    # Pairwise Kendall-tau over rank positions
    taus: list[float] = []
    exact_match = True
    if len(perm_ids) >= 2 and len(sent_ids) >= 2:
        for a, b in itertools.combinations(perm_ids, 2):
            ra = [rank_index[a][s] for s in sent_ids]
            rb = [rank_index[b][s] for s in sent_ids]
            tau, _ = kendalltau(ra, rb)
            taus.append(float(tau) if tau == tau else float("nan"))  # nan-safe
            if rankings_by_perm[a] != rankings_by_perm[b]:
                exact_match = False

    mean_tau = float(np.nanmean(taus)) if taus else float("nan")
    min_tau  = float(np.nanmin(taus))  if taus else float("nan")

    # Set stability
    top1_set   = {rankings_by_perm[p][0] for p in perm_ids} if sent_ids else set()
    topk_sets  = [frozenset(rankings_by_perm[p][:top_k]) for p in perm_ids] if sent_ids else []
    top1_stable = len(top1_set) == 1
    topk_stable = len(set(topk_sets)) == 1 if topk_sets else True

    # Positional stability: for each of the top-k ranks, does the SAME sentence
    # occupy that exact spot across every permutation?
    k_eff = min(top_k, len(sent_ids))
    position_stable = [
        len({rankings_by_perm[p][i] for p in perm_ids}) == 1
        for i in range(k_eff)
    ]

    # Per-sentence importance spread
    per_sentence: dict[str, dict] = {}
    for sid in sent_ids:
        rank_vals = [rank_index[p][sid] for p in perm_ids]
        per_sentence[sid] = {
            "mean_rank": float(np.mean(rank_vals)),
            "std_rank":  float(np.std(rank_vals)),
            "min_rank":  int(min(rank_vals)),
            "max_rank":  int(max(rank_vals)),
        }

    return {
        "num_permutations":       len(perm_ids),
        "mean_kendall_tau":       mean_tau,
        "min_kendall_tau":        min_tau,
        "exact_ranking_match":    exact_match,
        "top1_stable":            top1_stable,
        f"top{top_k}_stable":     topk_stable,
        "topk_positions_checked": k_eff,
        "topk_position_matches":  int(sum(position_stable)),
        "topk_position_stable":   position_stable,
        "per_sentence_rank":      per_sentence,
        "rankings":               rankings_by_perm,
    }


# ═════════════════════════════════════════════════════════════════════════════
# A lightweight stand-alone evaluator (no retrieval, reuses LLM/embed only)
# ═════════════════════════════════════════════════════════════════════════════

class _PermEvaluator:
    """Minimal wrapper around LightRAG's LLM + embedding, no retrieval."""

    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.rag = None

    async def setup(self) -> "_PermEvaluator":
        self.rag = await initialize_lightrag(self.working_dir)
        return self

    # ── generation ────────────────────────────────────────────────────────────
    async def generate(self, question: str, context: str) -> str:
        """Generate a single-sentence RAG answer using the QA_PROMPT."""
        system_prompt = PROMPTS["rag_response"].format(
            context_data=context,
            response_type="Single Sentence, without references and extra explanations.",
            user_prompt="",
        )
        return await self.rag.llm_model_func(question, system_prompt=system_prompt)

    # ── similarity ────────────────────────────────────────────────────────────
    async def similarity(self, text_a: str, text_b: str) -> float:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim  # type: ignore
        vecs = await self.rag.embedding_func([str(text_a), str(text_b)])
        return float(cos_sim([vecs[0]], [vecs[1]])[0][0])


# ═════════════════════════════════════════════════════════════════════════════
# Core per-permutation remove-one loop
# ═════════════════════════════════════════════════════════════════════════════

async def run_remove_one(
    ev: _PermEvaluator,
    ordered_sentences: list[str],
    ordered_sent_ids: list[str],
    question: str,
    baseline_answer: str,
    ground_truth: str,
) -> tuple[list[dict], int]:
    """Remove each sentence in turn; judge + measure similarity to baseline_answer.

    Returns
    -------
    records   : one entry per sentence with remove-one stats
    llm_calls : number of LLM calls made (2 per sentence: generate + judge)
    """
    records: list[dict] = []
    calls = 0

    for i, (sid, sent) in enumerate(zip(ordered_sent_ids, ordered_sentences)):
        remaining  = ordered_sentences[:i] + ordered_sentences[i + 1:]
        new_ctx    = render_context(remaining)

        new_answer = await ev.generate(question, new_ctx)
        calls += 1

        is_correct = await judge_response(
            question=question,
            generated_answer=new_answer,
            ground_truth=ground_truth,
        )
        calls += 1

        sim               = await ev.similarity(baseline_answer, new_answer)
        importance_weight = 1.0 - sim

        records.append({
            "sentence_id":        sid,
            "removed_sentence":   sent,
            "new_answer":         new_answer,
            "judge_result":       is_correct,
            "is_flip":            is_correct == 0,
            "similarity":         float(sim),
            "importance_weight":  float(importance_weight),
        })

    return records, calls


def importance_ranking(records: list[dict]) -> list[str]:
    """Sentence ids ranked by descending importance_weight (highest = most important)."""
    return [
        r["sentence_id"]
        for r in sorted(records, key=lambda r: r["importance_weight"], reverse=True)
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Case loader
# ═════════════════════════════════════════════════════════════════════════════

def _load_questions(questions_file: str | None, questions_dir: str | None) -> set[str] | None:
    """Build the question filter set from a file and/or a directory of JSON files."""
    if questions_file is None and questions_dir is None:
        return None
    qs: set[str] = set()
    if questions_file:
        with open(questions_file, encoding="utf-8") as f:
            raw = json.load(f)
        # Accept: list of strings  OR  list of {question: ...} dicts
        for item in raw:
            if isinstance(item, str):
                qs.add(item)
            elif isinstance(item, dict):
                qs.add(item.get("question", ""))
    if questions_dir:
        for fp in sorted(glob.glob(os.path.join(questions_dir, "**", "*.json"), recursive=True)):
            try:
                with open(fp, encoding="utf-8") as f:
                    raw = json.load(f)
                items = raw if isinstance(raw, list) else raw.get("questions", [raw])
                for item in items:
                    if isinstance(item, str):
                        qs.add(item)
                    elif isinstance(item, dict):
                        qs.add(item.get("question", ""))
            except Exception as e:
                print(f"  skip question file {fp}: {e}")
    qs.discard("")
    return qs or None


# def load_ragex_cases(
#     input_dir: str,
#     questions: set[str] | None = None,
# ) -> list[tuple[str, dict]]:
#     """Load case dicts from RAG-Ex output JSONs under input_dir.

#     Supports both formats:
#       • {"cases": [...]}   – wrapper produced by KGCasePerturbationEvaluator
#       • a bare case dict   – single-case JSON (with "original_context" at top level)

#     A case is included only if it has a non-empty original_context.
#     """
#     files = sorted(glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True))
#     cases: list[tuple[str, dict]] = []
#     for fp in files:
#         try:
#             with open(fp, encoding="utf-8") as f:
#                 data = json.load(f)
#         except Exception as e:
#             print(f"  skip {fp}: {e}")
#             continue

#         if isinstance(data, dict):
#             raw_cases = data.get("cases") or ([data] if "original_context" in data else [])
#         else:
#             raw_cases = []

#         for c in raw_cases:
#             if not isinstance(c, dict):
#                 continue
#             if not (c.get("original_context") or "").strip():
#                 continue
#             if questions is not None and c.get("question") not in questions:
#                 continue
#             cases.append((fp, c))

#     return cases

def load_ragex_cases(
    input_dir: str,
    questions: set[str] | None = None,
) -> list[tuple[str, dict]]:
    """Load case dicts from RAG-Ex output JSONs under input_dir.

    Supports both formats:
      • {"cases": [...]}   – wrapper produced by KGCasePerturbationEvaluator
      • a bare case dict   – single-case JSON (with "original_context" at top level)

    A case is included only if it has a non-empty original_context.
    """
    # files = sorted(glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True))

    if os.path.isfile(input_dir):
        files = [input_dir]
    else: 
        files = sorted(glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True))


    cases: list[tuple[str, dict]] = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  skip {fp}: {e}")
            continue

        if isinstance(data, dict):
            raw_cases = data.get("cases") or ([data] if "original_context" in data else [])
        else:
            raw_cases = []

        for c in raw_cases:
            if not isinstance(c, dict):
                continue
            if not (c.get("original_context") or "").strip():
                continue
            if questions is not None and c.get("question") not in questions:
                continue
            cases.append((fp, c))

    return cases


# ═════════════════════════════════════════════════════════════════════════════
# Main permutation loop
# ═════════════════════════════════════════════════════════════════════════════

async def run_permutation(args: argparse.Namespace) -> None:
    # ── load question filter ──────────────────────────────────────────────────
    questions = _load_questions(
        getattr(args, "questions_file", None),
        getattr(args, "questions_dir", None),
    )
    if questions is not None:
        print(f"Question filter: {len(questions)} question(s) loaded.")

    # ── load cases ────────────────────────────────────────────────────────────
    cases = load_ragex_cases(args.input_dir, questions=questions)
    if args.num_rows is not None:
        cases = cases[: args.num_rows]
    print(f"Loaded {len(cases)} case(s) from {args.input_dir}.")

    if not cases:
        print("No cases found. Check --input-dir and --questions-file / --questions-dir.")
        return

    # ── boot LLM + embedding (no retrieval) ───────────────────────────────────
    ev = await _PermEvaluator(args.working_dir).setup()

    # ── result accumulators ───────────────────────────────────────────────────
    results: dict[str, dict] = {}
    tau_list:       list[float] = []
    mintau_list:    list[float] = []
    top1_list:      list[bool]  = []
    topk_list:      list[bool]  = []
    exact_list:     list[bool]  = []
    posmatch_list:  list[int]   = []
    poschecked_list:list[int]   = []
    changed_list:   list[int]   = []
    global_calls    = 0

    for fp, case in tqdm(cases, desc="RAG-Ex permutation", total=len(cases)):
        cid      = str(case.get("case_id", os.path.splitext(os.path.basename(fp))[0]))
        question = case["question"]
        gt       = case.get("ground_truth", "")
        orig_ans = case.get("original_answer", "")
        ctx      = case.get("original_context", "")

        sents = split_sentences(ctx)
        if len(sents) < 2:
            print(f"[{cid}] only {len(sents)} sentence(s) after splitting; skipping.")
            continue

        sent_ids = [sentence_id(i) for i in range(len(sents))]
        rankings_by_perm: dict[str, list[str]] = {}
        perm_records: list[dict] = []
        row_t0 = time.perf_counter()

        # ── (A) Original sentence order ───────────────────────────────────────
        # Reuse the stored original_answer (no generation), matching Shapley's
        # run_permutation_from_json "original" entry.
        records, calls = await run_remove_one(
            ev, sents, sent_ids, question, orig_ans, gt
        )
        global_calls += calls
        ranking = importance_ranking(records)
        rankings_by_perm["original"] = ranking
        perm_records.append({
            "perm_id":        "original",
            "perm":           list(range(len(sents))),
            "sentence_order": sent_ids,
            "context":        render_context(sents),
            "target_answer":  orig_ans,
            "generated":      False,
            "answer_changed": False,
            "gen_time_s":     0.0,
            "remove_one":     records,
            "ranking":        ranking,
            "llm_calls":      calls,
        })

        # ── (B) Random sentence permutations ──────────────────────────────────
        perms     = random_sentence_permutations(sents, count=args.num_perms, seed=args.seed)
        n_changed = 0

        for p in perms:
            # Generate a fresh answer from the permuted context
            gt0     = time.perf_counter()
            new_ans = await ev.generate(question, p["context"])
            gen_t   = round(time.perf_counter() - gt0, 4)
            global_calls += 1

            answer_changed = new_ans.strip() != orig_ans.strip()
            if answer_changed:
                n_changed += 1

            # sent_ids in the order they appear in this permutation
            perm_sent_ids = [sent_ids[i] for i in p["perm"]]

            records, calls = await run_remove_one(
                ev, p["sentences"], perm_sent_ids, question, new_ans, gt
            )
            global_calls += calls

            ranking = importance_ranking(records)
            rankings_by_perm[p["perm_id"]] = ranking

            perm_records.append({
                "perm_id":        p["perm_id"],
                "perm":           p["perm"],
                "sentence_order": perm_sent_ids,
                "context":        p["context"],
                "target_answer":  new_ans,
                "generated":      True,
                "answer_changed": answer_changed,
                "gen_time_s":     gen_t,
                "remove_one":     records,
                "ranking":        ranking,
                "llm_calls":      calls,
            })

        # ── stats ─────────────────────────────────────────────────────────────
        row_time = round(time.perf_counter() - row_t0, 4)
        stats    = permutation_stats(rankings_by_perm, sent_ids, args.topk_stable)
        topk_key = f"top{args.topk_stable}_stable"

        results[cid] = {
            "filepath":           fp,
            "case_type":          case.get("case_type"),
            "mapped_label":       case.get("mapped_label"),
            "method":             case.get("method", "remove_sentence"),
            "question":           question,
            "ground_truth":       gt,
            "original_answer":    orig_ans,
            "n_sentences":        len(sents),
            "sentence_ids":       sent_ids,
            # sentences stored once for reference
            "sentences":          {sentence_id(i): s for i, s in enumerate(sents)},
            "num_permutations":   len(perm_records),
            "num_answer_changed": n_changed,
            "permutations":       perm_records,
            "stats":              stats,
            "total_llm_calls":    sum(r["llm_calls"] for r in perm_records) + n_changed + 1,
            "total_time_s":       row_time,
        }

        # accumulators
        tau_list.append(stats["mean_kendall_tau"])
        mintau_list.append(stats["min_kendall_tau"])
        top1_list.append(stats["top1_stable"])
        topk_list.append(stats.get(topk_key, False))
        exact_list.append(stats["exact_ranking_match"])
        posmatch_list.append(stats["topk_position_matches"])
        poschecked_list.append(stats["topk_positions_checked"])
        changed_list.append(n_changed)

        print(
            f"[{cid}] sents={len(sents)} perms={len(perm_records)} "
            f"answer_changed={n_changed}/{args.num_perms} "
            f"meanτ={stats['mean_kendall_tau']:.3f} "
            f"minτ={stats['min_kendall_tau']:.3f} "
            f"top1_stable={stats['top1_stable']} "
            f"exact={stats['exact_ranking_match']}"
        )

    # ── aggregate summary ─────────────────────────────────────────────────────
    rows = len(results)
    summary: dict = {
        "rows":                       rows,
        "num_perms_per_case":         args.num_perms,
        "topk_stable_k":              args.topk_stable,
        "avg_mean_kendall_tau":       float(np.nanmean(tau_list))    if tau_list else float("nan"),
        "avg_min_kendall_tau":        float(np.nanmean(mintau_list)) if mintau_list else float("nan"),
        "pct_top1_stable":            round(100 * sum(top1_list) / rows, 2)  if rows else 0.0,
        "pct_topk_stable":            round(100 * sum(topk_list) / rows, 2)  if rows else 0.0,
        "pct_exact_ranking_match":    round(100 * sum(exact_list) / rows, 2) if rows else 0.0,
        "avg_topk_position_matches":  round(sum(posmatch_list) / rows, 4)    if rows else 0.0,
        "avg_topk_positions_checked": round(sum(poschecked_list) / rows, 4)  if rows else 0.0,
        "avg_answer_changed_per_row": round(sum(changed_list) / rows, 4)     if rows else 0.0,
        "total_llm_calls":            global_calls,
    }
    results["__summary__"] = summary

    # ── write output ──────────────────────────────────────────────────────────
    out = args.output or f"benchmark/results/{args.dataset}_ragex_permutation.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _print_summary(summary, args)
    print(f"Results -> {out}")


def _print_summary(s: dict, args: argparse.Namespace) -> None:
    print("\n" + "=" * 64)
    print(f"  RAG-Ex sentence-permutation  ({s['rows']} rows, dataset={args.dataset})")
    print("=" * 64)
    print(f"avg mean Kendall-tau   : {s['avg_mean_kendall_tau']:.4f}")
    print(f"avg min  Kendall-tau   : {s['avg_min_kendall_tau']:.4f}")
    print(f"avg answer-changed     : {s['avg_answer_changed_per_row']} / {args.num_perms} permuted orders")
    print(f"top-1 stable rows      : {s['pct_top1_stable']}%")
    print(f"top-{args.topk_stable} stable rows      : {s['pct_topk_stable']}% (same set)")
    print(f"top-{args.topk_stable} same-position    : "
          f"{s['avg_topk_position_matches']}/{s['avg_topk_positions_checked']} ranks (avg)")
    print(f"exact-ranking rows     : {s['pct_exact_ranking_match']}%")
    print(f"total LLM calls        : {s['total_llm_calls']}")
    print("=" * 64)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ragex_permutation",
        description="RAG-Ex sentence-level permutation robustness (no retrieval).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── input ─────────────────────────────────────────────────────────────────
    # p.add_argument(
    #     "--input-dir", required=True,
    #     help="Directory of RAG-Ex output JSONs (recursively globbed). "
    #          "Supports the {'cases': [...]} wrapper format and bare case dicts.",
    # )
    p.add_argument(
        "--input", required=True,
        dest="input_dir",          # keep the internal name so nothing else breaks
        help="Path to a single RAG-Ex output JSON, or a directory of JSONs "
            "(recursively globbed). Supports the {'cases': [...]} wrapper "
            "format and bare case dicts.",
    )
    p.add_argument(
        "--working-dir", required=True,
        help="LightRAG working directory used to initialise the LLM and "
             "embedding models (no retrieval is performed, but the model "
             "handles are loaded from here).",
    )

    # ── question filter (at least one optional) ───────────────────────────────
    q_grp = p.add_argument_group("question filter (use one or both)")
    q_grp.add_argument(
        "--questions-file", default=None,
        help="JSON file containing a list of question strings (or dicts with "
             "a 'question' key).  Only matching cases are processed.",
    )
    q_grp.add_argument(
        "--questions-dir", default=None,
        help="Directory of JSON files; all question strings found inside are "
             "collected and used as the filter (union with --questions-file).",
    )

    # ── permutation settings ──────────────────────────────────────────────────
    p.add_argument("--num-perms",   type=int, default=5,
                   help="Number of random sentence orderings sampled per case "
                        "(in addition to the original order).")
    p.add_argument("--topk-stable", dest="topk_stable", type=int, default=2,
                   help="k used for the top-k set / positional stability checks.")
    p.add_argument("--seed",        type=int, default=42,
                   help="RNG seed for permutation sampling.")

    # ── run control ───────────────────────────────────────────────────────────
    p.add_argument("--dataset",  default="synthetic",
                   help="Dataset tag used in the default output filename.")
    p.add_argument("--num-rows", type=int, default=None,
                   help="Cap on the number of cases processed (default: all).")
    p.add_argument("--output",   default=None,
                   help="Path for the output JSON "
                        "(default: benchmark/results/<dataset>_ragex_permutation.json).")

    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    asyncio.run(run_permutation(args))