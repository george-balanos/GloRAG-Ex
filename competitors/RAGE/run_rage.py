"""RAGE — "RAGE Against the Machine: Retrieval-Augmented LLM Explanations"
(Rorseth, Godfrey, Golab, Srivastava, Szlichta, 2024; arXiv:2405.13000), implemented
as a competitor baseline that plugs into the same experimental setup as GloRAG-Ex and
the other text-span explainers (RAG-Ex, Shapley-Text).

This is the RAGE *method* (the counterfactual SEARCH), NOT the Plotly demo/dashboard
(no pie chart / answer rules / optimum-permutation assignment problem). It implements
BOTH counterfactual functions of the paper, over retrieved SENTENCES (text spans,
like Shapley-Text --granularity sentence), with flips decided by the SAME LLM judge
the method uses (src.llm_judge.judge_response) — never answer-similarity:

  --mode combination  (REGISTERED for correctness + noise robustness)
      Minimal-combination counterfactual. ft -> top-down: remove a minimal subset of
      D_q so the answer no longer matches the full-context answer a
      (`judge_response(q, perturbed, a) == 0`). ff -> bottom-up: retain a minimal
      subset so the answer reaches the corrective target ground_truth
      (`judge_response(q, retained, gt) == 1`). Combinations are tried in increasing
      subset size; equal-size combos in decreasing Sum S(q,s) (no normalisation). The
      search stops at the first flip or the LLM-call budget (--max-llm-calls, 200).
      This mirrors code/src/counterfactuals/generate.py (judge-decided flips) and
      Shapley-Text (sentence units).

  --mode permutation  (position-bias diagnostic; UNSCORED in the comparison tables)
      RAGE's order function: reorder D_q and evaluate permutations in decreasing
      Kendall's-Tau similarity to the given order until the answer changes
      (`judge_response(q, reordered, a) == 0`). Reveals context position bias and
      subsumes, for RAGE, the permutation-robustness question (does the answer survive
      reordering?). No supporting-facts ground truth and no baseline -> diagnostic only.

Per-sentence importance the correctness eval ranks (judge-based, similarity-free):
      score(s) = W_FLIP * cf(s) + S(q,s)
  cf(s) = 1 iff s is in the minimal judge-verified counterfactual (the "citation").
  S(q,s) = cosine(embed(q), embed(s)) with the retriever's all-MiniLM-L6-v2 — the
           paper's retrieval-relevance method (NOT answer similarity). Orders
           equal-size combinations and breaks ranking ties. W_FLIP=2 (S in [0,1])
           keeps the counterfactual sentences above all others.

Output mirrors the RAG-Ex {cases:[...]} schema so src.correctness.evaluate --method
ragex scores it as-is (removed_item_importance = {sentence_text: score}). CWD must be
code/ so relative dataset paths resolve:

  cd code && ../.venv/bin/python ../competitors/RAGE/run_rage.py \
      --dataset hotpotqa --mode combination \
      --comparison <path>/comparison_hotpotqa.json \
      --out-dir ../all_results/results_rage/hotpotqa
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

from run_shapley import RagCounter, load_qa
from run_shapley_text import split_into_players, unit_id  #  (sentence helper)

from src.retrieve import initialize_lightrag, sentence_transformer_embed
from src.query import query
from src.llm_judge import judge_response
from src.counterfactuals.utils import cosine_similarity_norm
from src.dataset_setup import WORKING_DIRS, QA_CSV_PATHS, DATASETS

from chunk_utils import retrieve_chunks, render_context_from_chunks

from tqdm import tqdm
import argparse
import asyncio
import itertools
import json
import logging
import time

import numpy as np

logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("nano-vectordb").setLevel(logging.WARNING)
logging.getLogger("lightrag").setLevel(logging.WARNING)

GRANULARITY = "sentence"   # RAGE "sources" are sentences (Shapley-Text sentence analog)
W_FLIP = 2.0               # lifts judge-verified counterfactual sentences above S in [0,1]


# ── Retrieval relevance S (paper's method 2) ─────────────────────────────────
async def relevance_scores(question: str, sentences: list[str]) -> list[float]:
    """S(q,s) = cosine(embed(q), embed(s)) with the retriever's embedding model.
    Used to order equal-size combinations and break output-ranking ties."""
    vecs = await sentence_transformer_embed([question] + sentences)
    qv = np.asarray(vecs[0])
    return [cosine_similarity_norm(qv, np.asarray(v)) for v in vecs[1:]]


# ── Judged subset/permutation evaluation (cached, call-counted) ──────────────
class JudgedEvaluator:
    """L(q, sources) + the shared LLM judge, with answer caches and call counters.
    `answer` is order-insensitive (combinations, cached by frozenset); `answer_ordered`
    is order-sensitive (permutations, cached by tuple)."""

    def __init__(self, rag, sentences: list[str], question: str):
        self.rag = rag
        self.sentences = sentences
        self.question = question
        self._set_cache: dict[frozenset, str] = {}
        self._ord_cache: dict[tuple, str] = {}
        self.query_calls = 0
        self.judge_calls = 0

    async def answer(self, retained_idx) -> str:
        key = frozenset(retained_idx)
        if key not in self._set_cache:
            ctx = render_context_from_chunks([self.sentences[i] for i in sorted(key)])
            self._set_cache[key] = await query(self.rag, ctx, self.question)
            self.query_calls += 1
        return self._set_cache[key]

    async def answer_ordered(self, order) -> str:
        key = tuple(order)
        if key not in self._ord_cache:
            ctx = render_context_from_chunks([self.sentences[i] for i in order])
            self._ord_cache[key] = await query(self.rag, ctx, self.question)
            self.query_calls += 1
        return self._ord_cache[key]

    async def judge(self, answer: str, reference: str) -> int:
        self.judge_calls += 1
        return await judge_response(self.question, answer, reference)


# ── Combination-based counterfactual (correctness + noise) ───────────────────
async def rage_combination(ev: JudgedEvaluator, rel, original_answer, ground_truth,
                           case_type, max_llm_calls=200, max_size=None):
    """Top-down (ft) removal / bottom-up (ff) retention minimal-combination search,
    flips decided by the LLM judge (mirrors generate.py): ft flips at score==0 vs the
    full-context answer; ff succeeds at score==1 vs ground_truth. Each perturbation
    (one query + one judge) consumes one unit of the LLM-call budget.

    Returns (removed_item_importance: {sentence: score}, meta)."""
    sentences = ev.sentences
    n = len(sentences)
    all_idx = list(range(n))
    bottom_up = (case_type == "ff")
    max_size = n if max_size is None else min(max_size, n)

    cf: set[int] = set()
    n_tests = 0
    flipped = False
    min_comb_size = None

    for m in range(1, max_size + 1):
        if flipped or n_tests >= max_llm_calls:
            break
        # The "combination" scored for ordering is the removed set (ft) / retained
        # set (ff); equal-size combos are tried by decreasing Sum S (no normalisation).
        combos = sorted(itertools.combinations(all_idx, m),
                        key=lambda c: sum(rel[i] for i in c), reverse=True)
        for combo in combos:
            if n_tests >= max_llm_calls:
                break
            retained = set(combo) if bottom_up else (set(all_idx) - set(combo))
            ans = await ev.answer(retained)
            n_tests += 1
            if bottom_up:
                hit = (await ev.judge(ans, ground_truth)) == 1
            else:
                hit = (await ev.judge(ans, original_answer)) == 0
            if hit:
                cf, flipped, min_comb_size = set(combo), True, m
                break

    importance = {sentences[i]: (W_FLIP if i in cf else 0.0) + max(0.0, rel[i])
                  for i in range(n)}
    meta = {
        "counterfactual_sentences": [sentences[i] for i in sorted(cf)],
        "min_comb_size": min_comb_size,
        "n_tests": n_tests,
        "flipped": flipped,
        "search": "bottom_up" if bottom_up else "top_down",
    }
    return importance, meta


# ── Permutation-based counterfactual (RAGE order function; position diagnostic) ─
def kendall_tau(perm) -> float:
    """Kendall's-Tau rank correlation of `perm` vs the identity order in [-1, 1]."""
    n = len(perm)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for a in range(n):
        for b in range(a + 1, n):
            if perm[a] < perm[b]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def candidate_permutations(n: int, cap: int):
    """Permutations to test, most-similar-first (decreasing Kendall's Tau). For small
    n, all n! non-identity orders; otherwise all single pairwise swaps (the most
    similar non-identity orders) so k! does not explode at sentence granularity."""
    idx = list(range(n))
    if n <= cap:
        perms = [list(p) for p in itertools.permutations(idx) if list(p) != idx]
    else:
        perms = []
        for a, b in itertools.combinations(idx, 2):
            p = idx[:]
            p[a], p[b] = p[b], p[a]
            perms.append(p)
    perms.sort(key=kendall_tau, reverse=True)
    return perms


async def rage_permutation(ev: JudgedEvaluator, original_answer,
                           max_llm_calls=200, perm_cap=6):
    """RAGE's permutation function: the most-similar reordering of D_q that changes
    the answer (judge vs the given-order answer). Quantifies position (in)stability —
    a flip means the answer is order-sensitive. Importance per sentence = its
    displacement in the flipping permutation (a diagnostic, not a provenance score).

    Returns (removed_item_importance, meta)."""
    sentences = ev.sentences
    n = len(sentences)
    importance = {s: 0.0 for s in sentences}
    n_tests = 0
    flipped = False
    flip_perm = None
    flip_tau = None

    for perm in candidate_permutations(n, perm_cap):
        if n_tests >= max_llm_calls:
            break
        ans = await ev.answer_ordered(perm)
        n_tests += 1
        if (await ev.judge(ans, original_answer)) == 0:       # answer changed by order
            flipped, flip_perm, flip_tau = True, perm, kendall_tau(perm)
            denom = max(1, n - 1)
            for new_pos, orig_idx in enumerate(perm):
                importance[sentences[orig_idx]] = abs(new_pos - orig_idx) / denom
            break

    meta = {
        "position_sensitive": flipped,
        "position_stable": not flipped,
        "flip_kendall_tau": flip_tau,
        "flip_perm": flip_perm,
        "counterfactual_sentences": ([sentences[i] for i in flip_perm]
                                     if flip_perm else []),
        "min_comb_size": None,
        "n_tests": n_tests,
        "flipped": flipped,
        "search": "permutation",
    }
    return importance, meta


# ── Per-case driver ──────────────────────────────────────────────────────────
async def _process_case(args, ev_factory, qid, case_type, question, ground_truth):
    mapped_label = "F->T" if case_type == "ff" else "T->F"
    _, chunks = await retrieve_chunks(args.rag, query=question, mode=args.rag_mode, top_k=args.top_k)
    sentences = split_into_players(chunks, GRANULARITY)
    if len(sentences) == 0:
        return None
    ev = JudgedEvaluator(args.rag, sentences, question)
    original_answer = await ev.answer(range(len(sentences)))   # full-context answer a (setup)

    if args.mode == "permutation":
        importance, meta = await rage_permutation(
            ev, original_answer, max_llm_calls=args.max_llm_calls, perm_cap=args.perm_cap)
    else:
        rel = await relevance_scores(question, sentences)
        importance, meta = await rage_combination(
            ev, rel, original_answer, ground_truth, case_type,
            max_llm_calls=args.max_llm_calls, max_size=args.max_size)

    case = {
        "case_id": qid,
        "question": question,
        "case_type": case_type,
        "mapped_label": mapped_label,
        "method": GRANULARITY,
        "ground_truth": ground_truth,
        "original_answer": original_answer,
        "original_context": "\n".join(sentences),
        "sentences": sentences,   # explicit D_q (sentences may embed newlines; don't split original_context)
        "removed_item_importance": importance,
        "counterfactual_sentences": meta["counterfactual_sentences"],
        "min_comb_size": meta["min_comb_size"],
        "n_items": len(sentences),
        "n_tests": meta["n_tests"],
        "flipped": meta["flipped"],
        "search": meta["search"],
        "query_calls": ev.query_calls,
        "judge_calls": ev.judge_calls,
    }
    if args.mode == "permutation":
        case.update({"position_sensitive": meta["position_sensitive"],
                     "position_stable": meta["position_stable"],
                     "flip_kendall_tau": meta["flip_kendall_tau"],
                     "flip_perm": meta["flip_perm"]})
    return case


def _summarize(cases, mode):
    summary = {"total_cases": len(cases), "mode": mode, "method": GRANULARITY,
               "ft_cases": sum(1 for c in cases if c["case_type"] == "ft"),
               "ff_cases": sum(1 for c in cases if c["case_type"] == "ff")}
    if mode == "permutation":
        ps = [c for c in cases if c.get("position_sensitive")]
        taus = [c["flip_kendall_tau"] for c in ps if c["flip_kendall_tau"] is not None]
        summary.update({
            "n_position_sensitive": len(ps),
            "pct_position_sensitive": round(100 * len(ps) / len(cases), 2) if cases else 0.0,
            "pct_position_stable": round(100 * (len(cases) - len(ps)) / len(cases), 2) if cases else 0.0,
            "avg_flip_kendall_tau": round(float(np.mean(taus)), 4) if taus else None,
        })
    else:
        summary["n_flipped"] = sum(1 for c in cases if c["flipped"])
        summary["pct_flipped"] = round(100 * summary["n_flipped"] / len(cases), 2) if cases else 0.0
    summary["total_query_calls"] = sum(c["query_calls"] for c in cases)
    summary["total_judge_calls"] = sum(c["judge_calls"] for c in cases)
    return summary


async def run_cases(args, records):
    """records: iterable of (qid, case_type, question, ground_truth)."""
    cases = []
    for qid, case_type, question, ground_truth in tqdm(
            records, desc=f"RAGE({args.mode})", total=len(records)):
        args.rag_counter.reset()
        t0 = time.perf_counter()
        case = await _process_case(args, None, qid, case_type, question, ground_truth)
        if case is None:
            print(f"[{qid}] no retrieved sentences; skipping.")
            continue
        case["elapsed_time"] = round(time.perf_counter() - t0, 4)
        cases.append(case)
        if args.mode == "permutation":
            print(f"[{qid} | {case['mapped_label']}] sents={case['n_items']} "
                  f"tests={case['n_tests']} position_sensitive={case['position_sensitive']} "
                  f"tau={case['flip_kendall_tau']}")
        else:
            print(f"[{qid} | {case['mapped_label']}] sents={case['n_items']} "
                  f"tests={case['n_tests']} flipped={case['flipped']} cf={case['min_comb_size']}")

    summary = _summarize(cases, args.mode)
    out_dir = args.out_dir or "benchmark/results"
    out = args.output or os.path.join(out_dir, f"{args.dataset}_{args.mode}_analysis.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"cases": cases, "summary": summary}, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 64)
    print(f"  RAGE {args.mode}  ({len(cases)} cases, dataset={args.dataset})")
    print(f"  ft={summary['ft_cases']} ff={summary['ff_cases']} | "
          + (f"position_sensitive={summary.get('n_position_sensitive')}" if args.mode == "permutation"
             else f"flipped={summary.get('n_flipped')}"))
    print("=" * 64)
    print(f"Results -> {out}")
    if args.mode == "combination":
        print("Score with: python -m src.correctness.evaluate --method ragex "
              f"--dataset {args.dataset} --facts datasets/{args.dataset}/supporting_facts_{args.dataset}.json "
              f"--results {out}")


def _records_from_comparison(path, num_rows):
    with open(path, encoding="utf-8") as f:
        comparison = json.load(f)
    results = comparison.get("results", comparison)
    recs = [(qid, str(rec["case"]).lower(), rec["question"], rec.get("ground_truth", ""))
            for qid, rec in results.items()
            if isinstance(rec, dict) and str(rec.get("case", "")).lower() in ("ft", "ff")]
    if num_rows is not None:
        recs = recs[:num_rows]
    print(f"Loaded {len(results)} cases from {path}; using {len(recs)} ft/ff flip cases.")
    return recs


def _records_from_csv(dataset, num_rows):
    data = load_qa(QA_CSV_PATHS[dataset])
    if num_rows is not None:
        data = data.head(num_rows)
    return [(str(row.get("id", i)), "ft", row["questions"], row.get("answers", ""))
            for i, row in data.iterrows()]


async def run_benchmark(args):
    args.rag_counter = RagCounter()
    import src.retrieve as _retr
    _retr.vllm_model_complete = args.rag_counter.make_wrapper()
    args.rag = await initialize_lightrag(working_dir=WORKING_DIRS[args.dataset])
    if args.comparison:
        records = _records_from_comparison(args.comparison, args.num_rows)
    else:
        records = _records_from_csv(args.dataset, args.num_rows)
    await run_cases(args, records)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_rage",
        description="RAGE counterfactual RAG explainer over retrieved SENTENCES with "
                    "judge-decided flips. --mode combination = top-down/bottom-up minimal "
                    "source-combination counterfactual (correctness + noise); --mode "
                    "permutation = position-bias diagnostic. Emits the RAG-Ex {cases:[...]} "
                    "schema for src.correctness.evaluate --method ragex.")
    p.add_argument("--dataset", choices=DATASETS, default="hotpotqa")
    p.add_argument("--mode", choices=["combination", "permutation"], default="combination")
    p.add_argument("--rag-mode", choices=["hybrid", "local", "global", "naive"], default="hybrid")
    p.add_argument("--top-k", type=int, default=2,
                   help="LightRAG top_k; retrieved chunks are split into sentences (sources).")
    p.add_argument("--num-rows", type=int, default=None, help="Cap on cases (default: all).")
    p.add_argument("--comparison", default=None,
                   help="FF/FT/TF/TT comparison JSON. Keeps the ft/ff flip cases and emits "
                        "<dataset>_<mode>_analysis.json. Without it, falls back to the QA CSV "
                        "(all rows treated as ft / top-down).")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for <dataset>_<mode>_analysis.json (default benchmark/results).")
    p.add_argument("--max-llm-calls", type=int, default=200,
                   help="Per-case perturbation budget (each = one query + one judge), like generate.py.")
    p.add_argument("--max-size", type=int, default=None,
                   help="[combination] largest subset size searched (default: #sentences).")
    p.add_argument("--perm-cap", type=int, default=6,
                   help="[permutation] n<=cap -> test all n! orders; else single swaps.")
    p.add_argument("--output", default=None, help="Explicit output path (overrides --out-dir).")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    asyncio.run(run_benchmark(args))
