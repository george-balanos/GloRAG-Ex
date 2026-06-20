"""RAG + TMC-Shapley benchmark over TEXT CHUNKS — the text-excerpt analog of
run_shapley.py. Everything is identical to the graph version EXCEPT the Shapley
"players": instead of the retrieved entities/relations, each retrieved DOCUMENT
CHUNK is one player. This mirrors how RAG-Ex
(competitors/RAGEX-RAGE-SHAPLEY/) explains a RAG answer over text rather than the
knowledge graph.

Retrieval is unchanged ("retrieve with LightRAG just like it is, but keep
chunks"): src.retrieve.retrieve_chunks copies KGCasePerturbationEvaluator
.retrieve_context — QueryParam(mode, top_k, only_need_context=True,
enable_rerank=False) -> aquery -> parse_context(...).chunks. A subset of chunks is
rendered back into the `Document Chunks` JSON block (src.parser
.render_context_from_chunks) so each subset is byte-faithful to what RAG feeds the
LLM, and scored under the EXACT RAG system prompt (query.build_rag_system_prompt).

The TMC machinery is reused verbatim from the graph Shapley driver: we subclass
run_shapley.RagContextAttribution (which subclasses the competitor
ContextAttribution) and only swap the two chunk-rendering points; compute_tmc_shap,
_logprob_of_answer, the LLM-call counter, the permutation stats and the per-row
metrics are imported unchanged.

Three modes (CWD must be code/ so relative dataset paths resolve):
  (default) plain     : per QA row, attribute the RAG answer over the retrieved
                        chunks; write <ds>_shapley_text_tmc.json + _metrics.json.
  --permute           : context-permutation robustness over chunk orderings
                        (Kendall-tau + per-chunk spread); <ds>_shapley_text_permutation.json.
  --comparison <json> : correctness-format output. Reads an FF/FT/TF/TT comparison
                        JSON (benchmark/evaluation.py), keeps the ft/ff flip cases,
                        and emits the RAG-Ex {cases:[...]} schema with
                        removed_item_importance = {chunk_text: shapley_value}, so
                        src.correctness.evaluate --method ragex scores it as-is.

  cd code && ../.venv/bin/python ../competitors/Shapley/run_shapley_text.py \
      --dataset synthetic --rag-mode hybrid --top-k 2 --shap-device cuda:1
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_CODE_DIR = os.path.join(_REPO_ROOT, "code")
for _p in (_CODE_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_shapley import (
    RagContextAttribution,
    RagCounter,
    load_qa,
    load_hf_utility_model,
    tmc_budget,
    permutation_stats,
)

from src.retrieve import initialize_lightrag
from src.query import query
from src.llm_judge import judge_response
from src.llm.utils import vllm_model_complete
from src.dataset_setup import WORKING_DIRS, QA_CSV_PATHS, DATASETS

from chunk_utils import retrieve_chunks, render_context_from_chunks

from tqdm import tqdm
import argparse
import asyncio
import json
import logging
import random
import time

import numpy as np

logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("nano-vectordb").setLevel(logging.WARNING)
logging.getLogger("lightrag").setLevel(logging.WARNING)

def split_into_players(chunks: list[str], granularity: str = "chunk") -> list[str]:
    if granularity == "chunk":
        return list(chunks)
    if granularity == "sentence":
        full_text = "\n\n".join(chunks)
        return [s.strip() + "." for s in full_text.split(".") if s.strip()]
    raise ValueError(f"Unknown granularity: {granularity!r}")


_ID_PREFIX = {"chunk": "C", "sentence": "S"}


def unit_id(i: int, granularity: str = "chunk") -> str:
    return f"{_ID_PREFIX.get(granularity, 'C')}::{i}"

def chunk_id(i: int) -> str:
    return unit_id(i, "chunk")

def player_id_map(players: list[str], granularity: str = "chunk") -> dict[str, str]:
    out: dict[str, str] = {}
    for i, text in enumerate(players):
        out.setdefault(text, unit_id(i, granularity))
    return out

class RagChunkContextAttribution(RagContextAttribution):
    """RagContextAttribution with the two graph-rendering points swapped for the
    `Document Chunks` JSON block. `objects` is the ordered list of chunk strings;
    everything else (TMC loop, _logprob_of_answer, prompt) is inherited."""

    def _get_ablated_context_from_vector(self, v_np) -> str:
        selected = [self.objects[i] for i, keep in enumerate(v_np) if keep == 1]
        return render_context_from_chunks(selected)

    def _compute_response_metric(self, context_str: str, mode: str, response: str = None) -> float:
        if response is None:
            response = self.target_response
        answer_ids = self.tokenizer(
            response, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        num_answer_tokens = answer_ids.shape[1]
        if num_answer_tokens == 0:
            return 0.0

        log_prob_with = self._logprob_of_answer(context_str, answer_ids)
        log_prob_empty = self._logprob_of_answer(render_context_from_chunks([]), answer_ids)

        if mode in ("log-perplexity", None):
            metric = (log_prob_with - log_prob_empty) / num_answer_tokens
        elif mode == "log-prob":
            metric = log_prob_with - log_prob_empty
        else:
            raise ValueError(f"Unsupported utility mode for RagChunkContextAttribution: {mode!r}")
        return metric.item()


def run_tmc_chunks(players, question, model, tokenizer, device, rag_answer, args):
    attributor = RagChunkContextAttribution(
        objects=players, query_text=question, model=model, tokenizer=tokenizer,
        device=device, target_response=rag_answer, verbose=args.verbose,
    )
    shared_cache: dict = {}  
    scores = attributor.compute_tmc_shap(
        num_iterations_max=args.tmc_iters,
        performance_tolerance=args.tmc_tol,
        max_unique_lookups=tmc_budget(len(players), args.tmc_budget),
        seed=args.seed,
        shared_cache=shared_cache,
        utility_mode="log-perplexity",
    )
    return [float(s) for s in scores], len(shared_cache)


def random_unit_permutations(units: list[str], count: int = 5, seed: int | None = None):
    if not units:
        return []
    rng = random.Random(seed)
    indices = list(range(len(units)))
    out = []
    for i in range(count):
        perm = tuple(rng.sample(indices, len(indices)))
        permuted = [units[j] for j in perm]
        out.append({
            "perm_id": f"rand_{i + 1:02d}",
            "perm": perm,
            "units": permuted,
            "render": render_context_from_chunks(permuted),
            "identity": perm == tuple(indices),
        })
    return out


async def run_plain(args, rag, rag_counter, hf_model, hf_tok, data):
    results, metrics = {}, {}
    for _, row in tqdm(data.iterrows(), desc="RAG+TMC-Shapley(text)", total=len(data)):
        rid = str(row["id"])
        question, ground_truth = row["questions"], row["answers"]
        row_t0 = time.perf_counter()

        rag_counter.reset()
        context, chunks = await retrieve_chunks(rag, query=question, mode=args.rag_mode, top_k=args.top_k)
        players = split_into_players(chunks, args.granularity)
        rag_answer = await query(rag, context, question)
        rag_calls, rag_time = rag_counter.calls, rag_counter.time

        judge_calls, judge_time, score = 0, 0.0, None
        if args.judge:
            jt0 = time.perf_counter()
            score = await judge_response(question, rag_answer, ground_truth)
            judge_time = time.perf_counter() - jt0
            judge_calls = 1

        n_items = len(players)
        shap_evals, shap_time, scores_by_id, player_texts = 0, 0.0, {}, {}
        if n_items == 0:
            print(f"[{rid}] no retrieved {args.granularity}s; skipping Shapley.")
        else:
            st0 = time.perf_counter()
            scores, shap_evals = run_tmc_chunks(players, question, hf_model, hf_tok, args.shap_device, rag_answer, args)
            shap_time = time.perf_counter() - st0
            scores_by_id = {unit_id(i, args.granularity): s for i, s in enumerate(scores)}
            player_texts = {unit_id(i, args.granularity): t for i, t in enumerate(players)}

        total_time = time.perf_counter() - row_t0
        total_calls = rag_calls + judge_calls + shap_evals

        results[rid] = {
            "question": question, "ground_truth": ground_truth, "rag_answer": rag_answer,
            "score": score, "granularity": args.granularity, "n_items": n_items,
            "shapley_scores": scores_by_id, "player_texts": player_texts,
        }
        metrics[rid] = {
            "rag_calls": rag_calls, "rag_time": round(rag_time, 4),
            "shap_utility_evals": shap_evals, "shap_forward_passes": shap_evals * 2,
            "shap_time": round(shap_time, 4),
            "judge_calls": judge_calls, "judge_time": round(judge_time, 4),
            "total_calls": total_calls, "total_time": round(total_time, 4), "n_items": n_items,
        }
        print(f"[{rid}] {args.granularity}s={n_items} score={score} | RAG: {rag_calls} calls / {rag_time:.2f}s | "
              f"Shapley: {shap_evals} evals / {shap_time:.2f}s | total: {total_calls} calls / {total_time:.2f}s")

    _write_plain_outputs(args, results, metrics)


def _write_plain_outputs(args, results, metrics):
    n = len(metrics) or 1

    def s(key):
        return sum(m[key] for m in metrics.values())

    aggregate = {
        "rows": len(metrics),
        "totals": {
            "rag_calls": s("rag_calls"), "rag_time": round(s("rag_time"), 4),
            "shap_utility_evals": s("shap_utility_evals"), "shap_forward_passes": s("shap_forward_passes"),
            "shap_time": round(s("shap_time"), 4),
            "judge_calls": s("judge_calls"), "judge_time": round(s("judge_time"), 4),
            "total_llm_calls": s("total_calls"), "total_wall_time": round(s("total_time"), 4),
        },
        "per_row_avg": {
            "rag_calls": round(s("rag_calls") / n, 4), "rag_time": round(s("rag_time") / n, 4),
            "shap_utility_evals": round(s("shap_utility_evals") / n, 4), "shap_time": round(s("shap_time") / n, 4),
            "judge_calls": round(s("judge_calls") / n, 4), "judge_time": round(s("judge_time") / n, 4),
            "total_calls": round(s("total_calls") / n, 4), "total_time": round(s("total_time") / n, 4),
            "n_items": round(s("n_items") / n, 4),
        },
    }
    metrics["__aggregate__"] = aggregate
    out_results = args.output or f"benchmark/results/{args.dataset}_shapley_text_{args.granularity}_tmc.json"
    out_metrics = args.metrics or f"benchmark/results/{args.dataset}_shapley_text_{args.granularity}_tmc_metrics.json"
    os.makedirs(os.path.dirname(out_results), exist_ok=True)
    with open(out_results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    a = aggregate
    print("\n" + "=" * 64)
    print(f"  RAG + TMC-Shapley (text) summary  ({a['rows']} rows, dataset={args.dataset})")
    print("=" * 64)
    print(f"{'bucket':<12}{'LLM calls (total)':>20}{'avg/row':>12}{'time/row(s)':>14}")
    print(f"{'RAG':<12}{a['totals']['rag_calls']:>20}{a['per_row_avg']['rag_calls']:>12}{a['per_row_avg']['rag_time']:>14}")
    print(f"{'Shapley':<12}{a['totals']['shap_utility_evals']:>20}{a['per_row_avg']['shap_utility_evals']:>12}{a['per_row_avg']['shap_time']:>14}")
    if args.judge:
        print(f"{'Judge':<12}{a['totals']['judge_calls']:>20}{a['per_row_avg']['judge_calls']:>12}{a['per_row_avg']['judge_time']:>14}")
    print(f"{'WHOLE':<12}{a['totals']['total_llm_calls']:>20}{a['per_row_avg']['total_calls']:>12}{a['per_row_avg']['total_time']:>14}")
    print("=" * 64)
    print(f"Total wall time: {a['totals']['total_wall_time']}s  (Shapley fwd passes: {a['totals']['shap_forward_passes']})")
    print(f"Results -> {out_results}\nMetrics -> {out_metrics}")


# ── Permutation mode ────────────────────────────────────────────────────────
async def run_permutation(args, rag, rag_counter, hf_model, hf_tok, data):
    results = {}
    tau_list, mintau_list, top1_list, topk_list, exact_list = [], [], [], [], []
    posmatch_list, poschecked_list = [], []
    for _, row in tqdm(data.iterrows(), desc="Shapley(text) permutation", total=len(data)):
        rid = str(row["id"])
        question, ground_truth = row["questions"], row["answers"]

        context, chunks = await retrieve_chunks(rag, query=question, mode=args.rag_mode, top_k=args.top_k)
        players = split_into_players(chunks, args.granularity)
        rag_answer = await query(rag, context, question)  # fixed target across permutations

        if len(players) == 0:
            print(f"[{rid}] no retrieved {args.granularity}s; skipping.")
            continue

        id_of = player_id_map(players, args.granularity)
        ids = [unit_id(i, args.granularity) for i in range(len(players))]
        perms = random_unit_permutations(players, count=5, seed=args.seed)

        scores_by_perm = {}
        perm_records = []
        for p in perms:
            order_units = p["units"]
            order_ids = [id_of[t] for t in order_units]
            st0 = time.perf_counter()
            scores, evals = run_tmc_chunks(order_units, question, hf_model, hf_tok, args.shap_device, rag_answer, args)
            elapsed = round(time.perf_counter() - st0, 4)
            sbi = {id_of[t]: s for t, s in zip(order_units, scores)}
            scores_by_perm[p["perm_id"]] = sbi
            ranking = sorted(sbi, key=lambda o: sbi[o], reverse=True)
            perm_records.append({
                "perm_id": p["perm_id"],
                "perm": list(p["perm"]),
                "object_order": order_ids,
                "shapley_scores": sbi,
                "ranking": ranking,
                "utility_evals": evals,
                "forward_passes": evals * 2,
                "shap_time": elapsed,
            })

        stats = permutation_stats(scores_by_perm, ids, args.topk_stable)
        topk_key = f"top{args.topk_stable}_stable"
        results[rid] = {
            "question": question, "ground_truth": ground_truth, "rag_answer": rag_answer,
            "granularity": args.granularity, "n_players": len(players),
            "object_ids": ids,
            "player_texts": {unit_id(i, args.granularity): t for i, t in enumerate(players)},
            "num_permutations": len(perm_records),
            "permutations": perm_records,
            "stats": stats,
            "perm_total_utility_evals": sum(r["utility_evals"] for r in perm_records),
            "perm_total_shap_time": round(sum(r["shap_time"] for r in perm_records), 4),
        }
        tau_list.append(stats["mean_kendall_tau"])
        mintau_list.append(stats["min_kendall_tau"])
        top1_list.append(stats["top1_stable"])
        topk_list.append(stats.get(topk_key, False))
        exact_list.append(stats["exact_ranking_match"])
        posmatch_list.append(stats["topk_position_matches"])
        poschecked_list.append(stats["topk_positions_checked"])
        print(f"[{rid}] perms={stats['num_permutations']} meanτ={stats['mean_kendall_tau']:.3f} "
              f"minτ={stats['min_kendall_tau']:.3f} top1_stable={stats['top1_stable']} "
              f"top{args.topk_stable}_pos_same={stats['topk_position_matches']}/{stats['topk_positions_checked']} "
              f"exact={stats['exact_ranking_match']}")

    out = args.output or f"benchmark/results/{args.dataset}_shapley_text_{args.granularity}_permutation.json"
    rows = len(results)
    summary = {
        "rows": rows,
        "topk_stable_k": args.topk_stable,
        "avg_mean_kendall_tau": float(np.nanmean(tau_list)) if tau_list else float("nan"),
        "avg_min_kendall_tau": float(np.nanmean(mintau_list)) if mintau_list else float("nan"),
        "pct_top1_stable": round(100 * sum(top1_list) / rows, 2) if rows else 0.0,
        "pct_topk_stable": round(100 * sum(topk_list) / rows, 2) if rows else 0.0,
        "pct_exact_ranking_match": round(100 * sum(exact_list) / rows, 2) if rows else 0.0,
        "avg_topk_position_matches": round(sum(posmatch_list) / rows, 4) if rows else 0.0,
        "avg_topk_positions_checked": round(sum(poschecked_list) / rows, 4) if rows else 0.0,
    }
    results["__summary__"] = summary
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 64)
    print(f"  Shapley(text) permutation robustness  ({rows} rows, dataset={args.dataset})")
    print("=" * 64)
    print(f"avg mean Kendall-tau : {summary['avg_mean_kendall_tau']:.4f}")
    print(f"avg min  Kendall-tau : {summary['avg_min_kendall_tau']:.4f}")
    print(f"top-1 stable rows    : {summary['pct_top1_stable']}%")
    print(f"top-{args.topk_stable} stable rows    : {summary['pct_topk_stable']}% (same set)")
    print(f"top-{args.topk_stable} same-position  : {summary['avg_topk_position_matches']}/{summary['avg_topk_positions_checked']} ranks (avg)")
    print(f"exact-ranking rows   : {summary['pct_exact_ranking_match']}%")
    print("=" * 64)
    print(f"Results -> {out}")


async def run_comparison(args, rag, rag_counter, hf_model, hf_tok):
    with open(args.comparison, encoding="utf-8") as f:
        comparison = json.load(f)
    results = comparison.get("results", comparison)  # accept {results:{...}} or a bare id->rec map
    relevant = [(qid, rec) for qid, rec in results.items()
                if isinstance(rec, dict) and str(rec.get("case", "")).lower() in ("ft", "ff")]
    if args.num_rows is not None:
        relevant = relevant[:args.num_rows]
    print(f"Loaded {len(results)} cases from {args.comparison}; using {len(relevant)} ft/ff flip cases.")

    cases, n_changed_total = [], 0
    ft_cases = ff_cases = 0
    for qid, rec in tqdm(relevant, desc="Shapley(text) correctness", total=len(relevant)):
        case_type = str(rec["case"]).lower()
        question = rec["question"]
        ground_truth = rec.get("ground_truth", "")
        mapped_label = "F->T" if case_type == "ff" else "T->F"

        q_t0 = time.perf_counter()
        rag_counter.reset()
        context, chunks = await retrieve_chunks(rag, query=question, mode=args.rag_mode, top_k=args.top_k)
        players = split_into_players(chunks, args.granularity)
        if len(players) == 0:
            print(f"[{qid}] no retrieved {args.granularity}s; skipping.")
            continue
        rag_answer = await query(rag, context, question)
        rag_calls, rag_time = rag_counter.calls, rag_counter.time

        st0 = time.perf_counter()
        scores, shap_evals = run_tmc_chunks(players, question, hf_model, hf_tok, args.shap_device, rag_answer, args)
        shap_time = time.perf_counter() - st0

        scores_by_id = {unit_id(i, args.granularity): s for i, s in enumerate(scores)}
        player_texts = {unit_id(i, args.granularity): t for i, t in enumerate(players)}
        removed_item_importance = {t: float(s) for t, s in zip(players, scores)}

        if case_type == "ff":
            ff_cases += 1
        else:
            ft_cases += 1

        cases.append({
            "case_id": qid,
            "question": question,
            "case_type": case_type,
            "mapped_label": mapped_label,
            "method": args.granularity,
            "ground_truth": ground_truth,
            "original_answer": rag_answer,
            "original_context": "\n\n".join(chunks),
            "removed_item_importance": removed_item_importance,
            "shapley_scores": scores_by_id,
            "player_texts": player_texts,
            "n_items": len(players),
            "shap_utility_evals": shap_evals,
            "shap_forward_passes": shap_evals * 2,
            "rag_calls": rag_calls,
            "rag_time": round(rag_time, 4),
            "shap_time": round(shap_time, 4),
            "elapsed_time": round(time.perf_counter() - q_t0, 4),
        })
        print(f"[{qid} | {mapped_label}] {args.granularity}s={len(players)} | Shapley: {shap_evals} evals / {shap_time:.2f}s")

    summary = {
        "total_cases": len(cases),
        "ft_cases": ft_cases,
        "ff_cases": ff_cases,
        "method": args.granularity,
        "total_shap_utility_evals": sum(c["shap_utility_evals"] for c in cases),
        "total_shap_time": round(sum(c["shap_time"] for c in cases), 4),
    }
    out_dir = args.out_dir or "benchmark/results"
    out = args.output or os.path.join(out_dir, f"{args.dataset}_{args.granularity}_analysis.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"cases": cases, "summary": summary}, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 64)
    print(f"  Shapley(text) correctness cases  ({len(cases)} cases, dataset={args.dataset})")
    print(f"  ft={ft_cases}  ff={ff_cases}  method={args.granularity}")
    print("=" * 64)
    print(f"Results -> {out}")
    print("Score with: python -m src.correctness.evaluate --method ragex "
          f"--dataset {args.dataset} --facts datasets/{args.dataset}/supporting_facts_{args.dataset}.json "
          f"--results {out}")


async def run_benchmark(args):
    rag_counter = RagCounter()
    import src.retrieve as _retr
    _retr.vllm_model_complete = rag_counter.make_wrapper()
    rag = await initialize_lightrag(working_dir=WORKING_DIRS[args.dataset])
    hf_model, hf_tok = load_hf_utility_model(args.shap_device, args.shap_load_8bit, args.shap_load_4bit)

    if args.comparison:
        await run_comparison(args, rag, rag_counter, hf_model, hf_tok)
        return

    data = load_qa(QA_CSV_PATHS[args.dataset])
    if args.num_rows is not None:
        data = data.head(args.num_rows)

    if args.permute:
        await run_permutation(args, rag, rag_counter, hf_model, hf_tok, data)
    else:
        await run_plain(args, rag, rag_counter, hf_model, hf_tok, data)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_shapley_text",
        description="RAG + TMC-Shapley benchmark over TEXT CHUNKS (text-excerpt analog "
                    "of run_shapley.py): plain metrics, --permute robustness, or "
                    "--comparison correctness-format output.")
    p.add_argument("--dataset", choices=DATASETS, default="synthetic")
    p.add_argument("--granularity", choices=["chunk", "sentence"], default="chunk",
                   help="Shapley player granularity: whole retrieved chunks, or sentences "
                        "split from the joined chunks (RAG-Ex remove_sentence analog).")
    p.add_argument("--rag-mode", choices=["hybrid", "local", "global", "naive"], default="hybrid")
    p.add_argument("--top-k", type=int, default=2,
                   help="LightRAG top_k; the number of retrieved chunks (players) scales with it.")
    p.add_argument("--num-rows", type=int, default=None, help="Cap on QA rows / cases (default: all).")
    p.add_argument("--comparison", default=None,
                   help="FF/FT/TF/TT comparison JSON (benchmark/evaluation.py). When set, keep the ft/ff "
                        "flip cases and emit the RAG-Ex {cases:[...]} correctness schema "
                        "(removed_item_importance = {chunk_text: shapley_value}). No QA-CSV / permute.")
    p.add_argument("--out-dir", default=None,
                   help="[--comparison] output directory for <dataset>_chunk_analysis.json (default benchmark/results).")
    p.add_argument("--shap-device", default="cuda:1", help="Device for the HF utility model.")
    p.add_argument("--shap-load-8bit", action="store_true")
    p.add_argument("--shap-load-4bit", action="store_true")
    p.add_argument("--tmc-iters", type=int, default=50, help="num_iterations_max (permutations).")
    p.add_argument("--tmc-tol", type=float, default=0.01, help="performance_tolerance for truncation.")
    p.add_argument("--tmc-budget", type=int, default=None, help="max_unique_lookups (default: min(2^n, 128)).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--permute", action="store_true",
                   help="Run context-permutation robustness (5 random chunk orderings) instead of plain metrics.")
    p.add_argument("--topk-stable", dest="topk_stable", type=int, default=2,
                   help="k for the top-k stability check in --permute mode.")
    p.add_argument("--judge", dest="judge", action="store_true", default=True,
                   help="Run the vLLM judge to score answers (default on; ignored in --permute / --comparison).")
    p.add_argument("--no-judge", dest="judge", action="store_false")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument("--metrics", default=None)
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    asyncio.run(run_benchmark(args))
