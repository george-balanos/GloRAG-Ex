"""RAG + TMC-Shapley benchmark with per-row LLM-call & timing instrumentation,
plus an optional context-permutation robustness mode (--permute).

For each QA row (synthetic by default) this:
  1. Retrieves the subgraph (entities + relations) via LightRAG  -> RAG bucket
  2. Generates an answer with the vLLM Mistral-24B                -> RAG bucket
  3. (optional) Judges the answer with the vLLM Qwen-32B          -> judge bucket
  4. Runs Truncated Monte Carlo Shapley over the retrieved OBJECTS,
     attributing the RAG-generated answer, using a HuggingFace copy of
     Mistral-24B on a separate GPU                                -> Shapley bucket

TMC implementation: the COMPETITOR class
`competitors/LLMX/SHapRAG/ragshap_experiments.py :: ContextAttribution`.
We subclass it (RagContextAttribution) to (a) pin the model to a chosen device,
(b) inject the RAG answer as the attribution target, (c) serialize every subset
in the EXACT RAG context format via parser.render_context, and (d) score each
subset under the EXACT RAG system prompt (query.build_rag_system_prompt). The
TMC algorithm (compute_tmc_shap) is inherited unmodified.

--permute: context-permutation robustness over the base retrieved objects.
The permutation mode now treats entities + relations as one ordered object bag
and samples 5 random object permutations. For each permutation we run TMC
(same fixed RAG answer as target) and then report Kendall-tau ranking
stability + per-object value spread.

Run from code/ (sys.path is bootstrapped below), e.g.:
  ../.venv/bin/python benchmark/run_shapley.py --dataset synthetic \
      --rag-mode hybrid --top-k 2 --shap-device cuda:1
  ../.venv/bin/python benchmark/run_shapley.py --permute --shap-device cuda:1 --no-judge

Requires (to import the competitor module) the SHapRAG deps it imports at module
top: shapiq, spectralexplain, tensorflow, fastFM.
"""
import os
import sys

# Make `src` importable when run directly (stock benchmark relies on
# PYTHONPATH=code, set by run_ablation.sh). code/ is the parent of benchmark/.
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src.retrieve import initialize_lightrag, retrieve_subgraph_objects
from src.query import query, build_rag_system_prompt
from src.llm_judge import judge_response
from src.llm.utils import vllm_model_complete, VLLM_MODEL
from src.dataset_setup import WORKING_DIRS, QA_CSV_PATHS, DATASETS
from src.parser import render_context_from_objects
from src.perm_utils import random_object_permutations

from tqdm import tqdm
import argparse
import asyncio
import itertools
import json
import logging
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("nano-vectordb").setLevel(logging.WARNING)
logging.getLogger("lightrag").setLevel(logging.WARNING)

# Make the competitor package importable: competitors/LLMX holds the `SHapRAG` pkg.
_REPO_ROOT = os.path.dirname(_CODE_DIR)
_LLMX_DIR = os.path.join(_REPO_ROOT, "competitors", "LLMX")
if _LLMX_DIR not in sys.path:
    sys.path.insert(0, _LLMX_DIR)
from SHapRAG.ragshap_experiments import ContextAttribution as CompetitorContextAttribution


# ── Object helpers ──────────────────────────────────────────────────────────
# A Shapley "object" is one retrieved entity or relation. We keep it tagged so a
# subset can be re-rendered into the exact two-section RAG context.
def build_objects(entities, relations):
    """Ordered list of (kind, obj): entities first, then relations."""
    return [("entity", e) for e in entities] + [("relation", r) for r in relations]


def object_id(kind, obj) -> str:
    """Stable id so a given object can be tracked across permutations."""
    if kind == "entity":
        return f"E::{obj.name}"
    return f"R::{obj.src}->{obj.tgt}"


def split_objects(objects):
    """Split a tagged object list back into (entities, relations) in order."""
    ents = [obj for kind, obj in objects if kind == "entity"]
    rels = [obj for kind, obj in objects if kind == "relation"]
    return ents, rels


# ── Adapter around the competitor's ContextAttribution ──────────────────────
class _MainProcAccelerator:
    """Minimal stand-in for accelerate.Accelerator (single process, no device grab)."""
    is_main_process = True

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self):
        pass


class RagContextAttribution(CompetitorContextAttribution):
    """Competitor ContextAttribution, RAG-faithful.

    Inherits compute_tmc_shap unchanged. Overrides:
      - __init__: pin device, inject RAG answer as target, no self-generation.
      - _get_ablated_context_from_vector: render the selected objects in the
        EXACT RAG two-section context format (parser.render_context).
      - _compute_response_metric: score logP(answer | subset) under the EXACT
        RAG system prompt (query.build_rag_system_prompt), tokenizing the chat
        template the same way vllm_model_complete does for Mistral (tokenize=True).
    """

    def __init__(self, objects, query_text, model, tokenizer, device, target_response, verbose=False):
        if not objects:
            raise ValueError("objects list cannot be empty")
        self.accelerator = _MainProcAccelerator()
        self.objects = objects
        self.items = objects  # competitor code reads self.n_items; keep alias
        self.query = query_text
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.n_items = len(objects)
        self.device = torch.device(device)
        self.utility_cache = defaultdict(dict)
        self._factorials = {k: math.factorial(k) for k in range(self.n_items + 1)}
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.target_response = target_response

    def _get_ablated_context_from_vector(self, v_np) -> str:
        selected = [self.objects[i] for i, keep in enumerate(v_np) if keep == 1]
        return render_context_from_objects(selected)

    def _logprob_of_answer(self, context_str: str, answer_ids: torch.Tensor) -> torch.Tensor:
        # Identical prompt construction to RAG: system = build_rag_system_prompt(ctx),
        # user = the question. Mistral path tokenizes the chat template directly
        # (tokenize=True), matching vllm_model_complete.
        system_prompt = build_rag_system_prompt(context_str)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.query},
        ]
        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt")
        # Mistral's tokenizer returns a BatchEncoding (dict-like); others return a
        # bare tensor. Normalize to the input_ids tensor.
        prompt_ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
        prompt_ids = prompt_ids.to(self.device)
        full_input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            logits = self.model(input_ids=full_input_ids).logits

        shift_logits = logits[..., prompt_len - 1:-1, :].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        answer_log_probs = torch.gather(log_probs, 2, answer_ids.unsqueeze(-1)).squeeze(-1)
        total = answer_log_probs.sum()
        del logits, shift_logits, log_probs, answer_log_probs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return total

    def _compute_response_metric(self, context_str: str, mode: str, response: str = None) -> float:
        if response is None:
            response = self.target_response
        answer_ids = self.tokenizer(
            response, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        num_answer_tokens = answer_ids.shape[1]
        if num_answer_tokens == 0:
            return 0.0

        log_prob_with = self._logprob_of_answer(context_str, answer_ids)
        # Empty-context baseline (renders empty sections). It is a constant
        # subtracted from every subset utility, so it cancels in TMC marginals;
        # kept for parity with the competitor's utility definition.
        log_prob_empty = self._logprob_of_answer(render_context_from_objects([]), answer_ids)

        if mode in ("log-perplexity", None):
            metric = (log_prob_with - log_prob_empty) / num_answer_tokens
        elif mode == "log-prob":
            metric = log_prob_with - log_prob_empty
        else:
            raise ValueError(f"Unsupported utility mode for RagContextAttribution: {mode!r}")
        return metric.item()


# ── RAG LLM-call counter ────────────────────────────────────────────────────
class RagCounter:
    def __init__(self):
        self.calls = 0
        self.time = 0.0

    def reset(self):
        self.calls = 0
        self.time = 0.0

    def make_wrapper(self):
        async def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            out = await vllm_model_complete(*args, **kwargs)
            self.time += time.perf_counter() - t0
            self.calls += 1
            return out
        return wrapped


def load_qa(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["questions"])
    return df.reset_index(drop=True)


def load_hf_utility_model(device: str, load_8bit: bool, load_4bit: bool):
    """Load Mistral-24B via HF transformers for Shapley log-prob utilities."""
    print(f"Loading HF utility model {VLLM_MODEL} on {device} "
          f"(8bit={load_8bit}, 4bit={load_4bit}) ...")
    tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL)
    kwargs = dict(torch_dtype=torch.bfloat16)
    if load_8bit or load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=load_8bit, load_in_4bit=load_4bit)
        kwargs["device_map"] = device
    else:
        kwargs["device_map"] = {"": device}
    # Mistral-Small-3.2-24B is a vision-language model (Mistral3Config), which
    # AutoModelForCausalLM rejects; fall back to the image-text-to-text class
    # (it scores text logits fine when called with input_ids only).
    try:
        model = AutoModelForCausalLM.from_pretrained(VLLM_MODEL, **kwargs)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        print("  (model is image-text-to-text; loading via AutoModelForImageTextToText)")
        model = AutoModelForImageTextToText.from_pretrained(VLLM_MODEL, **kwargs)
    model.eval()
    return model, tokenizer


def tmc_budget(n_items, arg_budget):
    return arg_budget if arg_budget is not None else min(2 ** n_items, 128)


def run_tmc(objects, question, model, tokenizer, device, rag_answer, args):
    """Run competitor TMC over `objects`. Returns (scores aligned to objects, evals)."""
    attributor = RagContextAttribution(
        objects=objects, query_text=question, model=model, tokenizer=tokenizer,
        device=device, target_response=rag_answer, verbose=args.verbose,
    )
    shared_cache: dict = {}  # fresh per call; len() == unique utility evals
    scores = attributor.compute_tmc_shap(
        num_iterations_max=args.tmc_iters,
        performance_tolerance=args.tmc_tol,
        max_unique_lookups=tmc_budget(len(objects), args.tmc_budget),
        seed=args.seed,
        shared_cache=shared_cache,
        utility_mode="log-perplexity",
    )
    return [float(s) for s in scores], len(shared_cache)


# ── Permutation-robustness stats ────────────────────────────────────────────
def permutation_stats(scores_by_perm: dict[str, dict[str, float]], ids: list[str], top_k: int):
    """scores_by_perm: perm_id -> {object_id -> shapley value}. Returns stats dict."""
    perm_ids = list(scores_by_perm.keys())

    # per-object spread across permutations
    per_object = {}
    for oid in ids:
        vals = np.array([scores_by_perm[p][oid] for p in perm_ids], dtype=float)
        per_object[oid] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "range": float(vals.max() - vals.min()),
        }

    # ranking per permutation (object ids ordered by descending value)
    rankings = {p: [oid for oid in sorted(ids, key=lambda o: scores_by_perm[p][o], reverse=True)]
                for p in perm_ids}
    rank_index = {p: {oid: i for i, oid in enumerate(rankings[p])} for p in perm_ids}

    # pairwise Kendall tau over the rank positions
    taus = []
    exact_match = True
    if len(perm_ids) >= 2 and len(ids) >= 2:
        for a, b in itertools.combinations(perm_ids, 2):
            ra = [rank_index[a][o] for o in ids]
            rb = [rank_index[b][o] for o in ids]
            tau, _ = kendalltau(ra, rb)
            taus.append(float(tau) if tau == tau else float("nan"))  # nan-safe
            if rankings[a] != rankings[b]:
                exact_match = False
    mean_tau = float(np.nanmean(taus)) if taus else float("nan")
    min_tau = float(np.nanmin(taus)) if taus else float("nan")

    # top-1 / top-k SET stability across all permutations (same members, any order)
    top1 = {rankings[p][0] for p in perm_ids} if ids else set()
    topk_sets = [frozenset(rankings[p][:top_k]) for p in perm_ids] if ids else []
    top1_stable = len(top1) == 1
    topk_stable = len(set(topk_sets)) == 1 if topk_sets else True

    # top-k POSITIONAL stability: for each of the top-k ranks, is the SAME object
    # in that exact spot across every permutation? (generalizes top1_stable)
    k_eff = min(top_k, len(ids))
    position_stable = [
        len({rankings[p][i] for p in perm_ids}) == 1
        for i in range(k_eff)
    ]
    topk_position_matches = int(sum(position_stable))

    return {
        "num_permutations": len(perm_ids),
        "mean_kendall_tau": mean_tau,
        "min_kendall_tau": min_tau,
        "exact_ranking_match": exact_match,
        "top1_stable": top1_stable,
        f"top{top_k}_stable": topk_stable,
        "topk_positions_checked": k_eff,
        "topk_position_matches": topk_position_matches,
        "topk_position_stable": position_stable,
        "per_object": per_object,
        "rankings": rankings,
    }


# ── Plain mode (Phase 1 metrics) ────────────────────────────────────────────
async def run_plain(args, rag, rag_counter, hf_model, hf_tok, data):
    results, metrics = {}, {}
    for _, row in tqdm(data.iterrows(), desc="RAG+TMC-Shapley", total=len(data)):
        rid = str(row["id"])
        question, ground_truth = row["questions"], row["answers"]
        row_t0 = time.perf_counter()

        rag_counter.reset()
        context, sg = await retrieve_subgraph_objects(rag, query=question, mode=args.rag_mode, top_k=args.top_k)
        rag_answer = await query(rag, context, question)
        rag_calls, rag_time = rag_counter.calls, rag_counter.time

        judge_calls, judge_time, score = 0, 0.0, None
        if args.judge:
            jt0 = time.perf_counter()
            score = await judge_response(question, rag_answer, ground_truth)
            judge_time = time.perf_counter() - jt0
            judge_calls = 1

        objects = build_objects(sg.entities, sg.relations)
        n_items = len(objects)
        shap_evals, shap_time, scores_by_id = 0, 0.0, {}
        if n_items == 0:
            print(f"[{rid}] no retrieved objects; skipping Shapley.")
        else:
            st0 = time.perf_counter()
            scores, shap_evals = run_tmc(objects, question, hf_model, hf_tok, args.shap_device, rag_answer, args)
            shap_time = time.perf_counter() - st0
            scores_by_id = {object_id(k, o): s for (k, o), s in zip(objects, scores)}

        total_time = time.perf_counter() - row_t0
        total_calls = rag_calls + judge_calls + shap_evals

        results[rid] = {
            "question": question, "ground_truth": ground_truth, "rag_answer": rag_answer,
            "score": score, "n_items": n_items, "shapley_scores": scores_by_id,
        }
        metrics[rid] = {
            "rag_calls": rag_calls, "rag_time": round(rag_time, 4),
            "shap_utility_evals": shap_evals, "shap_forward_passes": shap_evals * 2,
            "shap_time": round(shap_time, 4),
            "judge_calls": judge_calls, "judge_time": round(judge_time, 4),
            "total_calls": total_calls, "total_time": round(total_time, 4), "n_items": n_items,
        }
        print(f"[{rid}] items={n_items} score={score} | RAG: {rag_calls} calls / {rag_time:.2f}s | "
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
    out_results = args.output or f"benchmark/results/{args.dataset}_shapley_tmc.json"
    out_metrics = args.metrics or f"benchmark/results/{args.dataset}_shapley_tmc_metrics.json"
    os.makedirs(os.path.dirname(out_results), exist_ok=True)
    with open(out_results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    a = aggregate
    print("\n" + "=" * 64)
    print(f"  RAG + TMC-Shapley summary  ({a['rows']} rows, dataset={args.dataset})")
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
    for _, row in tqdm(data.iterrows(), desc="Shapley permutation", total=len(data)):
        rid = str(row["id"])
        question, ground_truth = row["questions"], row["answers"]

        context, sg = await retrieve_subgraph_objects(rag, query=question, mode=args.rag_mode, top_k=args.top_k)
        rag_answer = await query(rag, context, question)  # fixed target across permutations

        if len(sg.entities) + len(sg.relations) == 0:
            print(f"[{rid}] no retrieved objects; skipping.")
            continue

        ids = [object_id(k, o) for k, o in build_objects(sg.entities, sg.relations)]
        perms = random_object_permutations(sg.entities, sg.relations, count=5, seed=args.seed)

        # One self-contained record per permutation: presented order, the Shapley
        # result, the induced ranking, and that run's metrics.
        scores_by_perm = {}
        perm_records = []
        for p in perms:
            objects = p["objects"]
            order_ids = [object_id(k, o) for (k, o) in objects]
            st0 = time.perf_counter()
            scores, evals = run_tmc(objects, question, hf_model, hf_tok, args.shap_device, rag_answer, args)
            elapsed = round(time.perf_counter() - st0, 4)
            sbi = {object_id(k, o): s for (k, o), s in zip(objects, scores)}
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
            "n_entities": len(sg.entities), "n_relations": len(sg.relations),
            "object_ids": ids,
            "num_permutations": len(perm_records),
            "permutations": perm_records,            # all 5 results + their metrics
            "stats": stats,                          # kendall-tau, spread, rankings, ...
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

    out = args.output or f"benchmark/results/{args.dataset}_shapley_permutation.json"
    rows = len(results)
    summary = {
        "rows": rows,
        "topk_stable_k": args.topk_stable,
        "avg_mean_kendall_tau": float(np.nanmean(tau_list)) if tau_list else float("nan"),
        "avg_min_kendall_tau": float(np.nanmean(mintau_list)) if mintau_list else float("nan"),
        "pct_top1_stable": round(100 * sum(top1_list) / rows, 2) if rows else 0.0,
        "pct_topk_stable": round(100 * sum(topk_list) / rows, 2) if rows else 0.0,
        "pct_exact_ranking_match": round(100 * sum(exact_list) / rows, 2) if rows else 0.0,
        # positional top-k stability: avg # of top-k ranks holding the same object
        # across all permutations, and the avg # of positions actually checked.
        "avg_topk_position_matches": round(sum(posmatch_list) / rows, 4) if rows else 0.0,
        "avg_topk_positions_checked": round(sum(poschecked_list) / rows, 4) if rows else 0.0,
    }
    results["__summary__"] = summary
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 64)
    print(f"  Shapley permutation robustness  ({rows} rows, dataset={args.dataset})")
    print("=" * 64)
    print(f"avg mean Kendall-tau : {summary['avg_mean_kendall_tau']:.4f}")
    print(f"avg min  Kendall-tau : {summary['avg_min_kendall_tau']:.4f}")
    print(f"top-1 stable rows    : {summary['pct_top1_stable']}%")
    print(f"top-{args.topk_stable} stable rows    : {summary['pct_topk_stable']}% (same set)")
    print(f"top-{args.topk_stable} same-position  : {summary['avg_topk_position_matches']}/{summary['avg_topk_positions_checked']} ranks (avg)")
    print(f"exact-ranking rows   : {summary['pct_exact_ranking_match']}%")
    print("=" * 64)
    print(f"Results -> {out}")


async def run_benchmark(args):
    rag_counter = RagCounter()
    # Route RAG's LLM calls through the counter WITHOUT modifying retrieve.py:
    # initialize_lightrag resolves `vllm_model_complete` from the src.retrieve
    # module namespace at call time, so rebinding it there makes LightRAG use our
    # counting wrapper for both retrieval keyword-extraction and generation.
    import src.retrieve as _retr
    _retr.vllm_model_complete = rag_counter.make_wrapper()
    rag = await initialize_lightrag(working_dir=WORKING_DIRS[args.dataset])
    hf_model, hf_tok = load_hf_utility_model(args.shap_device, args.shap_load_8bit, args.shap_load_4bit)

    data = load_qa(QA_CSV_PATHS[args.dataset])
    if args.num_rows is not None:
        data = data.head(args.num_rows)

    if args.permute:
        await run_permutation(args, rag, rag_counter, hf_model, hf_tok, data)
    else:
        await run_plain(args, rag, rag_counter, hf_model, hf_tok, data)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_shapley",
                                description="RAG + TMC-Shapley benchmark (+ context-permutation robustness).")
    p.add_argument("--dataset", choices=DATASETS, default="synthetic")
    p.add_argument("--rag-mode", choices=["hybrid", "local", "global", "naive"], default="hybrid")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--num-rows", type=int, default=None, help="Cap on QA rows (default: all).")
    p.add_argument("--shap-device", default="cuda:1", help="Device for the HF utility model.")
    p.add_argument("--shap-load-8bit", action="store_true")
    p.add_argument("--shap-load-4bit", action="store_true")
    p.add_argument("--tmc-iters", type=int, default=50, help="num_iterations_max (permutations).")
    p.add_argument("--tmc-tol", type=float, default=0.01, help="performance_tolerance for truncation.")
    p.add_argument("--tmc-budget", type=int, default=None, help="max_unique_lookups (default: min(2^n, 128)).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--permute", action="store_true",
                   help="Run context-permutation robustness (5 random object orderings) instead of plain metrics.")
    p.add_argument("--topk-stable", dest="topk_stable", type=int, default=2,
                   help="k for the top-k stability check in --permute mode.")
    p.add_argument("--judge", dest="judge", action="store_true", default=True,
                   help="Run the vLLM judge to score answers (default on; ignored in --permute).")
    p.add_argument("--no-judge", dest="judge", action="store_false")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument("--metrics", default=None)
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    asyncio.run(run_benchmark(args))
