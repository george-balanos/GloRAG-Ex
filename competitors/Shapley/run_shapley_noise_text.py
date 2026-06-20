"""Shapley noise-resistance benchmark over TEXT CHUNKS — the text-excerpt analog of
run_shapley_noise.py.

Where the graph version injects random foreign NODES/EDGES into the retrieved
subgraph, this injects random foreign CHUNKS into the retrieved chunk bag. For each
QA row it:
  1. Retrieves the chunks via LightRAG and generates the clean RAG answer (kept only
     as the judge reference).
  2. For each noise level, inserts foreign chunks (sampled from the pool of chunks
     retrieved for OTHER rows of the same dataset) into the chunk bag, regenerates the
     answer on the NOISY chunk context, and runs Truncated Monte Carlo Shapley over the
     noisy chunk bag attributing `noisy_answer`.
  3. Measures how much of the answer's attribution mass lands on the injected noise
     chunks. A faithful attributor should give noise ~0 importance on judge-robust rows
     (noise did not change the answer).

The per-row noise metrics, aggregation, output writing and CLI parsing are reused
verbatim from run_shapley_noise.py; only the noise UNIT (chunks, not graph elements)
and the foreign pool differ.

Note: the foreign pool is the union of chunks retrieved across all rows, so run with
enough rows for a non-trivial pool (a single row yields no foreign chunks). Run with
CWD = code/ so relative dataset paths resolve, e.g.:
  cd code && ../.venv/bin/python ../competitors/Shapley/run_shapley_noise_text.py \
      --dataset synthetic --rag-mode hybrid --top-k 2 --shap-device cuda:1 \
      --noise-percentages 0.1,0.3,0.5,0.8
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_CODE_DIR = os.path.join(_REPO_ROOT, "code")
for _p in (_CODE_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_shapley import load_hf_utility_model, load_qa, RagCounter
from run_shapley_noise import (
    compute_noise_metrics,
    _write_outputs as _write_noise_outputs,
    _parse_noise_percentages,
    _parse_top_attr_ks,
)
from run_shapley_text import run_tmc_chunks, split_into_players, unit_id

from src.retrieve import initialize_lightrag
from src.query import query
from src.llm_judge import judge_response
from src.dataset_setup import WORKING_DIRS, QA_CSV_PATHS, DATASETS

from chunk_utils import retrieve_chunks, render_context_from_chunks

from tqdm import tqdm
import argparse
import asyncio
import logging
import random
import time

logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("nano-vectordb").setLevel(logging.WARNING)
logging.getLogger("lightrag").setLevel(logging.WARNING)


def add_random_noise_chunks(chunks: list[str], foreign_pool: list[str],
                            noise_pct: float, seed: int | None = None):
    rng = random.Random(seed)
    own = set(chunks)
    candidates = [c for c in foreign_pool if c not in own]
    if not candidates or not chunks:
        return list(chunks), set()
    n_noise = min(max(1, round(noise_pct * len(chunks))), len(candidates))
    noise = rng.sample(candidates, n_noise)
    noisy = list(chunks)
    for c in noise:
        noisy.insert(rng.randint(0, len(noisy)), c)
    return noisy, set(noise)


async def run_noise(args, rag, rag_counter, hf_model, hf_tok, data):
    noise_percentages = _parse_noise_percentages(args.noise_percentages)
    results = {f"noise_level_{int(p * 100)}": {} for p in noise_percentages}
    metrics = {f"noise_level_{int(p * 100)}": {} for p in noise_percentages}

    row_players, row_answer, row_meta, pool, seen = {}, {}, {}, [], set()
    for _, row in tqdm(data.iterrows(), desc="retrieve+pool", total=len(data)):
        rid = str(row["id"])
        question = row["questions"]
        rag_counter.reset()
        context, chunks = await retrieve_chunks(rag, query=question, mode=args.rag_mode, top_k=args.top_k)
        players = split_into_players(chunks, args.granularity)
        if len(players) == 0:
            print(f"[{rid}] no retrieved {args.granularity}s; skipping.")
            continue
        clean = await query(rag, context, question)
        row_players[rid] = players
        row_answer[rid] = clean
        row_meta[rid] = {"question": question, "ground_truth": row["answers"],
                         "setup_calls": rag_counter.calls, "setup_time": rag_counter.time}
        for c in players:
            if c not in seen:
                seen.add(c)
                pool.append(c)

    rids = list(row_players.keys())
    print(f"Foreign-{args.granularity} pool: {len(pool)} unique {args.granularity}s across {len(rids)} rows.")

    for row_idx, rid in enumerate(tqdm(rids, desc="Shapley(text) noise", total=len(rids))):
        players = row_players[rid]
        question = row_meta[rid]["question"]
        ground_truth = row_meta[rid]["ground_truth"]
        original_answer = row_answer[rid]
        setup_calls, setup_time = row_meta[rid]["setup_calls"], row_meta[rid]["setup_time"]

        for p in noise_percentages:
            level_key = f"noise_level_{int(p * 100)}"
            row_seed = args.seed + row_idx

            noisy_units, noise_set = add_random_noise_chunks(players, pool, p, seed=row_seed)
            noisy_context = render_context_from_chunks(noisy_units)

            rag_counter.reset()
            noisy_answer = await query(rag, noisy_context, question)
            gen_calls, gen_time = rag_counter.calls, rag_counter.time

            noise_score, noise_robust = None, None
            judge_calls, judge_time = 0, 0.0
            if args.judge:
                jt0 = time.perf_counter()
                noise_score = await judge_response(question, noisy_answer, original_answer)
                judge_time = time.perf_counter() - jt0
                judge_calls = 1
                noise_robust = noise_score != 0

            st0 = time.perf_counter()
            scores, shap_evals = run_tmc_chunks(
                noisy_units, question, hf_model, hf_tok, args.shap_device, noisy_answer, args)
            shap_time = time.perf_counter() - st0
            scores_by_id = {unit_id(i, args.granularity): s for i, s in enumerate(scores)}
            noise_ids = {unit_id(i, args.granularity) for i, t in enumerate(noisy_units) if t in noise_set}

            m = compute_noise_metrics(scores_by_id, noise_ids, args.top_attr_ks)

            results[level_key][rid] = {
                "question": question,
                "ground_truth": ground_truth,
                "original_answer": original_answer,
                "noisy_answer": noisy_answer,
                "noise_pct": p,
                "noise_score": noise_score,
                "noise_robust": noise_robust,
                "granularity": args.granularity,
                "num_noise_units": len(noise_set),
                "noise_ids": sorted(noise_ids),
                "shapley_scores": scores_by_id,
                "player_texts": {unit_id(i, args.granularity): t for i, t in enumerate(noisy_units)},
                "metrics": m,
            }
            metrics[level_key][rid] = {
                "rag_setup_calls": setup_calls, "rag_setup_time": round(setup_time, 4),
                "rag_gen_calls": gen_calls, "rag_gen_time": round(gen_time, 4),
                "judge_calls": judge_calls, "judge_time": round(judge_time, 4),
                "shap_utility_evals": shap_evals, "shap_forward_passes": shap_evals * 2,
                "shap_time": round(shap_time, 4),
                "n_objects": m["n_objects"], "n_noise": m["n_noise"],
            }

            topk_str = " ".join(
                f"t{k}={'Y' if m['topk'][str(k)]['in_topk'] else 'n'}({m['topk'][str(k)]['num_in_topk']})"
                for k in args.top_attr_ks)
            print(f"[{rid} | noise={int(p * 100)}%] chunks={m['n_objects']} (noise={m['n_noise']}) "
                  f"robust={noise_robust} noise_abs_frac={m['noise_abs_frac']:.3f} "
                  f"{topk_str} best_noise_rank={m['best_noise_rank']} | shap {shap_evals} evals/{shap_time:.1f}s")

    args.output = args.output or f"benchmark/results/{args.dataset}_shapley_text_noise.json"
    args.metrics = args.metrics or f"benchmark/results/{args.dataset}_shapley_text_noise_metrics.json"
    _write_noise_outputs(args, results, metrics, noise_percentages)


async def run_benchmark(args):
    rag_counter = RagCounter()
    import src.retrieve as _retr
    _retr.vllm_model_complete = rag_counter.make_wrapper()
    rag = await initialize_lightrag(working_dir=WORKING_DIRS[args.dataset])
    hf_model, hf_tok = load_hf_utility_model(args.shap_device, args.shap_load_8bit, args.shap_load_4bit)

    data = load_qa(QA_CSV_PATHS[args.dataset])
    if args.num_rows is not None:
        data = data.head(args.num_rows)

    await run_noise(args, rag, rag_counter, hf_model, hf_tok, data)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_shapley_noise_text",
        description="Shapley noise-resistance over text chunks: inject foreign chunks "
                    "into the RAG context, regenerate the answer, and measure how much "
                    "TMC-Shapley attribution lands on the injected noise chunks.")
    p.add_argument("--dataset", choices=DATASETS, default="synthetic")
    p.add_argument("--rag-mode", choices=["hybrid", "local", "global", "naive"], default="hybrid")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--num-rows", type=int, default=None, help="Cap on QA rows (default: all).")
    p.add_argument("--noise-percentages", default="0.1,0.3,0.5,0.8",
                   help="Comma-separated noise fractions in (0, 1). One Shapley run per fraction × row.")
    p.add_argument("--top-attr-ks", default="1,3,5",
                   help="Comma-separated k values for the 'noise in top-k attributions' check (e.g. 1,3,5).")
    p.add_argument("--shap-device", default="cuda:1", help="Device for the HF utility model.")
    p.add_argument("--shap-load-8bit", action="store_true")
    p.add_argument("--shap-load-4bit", action="store_true")
    p.add_argument("--tmc-iters", type=int, default=50, help="num_iterations_max (permutations).")
    p.add_argument("--tmc-tol", type=float, default=0.01, help="performance_tolerance for truncation.")
    p.add_argument("--tmc-budget", type=int, default=None, help="max_unique_lookups (default: min(2^n, 128)).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge", dest="judge", action="store_true", default=True,
                   help="Judge noisy answer vs clean answer to flag noise_robust (default on).")
    p.add_argument("--no-judge", dest="judge", action="store_false")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument("--metrics", default=None)
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    args.top_attr_ks = _parse_top_attr_ks(args.top_attr_ks)
    asyncio.run(run_benchmark(args))
