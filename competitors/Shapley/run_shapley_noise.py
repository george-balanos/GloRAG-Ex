"""Shapley noise-resistance benchmark — the TMC-Shapley analog of the CFE
noise-resistance test in src/quality_metrics/noise_resistance.py.

For each QA row this:
  1. Retrieves the subgraph (entities + relations) via LightRAG and generates the
     clean RAG answer (kept only as the judge reference).
  2. For each noise level, injects random foreign nodes/edges into the context
     graph using the SAME `add_random_noise_nodes` the CFE test uses, regenerates
     the answer on the NOISY context (`noisy_answer`), and runs Truncated Monte
     Carlo Shapley over the noisy object bag attributing `noisy_answer`.
  3. Measures how much of the answer's attribution mass lands on the injected
     noise objects (the Shapley analog of CFE's `noise_in_explanation`). A faithful
     attributor should give noise ~0 importance when the noise did not change the
     answer (judge-robust rows).

This script lives in competitors/Shapley/ (decoupled from code/) but still uses
code/src. Run with CWD = code/ so the relative dataset paths resolve, e.g.:
  cd code && ../.venv/bin/python ../competitors/Shapley/run_shapley_noise.py \
      --dataset synthetic --rag-mode hybrid --top-k 2 --shap-device cuda:1 \
      --noise-percentages 0.1,0.3,0.5,0.8
Or just use competitors/Shapley/run_shapley_experiments.sh, which sets this up.
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
    build_objects,
    object_id,
    run_tmc,
    load_hf_utility_model,
    load_qa,
    RagCounter,
)

from src.retrieve import initialize_lightrag, retrieve_subgraph_objects
from src.query import query
from src.llm_judge import judge_response
from src.parser import parse_graph, graph_to_subgraph, render_context_from_objects
from src.dataset_setup import WORKING_DIRS, QA_CSV_PATHS, DATASETS

from src.quality_metrics.noise_resistance import add_random_noise_nodes

from tqdm import tqdm
import argparse
import asyncio
import json
import logging
import time

import networkx as nx
import numpy as np

logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("nano-vectordb").setLevel(logging.WARNING)
logging.getLogger("lightrag").setLevel(logging.WARNING)


# ── Noise-object identification ─────────────────────────────────────────────
def noise_ids_from_ops(noise_ops) -> set[str]:
    """Map `add_random_noise_nodes` ops to the object_id space.

    add_node  -> "E::<new_node>"
    add_edge  -> "R::<new_node>-><anchor>"   (anchors are always original nodes,
                 and graph_to_subgraph yields src=new_node, tgt=anchor)
    """
    ids: set[str] = set()
    for op in noise_ops:
        kind = op[0]
        if kind == "add_node":
            ids.add(f"E::{op[1]}")
        elif kind == "add_edge":
            src, tgt = op[1]
            ids.add(f"R::{src}->{tgt}")
    return ids


# ── Per-row metrics ─────────────────────────────────────────────────────────
def compute_noise_metrics(scores_by_id: dict[str, float], noise_ids: set[str], top_attr_k: int) -> dict:
    """How much of the answer's Shapley attribution lands on the noise objects."""
    all_ids = list(scores_by_id.keys())
    noise_present = [i for i in all_ids if i in noise_ids]
    orig_present = [i for i in all_ids if i not in noise_ids]

    noise_vals = np.array([scores_by_id[i] for i in noise_present], dtype=float)
    orig_vals = np.array([scores_by_id[i] for i in orig_present], dtype=float)
    all_vals = np.array([scores_by_id[i] for i in all_ids], dtype=float)

    def frac(num: float, den: float) -> float:
        return float(num / den) if den != 0 else 0.0

    abs_total = float(np.abs(all_vals).sum())
    pos_total = float(np.clip(all_vals, 0, None).sum())
    signed_total = float(all_vals.sum())
    noise_abs = float(np.abs(noise_vals).sum())
    noise_pos = float(np.clip(noise_vals, 0, None).sum())
    noise_signed = float(noise_vals.sum())

    # Ranking: object ids by descending Shapley value (the "explanation").
    ranking = sorted(all_ids, key=lambda i: scores_by_id[i], reverse=True)
    k_eff = min(top_attr_k, len(ranking))
    topk = ranking[:k_eff]
    
    num_noise_in_topk = sum(1 for i in topk if i in noise_ids)
    best_noise_rank = next((pos for pos, i in enumerate(ranking, 1) if i in noise_ids), None)

    return {
        "n_objects": len(all_ids),
        "n_noise": len(noise_present),
        "n_original": len(orig_present),
        "noise_abs_frac": frac(noise_abs, abs_total),
        "noise_pos_frac": frac(noise_pos, pos_total),
        "noise_signed_frac": frac(noise_signed, signed_total),
        "noise_abs_mass": noise_abs,
        "total_abs_mass": abs_total,
        "mean_noise_shap": float(noise_vals.mean()) if noise_present else 0.0,
        "max_noise_shap": float(noise_vals.max()) if noise_present else 0.0,
        "mean_orig_shap": float(orig_vals.mean()) if orig_present else 0.0,
        "max_orig_shap": float(orig_vals.max()) if orig_present else 0.0,
        "top_attr_k": k_eff,
        "noise_in_topk": num_noise_in_topk > 0,
        "num_noise_in_topk": num_noise_in_topk,
        "noise_in_topk_frac": frac(num_noise_in_topk, len(noise_present)), # <-- NEW: Fraction of noise objects in top-k
        "best_noise_rank": best_noise_rank,
        "ranking": ranking,
    }


# ── Aggregation ─────────────────────────────────────────────────────────────
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def summarize_level(level_records: dict) -> dict:
    """Aggregate per-row metrics for one noise level (overall + robust split)."""
    rows = [r["metrics"] for r in level_records.values()]
    n = len(rows)
    robust = [r for r in level_records.values() if r.get("noise_robust") is True]
    fragile = [r for r in level_records.values() if r.get("noise_robust") is False]
    judged = robust + fragile

    def frac_block(records):
        ms = [r["metrics"] for r in records]
        return {
            "rows": len(ms),
            "avg_noise_abs_frac": _mean([m["noise_abs_frac"] for m in ms]),
            "avg_noise_pos_frac": _mean([m["noise_pos_frac"] for m in ms]),
            "pct_noise_in_topk": round(100 * sum(m["noise_in_topk"] for m in ms) / len(ms), 2) if ms else 0.0,
            "avg_noise_in_topk_frac": _mean([m.get("noise_in_topk_frac", 0.0) for m in ms]),
        }

    out = {
        "rows": n,
        "avg_noise_abs_frac": _mean([m["noise_abs_frac"] for m in rows]),
        "avg_noise_pos_frac": _mean([m["noise_pos_frac"] for m in rows]),
        "avg_num_noise_in_topk": _mean([m["num_noise_in_topk"] for m in rows]),
        "avg_noise_in_topk_frac": _mean([m.get("noise_in_topk_frac", 0.0) for m in rows]),
        "pct_noise_in_topk": round(100 * sum(m["noise_in_topk"] for m in rows) / n, 2) if n else 0.0,
        "avg_n_noise_objects": _mean([m["n_noise"] for m in rows]),
    }
    if judged:
        out["pct_noise_robust"] = round(100 * len(robust) / len(judged), 2)
        out["robust_rows"] = frac_block(robust)
        out["fragile_rows"] = frac_block(fragile)
    return out


# ── Main run ────────────────────────────────────────────────────────────────
async def run_noise(args, rag, rag_counter, hf_model, hf_tok, G, data):
    noise_percentages = _parse_noise_percentages(args.noise_percentages)
    # results[level_key][rid] = {...} ; metrics[level_key][rid] = {cost/timing}
    results = {f"noise_level_{int(p * 100)}": {} for p in noise_percentages}
    metrics = {f"noise_level_{int(p * 100)}": {} for p in noise_percentages}

    for row_idx, (_, row) in enumerate(
            tqdm(data.iterrows(), desc="Shapley noise", total=len(data))):
        rid = str(row["id"])
        question, ground_truth = row["questions"], row["answers"]

        # Retrieval + clean answer are noise-independent: compute once per row.
        rag_counter.reset()
        context, sg = await retrieve_subgraph_objects(
            rag, query=question, mode=args.rag_mode, top_k=args.top_k)
        if len(sg.entities) + len(sg.relations) == 0:
            print(f"[{rid}] no retrieved objects; skipping.")
            continue
        original_answer = await query(rag, context, question)
        setup_calls, setup_time = rag_counter.calls, rag_counter.time
        cg = parse_graph(sg)  # DiGraph rebuilt from the parsed subgraph

        for p in noise_percentages:
            level_key = f"noise_level_{int(p * 100)}"
            row_seed = args.seed + row_idx

            noisy_cg, noise_ops = add_random_noise_nodes(cg, G, noise_pct=p, seed=row_seed)
            noisy_sg = graph_to_subgraph(noisy_cg)
            objects = build_objects(noisy_sg.entities, noisy_sg.relations)
            # Same render used for both the generation prompt and TMC's full set.
            noisy_context = render_context_from_objects(objects)

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
            scores, shap_evals = run_tmc(
                objects, question, hf_model, hf_tok, args.shap_device, noisy_answer, args)
            shap_time = time.perf_counter() - st0
            scores_by_id = {object_id(k, o): s for (k, o), s in zip(objects, scores)}

            noise_ids = noise_ids_from_ops(noise_ops)
            m = compute_noise_metrics(scores_by_id, noise_ids, args.top_attr_k)

            results[level_key][rid] = {
                "question": question,
                "ground_truth": ground_truth,
                "original_answer": original_answer,
                "noisy_answer": noisy_answer,
                "noise_pct": p,
                "noise_score": noise_score,
                "noise_robust": noise_robust,
                "num_noise_ops": len(noise_ops),
                "noise_ids": sorted(noise_ids),
                "shapley_scores": scores_by_id,
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

            print(f"[{rid} | noise={int(p * 100)}%] objs={m['n_objects']} (noise={m['n_noise']}) "
                  f"robust={noise_robust} noise_abs_frac={m['noise_abs_frac']:.3f} "
                  f"in_top{m['top_attr_k']}={m['noise_in_topk']}({m['num_noise_in_topk']}/{m['n_noise']}) "
                  f"best_noise_rank={m['best_noise_rank']} | shap {shap_evals} evals/{shap_time:.1f}s")

    _write_outputs(args, results, metrics, noise_percentages)


def _write_outputs(args, results, metrics, noise_percentages):
    summary = {}
    for p in noise_percentages:
        level_key = f"noise_level_{int(p * 100)}"
        if results[level_key]:
            summary[level_key] = summarize_level(results[level_key])
    results["__summary__"] = summary
    metrics["__summary__"] = summary

    out_results = args.output or f"benchmark/results/{args.dataset}_shapley_noise.json"
    out_metrics = args.metrics or f"benchmark/results/{args.dataset}_shapley_noise_metrics.json"
    os.makedirs(os.path.dirname(out_results), exist_ok=True)
    with open(out_results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 85)
    print(f"  Shapley noise resistance  (dataset={args.dataset}, target=noisy-context answer)")
    print("=" * 85)
    print(f"{'noise':<8}{'rows':>6}{'avg|noise|frac':>16}{'avg pos frac':>14}"
          f"{'%noise_in_top':>16}{'avg_top_frac':>14}{'%robust':>10}")
    for p in noise_percentages:
        s = summary.get(f"noise_level_{int(p * 100)}")
        if not s:
            continue
        print(f"{int(p * 100):>3}% {'':<3}{s['rows']:>6}{s['avg_noise_abs_frac']:>16.4f}"
              f"{s['avg_noise_pos_frac']:>14.4f}{s['pct_noise_in_topk']:>16.2f}"
              f"{s['avg_noise_in_topk_frac']:>14.4f}{s.get('pct_noise_robust', float('nan')):>10}")
    # The headline result: among robust rows (noise didn't change the answer),
    # is the noise attribution near zero?
    print("-" * 85)
    for p in noise_percentages:
        s = summary.get(f"noise_level_{int(p * 100)}")
        if not s or "robust_rows" not in s:
            continue
        rb, fr = s["robust_rows"], s["fragile_rows"]
        print(f"{int(p * 100):>3}%  robust(n={rb['rows']}) avg|noise|frac={rb['avg_noise_abs_frac']:.4f} "
              f"| fragile(n={fr['rows']}) avg|noise|frac={fr['avg_noise_abs_frac']:.4f}")
    print("=" * 85)
    print(f"Results -> {out_results}\nMetrics -> {out_metrics}")


async def run_benchmark(args):
    rag_counter = RagCounter()
    # Route RAG's LLM calls through the counter WITHOUT modifying retrieve.py
    # (same trick as run_shapley): rebind the name in the src.retrieve namespace.
    import src.retrieve as _retr
    _retr.vllm_model_complete = rag_counter.make_wrapper()
    rag = await initialize_lightrag(working_dir=WORKING_DIRS[args.dataset])
    hf_model, hf_tok = load_hf_utility_model(args.shap_device, args.shap_load_8bit, args.shap_load_4bit)

    # Full dataset graph the noise injector samples foreign nodes from.
    G = nx.read_graphml(f"{WORKING_DIRS[args.dataset]}/graph_chunk_entity_relation.graphml")

    data = load_qa(QA_CSV_PATHS[args.dataset])
    if args.num_rows is not None:
        data = data.head(args.num_rows)

    await run_noise(args, rag, rag_counter, hf_model, hf_tok, G, data)


# ── CLI ─────────────────────────────────────────────────────────────────────
def _parse_noise_percentages(spec: str) -> list:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError:
            raise SystemExit(f"--noise-percentages: '{tok}' is not a float.")
        if not (0.0 < v < 1.0):
            raise SystemExit(f"--noise-percentages: {v} must be in (0, 1).")
        out.append(v)
    if not out:
        raise SystemExit("--noise-percentages: at least one value required.")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_shapley_noise",
        description="Shapley noise-resistance benchmark: inject noise into the RAG "
                    "context, regenerate the answer, and measure how much TMC-Shapley "
                    "attribution lands on the injected noise objects.")
    p.add_argument("--dataset", choices=DATASETS, default="synthetic")
    p.add_argument("--rag-mode", choices=["hybrid", "local", "global", "naive"], default="hybrid")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--num-rows", type=int, default=None, help="Cap on QA rows (default: all).")
    p.add_argument("--noise-percentages", default="0.1,0.3,0.5,0.8",
                   help="Comma-separated noise fractions in (0, 1). One Shapley run per fraction × row.")
    p.add_argument("--top-attr-k", type=int, default=5,
                   help="k for the 'noise in top-k attributions' check.")
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
    asyncio.run(run_benchmark(args))