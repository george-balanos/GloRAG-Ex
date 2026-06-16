"""
runner.py
=========
Two distinct entry points:

  python -m kg_smile.runner run        — normal KG-SMILE execution
                                         output: dict-of-dicts attribution schema
                                         consumed by aggregation.py / compare.py

  python -m kg_smile.runner robustness — noise-benchmark execution
                                         output: {benchmark: [{noise_pct, result}]}
                                         consumed by evaluation.py / evaluate_robustness.py

Output format (normal run)
--------------------------
Results are written as a JSON object keyed by sequential index string:

    {
        "0": {
            "question":        "...",
            "ground_truth":    "...",
            "rag_answer":      "...",
            "score":           null,
            "n_items":         15,
            "elapsed_seconds": 4.23,
            "llm_call_count":  6,
            "scores":          {"E::NodeA": 0.55, "R::A->B": 0.22, ...},
            ...
        },
        "1": { ... },
        ...
    }
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from src.retrieve import initialize_lightrag
from .kg_smile   import load_full_kg, run_kg_smile, KGSMILEConfig, result_to_dict
from .io_utils   import (
    load_questions_from_csv,
    load_questions_from_explanation,   # re-exported for callers
    load_completed,
    to_output_schema,
)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

ROBUSTNESS_NOISE_LEVELS: list[float] = [0.0, 0.10, 0.20, 0.30, 0.50]


# ─────────────────────────────────────────────────────────────
# Shared question loader
# ─────────────────────────────────────────────────────────────

def _load_questions_from_explanation_dir(explanation_dir: str) -> list[dict]:
    """
    Load questions from counterfactual explanation JSON files (found=True only).
    Returns a list of {question, ground_truth, id} dicts.
    """
    json_files = sorted(
        os.path.join(explanation_dir, f)
        for f in os.listdir(explanation_dir)
        if f.endswith(".json")
    )

    questions = []
    for i, fp in enumerate(json_files):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data["found"]:
            questions.append({
                "question":     data["question"],
                "ground_truth": data["answers"].get("ground_truth"),
                "id":           i,
            })

    print(f"[runner] {len(questions)} solved questions found in {explanation_dir}")
    return questions


# ─────────────────────────────────────────────────────────────
# PIPELINE 1 — Normal run (noise=0, dict-of-dicts output)
# ─────────────────────────────────────────────────────────────

async def run_normal(
    csv_path:         str                  = "/home/gbalanos/GloRAG-Ex/code/datasets/hotpotqa/qa_data_hotpotqa.csv",
    output_path:      str                  = "kg_smile/results/kg_smile_results.json",
    kg_working_dir:   str                  = "KGs/lightrag/hotpotqa",
    kg_graphml:       str                  = "KGs/lightrag/hotpotqa/graph_chunk_entity_relation.graphml",
    num_questions:    int                  = 100,
    explanation_dir:  str                  = "/home/gbalanos/GloRAG-Ex/code/src/counterfactuals/results/synthetic/delete_ops_ft",
    config:           KGSMILEConfig | None = None,
    explanation_mode: str | None           = None,
) -> None:
    """
    Run KG-SMILE once per question at noise=0.

    Questions are loaded from `explanation_dir` (JSON files produced by the
    counterfactual pipeline where found=True).  Completed-question resume is
    supported via the dict-of-dicts output file.

    Output schema — one entry per question (dict keyed by sequential index):
        {
            "question", "ground_truth", "rag_answer", "score", "n_items",
            "elapsed_seconds", "llm_call_count", "scores",
            "id", "surrogate_r2", "output_shift_std", "degenerate",
            "timestamp", "edge_attributions", "node_attributions", ...
        }

    Consumed by aggregation.py and compare.py.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build config, always enforcing noise_pct=0
    base = config or KGSMILEConfig()
    config = KGSMILEConfig(
        n_perturbations=base.n_perturbations,
        kernel_width=base.kernel_width,
        retrieval_mode=base.retrieval_mode,
        retrieval_top_k=base.retrieval_top_k,
        random_seed=base.random_seed,
        max_tokens=base.max_tokens,
        embedding_model=base.embedding_model,
        noise_pct=0.0,
    )

    kg_full = load_full_kg(kg_graphml)
    print("[runner:normal] Initialising LightRAG ...")
    rag = await initialize_lightrag(kg_working_dir)

    questions = _load_questions_from_explanation_dir(explanation_dir)

    completed_questions, results = load_completed(output_path)
    next_index = max((int(k) for k in results), default=-1) + 1

    remaining = [q for q in questions if q["question"] not in completed_questions]
    print(f"[runner:normal] {len(remaining)} questions remaining")

    if not remaining:
        print("[runner:normal] Nothing to do — all questions already completed.")
        return

    for i, item in enumerate(remaining):
        global_num = next_index + i
        print(f"\n[runner:normal] {i+1}/{len(remaining)}  "
              f"(global index {global_num}, total saved {len(results)})")

        t_start = time.perf_counter()
        try:
            result = await run_kg_smile(
                query=item["question"], rag=rag, KG_full=kg_full, config=config,
                mode=explanation_mode, ground_truth=item["ground_truth"],
            )
            elapsed   = time.perf_counter() - t_start
            llm_calls = getattr(result, "llm_call_count", None)

            entry = to_output_schema(
                question=item["question"],
                result=result,
                ground_truth=item.get("ground_truth"),
                question_id=item.get("id"),
                elapsed_seconds=round(elapsed, 4),
                llm_call_count=llm_calls,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t_start
            print(f"[runner:normal ERROR] {e}")
            entry = {
                "question":        item["question"],
                "error":           str(e),
                "elapsed_seconds": round(elapsed, 4),
            }

        results[str(global_num)] = entry
        _save(results, output_path)

    print(f"\n[runner:normal] Done. {len(results)} results saved to {output_path}")


# ─────────────────────────────────────────────────────────────
# PIPELINE 2 — Robustness benchmark (multiple noise levels)
# ─────────────────────────────────────────────────────────────

async def _run_at_noise(
    item:        dict,
    rag,
    kg_full,
    base_config: KGSMILEConfig,
    noise:       float,
) -> dict:
    """Run KG-SMILE at a single noise level; return one benchmark entry."""
    config = KGSMILEConfig(
        n_perturbations=base_config.n_perturbations,
        kernel_width=base_config.kernel_width,
        retrieval_mode=base_config.retrieval_mode,
        retrieval_top_k=base_config.retrieval_top_k,
        random_seed=base_config.random_seed,
        max_tokens=base_config.max_tokens,
        embedding_model=base_config.embedding_model,
        noise_pct=noise,
    )
    label = f"{noise * 100:.0f}%"
    print(f"  [robustness] noise={label}")

    t_start = time.perf_counter()
    try:
        result = await run_kg_smile(
            query=item["question"], rag=rag, KG_full=kg_full, config=config,
        )
        elapsed   = time.perf_counter() - t_start
        llm_calls = getattr(result, "llm_call_count", None)
        return {
            "noise_pct":       noise,
            "result":          result_to_dict(result),
            "elapsed_seconds": round(elapsed, 4),
            "llm_call_count":  llm_calls,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t_start
        print(f"  [robustness ERROR] noise={label}: {e}")
        return {
            "noise_pct":       noise,
            "error":           str(e),
            "elapsed_seconds": round(elapsed, 4),
        }


async def run_robustness(
    csv_path:        str                    = "/home/gbalanos/GloRAG-Ex/code/datasets/hotpotqa/qa_data_hotpotqa.csv",
    output_path:     str                    = "results/robustness_results.json",
    kg_working_dir:  str                    = "KGs/lightrag/hotpotqa",
    kg_graphml:      str                    = "KGs/lightrag/hotpotqa/graph_chunk_entity_relation.graphml",
    num_questions:   int                    = 100,
    explanation_dir: str | None             = None,
    noise_levels:    list[float] | None     = None,
    config:          KGSMILEConfig | None   = None,
) -> None:
    """
    Run KG-SMILE at each noise level for every question.

    Questions are loaded from `explanation_dir` (same logic as run_normal) when
    provided, otherwise from the CSV file.

    Output schema — one entry per question (dict keyed by sequential index):
        {
            "id", "question",
            "benchmark": [
                {
                    "noise_pct":       float,
                    "result":          dict,      # or absent on error
                    "elapsed_seconds": float,
                    "llm_call_count":  int | null,
                    "error":           str,       # only present on error
                },
                ...
            ]
        }

    noise=0.0 is always included as the baseline required by evaluation.py.
    Consumed by evaluation.py and evaluate_robustness.py.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    noise_levels = list(noise_levels or ROBUSTNESS_NOISE_LEVELS)
    if 0.0 not in noise_levels:
        noise_levels = [0.0] + noise_levels

    config = config or KGSMILEConfig(n_perturbations=20, kernel_width=0.25,
                                     retrieval_mode="hybrid", retrieval_top_k=2)

    kg_full = load_full_kg(kg_graphml)
    print("[runner:robustness] Initialising LightRAG ...")
    rag = await initialize_lightrag(kg_working_dir)

    if explanation_dir is not None:
        questions = _load_questions_from_explanation_dir(explanation_dir)
    else:
        questions = load_questions_from_csv(csv_path, num=num_questions)

    completed_questions, results = load_completed(output_path)
    next_index = max((int(k) for k in results), default=-1) + 1

    remaining = [q for q in questions if q["question"] not in completed_questions]
    print(f"[runner:robustness] {len(remaining)} questions remaining")

    if not remaining:
        print("[runner:robustness] Nothing to do — all questions already completed.")
        return

    for i, item in enumerate(remaining):
        global_num = next_index + i
        print(f"\n[runner:robustness] {i+1}/{len(remaining)}  "
              f"(global index {global_num}, total saved {len(results)})")

        bench = [
            await _run_at_noise(item, rag, kg_full, config, noise)
            for noise in sorted(noise_levels)
        ]

        results[str(global_num)] = {
            "id":        item["id"],
            "question":  item["question"],
            "benchmark": bench,
        }
        _save(results, output_path)

    print(f"\n[runner:robustness] Done. {len(results)} results saved to {output_path}")


# ─────────────────────────────────────────────────────────────
# Shared save helper
# ─────────────────────────────────────────────────────────────

def _save(results: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[runner] Saved -> {path}")


# ─────────────────────────────────────────────────────────────
# CLI dispatch
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="KG-SMILE runner — 'run' for normal execution, "
                    "'robustness' for noise benchmarking."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--csv",              default="/home/gbalanos/GloRAG-Ex/code/datasets/hotpotqa/qa_data_hotpotqa.csv")
    shared.add_argument("--explanation-dir",  default="/home/gbalanos/GloRAG-Ex/code/src/counterfactuals/results/synthetic/delete_ops_ft")
    shared.add_argument("--kg-dir",           default="KGs/lightrag/hotpotqa")
    shared.add_argument("--kg-graphml",       default="KGs/lightrag/hotpotqa/graph_chunk_entity_relation.graphml")
    shared.add_argument("--num",              type=int,   default=100)
    shared.add_argument("--n-pert",           type=int,   default=20)
    shared.add_argument("--kernel-width",     type=float, default=0.25)
    shared.add_argument("--top-k",            type=int,   default=2)
    shared.add_argument("--explanation-mode", type=str,   default="ft")

    p_run = sub.add_parser("run", parents=[shared],
                            help="Normal run — output for aggregation.py / compare.py")
    p_run.add_argument("--output", default="kg_smile/results/kg_smile_results.json")

    p_rob = sub.add_parser("robustness", parents=[shared],
                            help="Noise benchmark — output for evaluation.py / evaluate_robustness.py")
    p_rob.add_argument("--output", default="results/robustness_results.json")
    p_rob.add_argument("--noise-levels", nargs="+", type=float,
                       default=ROBUSTNESS_NOISE_LEVELS, metavar="NOISE")

    args = parser.parse_args()
    cfg  = KGSMILEConfig(n_perturbations=args.n_pert,
                         kernel_width=args.kernel_width,
                         retrieval_top_k=args.top_k)

    if args.mode == "run":
        asyncio.run(run_normal(
            csv_path=args.csv,
            output_path=args.output,
            kg_working_dir=args.kg_dir,
            kg_graphml=args.kg_graphml,
            num_questions=args.num,
            explanation_dir=args.explanation_dir,
            config=cfg,
            explanation_mode=args.explanation_mode,
        ))
    elif args.mode == "robustness":
        asyncio.run(run_robustness(
            csv_path=args.csv,
            output_path=args.output,
            kg_working_dir=args.kg_dir,
            kg_graphml=args.kg_graphml,
            num_questions=args.num,
            explanation_dir=args.explanation_dir,
            noise_levels=args.noise_levels,
            config=cfg,
        ))