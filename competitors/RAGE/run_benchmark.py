"""Sentence-context benchmark for RAGE — the fair analog of code/benchmark/run.py.

`code/benchmark/run.py` scores the RAG system over the QA set using the **graph**
context (`retrieve_subgraph`) and `code/benchmark/evaluation.py` turns the RAG +
LLM-only runs into the FF/FT/TF/TT `comparison_<ds>.json`. But RAGE explains the
**sentence** context, not the graph, so a case labelled `ft` (rag-correct on the
graph) need not be rag-correct on sentences — the comparison's `rag_answer` would
not match RAGE's `original_answer`, biasing its correctness.

This script reproduces `run.py` but scores RAG with the **exact same sentence
pipeline run_rage.py uses** — `retrieve_chunks` -> `split_into_players(_, "sentence")`
-> `render_context_from_chunks` -> `query` — so the comparison's `rag_answer` is
byte-identical to the answer RAGE actually explains. With `--build-comparison` it
also runs the LLM-only (`bypass`) pass and calls `export_performance_cases`
(reused verbatim from benchmark/evaluation.py) to emit a sentence-context
`comparison_<ds>_sentence_<top_k>.json` with the standard {summary, cases, results}
schema that run_rage.py / run_rage_noise.py consume via `--comparison`.

  cd code && ../.venv/bin/python ../competitors/RAGE/run_benchmark.py \
      --dataset hotpotqa --rag-mode hybrid --top-k 2 --build-comparison \
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

from run_shapley import load_qa  # noqa: E402
from run_shapley_text import split_into_players  # noqa: E402

from src.retrieve import initialize_lightrag  # noqa: E402
from src.query import query  # noqa: E402
from src.llm_judge import judge_response  # noqa: E402
from src.dataset_setup import WORKING_DIRS, QA_CSV_PATHS, DATASETS  # noqa: E402

from chunk_utils import retrieve_chunks, render_context_from_chunks  # noqa: E402
from benchmark.evaluation import export_performance_cases  # noqa: E402

from tqdm import tqdm  # noqa: E402
import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402

logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("nano-vectordb").setLevel(logging.WARNING)
logging.getLogger("lightrag").setLevel(logging.WARNING)

GRANULARITY = "sentence"


async def run_example(rag, question, ground_truth, mode, top_k):
    """RAG answer scored over the SENTENCE context (identical to run_rage.py's
    full-context answer). `mode == "bypass"` -> empty context = LLM-only baseline."""
    if mode == "bypass" or top_k == 0:
        sentences = []
    else:
        _, chunks = await retrieve_chunks(rag, query=question, mode=mode, top_k=top_k)
        sentences = split_into_players(chunks, GRANULARITY)
    context = render_context_from_chunks(sentences)        # empty block when no sentences
    generated_answer = await query(rag, context, question)
    score = await judge_response(question, generated_answer=generated_answer, ground_truth=ground_truth)
    return score, generated_answer, len(sentences)


async def run_pass(rag, data, mode, top_k, desc):
    """One pass (RAG-sentence or bypass) -> {id: {score, generated_answer, question,
    ground_truth}} (the run.py schema, so benchmark/evaluation.py consumes it as-is)."""
    out = {}
    for _, row in tqdm(data.iterrows(), desc=desc, total=len(data)):
        rid = str(row["id"])
        question, answer = row["questions"], row["answers"]
        score, generated_answer, n_sents = await run_example(rag, question, answer, mode, top_k)
        out[rid] = {"score": int(score), "generated_answer": generated_answer,
                    "question": question, "ground_truth": answer, "n_sentences": n_sents}
    return out


def _save(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {path}")


async def main(args):
    rag = await initialize_lightrag(working_dir=WORKING_DIRS[args.dataset])
    data = load_qa(QA_CSV_PATHS[args.dataset])
    if args.num_rows is not None:
        data = data.head(args.num_rows)

    out_dir = args.out_dir or "benchmark/results"
    rag_path = os.path.join(out_dir, f"{args.dataset}_sentence_{args.top_k}.json")
    rag_results = await run_pass(rag, data, args.rag_mode, args.top_k, "RAG (sentence)")
    _save(rag_results, rag_path)
    acc = sum(r["score"] for r in rag_results.values()) / max(1, len(rag_results))
    print(f"RAG (sentence) accuracy: {acc:.2%}")

    if args.build_comparison:
        if args.llm_results:                       # reuse an existing LLM-only run (no extra pass)
            llm_path = args.llm_results
            print(f"Using existing LLM-only results: {llm_path}")
        else:                                       # otherwise run the bypass (LLM-only) pass
            llm_path = os.path.join(out_dir, f"{args.dataset}_sentence_bypass.json")
            llm_results = await run_pass(rag, data, "bypass", 0, "LLM-only (bypass)")
            _save(llm_results, llm_path)
        cmp_out = args.comparison_out or os.path.join(
            out_dir, f"comparison_{args.dataset}_sentence_{args.top_k}.json")
        # FF/FT/TF/TT classification reused verbatim from benchmark/evaluation.py.
        export_performance_cases(
            rag_results_path=rag_path, dataset=args.dataset, top_k=args.top_k,
            llm_results_path=llm_path, output_path=cmp_out, rag_only=args.rag_only)
        print(f"\nSentence-context comparison for RAGE -> {cmp_out}")
        print(f"Drive RAGE with:  --comparison {cmp_out}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_benchmark",
        description="Sentence-context RAG benchmark (fair analog of benchmark/run.py for "
                    "RAGE): scores RAG over the SAME sentence context run_rage.py explains, "
                    "and (with --build-comparison) emits comparison_<ds>_sentence_<top_k>.json.")
    p.add_argument("--dataset", choices=DATASETS, default="hotpotqa")
    p.add_argument("--rag-mode", choices=["hybrid", "local", "global", "naive", "bypass"], default="hybrid")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--num-rows", type=int, default=None, help="Cap on QA rows (default: all).")
    p.add_argument("--build-comparison", action="store_true",
                   help="Emit the FF/FT/TF/TT comparison JSON. Runs an LLM-only (bypass) pass unless "
                        "--llm-results is supplied.")
    p.add_argument("--llm-results", default=None,
                   help="[--build-comparison] path to an existing LLM-only results JSON "
                        "(run.py-schema: id -> {score, generated_answer, question, ground_truth}); "
                        "reused as-is so no LLM-only pass is run.")
    p.add_argument("--rag-only", action="store_true",
                   help="[--build-comparison] classify by RAG score only (ft=rag correct, tf=rag wrong); "
                        "skips the LLM-only join (no ff cases for the F->T direction).")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default benchmark/results).")
    p.add_argument("--comparison-out", default=None,
                   help="[--build-comparison] explicit comparison JSON path.")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    asyncio.run(main(args))
