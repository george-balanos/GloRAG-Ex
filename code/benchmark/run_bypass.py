"""LLM-only baseline ("bypass") — answers each QA question without retrieving any context.

Produces benchmark/results/<dataset>_<bypass-suffix>.json in the same schema as
benchmark/run.py, so it can be paired with the RAG results by benchmark/evaluation.py
to classify each question into tt / tf / ft / ff.

Optional. The default ablation pipeline now uses `evaluation.py --rag-only`, which
classifies questions by RAG correctness alone (rag-correct -> ft, rag-wrong -> tf)
and does not need this script. Use run_bypass.py only when you want the full
4-bucket (LLM vs RAG) analysis.
"""
import argparse
import asyncio
import json
import logging
import os

import pandas as pd
from tqdm import tqdm

from src.llm.utils import vllm_model_complete
from src.llm_judge import judge_response
from src.dataset_setup import DATASETS, QA_CSV_PATHS

logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are answering a short factoid question. "
    "Return only the final answer — no explanation, no preamble, no full sentence."
)


async def run_bypass(qa_csv: str, output_path: str, num_rows):
    df = pd.read_csv(qa_csv).drop_duplicates(subset=["questions"]).reset_index(drop=True)
    if num_rows is not None:
        df = df.head(num_rows)

    results = {}
    for _, row in tqdm(df.iterrows(), desc="LLM-only baseline...", total=len(df)):
        question = row["questions"]
        ground_truth = row["answers"]

        prediction = await vllm_model_complete(prompt=question, system_prompt=SYSTEM_PROMPT)
        score = await judge_response(question, generated_answer=prediction, ground_truth=ground_truth)

        print(f"Score: {score} | Pred: {prediction} | GT: {ground_truth}")

        results[row["id"]] = {
            "score": score,
            "generated_answer": prediction,
            "question": question,
            "ground_truth": ground_truth,
        }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_bypass",
        description="LLM-only baseline (no retrieval) over the QA dataset.",
    )
    p.add_argument("--dataset", choices=DATASETS, default="synthetic")
    p.add_argument("--bypass-suffix", default="bypass_0",
                   help="Suffix used in the output filename: <dataset>_<suffix>.json. "
                        "Match this with benchmark/evaluation.py's --bypass-suffix.")
    p.add_argument("--num-rows", type=int, default=None,
                   help="Cap on QA rows (default: all).")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: benchmark/results/<dataset>_<bypass-suffix>.json).")
    return p


async def main(args: argparse.Namespace):
    output_path = args.output or f"benchmark/results/{args.dataset}_{args.bypass_suffix}.json"
    await run_bypass(QA_CSV_PATHS[args.dataset], output_path, args.num_rows)


if __name__ == "__main__":
    asyncio.run(main(build_arg_parser().parse_args()))
