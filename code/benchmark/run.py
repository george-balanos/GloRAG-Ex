from src.retrieve import *
from src.query import *
from src.llm_judge import *
from src.dataset_setup import WORKING_DIRS, QA_CSV_PATHS, DATASETS
from tqdm import tqdm

import argparse
import pandas as pd
import asyncio
import json
import logging
import os

logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("nano-vectordb").setLevel(logging.WARNING)
logging.getLogger("lightrag").setLevel(logging.WARNING)


def load_qa(path: str):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["questions"])
    return df.reset_index(drop=True)


async def run_example(rag, question, ground_truth, mode, top_k):
    context_graph = await retrieve_subgraph(rag, query=question, mode=mode, top_k=top_k)
    generated_answer = await query(rag, context_graph, question)
    score = await judge_response(question, generated_answer=generated_answer, ground_truth=ground_truth)
    return score, generated_answer


async def run_benchmark(rag, qa_csv: str, output_path: str, mode: str, top_k: int, num_rows):
    benchmark_data = load_qa(qa_csv)

    if num_rows is not None:
        benchmark_data = benchmark_data.head(num_rows)

    result_dict = {}

    for _, row in tqdm(benchmark_data.iterrows(), desc="Processing questions...", total=len(benchmark_data)):
        id = row["id"]
        question = row["questions"]
        answer = row["answers"]

        score, generated_answer = await run_example(rag, question, answer, mode, top_k)

        print(f"Score: {score}\nGenerated Answer: {generated_answer} VS Ground Truth: {answer}")

        result_dict[id] = {
            "score": score,
            "generated_answer": generated_answer,
            "question": question,
            "ground_truth": answer,
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
    print(f"Results saved to: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run",
        description="RAG-only benchmark over a LightRAG-backed KG.",
    )
    p.add_argument("--dataset", choices=DATASETS, default="synthetic",
                   help="Dataset name; selects working_dir, qa CSV, and output filename.")
    p.add_argument("--rag-mode", choices=["hybrid", "local", "global", "naive"], default="hybrid",
                   help="LightRAG retrieval mode.")
    p.add_argument("--top-k", type=int, default=2, help="LightRAG retrieval top_k.")
    p.add_argument("--num-rows", type=int, default=None,
                   help="Cap on QA rows (default: all).")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: benchmark/results/<dataset>_<rag-mode>_<top-k>.json).")
    return p


async def main(args: argparse.Namespace):
    rag = await initialize_lightrag(working_dir=WORKING_DIRS[args.dataset])

    output_path = args.output or f"benchmark/results/{args.dataset}_{args.rag_mode}_{args.top_k}.json"

    await run_benchmark(
        rag=rag,
        qa_csv=QA_CSV_PATHS[args.dataset],
        output_path=output_path,
        mode=args.rag_mode,
        top_k=args.top_k,
        num_rows=args.num_rows,
    )


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    asyncio.run(main(args))
