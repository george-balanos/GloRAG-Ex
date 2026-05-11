from src.retrieve import *
from src.query import *
from src.llm_judge import *
from tqdm import tqdm

import argparse
import os
import pandas as pd
import asyncio
import json
import logging

logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("nano-vectordb").setLevel(logging.WARNING)
logging.getLogger("lightrag").setLevel(logging.WARNING)


def load_qa(path: str):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["questions"])
    return df.reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser(description="Run the RAG benchmark over a QA CSV.")
    p.add_argument("--mode", default="local", help="LightRAG retrieval mode (local/global/hybrid)")
    p.add_argument("--top-k", type=int, default=10, help="Retrieval top-k")
    p.add_argument("--qa", default="qa/qa_data_synthetic.csv", help="Path to QA CSV")
    p.add_argument("--out", default=None,
                   help="Output JSON path. Default: benchmark/results/synthetic_{mode}_{top_k}.json")
    return p.parse_args()

async def run_example(rag, question, ground_truth, mode, top_k):
    # Retrieve Context
    context_graph = await retrieve_subgraph(rag, query=question, mode=mode, top_k=top_k)

    # Query LLM
    generated_answer = await query(rag, context_graph, question)

    # Compare to Ground Truth
    score = await judge_response(question, generated_answer=generated_answer, ground_truth=ground_truth)

    return score, generated_answer

async def run_benchmark(rag, qa_path="qa/qa_data_synthetic.csv", mode="local", top_k=10, out_path=None):
    benchmark_data = load_qa(qa_path)

    result_dict = {}

    for i, row in tqdm(benchmark_data.iterrows(), desc="Processing questions...", total=len(benchmark_data)):
        id = row["id"]
        question = row["questions"]
        answer = row["answers"]

        score, generated_answer = await run_example(rag, question, answer, mode, top_k)

        print(f"Score: {score}\nGenerated Answer: {generated_answer} VS Ground Truth: {answer}")

        result_dict[id] = {
            "score": score,
            "generated_answer": generated_answer,
            "question": question,
            "ground_truth": answer
        }

    if out_path is None:
        out_path = f"benchmark/results/synthetic_{mode}_{top_k}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
    print(f"Wrote {len(result_dict)} entries to {out_path}")


async def main():
    args = parse_args()
    rag = await initialize_lightrag()
    await run_benchmark(rag, qa_path=args.qa, mode=args.mode, top_k=args.top_k, out_path=args.out)


if __name__ == "__main__":
    asyncio.run(main())