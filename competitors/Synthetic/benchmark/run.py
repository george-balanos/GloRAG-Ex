from src.retrieve import *
from src.query import *
from src.llm_judge import *
from tqdm import tqdm

import pandas as pd
import asyncio
import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_qa(path: str):
    return pd.read_csv(path)

async def run_example(rag, question, ground_truth, mode, top_k):
    # Retrieve Context
    context_graph = await retrieve_subgraph(rag, question, mode=mode, top_k=top_k)

    # Query LLM
    generated_answer = await query(rag, context_graph)

    # Compare to Ground Truth
    score = judge_response(question, generated_answer=generated_answer, ground_truth=ground_truth)

    return score, generated_answer

async def run_benchmark(rag, mode="hybrid", top_k=2):
    benchmark_data = load_qa("qa/qa_data_synthetic.csv")

    result_dict = {}

    for i, row in tqdm(benchmark_data.iterrows(), desc="Processing questions...", total=len(benchmark_data)):
        id = row["id"]
        question = row["questions"]
        answer = row["answers"]

        # print(f"Row with ID {id}:\nQuestion: {question}\nAnswer: {answer}")

        score, generated_answer = await run_example(rag, question, answer, mode, top_k)

        print(f"Score: {score}\nGenerated Answer: {generated_answer} VS Ground Truth: {answer}")
        
        result_dict[id] = {
            "score": score,
            "generated_answer": generated_answer,
            "question": question,
            "ground_truth": answer
        }

    with open(f"benchmark/results/synthetic_{mode}_{top_k}.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

async def main():
    rag = await initialize_lightrag()

    await run_benchmark(rag)

if __name__ == "__main__":

    asyncio.run(main())