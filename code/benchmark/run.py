from src.retrieve import *
from src.query import *
from src.llm_judge import *
from tqdm import tqdm

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

async def run_example(rag, question, ground_truth, mode, top_k):
    # Retrieve Context
    context_graph = await retrieve_subgraph(rag, query=question, mode=mode, top_k=top_k)

    # Query LLM
    generated_answer = await query(rag, context_graph, question)

    # Compare to Ground Truth
    score = await judge_response(question, generated_answer=generated_answer, ground_truth=ground_truth)

    return score, generated_answer

async def run_benchmark(rag, mode="local", top_k=10, num_rows=100, dataset="synthetic"):
    benchmark_data = load_qa(f"datasets/{dataset}/qa_data_{dataset}.csv")

    if num_rows is not None:
        benchmark_data = benchmark_data.head(num_rows)

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

    with open(f"benchmark/results/{dataset}_{mode}_{top_k}.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

async def main():
    mode = "hybrid"
    top_k = 2

    rag = await initialize_lightrag(working_dir=WORKING_DIR_HOTPOTQA)

    await run_benchmark(rag, mode, top_k, dataset="hotpotqa")

if __name__ == "__main__":

    asyncio.run(main())