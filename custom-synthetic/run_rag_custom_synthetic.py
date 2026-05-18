import asyncio
import os
import json
import pandas as pd
from tqdm import tqdm
from typing import Literal
from pydantic import BaseModel, Field
from ollama import chat
from sentence_transformers import SentenceTransformer

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import EmbeddingFunc



os.environ["LLM_TIMEOUT"] = "1800"
os.environ["WORKER_TIMEOUT"] = "1800"
os.environ["EMBEDDING_TIMEOUT"] = "600"

WORKING_DIR = "./kg_dataset_storage"
JSON_FILE = "./kg_dataset.json"

LLM_MODEL = "mistral-small3.2:24b-instruct-2506-q4_K_M"

OUTPUT_CSV = "kg_rag_eval_results.csv"


class JudgeScore(BaseModel):
    reasoning: str = Field(description="short reasoning")
    score: Literal["1", "5"]



model = SentenceTransformer("all-MiniLM-L6-v2")

async def hf_embed(texts):
    return model.encode(texts, show_progress_bar=False)


async def initialize_rag():

    rag = LightRAG(
        working_dir=WORKING_DIR,

        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,

        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=8192,
            func=hf_embed,
        ),
    )

    await rag.initialize_storages()
    return rag


def judge_response(question, pred, gt):

    prompt = f"""
            You are a strict evaluator.

            Decide if the SYSTEM ANSWER matches the GROUND TRUTH.

            Ignore wording differences.
            Focus only on factual correctness.

            QUESTION:
            {question}

            GROUND TRUTH:
            {gt}

            SYSTEM ANSWER:
            {pred}

            Return ONLY JSON:
            {{
            "reasoning": "...",
            "score": "5"
            }}

            Use:
            5 = correct
            1 = incorrect
            """

    response = chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
        format=JudgeScore.model_json_schema(),
    )

    try:
        content = response["message"]["content"]
        parsed = json.loads(content)
        return parsed.get("score", "1")

    except Exception:
        print("JUDGE PARSE ERROR:", e)
        print("RAW RESPONSE:", response)
        return "1"


async def query_rag(rag, question, mode="hybrid"):

    prompt = f"""
            You are answering a factoid question.

            Return ONLY the final short answer.
            Do NOT explain.
            Do NOT use full sentences.
            Do NOT add extra words.

            Question:
            {question}

            Answer:
            """

    response = await rag.aquery(
        prompt,
        param=QueryParam(
            mode=mode,
            stream=False,
            top_k=2,
            enable_rerank=False,
        )
    )

    return str(response).strip()

def load_dataset():

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return pd.DataFrame(data)


async def run_benchmark(rag, df, mode):

    results = []

    print(f"\nRunning mode: {mode}")
    print(f"Samples: {len(df)}\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        question = row["question"]
        gt = row["ground_truth"]

        pred = await query_rag(rag, question, mode=mode)

        score = judge_response(question, pred, gt)

        results.append({
            "question": question,
            "ground_truth": gt,
            "prediction": pred,
            "score": 1 if score == "5" else 0
        })

    df_out = pd.DataFrame(results)

    file = f"{OUTPUT_CSV.replace('.csv','')}_{mode}.csv"

    df_out.to_csv(file, index=False)

    print("\n========================")
    print("MODE:", mode)
    print("Accuracy:", df_out["score"].mean())
    print("Saved:", file)
    print("========================\n")


async def main():

    df = load_dataset()

    rag = await initialize_rag()

    try:

        for mode in ["local"]:

            await run_benchmark(rag, df, mode)

    finally:

        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())