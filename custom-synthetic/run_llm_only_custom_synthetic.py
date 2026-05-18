import asyncio
import json
import pandas as pd
from tqdm import tqdm
from typing import Literal
from pydantic import BaseModel, Field
from ollama import chat

JSON_FILE = "./kg_dataset.json"
OUTPUT_CSV = "llm_only_eval_results.csv"


class JudgeScore(BaseModel):
    reasoning: str = Field(description="short reasoning")
    score: Literal["1", "5"]


def load_dataset():
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


async def query_llm(question):

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

    response = chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    return response["message"]["content"].strip()


def judge_response(question, pred, gt):

    prompt = f"""
            You are a strict evaluator.

            QUESTION:
            {question}

            GROUND TRUTH:
            {gt}

            SYSTEM ANSWER:
            {pred}

            Return JSON:
            {{
            "reasoning": "...",
            "score": "5"
            }}

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
        return "1"


async def run_benchmark(df):

    results = []

    print(f"Samples: {len(df)}")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        question = row["question"]
        gt = row["ground_truth"]

        pred = await query_llm(question)

        score = judge_response(question, pred, gt)

        results.append({
            "question": question,
            "ground_truth": gt,
            "prediction": pred,
            "score": 1 if score == "5" else 0
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print("\n========================")
    print("Accuracy:", df_out["score"].mean())
    print("Saved:", OUTPUT_CSV)
    print("========================")

async def main():
    df = load_dataset()
    await run_benchmark(df)

if __name__ == "__main__":
    asyncio.run(main())