from typing import Dict, Any
import json
from ollama import chat
from LLM.prompts import LLM_AS_A_JUDGE_PROMPT, QA_PROMPT
import os
import csv
import pandas as pd

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from retrieval.base import *
from retrieval.retrieve import (
    initialize_lightrag, 
    retrieve_subgraph, 
    print_subgraph,
    WORKING_DIR,
    QUERY,
    MODE,
    TOP_K
)
from retrieval.parser import parse_context
from sentence_transformers import SentenceTransformer

import asyncio

WORKING_DIR  = "/mnt/qnap/cs05058/LightRAG/xylotian_storage"
QUERY        = "What are the two primary materials used to construct a Xylotian 'Sky-Skiff' hull?"
MODE         = "hybrid"
TOP_K        = 2

OLLAMA_HOST  = "http://localhost:11434"
LLM_MODEL    = "mistral-small3.2:24b-instruct-2506-q4_K_M"
model = SentenceTransformer('all-MiniLM-L6-v2')


class LLMWrapper:
    def __init__(self, model: str = "mistral-small3.2:24b-instruct-2506-q4_K_M"):
        self.model = model
        self.COUNTER_FILE = "counter.txt"

    def _call(self, query: Dict[str, Any], prompt: str = None):

        if prompt == "LLM_AS_A_JUDGE_PROMPT":
            prompt_text = LLM_AS_A_JUDGE_PROMPT.format(**query)

        elif prompt == "QA_PROMPT":
            prompt_text = QA_PROMPT.format(**query)

        else:
            prompt_text = str(query)

        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt_text}],
            options={"temperature": 0}
        )

        content = response["message"]["content"]

        try:
            return json.loads(content)
        except Exception:
            return content

    def load_dataset(self, filename):
        file_path = os.path.join("..", "data", filename)
        return pd.read_csv(file_path)


    def _extract_score(self, judge_result):
        if isinstance(judge_result, dict):
            return judge_result.get("score", None)
        return judge_result


    def evaluate_file(self, filename, mode="llm_only"):

        df = self.load_dataset(filename)

        results_file_id = self.get_next_run_id()
        results_dir = os.path.join("..", "results")
        os.makedirs(results_dir, exist_ok=True)

        mode_dir = os.path.join(results_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        base_name = os.path.splitext(filename)[0]
        file_path = os.path.join(mode_dir, f"{base_name}.csv")
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "question",
                "context",
                "ground_truth",
                "predicted_answer",
                "judge_score"
            ])

            for _, row in df.iterrows():

                question = row["question"]
                if mode == "llm_only":
                    context = ""
                else:
                    context = row.get("context", "")
                
                ground_truth = row["answer"]

                predicted_answer = self._call(
                    {
                        "question": question,
                        "context": context
                    },
                    prompt="QA_PROMPT"
                )

                judge_result = self._call(
                    {
                        "question": question,
                        "context": context,
                        "system_generated_answer": predicted_answer,
                        "ground_truth_answer": ground_truth
                    },
                    prompt="LLM_AS_A_JUDGE_PROMPT"
                )

                score = self._extract_score(judge_result)

                writer.writerow([
                    question,
                    context,
                    ground_truth,
                    predicted_answer,
                    score
                ])

        return file_path

    def perturbe_context(self, context: str):

        if not context or not isinstance(context, str):
            return []

        sentences = [s.strip() for s in context.split(".") if s.strip()]

        perturbed_contexts = []

        for i in range(len(sentences)):
            perturbed = sentences[:i] + sentences[i+1:]
            new_context = ". ".join(perturbed)

            if new_context:
                new_context += "."

            perturbed_contexts.append(new_context)

        return perturbed_contexts
    
    def compare_answers(self, base_filename):

        llm_dir = os.path.join("..", "results", "llm_only")
        rag_dir = os.path.join("..", "results", "rag")

        base_name = os.path.splitext(base_filename)[0]

        llm_file = None
        rag_file = None

        for f in os.listdir(llm_dir):
            if f.startswith(base_name):
                llm_file = os.path.join(llm_dir, f)
                break

        for f in os.listdir(rag_dir):
            if f.startswith(base_name):
                rag_file = os.path.join(rag_dir, f)
                break

        if llm_file is None or rag_file is None:
            raise FileNotFoundError(
                f"Could not find matching files for base: {base_name}"
            )

        llm_df = pd.read_csv(llm_file)
        rag_df = pd.read_csv(rag_file)

        merged = pd.merge(
            llm_df,
            rag_df,
            on="question",
            suffixes=("_llm", "_rag")
        )

        both_correct = 0
        both_wrong = 0
        improvement = 0
        worsening = 0

        llm_correct = 0
        rag_correct = 0

        total = len(merged)

        for _, row in merged.iterrows():

            llm_score = int(row["judge_score_llm"])
            rag_score = int(row["judge_score_rag"])

            llm_correct += llm_score
            rag_correct += rag_score

            if llm_score == 1 and rag_score == 1:
                both_correct += 1

            elif llm_score == 0 and rag_score == 0:
                both_wrong += 1

            elif llm_score == 0 and rag_score == 1:
                improvement += 1

            elif llm_score == 1 and rag_score == 0:
                worsening += 1

        results = {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "improvement": improvement,
            "worsening": worsening,
            "llm_accuracy": llm_correct / total if total else 0,
            "rag_accuracy": rag_correct / total if total else 0
        }

        results_dir = os.path.join("..", "results", "comparisons")
        os.makedirs(results_dir, exist_ok=True)

        # file_id = self.get_next_run_id()
        out_path = os.path.join(results_dir, f"{base_name}_comparison.csv")

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(["metric", "value"])
            for k, v in results.items():
                writer.writerow([k, v])

        return results

    def get_next_run_id(self):
        if not os.path.exists(self.COUNTER_FILE):
            with open(self.COUNTER_FILE, "w") as f:
                f.write("0")

        with open(self.COUNTER_FILE, "r") as f:
            raw = f.read().strip()
            run_id = int(raw or "0")

        next_id = run_id + 1

        with open(self.COUNTER_FILE, "w") as f:
            f.write(str(next_id))

        return next_id


### test for evaluation and comparison

# if __name__ == "__main__":
#     lw = LLMWrapper()
#     lw.evaluate_file("20_duplicate.csv", "llm_only")
#     lw.evaluate_file("20_duplicate.csv", "rag")
#     lw.compare_answers("20_duplicate")



### test retrieval and perturbation
async def main():
    rag = await initialize_lightrag(WORKING_DIR)
    lw = LLMWrapper()
    
    context_str = await retrieve_subgraph(rag=rag, query=QUERY, mode=MODE, top_k=TOP_K)
    
    if context_str:
        parsed_subgraph = parse_context(context_str)
        # print_subgraph(parsed_subgraph)

        print(f"\n── Source Chunks {'─'*43}")    
        perturbed = lw.perturbe_context(parsed_subgraph.chunks[0] if parsed_subgraph.chunks else "")
        print("\nPERTURBED CONTEXTS:\n")
        for i, p in enumerate(perturbed):
            print(f"\n--- Version {i+1} ---")
            print(p)
    else:
        print("No context retrieved")

if __name__ == "__main__":
    asyncio.run(main())

