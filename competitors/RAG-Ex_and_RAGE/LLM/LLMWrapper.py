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
from sklearn.metrics.pairwise import cosine_similarity
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
        self.rag = None
    
    async def setup(self):
        """Asynchronously initialize the RAG engine"""
        self.rag = await initialize_lightrag(WORKING_DIR)
        return self

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
        
        stats = {
            "input_tokens": response.get("prompt_eval_count", 0),
            "output_tokens": response.get("eval_count", 0),
            "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
        }

        try:
            parsed_content = json.loads(content)
            return parsed_content, stats
        except Exception:
            return content, stats
    
    def _get_similarity(self, text1, text2):
        from sklearn.metrics.pairwise import cosine_similarity
        t1 = str(text1)
        t2 = str(text2)
        
        emb1 = model.encode([t1])
        emb2 = model.encode([t2])
        
        return float(cosine_similarity(emb1, emb2))

    def load_dataset(self, filename):
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "..", "data", filename)
        return pd.read_csv(file_path)


    def _extract_score(self, judge_result):
        if isinstance(judge_result, dict):
            score = judge_result.get("score", None)
            if score is not None:
                return int(score)
        
        text_score = str(judge_result).strip()
        
        if '1' in text_score:
            return 1
            
        return 0


    def evaluate_file(self, filename, mode="llm_only"):

        df = self.load_dataset(filename)

        results_file_id = self.get_next_run_id()
        results_dir = os.path.join(os.getcwd(), "results")
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

    def perturbe_context(self, context: str, method: str = "remove_word"):
        if not context or not isinstance(context, str):
            return []

        perturbed_data = [] 

        if method == "remove_sentence":
            sentences = [s.strip() for s in context.split(".") if s.strip()]
            for i in range(len(sentences)):
                removed = sentences[i]
                perturbed = sentences[:i] + sentences[i+1:]
                new_context = ". ".join(perturbed)
                if new_context:
                    new_context += "."
                perturbed_data.append((new_context, removed))

        elif method == "remove_word":
            words = context.split() 
            for i in range(len(words)):
                removed = words[i]
                perturbed = words[:i] + words[i+1:]
                new_context = " ".join(perturbed)
                perturbed_data.append((new_context, removed))

        elif method == "rage":
            paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]
            if len(paragraphs) < 2:
                return [(context, "none")] 

            for i in range(len(paragraphs)):
                for j in range(i + 1, len(paragraphs)):
                    swapped = paragraphs[:]
                    swapped[i], swapped[j] = swapped[j], swapped[i]
                    new_context = "\n\n".join(swapped)
                    perturbed_data.append((new_context, f"swapped_{i}_with_{j}"))

        return perturbed_data
    
    def compare_answers(self, base_filename):
        llm_dir = os.path.join("results", "llm_only")
        rag_dir = os.path.join("results", "rag")
        
        if base_filename.endswith(".csv"):
            base_name = base_filename[:-4]
        else:
            base_name = base_filename

        llm_path = os.path.join(llm_dir, f"{base_name}.csv")
        rag_path = os.path.join(rag_dir, f"{base_name}.csv")

        if not os.path.exists(llm_path) or not os.path.exists(rag_path):
            raise FileNotFoundError(f"Looking for:\n{llm_path}\n{rag_path}\nBut they don't exist!")

        llm_df = pd.read_csv(os.path.join(llm_dir, f"{base_name}.csv")).drop_duplicates(subset=["question"])
        rag_df = pd.read_csv(os.path.join(rag_dir, f"{base_name}.csv")).drop_duplicates(subset=["question"])

        merged = pd.merge(llm_df, rag_df, on="question", suffixes=("_llm", "_rag"))

        analysis_rows = []
        results = {"both_correct": 0, "both_wrong": 0, "improvement": 0, "worsening": 0}

        for idx, row in merged.iterrows():
            l_score = int(row["judge_score_llm"])
            r_score = int(row["judge_score_rag"])
            
            if l_score == 1 and r_score == 1: results["both_correct"] += 1
            elif l_score == 0 and r_score == 0: results["both_wrong"] += 1
            elif l_score == 0 and r_score == 1: results["improvement"] += 1
            elif l_score == 1 and r_score == 0: results["worsening"] += 1

            if l_score != r_score:
                context_to_perturb = row.get("context_rag", "")

                if isinstance(context_to_perturb, str):
                    for char in ["['", "']", '["', '"]']:
                        context_to_perturb = context_to_perturb.replace(char, "")
                    context_to_perturb = context_to_perturb.replace("', '", " ").replace('", "', " ")

                perturbations = self.perturbe_context(context_to_perturb, method="rage")

                total_calls = 0
                total_tokens = 0
                temp_perturbation_results = []

                truth_col_name = "ground_truth_rag" 
                actual_ground_truth = row.get(truth_col_name)
                
                for p_text, removed_token in perturbations:
                    new_predicted_answer, qa_stats = self._call(
                        {"question": row["question"], "context": p_text},
                        prompt="QA_PROMPT"
                    )
                    # print("xxxxxxxxxxxxxxx")
                    # print(f"QA Call - Question: {row['question']}, Removed Token: '{removed_token}', New Answer: '{new_predicted_answer}'")
                    # print(p_text)
                    # print(actual_ground_truth)
                    # print("xxxxxxxxxxxxxxx")
                    judge_result, judge_stats = self._call(
                        {
                            "question": row["question"],
                            "context": p_text,
                            "system_generated_answer": new_predicted_answer,
                            "ground_truth_answer": actual_ground_truth
                        },
                        prompt="LLM_AS_A_JUDGE_PROMPT"
                    )
                    print(f"Judge results {judge_result}")
                    new_score = self._extract_score(judge_result)

                    similarity = self._get_similarity(row["predicted_answer_rag"], new_predicted_answer)
                    
                    importance_weight_prime = 1.0 - similarity
                    
                    row_tokens = qa_stats["total_tokens"] + judge_stats["total_tokens"]
                    total_tokens += row_tokens
                    total_calls += 2

                    temp_perturbation_results.append({
                        "original_row_id": idx,
                        "removed_token": removed_token,
                        "new_score": new_score,
                        "importance_weight_prime": importance_weight_prime,
                        "tokens_consumed": row_tokens, 
                        "new_answer": new_predicted_answer,
                        "question": row["question"]
                    })
                
                max_w_prime = max([res["importance_weight_prime"] for res in temp_perturbation_results]) if temp_perturbation_results else 1.0
                
                if temp_perturbation_results:
                    max_w_prime = max([res["importance_weight_prime"] for res in temp_perturbation_results])
                    
                    for res in temp_perturbation_results:
                        res["token_importance_score"] = res["importance_weight_prime"] / max_w_prime if max_w_prime > 0 else 0.0
                        analysis_rows.append(res)

                print(f"Perturbation complete for Row {idx}. Calls: {total_calls}, Total Tokens: {total_tokens}")

        results_dir = os.path.join("results", "comparisons")
        os.makedirs(results_dir, exist_ok=True)
        
        analysis_df = pd.DataFrame(analysis_rows)
        analysis_df.to_csv(os.path.join(results_dir, f"{base_name}_rage_disagreement_analysis.csv"), index=False)

        with open(os.path.join(results_dir, f"{base_name}_rage_comparison.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in results.items(): writer.writerow([k, v])

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


    def collect_perturbations(self, context: str, method: str = "remove_sentence"):
        perturbations = []

        perturbed = self.perturbe_context(context)
        print("\nPERTURBED CONTEXTS:\n")
        for i, p in enumerate(perturbed):
            perturbations.append(p)
            print(f"\n--- Version {i+1} ---")
            print(p)
        
        return perturbations
    
    def extract_impactful_changes(self, base_filename):
        results_dir = os.path.join("results", "comparisons")
        rag_file = os.path.join("results", "rag", f"{base_filename}.csv")
        perturb_file = os.path.join(results_dir, f"{base_filename}_rage_disagreement_analysis.csv")
        output_file = os.path.join(results_dir, f"{base_filename}_rage_impactful_tokens.csv")

        if not os.path.exists(perturb_file) or not os.path.exists(rag_file):
            print(f"Files missing: {perturb_file} or {rag_file}")
            return

        rag_df = pd.read_csv(rag_file)
        perturb_df = pd.read_csv(perturb_file)

        merged = pd.merge(
            perturb_df, 
            rag_df, 
            on="question", 
            suffixes=("_perturbed", "_original")
        )

        impact_rows = []

        for _, row in merged.iterrows():
            original_score = row.get("judge_score_original") or row.get("judge_score")
            
            original_answer = row.get("predicted_answer_original") or row.get("predicted_answer")
            
            new_score = row.get("new_score")
            new_answer = row.get("new_answer")
            removed_token = row.get("removed_token")
            importance_score = row.get("token_importance_score", 0)

            # We save the token if:
            # A) The score dropped (Logic Flip)
            # B) The answer text drifted significantly (Semantic Change)
            score_dropped = (original_score == 1 and new_score == 0)
            answer_changed = (importance_score > 0.3) 

            if score_dropped or answer_changed:
                impact_rows.append({
                    "question": row["question"],
                    "removed_token": removed_token,
                    "token_impact_score": importance_score,
                    "original_score": original_score,
                    "new_score": new_score,
                    "score_flip": "YES" if score_dropped else "NO",
                    "original_rag_answer": original_answer,
                    "perturbed_answer": new_answer
                })

        if impact_rows:
            impact_df = pd.DataFrame(impact_rows)
            impact_df = impact_df.sort_values(by="token_impact_score", ascending=False)
            impact_df.to_csv(output_file, index=False)
            
            print(f"--- Impact Analysis Complete ---")
            print(f"Impactful tokens saved to: {output_file}")
            print(f"Analyzed {len(merged)} total perturbations.")
            print(f"Found {len(impact_df)} tokens that significantly altered the RAG output.")
        else:
            print("No impactful changes were found after perturbation.")
    

### test retrieval and perturbation
# async def main():
#     lw = LLMWrapper()
#     await lw.setup()
    
#     context_str = await retrieve_subgraph(rag=lw.rag, query=QUERY, mode=MODE, top_k=TOP_K)
    
#     if context_str:
#         parsed_subgraph = parse_context(context_str)
#         # print_subgraph(parsed_subgraph)

#         print(f"\n── Source Chunks {'─'*43}")    
#         perturbed = lw.collect_perturbations(parsed_subgraph.chunks[0] if parsed_subgraph.chunks else "", "rage") 
#         print("\nPERTURBED CONTEXTS:\n")
#         for i, p in enumerate(perturbed):
#             print(f"\n--- Version {i+1} ---")
#             print(p)
#     else:
#         print("No context retrieved")

# if __name__ == "__main__":
#     asyncio.run(main())



### test for evaluation and comparison

if __name__ == "__main__":
    lw = LLMWrapper()
#     # lw.evaluate_file("master_synthetic_dataset.csv", "llm_only")
#     # lw.evaluate_file("master_synthetic_dataset.csv", "rag")
    # lw.compare_answers("master_synthetic_dataset")
    lw.extract_impactful_changes("master_synthetic_dataset")
