import os
import json
import time
import csv
import re
import sys
import random
import numpy as np
import pandas as pd
import asyncio
import torch
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from LLM.prompts import LLM_AS_A_JUDGE_PROMPT, QA_PROMPT

from lightrag import QueryParam
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


from accelerate import Accelerator
from transformers import AutoModelForCausalLM

from retrieval.retrieve import initialize_lightrag
from retrieval.parser import parse_context



class KGCasePerturbationEvaluator:

    def __init__(self, model="Qwen/Qwen2.5-3B-Instruct", working_dir=None):
        self.model_name = model
        self.working_dir = working_dir
        self.rag = None
        self.vllm_model = None
        self.vllm_tokenizer = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2")


    async def setup(self):
        self.rag = await initialize_lightrag(self.working_dir)
        
        print(f"\n[vLLM Initialization] Initializing local vLLM engine: {self.model_name}...")
        self.vllm_model = LLM(
            model=self.model_name, 
            gpu_memory_utilization=0.8
        )
        self.vllm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self

    def _get_similarity(self, text1, text2):
        t1 = str(text1)
        t2 = str(text2)
        
        emb1 = self.model.encode([t1])
        emb2 = self.model.encode([t2])
        
        return float(cosine_similarity(emb1, emb2))

    
    def _extract_score(self, judge_output: str) -> int:
        if judge_output is None:
            return 0

        match = re.search(r"\b([01])\b", str(judge_output))
        if match:
            return int(match.group(1))

        # fallback: treat uncertain as wrong
        return 0


    def _call(self, query: dict | str, prompt: str = None):


        if isinstance(query, str):
            prompt_text = query

        elif prompt == "LLM_AS_A_JUDGE_PROMPT":
            prompt_text = LLM_AS_A_JUDGE_PROMPT.format(**query)

        elif prompt == "QA_PROMPT":
            prompt_text = QA_PROMPT.format(**query)

        else:
            # fallback: safe serialization
            prompt_text = str(query)

        sampling_params = SamplingParams(
            max_tokens=512,
            temperature=0.0
        )

        outputs = self.vllm_model.generate(prompt_text, sampling_params)
        return outputs[0].outputs[0].text.strip()


    async def retrieve_context(self, question, top_k=5, mode="hybrid"):
        try:
            param = QueryParam(
                mode=mode, top_k=top_k, only_need_context=True, enable_rerank=False
            )

            context_str = await self.rag.aquery(question, param=param)
            if not context_str:
                return []

            parsed_subgraph = parse_context(context_str)
            if not parsed_subgraph or not hasattr(parsed_subgraph, 'chunks') or not parsed_subgraph.chunks:
                return []
            
            cleaned_chunks = [chunk.strip() for chunk in parsed_subgraph.chunks if chunk.strip()]
            return cleaned_chunks

        except Exception as e:
            print(f"RAG ERROR: {e}")
            return []


    def perturb(self, chunks, method="remove_sentence"):
        if not chunks:
            return []

        out = []
        full_text = "\n\n".join(chunks)

        if method == "remove_sentence":
            sents = [s.strip() for s in full_text.split(".") if s.strip()]
            for i in range(len(sents)):
                new_ctx_list = sents[:i] + sents[i+1:]
                removed = sents[i]
                new_ctx = ". ".join(new_ctx_list) + "." if new_ctx_list else ""
                out.append((new_ctx, removed + "."))

        elif method == "remove_word":
            words = full_text.split()
            for i in range(len(words)):
                new_ctx = words[:i] + words[i+1:]
                removed = words[i]
                out.append((" ".join(new_ctx), removed))
        
        elif method == "remove_paragraph":
            paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]

            if len(paragraphs) < 2:
                return [(full_text, "none")]

            for i in range(len(paragraphs)):
                new_ctx_list = paragraphs[:i] + paragraphs[i+1:]
                removed = paragraphs[i]

                new_ctx = "\n\n".join(new_ctx_list)
                out.append((new_ctx, removed))

        elif method == "rage":
            paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            if len(paragraphs) < 2:
                return [(full_text, "none")]

            for i in range(len(paragraphs)):
                for j in range(i + 1, len(paragraphs)):
                    swapped = paragraphs[:]
                    swapped[i], swapped[j] = swapped[j], swapped[i]
                    new_ctx = "\n\n".join(swapped)
                    out.append((new_ctx, f"swapped_{i}_with_{j}"))

        return out


    def _get_utility(self, subset_items, query, model, tokenizer, device):
        context_str = " ".join(subset_items)
        prompt = f"Context: {context_str}\n\nQuestion: {query}\n\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            
        return -loss


    def load(self, path):
        with open(path, "r") as f:
            return json.load(f)


    async def evaluate(
        self,
        json_file,
        out_dir="results_kg",
        top_k=5,
        dataset="synthetic",
        method="remove_sentence",
    ):

        data = self.load(json_file)
        results = data["results"]
        ids = list(results.keys())

        os.makedirs(out_dir, exist_ok=True)

        json_results = []
        total_llm_calls = 0
        start_time = time.time()

        disagreement_results = {"T->F": 0, "F->T": 0}
        total_tf_time = 0
        total_ft_time = 0
        ft_cases = 0
        tf_cases = 0
        ft_calls = 0
        tf_calls = 0

        total_perturb_time = 0

        print("\n Checking for RAG vs LLM disagreement cases...")

        for i, qid in enumerate(ids):

            item = results[qid]
            case_type = item.get("case", "unknown").lower()

            if case_type not in ["ft", "ff"]:
                continue

            question = item["question"]
            ground_truth = item.get("ground_truth", "")

            display_label = "F->T" if case_type == "ff" else "T->F"
            disagreement_results[display_label] += 1

            chunks = await self.retrieve_context(question, top_k=top_k)
            if not chunks:
                continue

            context = "\n\n".join(chunks)

            original_answer = self._call(
                {"question": question, "context": context},
                prompt="QA_PROMPT"
            )

            perturbations = self.perturb(chunks, method=method)

            case_json = {
                "case_id": qid,
                "question": question,
                "case_type": case_type,
                "mapped_label": display_label,
                "method": method,
                "ground_truth": ground_truth,
                "original_answer": original_answer,
                "original_context": context,
                "analysis": [],
                "llm_calls": 1,
                "timestamp": pd.Timestamp.now().isoformat()
            }

            row_start = time.time()

            records = []




            for idx, (new_ctx, removed) in enumerate(perturbations):

                new_answer = self._call(
                    f"Context: {new_ctx}\n\nQuestion: {question}\n\nAnswer:"
                )

                print(60*"-")
                print(new_answer)
                print(60*"-")

                judge_result = self._call(
                    {
                        "question": question,
                        "system_generated_answer": new_answer,
                        "ground_truth_answer": ground_truth
                    },
                    prompt="LLM_AS_A_JUDGE_PROMPT"
                )

                total_llm_calls += 2

                similarity = self._get_similarity(original_answer, new_answer)
                importance_weight = 1.0 - similarity

                is_correct = self._extract_score(judge_result)
                is_flip = (is_correct == 0)

                record = {
                    "removed_item": removed,
                    "new_answer": new_answer,
                    "judge_result": judge_result,
                    "is_flip": is_flip,
                    "similarity": similarity,
                    "importance_weight": importance_weight
                }

                records.append(record)
                case_json["analysis"].append(record)
            
            case_json["removed_item_importance"] = {
                r["removed_item"]: r["importance_weight"]
                for r in records
            }


            row_elapsed = time.time() - row_start
            total_perturb_time += row_elapsed

            if case_type == "ff":
                total_ft_time += row_elapsed
                ft_cases += 1
            else:
                total_tf_time += row_elapsed
                tf_cases += 1

            case_json["llm_calls"] = 1 + 2 * len(perturbations)
            if case_type == "ff":
                ft_calls += case_json["llm_calls"]
            else:
                tf_calls += case_json["llm_calls"]
            json_results.append(case_json)


        json_path = os.path.join(
            out_dir,
            f"{dataset}_{method}_analysis.json"
        )


        # ----------------------------
        # SUMMARY
        # ----------------------------
        summary_path = os.path.join(out_dir, f"{dataset}_{method}_summary.csv")

        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in disagreement_results.items():
                writer.writerow([k, v])

        elapsed = time.time() - start_time

        print("\n DONE")
        print("JSON:", json_path)
        print("Summary:", summary_path)
        print("Time:", elapsed)

        summary = {
            "total_llm_calls": total_llm_calls,
            "total_ft_time": total_ft_time,
            "total_tf_time": total_tf_time,

            "avg_ft_time_per_case": total_ft_time / ft_cases if ft_cases > 0 else 0,
            "avg_tf_time_per_case": total_tf_time / tf_cases if tf_cases > 0 else 0,


            "avg_ft_calls_per_case": ft_calls / ft_cases if ft_cases > 0 else 0,
            "avg_tf_calls_per_case": tf_calls / tf_cases if tf_cases > 0 else 0,

            "ft_cases": ft_cases,
            "tf_cases": tf_cases
        }

        output = {
            "cases": json_results,
            "summary": summary
        }

        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)

        return disagreement_results



if __name__ == "__main__":

    evaluator = KGCasePerturbationEvaluator(
        model="Qwen/Qwen2.5-3B-Instruct",
        working_dir="/home/vchasanis/Documents/GitHub/LightRAG/GloRAG-Ex/xylotian_storage"
    )

    async def run_pipeline():
        await evaluator.setup()

        ## run for synthetic evaluation
        json_file = "/home/vchasanis/Documents/GitHub/LightRAG/GloRAG-Ex/competitors/RAGEX-RAGE-SHAPLEY/special_cases/comparison_synthetic_2.json"

        # await evaluator.evaluate(
        #     json_file,
        #     top_k=2,
        #     dataset="synthetic",
        #     method="remove_word"         
        # )
        # await evaluator.evaluate(
        #     json_file,
        #     top_k=2,
        #     dataset="synthetic",
        #     method="remove_sentence"  
        # )
        await evaluator.evaluate(
            json_file,
            top_k=2,
            dataset="synthetic",
            method="remove_paragraph"  
        )

        # evaluator2 = KGCasePerturbationEvaluator(
        #     model="Qwen/Qwen2.5-3B-Instruct",
        #     working_dir="/home/vchasanis/Documents/GitHub/LightRAG/GloRAG-Ex/competitors/RAGEX-RAGE-SHAPLEY/hotpotqa"
        # )

        # ## run for hotpotqa evaluation
        # json_file = "/home/vchasanis/Documents/GitHub/LightRAG/GloRAG-Ex/competitors/RAGEX-RAGE-SHAPLEY/special_cases/comparison_hotpotqa_2.json"
        
        # await evaluator2.evaluate(
        #     json_file,
        #     top_k=2,
        #     dataset="hotpotqa",
        #     method="remove_word"         
        # )
        # await evaluator2.evaluate(
        #     json_file,
        #     top_k=2,
        #     dataset="hotpotqa",
        #     method="remove_sentence"  
        # )
        # await evaluator2.evaluate(
        #     json_file,
        #     top_k=2,
        #     dataset="hotpotqa",
        #     method="remove_paragraph"  
        # )

    asyncio.run(run_pipeline())