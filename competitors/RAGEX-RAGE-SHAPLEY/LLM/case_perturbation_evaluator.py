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

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from lightrag import QueryParam


LLMX_DIR = "/home/vchasanis/Documents/GitHub/LightRAG/GloRAG-Ex/competitors/LLMX"
if LLMX_DIR not in sys.path:
    sys.path.insert(0, LLMX_DIR)

from accelerate import Accelerator
from transformers import AutoModelForCausalLM
from SHapRAG.rag_shap import ContextAttribution  

from retrieval.retrieve import initialize_lightrag
from retrieval.parser import parse_context



class KGCasePerturbationEvaluator:

    def __init__(self, model="Qwen/Qwen2.5-3B-Instruct", working_dir=None):
        self.model_name = model
        self.working_dir = working_dir
        self.rag = None
        self.vllm_model = None
        self.vllm_tokenizer = None


    async def setup(self):
        self.rag = await initialize_lightrag(self.working_dir)
        
        print(f"\n[vLLM Initialization] Initializing local vLLM engine: {self.model_name}...")
        self.vllm_model = LLM(
            model=self.model_name, 
            gpu_memory_utilization=0.8
        )
        self.vllm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self


    def _call(self, prompt):
        sampling_params = SamplingParams(
            max_tokens=40,
            temperature=0.0
        )

        outputs = self.vllm_model.generate(prompt, sampling_params)

        return outputs[0].outputs[0].text.strip()


    async def retrieve_context(self, question, top_k=5, mode="local"):
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


    async def evaluate(self, json_file, out_dir="results_kg", top_k=5, dataset="synthetic", method="remove_sentence", num_iterations=40, tolerance=0.002):
        data = self.load(json_file)
        results = data["results"]
        ids = list(results.keys())

        os.makedirs(out_dir, exist_ok=True)

        rows = []
        total_llm_calls = 0
        start_time = time.time()

        disagreement_results = {"T->F": 0, "F->T": 0}
        total_perturb_time = 0
        total_tf_time = 0
        total_ft_time = 0
        
        case_calls = {"ft": 0, "tf": 0, "ff": 0, "tt": 0}
        case_counts = {"ft": 0, "tf": 0, "ff": 0, "tt": 0}

        if method in ["shapley", "tmc"]:
            print(f"\n Initializing GPU-accelerated pipeline allocation via Qwen model for method: {method.upper()}...")
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            accelerator = Accelerator()
            device = accelerator.device

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map={"": device}
            )

        print("\n Checking for RAG vs LLM disagreement cases (FT: F->T / FF: T->F flips)...")

        for qid in ids:
            item = results[qid]
            case_type = item.get("case", "unknown").lower()

            if case_type not in ["ft", "ff"]:
                continue

            question = item["question"]
            ground_truth = item.get("ground_truth", "")

            if case_type == "ff":
                disagreement_results["T->F"] += 1
                display_label = "T->F"
            else:
                disagreement_results["F->T"] += 1
                display_label = "F->T"


            chunks = await self.retrieve_context(question, top_k=top_k)
            if not chunks:
                print(f"Skipping {qid}: no chunks extracted via parser.")
                continue

            context = "\n\n".join(chunks)


            print("\n" + "!" * 60)
            print(f" ANALYZING DISAGREEMENT ON CASE [{qid}]")
            print(f" Question: '{question}'")
            print(f" JSON Case: {case_type.upper()} -> Mapped Label: {display_label}")
            print("!" * 60)

            row_start = time.time()

            if method in ["shapley", "tmc"]:
                raw_sentences = []
                for chunk in chunks:
                    split_sents = [s.strip() + "." for s in chunk.split(".") if s.strip()]
                    raw_sentences.extend(split_sents)

                if not raw_sentences:
                    print(f" Skipping case {qid}: Empty context arrays discovered.")
                    continue

                query_words = set(re.findall(r'\w+', question.lower()))
                stop_words = {'what', 'are', 'the', 'two', 'of', 'in', 'a', 'is', 'does', 'primary', 'components', 'used', 'distinct'}
                keywords = query_words - stop_words

                def calculate_relevance(text):
                    text_lower = text.lower()
                    return sum(1 for word in keywords if word in text_lower)

                sorted_sentences = sorted(raw_sentences, key=calculate_relevance, reverse=True)
                
                list_of_chunks = sorted_sentences[:8]
                n_players = len(list_of_chunks)
                print(f" Context pools totaled {len(raw_sentences)} sentences. Selected top {n_players} elements for evaluation.")

                try:
                    print(f" Initializing Truncated Monte Carlo estimation (Iterations={num_iterations}, Early-Stop Tolerance={tolerance})...")
                    
                    marginal_contributions = np.zeros(n_players)
                    counts = np.zeros(n_players)
                    
                    v_empty = self._get_utility([], question, model, tokenizer, device)
                    v_full = self._get_utility(list_of_chunks, question, model, tokenizer, device)
                    
                    row_calls = 2
                    
                    for idx_iter in range(num_iterations):
                        permutation = list(range(n_players))
                        random.shuffle(permutation)
                        
                        current_subset = []
                        v_old = v_empty
                        
                        for step, player in enumerate(permutation):
                            if abs(v_old - v_full) < tolerance:
                                marginal_contribution = 0.0
                            else:
                                current_subset.append(list_of_chunks[player])
                                v_new = self._get_utility(current_subset, question, model, tokenizer, device)
                                row_calls += 1
                                
                                marginal_contribution = v_new - v_old
                                v_old = v_new
                            
                            marginal_contributions[player] += marginal_contribution
                            counts[player] += 1
                    
                    scores = marginal_contributions / np.where(counts > 0, counts, 1)

                    total_llm_calls += row_calls
                    if case_type in case_calls:
                        case_calls[case_type] += row_calls

                    if case_type in case_counts:
                        case_counts[case_type] += 1

                    print("\n" + "=" * 60)
                    print(f" TMC SHAPLEY VALUES ATTRIBUTION TABLE (CASE {qid})")
                    print("=" * 60)
                    for i, score in enumerate(scores):
                        truncated_text = list_of_chunks[i][:75]
                        print(f"  Player Chunk [{i}]: TMC Score = {score:.4f} | Text: {truncated_text}...")
                        
                        rows.append({
                            "case_id": qid,
                            "question": question,
                            "case_type_json": case_type,
                            "mapped_label": display_label,
                            "player_index": i,
                            "shapley_value": score,
                            "chunk_text": list_of_chunks[i],
                            "method": f"tmc_{method}",
                            "original_context": context
                        })
                    print("=" * 60)

                except Exception as e:
                    print(f" Error processing TMC Shapley engine on case {qid}: {e}")

            else:
                perturbations = self.perturb(chunks, method=method)
                print(f" Generated {len(perturbations)} mutations via '{method}' strategy...")

                if case_type in case_counts:
                    case_counts[case_type] += 1

                for new_ctx, removed in perturbations:
                    _ = self._call(f"Context: {new_ctx}\n\nQuestion: {question}\n\nAnswer:")
                    
                    total_llm_calls += 1
                    if case_type in case_calls:
                        case_calls[case_type] += 1

                    rows.append({
                        "case_id": qid,
                        "question": question,
                        "case_type_json": case_type,
                        "mapped_label": display_label,
                        "removed_token": removed,
                        "method": method,
                        "original_context": context,
                        "perturbed_context": new_ctx
                    })

            row_elapsed = time.time() - row_start
            total_perturb_time += row_elapsed
            if case_type == "ff":
                total_tf_time += row_elapsed
            else:
                total_ft_time += row_elapsed

            print(f"done: {qid}")


        if not rows:
            print("\n No flip or disagreement cases were processed in this execution run.")
            return {}

        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, f"{dataset}_{method}_disagreement_analysis.csv")
        df.to_csv(csv_path, index=False)

        summary_csv_path = os.path.join(out_dir, f"{dataset}_{method}_comparison.csv")
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in disagreement_results.items():
                writer.writerow([k, v])


        elapsed_total = time.time() - start_time
        report_path = os.path.join(out_dir, f"{dataset}_{method}_runtime.txt")
        
        with open(report_path, "w") as f:
            f.write(f"runtime_seconds: {elapsed_total:.2f}\n")
            f.write(f"runtime_human: {elapsed_total // 60:.0f}m {elapsed_total % 60:.0f}s\n")
            f.write(f"total_calls: {total_llm_calls}\n")
            f.write(f"total evaluation pipeline time: {total_perturb_time // 60:.0f}m {total_perturb_time % 60:.0f}s\n")
            f.write(f"total worsening time (T->F): {total_tf_time // 60:.0f}m {total_tf_time % 60:.0f}s\n")
            f.write(f"total improvement time (F->T): {total_ft_time // 60:.0f}m {total_ft_time % 60:.0f}s\n")
            f.write(f"method_used: {method}\n\n")
            
            f.write("=== CALL AVERAGES PER DISAGREEMENT FLIP ===\n")
            
            calls_ff = case_calls["ff"]
            items_ff = case_counts["ff"]
            avg_ff = calls_ff / items_ff if items_ff > 0 else 0.0
            f.write(f"Case [F->T] (from JSON ff): Total Calls = {calls_ff} | Items = {items_ff} | Avg Calls/Case = {avg_ff:.2f}\n")
            
            calls_ft = case_calls["ft"]
            items_ft = case_counts["ft"]
            avg_ft = calls_ft / items_ft if items_ft > 0 else 0.0
            f.write(f"Case [T->F] (from JSON ft): Total Calls = {calls_ft} | Items = {items_ft} | Avg Calls/Case = {avg_ft:.2f}\n")

        print("\n==============================================")
        print(" PIPELINE RUN PROCESSING COMPLETE")
        print("==============================================")
        print("Analysis Export:", csv_path)
        print("Summary Tallies:", summary_csv_path)
        print("Time Benchmarks:", report_path)

        return disagreement_results



if __name__ == "__main__":

    evaluator = KGCasePerturbationEvaluator(
        model="Qwen/Qwen2.5-3B-Instruct",
        working_dir="/home/vchasanis/Documents/GitHub/LightRAG/GloRAG-Ex/xylotian_storage"
    )

    async def run_pipeline():
        await evaluator.setup()
        # await evaluator.evaluate(
        #     json_file="/home/vchasanis/Documents/GitHub/LightRAG/GloRAG-Ex/competitors/RAGEX-RAGE-SHAPLEY/special_cases/comparison_synthetic_2.json",
        #     top_k=2,
        #     dataset="synthetic",
        #     method="tmc",               
        #     num_iterations=40,          
        #     tolerance=0.002             
        # )
        await evaluator.evaluate(
            json_file="/home/vchasanis/Documents/GitHub/LightRAG/GloRAG-Ex/competitors/RAGEX-RAGE-SHAPLEY/special_cases/comparison_synthetic_2.json",
            top_k=2,
            dataset="synthetic2",
            method="remove_word",               
            num_iterations=40,          
            tolerance=0.002             
        )
        await evaluator.evaluate(
            json_file="/home/vchasanis/Documents/GitHub/LightRAG/GloRAG-Ex/competitors/RAGEX-RAGE-SHAPLEY/special_cases/comparison_synthetic_2.json",
            top_k=2,
            dataset="synthetic",
            method="remove_sentence",               
            num_iterations=40,          
            tolerance=0.002             
        )
        await evaluator.evaluate(
            json_file="/home/vchasanis/Documents/GitHub/LightRAG/GloRAG-Ex/competitors/RAGEX-RAGE-SHAPLEY/special_cases/comparison_synthetic_2.json",
            top_k=2,
            dataset="synthetic",
            method="rage",               
            num_iterations=40,          
            tolerance=0.002             
        )

    asyncio.run(run_pipeline())