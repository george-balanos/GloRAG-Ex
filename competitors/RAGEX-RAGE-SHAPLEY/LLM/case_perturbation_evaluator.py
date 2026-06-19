import os
import json
import time
import csv
import re
import asyncio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.asyncio import tqdm

from lightrag.prompt import PROMPTS
from lightrag import QueryParam

from retrieval.retrieve import initialize_lightrag
from retrieval.parser import parse_context
from LLM.llm_judge import judge_response, get_binary_score


class KGCasePerturbationEvaluator:

    def __init__(self, working_dir=None):
        self.working_dir = working_dir
        self.rag = None

    async def setup(self):
        self.rag = await initialize_lightrag(self.working_dir)
        print("[Setup] RAG + LLM models ready.")
        return self

    async def _get_similarity(self, text1: str, text2: str) -> float:
        vecs = await self.rag.embedding_func([str(text1), str(text2)])
        return float(cosine_similarity([vecs[0]], [vecs[1]])[0][0])

    async def _call(self, query, prompt: str | None = None) -> str:
        if isinstance(query, str):
            return await self.rag.llm_model_func(query)

        if prompt == "QA_PROMPT":
            context_data  = query.get("context", "")
            question      = query.get("question", "")
            system_prompt = PROMPTS["rag_response"].format(
                context_data=context_data,
                response_type="Single Sentence, without references and extra explanations.",
                user_prompt=""
            )
            return await self.rag.llm_model_func(question, system_prompt=system_prompt)

        return await self.rag.llm_model_func(str(query))

    async def retrieve_context(self, question, top_k=5, mode="hybrid"):
        try:
            param = QueryParam(
                mode=mode, top_k=top_k, only_need_context=True, enable_rerank=False
            )
            context_str = await self.rag.aquery(question, param=param)
            if not context_str:
                return []

            parsed_subgraph = parse_context(context_str)
            if (
                not parsed_subgraph
                or not hasattr(parsed_subgraph, "chunks")
                or not parsed_subgraph.chunks
            ):
                return []

            return [chunk.strip() for chunk in parsed_subgraph.chunks if chunk.strip()]

        except Exception as e:
            print(f"RAG ERROR: {e}")
            return []

    def perturb(self, chunks, method="remove_sentence"):
        if not chunks:
            return []

        out       = []
        full_text = "\n\n".join(chunks)

        if method == "remove_sentence":
            sents = [s.strip() for s in full_text.split(".") if s.strip()]
            for i in range(len(sents)):
                new_ctx = ". ".join(sents[:i] + sents[i + 1:]) + "." if sents else ""
                out.append((new_ctx, sents[i] + "."))

        elif method == "remove_word":
            words = full_text.split()
            for i in range(len(words)):
                out.append((" ".join(words[:i] + words[i + 1:]), words[i]))

        elif method == "remove_paragraph":
            paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            for i in range(len(paragraphs)):
                new_ctx = "\n\n".join(paragraphs[:i] + paragraphs[i + 1:])
                out.append((new_ctx, paragraphs[i]))

        elif method == "rage":
            paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            if len(paragraphs) < 2:
                return [(full_text, "none")]
            for i in range(len(paragraphs)):
                for j in range(i + 1, len(paragraphs)):
                    swapped       = paragraphs[:]
                    swapped[i], swapped[j] = swapped[j], swapped[i]
                    out.append(("\n\n".join(swapped), f"swapped_{i}_with_{j}"))

        return out

    def load(self, path):
        with open(path) as f:
            return json.load(f)

    async def evaluate(
        self,
        json_file,
        out_dir="results_kg",
        top_k=5,
        dataset="synthetic",
        method="remove_sentence",
    ):
        data    = self.load(json_file)
        results = data["results"]
        ids     = list(results.keys())

        relevant_ids = [
            qid for qid in ids
            if results[qid].get("case", "unknown").lower() in ("ft", "ff")
        ]

        os.makedirs(out_dir, exist_ok=True)

        json_results         = []
        total_llm_calls      = 0
        start_time           = time.time()
        disagreement_results = {"T->F": 0, "F->T": 0}

        total_tf_time = total_ft_time = 0
        tf_cases = ft_cases = 0
        tf_calls = ft_calls = 0
        total_perturb_time  = 0

        print(f"\nFound {len(relevant_ids)} disagreement cases (ft/ff) out of {len(ids)} total.")

        for qid in tqdm(relevant_ids, desc="Processing cases", unit="case"):
            q_start = time.time()

            item          = results[qid]
            case_type     = item.get("case", "unknown").lower()
            question      = item["question"]
            ground_truth  = item.get("ground_truth", "")
            display_label = "F->T" if case_type == "ff" else "T->F"
            disagreement_results[display_label] += 1

            chunks = await self.retrieve_context(question, top_k=top_k)
            if not chunks:
                continue

            context = "\n\n".join(chunks)

            original_answer = await self._call(
                {"question": question, "context": context},
                prompt="QA_PROMPT",
            )

            perturbations = self.perturb(chunks, method=method)

            case_json = {
                "case_id":          qid,
                "question":         question,
                "case_type":        case_type,
                "mapped_label":     display_label,
                "method":           method,
                "ground_truth":     ground_truth,
                "original_answer":  original_answer,
                "original_context": context,
                "analysis":         [],
                "llm_calls":        1,
                "elapsed_time":     0.0,
            }

            row_start = time.time()
            records   = []

            for new_ctx, removed in perturbations:
                new_answer = await self._call(
                    {"question": question, "context": new_ctx},
                    prompt="QA_PROMPT",
                )

                print("-" * 60)
                print(new_answer)
                print("-" * 60)

                is_correct = await judge_response(
                    question=question,
                    generated_answer=new_answer,
                    ground_truth=ground_truth,
                )
                is_flip = is_correct == 0

                total_llm_calls += 2

                similarity        = await self._get_similarity(original_answer, new_answer)
                importance_weight = 1.0 - similarity

                record = {
                    "removed_item":      removed,
                    "new_answer":        new_answer,
                    "judge_result":      is_correct,
                    "is_flip":           is_flip,
                    "similarity":        similarity,
                    "importance_weight": importance_weight,
                }
                records.append(record)
                case_json["analysis"].append(record)

            case_json["removed_item_importance"] = {
                r["removed_item"]: r["importance_weight"] for r in records
            }

            row_elapsed         = time.time() - row_start
            total_perturb_time += row_elapsed

            if case_type == "ff":
                total_ft_time += row_elapsed
                ft_cases      += 1
            else:
                total_tf_time += row_elapsed
                tf_cases      += 1

            # case_json["llm_calls"]    = 1 + 2 * len(perturbations)
            case_json["llm_calls"]    = 1 + len(perturbations)
            case_json["elapsed_time"] = time.time() - q_start

            print(f"[Case {qid}] done in {case_json['elapsed_time']:.2f}s | perturbations: {len(perturbations)} | llm_calls: {case_json['llm_calls']}")

            if case_type == "ff":
                ft_calls += case_json["llm_calls"]
            else:
                tf_calls += case_json["llm_calls"]

            json_results.append(case_json)

        # ── write outputs ──────────────────────────────────────────────────────
        json_path    = os.path.join(out_dir, f"{dataset}_{method}_analysis.json")
        summary_path = os.path.join(out_dir, f"{dataset}_{method}_summary.csv")

        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in disagreement_results.items():
                writer.writerow([k, v])

        elapsed = time.time() - start_time
        print("\nDONE")
        print("JSON:", json_path)
        print("Summary:", summary_path)
        print(f"Total time: {elapsed:.2f}s")

        summary = {
            "total_llm_calls":       total_llm_calls,
            "total_ft_time":         total_ft_time,
            "total_tf_time":         total_tf_time,
            "avg_ft_time_per_case":  total_ft_time / ft_cases if ft_cases else 0,
            "avg_tf_time_per_case":  total_tf_time / tf_cases if tf_cases else 0,
            "avg_ft_calls_per_case": ft_calls / ft_cases if ft_cases else 0,
            "avg_tf_calls_per_case": tf_calls / tf_cases if tf_cases else 0,
            "ft_cases":              ft_cases,
            "tf_cases":              tf_cases,
        }

        with open(json_path, "w") as f:
            json.dump({"cases": json_results, "summary": summary}, f, indent=2)

        return disagreement_results


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    evaluator = KGCasePerturbationEvaluator(
        working_dir="/home/gbalanos/GloRAG-Ex/competitors/RAGEX-RAGE-SHAPLEY/KGs/lightrag/musique"
    )

    async def run_pipeline():
        await evaluator.setup()

        json_file = "/home/gbalanos/GloRAG-Ex/competitors/RAGEX-RAGE-SHAPLEY/results/comparison_musique_2_ff81.json"

        await evaluator.evaluate(
            json_file,
            top_k=2,
            dataset="musique",
            method="remove_sentence",
            out_dir="/home/gbalanos/GloRAG-Ex/competitors/RAGEX-RAGE-SHAPLEY/experiments/musique"
        )

    asyncio.run(run_pipeline())