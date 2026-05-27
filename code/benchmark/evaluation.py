import argparse
import json
import os

import pandas as pd

from src.dataset_setup import DATASETS


def accuracy(results_path: str) -> float:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results_df = pd.DataFrame.from_dict(data, orient="index")

    total = len(results_df)
    correct = results_df["score"].sum()
    acc = correct / total

    print(f"\nTotal: {total} | Correct: {int(correct)} | Accuracy: {acc:.2%}")

    return acc


def export_performance_cases(
    llm_results_path: str,
    rag_results_path: str,
    dataset: str,
    top_k: int,
    output_path: str | None = None,
) -> str:
    """Join LLM-only vs RAG results and classify each question into tt/tf/ft/ff.

    output_path default: benchmark/results/comparison_{dataset}_{top_k}.json
    Returns the actual output path written.
    """
    with open(llm_results_path, "r", encoding="utf-8") as f:
        data_llm = json.load(f)

    with open(rag_results_path, "r", encoding="utf-8") as f:
        data_rag = json.load(f)

    comparison = {}
    cases = {"tt": [], "tf": [], "ft": [], "ff": []}

    for id in data_llm:
        if id not in data_rag:
            continue

        llm = data_llm[id]
        rag = data_rag[id]

        llm_score = int(llm["score"])
        rag_score = int(rag["score"])

        entry = {
            "question":       llm["question"],
            "ground_truth":   llm["ground_truth"],
            "llm_answer":     llm["generated_answer"],
            "rag_answer":     rag["generated_answer"],
            "llm_score":      llm_score,
            "rag_score":      rag_score,
        }

        # tt = both correct, tf = llm correct rag wrong
        # ft = llm wrong rag correct, ff = both wrong
        if   llm_score == 1 and rag_score == 1: case = "tt"
        elif llm_score == 1 and rag_score == 0: case = "tf"
        elif llm_score == 0 and rag_score == 1: case = "ft"
        else:                                   case = "ff"

        entry["case"] = case
        cases[case].append(id)
        comparison[id] = entry

    output = {
        "summary": {
            "total":        len(comparison),
            "tt":           len(cases["tt"]),
            "tf":           len(cases["tf"]),
            "ft":           len(cases["ft"]),
            "ff":           len(cases["ff"]),
            "llm_accuracy": sum(v["llm_score"] for v in comparison.values()) / len(comparison),
            "rag_accuracy": sum(v["rag_score"] for v in comparison.values()) / len(comparison),
        },
        "cases": cases,
        "results": comparison,
    }

    if output_path is None:
        output_path = f"benchmark/results/comparison_{dataset}_{top_k}.json"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(comparison)} entries to {output_path}")
    print(f"  TT (both correct):                              {len(cases['tt'])}")
    print(f"  TF (llm correct, rag wrong - RAG worsened):     {len(cases['tf'])}")
    print(f"  FT (llm wrong, rag correct - RAG improved):     {len(cases['ft'])}")
    print(f"  FF (both wrong):                                {len(cases['ff'])}")
    print(f"  LLM Accuracy: {output['summary']['llm_accuracy']:.2%}")
    print(f"  RAG Accuracy: {output['summary']['rag_accuracy']:.2%}")

    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluation",
        description="Compare LLM-only vs RAG results and classify per-question cases.",
    )
    p.add_argument("--dataset", choices=DATASETS, default="synthetic")
    p.add_argument("--rag-mode", choices=["hybrid", "local", "global", "naive"], default="hybrid",
                   help="LightRAG retrieval mode used in the RAG run (for default RAG-results path).")
    p.add_argument("--top-k", type=int, default=2,
                   help="LightRAG top_k used in the RAG run (for default RAG-results path).")
    p.add_argument("--bypass-suffix", default="bypass_0",
                   help="Suffix in the LLM-only results filename, e.g. 'bypass_0' for '<dataset>_bypass_0.json'.")
    p.add_argument("--rag-results", default=None,
                   help="Explicit path to RAG results JSON (default derived from --dataset/--rag-mode/--top-k).")
    p.add_argument("--llm-results", default=None,
                   help="Explicit path to LLM-only results JSON (default derived from --dataset/--bypass-suffix).")
    p.add_argument("--output", default=None,
                   help="Output path (default: benchmark/results/comparison_<dataset>_<top-k>.json).")
    return p


def main(args: argparse.Namespace):
    rag_results = args.rag_results or f"benchmark/results/{args.dataset}_{args.rag_mode}_{args.top_k}.json"
    llm_results = args.llm_results or f"benchmark/results/{args.dataset}_{args.bypass_suffix}.json"

    export_performance_cases(
        llm_results_path=llm_results,
        rag_results_path=rag_results,
        dataset=args.dataset,
        top_k=args.top_k,
        output_path=args.output,
    )


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
