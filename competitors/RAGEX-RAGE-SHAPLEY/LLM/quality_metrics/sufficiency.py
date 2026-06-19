import json
import os
import asyncio
import argparse

from lightrag.prompt import PROMPTS
from LLM.llm_judge import judge_response
from LLM.llm_utils import vllm_model_complete


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_perturbation_results(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def load_found_cases(folder: str) -> set[str]:
    """Return the set of questions from found=True case files."""
    found_questions = set()
    for root, _, files in os.walk(folder):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            path = os.path.join(root, fname)
            with open(path, "r") as f:
                data = json.load(f)
            if data.get("found") is True:
                q = data.get("question")
                if q:
                    found_questions.add(q)
    return found_questions


def join_items(items: list[str], method: str) -> str:
    """Join context items based on the perturbation method."""
    if "sentence" in method:
        return " ".join(item.rstrip(".") + "." for item in items)
    else:  # paragraph or any other method
        return "\n\n".join(items)


def load_explanation(case: dict, top_k: int = 1, method: str = "remove_paragraph") -> dict | None:
    question     = case["question"]
    ground_truth = case["ground_truth"]

    removed_item_importance = case.get("removed_item_importance", {})
    if not removed_item_importance:
        return None

    # Sort by importance weight descending and take top_k
    sorted_items = sorted(
        removed_item_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # Filter out non-positive scores
    sorted_items = [(item, score) for item, score in sorted_items if score > 0]
    if not sorted_items:
        return None

    top_context = join_items([item for item, _ in sorted_items], method)
    top_scores  = {item: score for item, score in sorted_items}

    return {
        "question":     question,
        "ground_truth": ground_truth,
        "top_context":  top_context,
        "top_scores":   top_scores,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

async def evaluate_explanation(case: dict, top_k: int = 1, method: str = "remove_paragraph") -> bool | None:
    explanation_dict = load_explanation(case, top_k=top_k, method=method)

    if explanation_dict is None:
        return None

    system_prompt = PROMPTS["rag_response"].format(
        context_data=explanation_dict["top_context"],
        response_type="Single Sentence, without references and extra explanations.",
        user_prompt=""
    )

    response = await vllm_model_complete(
        prompt=explanation_dict["question"],
        system_prompt=system_prompt,
        temperature=0,
        max_tokens=512,
    )

    score = await judge_response(
        question=explanation_dict["question"],
        generated_answer=response,
        ground_truth=explanation_dict["ground_truth"],
    )

    return score == 1


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Evaluate perturbation-based explanations.")
    parser.add_argument("--dataset",     type=str, required=True, help="hotpotqa | synthetic | musique")
    parser.add_argument("--method",      type=str, default="remove_paragraph", help="remove_paragraph | remove_sentence")
    parser.add_argument("--mode",        type=str, default="ft",               help="ft | ff")
    parser.add_argument("--top_k",       type=int, default=1,                  help="number of top importance chunks to use as context")
    parser.add_argument("--input",       type=str, default=None,               help="override input JSON path")
    parser.add_argument("--output",      type=str, default=None,               help="override output JSON path")
    parser.add_argument("--found-cases", type=str, default=None,               help="folder of found=True counterfactual cases to filter by")
    args = parser.parse_args()

    input_path = args.input or (
        f"/home/gbalanos/GloRAG-Ex/competitors/RAGEX-RAGE-SHAPLEY/experiments/"
        f"{args.dataset}/{args.dataset}_{args.method}_analysis.json"
    )
    output_path = args.output or (
        f"perturbation/quality_metrics/"
        f"{args.dataset}_{args.method}_{args.mode}_top{args.top_k}_sufficiency.json"
    )

    join_style = "sentence" if "sentence" in args.method else "paragraph"
    print(f"Dataset: {args.dataset} | Method: {args.method} | Mode: {args.mode} | Top-K: {args.top_k} | Join: {join_style}")
    print(f"Input:   {input_path}")

    cases = load_perturbation_results(input_path)
    cases = [c for c in cases if c.get("case_type", "").lower() == args.mode]
    print(f"Found {len(cases)} cases with case_type='{args.mode}'")

    # Optionally filter to only questions present in the found_cases folder
    if args.found_cases:
        found_questions = load_found_cases(args.found_cases)
        print(f"Loaded {len(found_questions)} found questions from {args.found_cases}")
        before = len(cases)
        cases = [c for c in cases if c.get("question") in found_questions]
        print(f"Filtered {before} → {len(cases)} cases matching found questions")

    print()

    results       = []
    entry_results = []

    for i, case in enumerate(cases):
        case_id = case.get("case_id", i)
        print(f"[{i+1}/{len(cases)}] Case {case_id}...", end=" ", flush=True)

        try:
            result = await evaluate_explanation(case, top_k=args.top_k, method=args.method)
            if result is None:
                print("SKIPPED (no valid explanation)")
                entry_results.append({"id": case_id, "correct": None, "skipped": True})
            else:
                results.append(result)
                entry_results.append({"id": case_id, "correct": result})
                print("✓" if result else "✗")
        except Exception as e:
            print(f"SKIPPED ({type(e).__name__}: {e})")
            entry_results.append({"id": case_id, "correct": None, "error": str(e)})

    total    = len(results)
    correct  = sum(results)
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.2%})")

    output = {
        "dataset":  args.dataset,
        "method":   args.method,
        "mode":     args.mode,
        "top_k":    args.top_k,
        "accuracy": accuracy,
        "correct":  correct,
        "total":    total,
        "entries":  entry_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())