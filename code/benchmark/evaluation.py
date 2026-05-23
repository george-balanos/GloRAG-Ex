import json
import pandas as pd

def accuracy(results_path: str) -> float:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results_df = pd.DataFrame.from_dict(data, orient="index")

    total = len(results_df)
    correct = results_df["score"].sum()
    acc = correct / total

    print(f"\nTotal: {total} | Correct: {int(correct)} | Accuracy: {acc:.2%}")

    return acc

def export_performance_cases(llm_results_path: str, rag_results_path: str, output_path: str = "benchmark/results/comparison", dataset: str = "synthetic") -> None:
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

    with open(f"{output_path}_{dataset}_{top_k}.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(comparison)} entries to {output_path}")
    print(f"  TT (both correct):          {len(cases['tt'])}")
    print(f"  TF (llm correct, rag wrong - RAG worsened):  {len(cases['tf'])}")
    print(f"  FT (llm wrong, rag correct - RAG improved):  {len(cases['ft'])}")
    print(f"  FF (both wrong):            {len(cases['ff'])}")
    print(f"  LLM Accuracy: {output['summary']['llm_accuracy']:.2%}")
    print(f"  RAG Accuracy: {output['summary']['rag_accuracy']:.2%}")

if __name__ == "__main__":
    dataset = "hotpotqa"

    mode = "hybrid"
    top_k = 2

    rag_results = f"benchmark/results/{dataset}_{mode}_{top_k}.json"
    llm_results = f"benchmark/results/{dataset}_bypass_0.json"
    
    # rag_results = f"benchmark/results/hotpotqa_{mode}_{top_k}.json"
    # llm_results = f"benchmark/results/hotpotqa_bypass_0.json"
    
    export_performance_cases(llm_results_path=llm_results, rag_results_path=rag_results, dataset=dataset)