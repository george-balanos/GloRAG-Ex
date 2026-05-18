#!/usr/bin/env bash
# GloRAG-Ex experiment sweep — runs all 8 steps from the plan.
# Run from this directory: cd competitors/Synthetic && ./run_sweep.sh
#
# Env knobs:
#   STEPS="0 1 2 3 4 5 6 7 8"   subset of steps to run (default: all)
#   MAX_COST=20                 max-cost budget
#   MAX_LLM_CALLS=200           max-llm-calls budget
#   MAX_PIVOTS=3                PSP pivots for deletion-only (Step 4)
#   MAX_PIVOTS_ADD=5            PSP pivots for add+del runs (Steps 7-8)
#   TOP_K=10                    baseline RAG top-k
#   MODE=hybrid                 baseline RAG retrieval mode
#   QA=qa/qa_data_synthetic.csv QA dataset
#
# Examples:
#   ./run_sweep.sh                       # run everything
#   STEPS="3 4" ./run_sweep.sh           # only the deletion experiments
#   STEPS="5 6 7 8" ./run_sweep.sh       # only the addition experiments

set -euo pipefail

# Resolve uv project root (parent of this script's grandparent: …/glorag).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# All `uv run` invocations point at the root pyproject; cwd stays in Synthetic
# so relative paths (qa/, benchmark/, src/) resolve as the python modules expect.
UV_RUN=(uv run --project "$PROJECT_ROOT" python)

STEPS="${STEPS:-0 1 2 3 4 5 6 7 8}"
MAX_COST="${MAX_COST:-20}"
MAX_LLM_CALLS="${MAX_LLM_CALLS:-200}"
MAX_PIVOTS="${MAX_PIVOTS:-3}"
MAX_PIVOTS_ADD="${MAX_PIVOTS_ADD:-5}"
TOP_K="${TOP_K:-10}"
MODE="${MODE:-hybrid}"
QA="${QA:-qa/qa_data_synthetic.csv}"

mkdir -p benchmark/results

run_step() {
    local n="$1"
    [[ " $STEPS " == *" $n "* ]]
}

banner() {
    printf '\n========================================\n  %s\n========================================\n' "$1"
}

# ---- Step 0: LLM-only baseline (produces synthetic_bypass_0.json) ----
if run_step 0; then
    banner "Step 0 — LLM-only baseline"
    "${UV_RUN[@]}" - <<'PY'
import asyncio, json, os, pandas as pd
from src.retrieve import initialize_lightrag
from src.query import query
from src.llm_judge import judge_response

async def main():
    rag = await initialize_lightrag()
    df = pd.read_csv("qa/qa_data_synthetic.csv").drop_duplicates(subset=["questions"]).reset_index(drop=True)
    out = {}
    for _, row in df.iterrows():
        ans = await query(rag, context="", question=row["questions"])
        score = await judge_response(row["questions"], generated_answer=ans, ground_truth=row["answers"])
        out[row["id"]] = {
            "score": score,
            "generated_answer": ans,
            "question": row["questions"],
            "ground_truth": row["answers"],
        }
    os.makedirs("benchmark/results", exist_ok=True)
    with open("benchmark/results/synthetic_bypass_0.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} entries to benchmark/results/synthetic_bypass_0.json")

asyncio.run(main())
PY
fi

# ---- Step 1: Baseline RAG ----
if run_step 1; then
    banner "Step 1 — Baseline RAG (mode=$MODE top-k=$TOP_K)"
    "${UV_RUN[@]}" -m benchmark.run \
        --mode "$MODE" --top-k "$TOP_K" \
        --qa "$QA" \
        --out "benchmark/results/synthetic_${MODE}_${TOP_K}.json"
fi

# ---- Step 2: Build comparison.json ----
if run_step 2; then
    banner "Step 2 — Build comparison.json"
    "${UV_RUN[@]}" - <<PY
from benchmark.evaluation import export_performance_cases
export_performance_cases(
    llm_results_path="benchmark/results/synthetic_bypass_0.json",
    rag_results_path="benchmark/results/synthetic_${MODE}_${TOP_K}.json",
    output_path="benchmark/results/comparison.json",
)
PY
fi

# ---- Step 3: Deletion only ----
if run_step 3; then
    banner "Step 3 — Deletion only (case=tf)"
    "${UV_RUN[@]}" -m src.counterfactuals.generate \
        --input benchmark/results/comparison.json \
        --case tf \
        --max-cost "$MAX_COST" --max-llm-calls "$MAX_LLM_CALLS" \
        --ops delete_node delete_edge \
        --add-mode expand --replace-mode atomic \
        --f1-mode type-only --judge-against original \
        --suffix _delete_only
fi

# ---- Step 4: PSP, deletion only ----
if run_step 4; then
    banner "Step 4 — PSP + deletion only (case=tf, pivots=$MAX_PIVOTS)"
    "${UV_RUN[@]}" -m src.counterfactuals.generate \
        --input benchmark/results/comparison.json \
        --case tf \
        --max-cost "$MAX_COST" --max-llm-calls "$MAX_LLM_CALLS" \
        --ops delete_node delete_edge \
        --use-psp --max-pivots "$MAX_PIVOTS" \
        --add-mode expand --replace-mode atomic \
        --f1-mode type-only --judge-against original \
        --suffix _psp_delete_only
fi

# ---- Step 5: Del + Add (extension) ----
if run_step 5; then
    banner "Step 5 — Del + Add extension (case=ft, add-mode=expand)"
    "${UV_RUN[@]}" -m src.counterfactuals.generate \
        --input benchmark/results/comparison.json \
        --case ft \
        --max-cost "$MAX_COST" --max-llm-calls "$MAX_LLM_CALLS" \
        --ops delete_node delete_edge add_node add_edge \
        --add-mode expand --replace-mode atomic \
        --f1-mode type-only --judge-against original \
        --suffix _add_delete_extend
fi

# ---- Step 6: Del + Add (new component) ----
if run_step 6; then
    banner "Step 6 — Del + Add new component (case=ft, add-mode=retrieve)"
    "${UV_RUN[@]}" -m src.counterfactuals.generate \
        --input benchmark/results/comparison.json \
        --case ft \
        --max-cost "$MAX_COST" --max-llm-calls "$MAX_LLM_CALLS" \
        --ops delete_node delete_edge add_node add_edge \
        --add-mode retrieve --replace-mode atomic \
        --f1-mode type-only --judge-against original \
        --suffix _add_delete_retrieve
fi

# ---- Step 7: PSP + Del + Add (extension) ----
if run_step 7; then
    banner "Step 7 — PSP + Del + Add extension (case=ft, pivots=$MAX_PIVOTS_ADD)"
    "${UV_RUN[@]}" -m src.counterfactuals.generate \
        --input benchmark/results/comparison.json \
        --case ft \
        --max-cost "$MAX_COST" --max-llm-calls "$MAX_LLM_CALLS" \
        --ops delete_node delete_edge add_node add_edge \
        --use-psp --max-pivots "$MAX_PIVOTS_ADD" \
        --add-mode expand --replace-mode atomic \
        --f1-mode type-only --judge-against original \
        --suffix _psp_add_delete_extend
fi

# ---- Step 8: PSP + Del + Add (new component) ----
if run_step 8; then
    banner "Step 8 — PSP + Del + Add new component (case=ft, pivots=$MAX_PIVOTS_ADD)"
    "${UV_RUN[@]}" -m src.counterfactuals.generate \
        --input benchmark/results/comparison.json \
        --case ft \
        --max-cost "$MAX_COST" --max-llm-calls "$MAX_LLM_CALLS" \
        --ops delete_node delete_edge add_node add_edge \
        --use-psp --max-pivots "$MAX_PIVOTS_ADD" \
        --add-mode retrieve --replace-mode atomic \
        --f1-mode type-only --judge-against original \
        --suffix _psp_add_delete_retrieve
fi

banner "Sweep complete"
