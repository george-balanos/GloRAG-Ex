#!/usr/bin/env bash
# GloRAG-Ex experiment sweep.
# Run from the repo root: ./run_sweep.sh
#
# Grid:
#   Deletions (--case tf):
#     D1.  --ops delete_node delete_edge                              (suffix _del)
#     D2.  --ops delete_node delete_edge --use-psp --max-pivots K     (suffix _del_psp)
#
#   Additions (--case ft, --ops add_node add_edge delete_node delete_edge):
#     for ADDMODE in expand retrieve both:                   # extension / new-component / mixed pool
#       for COST in unit query context mix:                  # 4 cost variants
#         A_{ADDMODE,COST}                                   (suffix _add_ADDMODE_COST)
#         H_{ADDMODE,COST}  (with PFP heuristic, opt-in)     (suffix _add_ADDMODE_COST_pfp)
#
# Env knobs:
#   RUN_BASE=1                  run D1, D2, all 12 A_* (default 1)
#   RUN_HEURISTIC=0             also run all 12 H_* (PFP heuristic; default 0, opt-in)
#   RUNS="D1 D2 A_expand_unit"  whitelist of run IDs (overrides RUN_BASE/RUN_HEURISTIC)
#   ADDMODES="expand retrieve both"   subset of add-modes to sweep
#   COSTS="unit query context mix"    subset of cost modes to sweep
#   MAX_COST=20                 cost budget per run
#   MAX_LLM_CALLS=200           LLM call budget per run
#   MAX_PIVOTS=3                PSP top-K (deletions)
#   MAX_FRONTIERS=5             PFP top-K (additions)
#   MIX_ALPHA=0.5               α for --add-cost-mode mix
#   TOP_K=10                    baseline RAG top-k
#   MODE=hybrid                 baseline RAG retrieval mode
#   QA=qa/qa_data_synthetic.csv QA dataset (relative to code/)
#
# Examples:
#   ./run_sweep.sh                       # baseline-only (D1, D2, 12 additions)
#   RUN_HEURISTIC=1 ./run_sweep.sh       # baseline + PFP (26 runs)
#   RUNS="D1 D2" ./run_sweep.sh          # just the deletion runs
#   ADDMODES="expand" COSTS="unit mix" ./run_sweep.sh   # 2-cost × expand only

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT/code"
UV_RUN=(uv run --project "$PROJECT_ROOT" python)

RUN_BASE="${RUN_BASE:-1}"
RUN_HEURISTIC="${RUN_HEURISTIC:-0}"
ADDMODES_DEFAULT="expand retrieve both"
COSTS_DEFAULT="unit query context mix"
ADDMODES="${ADDMODES:-$ADDMODES_DEFAULT}"
COSTS="${COSTS:-$COSTS_DEFAULT}"
MAX_COST="${MAX_COST:-20}"
MAX_LLM_CALLS="${MAX_LLM_CALLS:-200}"
MAX_PIVOTS="${MAX_PIVOTS:-3}"
MAX_FRONTIERS="${MAX_FRONTIERS:-5}"
MIX_ALPHA="${MIX_ALPHA:-0.5}"
TOP_K="${TOP_K:-10}"
MODE="${MODE:-hybrid}"
QA="${QA:-qa/qa_data_synthetic.csv}"

mkdir -p benchmark/results

banner() {
    printf '\n========================================\n  %s\n========================================\n' "$1"
}

want_run() {
    local id="$1"
    if [[ -n "${RUNS:-}" ]]; then
        [[ " $RUNS " == *" $id "* ]]
        return
    fi
    # Default policy: D*/A* gated by RUN_BASE, H* gated by RUN_HEURISTIC.
    case "$id" in
        D*|A_*) [[ "$RUN_BASE" == 1 ]] ;;
        H_*)   [[ "$RUN_HEURISTIC" == 1 ]] ;;
        *)     return 1 ;;
    esac
}

# ---------------- Deletion runs (case=tf) ----------------

if want_run D1; then
    banner "D1 — Deletion only (case=tf)"
    "${UV_RUN[@]}" -m src.counterfactuals.generate \
        --input benchmark/results/comparison.json \
        --case tf \
        --max-cost "$MAX_COST" --max-llm-calls "$MAX_LLM_CALLS" \
        --ops delete_node delete_edge \
        --add-mode expand --replace-mode atomic \
        --f1-mode type-only --judge-against original \
        --suffix _del
fi

if want_run D2; then
    banner "D2 — PSP + deletion (case=tf, pivots=$MAX_PIVOTS)"
    "${UV_RUN[@]}" -m src.counterfactuals.generate \
        --input benchmark/results/comparison.json \
        --case tf \
        --max-cost "$MAX_COST" --max-llm-calls "$MAX_LLM_CALLS" \
        --ops delete_node delete_edge \
        --use-psp --max-pivots "$MAX_PIVOTS" \
        --add-mode expand --replace-mode atomic \
        --f1-mode type-only --judge-against original \
        --suffix _del_psp
fi

# ---------------- Addition runs (case=ft) ----------------

for ADDMODE in $ADDMODES; do
    for COST in $COSTS; do
        ID="A_${ADDMODE}_${COST}"
        if want_run "$ID"; then
            banner "$ID — Add+Del (case=ft, add-mode=$ADDMODE, cost=$COST)"
            "${UV_RUN[@]}" -m src.counterfactuals.generate \
                --input benchmark/results/comparison.json \
                --case ft \
                --max-cost "$MAX_COST" --max-llm-calls "$MAX_LLM_CALLS" \
                --ops delete_node delete_edge add_node add_edge \
                --add-mode "$ADDMODE" --add-cost-mode "$COST" --mix-alpha "$MIX_ALPHA" \
                --replace-mode atomic --f1-mode type-only --judge-against original \
                --suffix "_add_${ADDMODE}_${COST}"
        fi

        ID="H_${ADDMODE}_${COST}"
        if want_run "$ID"; then
            banner "$ID — PFP + Add+Del (case=ft, add-mode=$ADDMODE, cost=$COST, frontiers=$MAX_FRONTIERS)"
            "${UV_RUN[@]}" -m src.counterfactuals.generate \
                --input benchmark/results/comparison.json \
                --case ft \
                --max-cost "$MAX_COST" --max-llm-calls "$MAX_LLM_CALLS" \
                --ops delete_node delete_edge add_node add_edge \
                --add-mode "$ADDMODE" --add-cost-mode "$COST" --mix-alpha "$MIX_ALPHA" \
                --use-pfp --max-frontiers "$MAX_FRONTIERS" \
                --replace-mode atomic --f1-mode type-only --judge-against original \
                --suffix "_add_${ADDMODE}_${COST}_pfp"
        fi
    done
done

banner "Sweep complete"
