#!/usr/bin/env bash
# End-to-end driver: run the full GloRAG-Ex grid, then analyze.
# Run from the repo root: ./run_and_analyze.sh
#
# By default runs all 26 cases (2 deletions + 12 base additions + 12 PFP additions),
# then dumps per-directory summaries and the four natural pairwise comparisons
# (D1 vs D2 for deletions; A_* vs H_* for each addition combo).
#
# Env knobs:
#   STAGE=both                   "run" | "analyze" | "both" (default: both)
#   RUN_HEURISTIC=1              include the 12 H_{ADDMODE,COST} PFP variants (default: 1)
#   RUNS=...                     whitelist passed straight through to run_sweep.sh
#   ADDMODES="expand retrieve both"
#   COSTS="unit query context mix"
#   ALPHAS=""                    optional space-separated --mix-alpha sweep
#                                (e.g. ALPHAS="0.0 0.25 0.5 0.75 1.0"); when set,
#                                ONLY the *_mix runs are looped, and outputs are
#                                suffixed _mix_aN.NN to preserve each α's results.
#   MAX_COST=20  MAX_LLM_CALLS=200
#   MAX_PIVOTS=3  MAX_FRONTIERS=5  MIX_ALPHA=0.5
#   TOP_K=10  MODE=hybrid
#   QA=qa/qa_data_synthetic.csv
#   COMPARE_DIR=benchmark/results/compare    where the .csv pairwise tables land
#
# Examples:
#   ./run_and_analyze.sh                                  # full grid + analysis
#   STAGE=analyze ./run_and_analyze.sh                    # analyze existing results, no re-run
#   RUN_HEURISTIC=0 ./run_and_analyze.sh                  # 14 base runs + analysis
#   RUNS="D1 D2 A_both_mix H_both_mix" ./run_and_analyze.sh
#   ALPHAS="0.0 0.25 0.5 0.75 1.0" ./run_and_analyze.sh   # α sweep on mix-cost runs

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="$PROJECT_ROOT/run_sweep.sh"
VIEW="$PROJECT_ROOT/view_results.sh"
[[ -x "$SWEEP" ]] || { echo "missing $SWEEP" >&2; exit 1; }
[[ -x "$VIEW"  ]] || { echo "missing $VIEW"  >&2; exit 1; }

STAGE="${STAGE:-both}"
RUN_HEURISTIC="${RUN_HEURISTIC:-1}"
ADDMODES_DEFAULT="expand retrieve both"
COSTS_DEFAULT="unit query context mix"
ADDMODES="${ADDMODES:-$ADDMODES_DEFAULT}"
COSTS="${COSTS:-$COSTS_DEFAULT}"
ALPHAS="${ALPHAS:-}"
MAX_COST="${MAX_COST:-20}"
MAX_LLM_CALLS="${MAX_LLM_CALLS:-200}"
MAX_PIVOTS="${MAX_PIVOTS:-3}"
MAX_FRONTIERS="${MAX_FRONTIERS:-5}"
MIX_ALPHA="${MIX_ALPHA:-0.5}"
TOP_K="${TOP_K:-10}"
MODE="${MODE:-hybrid}"
QA="${QA:-qa/qa_data_synthetic.csv}"
COMPARE_DIR="${COMPARE_DIR:-benchmark/results/compare}"

banner() {
    printf '\n████████████████████████████████████████\n  %s\n████████████████████████████████████████\n' "$1"
}

export RUN_BASE RUN_HEURISTIC RUNS ADDMODES COSTS \
       MAX_COST MAX_LLM_CALLS MAX_PIVOTS MAX_FRONTIERS MIX_ALPHA \
       TOP_K MODE QA

# ----------------------- prerequisite: HNSW indexes -----------------------
# generate.py loads `src/embeddings/{dataset}/node_index.bin` and `…/edge_index.bin`
# at import time. Build them if missing.

DATASET="${DATASET:-synthetic}"
IDX_DIR="$PROJECT_ROOT/code/src/embeddings/$DATASET"
if [[ ! -f "$IDX_DIR/node_index.bin" || ! -f "$IDX_DIR/edge_index.bin" ]]; then
    banner "Building HNSW indexes for dataset=$DATASET"
    (cd "$PROJECT_ROOT/code" && \
        DATASET="$DATASET" uv run --project "$PROJECT_ROOT" python -m src.embeddings.build_index)
fi

# ----------------------- run -----------------------

if [[ "$STAGE" == "run" || "$STAGE" == "both" ]]; then
    if [[ -n "$ALPHAS" ]]; then
        # α sweep: loop only the mix-cost variants, varying MIX_ALPHA, and
        # snapshot each run's output dir to a per-α suffix so they don't collide.
        # The sweep script writes into suffix `_mix`; we rename it afterward.
        ROBUST="$PROJECT_ROOT/code/src/counterfactuals/robustness"
        for ALPHA in $ALPHAS; do
            banner "ALPHA sweep · MIX_ALPHA=$ALPHA"
            ALPHA_TAG="a$(printf '%s' "$ALPHA" | tr -d '.')"
            MIX_ALPHA="$ALPHA" COSTS="mix" "$SWEEP"
            # Re-tag the mix output directories so the next α doesn't overwrite them.
            for d in "$ROBUST"/*_mix "$ROBUST"/*_mix_pfp; do
                [[ -d "$d" ]] || continue
                mv "$d" "${d}_${ALPHA_TAG}"
            done
        done
    else
        banner "Full grid · $(date)"
        "$SWEEP"
    fi
fi

# ----------------------- analyze -----------------------

if [[ "$STAGE" == "analyze" || "$STAGE" == "both" ]]; then
    banner "Per-directory summary"
    "$VIEW" summary

    banner "Detailed analysis (per result dir)"
    "$VIEW" full

    # Pairwise comparisons:
    #   - D1 vs D2 (deletion baseline vs PSP)
    #   - For each (ADDMODE, COST): A_{ADDMODE,COST} vs H_{ADDMODE,COST} (base vs PFP)
    banner "Pairwise compare_runs"
    mkdir -p "$PROJECT_ROOT/code/$COMPARE_DIR"

    do_compare() {
        local label="$1" base_dir="$2" psp_dir="$3"
        local out="code/$COMPARE_DIR/${label}.csv"
        if [[ ! -d "$PROJECT_ROOT/code/$base_dir" || ! -d "$PROJECT_ROOT/code/$psp_dir" ]]; then
            echo "skip $label — missing $base_dir or $psp_dir"
            return
        fi
        echo
        echo "→ $label"
        (cd "$PROJECT_ROOT/code" && \
            uv run --project "$PROJECT_ROOT" python -m src.counterfactuals.compare_runs \
                --baseline-dir "$base_dir" --psp-dir "$psp_dir" --out "$COMPARE_DIR/${label}.csv")
    }

    # Deletion pair (kaoukis output-dir naming: stability/sem_delete_s_neither_<suffix>)
    do_compare "del_vs_psp" \
        "src/counterfactuals/robustness/stability/sem_delete_s_neither_del" \
        "src/counterfactuals/robustness/stability/sem_delete_s_neither_del_psp"

    # Addition pairs (output naming: stability_other_add_<addmode>_<cost>[_pfp])
    for ADDMODE in $ADDMODES; do
        for COST in $COSTS; do
            label="add_${ADDMODE}_${COST}_vs_pfp"
            base="src/counterfactuals/robustness/stability_other_add_${ADDMODE}_${COST}"
            psp="src/counterfactuals/robustness/stability_other_add_${ADDMODE}_${COST}_pfp"
            do_compare "$label" "$base" "$psp"
        done
    done

    banner "All compare CSVs"
    ls -1 "$PROJECT_ROOT/code/$COMPARE_DIR/"*.csv 2>/dev/null || echo "(no CSVs produced)"

    banner "Baseline accuracy"
    "$VIEW" baseline || true
fi

banner "Pipeline complete"
