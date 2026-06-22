#!/usr/bin/env bash
# Corrective F->T counterfactuals (mode ff) with the Pivotal-Star Probe (PSP)
# enabled at the default K, across all datasets.
#
# Uses the ablation entrypoint (src.counterfactuals.generate_ablation), where the
# corrective-PSP path is implemented. PSP --psp-k is intentionally omitted so the
# CLI default (5) is used.
#
# Per dataset, the script:
#   - builds HNSW indices if missing,
#   - builds the RAG-only / LLM-only baselines + comparison JSON if missing
#     (cached steps are skipped),
#   - runs the ff + PSP counterfactual search.
#
# Results land in:
#   ${OUT_ROOT}/<dataset>/all_ops_ff/counterfactual_*.json
# (save_operations_to_json appends "/<dataset>/all_ops_<mode>" to --output-dir,
#  because mode ff defaults to all four edit ops.)

set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

DATASETS=(synthetic hotpotqa musique 2wiki)
RAG_MODE="hybrid"
TOP_K=5
MAX_COST=20
MAX_LLM_CALLS=200
# NOTE: --psp-k deliberately not set -> uses the CLI default (K=5).

RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="src/counterfactuals/results/ff_psp_${RUN_TS}"

if [[ -x "../.venv/bin/python" ]]; then
  PYTHON_RUN="../.venv/bin/python"
  echo "Using project venv python: ${PYTHON_RUN}"
elif command -v uv >/dev/null 2>&1; then
  PYTHON_RUN="uv run --no-sync python"
  echo "Using uv-managed environment (venv not found at ../.venv)."
else
  PYTHON_RUN="python"
  echo "uv not found, falling back to system Python."
fi

GEN="$PYTHON_RUN -m src.counterfactuals.generate_psp_ff"

echo "Run timestamp: ${RUN_TS}"
echo "  ff + PSP (default K) -> ${OUT_ROOT}/<dataset>/all_ops_ff/"

for DATASET in "${DATASETS[@]}"; do
  echo ""
  echo "##########################################################################"
  echo "##### DATASET=${DATASET}  (mode=ff, F->T corrective, PSP on, default K)"
  echo "##########################################################################"

  RAG_RESULTS="benchmark/results/${DATASET}_${RAG_MODE}_${TOP_K}.json"
  LLM_RESULTS="benchmark/results/${DATASET}_bypass_0.json"
  INPUT_JSON="benchmark/results/comparison_${DATASET}_${TOP_K}.json"

  if [[ ! -f "src/embeddings/${DATASET}/node_index.bin" || ! -f "src/embeddings/${DATASET}/edge_index.bin" ]]; then
    echo "=== [0] Indices missing for '${DATASET}', building... ==="
    $PYTHON_RUN -m src.embeddings.build_index --dataset "$DATASET"
  fi

  if [[ ! -f "$INPUT_JSON" ]]; then
    echo "=== [1a] RAG-only baseline ==="
    [[ -f "$RAG_RESULTS" ]] || $PYTHON_RUN benchmark/run.py \
      --dataset "$DATASET" --rag-mode "$RAG_MODE" --top-k "$TOP_K"

    echo "=== [1b] LLM-only baseline ==="
    [[ -f "$LLM_RESULTS" ]] || $PYTHON_RUN benchmark/run.py \
      --dataset "$DATASET" --rag-mode bypass --top-k 0

    echo "=== [1c] Building comparison file (--rag-only) ==="
    $PYTHON_RUN -m benchmark.evaluation \
      --dataset "$DATASET" --rag-mode "$RAG_MODE" --top-k "$TOP_K" --rag-only

    if [[ ! -f "$INPUT_JSON" ]]; then
      echo "ERROR: $INPUT_JSON still missing after evaluation step." >&2
      exit 1
    fi
  else
    echo "=== [1] Comparison file cached: ${INPUT_JSON} ==="
  fi

  echo "=== [2] Counterfactuals F->T (ff) + PSP (default K) ==="
  $GEN \
    --dataset "$DATASET" \
    --rag-mode "$RAG_MODE" \
    --top-k "$TOP_K" \
    --input "$INPUT_JSON" \
    --mode ff \
    --psp \
    --max-cost "$MAX_COST" \
    --max-llm-calls "$MAX_LLM_CALLS" \
    --output-dir "$OUT_ROOT"
done

echo ""
echo "Done. F->T (ff) + PSP results under ${OUT_ROOT}/<dataset>/all_ops_ff/"
