#!/usr/bin/env bash
# RUN_TS folder isolates every run. Per dataset (synthetic, hotpotqa):
#   Step 0 : Auto-build HNSW indices if missing.
#   Step 1a: RAG-only baseline (benchmark/run.py).        [stable cached path]
#   Step 1b: LLM-only baseline (bypass).                  [stable cached path]
#   Step 1c: Build comparison JSON (evaluation.py --rag-only).  [stable cached path]
#   Step 2 : Counterfactual deletions only, T->F (--mode ft, no PSP).
#   Step 3 : Counterfactual deletions + PSP, T->F.
#   Step 4 : Counterfactual additions, F->T corrective (ff),

set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

DATASETS=(synthetic hotpotqa) # medical
RAG_MODE="hybrid"
TOP_K=2
NUM_ROWS=   # empty = all rows
MAX_COST=20
MAX_LLM_CALLS=200
PSP_K=3
ADM_MODES=(2)
CORRECTIVE_MODES=(ff)
SHAP_DEVICE="cuda:1"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="src/counterfactuals/results/ablation/${RUN_TS}"

echo "Run timestamp: ${RUN_TS}"
echo "  Counterfactuals -> ${OUT_ROOT}"

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

GEN="$PYTHON_RUN -m src.counterfactuals.generate"

for DATASET in "${DATASETS[@]}"; do
  echo ""
  echo "##########################################################################"
  echo "##### DATASET=${DATASET}"
  echo "##########################################################################"

  RAG_RESULTS="benchmark/results/${DATASET}_${RAG_MODE}_${TOP_K}.json"
  LLM_RESULTS="benchmark/results/${DATASET}_bypass_0.json"
  INPUT_JSON="benchmark/results/comparison_${DATASET}_${TOP_K}.json"

  if [[ ! -f "src/embeddings/${DATASET}/node_index.bin" || ! -f "src/embeddings/${DATASET}/edge_index.bin" ]]; then
    echo "Indices missing for '${DATASET}', building..."
    $PYTHON_RUN -m src.embeddings.build_index --dataset "$DATASET"
  fi

  if [[ -f "$RAG_RESULTS" ]]; then
    echo "=== [1a] RAG-only baseline (cached: ${RAG_RESULTS}) ==="
  else
    echo "=== [1a] RAG-only baseline ==="
    $PYTHON_RUN benchmark/run.py \
      --dataset "$DATASET" \
      --rag-mode "$RAG_MODE" \
      --top-k "$TOP_K" \
      ${NUM_ROWS:+--num-rows "$NUM_ROWS"}
  fi

  if [[ -f "$LLM_RESULTS" ]]; then
    echo "=== [1b] LLM-only baseline (cached: ${LLM_RESULTS}) ==="
  else
    echo "=== [1b] LLM-only baseline ==="
    $PYTHON_RUN benchmark/run.py \
      --dataset "$DATASET" \
      --rag-mode bypass \
      --top-k 0 \
      ${NUM_ROWS:+--num-rows "$NUM_ROWS"}
  fi

  echo "=== [1c] Building comparison file (--rag-only) ==="
  $PYTHON_RUN -m benchmark.evaluation \
    --dataset "$DATASET" \
    --rag-mode "$RAG_MODE" \
    --top-k "$TOP_K" \
    --rag-only

  if [[ ! -f "$INPUT_JSON" ]]; then
    echo "ERROR: $INPUT_JSON still missing after evaluation step." >&2
    exit 1
  fi

  echo "=== [2] Deletions only, T->F (no PSP) ==="
  $GEN \
    --dataset "$DATASET" \
    --rag-mode "$RAG_MODE" \
    --top-k "$TOP_K" \
    --input "$INPUT_JSON" \
    --mode ft \
    --ops delete_node,delete_edge \
    --max-cost "$MAX_COST" \
    --max-llm-calls "$MAX_LLM_CALLS" \
    --output-dir "${OUT_ROOT}/ft_delete_no_psp"

  echo "=== [3] Deletions + PSP, T->F ==="
  $GEN \
    --dataset "$DATASET" \
    --rag-mode "$RAG_MODE" \
    --top-k "$TOP_K" \
    --input   "$INPUT_JSON" \
    --mode    ft \
    --ops     delete_node,delete_edge \
    --psp --psp-k "$PSP_K" \
    --max-cost "$MAX_COST" \
    --max-llm-calls "$MAX_LLM_CALLS" \
    --output-dir "${OUT_ROOT}/ft_delete_psp_k${PSP_K}"

  echo "=== [4] Deletions + Additions, F->T ==="
  $GEN \
    --dataset "$DATASET" --rag-mode "$RAG_MODE" --top-k "$TOP_K" \
    --input "$INPUT_JSON" \
    --mode ff --ops add_node,add_edge,delete_node,delete_edge --adm "$adm" \
    --add-heuristic none \
    --max-cost "$MAX_COST" --max-llm-calls "$MAX_LLM_CALLS" \
    --output-dir "${OUT_ROOT}/ff_add_adm${adm}_none"
done

echo "Done. All counterfactuals written to ${OUT_ROOT}."