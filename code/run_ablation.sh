#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

# ------------------------------------------------------------------
# Global parameters
# ------------------------------------------------------------------

DATASETS=("hotpotqa" "synthetic")

RAG_MODE="hybrid"
TOP_K=2
NUM_ROWS=
MAX_COST=20
MAX_LLM_CALLS=200
PSP_K=5

TIER_WIDTHS=(2.0)
ALPHAS=(0.5)
ADM_MODES=(1 2)

# ------------------------------------------------------------------
# Python runner
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
# Run pipeline for each dataset
# ------------------------------------------------------------------

for DATASET in "${DATASETS[@]}"; do
  echo
  echo "============================================================"
  echo "DATASET: ${DATASET}"
  echo "============================================================"

  OUT_ROOT="src/counterfactuals/results/ablation/${DATASET}"
  INPUT_JSON="benchmark/results/comparison_${DATASET}_${TOP_K}.json"

  # --------------------------------------------------------------
  # Build indices if missing
  # --------------------------------------------------------------

  if [[ ! -f "src/embeddings/${DATASET}/node_index.bin" || \
        ! -f "src/embeddings/${DATASET}/edge_index.bin" ]]; then
    echo "Indices missing for '${DATASET}', building..."
    $PYTHON_RUN -m src.embeddings.build_index --dataset "$DATASET"
  fi

  GEN="$PYTHON_RUN -m src.counterfactuals.generate"

  # --------------------------------------------------------------
  # RAG baseline
  # --------------------------------------------------------------

  RAG_RESULTS="benchmark/results/${DATASET}_${RAG_MODE}_${TOP_K}.json"

  if [[ -f "$RAG_RESULTS" ]]; then
    echo "=== [1a/4] RAG-only baseline (cached) ==="
  else
    echo "=== [1a/4] RAG-only baseline ==="
    $PYTHON_RUN benchmark/run.py \
      --dataset "$DATASET" \
      --rag-mode "$RAG_MODE" \
      --top-k "$TOP_K" \
      ${NUM_ROWS:+--num-rows "$NUM_ROWS"}
  fi

  # --------------------------------------------------------------
  # LLM baseline
  # --------------------------------------------------------------

  LLM_RESULTS="benchmark/results/${DATASET}_bypass_0.json"

  if [[ -f "$LLM_RESULTS" ]]; then
    echo "=== [1b/4] LLM-only baseline (cached) ==="
  else
    echo "=== [1b/4] LLM-only baseline ==="
    $PYTHON_RUN benchmark/run.py \
      --dataset "$DATASET" \
      --rag-mode bypass \
      --top-k 0 \
      ${NUM_ROWS:+--num-rows "$NUM_ROWS"}
  fi

  # --------------------------------------------------------------
  # Build comparison file
  # --------------------------------------------------------------

  echo "=== [1c/4] Building comparison file ==="

  $PYTHON_RUN -m benchmark.evaluation \
    --dataset "$DATASET" \
    --top-k "$TOP_K"

  if [[ ! -f "$INPUT_JSON" ]]; then
    echo "ERROR: $INPUT_JSON missing after evaluation step." >&2
    exit 1
  fi

  # --------------------------------------------------------------
  # Deletions only (T -> F)
  # --------------------------------------------------------------

  echo "=== [2/4] Deletions only, T->F (no PSP) ==="

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

  # --------------------------------------------------------------
  # Deletions + PSP
  # --------------------------------------------------------------

  echo "=== [3/4] Deletions + PSP, T->F ==="

  $GEN \
    --dataset "$DATASET" \
    --rag-mode "$RAG_MODE" \
    --top-k "$TOP_K" \
    --input "$INPUT_JSON" \
    --mode ft \
    --ops delete_node,delete_edge \
    --psp \
    --psp-k "$PSP_K" \
    --max-cost "$MAX_COST" \
    --max-llm-calls "$MAX_LLM_CALLS" \
    --output-dir "${OUT_ROOT}/ft_delete_psp_k${PSP_K}"

  # --------------------------------------------------------------
  # Additions ablation (F -> T)
  # --------------------------------------------------------------

  echo "=== [4/4] Additions ablation, F->T ==="

  for adm in "${ADM_MODES[@]}"; do

    echo "--- adm=${adm} | heuristic=none ---"

    $GEN \
      --dataset "$DATASET" \
      --rag-mode "$RAG_MODE" \
      --top-k "$TOP_K" \
      --input "$INPUT_JSON" \
      --mode tf \
      --ops add_node,add_edge \
      --adm "$adm" \
      --add-heuristic none \
      --max-cost "$MAX_COST" \
      --max-llm-calls "$MAX_LLM_CALLS" \
      --output-dir "${OUT_ROOT}/tf_add_adm${adm}_none"

    for tw in "${TIER_WIDTHS[@]}"; do
      echo "--- adm=${adm} | heuristic=tier | width=${tw} ---"

      $GEN \
        --dataset "$DATASET" \
        --rag-mode "$RAG_MODE" \
        --top-k "$TOP_K" \
        --input "$INPUT_JSON" \
        --mode tf \
        --ops add_node,add_edge \
        --adm "$adm" \
        --add-heuristic tier \
        --tier-width "$tw" \
        --max-cost "$MAX_COST" \
        --max-llm-calls "$MAX_LLM_CALLS" \
        --output-dir "${OUT_ROOT}/tf_add_adm${adm}_tier_w${tw}"
    done

    for alpha in "${ALPHAS[@]}"; do
      echo "--- adm=${adm} | heuristic=blend | alpha=${alpha} ---"

      $GEN \
        --dataset "$DATASET" \
        --rag-mode "$RAG_MODE" \
        --top-k "$TOP_K" \
        --input "$INPUT_JSON" \
        --mode tf \
        --ops add_node,add_edge \
        --adm "$adm" \
        --add-heuristic blend \
        --alpha "$alpha" \
        --max-cost "$MAX_COST" \
        --max-llm-calls "$MAX_LLM_CALLS" \
        --output-dir "${OUT_ROOT}/tf_add_adm${adm}_blend_a${alpha}"
    done
  done

  echo "Completed dataset: ${DATASET}"
done

echo
echo "=== All experiments completed ==="