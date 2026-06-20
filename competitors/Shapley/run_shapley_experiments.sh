#!/usr/bin/env bash
# Decoupled Shapley experiment runner.
#
# This script lives in competitors/Shapley/ but runs with CWD = code/ so the
# relative dataset paths (KGs/, datasets/, src/embeddings/) resolve exactly as in
# the main pipeline. The Python entrypoints (run_shapley.py, run_shapley_noise.py)
# live next to this script and bootstrap code/src onto sys.path themselves.
#
# Per dataset:
#   S1: TMC-Shapley attribution + per-row LLM-call/time metrics   (run_shapley.py)
#   S2: Shapley context-permutation robustness                    (run_shapley.py --permute)
#   S3: Shapley noise resistance                                  (run_shapley_noise.py)
#
# Outputs go to competitors/Shapley/results/<timestamp>/ (decoupled from code/).

set -euo pipefail

# ── Locations ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # competitors/Shapley
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CODE_DIR="${REPO_ROOT}/code"
cd "${CODE_DIR}"                                             # relative data paths resolve here
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

# ── Parameters ───────────────────────────────────────────────────────────────
DATASETS=(synthetic hotpotqa)   # also configured: musique
RAG_MODE="hybrid"
TOP_K=2
NUM_ROWS=                       # empty = all rows
SHAP_DEVICE="cuda:1"            # GPU for the HF Mistral utility model
NOISE_PERCENTAGES="0.1,0.2,0.3,0.5"
TOP_ATTR_KS="1,3,5"             # k values for the "noise in top-k attributions" check
RUN_PLAIN=1                     # S1
RUN_PERMUTE=1                   # S2
RUN_NOISE=1                     # S3

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="${SCRIPT_DIR}/results/${RUN_TS}"
mkdir -p "${RESULTS_ROOT}"
echo "Run timestamp: ${RUN_TS}"
echo "  Shapley results -> ${RESULTS_ROOT}"

# ── Python runner ────────────────────────────────────────────────────────────
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_RUN="${REPO_ROOT}/.venv/bin/python"
  echo "Using project venv python: ${PYTHON_RUN}"
elif command -v uv >/dev/null 2>&1; then
  PYTHON_RUN="uv run --no-sync python"
  echo "Using uv-managed environment (venv not found at ${REPO_ROOT}/.venv)."
else
  PYTHON_RUN="python"
  echo "uv not found, falling back to system Python."
fi

# ── Pre-build indices ────────────────────────────────────────────────────────
# run_shapley_noise imports src.quality_metrics.noise_resistance, which loads the
# synthetic HNSW index at import time regardless of --dataset, so synthetic plus
# every run dataset must have its index built before anything runs.
for DATASET in "${DATASETS[@]}" synthetic; do
  if [[ ! -f "src/embeddings/${DATASET}/node_index.bin" || \
        ! -f "src/embeddings/${DATASET}/edge_index.bin" ]]; then
    echo "Indices missing for '${DATASET}', building..."
    $PYTHON_RUN -m src.embeddings.build_index --dataset "$DATASET"
  fi
done

# ── Per-dataset Shapley experiments ──────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
  echo ""
  echo "##########################################################################"
  echo "##### DATASET=${DATASET}"
  echo "##########################################################################"

  if [[ "${RUN_PLAIN}" == "1" ]]; then
    echo "=== [S1] Shapley TMC attribution (RAG/Shapley/whole metrics) ==="
    $PYTHON_RUN "${SCRIPT_DIR}/run_shapley.py" \
      --dataset "$DATASET" \
      --rag-mode "$RAG_MODE" \
      --top-k "$TOP_K" \
      --shap-device "$SHAP_DEVICE" \
      --output  "${RESULTS_ROOT}/${DATASET}_shapley_tmc.json" \
      --metrics "${RESULTS_ROOT}/${DATASET}_shapley_tmc_metrics.json" \
      ${NUM_ROWS:+--num-rows "$NUM_ROWS"}
  fi

  if [[ "${RUN_PERMUTE}" == "1" ]]; then
    echo "=== [S2] Shapley permutation robustness (Kendall-tau + per-object spread) ==="
    $PYTHON_RUN "${SCRIPT_DIR}/run_shapley.py" \
      --permute \
      --dataset "$DATASET" \
      --rag-mode "$RAG_MODE" \
      --top-k "$TOP_K" \
      --shap-device "$SHAP_DEVICE" \
      --output "${RESULTS_ROOT}/${DATASET}_shapley_permutation.json" \
      ${NUM_ROWS:+--num-rows "$NUM_ROWS"}
  fi

  if [[ "${RUN_NOISE}" == "1" ]]; then
    echo "=== [S3] Shapley noise resistance (noise attribution share) ==="
    $PYTHON_RUN "${SCRIPT_DIR}/run_shapley_noise.py" \
      --dataset "$DATASET" \
      --rag-mode "$RAG_MODE" \
      --top-k "$TOP_K" \
      --shap-device "$SHAP_DEVICE" \
      --noise-percentages "$NOISE_PERCENTAGES" \
      --top-attr-ks "$TOP_ATTR_KS" \
      --output  "${RESULTS_ROOT}/${DATASET}_shapley_noise.json" \
      --metrics "${RESULTS_ROOT}/${DATASET}_shapley_noise_metrics.json" \
      ${NUM_ROWS:+--num-rows "$NUM_ROWS"}
  fi

  echo "Completed dataset: ${DATASET}"
done

echo ""
echo "=== Done. Shapley results under ${RESULTS_ROOT}/ ==="
