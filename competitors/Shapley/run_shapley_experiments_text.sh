#!/usr/bin/env bash
# Decoupled TEXT-CHUNK Shapley experiment runner (text-excerpt analog of
# run_shapley_experiments.sh).
#
# Runs with CWD = code/ so the relative dataset paths (KGs/, datasets/,
# src/embeddings/) resolve exactly as in the main pipeline. The Python entrypoints
# (run_shapley_text.py, run_shapley_noise_text.py) live next to this script and
# bootstrap code/src onto sys.path themselves.
#
# Per dataset:
#   S1: TMC-Shapley attribution over chunks + per-row metrics  (run_shapley_text.py)
#   S2: chunk context-permutation robustness                   (run_shapley_text.py --permute)
#   S3: chunk noise resistance                                 (run_shapley_noise_text.py)
#   S4: correctness-format cases (RAG-Ex {cases:[...]} schema) (run_shapley_text.py --comparison)
#       -> all_results/results_shap_text/<ds>/<ds>_chunk_analysis.json, scored by
#          code/run_correctness.sh via `src.correctness.evaluate --method ragex`.
#
# S1-S3 outputs go to competitors/Shapley/results_text/<timestamp>/ (decoupled).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # competitors/Shapley
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CODE_DIR="${REPO_ROOT}/code"
cd "${CODE_DIR}"                                             # relative data paths resolve here
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

DATASETS=(synthetic hotpotqa)   # also configured: musique
RAG_MODE="hybrid"
TOP_K=2
NUM_ROWS=                       # empty = all rows
NOISE_PERCENTAGES="0.1,0.2,0.3,0.5"
RUN_PLAIN=1                     # S1
RUN_PERMUTE=1                   # S2
RUN_NOISE=1                     # S3
RUN_CORRECTNESS=1               # S4

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="${SCRIPT_DIR}/results_text/${RUN_TS}"
CORRECTNESS_ROOT="${REPO_ROOT}/all_results/results_shap_text"
mkdir -p "${RESULTS_ROOT}"
echo "Run timestamp: ${RUN_TS}"
echo "  Shapley(text) results     -> ${RESULTS_ROOT}"
echo "  Shapley(text) correctness -> ${CORRECTNESS_ROOT}"

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

for DATASET in "${DATASETS[@]}" synthetic; do
  if [[ ! -f "src/embeddings/${DATASET}/node_index.bin" || \
        ! -f "src/embeddings/${DATASET}/edge_index.bin" ]]; then
    echo "Indices missing for '${DATASET}', building..."
    $PYTHON_RUN -m src.embeddings.build_index --dataset "$DATASET"
  fi
done

for DATASET in "${DATASETS[@]}"; do
  echo ""
  echo "##########################################################################"
  echo "##### DATASET=${DATASET}"
  echo "##########################################################################"

  for GRAN in "${GRANULARITIES[@]}"; do
    echo ""
    echo "  ---------- granularity=${GRAN} ----------"

    if [[ "${RUN_PLAIN}" == "1" ]]; then
      echo "=== [S1] Shapley(text/${GRAN}) TMC attribution (RAG/Shapley/whole metrics) ==="
      $PYTHON_RUN "${SCRIPT_DIR}/run_shapley_text.py" \
        --granularity "$GRAN" \
        --dataset "$DATASET" \
        --rag-mode "$RAG_MODE" \
        --top-k "$TOP_K" \
        --shap-device "$SHAP_DEVICE" \
        --output  "${RESULTS_ROOT}/${DATASET}_shapley_text_${GRAN}_tmc.json" \
        --metrics "${RESULTS_ROOT}/${DATASET}_shapley_text_${GRAN}_tmc_metrics.json" \
        ${NUM_ROWS:+--num-rows "$NUM_ROWS"}
    fi

    if [[ "${RUN_PERMUTE}" == "1" ]]; then
      echo "=== [S2] Shapley(text/${GRAN}) permutation robustness (Kendall-tau + per-unit spread) ==="
      $PYTHON_RUN "${SCRIPT_DIR}/run_shapley_text.py" \
        --permute \
        --granularity "$GRAN" \
        --dataset "$DATASET" \
        --rag-mode "$RAG_MODE" \
        --top-k "$TOP_K" \
        --shap-device "$SHAP_DEVICE" \
        --output "${RESULTS_ROOT}/${DATASET}_shapley_text_${GRAN}_permutation.json" \
        ${NUM_ROWS:+--num-rows "$NUM_ROWS"}
      $PYTHON_RUN "${SCRIPT_DIR}/analyze_shapley_permutation_text.py" \
        --input "${RESULTS_ROOT}/${DATASET}_shapley_text_${GRAN}_permutation.json" || true
    fi

    if [[ "${RUN_NOISE}" == "1" ]]; then
      echo "=== [S3] Shapley(text/${GRAN}) noise resistance (noise attribution share) ==="
      $PYTHON_RUN "${SCRIPT_DIR}/run_shapley_noise_text.py" \
        --granularity "$GRAN" \
        --dataset "$DATASET" \
        --rag-mode "$RAG_MODE" \
        --top-k "$TOP_K" \
        --shap-device "$SHAP_DEVICE" \
        --noise-percentages "$NOISE_PERCENTAGES" \
        --top-attr-ks "$TOP_ATTR_KS" \
        --output  "${RESULTS_ROOT}/${DATASET}_shapley_text_${GRAN}_noise.json" \
        --metrics "${RESULTS_ROOT}/${DATASET}_shapley_text_${GRAN}_noise_metrics.json" \
        ${NUM_ROWS:+--num-rows "$NUM_ROWS"}
    fi

    if [[ "${RUN_CORRECTNESS}" == "1" ]]; then
      CMP="benchmark/results/comparison_${DATASET}_${TOP_K}.json"
      if [[ -f "$CMP" ]]; then
        echo "=== [S4] Shapley(text/${GRAN}) correctness cases (RAG-Ex format) ==="
        mkdir -p "${CORRECTNESS_ROOT}/${DATASET}"
        # writes ${CORRECTNESS_ROOT}/${DATASET}/${DATASET}_${GRAN}_analysis.json
        $PYTHON_RUN "${SCRIPT_DIR}/run_shapley_text.py" \
          --comparison "$CMP" \
          --granularity "$GRAN" \
          --dataset "$DATASET" \
          --rag-mode "$RAG_MODE" \
          --top-k "$TOP_K" \
          --shap-device "$SHAP_DEVICE" \
          --out-dir "${CORRECTNESS_ROOT}/${DATASET}" \
          ${NUM_ROWS:+--num-rows "$NUM_ROWS"}
      else
        echo "=== [S4/${GRAN}] skipped: no comparison JSON at ${CMP} (build it via benchmark/evaluation.py) ==="
      fi
    fi
  done

  echo "Completed dataset: ${DATASET}"
done

echo ""
echo "=== Done. Shapley(text) results under ${RESULTS_ROOT}/ ==="
echo "    Correctness cases under ${CORRECTNESS_ROOT}/ (scored by code/run_correctness.sh)"
