#!/usr/bin/env bash
# RAGE (arXiv:2405.13000) counterfactual RAG explainer — competitor baseline.
# Runs the combination-based counterfactual over retrieved SENTENCES for the ft/ff
# flip cases of each comparison file, writing the RAG-Ex {cases:[...]} schema that
# src.correctness.evaluate --method ragex scores. Set MODE=permutation for RAGE's
# position-bias diagnostic (writes <ds>_permutation_analysis.json; not a comparison row).
#
#   results -> all_results/results_rage/<ds>/<ds>_combination_analysis.json
#
# Prereqs (same as every other competitor): local GPU vLLM models, KGs under
# KGs/lightrag/<ds>/, HNSW indices, and the (untracked) comparison_<ds>.json files.
set -euo pipefail

cd "$(dirname "$0")/../../code"      # run from code/ so dataset paths resolve
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

if [[ -x "../.venv/bin/python" ]]; then PYTHON_RUN="../.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then PYTHON_RUN="uv run --no-sync python"
else PYTHON_RUN="python"; fi
echo "Python: ${PYTHON_RUN}"

RAGE="../competitors/RAGE/run_rage.py"
AR="../all_results"
MODE="${MODE:-combination}"
TOP_K="${TOP_K:-2}"
RAG_MODE="${RAG_MODE:-hybrid}"
MAX_LLM_CALLS="${MAX_LLM_CALLS:-200}"
# Per dataset: path to its FF/FT/TF/TT comparison JSON (override via COMPARISON_<DS>).
DATASETS=(synthetic hotpotqa musique 2wiki)

for DS in "${DATASETS[@]}"; do
  echo ""; echo "############ RAGE ${MODE} | DATASET=${DS} ############"
  CMP_VAR="COMPARISON_${DS}"
  CMP="${!CMP_VAR:-../comparison_${DS}.json}"
  if [[ ! -f "$CMP" ]]; then echo "  no comparison file ($CMP); skipping ${DS}"; continue; fi
  OUT_DIR="$AR/results_rage/${DS}"
  mkdir -p "$OUT_DIR"
  $PYTHON_RUN "$RAGE" --dataset "$DS" --mode "$MODE" --rag-mode "$RAG_MODE" \
      --top-k "$TOP_K" --max-llm-calls "$MAX_LLM_CALLS" --comparison "$CMP" --out-dir "$OUT_DIR"
done
