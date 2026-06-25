#!/usr/bin/env bash
# Sentence-context benchmark for RAGE (fair analog of code/benchmark/run.py +
# evaluation.py). Scores RAG over the SAME sentence context run_rage.py explains and
# builds comparison_<ds>_sentence_<top_k>.json (FF/FT/TF/TT) that drives RAGE's
# experiments via --comparison.
#
#   results -> all_results/results_rage/<ds>/comparison_<ds>_sentence_<top_k>.json
set -euo pipefail

cd "$(dirname "$0")/../../code"      # run from code/ so dataset paths resolve
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

if [[ -x "../.venv/bin/python" ]]; then PYTHON_RUN="../.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then PYTHON_RUN="uv run --no-sync python"
else PYTHON_RUN="python"; fi
echo "Python: ${PYTHON_RUN}"

BENCH="../competitors/RAGE/run_benchmark.py"
AR="../all_results"
TOP_K="${TOP_K:-2}"
RAG_MODE="${RAG_MODE:-hybrid}"
NUM_ROWS="${NUM_ROWS:-}"
DATASETS=(hotpotqa musique)

for DS in "${DATASETS[@]}"; do
  echo ""; echo "############ RAGE sentence-comparison | DATASET=${DS} ############"
  OUT_DIR="$AR/results_rage/${DS}"
  mkdir -p "$OUT_DIR"
  EXTRA=(); [[ -n "$NUM_ROWS" ]] && EXTRA=(--num-rows "$NUM_ROWS")
  # Reuse an existing LLM-only run by setting LLM_RESULTS_<DS> (skips the bypass pass).
  LLM_VAR="LLM_RESULTS_${DS}"; LLM="${!LLM_VAR:-}"
  [[ -n "$LLM" ]] && EXTRA+=(--llm-results "$LLM")
  $PYTHON_RUN "$BENCH" --dataset "$DS" --rag-mode "$RAG_MODE" --top-k "$TOP_K" \
      --build-comparison --out-dir "$OUT_DIR" "${EXTRA[@]}"
done
