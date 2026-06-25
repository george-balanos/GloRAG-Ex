#!/usr/bin/env bash
# RAGE noise-resistance benchmark (arXiv:2405.13000), competitor baseline.
# Injects foreign sentences into the retrieved context, regenerates the answer, and
# measures how much RAGE combination-counterfactual importance lands on the noise.
#
#   results -> all_results/results_rage/<ds>/<ds>_rage_noise.json (+ _metrics.json)
#
# Prereqs: local GPU vLLM models, KGs under KGs/lightrag/<ds>/ (incl.
# kv_store_text_chunks.json for the foreign-sentence pool), HNSW indices.
set -euo pipefail

cd "$(dirname "$0")/../../code"      # run from code/ so dataset paths resolve
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

if [[ -x "../.venv/bin/python" ]]; then PYTHON_RUN="../.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then PYTHON_RUN="uv run --no-sync python"
else PYTHON_RUN="python"; fi
echo "Python: ${PYTHON_RUN}"

RAGE_NOISE="../competitors/RAGE/run_rage_noise.py"
AR="../all_results"
TOP_K="${TOP_K:-2}"
RAG_MODE="${RAG_MODE:-hybrid}"
NOISE_PCTS="${NOISE_PCTS:-0.1,0.2,0.3,0.5}"
MAX_LLM_CALLS="${MAX_LLM_CALLS:-200}"
NUM_ROWS="${NUM_ROWS:-}"
DATASETS=(synthetic hotpotqa musique 2wiki)

for DS in "${DATASETS[@]}"; do
  echo ""; echo "############ RAGE noise | DATASET=${DS} ############"
  # ft cases -> top-down (check removed set); ff cases -> bottom-up (check retained set).
  CMP_VAR="COMPARISON_${DS}"
  CMP="${!CMP_VAR:-../comparison_${DS}.json}"
  if [[ ! -f "$CMP" ]]; then echo "  no comparison file ($CMP); skipping ${DS}"; continue; fi
  OUT_DIR="$AR/results_rage/${DS}"
  mkdir -p "$OUT_DIR"
  EXTRA=(); [[ -n "$NUM_ROWS" ]] && EXTRA=(--num-rows "$NUM_ROWS")
  $PYTHON_RUN "$RAGE_NOISE" --dataset "$DS" --rag-mode "$RAG_MODE" --top-k "$TOP_K" \
      --comparison "$CMP" --noise-percentages "$NOISE_PCTS" --max-llm-calls "$MAX_LLM_CALLS" \
      --output "$OUT_DIR/${DS}_rage_noise.json" "${EXTRA[@]}"
done
