#!/usr/bin/env bash
# RAGE combination-counterfactual permutation robustness (experiment B), the analog
# of code/src/counterfactuals/permutation_robustness.py. Post-hoc: reads each
# dataset's <ds>_combination_analysis.json (from run_rage.sh) and checks whether the
# flipped combination cases survive reordering of their perturbed source set.
#
#   results -> all_results/results_rage/<ds>/<ds>_combination_permutation_robustness.json
set -euo pipefail

cd "$(dirname "$0")/../../code"      # run from code/ so dataset paths resolve
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

if [[ -x "../.venv/bin/python" ]]; then PYTHON_RUN="../.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then PYTHON_RUN="uv run --no-sync python"
else PYTHON_RUN="python"; fi
echo "Python: ${PYTHON_RUN}"

PR="../competitors/RAGE/run_rage_perm_robustness.py"
AR="../all_results"
COUNT="${COUNT:-5}"
DATASETS=(synthetic hotpotqa musique 2wiki)

for DS in "${DATASETS[@]}"; do
  echo ""; echo "############ RAGE perm-robustness | DATASET=${DS} ############"
  IN="$AR/results_rage/${DS}/${DS}_combination_analysis.json"
  if [[ ! -f "$IN" ]]; then echo "  no combination output ($IN); run run_rage.sh first"; continue; fi
  $PYTHON_RUN "$PR" --dataset "$DS" --input "$IN" --count "$COUNT"
done
