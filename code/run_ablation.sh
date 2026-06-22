#!/usr/bin/env bash
# Ablation of GLoRAG-Ex components on the Synthetic T->F benchmark (Section 6.4.5,
# Table tab:ablation-psp). Reference = full cost-ordered search; each ablation
# toggles one component and writes to its own output dir. Metrics (cost, LLM
# calls, #ops, time) are recorded per instance by save_operations_to_json.
#
# Settings:
#   reference  : full method (precomputed I_E, cache M on)
#   psp_k* : reference + Pivotal-Star Probe evaluated at varying k values
#   no_cache   : disable f-output cache M
#   no_ie      : disable embedding index I_E (recompute node embeddings on the fly)
#   llm_<name> : swap the generation LLM (one run per entry in GEN_LLMS)

set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

DATASET="synthetic"
MODE="ft"                       # T->F (breaking, deletion-driven)
OPS="delete_node,delete_edge"   # T->F uses deletions
TOP_K=2
MAX_COST=20
MAX_LLM_CALLS=200

# PSP-K values to evaluate
PSP_KS=(2 3 5)

# Generation-LLM swaps to evaluate (HF model ids); leave empty to skip.
GEN_LLMS=(
  "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
  "google/gemma-3-27b-it"
  "meta-llama/Llama-3.1-8B-Instruct"
)

RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="src/counterfactuals/results/ablation/${RUN_TS}"
echo "Ablation run timestamp: ${RUN_TS}"
echo "  Outputs -> ${OUT_ROOT}"

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

GEN="$PYTHON_RUN -m src.counterfactuals.generate_ablation"

if [[ ! -f "src/embeddings/${DATASET}/node_index.bin" || ! -f "src/embeddings/${DATASET}/edge_index.bin" ]]; then
  echo "Indices missing for '${DATASET}', building..."
  $PYTHON_RUN -m src.embeddings.build_index --dataset "$DATASET"
fi

COMMON=(--dataset "$DATASET" --mode "$MODE" --ops "$OPS" --top-k "$TOP_K"
        --max-cost "$MAX_COST" --max-llm-calls "$MAX_LLM_CALLS")

run() {  # run <setting-tag> <extra args...>
  local tag="$1"; shift
  local out="${OUT_ROOT}/${tag}"
  echo ""
  echo "##### ABLATION: ${tag}"
  mkdir -p "$out"
  $GEN "${COMMON[@]}" --output-dir "$out" "$@"
}

run "reference"
run "no_cache"  --no-cache
run "no_ie"     --no-ie

for k in "${PSP_KS[@]}"; do
  run "psp_k${k}" --psp --psp-k "$k"
done

for model in "${GEN_LLMS[@]:-}"; do
  [[ -z "$model" ]] && continue
  tag="llm_$(echo "$model" | tr '/:' '__')"
  run "$tag" --gen-llm "$model"
done

echo ""
echo "Ablation complete. Per-instance JSON under ${OUT_ROOT}/<setting>/${DATASET}/delete_ops_${MODE}/"