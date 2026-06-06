#!/usr/bin/env bash
# ablation runner
#
# Step 0: Auto-build HNSW indices if missing.
# Step 1: RAG-only baseline (benchmark/run.py).
# Step 1b: Build comparison JSON from RAG results only (evaluation.py --rag-only).
# Step 2: Counterfactual deletions only, T->F (--mode ft).
# Step 3: Counterfactual deletions + PSP, T->F.
# Step 4: Counterfactual additions only, F->T (--mode tf),
#         ablated over --adm {1,2,3} x --add-heuristic {none, tier(w...), blend(a...)}.

set -euo pipefail

cd "$(dirname "$0")"  # ensure CWD = code/ for relative paths
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)" 
# params
DATASET="synthetic"
RAG_MODE="hybrid"
TOP_K=2
NUM_ROWS=   # empty = all rows
MAX_COST=20
MAX_LLM_CALLS=200
PSP_K=5
TIER_WIDTHS=(0.25 0.5 1.0 2.0)
ALPHAS=(0.1 0.25 0.5 0.75 1.0)
ADM_MODES=(1 2 3)
OUT_ROOT="src/counterfactuals/results/ablation"

INPUT_JSON="benchmark/results/comparison_${DATASET}_${TOP_K}.json"

# check for uv
if command -v uv >/dev/null 2>&1; then
  PYTHON_RUN="uv run python"
  echo "Using uv-managed environment."
else
  PYTHON_RUN="python"
  echo "uv not found, falling back to system Python."
fi

# build indices if missing
if [[ ! -f "src/embeddings/${DATASET}/node_index.bin" || ! -f "src/embeddings/${DATASET}/edge_index.bin" ]]; then
  echo "Indices missing for '${DATASET}', building..."
  $PYTHON_RUN -m src.embeddings.build_index --dataset "$DATASET"
fi

GEN="$PYTHON_RUN -m src.counterfactuals.generate"

# ─── 1. RAG-only baseline ────────────────────────────────────────────────────
RAG_RESULTS="benchmark/results/${DATASET}_${RAG_MODE}_${TOP_K}.json"
if [[ -f "$RAG_RESULTS" ]]; then
  echo "=== [1/4] RAG-only baseline (cached: ${RAG_RESULTS}) ==="
else
  echo "=== [1/4] RAG-only baseline ==="
  $PYTHON_RUN benchmark/run.py \
    --dataset "$DATASET" \
    --rag-mode "$RAG_MODE" \
    --top-k "$TOP_K" \
    ${NUM_ROWS:+--num-rows "$NUM_ROWS"}
fi

# ─── 1b. Build comparison JSON from RAG results only ─────────────────────────
echo "=== [1b/4] Building comparison file (--rag-only) ==="
$PYTHON_RUN -m benchmark.evaluation \
  --dataset "$DATASET" \
  --rag-mode "$RAG_MODE" \
  --top-k "$TOP_K" \
  --rag-only

if [[ ! -f "$INPUT_JSON" ]]; then
  echo "ERROR: $INPUT_JSON still missing after evaluation step." >&2
  exit 1
fi

# 2. Deletions only, T->F (no PSP)
echo "=== [2/4] Deletions only, T->F (no PSP) ==="
$GEN \
  --dataset "$DATASET" \
  --rag-mode "$RAG_MODE" \
  --top-k "$TOP_K" \
  --input   "$INPUT_JSON" \
  --mode    ft \
  --ops     delete_node,delete_edge \
  --max-cost "$MAX_COST" \
  --max-llm-calls "$MAX_LLM_CALLS" \
  --output-dir "${OUT_ROOT}/ft_delete_no_psp"

# 3. Deletions + PSP, T->F
echo "=== [3/4] Deletions + PSP, T->F ==="
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

# 4. Additions ablation, F->T
echo "=== [4/4] Additions ablation, F->T ==="
for adm in "${ADM_MODES[@]}"; do
  echo "--- adm=${adm} | --add-heuristic none ---"
  $GEN \
    --dataset "$DATASET" --rag-mode "$RAG_MODE" --top-k "$TOP_K" \
    --input "$INPUT_JSON" \
    --mode tf --ops add_node,add_edge --adm "$adm" \
    --add-heuristic none \
    --max-cost "$MAX_COST" --max-llm-calls "$MAX_LLM_CALLS" \
    --output-dir "${OUT_ROOT}/tf_add_adm${adm}_none"

  for tw in "${TIER_WIDTHS[@]}"; do
    echo "--- adm=${adm} | --add-heuristic tier --tier-width ${tw} ---"
    $GEN \
      --dataset "$DATASET" --rag-mode "$RAG_MODE" --top-k "$TOP_K" \
      --input "$INPUT_JSON" \
      --mode tf --ops add_node,add_edge --adm "$adm" \
      --add-heuristic tier --tier-width "$tw" \
      --max-cost "$MAX_COST" --max-llm-calls "$MAX_LLM_CALLS" \
      --output-dir "${OUT_ROOT}/tf_add_adm${adm}_tier_w${tw}"
  done

  for a in "${ALPHAS[@]}"; do
    echo "--- adm=${adm} | --add-heuristic blend --alpha ${a} ---"
    $GEN \
      --dataset "$DATASET" --rag-mode "$RAG_MODE" --top-k "$TOP_K" \
      --input "$INPUT_JSON" \
      --mode tf --ops add_node,add_edge --adm "$adm" \
      --add-heuristic blend --alpha "$a" \
      --max-cost "$MAX_COST" --max-llm-calls "$MAX_LLM_CALLS" \
      --output-dir "${OUT_ROOT}/tf_add_adm${adm}_blend_a${a}"
  done
done

echo "=== Done. Results under ${OUT_ROOT}/ ==="
