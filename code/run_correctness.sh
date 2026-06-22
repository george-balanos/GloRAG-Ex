#!/usr/bin/env bash
# Correctness (precision vs ground-truth supporting facts) for ALL methods, on the
# two GT benchmarks (HotpotQA, musique; synthetic has no GT facts and is skipped).
#
# Reads the saved artifacts under all_results/ (post-hoc, no GPU/LLM) and writes one
# correctness JSON per (method, dataset, variant) into all_results/correctness/, then
# a combined CSV/JSON summary.
#
#   GloRAG-Ex      : results_our/<ds>/{all_ops_ff (F->T), delete_ops_ft (T->F)}
#   GloRAG-Ex+PSP  : results_our/psp/<ds>/delete_ops_ft (T->F)
#   Shapley        : results_shap/<ds>/shap_{ff,ft}.json          (attribution)
#   KG-SMILE       : results_kg_smile/kg_smile_<ds>_{ff,ft}.json  (attribution)
#   RAG-Ex         : results_rag_ex/<ds>/<ds>_remove_{sentence,paragraph}_analysis.json (text spans)
#   Shapley-Text   : results_shap_text/<ds>/<ds>_{chunk,sentence}_analysis.json   (text spans)
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

if [[ -x "../.venv/bin/python" ]]; then PYTHON_RUN="../.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then PYTHON_RUN="uv run --no-sync python"
else PYTHON_RUN="python"; fi
echo "Python: ${PYTHON_RUN}"

EVAL="$PYTHON_RUN -m src.correctness.evaluate"
MATCH="${MATCH:-name+desc}"      # name = surface-only; name+desc adds the embedding/desc fallback
AR="../all_results"
OUTDIR="$AR/correctness"
mkdir -p "$OUTDIR"
DATASETS=(hotpotqa musique)

# Matcher backend: the LLM judge is canonical; set HEURISTIC=1 (or JUDGE=0) for the
# fast string+embedding fallback, written to *_heur.json so both can coexist.
JUDGE_FLAG=""; SUF=""
if [[ "${HEURISTIC:-0}" == "1" || "${JUDGE:-1}" == "0" ]]; then JUDGE_FLAG="--heuristic"; SUF="_heur"; fi
echo "match=${MATCH} | backend=$([[ -n "$JUDGE_FLAG" ]] && echo heuristic || echo judge) -> ${OUTDIR}"

# Build GT facts if missing (hotpotqa needs hotpot_dev_distractor_v1.json; musique needs the train jsonl).
[[ -f datasets/hotpotqa/supporting_facts_hotpotqa.json ]] || $PYTHON_RUN datasets/build_supporting_facts_hotpotqa.py || true
[[ -f datasets/musique/supporting_facts_musique.json ]]   || $PYTHON_RUN datasets/build_musique.py || true

glorag() {  # <ds> <relpath-under-all_results> <tag>
  local in="$AR/$2"
  [[ -d "$in" ]] && $EVAL --method glorag --dataset "$1" --facts "$FACTS" --match "$MATCH" $JUDGE_FLAG \
      --input-dir "$in" --output "$OUTDIR/${1}_${3}_correctness${SUF}.json" || echo "  skip glorag: $in"
}
attribution() {  # <ds> <relpath> <tag>
  local in="$AR/$2"
  [[ -f "$in" ]] && $EVAL --method attribution --dataset "$1" --facts "$FACTS" --match "$MATCH" $JUDGE_FLAG \
      --results "$in" --output "$OUTDIR/${1}_${3}_correctness${SUF}.json" || echo "  skip attribution: $in"
}
ragex() {  # <ds> <relpath> <tag>
  local in="$AR/$2"
  [[ -f "$in" ]] && $EVAL --method ragex --dataset "$1" --facts "$FACTS" $JUDGE_FLAG \
      --results "$in" --output "$OUTDIR/${1}_${3}_correctness${SUF}.json" || echo "  skip ragex: $in"
}

for DS in "${DATASETS[@]}"; do
  echo ""; echo "############ DATASET=${DS} ############"
  FACTS="datasets/${DS}/supporting_facts_${DS}.json"
  if [[ ! -f "$FACTS" ]]; then echo "  no facts ($FACTS); skipping ${DS}"; continue; fi

  glorag      "$DS" "results_our/${DS}/all_ops_ff"          "our_ff"
  glorag      "$DS" "results_our/${DS}/delete_ops_ft"       "our_ft"
  glorag      "$DS" "results_our/psp/${DS}/delete_ops_ft"   "our_psp_ft"

  attribution "$DS" "results_shap/${DS}/shap_ff.json"       "shap_ff"
  attribution "$DS" "results_shap/${DS}/shap_ft.json"       "shap_ft"
  attribution "$DS" "results_kg_smile/kg_smile_${DS}_ff.json" "kgsmile_ff"
  attribution "$DS" "results_kg_smile/kg_smile_${DS}_ft.json" "kgsmile_ft"

  ragex       "$DS" "results_rag_ex/${DS}/${DS}_remove_sentence_analysis.json"  "ragex_sentence"
  ragex       "$DS" "results_rag_ex/${DS}/${DS}_remove_paragraph_analysis.json" "ragex_paragraph"

  # Text-chunk Shapley (competitors/Shapley/run_shapley_text.py --comparison): chunk-
  # and sentence-grained spans, scored by the same ragex text-span precision path.
  ragex       "$DS" "results_shap_text/${DS}/${DS}_chunk_analysis.json"    "shaptext_chunk"
  ragex       "$DS" "results_shap_text/${DS}/${DS}_sentence_analysis.json" "shaptext_sent"
done

echo ""; echo "############ COMBINE ############"
$PYTHON_RUN -m src.correctness.analyze_correctness "$OUTDIR"/*_correctness${SUF}.json \
    --csv "$OUTDIR/correctness_summary${SUF}.csv" --json "$OUTDIR/correctness_summary${SUF}.json"
