#!/usr/bin/env bash
# View / summarize the GloRAG-Ex sweep outputs.
# Run from the repo root: ./view_results.sh
#
# Modes:
#   ./view_results.sh                  list all result dirs + one-line summary each
#   ./view_results.sh full             list + full analyze.py output for each dir
#   ./view_results.sh compare          run compare_runs.py for the natural baseline/PSP pairs
#   ./view_results.sh dir <path>       full analyze.py for one directory
#   ./view_results.sh question <substr> [dir]   show ops+answers for any counterfactual whose question matches
#   ./view_results.sh baseline         print baseline RAG vs LLM-only accuracy from comparison.json

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT/code"
UV_RUN=(uv run --project "$PROJECT_ROOT" python)

ROBUST_DIR="src/counterfactuals/robustness"

find_result_dirs() {
    # Any directory under robustness/ that contains at least one counterfactual_*.json.
    [[ -d "$ROBUST_DIR" ]] || return 0
    find "$ROBUST_DIR" -type f -name 'counterfactual_*.json' \
        -printf '%h\n' 2>/dev/null | sort -u
}

# One-line summary per dir: counts + success rate, computed inline so we
# don't depend on analyze.py output format.
quick_summary() {
    local dir="$1"
    "${UV_RUN[@]}" - "$dir" <<'PY'
import json, sys, glob, os
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "counterfactual_*.json")))
n = len(files)
found = 0
costs = []
ops_lens = []
llm_calls = []
sims = []
for fp in files:
    try:
        with open(fp) as f:
            r = json.load(f)
    except Exception:
        continue
    if r.get("found"):
        found += 1
    if r.get("cost") is not None:
        costs.append(r["cost"])
    if r.get("operations") is not None:
        ops_lens.append(len(r["operations"]))
    if r.get("llm_calls") is not None:
        llm_calls.append(r["llm_calls"])
    sim = (r.get("answers") or {}).get("similarity")
    if sim is not None:
        sims.append(sim)

def mean(xs): return sum(xs) / len(xs) if xs else float("nan")

print(f"  files: {n:>4}  found: {found:>4} ({(found/n*100 if n else 0):.1f}%)  "
      f"mean_cost: {mean(costs):.2f}  mean_ops: {mean(ops_lens):.2f}  "
      f"mean_llm: {mean(llm_calls):.1f}  mean_sim: {mean(sims):.3f}")
PY
}

cmd="${1:-summary}"

case "$cmd" in
    summary|"")
        echo "Result directories under $ROBUST_DIR:"
        echo
        dirs=$(find_result_dirs)
        if [[ -z "$dirs" ]]; then
            echo "  (none — sweep hasn't produced any counterfactual_*.json yet)"
            exit 0
        fi
        while IFS= read -r d; do
            printf '%s\n' "${d#$ROBUST_DIR/}"
            quick_summary "$d"
            echo
        done <<< "$dirs"
        ;;

    full)
        dirs=$(find_result_dirs)
        if [[ -z "$dirs" ]]; then
            echo "No result dirs found."
            exit 0
        fi
        while IFS= read -r d; do
            echo "============================================================"
            echo "  $d"
            echo "============================================================"
            "${UV_RUN[@]}" -m src.counterfactuals.analyze --dir "$d"
            echo
        done <<< "$dirs"
        ;;

    dir)
        target="${2:?usage: view_results.sh dir <path>}"
        "${UV_RUN[@]}" -m src.counterfactuals.analyze --dir "$target"
        ;;

    compare)
        # Natural pairings: a baseline run vs its PSP counterpart.
        # We find pairs by matching basename without the "_psp" infix.
        mkdir -p benchmark/results
        declare -A baseline_for
        while IFS= read -r d; do
            base="$(basename "$d")"
            if [[ "$base" == *"_psp_"* ]]; then
                continue
            fi
            baseline_for["$base"]="$d"
        done <<< "$(find_result_dirs)"

        any=0
        while IFS= read -r d; do
            base="$(basename "$d")"
            [[ "$base" == *"_psp_"* ]] || continue
            sibling="${base/_psp_/_}"
            base_path="${baseline_for[$sibling]:-}"
            if [[ -z "$base_path" ]]; then
                echo "skip: $base — no matching baseline dir for '$sibling'"
                continue
            fi
            out="benchmark/results/compare_${base}.csv"
            echo "comparing:"
            echo "  baseline: $base_path"
            echo "  psp:      $d"
            echo "  out:      $out"
            "${UV_RUN[@]}" -m src.counterfactuals.compare_runs \
                --baseline-dir "$base_path" \
                --psp-dir "$d" \
                --out "$out"
            echo
            any=1
        done <<< "$(find_result_dirs)"
        [[ "$any" == 1 ]] || echo "No PSP/baseline pairs found."
        ;;

    question)
        needle="${2:?usage: view_results.sh question <substring> [dir]}"
        scope="${3:-$ROBUST_DIR}"
        "${UV_RUN[@]}" - "$needle" "$scope" <<'PY'
import json, sys, glob, os
needle, scope = sys.argv[1], sys.argv[2]
hits = 0
for fp in sorted(glob.glob(os.path.join(scope, "**", "counterfactual_*.json"), recursive=True)):
    try:
        with open(fp) as f:
            r = json.load(f)
    except Exception:
        continue
    q = r.get("question", "")
    if needle.lower() not in q.lower():
        continue
    hits += 1
    print(f"\n--- {fp}")
    print(f"  question:   {q}")
    print(f"  found:      {r.get('found')}  cost: {r.get('cost')}  llm_calls: {r.get('llm_calls')}")
    a = r.get("answers") or {}
    print(f"  original:   {a.get('original')}")
    print(f"  perturbed:  {a.get('perturbed')}")
    print(f"  similarity: {a.get('similarity')}")
    print(f"  operations: {r.get('operations')}")
if hits == 0:
    print(f"No counterfactuals match '{needle}' under {scope}")
else:
    print(f"\n{hits} match(es).")
PY
        ;;

    baseline)
        if [[ ! -f benchmark/results/comparison.json ]]; then
            echo "benchmark/results/comparison.json not found — run Steps 0-2 first."
            exit 1
        fi
        "${UV_RUN[@]}" - <<'PY'
import json
with open("benchmark/results/comparison.json") as f:
    d = json.load(f)
s = d.get("summary", {})
print("baseline summary (benchmark/results/comparison.json):")
for k in ("total", "tt", "tf", "ft", "ff", "llm_accuracy", "rag_accuracy"):
    print(f"  {k:<14} {s.get(k)}")
PY
        ;;

    *)
        echo "Unknown command: $cmd" >&2
        echo "usage:" >&2
        sed -n '/^# Modes:/,/^$/p' "$0" | sed 's/^# *//' >&2
        exit 2
        ;;
esac
