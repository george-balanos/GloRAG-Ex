"""Run correctness for every method on all benchmarks and emit ONE merged LaTeX table.

Python equivalent of run_correctness.sh: scores each method in all_results/ on the two
ground-truth benchmarks (HotpotQA, musique; synthetic has no GT facts), writes a per-run
correctness JSON under all_results/correctness/, and renders a single table that merges
the hit-rate and precision results.

Cell convention in the merged table:
  - GloRAG-Ex / +PSP : a single value computed over the returned edit set (all operations)
                       -- Hit = >=1 edited element is a fact; Prec = TP/(TP+FP).
  - ranking baselines: the top-k curve "@1/@2/@3/@5" (Hit@k and Precision@k).

Matching (entity/relation/span -> supporting fact) is whatever src.correctness.agreement
implements (token-overlap or the LLM-judge variant); this orchestrator is matching-agnostic.

Run from code/ (PYTHONPATH=code):
  ../.venv/bin/python -m src.correctness.run_all
  ../.venv/bin/python -m src.correctness.run_all --judge
  ../.venv/bin/python -m src.correctness.run_all --match name+desc --tex /path/out.tex
"""
import argparse
import csv
import json
import os

from src.correctness.evaluate import (KS, aggregate, build_desc_provider, eval_attribution,
                                       eval_glorag, eval_ragex, load_facts, set_backend)

_THIS = os.path.abspath(__file__)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))  # .../GloRAG-Ex
_CODE = os.path.join(_REPO, "code")
_AR = os.path.join(_REPO, "all_results")

DATASETS = ["hotpotqa", "musique"]
DS_LABEL = {"hotpotqa": "Open-domain QA", "musique": "Multi-hop QA"}

# (tag, label, kind, relpath-template-under-all_results). kind: glorag|attribution|ragex.
JOBS = [
    ("our_ff",          "GloRAG-Ex",            "glorag",      "results_our/{ds}/all_ops_ff"),
    ("our_ft",          "GloRAG-Ex",            "glorag",      "results_our/{ds}/delete_ops_ft"),
    ("our_psp_ft",      "GloRAG-Ex + PSP",      "glorag",      "results_our/psp/{ds}/delete_ops_ft"),
    ("shap_ff",         "Shapley-RAG",          "attribution", "results_shap/{ds}/shap_ff.json"),
    ("shap_ft",         "Shapley-RAG",          "attribution", "results_shap/{ds}/shap_ft.json"),
    ("kgsmile_ff",      "KG-SMILE",             "attribution", "results_kg_smile/kg_smile_{ds}_ff.json"),
    ("kgsmile_ft",      "KG-SMILE",             "attribution", "results_kg_smile/kg_smile_{ds}_ft.json"),
    ("ragex_sentence",  "RAG-Ex (sentence)",    "ragex",       "results_rag_ex/{ds}/{ds}_remove_sentence_analysis.json"),
    ("ragex_paragraph", "RAG-Ex (paragraph)",   "ragex",       "results_rag_ex/{ds}/{ds}_remove_paragraph_analysis.json"),
    ("shaptext_chunk",  "Shapley-Text (chunk)", "ragex",       "results_shap_text/{ds}/{ds}_chunk_analysis.json"),
    ("shaptext_sent",   "Shapley-Text (sent.)", "ragex",       "results_shap_text/{ds}/{ds}_sentence_analysis.json"),
]
OURS = {"our_ff", "our_ft", "our_psp_ft"}
LABEL = {tag: lab for tag, lab, _, _ in JOBS}
KIND = {tag: kind for tag, _, kind, _ in JOBS}
# Which tags supply each direction's rows (baselines first, then ours -- paper order).
ROW_ORDER = {
    "T->F": ["shap_ft", "kgsmile_ft", "ragex_sentence", "ragex_paragraph",
             "shaptext_chunk", "shaptext_sent", "our_ft", "our_psp_ft"],
    "F->T": ["shap_ff", "kgsmile_ff", "ragex_sentence", "ragex_paragraph",
             "shaptext_chunk", "shaptext_sent", "our_ff"],
}


def run_one(tag, kind, path, facts, q2id, dataset, match, desc_provider):
    if kind == "glorag":
        return eval_glorag(path, facts, q2id, match, desc_provider=desc_provider) \
            if os.path.isdir(path) else None
    if kind == "attribution":
        return eval_attribution(path, facts, q2id, match=match, dataset=dataset,
                                desc_provider=desc_provider) if os.path.isfile(path) else None
    if kind == "ragex":
        return eval_ragex(path, facts, q2id) if os.path.isfile(path) else None
    return None


def _pct(x):
    return "--" if x is None else f"{x * 100:.1f}"


def _ours_cell(block, field):
    return _pct(block.get(field))


def _rank_cell(block, prefix):
    vals = [block.get(f"{prefix}{k}") for k in KS]
    return "--" if all(v is None for v in vals) else "/".join(_pct(v) for v in vals)


def build_merged_table(summ):
    """summ[(ds, tag)] = aggregate-summary dict. Returns a standalone LaTeX document."""
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{caption}",
         r"\label{tab:correctness}", r"\small", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{l l l c c}", r"\toprule",
         r"Dir. & Dataset & Method & Hit rate (\%) $\uparrow$ & Precision (\%) $\uparrow$ \\",
         r"\midrule"]
    for di, direc in enumerate(("T->F", "F->T")):
        dlabel = r"$T\!\to\!F$" if direc == "T->F" else r"$F\!\to\!T$"
        # rows present per dataset for this direction
        present = {ds: [t for t in ROW_ORDER[direc]
                        if (ds, t) in summ and summ[(ds, t)].get(direc, {}).get("n")]
                   for ds in DATASETS}
        n_dir = sum(len(present[ds]) for ds in DATASETS)
        if n_dir == 0:
            continue
        first_dir = True
        for ds in DATASETS:
            tags = present[ds]
            first_ds = True
            for t in tags:
                b = summ[(ds, t)][direc]
                if t in OURS:
                    # micro precision = pooled TP/(TP+FP) over all flagged edits
                    # (macro-mean collapses to hit-rate on single-op instances).
                    hit, prec = _ours_cell(b, "hit_rate"), _ours_cell(b, "precision_micro")
                else:
                    hit, prec = _rank_cell(b, "Hit@"), _rank_cell(b, "P@")
                dc = rf"\multirow{{{n_dir}}}{{*}}{{{dlabel}}}" if first_dir else ""
                sc = rf"\multirow{{{len(tags)}}}{{*}}{{{DS_LABEL[ds]}}}" if first_ds else ""
                L.append(f"{dc} & {sc} & {LABEL[t]} & {hit} & {prec} " + r"\\")
                first_dir = first_ds = False
            if ds != DATASETS[-1] and tags:
                L.append(r"\cmidrule(l){2-5}")
        if di == 0:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    table = "\n".join(L)
    doc = ["\\documentclass{article}", "\\usepackage{booktabs}", "\\usepackage{multirow}",
           "\\usepackage[margin=0.6in,landscape]{geometry}", "\\pagestyle{empty}",
           "\\begin{document}", "", table, "", "\\end{document}", ""]
    return "\n".join(doc), table


def write_csv(summ, path):
    cols = ["dataset", "direction", "method", "tag", "n", "hit_rate", "precision",
            *(f"Hit@{k}" for k in KS), *(f"P@{k}" for k in KS)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for (ds, tag), s in summ.items():
            for direc in ("T->F", "F->T", "overall"):
                b = s.get(direc)
                if not b or not b.get("n"):
                    continue
                w.writerow({"dataset": ds, "direction": direc, "method": LABEL[tag], "tag": tag,
                            "n": b["n"], "hit_rate": b.get("hit_rate"), "precision": b.get("precision"),
                            **{f"Hit@{k}": b.get(f"Hit@{k}") for k in KS},
                            **{f"P@{k}": b.get(f"P@{k}") for k in KS}})


def main():
    p = argparse.ArgumentParser(description="Run correctness for all methods and emit a merged LaTeX table.")
    p.add_argument("--match", choices=["name", "name+desc"], default="name+desc")
    p.add_argument("--outdir", default=os.path.join(_AR, "correctness"))
    p.add_argument("--tex", default=None, help="merged table output (default: <outdir>/correctness_table.tex).")
    p.add_argument("--csv", default=None, help="flat summary CSV (default: <outdir>/correctness_merged.csv).")

    # Matcher backend. The LLM judge is canonical (produces the paper numbers);
    # --heuristic selects the fast string+embedding fallback.
    p.add_argument("--judge", dest="judge", action="store_true", default=False,
                   help="(default) use the LLM judge (agreement_judge).")
    p.add_argument("--heuristic", dest="judge", action="store_false",
                   help="use the string-heuristic fallback (agreement) instead of the judge.")

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_backend(args.judge)

    # Canonical (judge) output uses the plain name; the heuristic fallback is suffixed.
    suffix = "" if args.judge else "_heur"
    tex_out = args.tex or os.path.join(args.outdir, f"correctness_table{suffix}.tex")
    csv_out = args.csv or os.path.join(args.outdir, f"correctness_merged{suffix}.csv")

    # GloRAG result dirs feed the shared description provider (graphml fallback).
    glorag_dirs_tmpl = [tmpl for _, _, kind, tmpl in JOBS if kind == "glorag"]

    summ = {}
    for ds in DATASETS:
        facts_path = os.path.join(_CODE, "datasets", ds, f"supporting_facts_{ds}.json")
        if not os.path.exists(facts_path):
            print(f"[skip {ds}] no facts at {facts_path}")
            continue
        facts, q2id = load_facts(facts_path)
        glorag_dirs = [os.path.join(_AR, t.format(ds=ds)) for t in glorag_dirs_tmpl]
        desc_provider = build_desc_provider(ds, glorag_dirs)
        print(f"\n#### {ds}: {len(facts)} GT facts (match={args.match}, judge={args.judge}) "
              f"| desc provider: {len(desc_provider)} elements")
        for tag, label, kind, tmpl in JOBS:
            path = os.path.join(_AR, tmpl.format(ds=ds))
            per = run_one(tag, kind, path, facts, q2id, ds, args.match, desc_provider)
            if per is None:
                continue
            method = "glorag" if kind == "glorag" else ("ragex" if kind == "ragex" else "attribution")
            summary = aggregate(per, method)
            summary["_meta"] = {"method": method, "dataset": ds, "tag": tag, "match": args.match, "used_judge": args.judge}
            summ[(ds, tag)] = summary
            
            # Apply suffix to individual JSONs
            json_out = os.path.join(args.outdir, f"{ds}_{tag}_correctness{suffix}.json")
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump({**per, "__summary__": summary}, f, indent=2, ensure_ascii=False)

    if not summ:
        raise SystemExit("No results scored -- check that all_results/ and the GT facts files exist.")

    doc, _table = build_merged_table(summ)
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write(doc)
    write_csv(summ, csv_out)
    print(f"\nScored {len(summ)} (dataset, method) runs.")
    print(f"-> merged table : {tex_out}   (standalone; `pdflatex {os.path.basename(tex_out)}`)")
    print(f"-> summary csv  : {csv_out}")


if __name__ == "__main__":
    main()