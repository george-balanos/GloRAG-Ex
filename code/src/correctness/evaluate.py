"""Correctness evaluation: precision of flagged elements vs ground-truth facts.

Post-hoc and offline -- reads saved artifacts only (no LLM / GPU / LightRAG). One
precision metric (src.correctness.agreement), applied to every method:

  --method glorag      : GloRAG-Ex counterfactuals (dir of counterfactual_*.json).
                         flagged = ops objects; precision = TP/(TP+FP).
  --method attribution : Shapley / KG-SMILE (id- or index-keyed json with a
                         per-object score map). precision@k over the top-k objects
                         restricted to score > --shap-min-score.
  --method ragex       : RAG-Ex ({cases:[...]}); flagged = removed text spans ranked
                         by importance; precision@k with span<->fact text matching.

Alongside precision we report graph fact-coverage: how many ground-truth facts each
graph carries (GloRAG original AND perturbed subgraphs; the retrieved graph for
attribution; the original context for RAG-Ex), at two granularities (grounded
elements and covered supporting facts).

Join is by question (universal) with an id fast-path: rid = key if key in facts
else q2id[norm(question)]. Aggregated per flip direction (T->F / F->T) and overall.

Run from code/ (PYTHONPATH=code), e.g.:
  ../.venv/bin/python -m src.correctness.evaluate --method attribution \
      --dataset hotpotqa --facts datasets/hotpotqa/supporting_facts_hotpotqa.json \
      --results all_results/results_shap/hotpotqa/shap_ft.json
"""
import argparse
import glob
import json
import os

from src.correctness.agreement import (fact_coverage, gt_relevant_set, id_relevant, normalized_contains,
                                        parse_id, precision_at_k, set_precision, span_relevant, tokens)

KS = (1, 2, 3, 5)
DIRECTION = {"ft": "T->F", "ff": "F->T", "tf": "F->T"}  # generate.py mode / case_type -> flip direction
ADD_MODES = {"ff", "tf"}


def _norm_q(q: str) -> str:
    return " ".join((q or "").lower().split())


def load_facts(path: str):
    with open(path, encoding="utf-8") as f:
        facts = json.load(f)
    q2id = {_norm_q(v.get("question", "")): rid for rid, v in facts.items()}
    return facts, q2id


def _resolve_id(key, question, facts, q2id):
    """Universal join: prefer the record key if it is a facts id, else map by question."""
    if key in facts:
        return key
    return q2id.get(_norm_q(question))


def _supporting_units(fact_rec) -> list[str]:
    """Per-fact gold texts: HotpotQA supporting sentences, else musique paragraphs, else gold_text."""
    ss = fact_rec.get("supporting_sentences")
    if ss:
        return ss
    sp = fact_rec.get("supporting_paragraphs")
    if sp:
        return [p.get("text", "") for p in sp]
    gt = fact_rec.get("gold_text", "")
    return [gt] if gt else []


def ids_from_operations(ops) -> list[str]:
    """Map counterfactual operations to flagged element ids (in op/cost order)."""
    out = []
    for op in ops or []:
        if not isinstance(op, list) or len(op) < 2:
            continue
        typ, arg = op[0], op[1]
        if "node" in typ:
            out.append(f"E::{arg}")
        elif "edge" in typ and isinstance(arg, (list, tuple)) and len(arg) == 2:
            out.append(f"R::{arg[0]}->{arg[1]}")
    return list(dict.fromkeys(out))


def _elements_from_subgraph(sg: dict) -> list[str]:
    ents = [f"E::{e.get('name', '')}" for e in (sg or {}).get("entities", [])]
    rels = [f"R::{r.get('src', '')}->{r.get('tgt', '')}" for r in (sg or {}).get("relations", [])]
    return ents + rels


def _desc_from_subgraph(sg: dict) -> dict:
    d = {}
    for e in (sg or {}).get("entities", []):
        d[f"E::{e.get('name', '')}"] = e.get("description", "")
    for r in (sg or {}).get("relations", []):
        d[f"R::{r.get('src', '')}->{r.get('tgt', '')}"] = r.get("description", "")
    return d


def _load_kg_graph(dataset: str, kg_graph: str | None):
    import networkx as nx
    path = kg_graph
    if path is None:
        from src.dataset_setup import WORKING_DIRS
        path = os.path.join(WORKING_DIRS.get(dataset, ""), "graph_chunk_entity_relation.graphml")
    if not path or not os.path.exists(path):
        print(f"  [name+desc] KG graph not found ({path!r}); descriptions unavailable -> name-only match.")
        return None
    return nx.read_graphml(path)


def _desc_from_graph(G, ids) -> dict:
    if G is None:
        return {}
    d = {}
    for eid in ids:
        kind, payload = parse_id(eid)
        if kind == "entity":
            if G.has_node(payload):
                d[eid] = G.nodes[payload].get("description", "")
        else:
            s, t = payload
            if G.has_edge(s, t):
                d[eid] = G.edges[s, t].get("description", "")
            elif G.has_edge(t, s):
                d[eid] = G.edges[t, s].get("description", "")
    return d


def _file_direction(path, override):
    """Flip direction for a per-direction file (attribution); ff->F->T, ft->T->F."""
    if override and override != "auto":
        return DIRECTION.get(override, override)
    n = os.path.basename(path).lower()
    if "ff" in n:
        return "F->T"
    if "ft" in n:
        return "T->F"
    return "any"


# ── GloRAG-Ex ────────────────────────────────────────────────────────────────
def eval_glorag(input_dir, facts, q2id, match="name", desc_ngram=3):
    use_desc = match == "name+desc"
    files = sorted(glob.glob(os.path.join(input_dir, "**", "counterfactual_*.json"), recursive=True))
    per_instance, unmatched, skipped = {}, 0, 0
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"  skip {fp}: {e}")
            continue
        if not payload.get("found"):
            skipped += 1
            continue
        rid = _resolve_id(None, payload.get("question", ""), facts, q2id)
        if rid is None or rid not in facts:
            unmatched += 1
            continue

        gold = facts[rid]["gold_text"]
        gold_tokens = tokens(gold)
        units = _supporting_units(facts[rid])
        desc_by_id = None
        if use_desc:
            desc_by_id = {**_desc_from_subgraph(payload.get("original_subgraph")),
                          **_desc_from_subgraph(payload.get("perturbed_subgraph"))}
        flagged = ids_from_operations(payload.get("operations"))
        gold_set = {f for f in flagged if id_relevant(f, gold_tokens, desc_by_id, desc_ngram)}
        s = set_precision(flagged, gold_set)
        # P@k over the cost-ordered edits (all edits are "flagged"/positive), so the
        # precision@k column is comparable to the ranking baselines. For k >= |edits|
        # this saturates to the set precision above.
        s.update(precision_at_k(flagged, set(flagged), gold_set, KS))

        # graph fact-coverage: how many GT facts the original vs perturbed graph carries
        orig_ids = _elements_from_subgraph(payload.get("original_subgraph"))
        pert_ids = _elements_from_subgraph(payload.get("perturbed_subgraph"))
        s["orig_graph"] = fact_coverage(orig_ids, units, gold, desc_by_id, desc_ngram)
        s["pert_graph"] = fact_coverage(pert_ids, units, gold, desc_by_id, desc_ngram)

        mode = payload.get("mode", "")
        s.update({"mode": mode, "direction": DIRECTION.get(mode, "?"),
                  "is_add_mode": mode in ADD_MODES, "flagged": flagged, "file": os.path.basename(fp)})
        per_instance[rid] = s
    print(f"  glorag: {len(per_instance)} scored | unmatched={unmatched} | not-found-skipped={skipped}")
    return per_instance


# ── Attribution (Shapley / KG-SMILE) ─────────────────────────────────────────
def eval_attribution(results_path, facts, q2id, score_field="auto", direction="auto",
                     match="name", desc_ngram=3, shap_min_score=None, kg_graph=None, dataset="hotpotqa"):
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
    G = _load_kg_graph(dataset, kg_graph) if match == "name+desc" else None
    file_dir = _file_direction(results_path, direction)
    per_instance, unmatched, empty = {}, 0, 0
    for key, rec in results.items():
        if str(key).startswith("__") or not isinstance(rec, dict):
            continue
        scores = rec.get(score_field) if score_field not in (None, "auto") else \
            (rec.get("shapley_scores") or rec.get("scores"))
        rid = _resolve_id(key, rec.get("question", ""), facts, q2id)
        if rid is None or rid not in facts:
            unmatched += 1
            continue
        if not scores:
            empty += 1
            continue
        gold = facts[rid]["gold_text"]
        units = _supporting_units(facts[rid])
        ranked = sorted(scores, key=lambda o: scores[o], reverse=True)
        # no score filter by default: top-k is purely by rank (the #1 is the highest score,
        # even if negative). --shap-min-score optionally restricts to score > threshold.
        positive_ids = set(scores) if shap_min_score is None else {o for o, v in scores.items() if v > shap_min_score}
        desc_by_id = _desc_from_graph(G, ranked) if G is not None else None
        gold_set = gt_relevant_set(ranked, gold, desc_by_id, desc_ngram)
        s = precision_at_k(ranked, positive_ids, gold_set, KS)
        s["orig_graph"] = fact_coverage(ranked, units, gold, desc_by_id, desc_ngram)
        s.update({"mode": "attr", "direction": file_dir,
                  "n_universe": len(ranked), "n_positive": len(positive_ids)})
        per_instance[rid] = s
    print(f"  attribution: {len(per_instance)} scored | unmatched={unmatched} | empty-scores={empty} | dir={file_dir}")
    return per_instance


# ── RAG-Ex (text spans) ──────────────────────────────────────────────────────
def eval_ragex(results_path, facts, q2id, span_jaccard=0.5, min_weight=None):
    analysis = json.load(open(results_path, encoding="utf-8"))
    cases = analysis.get("cases", [])
    per_instance, unmatched = {}, 0
    for c in cases:
        rid = _resolve_id(c.get("case_id"), c.get("question", ""), facts, q2id)
        if rid is None or rid not in facts:
            unmatched += 1
            continue
        units = _supporting_units(facts[rid])
        imp = c.get("removed_item_importance") or {}
        ranked = sorted(imp, key=lambda sp: imp[sp], reverse=True)            # spans by importance
        # no filter by default: top-k spans purely by importance rank (greatest first).
        positive_ids = set(imp) if min_weight is None else {sp for sp, w in imp.items() if w > min_weight}
        gold_set = {sp for sp in ranked if span_relevant(sp, units, span_jaccard)}
        s = precision_at_k(ranked, positive_ids, gold_set, KS)

        ctx = c.get("original_context", "")
        covered = sum(1 for u in units if normalized_contains(ctx, u))
        s["orig_graph"] = {"n_elements": len(imp), "n_gold_elements": len(gold_set),
                           "n_facts_covered": covered, "n_facts_total": len(units)}
        direction = c.get("mapped_label") or DIRECTION.get(c.get("case_type", ""), "?")
        s.update({"mode": "ragex", "direction": direction, "n_universe": len(imp),
                  "n_positive": len(positive_ids), "granularity": c.get("method", "")})
        per_instance[rid] = s
    print(f"  ragex: {len(per_instance)} scored | unmatched={unmatched}")
    return per_instance


# ── Aggregation ──────────────────────────────────────────────────────────────
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _coverage_means(items, gkey, prefix):
    blocks = [it[gkey] for it in items if gkey in it]
    if not blocks:
        return {}
    out = {f"{prefix}_gold_elems": _mean([b["n_gold_elements"] for b in blocks]),
           f"{prefix}_facts_cov": _mean([b["n_facts_covered"] for b in blocks])}
    ratios = [b["n_facts_covered"] / b["n_facts_total"] for b in blocks if b["n_facts_total"]]
    out[f"{prefix}_cov_ratio"] = round(sum(ratios) / len(ratios), 4) if ratios else None
    if prefix == "orig":
        out["facts_total"] = _mean([b["n_facts_total"] for b in blocks])
    return out


def _glorag_block(items):
    if not items:
        return {"n": 0}
    tp = sum(it["tp"] for it in items)
    fp = sum(it["fp"] for it in items)
    return {"n": len(items),
            "precision": _mean([it["precision"] for it in items]),
            "precision_micro": round(tp / (tp + fp), 4) if (tp + fp) else None,
            "hit_rate": round(sum(1 for it in items if it["tp"] > 0) / len(items), 4),  # >=1 edit is a fact
            "tp_total": tp, "fp_total": fp,
            **{f"P@{k}": _mean([it["precision_at"].get(str(k)) for it in items]) for k in KS},
            **_coverage_means(items, "orig_graph", "orig"),
            **_coverage_means(items, "pert_graph", "pert")}


def _pk_block(items):
    if not items:
        return {"n": 0}
    out = {"n": len(items)}
    for k in KS:
        out[f"P@{k}"] = _mean([it["precision_at"].get(str(k)) for it in items])       # precision@k
        out[f"Hit@{k}"] = _mean([it["hit_at"].get(str(k)) for it in items])           # any-fact-in-top-k rate
    out.update(_coverage_means(items, "orig_graph", "orig"))
    return out


def aggregate(per_instance, method) -> dict:
    block = _glorag_block if method == "glorag" else _pk_block
    items = list(per_instance.values())
    summary = {"overall": block(items)}
    for direction in ("T->F", "F->T"):
        summary[direction] = block([it for it in items if it.get("direction") == direction])
    return summary


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else "  -  "


def _print_table(method, dataset, summary):
    print("\n" + "=" * 84)
    print(f"  Correctness (text-mention, precision)  method={method} dataset={dataset}")
    print("=" * 84)
    if method == "glorag":
        hdr = f"{'Dir':<8}{'n':>5}  {'Prec':>7}{'micro':>7}{'TP':>6}{'FP':>6}  {'orig_cov':>9}{'pert_cov':>9}{'facts':>7}"
        cov = lambda b, p: f"{b.get(p+'_facts_cov') or 0:.2f}"
        print(hdr); print("-" * len(hdr))
        for key in ("overall", "T->F", "F->T"):
            b = summary.get(key, {"n": 0})
            if not b.get("n"):
                print(f"{key:<8}{0:>5}"); continue
            print(f"{key:<8}{b['n']:>5}  {_fmt(b.get('precision')):>7}{_fmt(b.get('precision_micro')):>7}"
                  f"{b.get('tp_total', 0):>6}{b.get('fp_total', 0):>6}  "
                  f"{cov(b,'orig'):>9}{cov(b,'pert'):>9}{_fmt(b.get('facts_total')):>7}")
    else:
        hdr = f"{'Dir':<8}{'n':>5}  {'P@1':>7}{'P@2':>7}{'P@3':>7}{'P@5':>7}  {'orig_cov':>9}{'facts':>7}"
        print(hdr); print("-" * len(hdr))
        for key in ("overall", "T->F", "F->T"):
            b = summary.get(key, {"n": 0})
            if not b.get("n"):
                print(f"{key:<8}{0:>5}"); continue
            oc = f"{b.get('orig_facts_cov') or 0:.2f}"
            print(f"{key:<8}{b['n']:>5}  {_fmt(b.get('P@1')):>7}{_fmt(b.get('P@2')):>7}"
                  f"{_fmt(b.get('P@3')):>7}{_fmt(b.get('P@5')):>7}  {oc:>9}{_fmt(b.get('facts_total')):>7}")
    print("=" * 84)


def main():
    p = argparse.ArgumentParser(description="Precision of flagged elements vs GT supporting facts.")
    p.add_argument("--method", choices=["glorag", "attribution", "ragex"], required=True)
    p.add_argument("--dataset", default="hotpotqa")
    p.add_argument("--facts", required=True, help="id-keyed supporting-facts JSON (datasets/build_*.py output).")
    p.add_argument("--input-dir", help="[glorag] dir of counterfactual_*.json (searched recursively).")
    p.add_argument("--results", help="[attribution/ragex] results json.")
    p.add_argument("--shap-results", help="alias of --results (attribution).")
    p.add_argument("--score-field", default="auto", help="[attribution] score map field; auto = shapley_scores|scores.")
    p.add_argument("--direction", default="auto", choices=["auto", "ff", "ft", "tf"],
                   help="[attribution] flip direction for a per-direction file (auto = infer from filename).")
    p.add_argument("--shap-min-score", type=float, default=None,
                   help="[attribution] if set, only objects with score strictly greater than this count "
                        "(default: no filter -- top-k purely by rank, #1 is the highest score).")
    p.add_argument("--span-jaccard", type=float, default=0.5, help="[ragex] span<->fact token-set Jaccard threshold.")
    p.add_argument("--min-weight", type=float, default=None,
                   help="[ragex] if set, only spans with importance > this count (default: no filter).")
    p.add_argument("--match", choices=["name", "name+desc"], default="name")
    p.add_argument("--desc-ngram", type=int, default=3)
    p.add_argument("--kg-graph", default=None, help="[attribution, name+desc] graphml for description lookup.")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    facts, q2id = load_facts(args.facts)
    print(f"GT facts: {len(facts)} questions  (dataset={args.dataset}, method={args.method}, match={args.match})")
    results = args.results or args.shap_results

    if args.method == "glorag":
        if not args.input_dir:
            raise SystemExit("--input-dir is required for --method glorag")
        per_instance = eval_glorag(args.input_dir, facts, q2id, args.match, args.desc_ngram)
    elif args.method == "attribution":
        if not results:
            raise SystemExit("--results is required for --method attribution")
        per_instance = eval_attribution(results, facts, q2id, args.score_field, args.direction,
                                        args.match, args.desc_ngram, args.shap_min_score,
                                        args.kg_graph, args.dataset)
    else:  # ragex
        if not results:
            raise SystemExit("--results is required for --method ragex")
        per_instance = eval_ragex(results, facts, q2id, args.span_jaccard, args.min_weight)

    summary = aggregate(per_instance, args.method)
    summary["_meta"] = {"method": args.method, "dataset": args.dataset, "facts": args.facts,
                        "match": args.match, "desc_ngram": args.desc_ngram, "ks": list(KS),
                        "results": results, "shap_min_score": args.shap_min_score}
    out_obj = dict(per_instance)
    out_obj["__summary__"] = summary

    out = args.output or f"benchmark/results/{args.dataset}_{args.method}_correctness.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    _print_table(args.method, args.dataset, summary)
    print(f"Results -> {out}")


if __name__ == "__main__":
    main()
