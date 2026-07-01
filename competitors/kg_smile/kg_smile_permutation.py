from __future__ import annotations

import argparse
import asyncio
import glob
import itertools
import json
import os
import random
import sys
import time
import networkx as nx
import numpy as np
from scipy.stats import kendalltau

# ── make repo importable (same pattern as run_shapley.py) ───────────────────
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _THIS_DIR  # adjust if this file moves deeper
_CODE_DIR  = os.path.join(_REPO_ROOT, "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src.retrieve import initialize_lightrag
from src.dataset_setup import WORKING_DIRS, QA_CSV_PATHS, DATASETS
from tqdm import tqdm

from kg_smile.kg_smile import (          # type: ignore
    KGSMILEConfig,
    KGSMILEResult,
    load_full_kg,
    run_kg_smile,
    _extract_triples,
    _build_perturbed_graph,
    _query as kg_smile_query,
)
from kg_smile.runner import (            # type: ignore
    _load_questions_from_explanation_dir,
)
from kg_smile.io_utils import (          # type: ignore
    load_questions_from_csv,
    load_completed,
)


# ── edge / node id helpers ───────────────────────────────────────────────────

# def edge_id(src: str, tgt: str) -> str:
#     """Stable id for a (src, tgt) edge — order-normalised so A->B == B->A."""
#     a, b = (src, tgt) if src <= tgt else (tgt, src)
#     return f"E::{a}->{b}"

def edge_id(src: str, tgt: str) -> str:
    """Stable id for a (src, tgt) edge — order-normalised so A->B == B->A."""
    a, b = (src, tgt) if src <= tgt else (tgt, src)
    return f"R::{a}->{b}"

def triple_edge_id(triple: tuple[str, str, str]) -> str:
    src, _desc, tgt = triple
    return edge_id(src, tgt)


# ── permutation helpers ──────────────────────────────────────────────────────

def random_triple_permutations(
    triples: list[tuple[str, str, str]],
    count:   int,
    seed:    int,
) -> list[dict]:
    """
    Generate `count` distinct random permutations of `triples`.

    Returns a list of dicts:
        {
            "perm_id": "perm_0" | ...,
            "perm":    [int, ...],          # index permutation
            "triples": [(src, desc, tgt), ...],
        }
    """
    rng = random.Random(seed)
    n   = len(triples)
    results, seen = [], set()
    attempts = 0
    max_attempts = count * 20

    while len(results) < count and attempts < max_attempts:
        attempts += 1
        idx   = list(range(n))
        rng.shuffle(idx)
        key   = tuple(idx)
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "perm_id": f"perm_{len(results)}",
            "perm":    idx,
            "triples": [triples[i] for i in idx],
        })

    return results


# ── ranking / stability statistics (identical logic to Shapley benchmark) ───

def permutation_stats(
    scores_by_perm: dict[str, dict[str, float]],
    ids:            list[str],
    top_k:          int,
) -> dict:
    """
    scores_by_perm : perm_id -> {object_id -> attribution value}
    ids            : canonical ordered list of object ids
    top_k          : k for top-k stability check

    Returns the same statistics dict shape as the Shapley benchmark so that
    downstream comparison / evaluation scripts remain unchanged.
    """
    perm_ids = list(scores_by_perm.keys())

    # ── per-object spread across permutations ────────────────────────────────
    per_object: dict[str, dict] = {}
    for oid in ids:
        vals = np.array([scores_by_perm[p].get(oid, 0.0) for p in perm_ids], dtype=float)
        per_object[oid] = {
            "mean":  float(vals.mean()),
            "std":   float(vals.std()),
            "min":   float(vals.min()),
            "max":   float(vals.max()),
            "range": float(vals.max() - vals.min()),
        }

    # ── ranking per permutation (desc by absolute attribution) ───────────────
    # KG-SMILE coefficients can be negative; rank by absolute value so the
    # "most important" edge is top regardless of sign — matching top_edges logic.
    rankings = {
        p: sorted(ids, key=lambda o: abs(scores_by_perm[p].get(o, 0.0)), reverse=True)
        for p in perm_ids
    }
    rank_index = {p: {oid: i for i, oid in enumerate(rankings[p])} for p in perm_ids}

    # ── pairwise Kendall-tau ─────────────────────────────────────────────────
    taus: list[float] = []
    exact_match = True
    if len(perm_ids) >= 2 and len(ids) >= 2:
        for a, b in itertools.combinations(perm_ids, 2):
            ra = [rank_index[a][o] for o in ids]
            rb = [rank_index[b][o] for o in ids]
            tau, _ = kendalltau(ra, rb)
            taus.append(float(tau) if tau == tau else float("nan"))
            if rankings[a] != rankings[b]:
                exact_match = False

    mean_tau = float(np.nanmean(taus)) if taus else float("nan")
    min_tau  = float(np.nanmin(taus))  if taus else float("nan")

    # ── top-k SET stability ──────────────────────────────────────────────────
    top1      = {rankings[p][0] for p in perm_ids} if ids else set()
    topk_sets = [frozenset(rankings[p][:top_k]) for p in perm_ids] if ids else []
    top1_stable  = len(top1) == 1
    topk_stable  = len(set(topk_sets)) == 1 if topk_sets else True

    # ── top-k POSITIONAL stability ───────────────────────────────────────────
    k_eff = min(top_k, len(ids))
    position_stable = [
        len({rankings[p][i] for p in perm_ids}) == 1
        for i in range(k_eff)
    ]

    return {
        "num_permutations":       len(perm_ids),
        "mean_kendall_tau":       mean_tau,
        "min_kendall_tau":        min_tau,
        "exact_ranking_match":    exact_match,
        "top1_stable":            top1_stable,
        f"top{top_k}_stable":     topk_stable,
        "topk_positions_checked": k_eff,
        "topk_position_matches":  int(sum(position_stable)),
        "topk_position_stable":   position_stable,
        "per_object":             per_object,
        "rankings":               rankings,
    }


# ── result → lightweight dict (edge attributions only, for permutation tracking)

def _attribution_map(result: KGSMILEResult) -> dict[str, float]:
    """Flatten edge_attributions into {edge_id: float}."""
    return {edge_id(s, t): v for (s, t), v in result.edge_attributions.items()}


# def _node_map(result: KGSMILEResult) -> dict[str, float]:
#     return dict(result.node_attributions)

def _node_map(result: KGSMILEResult) -> dict[str, float]:
    return {f"E::{node}": v for node, v in result.node_attributions.items()}

# ── graph helpers ─────────────────────────────────────────────────────────────

def _nx_graph_from_subgraph(og: dict) -> nx.Graph:
    """
    Build an nx.Graph from a stored original_subgraph dict so that
    _extract_triples / _build_perturbed_graph work correctly.

    Only the fields actually used by the KG-SMILE graph pipeline are stored:
      - Node attrs: type, description  (carried over by _build_perturbed_graph)
      - Edge attrs: description        (extracted by _extract_triples)
    rank, keywords, and weight are present in the source data but are never
    read by any downstream KG-SMILE function, so they are not stored.
    """
    G = nx.Graph()
    for e in (og or {}).get("entities") or []:
        G.add_node(e.get("name", ""), type=e.get("type", ""), description=e.get("description", ""))
    for r in (og or {}).get("relations") or []:
        src, tgt = r.get("src", ""), r.get("tgt", "")
        if src not in G:
            G.add_node(src)
        if tgt not in G:
            G.add_node(tgt)
        G.add_edge(src, tgt, description=r.get("description", ""))
    return G


# ── per-case runner ───────────────────────────────────────────────────────────

async def _run_case_permutation(
    question:        str,
    ground_truth:    str | None,
    original_answer: str,
    G:               nx.Graph,
    rag,
    kg_full:         nx.Graph,
    base_config:     KGSMILEConfig,
    num_perms:       int,
    topk_stable:     int,
    seed:            int,
) -> dict:
    """
    Run the full permutation robustness experiment for a single case.

    Mirrors run_permutation_from_json in the Shapley benchmark:
      - Original order: stored answer used as-is (ff mode, no generation).
      - Permuted orders: fresh answer generated per permuted context, then
        attributed against that new answer (ft mode).

    Returns the per-case result dict (without the top-level 'id' / 'question' wrapper).
    """
    all_triples = _extract_triples(G)
    all_ids     = [triple_edge_id(t) for t in all_triples]

    if not all_triples:
        return {
            "question": question, "ground_truth": ground_truth,
            "original_answer": original_answer,
            "n_triples": 0, "n_nodes": G.number_of_nodes(),
            "object_ids": [], "num_permutations": 0, "num_answer_changed": 0,
            "permutations": [], "stats": {}, "error": "no triples",
        }

    # Config with noise disabled — used for both original and permuted runs.
    cfg = KGSMILEConfig(
        n_perturbations=base_config.n_perturbations,
        kernel_width=base_config.kernel_width,
        retrieval_mode=base_config.retrieval_mode,
        retrieval_top_k=base_config.retrieval_top_k,
        random_seed=base_config.random_seed,
        max_tokens=base_config.max_tokens,
        embedding_model=base_config.embedding_model,
        noise_pct=0.0,
    )

    perm_records:   list[dict]               = []
    scores_by_perm: dict[str, dict[str, float]] = {}

    # ── original order → pin stored answer, no generation (ff mode) ─────────
    st0 = time.perf_counter()
    orig_result = await run_kg_smile(
        query=question, rag=rag, KG_full=kg_full,
        config=cfg, ground_truth=original_answer, mode="ff",
    )
    orig_elapsed = round(time.perf_counter() - st0, 4)

    orig_attr_map = _attribution_map(orig_result)
    orig_ranking  = sorted(all_ids, key=lambda o: abs(orig_attr_map.get(o, 0.0)), reverse=True)

    scores_by_perm["original"] = orig_attr_map
    perm_records.append({
        "perm_id":           "original",
        "perm":              list(range(len(all_triples))),
        "triple_order":      all_ids,
        "target_answer":     original_answer,
        "generated":         False,
        "answer_changed":    False,
        "edge_attributions": orig_attr_map,
        "node_attributions": _node_map(orig_result),
        "ranking":           orig_ranking,
        "surrogate_r2":      orig_result.surrogate_r2,
        "degenerate":        orig_result.degenerate,
        "llm_call_count":    orig_result.llm_call_count,
        "shap_time":         orig_elapsed,
        "gen_time":          0.0,
    })

    # ── permuted orders → regenerate answer, then attribute it (ft mode) ─────
    perms = random_triple_permutations(all_triples, count=num_perms, seed=seed)

    for p in perms:
        perm_triples    = p["triples"]
        perm_triple_ids = [triple_edge_id(t) for t in perm_triples]

        # Reconstruct graph with edges in permuted order; node attrs carried over.
        surviving_entities: list[str] = []
        seen: set[str] = set()
        for (src, _d, tgt) in perm_triples:
            for node in (src, tgt):
                if node not in seen:
                    seen.add(node)
                    surviving_entities.append(node)
        G_perm = _build_perturbed_graph(perm_triples, surviving_entities, G)

        # Generate a fresh answer for this permuted context (mirrors Shapley).
        gt0        = time.perf_counter()
        new_answer = await kg_smile_query(question, G_perm, base_config.max_tokens)
        gen_time   = round(time.perf_counter() - gt0, 4)

        answer_changed = new_answer.strip() != (original_answer or "").strip()

        # Attribute the new answer against the permuted graph (ff mode, no re-retrieval).
        st0 = time.perf_counter()
        perm_result = await run_kg_smile(
            query=question, rag=rag, KG_full=kg_full,
            config=cfg, ground_truth=new_answer, mode="ff",
        )
        elapsed = round(time.perf_counter() - st0, 4)

        attr_map = _attribution_map(perm_result)
        ranking  = sorted(perm_triple_ids, key=lambda o: abs(attr_map.get(o, 0.0)), reverse=True)

        scores_by_perm[p["perm_id"]] = attr_map
        perm_records.append({
            "perm_id":           p["perm_id"],
            "perm":              p["perm"],
            "triple_order":      perm_triple_ids,
            "target_answer":     new_answer,
            "generated":         True,
            "answer_changed":    answer_changed,
            "edge_attributions": attr_map,
            "node_attributions": _node_map(perm_result),
            "ranking":           ranking,
            "surrogate_r2":      perm_result.surrogate_r2,
            "degenerate":        perm_result.degenerate,
            "llm_call_count":    perm_result.llm_call_count,
            "shap_time":         elapsed,
            "gen_time":          gen_time,
        })

    stats = permutation_stats(scores_by_perm, all_ids, topk_stable)
    n_changed = sum(1 for r in perm_records if r["answer_changed"])

    return {
        "question":           question,
        "ground_truth":       ground_truth,
        "original_answer":    original_answer,
        "n_triples":          len(all_triples),
        "n_nodes":            G.number_of_nodes(),
        "object_ids":         all_ids,
        "num_permutations":   len(perm_records),
        "num_answer_changed": n_changed,
        "permutations":       perm_records,
        "stats":              stats,
        "perm_total_llm_calls": sum(r["llm_call_count"] for r in perm_records),
        "perm_total_shap_time": round(sum(r["shap_time"] for r in perm_records), 4),
    }


# ── CFE JSON loader (mirrors Shapley benchmark) ───────────────────────────────

def load_cf_cases(input_dir: str, questions: set[str] | None = None) -> list[tuple[str, dict]]:
    """
    Load (filepath, payload) pairs from CFE counterfactual_*.json files.
    Filters to cases with a non-empty original_subgraph.
    """
    files = sorted(
        glob.glob(os.path.join(input_dir, "**", "counterfactual_*.json"), recursive=True)
    )
    cases: list[tuple[str, dict]] = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"  skip {fp}: {e}")
            continue

        if questions is not None and payload.get("question") not in questions:
            continue

        og = payload.get("original_subgraph") or {}
        if (og.get("entities") or []) or (og.get("relations") or []):
            cases.append((fp, payload))
    return cases


def _case_id(fp: str) -> str:
    return os.path.splitext(os.path.basename(fp))[0]


# ── main entry points ─────────────────────────────────────────────────────────

async def run_permutation_from_json(args, rag, kg_full: nx.Graph, cases: list) -> None:
    """
    Permutation robustness from CFE JSONs.
    Original subgraph + original answer are read from the JSON (no retrieval).
    """
    results: dict[str, dict]   = {}
    tau_list, mintau_list       = [], []
    top1_list, topk_list        = [], []
    exact_list                  = []
    posmatch_list, poschecked_list, changed_list = [], [], []

    base_config = KGSMILEConfig(
        n_perturbations=args.n_pert,
        kernel_width=args.kernel_width,
        retrieval_top_k=args.top_k,
        random_seed=args.seed,
    )

    # for fp, payload in cases:
    for fp, payload in tqdm(cases, desc="Cases", unit="case"):
        rid      = _case_id(fp)
        question = payload["question"]
        gt       = payload["answers"].get("ground_truth")
        orig_ans = payload["answers"]["original"]
        og       = payload.get("original_subgraph") or {}
        G        = _nx_graph_from_subgraph(og)

        if G.number_of_edges() == 0:
            print(f"[{rid}] no edges; skipping.")
            continue

        print(f"\n[permutation] {rid}")
        case_result = await _run_case_permutation(
            question=question, ground_truth=gt, original_answer=orig_ans,
            G=G, rag=rag, kg_full=kg_full, base_config=base_config,
            num_perms=args.num_perms,
            topk_stable=args.topk_stable, seed=args.seed,
        )
        case_result["filepath"] = fp
        case_result["mode"]     = payload.get("mode")
        results[rid]            = case_result

        stats     = case_result["stats"]
        topk_key  = f"top{args.topk_stable}_stable"
        n_changed = case_result["num_answer_changed"]
        tau_list.append(stats["mean_kendall_tau"])
        mintau_list.append(stats["min_kendall_tau"])
        top1_list.append(stats["top1_stable"])
        topk_list.append(stats.get(topk_key, False))
        exact_list.append(stats["exact_ranking_match"])
        posmatch_list.append(stats["topk_position_matches"])
        poschecked_list.append(stats["topk_positions_checked"])
        changed_list.append(n_changed)

        print(
            f"[{rid}] perms={stats['num_permutations']} "
            f"answer_changed={n_changed}/{args.num_perms} "
            f"meanτ={stats['mean_kendall_tau']:.3f} "
            f"minτ={stats['min_kendall_tau']:.3f} "
            f"top1_stable={stats['top1_stable']} "
            f"exact={stats['exact_ranking_match']}"
        )

    _write_summary(results, tau_list, mintau_list, top1_list, topk_list,
                   exact_list, posmatch_list, poschecked_list, changed_list,
                   args)


async def run_permutation_from_csv(args, rag, kg_full: nx.Graph) -> None:
    """
    Permutation robustness from a QA CSV.
    Subgraph is retrieved via LightRAG; the original answer is generated fresh.
    """
    from src.retrieve import retrieve_subgraph_objects  # type: ignore
    from src.query import query as rag_query            # type: ignore
    import pandas as pd

    df = pd.read_csv(QA_CSV_PATHS[args.dataset]).drop_duplicates(subset=["questions"])
    if args.num_rows is not None:
        df = df.head(args.num_rows)

    results: dict[str, dict]                          = {}
    tau_list, mintau_list, top1_list, topk_list       = [], [], [], []
    exact_list, posmatch_list, poschecked_list        = [], [], []
    changed_list: list[int]                           = []

    base_config = KGSMILEConfig(
        n_perturbations=args.n_pert,
        kernel_width=args.kernel_width,
        retrieval_top_k=args.top_k,
        random_seed=args.seed,
    )

    for _, row in df.iterrows():
        rid      = str(row["id"])
        question = row["questions"]
        gt       = row.get("answers")

        # Retrieve subgraph as nx.Graph
        from src.retrieve import retrieve_subgraph_objects  # type: ignore
        context, sg = await retrieve_subgraph_objects(
            rag, query=question, mode=args.rag_mode, top_k=args.top_k)
        G = _nx_graph_from_subgraph({
            "entities": [
                {"name": e.name, "type": e.type,
                 "description": e.description, "rank": e.rank}
                for e in sg.entities
            ],
            "relations": [
                {"src": r.src, "tgt": r.tgt,
                 "keywords": r.keywords, "description": r.description,
                 "weight": r.weight}
                for r in sg.relations
            ],
        })

        if G.number_of_edges() == 0:
            print(f"[{rid}] no edges; skipping.")
            continue

        # Generate original answer via KG-SMILE's own query helper
        from kg_smile.kg_smile import graph_to_context  # type: ignore
        orig_ans = await kg_smile_query(question, G, base_config.max_tokens)

        print(f"\n[permutation] {rid}")
        case_result = await _run_case_permutation(
            question=question, ground_truth=gt, original_answer=orig_ans,
            G=G, rag=rag, kg_full=kg_full, base_config=base_config,
            num_perms=args.num_perms,
            topk_stable=args.topk_stable, seed=args.seed,
        )
        results[rid] = case_result

        stats     = case_result["stats"]
        topk_key  = f"top{args.topk_stable}_stable"
        n_changed = case_result["num_answer_changed"]
        tau_list.append(stats["mean_kendall_tau"])
        mintau_list.append(stats["min_kendall_tau"])
        top1_list.append(stats["top1_stable"])
        topk_list.append(stats.get(topk_key, False))
        exact_list.append(stats["exact_ranking_match"])
        posmatch_list.append(stats["topk_position_matches"])
        poschecked_list.append(stats["topk_positions_checked"])
        changed_list.append(n_changed)

        print(
            f"[{rid}] perms={stats['num_permutations']} "
            f"answer_changed={n_changed}/{args.num_perms} "
            f"meanτ={stats['mean_kendall_tau']:.3f} "
            f"top1_stable={stats['top1_stable']}"
        )

    _write_summary(results, tau_list, mintau_list, top1_list, topk_list,
                   exact_list, posmatch_list, poschecked_list, changed_list,
                   args)


# ── shared output writer ──────────────────────────────────────────────────────

def _write_summary(
    results:         dict,
    tau_list:        list[float],
    mintau_list:     list[float],
    top1_list:       list[bool],
    topk_list:       list[bool],
    exact_list:      list[bool],
    posmatch_list:   list[int],
    poschecked_list: list[int],
    changed_list:    list[int],
    args,
) -> None:
    rows = len(results)

    def _safe_pct(lst):
        return round(100 * sum(lst) / rows, 2) if rows else 0.0

    def _safe_avg(lst):
        return round(sum(lst) / rows, 4) if rows else 0.0

    summary = {
        "rows":                        rows,
        "topk_stable_k":               args.topk_stable,
        "avg_mean_kendall_tau":        float(np.nanmean(tau_list))    if tau_list    else float("nan"),
        "avg_min_kendall_tau":         float(np.nanmean(mintau_list)) if mintau_list else float("nan"),
        "pct_top1_stable":             _safe_pct(top1_list),
        "pct_topk_stable":             _safe_pct(topk_list),
        "pct_exact_ranking_match":     _safe_pct(exact_list),
        "avg_topk_position_matches":   _safe_avg(posmatch_list),
        "avg_topk_positions_checked":  _safe_avg(poschecked_list),
        "avg_answer_changed_per_row":  _safe_avg(changed_list),
    }
    results["__summary__"] = summary

    out = args.output or f"benchmark/results/{args.dataset}_kgsmile_permutation.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    k = args.topk_stable
    print("\n" + "=" * 64)
    print(f"  KG-SMILE permutation robustness  ({rows} rows, dataset={args.dataset})")
    print("=" * 64)
    print(f"avg mean Kendall-tau  : {summary['avg_mean_kendall_tau']:.4f}")
    print(f"avg min  Kendall-tau  : {summary['avg_min_kendall_tau']:.4f}")
    print(f"top-1 stable rows     : {summary['pct_top1_stable']}%")
    print(f"top-{k} stable rows   : {summary['pct_topk_stable']}% (same set)")
    print(f"top-{k} same-position : {summary['avg_topk_position_matches']}/{summary['avg_topk_positions_checked']} ranks (avg)")
    print(f"exact-ranking rows    : {summary['pct_exact_ranking_match']}%")
    print(f"avg answer-changed    : {summary['avg_answer_changed_per_row']} / {args.num_perms} permuted orders")
    print("=" * 64)
    print(f"Results -> {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

async def _main(args) -> None:
    questions: set[str] | None = None
    if args.questions_file:
        with open(args.questions_file, encoding="utf-8") as f:
            questions = set(json.load(f))
        print(f"Filtering to {len(questions)} question(s) from {args.questions_file}")

    kg_full = load_full_kg(args.kg_graphml)

    if args.input_dir:
        # ── from-JSON mode (no retrieval) ─────────────────────────────────
        cases = load_cf_cases(args.input_dir, questions=questions)
        if args.num_rows is not None:
            cases = cases[: args.num_rows]
        print(f"Loaded {len(cases)} CFE case(s) from {args.input_dir}")

        # In from-JSON mode we still need a LightRAG handle because run_kg_smile
        # calls rag.aquery internally for retrieval — but the graph we pass in
        # overrides that path when mode='ff'.  Pass a dummy rag if your
        # run_kg_smile supports it, or initialise a real one:
        rag = await initialize_lightrag(WORKING_DIRS[args.dataset])
        await run_permutation_from_json(args, rag, kg_full, cases)
    else:
        # ── live retrieval mode ───────────────────────────────────────────
        rag = await initialize_lightrag(WORKING_DIRS[args.dataset])
        await run_permutation_from_csv(args, rag, kg_full)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kg_smile_permutation",
        description="KG-SMILE context-permutation robustness benchmark.",
    )
    p.add_argument("--dataset",        choices=DATASETS, default="synthetic")
    p.add_argument("--rag-mode",       choices=["hybrid", "local", "global", "naive"],
                   default="hybrid")
    p.add_argument("--top-k",          type=int,   default=2)
    p.add_argument("--num-rows",       type=int,   default=None,
                   help="Cap on number of cases processed (default: all).")
    p.add_argument("--input-dir",      default=None,
                   help="Directory of CFE counterfactual_*.json files. "
                        "When set, the original_subgraph + original answer are read "
                        "from JSON (no LightRAG retrieval).")
    p.add_argument("--questions-file", default=None,
                   help="JSON list of question strings to filter cases by.")
    p.add_argument("--kg-graphml",     required=True,
                   help="Path to the full KG .graphml file (required by KG-SMILE).")
    p.add_argument("--num-perms",      type=int,   default=5,
                   help="Number of random triple-order permutations per case.")
    p.add_argument("--n-pert",         type=int,   default=20,
                   help="KG-SMILE n_perturbations per run.")
    p.add_argument("--kernel-width",   type=float, default=0.25)
    p.add_argument("--topk-stable",    type=int,   default=2,
                   help="k for the top-k stability check.")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--output",         default=None)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    asyncio.run(_main(args))