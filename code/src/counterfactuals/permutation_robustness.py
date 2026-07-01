"""Post-hoc context-permutation robustness for counterfactual explanations.

Reads saved counterfactual JSONs (the flipping cases, written by
counterfactuals/generate.py :: save_operations_to_json), and for each one tests
whether the answer still flips under context permutation.

Permutation style (shared with the Shapley experiment, src/perm_utils.py):
treat the perturbed graph's entities + relations as one bag of objects and
sample 5 random object orderings, re-split into the two-section RAG layout. For
each permutation we re-render the context, re-query the RAG LLM, and judge
against the ORIGINAL answer — flip = (judge score == 0), matching generate.py.

This experiment uses the vLLM generation model + the judge directly — no LightRAG,
no retrieval (the perturbed subgraph already lives in the input JSONs), and no
HF/Shapley model.

Run from code/ (PYTHONPATH=code), e.g.:
  ../.venv/bin/python -m src.counterfactuals.permutation_robustness \
      --dataset synthetic \
      --input-dir src/counterfactuals/results/ablation/ft_delete_no_psp/synthetic/delete_ops_ft
"""
from src.query import build_rag_system_prompt
from src.llm.utils import vllm_model_complete
from src.llm_judge import judge_response
from src.base import Entity, Relation
from src.dataset_setup import DATASETS
from src.perm_utils import random_object_permutations

from tqdm import tqdm
import argparse
import asyncio
import glob
import json
import logging
import os

logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("lightrag").setLevel(logging.WARNING)


def _object_id(kind, obj) -> str:
    """Stable id matching run_shapley's convention, so an object is traceable."""
    return f"E::{obj.name}" if kind == "entity" else f"R::{obj.src}->{obj.tgt}"


async def _generate_answer(context: str, question: str) -> str:
    """Generate the RAG answer for `context` WITHOUT LightRAG/retrieval.

    Byte-identical to query()'s generation: same system prompt and the same vLLM
    model (vllm_model_complete defaults temperature=0, max_tokens=8192, matching
    LightRAG's llm_model_kwargs).
    """
    return await vllm_model_complete(question, system_prompt=build_rag_system_prompt(context))


def _entities_from_dict(d: dict) -> list[Entity]:
    return [Entity(name=e.get("name", ""), type=e.get("type", ""),
                   description=e.get("description", ""), rank=e.get("rank", 0.0))
            for e in (d.get("entities") or [])]


def _relations_from_dict(d: dict) -> list[Relation]:
    return [Relation(src=r.get("src", ""), tgt=r.get("tgt", ""),
                     keywords=r.get("keywords", ""), description=r.get("description", ""),
                     weight=r.get("weight", 0.0))
            for r in (d.get("relations") or [])]


def _is_nonempty_subgraph(sg) -> bool:
    """True if the subgraph dict has at least one entity or relation."""
    return bool(sg) and bool((sg.get("entities") or []) or (sg.get("relations") or []))


# def load_flip_cases(input_dir: str) -> tuple[list[tuple[str, dict]], dict]:
#     """Return (cases, counts).

#     cases  : [(filepath, payload)] for JSONs that found a flip AND carry a
#              non-empty perturbed_subgraph (the ones we actually permute).
#     counts : {total_cf_files, n_found, n_nonempty_perturbed, n_empty_or_not_found}.
#     """
#     files = sorted(glob.glob(os.path.join(input_dir, "**", "counterfactual_*.json"), recursive=True))
#     cases = []
#     total, n_found, n_nonempty = 0, 0, 0
#     for fp in files:
#         try:
#             with open(fp, encoding="utf-8") as f:
#                 payload = json.load(f)
#         except Exception as e:
#             print(f"  skip {fp}: {e}")
#             continue
#         total += 1
#         found = bool(payload.get("found"))
#         nonempty = _is_nonempty_subgraph(payload.get("perturbed_subgraph"))
#         n_found += int(found)
#         n_nonempty += int(nonempty)
#         if found and nonempty:
#             cases.append((fp, payload))
#     counts = {
#         "total_cf_files": total,
#         "n_found": n_found,
#         "n_nonempty_perturbed": n_nonempty,
#         "n_empty_or_not_found": total - len(cases),
#     }
#     return cases, counts

def load_flip_cases(input_dir: str, questions: set[str]) -> tuple[list[tuple[str, dict]], dict]:
    """Return (cases, counts).

    cases  : [(filepath, payload)] for JSONs that found a flip AND carry a
             non-empty perturbed_subgraph (the ones we actually permute).
    counts : {total_cf_files, n_found, n_nonempty_perturbed, n_empty_or_not_found}.
    """
    files = sorted(glob.glob(os.path.join(input_dir, "**", "counterfactual_*.json"), recursive=True))
    cases = []
    total, n_found, n_nonempty = 0, 0, 0
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"  skip {fp}: {e}")
            continue
        if questions is not None and payload.get("question") not in questions:
            continue
        total += 1
        found = bool(payload.get("found"))
        nonempty = _is_nonempty_subgraph(payload.get("perturbed_subgraph"))
        n_found += int(found)
        n_nonempty += int(nonempty)
        if found and nonempty:
            cases.append((fp, payload))
    counts = {
        "total_cf_files": total,
        "n_found": n_found,
        "n_nonempty_perturbed": n_nonempty,
        "n_empty_or_not_found": total - len(cases),
    }
    return cases, counts

async def run(args):
    questions = None
    if args.questions_file:
        with open(args.questions_file, encoding="utf-8") as f:
            questions = set(json.load(f))
        print(f"Filtering to {len(questions)} sampled question(s) from {args.questions_file}")

    cases, counts = load_flip_cases(args.input_dir, questions=questions)
    if args.num_files is not None:
        cases = cases[:args.num_files]
    print(f"Scanned {counts['total_cf_files']} counterfactual JSON(s) in {args.input_dir}")
    print(f"  found={counts['n_found']} | non-empty perturbed_subgraph={counts['n_nonempty_perturbed']} "
          f"(permuted) | empty/not-found={counts['n_empty_or_not_found']}")

    results = {}
    all6_count = 0
    stability_sum = 0.0

    for fp, payload in tqdm(cases, desc="CF permutation robustness", total=len(cases)):
        question = payload["question"]
        original_answer = payload["answers"]["original"]
        psg = payload["perturbed_subgraph"]
        entities = _entities_from_dict(psg)
        relations = _relations_from_dict(psg)

        perms = random_object_permutations(entities, relations, count=5, seed=args.seed)
        per_perm = {}
        n_flipped = 0
        for p in perms:
            new_response = await _generate_answer(p["render"], question)
            score = await judge_response(question, new_response, original_answer)
            flipped = (score == 0)
            n_flipped += int(flipped)
            per_perm[p["perm_id"]] = {
                "perm_id": p["perm_id"],
                "perm": list(p["perm"]),
                "identity": p["identity"],
                "object_order": [_object_id(k, o) for (k, o) in p["objects"]],
                "context": p["render"],          # exact context shown to the LLM
                "response": new_response,
                "judge_score": score,
                "flipped": flipped,
            }

        n_perms = len(perms)
        stability = n_flipped / n_perms if n_perms else 0.0
        all_flip = (n_perms > 0 and n_flipped == n_perms)
        all6_count += int(all_flip)
        stability_sum += stability

        results[os.path.basename(fp)] = {
            "filepath": fp,
            "mode": payload.get("mode"),
            "question": question,
            "original_answer": original_answer,
            "saved_perturbed_answer": payload["answers"].get("perturbed"),
            "operations": payload.get("operations"),
            "num_operations": payload.get("num_operations"),
            "cost": payload.get("cost"),
            "n_entities": len(entities),
            "n_relations": len(relations),
            "object_ids": [_object_id(k, o) for (k, o)
                           in ([("entity", e) for e in entities] + [("relation", r) for r in relations])],
            "num_permutations": n_perms,
            "num_flipped": n_flipped,
            "flip_stability": round(stability, 4),
            "flip_under_all_permutations": all_flip,
            "permutations": per_perm,
        }
        print(f"[{os.path.basename(fp)}] perms={n_perms} flipped={n_flipped}/{n_perms} "
              f"stability={stability:.2f} all={all_flip}")

    n = len(results) or 1
    summary = {
        "cases": len(results),
        "avg_flip_stability": round(stability_sum / n, 4),
        "pct_flip_under_all_permutations": round(100 * all6_count / n, 2),
        **counts,
    }
    results["__summary__"] = summary

    # Default output co-located with the input cell so per-subdir runs don't clobber
    # a single shared file.
    out = args.output or os.path.join(args.input_dir, "permutation_robustness.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print(f"  CF permutation robustness  ({summary['cases']} flip cases, dataset={args.dataset})")
    print("=" * 64)
    print(f"counterfactual files / found / non-empty perturbed : "
          f"{counts['total_cf_files']} / {counts['n_found']} / {counts['n_nonempty_perturbed']}")
    print(f"avg flip-stability (frac of perms still flipping)  : {summary['avg_flip_stability']}")
    print(f"cases flipping under ALL permutations              : {summary['pct_flip_under_all_permutations']}%")
    print("=" * 64)
    print(f"Results -> {out}")

# async def run(args):
#     cases, counts = load_flip_cases(args.input_dir)
#     if args.num_files is not None:
#         cases = cases[:args.num_files]
#     print(f"Scanned {counts['total_cf_files']} counterfactual JSON(s) in {args.input_dir}")
#     print(f"  found={counts['n_found']} | non-empty perturbed_subgraph={counts['n_nonempty_perturbed']} "
#           f"(permuted) | empty/not-found={counts['n_empty_or_not_found']}")

#     results = {}
#     all6_count = 0
#     stability_sum = 0.0

#     for fp, payload in tqdm(cases, desc="CF permutation robustness", total=len(cases)):
#         question = payload["question"]
#         original_answer = payload["answers"]["original"]
#         psg = payload["perturbed_subgraph"]
#         entities = _entities_from_dict(psg)
#         relations = _relations_from_dict(psg)

#         perms = random_object_permutations(entities, relations, count=5, seed=args.seed)
#         per_perm = {}
#         n_flipped = 0
#         for p in perms:
#             new_response = await _generate_answer(p["render"], question)
#             score = await judge_response(question, new_response, original_answer)
#             flipped = (score == 0)
#             n_flipped += int(flipped)
#             per_perm[p["perm_id"]] = {
#                 "perm_id": p["perm_id"],
#                 "perm": list(p["perm"]),
#                 "identity": p["identity"],
#                 "object_order": [_object_id(k, o) for (k, o) in p["objects"]],
#                 "context": p["render"],          # exact context shown to the LLM
#                 "response": new_response,
#                 "judge_score": score,
#                 "flipped": flipped,
#             }

#         n_perms = len(perms)
#         stability = n_flipped / n_perms if n_perms else 0.0
#         all_flip = (n_perms > 0 and n_flipped == n_perms)
#         all6_count += int(all_flip)
#         stability_sum += stability

#         results[os.path.basename(fp)] = {
#             "filepath": fp,
#             "mode": payload.get("mode"),
#             "question": question,
#             "original_answer": original_answer,
#             "saved_perturbed_answer": payload["answers"].get("perturbed"),
#             "operations": payload.get("operations"),
#             "num_operations": payload.get("num_operations"),
#             "cost": payload.get("cost"),
#             "n_entities": len(entities),
#             "n_relations": len(relations),
#             "object_ids": [_object_id(k, o) for (k, o)
#                            in ([("entity", e) for e in entities] + [("relation", r) for r in relations])],
#             "num_permutations": n_perms,
#             "num_flipped": n_flipped,
#             "flip_stability": round(stability, 4),
#             "flip_under_all_permutations": all_flip,
#             "permutations": per_perm,
#         }
#         print(f"[{os.path.basename(fp)}] perms={n_perms} flipped={n_flipped}/{n_perms} "
#               f"stability={stability:.2f} all={all_flip}")

#     n = len(results) or 1
#     summary = {
#         "cases": len(results),
#         "avg_flip_stability": round(stability_sum / n, 4),
#         "pct_flip_under_all_permutations": round(100 * all6_count / n, 2),
#         **counts,
#     }
#     results["__summary__"] = summary

#     # Default output co-located with the input cell so per-subdir runs don't clobber
#     # a single shared file.
#     out = args.output or os.path.join(args.input_dir, "permutation_robustness.json")
#     os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
#     with open(out, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)

#     print("\n" + "=" * 64)
#     print(f"  CF permutation robustness  ({summary['cases']} flip cases, dataset={args.dataset})")
#     print("=" * 64)
#     print(f"counterfactual files / found / non-empty perturbed : "
#           f"{counts['total_cf_files']} / {counts['n_found']} / {counts['n_nonempty_perturbed']}")
#     print(f"avg flip-stability (frac of perms still flipping)  : {summary['avg_flip_stability']}")
#     print(f"cases flipping under ALL permutations              : {summary['pct_flip_under_all_permutations']}%")
#     print("=" * 64)
#     print(f"Results -> {out}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="permutation_robustness",
        description="Post-hoc context-permutation robustness for counterfactual flips.")
    p.add_argument("--dataset", choices=DATASETS, default="synthetic")
    p.add_argument("--input-dir", required=True,
                   help="Directory of saved counterfactual_*.json (searched recursively).")
    p.add_argument("--num-files", type=int, default=None, help="Cap on number of flip cases.")
    p.add_argument("--seed", type=int, default=42, help="Seed for the 5 random object permutations.")
    p.add_argument("--output", default=None)
    p.add_argument("--questions-file", default=None,
                   help="Path to a sampled questions JSON (list of question strings). "
                        "If provided, only flip cases whose question appears in this list are processed.")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    asyncio.run(run(args))
