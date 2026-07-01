from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

from .graph_utils import build_graph


# ─────────────────────────────────────────────────────────────
# Question loaders
# ─────────────────────────────────────────────────────────────

def load_questions_from_csv(csv_path: str, num: int = 100) -> list[dict]:
    """Load up to `num` questions from a CSV with columns: id, questions, answers."""
    questions = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                "id":           row["id"],
                "question":     row["questions"].strip(),
                "ground_truth": row["answers"].strip() or None,
            })
    print(f"[io] Loaded {min(num, len(questions))} questions from {csv_path}")
    return questions[:num]


def load_questions_from_explanation(filepath: str) -> dict | None:
    """
    Load a single question from an explanation JSON produced by the main pipeline.
    Returns None if the question was not solved (found=False).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("found") is True:
        return {
            "question":     data["question"],
            "ground_truth": data["answers"]["ground_truth"],
        }
    return None


# ─────────────────────────────────────────────────────────────
# Attribution / result loaders
# ─────────────────────────────────────────────────────────────

def load_attributions(path: str) -> list[dict]:
    """
    Load attribution data produced by runner.py (normal mode).

    Accepts both the legacy list format and the new dict-of-dicts format:
        {"0": {...}, "1": {...}, ...}
    Always returns a plain list of entry dicts for downstream compatibility.
    """
    with open(path, "r") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return list(raw.values())
    return raw


def load_results(results_dir: str) -> tuple[dict, dict, dict, dict]:
    """
    Load all per-question result JSON files from a directory.

    Returns four dicts keyed by question string:
        graphs_by_question  — nx.DiGraph of the original subgraph
        ops_by_question     — list of (op_type, target) operations
        costs_by_question   — integer edit cost (or None)
        found_by_question   — bool
    """
    graphs_by_question: dict[str, nx.DiGraph] = {}
    ops_by_question:    dict[str, list]       = {}
    costs_by_question:  dict[str, int | None] = {}
    found_by_question:  dict[str, bool]       = {}

    for fp in Path(results_dir).glob("*.json"):
        with open(fp, "r") as f:
            obj = json.load(f)

        q = obj["question"]
        graphs_by_question[q] = build_graph(obj["original_subgraph"])
        ops_by_question[q]    = obj.get("operations", [])
        costs_by_question[q]  = obj.get("cost", None)
        found_by_question[q]  = obj.get("found", False)

    return graphs_by_question, ops_by_question, costs_by_question, found_by_question


def load_folder(folder_path: str) -> dict[str, dict]:
    """
    Load all JSON files in a folder and index them by question string.
    Used by compare.py to load both the 'your method' and KG-SMILE result sets.

    Handles both the legacy list format and the new dict-of-dicts format.
    """
    results: dict[str, dict] = {}
    for path in Path(folder_path).glob("*.json"):
        with open(path) as f:
            raw = json.load(f)

        if isinstance(raw, dict) and all(k.isdigit() for k in raw):
            for entry in raw.values():
                if "question" in entry:
                    results[entry["question"]] = entry

        elif isinstance(raw, list):
            for entry in raw:
                if "question" in entry:
                    results[entry["question"]] = entry

        elif "question" in raw:
            results[raw["question"]] = raw

    return results


# ─────────────────────────────────────────────────────────────
# Resume support (runner.py)
# ─────────────────────────────────────────────────────────────

def load_completed(output_path: str) -> tuple[set[str], dict]:
    if not Path(output_path).exists():
        return set(), {}

    with open(output_path, encoding="utf-8") as f:
        raw = json.load(f)


    if isinstance(raw, list):
        existing = {str(i): entry for i, entry in enumerate(raw)}
    else:
        existing = raw

    completed = {
        entry["question"]
        for entry in existing.values()
        if "error" not in entry and "question" in entry
    }
    print(f"[io] Resuming — {len(completed)} questions already completed")
    return completed, existing


# ─────────────────────────────────────────────────────────────
# Attribution score builder
# ─────────────────────────────────────────────────────────────

def _build_scores(result) -> dict[str, float]:
    """
    Flatten node and edge attributions into a single dict with prefixed keys:
        "E::<node_name>"          for node (entity) attributions
        "R::<src>-><tgt>"         for edge (relation) attributions

    Both are sorted by absolute attribution value descending so the most
    influential items appear first.
    """
    scores: dict[str, float] = {}

    node_attrs = sorted(
        result.node_attributions.items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    for node, val in node_attrs:
        scores[f"E::{node}"] = val

    edge_attrs = sorted(
        result.edge_attributions.items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    for (src, tgt), val in edge_attrs:
        scores[f"R::{src}->{tgt}"] = val

    return scores


# ─────────────────────────────────────────────────────────────
# Output schema serialisation (runner.py)
# ─────────────────────────────────────────────────────────────

def to_output_schema(
    question:       str,
    result,
    ground_truth:   str | None = None,
    question_id:    str | None = None,
    elapsed_seconds: float | None = None,
    llm_call_count:  int   | None = None,
) -> dict:
    from .kg_smile import result_to_dict   # deferred to avoid circular import

    d = result_to_dict(result)

    n_items = len(result.node_attributions) + len(result.edge_attributions)

    return {
        # ── New primary fields ──────────────────────────────────
        "question":         question,
        "ground_truth":     ground_truth,
        "rag_answer":       result.original_response,
        "score":            None,
        "n_items":          n_items,
        "elapsed_seconds":  elapsed_seconds,
        "llm_call_count":   llm_call_count,
        "scores":           _build_scores(result),

        # ── Legacy / downstream fields ──────────────────────────
        "id":                    question_id,
        "surrogate_r2":          result.surrogate_r2,
        "output_shift_std":      result.output_shift_std,
        "mean_graph_cosine_sim": result.mean_graph_cosine_sim,
        "mean_kernel_weight":    result.mean_kernel_weight,
        "degenerate":            result.degenerate,
        "noise_robust":          result.noise_robust,
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "edge_attributions":     d["edge_attributions"],
        "node_attributions":     d["node_attributions"],
    }