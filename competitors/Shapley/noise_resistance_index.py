#!/usr/bin/env python3
"""
Shared loader for the GLoRAG-Ex noise_resistance directory structure:

    {dataset}/noise_resistance/{variant}/noise_level_{X}/{question_id}.json

Each JSON file represents one question at one noise level and contains:
  - noise.noise_robust  (bool)  — whether the answer survived noise injection
  - mode                (str)   — "ff" or "ft" (same as the parent variant dir)

load_index(noise_resistance_root, dataset, variant) -> NoiseLevelIndex

NoiseLevelIndex is keyed by the QUESTION STRING (not file-based ids), so that
different methods can use different internal id schemes and still be compared
on exactly the same set of questions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeAlias

# {noise_pct_int: {question_str: noise_robust_bool}}
# e.g. {10: {"What are the two primary materials...": True, ...}, 20: {...}}
NoiseLevelIndex = dict[int, dict[str, bool]]


def load_index(
    noise_resistance_root: str | Path,
    dataset: str,
    variant: str,
) -> dict[int, dict[str, bool]]:
    """
    Walk  {noise_resistance_root}/{dataset}/noise_resistance/{variant}/noise_level_*/
    and collect the noise_robust flag keyed by the question string.

    Args:
        noise_resistance_root: root dir containing dataset sub-dirs
        dataset:  e.g. "hotpotqa", "musique", "synthetic"
        variant:  "ff" or "ft"

    Returns:
        {noise_pct_int: {question_str: noise_robust_bool}}
    """
    root    = Path(noise_resistance_root)
    var_dir = root / dataset / "noise_resistance" / variant
    if not var_dir.exists():
        raise FileNotFoundError(
            f"Noise-resistance directory not found: {var_dir}"
        )

    level_pattern = re.compile(r"^noise_level_(\d+)$")
    index: dict[int, dict[str, bool]] = {}

    for level_dir in sorted(var_dir.iterdir()):
        if not level_dir.is_dir():
            continue
        m = level_pattern.match(level_dir.name)
        if not m:
            continue
        noise_pct_int = int(m.group(1))
        level_index: dict[str, bool] = {}

        for json_file in sorted(level_dir.glob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                question     = data["question"].strip()
                noise_robust = bool(data["noise"]["noise_robust"])
            except (KeyError, TypeError, json.JSONDecodeError) as e:
                print(f"  [warn] Could not read {json_file}: {e}")
                continue
            level_index[question] = noise_robust

        if level_index:
            index[noise_pct_int] = level_index

    if not index:
        raise ValueError(
            f"No noise-level data found under {var_dir}. "
            f"Expected sub-dirs named noise_level_{{N}}/ containing .json files."
        )

    total_q  = len(next(iter(index.values())))
    robust_q = sum(1 for v in next(iter(index.values())).values() if v)
    print(f"  [{dataset}/{variant}] levels={sorted(index)}, "
          f"questions={total_q}, noise-robust in first level={robust_q}")
    return index


def robust_questions(index: dict[int, dict[str, bool]], noise_pct_int: int) -> set[str]:
    """Return question strings that are noise-robust at a given noise level."""
    return {q for q, robust in index.get(noise_pct_int, {}).items() if robust}


def all_questions(index: dict[int, dict[str, bool]]) -> set[str]:
    """Union of all question strings across all noise levels."""
    return {q for level in index.values() for q in level}