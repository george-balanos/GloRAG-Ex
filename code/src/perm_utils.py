"""Context-permutation utilities shared by the Shapley and counterfactual
robustness experiments.

The Shapley permutation experiment now treats entities + relations as a single
bag of objects and samples random object orderings. The counterfactual
robustness experiment still uses the older section-wise thirds permutation.
"""
import itertools
import random

import numpy as np

from src.parser import render_context

GROUP_NAMES = ("start", "mid", "last")


def thirds(seq: list) -> list[list]:
    """Split `seq` into 3 contiguous groups (as equal as possible).

    Uneven counts put the remainder in the earlier groups (np.array_split
    convention). Empty groups are allowed (e.g. when len(seq) < 3).
    """
    return [list(g) for g in np.array_split(np.array(seq, dtype=object), 3)]


def section_permutations(entities: list, relations: list):
    """Yield the (<=6) per-section-thirds permutations of (entities, relations).

    Returns a list of dicts: {perm_id, perm, entities, relations, identity}.
    Permutations whose rendered context is identical (common when a section has
    fewer than 3 non-empty groups) are de-duplicated to avoid wasted LLM calls;
    the identity permutation is always kept.
    """
    e_groups = thirds(entities)
    r_groups = thirds(relations)

    seen_renders: dict[str, None] = {}
    out = []
    for perm in itertools.permutations(range(3)):
        ents = [x for g in perm for x in e_groups[g]]
        rels = [x for g in perm for x in r_groups[g]]
        render = render_context(ents, rels)
        is_identity = perm == (0, 1, 2)
        if render in seen_renders and not is_identity:
            continue
        seen_renders[render] = None
        out.append({
            "perm_id": "_".join(GROUP_NAMES[i] for i in perm),
            "perm": perm,
            "entities": ents,
            "relations": rels,
            "identity": is_identity,
        })
    return out


def random_object_permutations(entities: list, relations: list, count: int = 5, seed: int | None = None):
    """Yield `count` random permutations over the combined entity+relation objects.

    Each permutation shuffles the full object list, then re-splits it into
    ordered entity/relation sections while preserving the shuffled order within
    each section. The result mirrors the standard RAG context format.
    """
    objects = [("entity", e) for e in entities] + [("relation", r) for r in relations]
    if not objects:
        return []

    rng = random.Random(seed)
    indices = list(range(len(objects)))
    out = []
    for i in range(count):
        perm = tuple(rng.sample(indices, len(indices)))
        permuted = [objects[j] for j in perm]
        ents = [obj for kind, obj in permuted if kind == "entity"]
        rels = [obj for kind, obj in permuted if kind == "relation"]
        render = render_context(ents, rels)
        out.append({
            "perm_id": f"rand_{i + 1:02d}",
            "perm": perm,
            "entities": ents,
            "relations": rels,
            "objects": permuted,
            "render": render,
            "identity": perm == tuple(indices),
        })
    return out
