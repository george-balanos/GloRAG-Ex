"""Text-mention correctness core (Phase 5) - EMBEDDING & STRING HEURISTIC VERSION.

Maps flagged KG elements to a dataset's ground-truth supporting facts using a 
combination of normalized string matching and dense embeddings (Sentence-BERT).
Replaces the LLM-as-a-judge with deterministic and semantic heuristics.
"""
import os
import re
from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer

# --- EMBEDDING CONFIGURATION ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, highly accurate for sentence semantics
SIMILARITY_THRESHOLD = 0.65                # Cosine similarity cutoff (tune this based on your dataset)
JACCARD_THRESHOLD = 0.4                    # Token overlap cutoff for spans

_embedder_instance = None

def get_embedder():
    """Singleton pattern for the embedding model."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder_instance

_WORD = re.compile(r"[a-z0-9]+")

def normalize(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    return " ".join(_WORD.findall((s or "").lower()))

def tokens(s: str) -> list[str]:
    return normalize(s).split()

def normalized_contains(haystack: str, needle: str) -> bool:
    """True if `needle` is a (normalised) substring of `haystack`."""
    hn, nn = normalize(haystack), normalize(needle)
    return bool(nn) and nn in hn

def token_jaccard(s1: str, s2: str) -> float:
    """Computes the Jaccard similarity between the token sets of two strings."""
    set1, set2 = set(tokens(s1)), set(tokens(s2))
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def parse_id(eid: str):
    """('entity', name) | ('relation', (src, tgt)) for an "E::"/"R::" id."""
    if eid.startswith("E::"):
        return "entity", eid[3:]
    if eid.startswith("R::"):
        src, _, tgt = eid[3:].partition("->")
        return "relation", (src, tgt)
    return "entity", eid 

@lru_cache(maxsize=20000)
def _get_embedding(text: str) -> np.ndarray:
    if not text.strip():
        return np.zeros(384) # Match MiniLM dimensions
    return get_embedder().encode(text, normalize_embeddings=True)

def check_semantic_similarity(text1: str, text2: str, threshold: float) -> bool:
    """Returns True if the cosine similarity between two texts exceeds the threshold."""
    if not text1.strip() or not text2.strip():
        return False
    emb1 = _get_embedding(text1)
    emb2 = _get_embedding(text2)
    similarity = np.dot(emb1, emb2) # Already normalized, so dot product == cosine similarity
    return float(similarity) >= threshold


# ── EVALUATION LOGIC ─────────────────────────────────────────────────────────
# Relevance is decided against individual supporting-fact *units* (a single sentence
# / paragraph), not the concatenated gold blob: this keeps token-Jaccard meaningful
# and scopes an entity name match to one fact. Relations are the exception -- their
# endpoints may bridge two facts (multi-hop), so ``id_relevant_any`` matches relation
# endpoints across the UNION of units. ``id_relevant_any`` / ``span_relevant_any`` are
# the entry points the evaluator uses.

def _id_relevant_unit(eid: str, unit: str, desc_by_id: dict | None, match: str) -> bool:
    """Is element ``eid`` grounded in a single supporting-fact ``unit``?

    Surface match first (entity name as a normalised substring; relation = both
    endpoints present). KG entities are extracted from the same corpus the facts
    come from, so an entity name appearing in a fact is taken as grounded -- no
    token-length gate. The semantic fallback on the element's description runs only
    when ``match == "name+desc"``.
    """
    if not unit or not unit.strip():
        return False

    kind, payload = parse_id(eid)
    desc = desc_by_id.get(eid, "") if desc_by_id else ""
    use_semantic = match == "name+desc"

    if kind == "entity":
        if normalized_contains(unit, payload):
            return True
    else:  # relation -- require both endpoints to be mentioned
        src, tgt = payload
        if normalized_contains(unit, src) and normalized_contains(unit, tgt):
            return True

    # semantic fallback on name (+description) -- name+desc mode only
    if use_semantic:
        payload_str = f"{payload[0]} -> {payload[1]}" if kind == "relation" else payload
        combined = f"{payload_str}: {desc}" if desc else payload_str
        return check_semantic_similarity(combined, unit, SIMILARITY_THRESHOLD)
    return False


def id_relevant(eid: str, gold_text: str, desc_by_id: dict | None = None,
                match: str = "name+desc", **kwargs) -> bool:
    """Element vs. a single gold text (treats ``gold_text`` as one unit)."""
    return _id_relevant_unit(eid, gold_text, desc_by_id, match)


def id_relevant_any(eid: str, units, desc_by_id: dict | None = None,
                    match: str = "name+desc", **kwargs) -> bool:
    """True if ``eid`` is grounded in the supporting facts.

    Entities are matched **per unit** (the name must appear within a single fact).
    Relations use a **union** rule: each endpoint need only appear in *some* fact,
    not the same one -- a supporting relation can legitimately bridge two facts
    (the defining case in multi-hop QA), and requiring both endpoints in one unit
    would wrongly drop those bridge edges.
    """
    units = list(units or [])
    if not units:
        return False

    kind, payload = parse_id(eid)
    if kind == "relation":
        src, tgt = payload
        # surface: both endpoints supported, possibly in different facts
        if any(normalized_contains(u, src) for u in units) and \
           any(normalized_contains(u, tgt) for u in units):
            return True
        # semantic fallback on the relation phrase (+description), name+desc mode only
        if match == "name+desc":
            desc = desc_by_id.get(eid, "") if desc_by_id else ""
            phrase = f"{src} -> {tgt}: {desc}" if desc else f"{src} -> {tgt}"
            return any(check_semantic_similarity(phrase, u, SIMILARITY_THRESHOLD) for u in units)
        return False

    return any(_id_relevant_unit(eid, u, desc_by_id, match) for u in units)


def _span_relevant_unit(span: str, unit: str, jaccard: float) -> bool:
    if not span or not span.strip() or not unit or not unit.strip():
        return False
    if token_jaccard(span, unit) >= jaccard:        # literal extraction
        return True
    return check_semantic_similarity(span, unit, SIMILARITY_THRESHOLD)  # paraphrase


def span_relevant(span: str, gold_text: str, jaccard: float = JACCARD_THRESHOLD, **kwargs) -> bool:
    """Text-span vs. a single gold text via token Jaccard or dense embeddings."""
    return _span_relevant_unit(span, gold_text, jaccard)


def span_relevant_any(span, units, jaccard: float = JACCARD_THRESHOLD, **kwargs) -> bool:
    """True if ``span`` matches ANY supporting-fact unit (per-unit matching)."""
    return any(_span_relevant_unit(span, u, jaccard) for u in (units or []))


# ── METRICS ──────────────────────────────────────────────────────────────────

def gt_relevant_set(universe_ids, gold_text: str, desc_by_id: dict | None = None,
                    match: str = "name+desc", **kwargs) -> set:
    return {eid for eid in universe_ids if id_relevant(eid, gold_text, desc_by_id, match)}

def set_precision(flagged_ids, gold_set) -> dict:
    flagged = list(dict.fromkeys(flagged_ids))
    gold = set(gold_set)
    tp = sum(1 for f in flagged if f in gold)
    n = len(flagged)
    return {
        "precision": (tp / n) if n else None,
        "tp": tp, "fp": n - tp, "n_flagged": n,
        "gold_flagged": sorted(f for f in flagged if f in gold),
    }

def fact_coverage(element_ids, supporting_units, gold_text, desc_by_id=None,
                  match: str = "name+desc", **kwargs) -> dict:
    els = list(dict.fromkeys(element_ids))
    units = list(supporting_units or [])
    n_gold = sum(1 for e in els if id_relevant_any(e, units, desc_by_id, match))
    covered = 0
    for u in units:
        if any(_id_relevant_unit(e, u, desc_by_id, match) for e in els):
            covered += 1
    return {"n_elements": len(els), "n_gold_elements": n_gold,
            "n_facts_covered": covered, "n_facts_total": len(units)}

def precision_at_k(ranked_ids, positive_ids, gold_set, ks=(1, 2, 3, 5)) -> dict:
    """Precision@k over the positive-filtered ranking (filter-then-slice).

    Kept byte-identical with ``agreement_judge.precision_at_k`` so the metric does
    not change meaning between the heuristic and the LLM-judge backends:
    ``prec = tp/n`` over the items present at cutoff k (None when empty)."""
    gold = set(gold_set)
    pos = set(positive_ids)
    retrieved = [x for x in ranked_ids if x in pos]
    prec, hit, tp_at, n_at = {}, {}, {}, {}
    for k in ks:
        flagged_k = retrieved[:k]
        tp = sum(1 for x in flagged_k if x in gold)
        n = len(flagged_k)
        prec[str(k)] = (tp / n) if n else None
        hit[str(k)] = 1 if tp > 0 else 0
        tp_at[str(k)] = tp
        n_at[str(k)] = n
    return {"precision_at": prec, "hit_at": hit, "tp_at": tp_at, "n_at": n_at}