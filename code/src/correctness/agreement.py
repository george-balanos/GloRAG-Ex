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

def id_relevant(eid: str, gold_text: str, desc_by_id: dict | None = None, **kwargs) -> bool:
    """Is element ``eid`` grounded in the gold text using heuristics?"""
    if not gold_text.strip():
        return False
        
    kind, payload = parse_id(eid)
    
    # 1. High-Precision String Matching on Names
    # If the entity name or relation endpoints are explicitly stated in the ground truth, it's a hit.
    if kind == "entity":
        if normalized_contains(gold_text, payload):
            return True
    elif kind == "relation":
        src, tgt = payload
        # If both source and target are mentioned in the gold text, it's highly likely relevant
        if normalized_contains(gold_text, src) and normalized_contains(gold_text, tgt):
            return True

    # 2. Semantic Fallback on Descriptions
    # If the exact names aren't there, check if the element's description aligns with the ground truth.
    desc = desc_by_id.get(eid, "") if desc_by_id else ""
    if desc:
        payload_str = f"{payload[0]} -> {payload[1]}" if kind == "relation" else payload
        combined_text = f"{payload_str}: {desc}"
        return check_semantic_similarity(combined_text, gold_text, SIMILARITY_THRESHOLD)

    return False


def span_relevant(span: str, gold_text: str, **kwargs) -> bool:
    """Text-span relevance via Token Jaccard and Dense Embeddings."""
    if not span or not span.strip() or not gold_text or not gold_text.strip():
        return False
        
    # 1. Fast Token Overlap (Catches literal extractions)
    if token_jaccard(span, gold_text) >= JACCARD_THRESHOLD:
        return True
        
    # 2. Semantic Overlap (Catches paraphrasing)
    return check_semantic_similarity(span, gold_text, SIMILARITY_THRESHOLD)


# ── METRICS (Unchanged) ──────────────────────────────────────────────────────

def gt_relevant_set(universe_ids, gold_text: str, desc_by_id: dict | None = None, **kwargs) -> set:
    return {eid for eid in universe_ids if id_relevant(eid, gold_text, desc_by_id)}

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

def fact_coverage(element_ids, supporting_units, gold_text, desc_by_id=None, **kwargs) -> dict:
    els = list(dict.fromkeys(element_ids))
    n_gold = sum(1 for e in els if id_relevant(e, gold_text, desc_by_id))
    covered = 0
    for u in supporting_units:
        if any(id_relevant(e, u, desc_by_id) for e in els):
            covered += 1
    return {"n_elements": len(els), "n_gold_elements": n_gold,
            "n_facts_covered": covered, "n_facts_total": len(supporting_units)}

def precision_at_k(ranked_ids, positive_ids, gold_set, ks=(1, 2, 3, 5)) -> dict:
    gold = set(gold_set)
    pos = set(positive_ids)
    
    retrieved = [x for x in ranked_ids if x in pos]
    prec, hit, tp_at, n_at = {}, {}, {}, {}
    for k in ks:
        flagged_k = retrieved[:k]
        tp = sum(1 for x in flagged_k if x in gold)
        prec[str(k)] = (tp / k)     
        hit[str(k)] = 1 if tp > 0 else 0           
        tp_at[str(k)] = tp
        n_at[str(k)] = len(flagged_k)
    return {"precision_at": prec, "hit_at": hit, "tp_at": tp_at, "n_at": n_at}