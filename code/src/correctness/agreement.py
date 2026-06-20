"""Text-mention correctness core (Phase 5).

Maps flagged KG elements to a dataset's ground-truth supporting facts on a common
granularity by **text mention**, so the same metric applies to GloRAG-Ex (the
elements edited by the counterfactual sequence) and Shapley-RAG (the elements
ranked by attribution score).

Element identity is the shared scheme used by both methods:
  entity   -> id "E::<name>"
  relation -> id "R::<src>-><tgt>"

Relevance: an entity is GT-relevant iff its (normalised) name occurs as a
contiguous, word-bounded token sub-sequence of the supporting ``gold_text``; a
relation iff BOTH endpoints occur. Working in normalised tokens makes the match
word-boundary safe (e.g. "art" does not match inside "smart").

Metric (per instance): precision only -- TP/(TP+FP) over the flagged set
(``set_precision``, GloRAG-Ex) or precision@k over the positive-score top-k
ranking (``precision_at_k``, attribution baselines). No recall/F1: a gold element
absent from a minimal edit set is not a false negative, so the FN side is undefined.
"""
import re

_WORD = re.compile(r"[a-z0-9]+")


def normalize(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    return " ".join(_WORD.findall((s or "").lower()))


def tokens(s: str) -> list[str]:
    return normalize(s).split()


def _contains_subsequence(hay: list[str], needle: list[str]) -> bool:
    """True if ``needle`` is a contiguous sub-list of ``hay`` (word-boundary safe)."""
    n, h = len(needle), len(hay)
    if n == 0 or n > h:
        return False
    first = needle[0]
    for i in range(h - n + 1):
        if hay[i] == first and hay[i:i + n] == needle:
            return True
    return False


def _shares_ngram(a: list[str], b: list[str], n: int) -> bool:
    """True if token-lists ``a`` and ``b`` share any contiguous n-gram (n>=1).

    Used to match an element's *description* against the gold text: a shared
    salient phrase is a strong relevance signal, far less noisy than fractional
    content-word overlap.
    """
    if n <= 0 or len(a) < n or len(b) < n:
        return False
    grams = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    return any(tuple(a[i:i + n]) in grams for i in range(len(a) - n + 1))


def parse_id(eid: str):
    """('entity', name) | ('relation', (src, tgt)) for an "E::"/"R::" id."""
    if eid.startswith("E::"):
        return "entity", eid[3:]
    if eid.startswith("R::"):
        src, _, tgt = eid[3:].partition("->")
        return "relation", (src, tgt)
    return "entity", eid  # be lenient with un-prefixed names


def id_relevant(eid: str, gold_tokens: list[str], desc_by_id: dict | None = None,
                desc_ngram: int = 3) -> bool:
    """Is element ``eid`` grounded in the gold text?

    Name match: entity name (or BOTH relation endpoints) occurs in ``gold_tokens``.
    If ``desc_by_id`` is given, an element ALSO matches when its description shares
    a contiguous ``desc_ngram``-gram with the gold text -- this brings the
    relationship/entity semantics (not just the surface name) into the decision.
    """
    kind, payload = parse_id(eid)
    if kind == "entity":
        if _contains_subsequence(gold_tokens, tokens(payload)):
            return True
    else:
        src, tgt = payload
        if (_contains_subsequence(gold_tokens, tokens(src))
                and _contains_subsequence(gold_tokens, tokens(tgt))):
            return True
    if desc_by_id:
        desc = desc_by_id.get(eid)
        if desc and _shares_ngram(tokens(desc), gold_tokens, desc_ngram):
            return True
    return False


def gt_relevant_set(universe_ids, gold_text: str, desc_by_id: dict | None = None,
                    desc_ngram: int = 3) -> set:
    """Subset of the candidate universe whose elements are grounded in GT."""
    gold_tokens = tokens(gold_text)
    return {eid for eid in universe_ids
            if id_relevant(eid, gold_tokens, desc_by_id, desc_ngram)}


def set_precision(flagged_ids, gold_set) -> dict:
    """Precision of a flagged SET against the gold-relevant set (GloRAG-Ex).

    precision = TP / (TP + FP) = TP / |flagged|, where TP = flagged elements that
    are grounded in the gold facts. ``None`` when nothing was flagged. No recall:
    a gold element absent from the (minimal) edit set is not a false negative, so
    the FN side is ill-defined here.
    """
    flagged = list(dict.fromkeys(flagged_ids))
    gold = set(gold_set)
    tp = sum(1 for f in flagged if f in gold)
    n = len(flagged)
    return {
        "precision": (tp / n) if n else None,
        "tp": tp,
        "fp": n - tp,
        "n_flagged": n,
        "gold_flagged": sorted(f for f in flagged if f in gold),
    }


def fact_coverage(element_ids, supporting_units, gold_text, desc_by_id=None, desc_ngram=3) -> dict:
    """How many ground-truth facts a graph carries (two granularities).

    element_ids      : the graph's `E::`/`R::` ids (GloRAG subgraph or attribution universe).
    supporting_units : the per-fact gold texts (HotpotQA supporting sentences / musique
                       supporting paragraphs); each is "covered" if some element is grounded in it.
    Returns: n_elements, n_gold_elements (elements grounded in gold_text), n_facts_covered, n_facts_total.
    """
    els = list(dict.fromkeys(element_ids))
    gold_tokens = tokens(gold_text)
    n_gold = sum(1 for e in els if id_relevant(e, gold_tokens, desc_by_id, desc_ngram))
    covered = 0
    for u in supporting_units:
        u_tokens = tokens(u)
        if any(id_relevant(e, u_tokens, desc_by_id, desc_ngram) for e in els):
            covered += 1
    return {"n_elements": len(els), "n_gold_elements": n_gold,
            "n_facts_covered": covered, "n_facts_total": len(supporting_units)}


def normalized_contains(haystack: str, needle: str) -> bool:
    """True if `needle` is a (normalised) substring of `haystack` (used for context coverage)."""
    hn, nn = normalize(haystack), normalize(needle)
    return bool(nn) and nn in hn


def _token_set(s: str) -> set:
    return set(tokens(s))


def span_relevant(span: str, supporting_units, jaccard_tau: float = 0.5) -> bool:
    """Does a RAG-Ex text span (a sentence/paragraph) correspond to a supporting fact?

    Matches a flagged span to a gold unit by normalised containment (either direction --
    handles sentence in paragraph and vice versa) or token-set Jaccard >= jaccard_tau
    (handles sentence ~= sentence). Brings token/span-level outputs to the GT granularity.
    """
    sn, st = normalize(span), _token_set(span)
    if not st:
        return False
    for u in supporting_units:
        un, ut = normalize(u), _token_set(u)
        if not ut:
            continue
        if sn in un or un in sn:
            return True
        inter = len(st & ut)
        if inter and inter / len(st | ut) >= jaccard_tau:
            return True
    return False


def precision_at_k(ranked_ids, positive_ids, gold_set, ks=(1, 2, 3, 5)) -> dict:
    """Precision@k of an attribution RANKING (Shapley / attribution baselines).

    The predicted-positive set at k is the top-k ranked elements restricted to
    those with a strictly positive score (``positive_ids``); precision@k =
    (#positive-top-k grounded in gold) / |positive-top-k|. ``None`` when no
    positive-score element falls within the top-k.
    """
    gold = set(gold_set)
    pos = set(positive_ids)
    prec, hit, tp_at, n_at = {}, {}, {}, {}
    for k in ks:
        flagged_k = [x for x in ranked_ids[:k] if x in pos]
        tp = sum(1 for x in flagged_k if x in gold)
        n = len(flagged_k)
        prec[str(k)] = (tp / n) if n else None     # precision@k (None if no positive in top-k)
        hit[str(k)] = 1 if tp > 0 else 0           # hit@k: does the top-k contain >=1 fact
        tp_at[str(k)] = tp
        n_at[str(k)] = n
    return {"precision_at": prec, "hit_at": hit, "tp_at": tp_at, "n_at": n_at}
