"""Text-mention correctness core (Phase 5) - LLM AS A JUDGE VERSION.

Maps flagged KG elements to a dataset's ground-truth supporting facts using a 
small local LLM to determine semantic entailment and relevance.
Uses a direct in-memory vLLM instance rather than an HTTP API.

Element identity is the shared scheme used by both methods:
  entity   -> id "E::<name>"
  relation -> id "R::<src>-><tgt>"
"""
import os
import re
from functools import lru_cache
from vllm import LLM, SamplingParams

# --- vLLM CONFIGURATION ---
VLLM_MODEL = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
_llm_instance: LLM | None = None

def get_llm() -> LLM:
    """Singleton pattern to ensure the LLM is loaded into VRAM only once."""
    global _llm_instance
    if _llm_instance is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Bind to your preferred GPU
        # Adjust gpu_memory_utilization if you need room for embeddings or other models
        _llm_instance = LLM(model=VLLM_MODEL, gpu_memory_utilization=0.6, limit_mm_per_prompt={"image": 0} )
    return _llm_instance

_WORD = re.compile(r"[a-z0-9]+")

def normalize(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    return " ".join(_WORD.findall((s or "").lower()))

def tokens(s: str) -> list[str]:
    return normalize(s).split()

def parse_id(eid: str):
    """('entity', name) | ('relation', (src, tgt)) for an "E::"/"R::" id."""
    if eid.startswith("E::"):
        return "entity", eid[3:]
    if eid.startswith("R::"):
        src, _, tgt = eid[3:].partition("->")
        return "relation", (src, tgt)
    return "entity", eid  # be lenient with un-prefixed names

_SYSTEM_PROMPT = ("You are a strict validator. Answer ONLY with 'YES' or 'NO'. "
                  "Do not explain your reasoning.")


@lru_cache(maxsize=200000)
def _judge_yes_no(user_prompt: str) -> bool:
    llm = get_llm()
    messages = [{"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8)  # only need YES/NO
    try:
        outputs = llm.chat(messages, sampling_params, use_tqdm=False)
        text = outputs[0].outputs[0].text.strip().upper()
        return "YES" in text[:8]
    except Exception as e:
        print(f"vLLM Inference Error: {e}")
        return False


def query_llm_judge(gold_text: str, element_type: str, payload: str, description: str) -> bool:
    user_prompt = f"""You are given the Supporting Facts for a question and a Knowledge Graph Element retrieved by a system.

Ground Truth Text: "{gold_text}"

Knowledge Graph Element:
- Type: {element_type}
- Name/Endpoints: {payload}
- Description: {description}

Task: Does the Knowledge Graph Element contain, restate, or correspond to at least one of the Supporting Facts -- even if the span also contains other, unrelated information?"""
    return _judge_yes_no(user_prompt)


def query_span_judge(gold_text: str, span: str) -> bool:
    user_prompt = f"""You are given the Supporting Facts for a question and a Candidate Text Span retrieved by a system.

Supporting Facts: "{gold_text}"

Candidate Text Span: "{span}"

Task: Does the Candidate Text Span contain, restate, or correspond to at least one of the Supporting Facts -- even if the span also contains other, unrelated information?"""
    return _judge_yes_no(user_prompt)

def id_relevant(eid: str, gold_text: str, desc_by_id: dict | None = None, **kwargs) -> bool:
    """Is element ``eid`` grounded in the gold text using LLM semantic entailment?"""
    if not gold_text.strip():
        return False
        
    kind, payload = parse_id(eid)

    # Name/endpoints for the prompt: "Name" for an entity, "src -> tgt" for a relation.
    payload_str = f"{payload[0]} -> {payload[1]}" if kind == "relation" else payload

    # Entity/relation description (used together with the name -- graph elements are judged
    # by name + description).
    desc = desc_by_id.get(eid, "None provided") if desc_by_id else "None provided"

    return query_llm_judge(gold_text, kind, payload_str, desc)

def gt_relevant_set(universe_ids, gold_text: str, desc_by_id: dict | None = None, **kwargs) -> set:
    """Subset of the candidate universe whose elements are grounded in GT."""
    return {eid for eid in universe_ids
            if id_relevant(eid, gold_text, desc_by_id)}

def set_precision(flagged_ids, gold_set) -> dict:
    """Precision of a flagged SET against the gold-relevant set (GloRAG-Ex)."""
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

def fact_coverage(element_ids, supporting_units, gold_text, desc_by_id=None, **kwargs) -> dict:
    """How many ground-truth facts a graph carries."""
    els = list(dict.fromkeys(element_ids))
    
    n_gold = sum(1 for e in els if id_relevant(e, gold_text, desc_by_id))
    
    covered = 0
    for u in supporting_units:
        if any(id_relevant(e, u, desc_by_id) for e in els):
            covered += 1
            
    return {"n_elements": len(els), "n_gold_elements": n_gold,
            "n_facts_covered": covered, "n_facts_total": len(supporting_units)}

def normalized_contains(haystack: str, needle: str) -> bool:
    """True if `needle` is a (normalised) substring of `haystack`."""
    hn, nn = normalize(haystack), normalize(needle)
    return bool(nn) and nn in hn

def _token_set(s: str) -> set:
    return set(tokens(s))

def span_relevant(span: str, gold_text: str, **kwargs) -> bool:
    """Text-span relevance via the SAME LLM judge (RAG-Ex / Shapley-Text chunks/sentences).

    The flagged element is the span text itself; it is judged for entailment against the
    supporting facts (gold_text), mirroring how graph elements are judged by name+description.
    """
    if not span or not span.strip() or not gold_text or not gold_text.strip():
        return False
    return query_span_judge(gold_text, span)

def precision_at_k(ranked_ids, positive_ids, gold_set, ks=(1, 2, 3, 5)) -> dict:
    """Precision@k of an attribution RANKING (Shapley / attribution baselines)."""
    gold = set(gold_set)
    pos = set(positive_ids)
    prec, hit, tp_at, n_at = {}, {}, {}, {}
    for k in ks:
        flagged_k = [x for x in ranked_ids[:k] if x in pos]
        tp = sum(1 for x in flagged_k if x in gold)
        n = len(flagged_k)
        prec[str(k)] = (tp / n) if n else None     
        hit[str(k)] = 1 if tp > 0 else 0           
        tp_at[str(k)] = tp
        n_at[str(k)] = n
    return {"precision_at": prec, "hit_at": hit, "tp_at": tp_at, "n_at": n_at}