from __future__ import annotations

import random
import networkx as nx
import numpy as np

from dataclasses import dataclass
from scipy.stats import wasserstein_distance
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

from src.llm.utils import vllm_model_complete
from src.parser import graph_to_context, parse_graph, parse_context
from lightrag import LightRAG, QueryParam
from src.query import query as llm_query
from lightrag.prompt import PROMPTS
from src.llm_judge import judge_response


# ─────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────

print("Final Implementation!")

@dataclass
class KGSMILEResult:
    original_response:     str
    edge_attributions:     dict[tuple[str, str], float]
    node_attributions:     dict[str, float]
    top_edges:             list[tuple[str, str]]
    surrogate_r2:          float
    output_shift_std:      float
    mean_graph_cosine_sim: float
    mean_kernel_weight:    float
    llm_call_count:        int  = 0
    noise_robust:          bool = True
    degenerate:            bool = False


def result_to_dict(result: KGSMILEResult) -> dict:
    return {
        "edge_attributions": [
            {"source": src, "target": tgt, "attribution": score}
            for (src, tgt), score in result.edge_attributions.items()
        ],
        "node_attributions": [
            {"node": node, "attribution": score}
            for node, score in result.node_attributions.items()
        ],
        "noise_robust": result.noise_robust,
        "degenerate":   result.degenerate,
    }


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class KGSMILEConfig:
    n_perturbations: int   = 20
    kernel_width:    float = 0.25   # σ in Eq. 12
    retrieval_mode:  str   = "hybrid"
    retrieval_top_k: int   = 2
    random_seed:     int   = 42
    max_tokens:      int   = 512
    embedding_model: str   = "all-MiniLM-L6-v2"
    noise_pct:       float = 0.0


# ─────────────────────────────────────────────────────────────
# FULL KG LOADING
# ─────────────────────────────────────────────────────────────

def load_full_kg(path: str) -> nx.Graph:
    print(f"[KG-SMILE] Loading FULL KG from {path}")
    G = nx.read_graphml(path)
    print(f"[KG-SMILE] Full KG: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")
    return G


# ─────────────────────────────────────────────────────────────
# NOISE INJECTION
# ─────────────────────────────────────────────────────────────

def add_random_noise_nodes(
    cg:        nx.Graph,
    KG:        nx.Graph,
    n:         int   = None,
    noise_pct: float = None,
    seed:      int   = None,
):
    if seed is not None:
        random.seed(seed)

    if noise_pct is not None:
        n = max(1, round(len(cg.nodes()) * noise_pct))
    elif n is None:
        raise ValueError("Either n or noise_pct must be provided.")

    cg = cg.copy()
    ops_applied = []

    candidate_nodes = [node for node in KG.nodes() if node not in cg.nodes()]
    if not candidate_nodes:
        print("[KG-SMILE] No candidate nodes in FULL KG outside subgraph.")
        return cg, ops_applied

    eligible_anchors = list(cg.nodes())
    if not eligible_anchors:
        print("[KG-SMILE] No anchors in subgraph. Skipping noise.")
        return cg, ops_applied

    all_KG_edges  = list(KG.edges(data=True))
    sampled_nodes = random.sample(candidate_nodes, min(n, len(candidate_nodes)))

    for new_node in sampled_nodes:
        anchor = random.choice(eligible_anchors)
        node_attr = KG.nodes[new_node]
        _, _, random_edge_attr = random.choice(all_KG_edges)

        cg.add_node(new_node, **node_attr)
        cg.add_edge(new_node, anchor, **random_edge_attr)

        ops_applied.append(("add_node", new_node))
        ops_applied.append(("add_edge", (new_node, anchor)))

    print(f"[KG-SMILE] Added {len(sampled_nodes)} noise nodes")
    return cg, ops_applied


# ─────────────────────────────────────────────────────────────
# GRAPH RETRIEVAL
# ─────────────────────────────────────────────────────────────

async def retrieve_graph(rag: LightRAG, query: str, mode: str, top_k: int) -> nx.Graph:
    param = QueryParam(
        mode=mode,
        only_need_context=True,
        enable_rerank=False,
        top_k=top_k,
        include_references=False,
    )
    context = await rag.aquery(query, param=param)
    return parse_graph(parse_context(context))


# ─────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────

def _build_system_prompt(context: str) -> str:
    return PROMPTS["rag_response"].format(
        context_data=context,
        response_type="Single Sentence, without references and extra explanations.",
        user_prompt=""
    )


async def _query(query: str, G: nx.Graph, max_tokens: int) -> str:
    context = graph_to_context(G)
    return await vllm_model_complete(
        query,
        system_prompt=_build_system_prompt(context),
        temperature=0,
        max_tokens=max_tokens,
    )


# ─────────────────────────────────────────────────────────────
# PERTURBATION HELPERS
# ─────────────────────────────────────────────────────────────

def _extract_triples(G: nx.Graph) -> list[tuple[str, str, str]]:
    """Extract (src, description, tgt) triples from graph edges.
    Uses 'description' to match the key stored by parse_graph / graph_to_context.
    """
    return [
        (u, d.get("description", ""), v)
        for u, v, d in G.edges(data=True)
    ]


def _graph_to_binary_mask(
    surviving_triples: list[tuple[str, str, str]],
    all_triples:       list[tuple[str, str, str]],
) -> np.ndarray:
    """1 if triple survived perturbation, 0 otherwise."""
    surviving_set = set(surviving_triples)
    return np.array([1 if t in surviving_set else 0 for t in all_triples], dtype=float)


def _perturb_graph(
    all_triples: list[tuple[str, str, str]],
    rng:         random.Random,
) -> tuple[list[tuple[str, str, str]], list[str], np.ndarray]:
    """Randomly remove triples; derive surviving entities from them.

    Returns:
        surviving_triples  : subset of all_triples that remain
        surviving_entities : unique entities parsed from surviving_triples
                             (order-preserving, no isolated nodes possible)
        mask               : binary array over all_triples (1=kept, 0=removed)
    """
    num_to_remove   = rng.randint(1, len(all_triples))
    removed_indices = set(rng.sample(range(len(all_triples)), num_to_remove))

    surviving_triples = [t for i, t in enumerate(all_triples) if i not in removed_indices]
    mask              = _graph_to_binary_mask(surviving_triples, all_triples)

    # Entities derived purely from surviving triples — no isolated nodes possible
    seen               = set()
    surviving_entities = []
    for (src, _, tgt) in surviving_triples:
        for entity in (src, tgt):
            if entity not in seen:
                seen.add(entity)
                surviving_entities.append(entity)

    return surviving_triples, surviving_entities, mask


def _build_perturbed_graph(
    surviving_triples:  list[tuple[str, str, str]],
    surviving_entities: list[str],
    G_original:         nx.Graph,
) -> nx.Graph:
    G_p = nx.Graph()
    for entity in surviving_entities:
        attrs = G_original.nodes[entity] if entity in G_original else {}
        G_p.add_node(entity, **attrs)
    for src, desc, tgt in surviving_triples:
        G_p.add_edge(src, tgt, description=desc)
    return G_p


def _inverse_wasserstein(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    epsilon = 1e-6
    wd = wasserstein_distance(emb_a.flatten(), emb_b.flatten())
    return 1.0 / (wd + epsilon)


def _scale_inv_wds(raw_inv_wds: list[float]) -> list[float]:
    min_v = min(raw_inv_wds)
    max_v = max(raw_inv_wds)
    if min_v == max_v:
        return [1.0] * len(raw_inv_wds)
    return [(v - min_v) / (max_v - min_v) for v in raw_inv_wds]


def _kernel_weight(graph_cos_sim: float, sigma: float) -> float:
    return float(np.exp(-(graph_cos_sim ** 2) / (sigma ** 2)))


# ─────────────────────────────────────────────────────────────
# EARLY EXIT HELPER
# ─────────────────────────────────────────────────────────────

def _zero_result(
    original_response:     str,
    all_triples:           list[tuple[str, str, str]],
    all_nodes:             list[str],
    llm_call_count:        int,
    mean_graph_cosine_sim: float = 0.0,
    noise_robust:          bool  = True,
) -> KGSMILEResult:
    """Return a zeroed-out result when perturbations are insufficient."""
    return KGSMILEResult(
        original_response=original_response,
        edge_attributions={(src, tgt): 0.0 for src, _, tgt in all_triples},
        node_attributions={n: 0.0 for n in all_nodes},
        top_edges=[],
        surrogate_r2=0.0,
        output_shift_std=0.0,
        mean_graph_cosine_sim=mean_graph_cosine_sim,
        mean_kernel_weight=0.0,
        llm_call_count=llm_call_count,
        noise_robust=noise_robust,
        degenerate=True,
    )


# ─────────────────────────────────────────────────────────────
# MAIN KG-SMILE ENTRY POINT
# ─────────────────────────────────────────────────────────────

async def run_kg_smile(
    query:        str,
    rag:          LightRAG,
    KG_full:      nx.Graph,
    config:       KGSMILEConfig | None = None,
    ground_truth: str | None           = None,
    mode:         str                  = "ft",
) -> KGSMILEResult:
    llm_call_count = 0
    noise_robust   = True

    if config is None:
        config = KGSMILEConfig()

    rng       = random.Random(config.random_seed)
    emb_model = SentenceTransformer(config.embedding_model)

    print("[KG-SMILE] Retrieving subgraph...")
    G = await retrieve_graph(rag, query, config.retrieval_mode, config.retrieval_top_k)
    print(f"[KG-SMILE] Subgraph: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")

    G_clean = G.copy()

    if mode == "ft":
        original_response = await llm_query(rag=rag, context=graph_to_context(G_clean), question=query)
        llm_call_count += 1
    elif mode == "ff":
        original_response = ground_truth

    original_emb       = emb_model.encode([original_response])
    graph_emb_baseline = emb_model.encode([graph_to_context(G_clean)])

    if config.noise_pct > 0:
        print(f"[KG-SMILE] Injecting noise: {config.noise_pct * 100:.0f}%")
        G, _ = add_random_noise_nodes(
            cg=G_clean, KG=KG_full, noise_pct=config.noise_pct, seed=config.random_seed
        )

        noisy_response = await llm_query(rag=rag, context=graph_to_context(G), question=query)
        llm_call_count += 1

        judge_score = await judge_response(
            question=query,
            generated_answer=noisy_response,
            ground_truth=original_response,
        )

        if judge_score == 0:
            noise_robust = False
            print("[KG-SMILE] Noise altered the answer (judge=0). "
                  "Skipping — perturbations unreliable.")
        else:
            print("[KG-SMILE] Noise check passed (judge=1). noise_robust=True.")

        graph_emb_baseline = emb_model.encode([graph_to_context(G)])

    # ── 2. Build triple index ─────────────────────────────────
    all_triples = _extract_triples(G)
    all_nodes   = list(G.nodes())

    # ── 3. Perturbation loop ──────────────────────────────────
    masks        = []
    raw_inv_wds  = []
    kernel_weights = []
    cosine_sims  = []

    for _ in range(config.n_perturbations):
        surviving_triples, surviving_entities, z_p = _perturb_graph(all_triples, rng)

        if not surviving_triples:
            continue

        # Rebuild graph from surviving triples; entities derived from them
        G_p = _build_perturbed_graph(surviving_triples, surviving_entities, G)

        response_p = await _query(query, G_p, config.max_tokens)
        llm_call_count += 1
        emb_p = emb_model.encode([response_p])

        # Text similarity: raw inverse Wasserstein distance (Eq. 11)
        # Will be min-max scaled across all perturbations after the loop
        inv_wd = _inverse_wasserstein(original_emb, emb_p)

        # Graph structure similarity: cosine between graph context embeddings (Eq. 10)
        graph_emb_p   = emb_model.encode([graph_to_context(G_p)])
        graph_cos_sim = float(cosine_similarity(graph_emb_baseline, graph_emb_p)[0][0])

        # Kernel weight on graph cosine similarity (Eq. 12)
        kw = _kernel_weight(graph_cos_sim, config.kernel_width)

        masks.append(z_p)
        raw_inv_wds.append(inv_wd)
        kernel_weights.append(kw)
        cosine_sims.append(graph_cos_sim)

    # ── 4. Fit surrogate linear model ─────────────────────────
    if len(masks) < 2:
        print("[KG-SMILE] Not enough valid perturbations survived.")
        return _zero_result(
            original_response, all_triples, all_nodes, llm_call_count,
            mean_graph_cosine_sim=float(np.mean(cosine_sims)) if cosine_sims else 0.0,
            noise_robust=noise_robust,
        )

    # Min-max scale inv_wd across all perturbations — higher = more similar to original
    scaled_y = _scale_inv_wds(raw_inv_wds)

    X = np.array(masks)          # shape: (n_perturbations, n_triples)
    y = np.array(scaled_y)       # shape: (n_perturbations,)
    w = np.array(kernel_weights) # shape: (n_perturbations,)

    reg    = LinearRegression()
    reg.fit(X, y, sample_weight=w)
    r2     = float(reg.score(X, y, sample_weight=w))
    coeffs = reg.coef_           # one coefficient per triple

    # ── 5. Build attribution dicts ────────────────────────────
    # Each triple coefficient → edge attribution.
    # Node attribution = mean of its incident triple coefficients.
    edge_attributions:    dict[tuple[str, str], float] = {}
    node_attribution_acc: dict[str, list[float]]       = {}

    for i, (src, desc, tgt) in enumerate(all_triples):
        coeff = float(coeffs[i])
        edge_attributions[(src, tgt)] = coeff
        for entity in (src, tgt):
            node_attribution_acc.setdefault(entity, []).append(coeff)

    node_attributions = {
        n: float(np.mean(v)) for n, v in node_attribution_acc.items()
    }

    # Safety: cover any node in G that had no triples
    for n in all_nodes:
        if n not in node_attributions:
            node_attributions[n] = 0.0

    top_edges = sorted(
        edge_attributions,
        key=lambda e: abs(edge_attributions[e]),
        reverse=True,
    )

    return KGSMILEResult(
        original_response=original_response,
        edge_attributions=edge_attributions,
        node_attributions=node_attributions,
        top_edges=top_edges,
        surrogate_r2=r2,
        output_shift_std=float(np.std(scaled_y)),
        mean_graph_cosine_sim=float(np.mean(cosine_sims)),
        mean_kernel_weight=float(np.mean(kernel_weights)),
        llm_call_count=llm_call_count,
        noise_robust=noise_robust,
    )