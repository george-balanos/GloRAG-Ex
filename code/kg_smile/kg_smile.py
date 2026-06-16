"""
KG-SMILE Pipeline (ROBUSTNESS-READY VERSION)
"""

from __future__ import annotations

import math
import random
import networkx as nx
import numpy as np

from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

from src.llm.utils import vllm_model_complete
from src.parser import graph_to_context, parse_graph, parse_context
from lightrag import LightRAG, QueryParam
from src.query import query as llm_query

# ─────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────

# @dataclass
# class KGSMILEResult:
#     original_response:     str
#     edge_attributions:     dict[tuple[str, str], float]  # (src, tgt) -> score
#     node_attributions:     dict[str, float]              # node -> score
#     top_edges:             list[tuple[str, str]]
#     surrogate_r2:          float
#     output_shift_std:      float
#     mean_graph_cosine_sim: float
#     mean_kernel_weight:    float

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
    llm_call_count:        int = 0 

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
    }


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class KGSMILEConfig:
    n_perturbations: int   = 20
    kernel_width:    float = 0.25
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
    cg: nx.Graph,
    KG: nx.Graph,
    n: int = None,
    noise_pct: float = None,
    seed: int = None,
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

_SYSTEM_PROMPT = "Answer using ONLY the knowledge graph provided. Be concise."


async def _query(query: str, G: nx.Graph, max_tokens: int) -> str:
    context = graph_to_context(G)
    return await vllm_model_complete(
        prompt=f"{context}\n\nQuestion: {query}",
        system_prompt=_SYSTEM_PROMPT,
        temperature=0,
        max_tokens=max_tokens,
    )


# ─────────────────────────────────────────────────────────────
# PERTURBATION HELPERS
# ─────────────────────────────────────────────────────────────

def _graph_to_binary_mask(G: nx.Graph, all_nodes: list, all_edges: list) -> np.ndarray:
    """1 if node/edge present in G, 0 otherwise."""
    node_mask = [1 if n in G.nodes() else 0 for n in all_nodes]
    edge_mask = [1 if G.has_edge(u, v) else 0 for u, v in all_edges]
    return np.array(node_mask + edge_mask, dtype=float)


def _perturb_graph(
    G: nx.Graph,
    all_nodes: list,
    all_edges: list,
    rng: random.Random,
) -> tuple[nx.Graph, np.ndarray]:
    """Randomly drop nodes and edges; return perturbed graph + its binary mask."""
    G_p = G.copy()

    # randomly remove ~half the nodes
    for node in list(G_p.nodes()):
        if rng.random() < 0.5:
            G_p.remove_node(node)

    # randomly remove ~half the remaining edges
    for u, v in list(G_p.edges()):
        if rng.random() < 0.5:
            G_p.remove_edge(u, v)

    mask = _graph_to_binary_mask(G_p, all_nodes, all_edges)
    return G_p, mask


def _kernel_weight(z: np.ndarray, z0: np.ndarray, width: float) -> float:
    """Exponential kernel on Hamming distance."""
    d = np.sum(z != z0) / max(len(z), 1)
    return float(np.exp(-(d ** 2) / (width ** 2)))


# ─────────────────────────────────────────────────────────────
# MAIN KG-SMILE ENTRY POINT
# ─────────────────────────────────────────────────────────────

async def run_kg_smile(
    query: str,
    rag: LightRAG,
    KG_full: nx.Graph,
    config: KGSMILEConfig | None = None,
    ground_truth: str | None = None,
    mode = "ft"
) -> KGSMILEResult:
    llm_call_count = 0

    if config is None:
        config = KGSMILEConfig()

    rng       = random.Random(config.random_seed)
    emb_model = SentenceTransformer(config.embedding_model)

    # ── 1. Retrieve subgraph ──────────────────────────────────
    print("[KG-SMILE] Retrieving subgraph...")
    G = await retrieve_graph(rag, query, config.retrieval_mode, config.retrieval_top_k)
    print(f"[KG-SMILE] Subgraph: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")

    # ── 2. Inject noise ───────────────────────────────────────
    if config.noise_pct > 0:
        print(f"[KG-SMILE] Injecting noise: {config.noise_pct*100:.0f}%")
        G, _ = add_random_noise_nodes(
            cg=G, KG=KG_full, noise_pct=config.noise_pct, seed=config.random_seed
        )

    # ── 3. Baseline response ──────────────────────────────────
    # original_response = await _query(query, G, config.max_tokens)
    if mode == "ft":
        original_response = await llm_query(rag=rag, context=graph_to_context(G), question=query)
        llm_call_count += 1
    elif mode == "ff":
        original_response = ground_truth
        
    original_emb      = emb_model.encode([original_response])

    # ── 4. Build feature index ────────────────────────────────
    all_nodes = list(G.nodes())
    all_edges = list(G.edges())
    z0        = _graph_to_binary_mask(G, all_nodes, all_edges)

    # ── 5. Perturbation loop ──────────────────────────────────
    masks          = []
    output_shifts  = []
    kernel_weights = []
    cosine_sims    = []

    for _ in range(config.n_perturbations):
        G_p, z_p = _perturb_graph(G, all_nodes, all_edges, rng)

        if G_p.number_of_nodes() == 0:
            continue

        response_p = await _query(query, G_p, config.max_tokens)
        llm_call_count += 1
        emb_p      = emb_model.encode([response_p])

        cos_sim = float(cosine_similarity(original_emb, emb_p)[0][0])
        kw      = _kernel_weight(z_p, z0, config.kernel_width)

        masks.append(z_p)
        output_shifts.append(1.0 - cos_sim)   # shift = how much the answer changed
        kernel_weights.append(kw)
        cosine_sims.append(cos_sim)

    # ── 6. Fit surrogate linear model ─────────────────────────
    if len(masks) < 2:
        # degenerate — not enough perturbations survived
        zero_edge_attr = {(u, v): 0.0 for u, v in all_edges}
        zero_node_attr = {n: 0.0 for n in all_nodes}
        return KGSMILEResult(
            original_response=original_response,
            edge_attributions=zero_edge_attr,
            node_attributions=zero_node_attr,
            top_edges=[],
            surrogate_r2=0.0,
            output_shift_std=0.0,
            mean_graph_cosine_sim=float(np.mean(cosine_sims)) if cosine_sims else 0.0,
            mean_kernel_weight=float(np.mean(kernel_weights)) if kernel_weights else 0.0,
            llm_call_count=llm_call_count,
        )

    X      = np.array(masks)
    y      = np.array(output_shifts)
    w      = np.array(kernel_weights)

    reg    = LinearRegression()
    reg.fit(X, y, sample_weight=w)
    r2     = float(reg.score(X, y, sample_weight=w))
    coeffs = reg.coef_   # one per node + one per edge

    n_nodes = len(all_nodes)
    node_coeffs = coeffs[:n_nodes]
    edge_coeffs = coeffs[n_nodes:]

    # ── 7. Build attribution dicts ────────────────────────────
    node_attributions = {n: float(node_coeffs[i]) for i, n in enumerate(all_nodes)}
    edge_attributions = {(u, v): float(edge_coeffs[i]) for i, (u, v) in enumerate(all_edges)}

    top_edges = sorted(edge_attributions, key=lambda e: abs(edge_attributions[e]), reverse=True)

    return KGSMILEResult(
        original_response=original_response,
        edge_attributions=edge_attributions,
        node_attributions=node_attributions,
        top_edges=top_edges,
        surrogate_r2=r2,
        output_shift_std=float(np.std(output_shifts)),
        mean_graph_cosine_sim=float(np.mean(cosine_sims)),
        mean_kernel_weight=float(np.mean(kernel_weights)),
    )