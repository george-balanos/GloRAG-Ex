"""Empirical optimality check: minimum-cost CFE must not include injected noise.

Runs find_counterfactuals on a few benchmark queries with a single irrelevant
(node, edge) pair injected. Since deleting/replacing noise cannot flip the
output, the optimal sigma should touch real evidence, not the noise.
"""

import asyncio
import json
import os

from src.retrieve import initialize_lightrag, retrieve_subgraph, sentence_transformer_embed
from src.counterfactuals.generate import find_counterfactuals, G as KG
from src.counterfactuals.robustness import inject_noise
from src.parser import parse_context, parse_graph, graph_to_context


def _ops_mention(ops, target_node, target_edge):
    for op in ops or []:
        kind = op[0]
        payload = op[1] if len(op) > 1 else None
        if kind in ("delete_node", "add_node") and payload == target_node:
            return True
        if kind == "replace_node" and isinstance(payload, tuple) and payload[0] == target_node:
            return True
        if kind in ("delete_edge", "add_edge") and tuple(payload) in (target_edge, target_edge[::-1]):
            return True
        if kind == "replace_edge" and isinstance(payload, tuple) and tuple(payload[0]) in (target_edge, target_edge[::-1]):
            return True
    return False


async def main(n_queries: int = 5):
    rag = await initialize_lightrag()
    with open("benchmark/results/comparison.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    table = []
    count = 0
    for idx, r in data["results"].items():
        if count >= n_queries:
            break
        if r.get("case") != "ft":
            continue
        question = r["question"]
        context = await retrieve_subgraph(rag, query=question, mode="hybrid", top_k=2)

        q_emb = (await sentence_transformer_embed([question]))[0]
        cg = parse_graph(parse_context(context))
        noisy_cg, v_noise, e_noise = inject_noise(cg, KG, q_emb)
        if v_noise is None:
            continue
        noisy_context = graph_to_context(noisy_cg)

        ops = await find_counterfactuals(rag, question, noisy_context, max_cost=5, max_llm_calls=50)
        appears = _ops_mention(ops, v_noise, e_noise)

        table.append((idx, question[:50], v_noise, appears, ops))
        count += 1

    print("\n=== Optimality validation ===")
    print(f"{'idx':<6}{'question':<55}{'noise_node':<25}{'noise_in_ops':<14}")
    for idx, q, v, app, _ in table:
        print(f"{idx:<6}{q:<55}{str(v):<25}{str(app):<14}")
    fails = [t for t in table if t[3]]
    print(f"\n{len(table)} queries; {len(fails)} include noise (expect 0).")


def cost_unit_check():
    from src.counterfactuals.edit_costs import (
        replace_edge_cost, replace_node_cost, add_edge_cost
    )
    import numpy as np
    import networkx as nx

    e = np.array([1.0, 0.0, 0.0])
    assert abs(replace_edge_cost(e, e) - 1.0) < 1e-6, "rep_e self should be 1.0"

    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3)])
    cost = replace_node_cost(e, e, G, 1)
    assert abs(cost - (1 + 2)) < 1e-6, f"rep_n self with 2 edges should be 3, got {cost}"

    print("cost_unit_check passed")


if __name__ == "__main__":
    cost_unit_check()
    asyncio.run(main())
