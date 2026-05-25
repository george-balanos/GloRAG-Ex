"""Empirical optimality check: minimum-cost CFE must not include injected noise.

Runs find_counterfactuals on a few benchmark queries with a single irrelevant
(node, edge) pair injected. Since deleting/replacing noise cannot flip the
output, the optimal sigma should touch real evidence, not the noise.
"""

import argparse
import asyncio
import csv
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


def parse_args():
    p = argparse.ArgumentParser(description="Optimality validation via noise injection.")
    p.add_argument("--input", default="benchmark/results/comparison.json",
                   help="Path to comparison.json")
    p.add_argument("--case", choices=["all", "tf", "ft"], default="all")
    p.add_argument("--n-queries", type=int, default=5,
                   help="How many entries to probe")
    p.add_argument("--max-cost", type=float, default=5.0)
    p.add_argument("--max-llm-calls", type=int, default=50)
    p.add_argument("--unit-cost", action="store_true")
    p.add_argument("--ops", nargs="+",
                   default=["delete_node", "delete_edge", "replace_node",
                            "replace_edge", "add_node", "add_edge"])
    p.add_argument("--use-psp", action="store_true")
    p.add_argument("--max-pivots", type=int, default=3)
    p.add_argument("--retrieve-mode", default="hybrid")
    p.add_argument("--retrieve-top-k", type=int, default=2)
    p.add_argument("--f1-mode", choices=["type-only", "strict-label", "off"], default="type-only")
    p.add_argument("--add-mode", choices=["expand", "retrieve", "both"], default="both")
    p.add_argument("--replace-mode", choices=["atomic", "decomposed"], default="atomic")
    p.add_argument("--judge-against", choices=["original", "ground_truth"], default="original")
    p.add_argument("--out", default="benchmark/results/optimality.csv",
                   help="CSV summary path")
    return p.parse_args()


async def main():
    args = parse_args()
    rag = await initialize_lightrag()
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    table = []
    count = 0
    for idx, r in data["results"].items():
        if count >= args.n_queries:
            break
        case = r.get("case", "all")
        if args.case != "all" and case != args.case:
            continue

        question = r["question"]
        context = await retrieve_subgraph(rag, query=question,
                                          mode=args.retrieve_mode,
                                          top_k=args.retrieve_top_k)

        q_emb = (await sentence_transformer_embed([question]))[0]
        cg = parse_graph(parse_context(context))
        noisy_cg, v_noise, e_noise = inject_noise(cg, KG, q_emb)
        if v_noise is None:
            continue
        noisy_context = graph_to_context(noisy_cg)

        ops = await find_counterfactuals(
            rag, question, noisy_context,
            max_cost=args.max_cost,
            max_llm_calls=args.max_llm_calls,
            unit_cost=args.unit_cost,
            current_ops=args.ops,
            use_pivotal_probe=args.use_psp,
            max_pivots=args.max_pivots,
            suffix="_validate",
            f1_mode=args.f1_mode,
            add_mode=args.add_mode,
            replace_mode=args.replace_mode,
            judge_against=args.judge_against,
            ground_truth=r.get("ground_truth", ""),
        )
        appears = _ops_mention(ops, v_noise, e_noise)

        table.append({
            "idx": idx,
            "case": case,
            "question": question,
            "noise_node": str(v_noise),
            "noise_edge": str(e_noise),
            "n_ops": len(ops or []),
            "noise_in_ops": appears,
        })
        count += 1

    print("\n=== Optimality validation ===")
    print(f"{'idx':<6}{'case':<6}{'question':<55}{'noise_node':<25}{'noise_in_ops':<14}")
    for row in table:
        print(f"{row['idx']:<6}{row['case']:<6}{row['question'][:50]:<55}"
              f"{row['noise_node']:<25}{str(row['noise_in_ops']):<14}")
    fails = [t for t in table if t["noise_in_ops"]]
    print(f"\n{len(table)} queries; {len(fails)} include noise (expect 0).")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0].keys()) if table else
                                    ["idx", "case", "question", "noise_node",
                                     "noise_edge", "n_ops", "noise_in_ops"])
            writer.writeheader()
            for row in table:
                writer.writerow(row)
        print(f"CSV saved to: {args.out}")


def cost_unit_check():
    from src.counterfactuals.edit_costs import (
        replace_edge_cost, replace_node_cost
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
