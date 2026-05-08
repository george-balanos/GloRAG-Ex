import json
import networkx as nx
from pathlib import Path


def build_graph_from_subgraph(subgraph: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for entity in subgraph.get("entities", []):
        G.add_node(entity["name"], **{k: v for k, v in entity.items() if k != "name"})
    for rel in subgraph.get("relations", []):
        G.add_edge(rel["src"], rel["tgt"], **{k: v for k, v in rel.items() if k not in ("src", "tgt")})
    return G


def check_single_op_feasibility(G: nx.DiGraph, op, op_type: str):
    """Check if a single operation is feasible on graph G. Returns (is_feasible, reason)."""
    U = G.to_undirected()
    cut_vertices = set(nx.articulation_points(U))
    bridges = set(nx.bridges(U))

    if op_type == "delete_node" and isinstance(op, str):
        is_cut = op in cut_vertices
        return not is_cut, "cut vertex" if is_cut else None

    elif op_type == "delete_edge" and isinstance(op, list) and len(op) == 2 and isinstance(op[0], str):
        edge = tuple(op)
        is_bridge = edge in bridges or (edge[1], edge[0]) in bridges
        return not is_bridge, "bridge" if is_bridge else None

    elif op_type == "replace_node" and isinstance(op, list) and len(op) == 2 and isinstance(op[0], str):
        old = op[0]
        is_cut = old in cut_vertices
        return not is_cut, "cut vertex" if is_cut else None

    elif op_type == "replace_edge" and isinstance(op, list) and len(op) == 2 and isinstance(op[0], list):
        old_edge = tuple(op[0])
        is_bridge = old_edge in bridges or (old_edge[1], old_edge[0]) in bridges
        return not is_bridge, "bridge" if is_bridge else None

    return True, None


def apply_op(G: nx.DiGraph, op, op_type: str) -> nx.DiGraph:
    """Apply an operation to G and return the modified copy."""
    G = G.copy()

    if op_type == "delete_node" and isinstance(op, str):
        if op in G:
            G.remove_node(op)

    elif op_type == "delete_edge" and isinstance(op, list) and len(op) == 2 and isinstance(op[0], str):
        edge = tuple(op)
        if G.has_edge(*edge):
            G.remove_edge(*edge)

    elif op_type == "replace_node" and isinstance(op, list) and len(op) == 2 and isinstance(op[0], str):
        old, new = op
        if old in G:
            G = nx.relabel_nodes(G, {old: new})

    elif op_type == "replace_edge" and isinstance(op, list) and len(op) == 2 and isinstance(op[0], list):
        old_edge = tuple(op[0])
        # For feasibility purposes just keep the edge endpoints, attrs don't matter here
        if G.has_edge(*old_edge):
            pass  # replace_edge keeps same (u,v), only attrs change — always structurally feasible

    return G


def check_operation_feasibility(record: dict) -> dict:
    """
    Returns a summary dict for this record:
      - all_feasible: True if every op was feasible on the remaining graph
      - op_results: list of (op, feasible, reason, checked_on_reduced_graph)
    """
    op_type = record.get("operation_type", "")
    operations = record.get("operations", [])
    subgraph = record.get("original_subgraph")

    if not subgraph:
        return {"all_feasible": None, "op_results": []}

    G = build_graph_from_subgraph(subgraph)
    U = G.to_undirected()

    cut_vertices = set(nx.articulation_points(U))
    bridges = set(nx.bridges(U))

    print(f"\n{'=' * 50}")
    print(f"Question:       {record['question']}")
    print(f"Operation type: {op_type}")
    print(f"Found:          {record['found']}")
    print(f"Graph:          {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Cut vertices:   {cut_vertices or 'none'}")
    print(f"Bridges:        {bridges or 'none'}")

    if not operations:
        print("No operations to check.")
        return {"all_feasible": None, "op_results": []}

    print("Operation feasibility:")

    current_G = G.copy()
    op_results = []
    all_feasible = True

    for i, op in enumerate(operations):
        is_feasible, reason = check_single_op_feasibility(current_G, op, op_type)
        reduced = i > 0  # whether we're checking on a reduced graph

        label = f"(on reduced graph after {i} op(s))" if reduced else ""
        status = "feasible" if is_feasible else f"INFEASIBLE ({reason})"

        if op_type == "delete_node" and isinstance(op, str):
            print(f"  [{i+1}] DELETE NODE '{op}' {label} -> {status}")
        elif op_type == "delete_edge" and isinstance(op, list) and isinstance(op[0], str):
            print(f"  [{i+1}] DELETE EDGE {tuple(op)} {label} -> {status}")
        elif op_type == "replace_node" and isinstance(op, list) and isinstance(op[0], str):
            print(f"  [{i+1}] REPLACE NODE '{op[0]}' -> '{op[1]}' {label} -> {status}")
        elif op_type == "replace_edge" and isinstance(op, list) and isinstance(op[0], list):
            print(f"  [{i+1}] REPLACE EDGE {tuple(op[0])} -> {tuple(op[1])} {label} -> {status}")

        op_results.append({"op": op, "feasible": is_feasible, "reason": reason, "reduced": reduced})

        if not is_feasible:
            all_feasible = False

        # Apply op to get reduced graph for next iteration
        current_G = apply_op(current_G, op, op_type)

    print(f"  => All feasible: {all_feasible}")
    return {"all_feasible": all_feasible, "op_results": op_results, "question": record["question"]}

def compare_infeasible(results1: list[dict], results2: list[dict]):
    """Compare infeasible operations between two result sets by question."""

    def get_infeasible_map(results):
        mapping = {}
        for r in results:
            if r["all_feasible"] is False:
                question = r.get("question", "")
                infeasible_ops = [o for o in r["op_results"] if not o["feasible"]]
                mapping[question] = infeasible_ops
        return mapping

    infeasible1 = get_infeasible_map(results1)
    infeasible2 = get_infeasible_map(results2)

    questions1 = set(infeasible1.keys())
    questions2 = set(infeasible2.keys())

    common_questions = questions1 & questions2
    union_questions = questions1 | questions2

    jaccard = len(common_questions) / len(union_questions) if union_questions else 0.0

    print(f"\n{'=' * 50}")
    print(f"INFEASIBLE COMPARISON")
    print(f"{'=' * 50}")
    print(f"  Infeasible cases in dir1:   {len(questions1)}")
    print(f"  Infeasible cases in dir2:   {len(questions2)}")
    print(f"  Common infeasible cases:    {len(common_questions)}")
    print(f"  Jaccard similarity:         {jaccard:.4f}")

    if common_questions:
        print(f"\n  Common infeasible questions — op comparison:")
        for q in sorted(common_questions):
            ops1 = {str(o["op"]) for o in infeasible1[q]}
            ops2 = {str(o["op"]) for o in infeasible2[q]}
            common_ops = ops1 & ops2
            only1 = ops1 - ops2
            only2 = ops2 - ops1
            op_jaccard = len(common_ops) / len(ops1 | ops2) if (ops1 | ops2) else 0.0

            print(f"\n    Q: {q[:80]}")
            print(f"       Op Jaccard:  {op_jaccard:.4f}")
            if common_ops:
                print(f"       Common ops: {common_ops}")
            if only1:
                print(f"       Only dir1:  {only1}")
            if only2:
                print(f"       Only dir2:  {only2}")

    only_in_1 = questions1 - questions2
    only_in_2 = questions2 - questions1

    if only_in_1:
        print(f"\n  Infeasible only in dir1 ({len(only_in_1)}):")
        for q in sorted(only_in_1):
            print(f"    - {q[:80]}")

    if only_in_2:
        print(f"\n  Infeasible only in dir2 ({len(only_in_2)}):")
        for q in sorted(only_in_2):
            print(f"    - {q[:80]}")

def load_and_check_all(results_dir: str) -> list[dict]:
    paths = list(Path(results_dir).glob("*.json"))

    if not paths:
        print(f"No JSON files found in '{results_dir}'")
        return []

    print(f"Loaded {len(paths)} files from '{results_dir}'")

    all_results = []
    for path in sorted(paths):
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)
        result = check_operation_feasibility(record)
        result["_filename"] = path.name
        all_results.append(result)

    with_ops = [r for r in all_results if r["all_feasible"] is not None]
    fully_feasible = [r for r in with_ops if r["all_feasible"]]
    partially_infeasible = [r for r in with_ops if not r["all_feasible"]]

    print(f"\n{'=' * 50}")
    print(f"SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Files checked:         {len(paths)}")
    print(f"  With operations:       {len(with_ops)}")
    print(f"  All feasible:          {len(fully_feasible)}")
    print(f"  Contains infeasible:   {len(partially_infeasible)}")

    if fully_feasible:
        print(f"\n  Fully feasible cases:")
        for r in fully_feasible:
            print(f"    [{r['_filename']}] {r.get('question', '')[:80]}")

    if partially_infeasible:
        print(f"\n  Cases with infeasible ops:")
        for r in partially_infeasible:
            infeasible_ops = [o for o in r["op_results"] if not o["feasible"]]
            print(f"    [{r['_filename']}] {r.get('question', '')[:80]}")
            for o in infeasible_ops:
                reduced_label = " on reduced graph" if o["reduced"] else ""
                print(f"      -> {o['op']} ({o['reason']}{reduced_label})")

    return all_results


def load_and_check_all(results_dir: str) -> list[dict]:
    paths = list(Path(results_dir).glob("*.json"))

    if not paths:
        print(f"No JSON files found in '{results_dir}'")
        return []

    print(f"Loaded {len(paths)} files from '{results_dir}'")

    all_results = []
    for path in sorted(paths):
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)
        result = check_operation_feasibility(record)
        result["_filename"] = path.name
        all_results.append(result)

    with_ops = [r for r in all_results if r["all_feasible"] is not None]
    fully_feasible = [r for r in with_ops if r["all_feasible"]]
    partially_infeasible = [r for r in with_ops if not r["all_feasible"]]

    print(f"\n{'=' * 50}")
    print(f"SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Files checked:         {len(paths)}")
    print(f"  With operations:       {len(with_ops)}")
    print(f"  All feasible:          {len(fully_feasible)}")
    print(f"  Contains infeasible:   {len(partially_infeasible)}")

    if fully_feasible:
        print(f"\n  Fully feasible cases:")
        for r in fully_feasible:
            print(f"    [{r['_filename']}] {r.get('question', '')[:80]}")

    if partially_infeasible:
        print(f"\n  Cases with infeasible ops:")
        for r in partially_infeasible:
            infeasible_ops = [o for o in r["op_results"] if not o["feasible"]]
            print(f"    [{r['_filename']}] {r.get('question', '')[:80]}")
            for o in infeasible_ops:
                reduced_label = " on reduced graph" if o["reduced"] else ""
                print(f"      -> {o['op']} ({o['reason']}{reduced_label})")

    return all_results

def compare_cut_vertices(results_dir1: str, results_dir2: str):
    """Compare cut vertices of original graphs for same questions across two dirs."""

    def load_cut_vertices(results_dir):
        mapping = {}
        for path in sorted(Path(results_dir).glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
            subgraph = record.get("original_subgraph")
            if not subgraph:
                continue
            G = build_graph_from_subgraph(subgraph)
            U = G.to_undirected()
            cut_vertices = set(nx.articulation_points(U))
            bridges = set(nx.bridges(U))
            mapping[record["question"]] = {
                "cut_vertices": cut_vertices,
                "bridges": bridges,
                "filename": path.name
            }
        return mapping

    map1 = load_cut_vertices(results_dir1)
    map2 = load_cut_vertices(results_dir2)

    common_questions = set(map1.keys()) & set(map2.keys())

    print(f"\n{'=' * 50}")
    print(f"CUT VERTEX / BRIDGE COMPARISON")
    print(f"{'=' * 50}")
    print(f"  Questions in dir1:     {len(map1)}")
    print(f"  Questions in dir2:     {len(map2)}")
    print(f"  Common questions:      {len(common_questions)}")

    if not common_questions:
        print("  No common questions to compare.")
        return

    cv_jaccards = []
    bridge_jaccards = []

    for q in sorted(common_questions):
        cv1 = map1[q]["cut_vertices"]
        cv2 = map2[q]["cut_vertices"]
        b1 = map1[q]["bridges"]
        b2 = map2[q]["bridges"]

        cv_intersection = cv1 & cv2
        cv_union = cv1 | cv2
        cv_jaccard = len(cv_intersection) / len(cv_union) if cv_union else 1.0

        b_intersection = b1 & b2
        b_union = b1 | b2
        b_jaccard = len(b_intersection) / len(b_union) if b_union else 1.0

        cv_jaccards.append(cv_jaccard)
        bridge_jaccards.append(b_jaccard)

        print(f"\n  Q: {q[:80]}")
        print(f"     Cut vertices dir1:  {cv1 or 'none'}")
        print(f"     Cut vertices dir2:  {cv2 or 'none'}")
        print(f"     Common:             {cv_intersection or 'none'}")
        print(f"     CV Jaccard:         {cv_jaccard:.4f}")
        print(f"     Bridges dir1:       {b1 or 'none'}")
        print(f"     Bridges dir2:       {b2 or 'none'}")
        print(f"     Common bridges:     {b_intersection or 'none'}")
        print(f"     Bridge Jaccard:     {b_jaccard:.4f}")

    print(f"\n{'=' * 50}")
    print(f"AGGREGATE")
    print(f"{'=' * 50}")
    print(f"  Avg CV Jaccard:        {sum(cv_jaccards) / len(cv_jaccards):.4f}")
    print(f"  Avg Bridge Jaccard:    {sum(bridge_jaccards) / len(bridge_jaccards):.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="src/counterfactuals/counterfactual_results",
                        help="Directory containing counterfactual JSON files")
    parser.add_argument("--dir2", type=str, default=None,
                        help="Optional second directory for infeasible comparison")
    args = parser.parse_args()

    results1 = load_and_check_all(args.dir)

    if args.dir2:
        print(f"\n{'=' * 50}")
        print(f"DIR2: {args.dir2}")
        results2 = load_and_check_all(args.dir2)
        compare_infeasible(results1, results2)
        compare_cut_vertices(args.dir, args.dir2)

if __name__ == "__main__":
    main()