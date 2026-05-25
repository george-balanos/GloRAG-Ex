"""
explain_result.py  –  Renders a graph-perturbation result JSON as a human-readable report.

Usage:
    python explain_result.py [--source FILE] [--format text|md|html] [--out FILE]
    cat result.json | python explain_result.py --source -
"""

import argparse, base64, io, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


# ─── helpers ──────────────────────────────────────────────────────────────────

def load(source: str) -> dict:
    return json.load(sys.stdin) if source == "-" else json.loads(Path(source).read_text())

def esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def describe_op(op: list) -> str:
    """Return a human-readable one-liner for any operation type."""
    if not op:
        return "(none)"
    name, *args = op
    if name == "delete_node":
        node = args[0] if args else "?"
        return f'delete_node("{node}") — removed node and all attached edges'
    if name == "delete_edge":
        edge = args[0] if args else ["?", "?"]
        src, tgt = (edge[0], edge[1]) if isinstance(edge, (list, tuple)) and len(edge) >= 2 else ("?", "?")
        return f'delete_edge("{src}" → "{tgt}") — removed this relation'
    if name == "add_node":
        node = args[0] if args else "?"
        return f'add_node("{node}") — inserted new node into the graph'
    if name == "add_edge":
        edge = args[0] if args else ["?", "?"]
        src, tgt = (edge[0], edge[1]) if isinstance(edge, (list, tuple)) and len(edge) >= 2 else ("?", "?")
        return f'add_edge("{src}" → "{tgt}") — added new relation'
    # fallback for unknown op types
    return f"{name}({', '.join(str(a) for a in args)})"


def _op_sets(ops: list) -> tuple[set, set, set, set]:
    """
    Returns (deleted_nodes, deleted_edges, added_nodes, added_edges).
    deleted_edges / added_edges are frozensets of (src, tgt) tuples.
    """
    deleted_nodes, added_nodes = set(), set()
    deleted_edges, added_edges = set(), set()
    for op in ops:
        if not op:
            continue
        name, *args = op
        if name == "delete_node" and args:
            deleted_nodes.add(args[0])
        elif name == "add_node" and args:
            added_nodes.add(args[0])
        elif name == "delete_edge" and args:
            edge = args[0]
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                deleted_edges.add((edge[0], edge[1]))
        elif name == "add_edge" and args:
            edge = args[0]
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                added_edges.add((edge[0], edge[1]))
    return deleted_nodes, deleted_edges, added_nodes, added_edges


# ─── graph drawing ────────────────────────────────────────────────────────────

def _build_graph(entities, relations) -> nx.DiGraph:
    G = nx.DiGraph()
    for e in entities:
        G.add_node(e["name"], etype=e.get("type", ""), desc=e.get("description", ""))
    for r in relations:
        G.add_edge(r["src"], r["tgt"], desc=r.get("description", ""))
    return G


def _spread_positions(pos: dict, min_dist: float = 0.35) -> dict:
    """Iteratively push nodes apart until no two are closer than min_dist."""
    nodes = list(pos)
    coords = {n: list(pos[n]) for n in nodes}
    for _ in range(200):
        moved = False
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                dx = coords[a][0] - coords[b][0]
                dy = coords[a][1] - coords[b][1]
                dist = (dx**2 + dy**2) ** 0.5 or 1e-6
                if dist < min_dist:
                    push = (min_dist - dist) / 2 + 1e-6
                    nx_ = dx / dist * push
                    ny_ = dy / dist * push
                    coords[a][0] += nx_; coords[a][1] += ny_
                    coords[b][0] -= nx_; coords[b][1] -= ny_
                    moved = True
        if not moved:
            break
    return {n: tuple(coords[n]) for n in nodes}


def _draw_graph(
    G,
    deleted_nodes: set = None,
    added_nodes: set = None,
    deleted_edges: set = None,
    added_edges: set = None,
    title: str = "",
    fig_bg: str = "#0f1117",
) -> str:
    """Return a base64 PNG of the graph.

    Colour legend
    ─────────────
    Nodes  purple  (#6366f1) normal
           red     (#ef4444) deleted
           green   (#10b981) added
    Edges  grey    (#94a3b8) normal
           red     (#ef4444) deleted / touching a deleted node
           green   (#10b981) added
    """
    deleted_nodes = deleted_nodes or set()
    added_nodes   = added_nodes   or set()
    deleted_edges = deleted_edges or set()
    added_edges   = added_edges   or set()

    n = G.number_of_nodes()
    side = max(6.0, n * 1.1)
    fig, ax = plt.subplots(figsize=(side, side * 0.65), facecolor=fig_bg)
    ax.set_facecolor(fig_bg)
    ax.set_title(title, color="#e2e8f0", fontsize=11, fontweight="bold", pad=10)

    if n == 0:
        ax.text(0.5, 0.5, "— empty —", ha="center", va="center",
                color="#64748b", fontsize=13, transform=ax.transAxes)
        ax.axis("off")
    else:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=2.5 / max(n**0.5, 1))

        pos = _spread_positions(pos, min_dist=0.45)

        def _node_color(nd):
            if nd in deleted_nodes: return "#ef4444"
            if nd in added_nodes:   return "#10b981"
            return "#6366f1"

        def _edge_color(u, v):
            if (u, v) in deleted_edges or u in deleted_nodes or v in deleted_nodes:
                return "#ef4444"
            if (u, v) in added_edges:
                return "#10b981"
            return "#94a3b8"

        node_colors = [_node_color(nd) for nd in G.nodes()]
        edge_colors = [_edge_color(u, v) for u, v in G.edges()]

        max_label_len = max((len(str(nd)) for nd in G.nodes()), default=1)
        node_size = max(1400, max_label_len * 100)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=node_size, alpha=0.92)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=15,
                                font_color="#f1f5f9", font_weight="bold")
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                               arrows=True, arrowsize=16, width=1.4,
                               node_size=node_size,
                               connectionstyle="arc3,rad=0.08")
        ax.axis("off")

    plt.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def make_graph_images(data: dict) -> tuple[str, str]:
    """Return (before_b64, after_b64) PNG images."""
    ops = data.get("operations", [])
    deleted_nodes, deleted_edges, added_nodes, added_edges = _op_sets(ops)

    orig_sg = data.get("original_subgraph") or {}
    pert_sg = data.get("perturbed_subgraph") or {}

    G_before = _build_graph(orig_sg.get("entities", []), orig_sg.get("relations", []))
    G_after  = _build_graph(pert_sg.get("entities", []), pert_sg.get("relations", []))

    # Before graph: mark nodes/edges that are about to be removed or were not yet present
    b64_before = _draw_graph(
        G_before,
        deleted_nodes=deleted_nodes,
        deleted_edges=deleted_edges,
        title="Before perturbation",
    )
    # After graph: highlight what was freshly added
    b64_after = _draw_graph(
        G_after,
        added_nodes=added_nodes,
        added_edges=added_edges,
        title="After perturbation",
    )
    return b64_before, b64_after


# ─── plain-text renderer ──────────────────────────────────────────────────────

def render_text(data: dict) -> str:
    answers  = data.get("answers") or {}
    ops      = data.get("operations") or []
    orig_sg  = data.get("original_subgraph") or {}
    ents     = orig_sg.get("entities",  [])
    rels     = orig_sg.get("relations", [])
    found    = data.get("found", False)

    deleted_nodes, deleted_edges, added_nodes, added_edges = _op_sets(ops)

    W = 60
    def h1(t): return f"\n{'═'*W}\n  {t}\n{'═'*W}"
    def h2(t): return f"\n{'─'*W}\n  {t}\n{'─'*W}"
    def kv(k, v): return f"  {k:<22} {v}"

    out = [h1("PERTURBATION RESULT REPORT"), f"\n  Question: {data.get('question','N/A')}\n"]

    out.append(h2("SUMMARY"))
    for k, v in [("Answer found:",        "YES ✓" if found else "NO ✗"),
                 ("Operations applied:",   data.get("num_operations", len(ops))),
                 ("Perturbation cost:",    data.get("cost", "N/A")),
                 ("LLM calls:",            data.get("llm_calls", "N/A")),
                 ("Mode:",                 data.get("mode", "N/A")),
                 ("Timestamp:",            data.get("timestamp", "N/A"))]:
        out.append(kv(k, v))

    if not found:
        out.append(f"\n  ⚠  No valid perturbation was found after {data.get('llm_calls','?')} LLM calls.")

    out.append(h2("WHAT WAS CHANGED"))
    if not ops:
        out.append("  (no operations — perturbation search exhausted without a result)")
    else:
        for i, op in enumerate(ops, 1):
            out.append(f"  Operation {i}: {describe_op(op)}")

    out.append(h2("ANSWERS COMPARISON"))
    perturbed_ans = answers.get("perturbed")
    for label, key in [("Ground truth", "ground_truth"),
                       ("Original graph", "original"),
                       ("After perturbation", "perturbed")]:
        val = answers.get(key) if key != "perturbed" else perturbed_ans
        out += [f"  [{label}]",
                f"    {val if val is not None else '— (no answer generated)'}",
                ""]

    out.append(h2("ORIGINAL SUBGRAPH"))
    out.append(f"  Entities ({len(ents)}):")
    for e in ents:
        name = e["name"]
        mark = (" ← DELETED" if name in deleted_nodes
                else " ← ADDED"   if name in added_nodes else "")
        out.append(f"    • {name}" + (f" [{e.get('type','')}]" if e.get("type") else "") + mark)
        if e.get("description"): out.append(f"      {e['description']}")

    out.append(f"\n  Relations ({len(rels)}):")
    for r in rels:
        edge_key = (r["src"], r["tgt"])
        sev_mark = ""
        if r["src"] in deleted_nodes or r["tgt"] in deleted_nodes or edge_key in deleted_edges:
            sev_mark = " [SEVERED]"
        elif edge_key in added_edges:
            sev_mark = " [ADDED]"
        out.append(f"    • {r['src']}  →  {r['tgt']}{sev_mark}")
        if r.get("description"): out.append(f"      {r['description']}")

    pert_sg = data.get("perturbed_subgraph")
    out.append(h2("PERTURBED SUBGRAPH"))
    if pert_sg is None:
        out.append("  (null — no valid perturbed subgraph was produced)")
    else:
        pe, pr = pert_sg.get("entities", []), pert_sg.get("relations", [])
        if pe or pr:
            out.append(f"  Entities: {len(pe)}, Relations: {len(pr)}")
        else:
            out.append("  (empty — all information was removed)")

    return "\n".join(out)


# ─── markdown renderer ────────────────────────────────────────────────────────

def render_md(data: dict, img_before="", img_after="") -> str:
    answers  = data.get("answers") or {}
    ops      = data.get("operations") or []
    orig_sg  = data.get("original_subgraph") or {}
    ents     = orig_sg.get("entities",  [])
    rels     = orig_sg.get("relations", [])
    found    = data.get("found", False)

    deleted_nodes, deleted_edges, added_nodes, added_edges = _op_sets(ops)

    lines = [f"# Perturbation Result Report\n",
             f"**Question:** {data.get('question','N/A')}\n",
             "## Summary\n",
             "| Field | Value |", "|---|---|",
             f"| Answer found | {'✅ Yes' if found else '❌ No'} |",
             f"| Operations | {data.get('num_operations', len(ops))} |",
             f"| Cost | {data.get('cost','N/A')} |",
             f"| LLM calls | {data.get('llm_calls','N/A')} |",
             f"| Mode | {data.get('mode','N/A')} |",
             f"| Timestamp | {data.get('timestamp','N/A')} |", ""]

    if not found:
        lines.append(f"> ⚠️ No valid perturbation was found after {data.get('llm_calls','?')} LLM calls.\n")

    lines.append("## What Was Changed\n")
    if not ops:
        lines.append("_No operations were applied — the perturbation search was exhausted._\n")
    else:
        for i, op in enumerate(ops, 1):
            lines.append(f"**Operation {i}:** `{describe_op(op)}`\n")

    lines += ["## Graph Visualisation\n",
              "| Before | After |", "|---|---|",
              f'| ![before](data:image/png;base64,{img_before}) | ![after](data:image/png;base64,{img_after}) |',
              "",
              "_🟣 Normal &nbsp; 🔴 Deleted &nbsp; 🟢 Added_\n"]

    perturbed_ans = answers.get("perturbed")
    lines += ["## Answers\n",
              f"**Ground truth**\n> {answers.get('ground_truth','—')}\n",
              f"**Original**\n> {answers.get('original','—')}\n",
              f"**After perturbation**\n> {perturbed_ans if perturbed_ans is not None else '_No answer generated_'}\n"]

    lines.append("## Original Subgraph\n")
    for e in ents:
        name = e["name"]
        mark = (" ⚠ *deleted*" if name in deleted_nodes
                else " ✚ *added*" if name in added_nodes else "")
        tag  = f" `{e['type']}`" if e.get("type") else ""
        lines.append(f"- **{name}**{tag}{mark}")
        if e.get("description"): lines.append(f"  {e['description']}")
    lines.append("")

    for r in rels:
        edge_key = (r["src"], r["tgt"])
        sev = ""
        if r["src"] in deleted_nodes or r["tgt"] in deleted_nodes or edge_key in deleted_edges:
            sev = " *(severed)*"
        elif edge_key in added_edges:
            sev = " *(added)*"
        lines.append(f"- `{r['src']}` → `{r['tgt']}`{sev}")
        if r.get("description"): lines.append(f"  {r['description']}")

    pert_sg = data.get("perturbed_subgraph")
    lines.append("\n## Perturbed Subgraph\n")
    if pert_sg is None:
        lines.append("_null — no valid perturbed subgraph was produced._")
    else:
        pe, pr = pert_sg.get("entities", []), pert_sg.get("relations", [])
        lines.append(f"- Entities: {len(pe)}\n- Relations: {len(pr)}" if pe or pr
                     else "*(empty — all information was removed)*")

    return "\n".join(lines)


# ─── html renderer ────────────────────────────────────────────────────────────

def render_html(data: dict, img_before: str, img_after: str) -> str:
    answers  = data.get("answers") or {}
    ops      = data.get("operations") or []
    orig_sg  = data.get("original_subgraph") or {}
    ents     = orig_sg.get("entities",  [])
    rels     = orig_sg.get("relations", [])
    found    = data.get("found", False)

    deleted_nodes, deleted_edges, added_nodes, added_edges = _op_sets(ops)

    badge = lambda t, c: f'<span class="badge badge-{c}">{esc(t)}</span>'
    card  = lambda h: f'<div class="card">{h}</div>'
    sect  = lambda t: f'<p class="section-label">{esc(t)}</p>'

    # ── not-found banner ──
    not_found_banner = ""
    if not found:
        not_found_banner = (
            f'<div class="banner-warn">'
            f'⚠ No valid perturbation was found after {data.get("llm_calls","?")} LLM calls. '
            f'The perturbed answer and subgraph may be absent.'
            f'</div>'
        )

    # ── answers ──
    sim       = answers.get("similarity", 0.0)
    sim_pct   = max(round(sim * 100), 1) if sim else 0
    bar_color = "#ef4444" if sim < 0.1 else "#f59e0b" if sim < 0.4 else "#10b981"
    perturbed_ans = answers.get("perturbed")

    answers_html = "".join([
        badge("ground truth", "green"),
        f'<div class="answer-block border-green">{esc(answers.get("ground_truth","—"))}</div>',
        badge("original graph", "blue"),
        f'<div class="answer-block border-blue">{esc(answers.get("original","—"))}</div>',
        badge("after perturbation", "red"),
        (f'<div class="answer-block border-red">{esc(perturbed_ans)}</div>'
         if perturbed_ans is not None
         else '<div class="answer-block border-red muted"><em>No answer generated</em></div>'),
    ])

    # ── entities ──
    def _entity_class(name):
        if name in deleted_nodes: return "deleted"
        if name in added_nodes:   return "added"
        return "bold"

    def _entity_tag(name):
        if name in deleted_nodes: return ' <span class="deleted-tag">(deleted)</span>'
        if name in added_nodes:   return ' <span class="added-tag">(added)</span>'
        return ""

    ent_rows = "".join(
        f'<div class="row">'
        f'<span class="{_entity_class(e["name"])}">{esc(e["name"])}</span>'
        + (f'<span class="tag">{esc(e["type"])}</span>' if e.get("type") else "")
        + _entity_tag(e["name"])
        + (f'<div class="desc">{esc(e["description"])}</div>' if e.get("description") else "")
        + '</div>'
        for e in ents
    )

    def _edge_tag(src, tgt):
        edge_key = (src, tgt)
        if src in deleted_nodes or tgt in deleted_nodes or edge_key in deleted_edges:
            return ' <span class="deleted-tag">[severed]</span>'
        if edge_key in added_edges:
            return ' <span class="added-tag">[added]</span>'
        return ""

    rel_rows = "".join(
        f'<div class="row"><code>{esc(r["src"])}</code> <span class="muted">→</span> <code>{esc(r["tgt"])}</code>'
        + _edge_tag(r["src"], r["tgt"])
        + (f'<div class="desc">{esc(r["description"])}</div>' if r.get("description") else "")
        + '</div>'
        for r in rels
    )

    # ── op pills ──
    def _op_pill_html(op):
        if not op:
            return ""
        name, *args = op
        label_map = {
            "delete_node": ("delete_node", "pill-red",   lambda a: f'<code class="pill pill-red">{esc(a[0])}</code>'   if a else ""),
            "add_node":    ("add_node",    "pill-green", lambda a: f'<code class="pill pill-green">{esc(a[0])}</code>' if a else ""),
            "delete_edge": ("delete_edge", "pill-red",   lambda a: (
                f'<code class="pill pill-red">{esc(a[0][0])}</code>'
                f'<span class="muted op-arrow">→</span>'
                f'<code class="pill pill-red">{esc(a[0][1])}</code>'
            ) if a and isinstance(a[0], (list, tuple)) and len(a[0]) >= 2 else ""),
            "add_edge":    ("add_edge",    "pill-green", lambda a: (
                f'<code class="pill pill-green">{esc(a[0][0])}</code>'
                f'<span class="muted op-arrow">→</span>'
                f'<code class="pill pill-green">{esc(a[0][1])}</code>'
            ) if a and isinstance(a[0], (list, tuple)) and len(a[0]) >= 2 else ""),
        }
        op_name, pill_cls, arg_fn = label_map.get(name, (name, "", lambda a: ""))
        return (
            f'<code class="pill {pill_cls}">{esc(op_name)}</code>'
            f'<span class="muted op-arrow">→</span>'
            + arg_fn(args)
        )

    if ops:
        op_pills = "".join(
            f'<div class="op-row"><span class="op-index">#{i}</span>{_op_pill_html(op)}</div>'
            for i, op in enumerate(ops, 1)
        )
        op_note = "Nodes/edges highlighted in the graph according to their operation type."
    else:
        op_pills = "<em>No operations — perturbation search was exhausted without finding a result.</em>"
        op_note  = ""

    pert_sg = data.get("perturbed_subgraph")
    if pert_sg is None:
        pert_info_html = '<p class="muted sm">null — no valid perturbed subgraph was produced.</p>'
    else:
        pe = pert_sg.get("entities", [])
        pr = pert_sg.get("relations", [])
        
        pert_info_text = (f"Entities: {len(pe)} &nbsp;·&nbsp; Relations: {len(pr)}"
                          if pe or pr else "Empty — all information removed.")
        
        pert_ent_rows = "".join(
            f'<div class="row">'
            f'<span class="{_entity_class(e["name"])}">{esc(e["name"])}</span>'
            + (f'<span class="tag">{esc(e["type"])}</span>' if e.get("type") else "")
            + _entity_tag(e["name"])
            + (f'<div class="desc">{esc(e["description"])}</div>' if e.get("description") else "")
            + '</div>'
            for e in pe
        )

        pert_rel_rows = "".join(
            f'<div class="row"><code>{esc(r["src"])}</code> <span class="muted">→</span> <code>{esc(r["tgt"])}</code>'
            + _edge_tag(r["src"], r["tgt"])
            + (f'<div class="desc">{esc(r["description"])}</div>' if r.get("description") else "")
            + '</div>'
            for r in pr
        )

        pert_info_html = (
            f'<p class="muted sm" style="margin-bottom:12px">{pert_info_text}</p>'
            f'<h3>Entities</h3>{pert_ent_rows}'
            f'<h3 style="margin-top:14px">Relations</h3>{pert_rel_rows}'
        )

    body = f"""
{not_found_banner}

{sect("Question")}
{card(f'<p class="question">{esc(data.get("question",""))}</p>')}

{sect("Summary")}
<div class="grid-3">
  {card(f'<p class="muted sm">Result</p>{ badge("✓ Found","green") if found else badge("✗ Not found","red")}')}
  {card(f'<p class="muted sm">Cost</p><span class="big-num">{round(data.get("cost","?"), 3)}</span>')}
  {card(f'<p class="muted sm">LLM calls</p><span class="big-num">{data.get("llm_calls","?")}</span>')}
</div>

{sect("What Was Changed")}
{card(f'<h3>Graph perturbation</h3><div class="pills">{op_pills}</div><p class="muted sm">{esc(op_note)}</p>')}

{sect("Graph — Before & After")}
{card(f'''<div class="grid-2">
  <div><p class="muted sm centre">Before</p><img src="data:image/png;base64,{img_before}" alt="before" class="graph-img"></div>
  <div><p class="muted sm centre">After</p><img src="data:image/png;base64,{img_after}"  alt="after"  class="graph-img"></div>
</div>
<p class="muted sm" style="margin-top:8px">
  <span style="color:#6366f1">&#9632;</span> Normal &nbsp;
  <span style="color:#ef4444">&#9632;</span> Deleted/severed &nbsp;
  <span style="color:#10b981">&#9632;</span> Added
</p>''')}

{sect("Answer Comparison")}
{card(f'<h3>How answers changed</h3>{answers_html}')}

{sect(f"Original Subgraph · {len(ents)} entities · {len(rels)} relations")}
{card(f'<h3>Entities</h3>{ent_rows}<h3 style="margin-top:14px">Relations</h3>{rel_rows}')}

{sect(f"Perturbed Subgraph" + (f" · {len(pe)} entities · {len(pr)} relations" if pert_sg else ""))}
{card(pert_info_html)}
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Perturbation Result</title>
<style>
  :root {{
    --bg:#0f1117; --surface:#1a1d27; --border:#2a2d3a;
    --text:#e2e8f0; --muted:#64748b; --accent:#6366f1;
    --green:#10b981; --red:#ef4444; --blue:#3b82f6; --amber:#f59e0b;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        background:var(--bg);color:var(--text);max-width:800px;margin:2rem auto;padding:0 1rem;font-size:14px}}
  .card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;
         padding:1rem 1.25rem;margin-bottom:10px}}
  .grid-3{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:10px}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  .section-label{{font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
                  color:var(--muted);margin:18px 0 6px}}
  h3{{font-size:14px;font-weight:600;margin-bottom:10px;color:var(--text)}}
  .question{{font-size:14px;line-height:1.65}}
  .big-num{{font-size:24px;font-weight:600}}
  .muted{{color:var(--muted)}} .sm{{font-size:12px;margin:4px 0}} .bold{{font-weight:600}}
  .centre{{text-align:center}}
  .badge{{padding:2px 9px;border-radius:5px;font-size:11px;font-weight:700;display:inline-block;margin:4px 0}}
  .badge-green{{background:#064e3b;color:#34d399}} .badge-red{{background:#450a0a;color:#f87171}}
  .badge-blue{{background:#1e3a5f;color:#93c5fd}} .badge-amber{{background:#451a03;color:#fbbf24}}
  .answer-block{{border-left:2.5px solid;padding:6px 10px;margin:6px 0;font-size:13px;line-height:1.6}}
  .border-green{{border-color:var(--green)}} .border-blue{{border-color:var(--blue)}} .border-red{{border-color:var(--red)}}
  .bar-track{{height:6px;background:#2a2d3a;border-radius:4px;margin:6px 0}}
  .bar-fill{{height:6px;border-radius:4px;transition:width .4s}}
  .row{{padding:5px 0;border-bottom:1px solid var(--border);font-size:13px}}
  .desc{{font-size:11px;color:var(--muted);margin-top:2px}}
  .tag{{background:#2a2d3a;border-radius:4px;padding:1px 6px;font-size:11px;margin-left:6px}}
  .deleted{{color:var(--red);font-weight:600}}
  .deleted-tag{{color:var(--red);font-size:11px;margin-left:6px}}
  .added{{color:#10b981;font-weight:600}}
  .added-tag{{color:#10b981;font-size:11px;margin-left:6px}}
  .pills{{display:flex;flex-direction:column;gap:8px;margin:8px 0}}
  .op-row{{display:flex;align-items:center;gap:8px}}
  .op-index{{font-size:11px;color:var(--muted);font-weight:600;min-width:20px}}
  .op-arrow{{padding:0 2px}}
  .pill{{background:#2a2d3a;border:1px solid var(--border);border-radius:6px;
         padding:3px 10px;font-family:monospace;font-size:12px}}
  .pill-red{{border-color:#7f1d1d;color:var(--red)}}
  .pill-green{{border-color:#064e3b;color:#10b981}}
  .graph-img{{width:100%;border-radius:8px;border:1px solid var(--border);display:block}}
  code{{font-family:monospace;font-size:12px;background:#2a2d3a;padding:1px 5px;border-radius:4px}}
  .banner-warn{{background:#451a03;border:1px solid #92400e;color:#fbbf24;border-radius:10px;
                padding:10px 16px;margin-bottom:14px;font-size:13px;line-height:1.5}}
</style>
</head>
<body>{body}</body>
</html>"""


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser  = argparse.ArgumentParser(description="Explain a graph-perturbation result JSON.")
    parser.add_argument("--source",  default="/home/gbalanos/GloRAG-Ex/code/src/counterfactuals/results/synthetic/without_f3_all/ff-case-1/all_ops_ff/counterfactual_20260525_082157.json")
    parser.add_argument("--format",  choices=["text","md","html"], default="text")
    parser.add_argument("--out",     default="src/html_results/result.html")
    args = parser.parse_args()

    data = load(args.source)
    img_before, img_after = make_graph_images(data)

    if args.format == "html":
        output = render_html(data, img_before, img_after)
    elif args.format == "md":
        output = render_md(data, img_before, img_after)
    else:
        output = render_text(data)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Saved to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()