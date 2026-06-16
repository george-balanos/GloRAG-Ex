from src.base import *

import re
import networkx as nx
import json

###### Networkx -> Context (String)

def graph_to_subgraph(G: nx.DiGraph) -> Subgraph:
    entities = []
    for name, attrs in G.nodes(data=True):
        # Parse name|Category from node ID as fallback
        parts = name.split("|")
        node_name = parts[0] if parts else name
        node_category = parts[1] if len(parts) > 1 else ""

        entities.append(Entity(
            name=name,
            type=attrs.get("category", node_category),
            description=attrs.get("label", attrs.get("name", node_name)),
            rank=0.0,
            raw=dict(attrs),
        ))

    relations = []
    for src, tgt, attrs in G.edges(data=True):
        relations.append(Relation(
            src=src,
            tgt=tgt,
            keywords=attrs.get("relation", ""),
            description=attrs.get("relation", ""),
            weight=0.0,
            raw=dict(attrs),
        ))

    return Subgraph(
        entities=entities,
        relations=relations,
    )

def graph_to_context(G: nx.DiGraph) -> str:
    if len(G.nodes) == 0:
        return ""

    subgraph = graph_to_subgraph(G)

    lines = []

    # ── Entities ──────────────────────────────────────────────────────────────
    lines.append("Knowledge Graph Data (Entity):")
    lines.append("```json")
    for e in subgraph.entities:
        lines.append(json.dumps({
            "entity": e.name,
            "type": e.type,
            "description": e.description,
        }))
    lines.append("```")
    lines.append("")

    # ── Relations ─────────────────────────────────────────────────────────────
    lines.append("Knowledge Graph Data (Relationship):")
    lines.append("```json")
    for r in subgraph.relations:
        lines.append(json.dumps({
            "entity1": r.src,
            "entity2": r.tgt,
            "description": r.description,
        }))
    lines.append("```")
    lines.append("")

    # # ── Chunks ────────────────────────────────────────────────────────────────
    # lines.append("Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):")
    # lines.append("```json")
    # for chunk in subgraph.chunks:
    #     lines.append(json.dumps({
    #         "reference_id": "",
    #         "content": chunk,
    #     }))
    # lines.append("```")
    # lines.append("")

    # # ── References ────────────────────────────────────────────────────────────
    # lines.append("Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):")
    # lines.append("```")
    # lines.append("```")

    return "\n".join(lines)