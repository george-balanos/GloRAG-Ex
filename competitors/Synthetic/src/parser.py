from src.base import *

import re
import networkx as nx
import json

def _split_sections(context: str) -> dict[str, str]:
    """
    Split the context string into named sections.
    Handles two known header styles:
 
    Style A (older):   -----Entities-----
    Style B (newer):   Knowledge Graph Data (Entity):
                       Knowledge Graph Data (Relationship):
                       Document Chunks (...):
                       Reference Document List (...):
    """
    sections: dict[str, str] = {}
    current_key = "header"
    current_lines: list[str] = []
 
    for line in context.splitlines():
        stripped = line.strip()
 
        # Style A: ---SectionName---
        m = re.match(r"^-{3,}\s*(\w[\w\s]*\w|\w)\s*-{3,}$", stripped)
        if m:
            sections[current_key] = "\n".join(current_lines)
            current_key = m.group(1).strip().lower()
            current_lines = []
            continue
 
        # Style B: "Knowledge Graph Data (Entity):" etc.
        m = re.match(r"^Knowledge Graph Data\s*\((\w+)\)\s*:", stripped, re.IGNORECASE)
        if m:
            sections[current_key] = "\n".join(current_lines)
            current_key = m.group(1).strip().lower()   # "entity" | "relationship"
            current_lines = []
            continue
 
        m = re.match(r"^Document Chunks", stripped, re.IGNORECASE)
        if m:
            sections[current_key] = "\n".join(current_lines)
            current_key = "chunks"
            current_lines = []
            continue
 
        m = re.match(r"^Reference Document", stripped, re.IGNORECASE)
        if m:
            sections[current_key] = "\n".join(current_lines)
            current_key = "references"
            current_lines = []
            continue
 
        current_lines.append(line)
 
    sections[current_key] = "\n".join(current_lines)
    return sections
 
 
def _parse_jsonlines(text: str) -> list[dict]:
    """Parse a block of JSON-lines, skipping blank lines and fenced code blocks."""
    import json
    results = []
    for line in text.splitlines():
        line = line.strip().lstrip("`")
        if not line or line.startswith("```"):
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return results
 
 
def _parse_pipe_delimited(text: str) -> list[dict]:
    """
    Parse the older <|>-delimited format:
      ("entity"<|>NAME<|>TYPE<|>DESCRIPTION<|>RANK)
      ("relationship"<|>SRC<|>TGT<|>KEYWORDS<|>DESCRIPTION<|>WEIGHT)
    """
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("("):
            continue
        inner = line[1:-1] if line.endswith(")") else line[1:]
        parts = [p.strip().strip('"') for p in inner.split("<|>")]
        if parts:
            results.append({"_parts": parts})
    return results
 
 
def parse_context(context: str) -> Subgraph:
    """
    Parse the context string LightRAG returns for only_need_context=True.
 
    Supports two formats emitted by different LightRAG versions:
      - JSON-lines under headers like "Knowledge Graph Data (Entity):"
      - <|>-pipe-delimited lines under "-----Entities-----" headers
    """
    subgraph = Subgraph(raw_context=context)
    sections = _split_sections(context)
 
    # ── Entities ──────────────────────────────────────────────────────────────
    # Try JSON-lines format first (newer), fall back to pipe-delimited (older)
    entity_text = sections.get("entity", sections.get("entities", ""))
    raw_entities = _parse_jsonlines(entity_text)
 
    if raw_entities and "entity" in raw_entities[0]:
        # JSON-lines: {"entity": "Name", "type": "...", "description": "..."}
        for r in raw_entities:
            subgraph.entities.append(Entity(
                name=r.get("entity", ""),
                type=r.get("type", ""),
                description=r.get("description", ""),
                rank=float(r.get("rank", 0.0)),
                raw=r,
            ))
    else:
        # Pipe-delimited: ("entity"<|>NAME<|>TYPE<|>DESCRIPTION<|>RANK)
        for r in _parse_pipe_delimited(entity_text):
            parts = r["_parts"]
            offset = 1 if parts[0].lower() == "entity" else 0
            subgraph.entities.append(Entity(
                name=parts[offset]     if len(parts) > offset     else "",
                type=parts[offset + 1] if len(parts) > offset + 1 else "",
                description=parts[offset + 2] if len(parts) > offset + 2 else "",
                rank=float(parts[offset + 3]) if len(parts) > offset + 3 else 0.0,
                raw={"parts": parts},
            ))
 
    # ── Relations ─────────────────────────────────────────────────────────────
    rel_text = sections.get("relationship", sections.get("relationships", ""))
    raw_rels = _parse_jsonlines(rel_text)
 
    if raw_rels and ("entity1" in raw_rels[0] or "src_id" in raw_rels[0]):
        # JSON-lines: {"entity1": "A", "entity2": "B", "description": "..."}
        for r in raw_rels:
            src = r.get("entity1", r.get("src_id", ""))
            tgt = r.get("entity2", r.get("tgt_id", ""))
            subgraph.relations.append(Relation(
                src=src,
                tgt=tgt,
                keywords=r.get("keywords", ""),
                description=r.get("description", ""),
                weight=float(r.get("weight", 0.0)),
                raw=r,
            ))
    else:
        # Pipe-delimited: ("relationship"<|>SRC<|>TGT<|>KEYWORDS<|>DESCRIPTION<|>WEIGHT)
        for r in _parse_pipe_delimited(rel_text):
            parts = r["_parts"]
            offset = 1 if parts[0].lower() == "relationship" else 0
            subgraph.relations.append(Relation(
                src=parts[offset]     if len(parts) > offset     else "",
                tgt=parts[offset + 1] if len(parts) > offset + 1 else "",
                keywords=parts[offset + 2] if len(parts) > offset + 2 else "",
                description=parts[offset + 3] if len(parts) > offset + 3 else "",
                weight=float(parts[offset + 4]) if len(parts) > offset + 4 else 0.0,
                raw={"parts": parts},
            ))
 
    # ── Source chunks ─────────────────────────────────────────────────────────
    chunk_text = sections.get("chunks", sections.get("sources", ""))
    raw_chunks = _parse_jsonlines(chunk_text)
    if raw_chunks and "content" in raw_chunks[0]:
        # JSON-lines: {"reference_id": "...", "content": "..."}
        subgraph.chunks = [r["content"] for r in raw_chunks if r.get("content")]
    else:
        # Plain text blocks separated by blank lines or [N] markers
        blocks = re.split(r"\n{2,}|\[\d+\]", chunk_text)
        subgraph.chunks = [b.strip() for b in blocks if b.strip()]
 
    return subgraph

def parse_graph(subgraph: Subgraph) -> nx.Graph:
    G = nx.DiGraph()

    for entity in subgraph.entities:
        G.add_node(
            entity.name,
            type=entity.type,
            description=entity.description,
        )

    for relation in subgraph.relations:
        for name in (relation.src, relation.tgt):
            if name and name not in G:
                G.add_node(name)

        if relation.src and relation.tgt:
            G.add_edge(
                relation.src,
                relation.tgt,
                keywords=relation.keywords,
                description=relation.description,
                weight=relation.weight
            )

    return G


###### Networkx -> Context (String)

def graph_to_subgraph(G: nx.DiGraph, original: Subgraph | None = None) -> Subgraph:
    entities = []
    for name, attrs in G.nodes(data=True):
        entities.append(Entity(
            name=name,
            type=attrs.get("type", ""),
            description=attrs.get("description", ""),
            rank=attrs.get("rank", 0.0),
            raw=dict(attrs),
        ))

    relations = []
    for src, tgt, attrs in G.edges(data=True):
        relations.append(Relation(
            src=src,
            tgt=tgt,
            keywords=attrs.get("keywords", ""),
            description=attrs.get("description", ""),
            weight=attrs.get("weight", 0.0),
            raw=dict(attrs),
        ))

    return Subgraph(
        entities=entities,
        relations=relations,
        chunks=original.chunks if original else [],
        raw_context=original.raw_context if original else "",
    )

def graph_to_context(G: nx.DiGraph, original: Subgraph | None = None) -> str:
    subgraph = graph_to_subgraph(G, original=original)

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