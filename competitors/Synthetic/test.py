"""
gather_subgraph.py
==================
Retrieve and inspect the raw graph elements (entities + relations)
that LightRAG would send to the LLM for a given query — without
actually calling the LLM.

Works with the current LightRAG (HKUDS/LightRAG, 2025).

The key insight: QueryParam(only_need_context=True) makes LightRAG
return the assembled context string instead of an LLM answer.
We parse that string back into structured entities and relations.

Requirements:
    pip install lightrag-hku
    ollama pull <LLM_MODEL>
    ollama pull <EMBED_MODEL>

Usage:
    python gather_subgraph.py
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import setup_logger
from lightrag.kg.shared_storage import initialize_pipeline_status

setup_logger("lightrag", level="WARNING")  # suppress noise

# ── Config ────────────────────────────────────────────────────────────────────

WORKING_DIR  = "./synthetic"   # path to your existing LightRAG index
QUERY        = "What are the two primary materials used to construct a Xylotian 'Sky-Skiff' hull?"
MODE         = "hybrid"          # local | global | hybrid | naive

OLLAMA_HOST  = "http://localhost:11434"
LLM_MODEL    = "qwen3:latest"         # any model you have pulled in ollama
EMBED_MODEL  = "all-minilm:latest"
EMBED_DIM    = 384               # nomic-embed-text output dimension

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Entity:
    name: str
    type: str = ""
    description: str = ""
    rank: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

@dataclass
class Relation:
    src: str
    tgt: str
    keywords: str = ""
    description: str = ""
    weight: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

@dataclass
class Subgraph:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)
    raw_context: str = ""

# ── Parser ────────────────────────────────────────────────────────────────────
 
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

# ── Main ──────────────────────────────────────────────────────────────────────

async def gather_subgraph(
    query: str,
    mode: str = "hybrid",
    working_dir: str = WORKING_DIR,
) -> Subgraph:
    """
    Initialize LightRAG, run a retrieval-only query, and return
    the parsed Subgraph (entities + relations + chunks).
    """
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        summary_max_tokens=8192,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"temperature": 0},
            "timeout": int("200")
        },
        embedding_func=ollama_embed,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    # only_need_context=True → skip LLM, return the raw context string
    param = QueryParam(mode=mode, only_need_context=True, enable_rerank=False, top_k=2)
    context: str = await rag.aquery(query, param=param)

    print(context)

    return parse_context(context)


def print_subgraph(sg: Subgraph) -> None:
    print(f"\n{'='*60}")
    print(f"  SUBGRAPH SUMMARY")
    print(f"{'='*60}")
    print(f"  Entities  : {len(sg.entities)}")
    print(f"  Relations : {len(sg.relations)}")
    print(f"  Chunks    : {len(sg.chunks)}")

    print(f"\n── Entities {'─'*48}")
    for e in sg.entities:
        print(f"  [{e.type}] {e.name}  (rank={e.rank:.2f})")
        if e.description:
            preview = e.description[:120].replace("\n", " ")
            print(f"    └─ {preview}{'...' if len(e.description) > 120 else ''}")

    print(f"\n── Relations {'─'*47}")
    for r in sg.relations:
        print(f"  {r.src}  →  {r.tgt}  (w={r.weight:.2f})  [{r.keywords}]")
        if r.description:
            preview = r.description[:120].replace("\n", " ")
            print(f"    └─ {preview}{'...' if len(r.description) > 120 else ''}")

    print(f"\n── Source Chunks {'─'*43}")
    for i, chunk in enumerate(sg.chunks, 1):
        preview = chunk[:200].replace("\n", " ")
        print(f"  [{i}] {preview}{'...' if len(chunk) > 200 else ''}")

    print(f"\n{'='*60}\n")


async def main():
    sg = await gather_subgraph(query=QUERY, mode=MODE)
    print_subgraph(sg)

    # ── Ready for perturbation ────────────────────────────────────────────────
    # At this point sg.entities and sg.relations are plain Python lists
    # of dataclasses — easy to filter, shuffle, corrupt, or augment
    # before feeding back to the LLM context.
    #
    # Example: drop low-rank entities
    # sg.entities = [e for e in sg.entities if e.rank > 1.0]
    #
    # Example: drop weak relations
    # sg.relations = [r for r in sg.relations if r.weight > 0.5]

    return sg


if __name__ == "__main__":
    asyncio.run(main())