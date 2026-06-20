import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))            # competitors/Shapley
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))          # repo root
_RAGEX_DIR = os.path.join(_REPO_ROOT, "competitors", "RAGEX-RAGE-SHAPLEY")
if _RAGEX_DIR not in sys.path:
    sys.path.insert(0, _RAGEX_DIR)

from retrieval.parser import parse_context as _ragex_parse_context  # noqa: E402


def parse_chunks(context: str) -> list[str]:
    if not context:
        return []
    sg = _ragex_parse_context(context)
    return [c.strip() for c in (sg.chunks or []) if c and c.strip()]


def render_context_from_chunks(chunks) -> str:
    lines = [
        "Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):",
        "```json",
    ]
    for i, chunk in enumerate(chunks, 1):
        lines.append(json.dumps({"reference_id": str(i), "content": chunk}))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


async def retrieve_chunks(rag, query: str, mode: str = "hybrid", top_k: int = 2):
    from lightrag import QueryParam  
    param = QueryParam(mode=mode, top_k=top_k, only_need_context=True, enable_rerank=False)
    context_str: str = await rag.aquery(query, param=param)
    if not context_str:
        return "", []
    chunks = parse_chunks(context_str)
    return render_context_from_chunks(chunks), chunks
