from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import setup_logger
from lightrag.kg.shared_storage import initialize_pipeline_status
from src.base import *
from src.parser import parse_context

import asyncio

setup_logger("lightrag", level="WARNING")

# ── Config ────────────────────────────────────────────────────────────────────

WORKING_DIR  = "./synthetic"
QUERY        = "What are the two primary materials used to construct a Xylotian 'Sky-Skiff' hull?"
MODE         = "hybrid"
TOP_K        = 2

OLLAMA_HOST  = "http://localhost:11434"
LLM_MODEL    = "mistral-small3.2:24b-instruct-2506-q4_K_M"
EMBED_MODEL  = "all-minilm:latest"
EMBED_DIM    = 384

# ──────────────────────────────────────────────────────────────────────────────

async def initialize_lightrag(working_dir: str = WORKING_DIR):
    '''Initialize LightRAG'''

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

    return rag

async def retrieve_subgraph(rag: LightRAG, query: str=QUERY, mode: str = MODE, top_k: int = TOP_K) -> Subgraph:
    '''
    Retrieve relevant subgraph (entities/relations/chunks)
    '''

    param = QueryParam(mode=mode, only_need_context=True, enable_rerank=False, top_k=top_k)
    context: str = await rag.aquery(query, param=param)

    # print(context)

    return context

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
    rag = await initialize_lightrag(WORKING_DIR)
    sg = await retrieve_subgraph(rag=rag, query=QUERY, mode=MODE, top_k=TOP_K)
    parsed_subgraph = parse_context(sg)

    print_subgraph(parsed_subgraph)

if __name__ == "__main__":
    asyncio.run(main())