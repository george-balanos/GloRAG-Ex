from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from retrieval.base import *
from retrieval.parser import parse_context
from sentence_transformers import SentenceTransformer

import asyncio

setup_logger("lightrag", level="WARNING")

# ── Config ────────────────────────────────────────────────────────────────────

WORKING_DIR  = "/mnt/qnap/cs05058/LightRAG/xylotian_storage"
QUERY        = "What are the two primary materials used to construct a Xylotian 'Sky-Skiff' hull?"
MODE         = "hybrid"
TOP_K        = 2

OLLAMA_HOST  = "http://localhost:11434"
LLM_MODEL    = "mistral-small3.2:24b-instruct-2506-q4_K_M"

# 1. Correctly initialize the model globally
model = SentenceTransformer('all-MiniLM-L6-v2')

# ──────────────────────────────────────────────────────────────────────────────

async def initialize_lightrag(working_dir: str = WORKING_DIR):
    '''Initialize LightRAG'''

    # The adapter now correctly references the global 'model' variable
    async def huggingface_embedding_adapter(texts):
        return model.encode(texts, show_progress_bar=False)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={
            "host": OLLAMA_HOST,
            "options": {"temperature": 0},
            "timeout": 200
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=384, 
            max_token_size=8192,
            func=huggingface_embedding_adapter,
        ),
    )
    
    await rag.initialize_storages()
    # Ensure pipeline status is also initialized
    await initialize_pipeline_status()
    
    return rag

async def retrieve_subgraph(rag: LightRAG, query: str=QUERY, mode: str = MODE, top_k: int = TOP_K) -> Subgraph:
    '''Retrieve relevant subgraph'''
    param = QueryParam(mode=mode, only_need_context=True, enable_rerank=False, top_k=top_k)
    context: str = await rag.aquery(query, param=param)

    # If context is None (query failed), return an empty Subgraph to avoid crashes
    if context is None:
        print("⚠️ Warning: Retrieval failed and returned None.")
        return Subgraph(raw_context="")

    return context

def print_subgraph(sg: Subgraph) -> None:
    # Logic remains the same as your source
    print(f"\n{'='*60}")
    print(f"  SUBGRAPH SUMMARY")
    print(f"{'='*60}")
    if not sg or not hasattr(sg, 'entities'):
        print("  No data found in subgraph.")
        return

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
        preview = chunk
        print(f"  [{i}] {preview}{'...' if len(chunk) > 200 else ''}")

    print(f"\n{'='*60}\n")


async def main():
    rag = await initialize_lightrag(WORKING_DIR)
    
    # Retrieve the raw context string
    context_str = await retrieve_subgraph(rag=rag, query=QUERY, mode=MODE, top_k=TOP_K)
    
    # Check if context exists before parsing to prevent 'NoneType' error
    if context_str:
        parsed_subgraph = parse_context(context_str)
        print_subgraph(parsed_subgraph)
    else:
        print("No context retrieved. Check embedding dimension or storage path.")

if __name__ == "__main__":
    asyncio.run(main())