from lightrag import LightRAG, QueryParam
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from retrieval.base import *
from retrieval.parser import *
from LLM.llm_utils import vllm_model_complete, VLLM_MODEL, EMBEDDING_DIM, sentence_transformer_embed

setup_logger("lightrag", level="WARNING")

# ── Config ────────────────────────────────────────────────────────────────────

# WORKING_DIR_SYNTHETIC  = "KGs/lightrag/synthetic"
# WORKING_DIR_HOTPOTQA  = "KGs/lightrag/hotpotqa"
MODE         = "hybrid"
TOP_K        = 2

# ──────────────────────────────────────────────────────────────────────────────

# async def initialize_lightrag(working_dir: str = WORKING_DIR_SYNTHETIC):
async def initialize_lightrag(working_dir):
    '''Initialize LightRAG'''

    rag = LightRAG(
        working_dir=working_dir,

        llm_model_func=vllm_model_complete,
        llm_model_name=VLLM_MODEL,
        summary_max_tokens=8192,
        llm_model_kwargs={
            "temperature": 0,
            "max_tokens": 8192,
        },
        
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=512,          
            func=sentence_transformer_embed,
        ),

        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract= False,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    await rag.aclear_cache()

    return rag

async def retrieve_subgraph(rag: LightRAG, query: str, mode: str = MODE, top_k: int = TOP_K):
    '''
    Retrieve relevant subgraph (entities/relations/chunks)
    '''

    param = QueryParam(mode=mode, only_need_context=True, enable_rerank=False, top_k=top_k, include_references=False)
    context: str = await rag.aquery(query, param=param)

    parsed_context = parse_context(context)
    parsed_graph = parse_graph(parsed_context)
    filtered_context = graph_to_context(parsed_graph)

    return filtered_context

async def retrieve_subgraph_objects(rag: LightRAG, query: str, mode: str = MODE, top_k: int = TOP_K):
    '''
    Like retrieve_subgraph, but also returns the parsed Subgraph (ordered
    entities + relations) so callers can attribute/permute over the discrete
    objects. The context string is rendered with render_context, so it is
    byte-identical to what retrieve_subgraph / RAG feeds the LLM.
    '''
    param = QueryParam(mode=mode, only_need_context=True, enable_rerank=False, top_k=top_k, include_references=False)
    context: str = await rag.aquery(query, param=param)

    parsed_context = parse_context(context)
    parsed_graph = parse_graph(parsed_context)
    sg = graph_to_subgraph(parsed_graph)
    filtered_context = render_context(sg.entities, sg.relations)

    return filtered_context, sg

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
