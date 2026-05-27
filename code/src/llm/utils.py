from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer

import asyncio 
import numpy as np
import os

VLLM_MODEL      = "mistralai/Mistral-Small-3.2-24B-Instruct-2506" ## Original
# VLLM_MODEL      = "google/gemma-3-27b-it"
# VLLM_MODEL      = "meta-llama/Llama-3.1-8B-Instruct"

JUDGE_MODEL     = "Qwen/Qwen2.5-7B-Instruct"                      ## Original
# JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_llm_instance: LLM | None = None
_emb_instance: SentenceTransformer | None = None
_judge_instance: LLM | None = None

def get_llm() -> LLM:
    global _llm_instance

    if _llm_instance is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        _llm_instance = LLM(model=VLLM_MODEL, gpu_memory_utilization=0.5, max_model_len=16768)
    return _llm_instance

def get_judge_llm() -> LLM:
    global _judge_instance
    
    if _judge_instance is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        _judge_instance = LLM(model=JUDGE_MODEL, gpu_memory_utilization=0.35, max_model_len=8192)
    return _judge_instance

def get_embedding_model() -> SentenceTransformer:
    global _emb_instance

    if _emb_instance is None:
        _emb_instance = SentenceTransformer(EMBEDDING_MODEL)
    return _emb_instance

async def vllm_model_complete(
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 8192,
    temperature: float = 0,
    **kwargs,
) -> str:
    llm       = get_llm()
    tokenizer = llm.get_tokenizer()
 
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if VLLM_MODEL == "mistralai/Mistral-Small-3.2-24B-Instruct-2506":
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    else: 
        full_prompt = tokenizer.apply_chat_template(
            messages,
            # tokenize=True,
            tokenize=False,
            add_generation_prompt=True,
        )
 
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    loop    = asyncio.get_event_loop()
    outputs = await loop.run_in_executor(
        None,
        lambda: llm.generate([full_prompt], sampling_params, use_tqdm=False),
    )
 
    return outputs[0].outputs[0].text.strip()

async def sentence_transformer_embed(texts: list[str], **kwargs,) -> list[list[float]]:
    model = get_embedding_model()
 
    loop = asyncio.get_event_loop()
    vecs = await loop.run_in_executor(
        None,
        lambda: model.encode(
            texts,
            normalize_embeddings=True,   
            batch_size=32,               
            show_progress_bar=False,
        ),
    )
 
    return vecs

if __name__ == "__main__":
    async def _test():
        # Test LLM completion
        answer = await vllm_model_complete(
            prompt="What is the capital of France?",
            system_prompt="You are a helpful assistant. Answer concisely.",
        )
        print("LLM answer:", answer)
 
        # Test embeddings
        vecs = await sentence_transformer_embed(["Hello world", "Bonjour le monde"])
        print(f"Embedding shape: {vecs.shape}")
        print(f"First 5 dims:   {vecs[0][:5]}")
        print(f"Vector norm:    {np.linalg.norm(vecs[0]):.4f}")
 
    asyncio.run(_test())