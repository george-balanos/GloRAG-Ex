import json
import base64
import zlib
import asyncio
import numpy as np
from pydantic import BaseModel, Field
from typing import Literal
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from sentence_transformers import SentenceTransformer

from src.prompts.prompts import MEDICAL_RAG_PROMPT
from src.llm.utils import get_llm

VDB_ENTITIES  = "KGs/medical/vdb_entities.json"
VDB_RELATIONS = "KGs/medical/vdb_relationships.json"


class SelectedOption(BaseModel):
    selected_option: Literal["A", "B", "C", "D"]


def decode_vector(vec_str: str) -> np.ndarray:

    compressed = base64.b64decode(vec_str.encode())
    decompressed = zlib.decompress(compressed)
    return np.frombuffer(decompressed, dtype=np.float16).astype(np.float32)


def retrieve_kg_context(query: str, top_k_entities: int = 3, top_k_relations: int = 3) -> str:

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode(query, convert_to_numpy=True).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec) + 1e-10

    context_segments = []

    try:
        with open(VDB_ENTITIES, "r") as f:
            entity_data = json.load(f)["data"]
        
        entity_scores = []
        for item in entity_data:
            vec = decode_vector(item["vector"])
            score = float(np.dot(query_vec, vec))
            entity_scores.append((score, item))
        
        entity_scores.sort(key=lambda x: x, reverse=True)
        
        context_segments.append("=== RELEVANT MEDICAL ENTITIES ===")
        for score, item in entity_scores[:top_k_entities]:
            desc = item['content'] if item['content'] else "No description available."
            context_segments.append(f"Entity: {item['entity_name']} ({item['entity_category']})\nDescription: {desc}\n")
            
    except FileNotFoundError:
        context_segments.append("[Warning: Entity VDB file not found]")

    try:
        with open(VDB_RELATIONS, "r") as f:
            rel_data = json.load(f)["data"]
            
        rel_scores = []
        for item in rel_data:
            vec = decode_vector(item["vector"])
            score = float(np.dot(query_vec, vec))
            rel_scores.append((score, item))
            
        rel_scores.sort(key=lambda x: x, reverse=True)
        
        context_segments.append("=== RELEVANT MEDICAL RELATIONSHIPS ===")
        for score, item in rel_scores[:top_k_relations]:
            context_segments.append(f"{item['content']}\n")
            
    except FileNotFoundError:
        context_segments.append("[Warning: Relationship VDB file not found]")

    return "\n".join(context_segments)


async def query_rag(input_question: str, options: str, context: str):
    prompt = MEDICAL_RAG_PROMPT.format(input_question=input_question, options=options, context=context)

    llm = get_llm()
    tokenizer = llm.get_tokenizer()

    full_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenizer=True, 
        add_generation_prompt=True
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        structured_outputs=StructuredOutputsParams(
            json=SelectedOption.model_json_schema()
        )
    )

    loop = asyncio.get_event_loop()
    outputs = await loop.run_in_executor(
        None,
        lambda: llm.generate([full_prompt], sampling_params, use_tqdm=False),
    )

    raw = outputs.outputs.text.strip()

    try:
        parsed = SelectedOption.model_validate_json(raw)
        return parsed.selected_option
    except Exception:
        return ""
    

async def query_llm_only(input_question: str, options: str):
    prompt = MEDICAL_RAG_PROMPT.format(input_question=input_question, options=options, context="")

    llm = get_llm()
    tokenizer = llm.get_tokenizer()

    full_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenizer=True, ### For Mistral,
        add_generation_prompt=True
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        structured_outputs=StructuredOutputsParams(
            json=SelectedOption.model_json_schema()
        )
    )

    loop = asyncio.get_event_loop()
    outputs = await loop.run_in_executor(
        None,
        lambda: llm.generate([full_prompt], sampling_params, use_tqdm=False),
    )

    raw = outputs.outputs.text.strip()

    try:
        parsed = SelectedOption.model_validate_json(raw)
        return parsed.selected_option
    except Exception:
        return ""
    
async def main():
    question = "Can bedside assessment reliably exclude aspiration following acute stroke?"
    
    options =  """{
        "A": "yes",
        "B": "no",
        "C": "maybe"
    }"""

    print("Retrieving node and edge descriptions from Knowledge Graph...")
    kg_context = retrieve_kg_context(question, top_k_entities=3, top_k_relations=3)

    print("Submitting query to vLLM engine with Graph Context (RAG)...")
    response = await query_rag(input_question=question, options=options, context=kg_context)
    print(f"\nRAG Response: {response}")
    
    print("\nSubmitting query without Context (Baseline)...")
    baseline_response = await query_llm_only(input_question=question, options=options)
    print(f"Baseline Response: {baseline_response}")
    

if __name__ == "__main__":
    asyncio.run(main())