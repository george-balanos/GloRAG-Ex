from pydantic import BaseModel, Field
from typing import Literal
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from src.prompts.prompts import MEDICAL_RAG_PROMPT
from src.llm.utils import get_llm

import asyncio

# ---------------------------------------------
#  Schema
# ---------------------------------------------

class SelectedOption(BaseModel):
    selected_option: Literal[
        "A", "B", "C", "D"
    ]

# ---------------------------------------------
#  Core function
# ---------------------------------------------

async def query_rag(input_question: str, options: str, context: str):
    prompt = MEDICAL_RAG_PROMPT.format(input_question=input_question, options=options, context=context)

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

    raw = outputs[0].outputs[0].text.strip()

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

    raw = outputs[0].outputs[0].text.strip()

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

    response = await query_llm_only(input_question=question, options=options)
    print(f"Response: {response}")
    

if __name__ == "__main__":
    asyncio.run(main())