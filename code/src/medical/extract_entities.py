from pydantic import BaseModel, Field
from typing import Literal
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from src.prompts.prompts import EXTRACT_MEDICAL_ENTITIES
from src.llm.utils import get_llm

import asyncio

# ---------------------------------------------
#  Schema
# ---------------------------------------------

class ExtractedEntity(BaseModel):
    entity_category: Literal[
        "Disease", "Symptom", "Drug", "Anatomy", "Treatment",
        "Complication", "Etiology", "Patient_Pop", "Professional_Role", "Diagnostic"
    ]
    entity_name: str = Field(description="Standardized name of the extracted entity.")

class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]

# ---------------------------------------------
#  Core function
# ---------------------------------------------

async def extract_entities(input_text: str) -> list[dict]:
    prompt = EXTRACT_MEDICAL_ENTITIES.format(input_text=input_text)

    llm = get_llm() 
    tokenizer = llm.get_tokenizer()

    full_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )

    sampling_params = SamplingParams(
        temperature=0,
        seed=42,
        max_tokens=1024,
        structured_outputs=StructuredOutputsParams(
            json=ExtractionResult.model_json_schema()
        ),
    )

    loop = asyncio.get_event_loop()
    outputs = await loop.run_in_executor(
        None,
        lambda: llm.generate([full_prompt], sampling_params, use_tqdm=False),
    )

    raw = outputs[0].outputs[0].text.strip()

    try:
        parsed = ExtractionResult.model_validate_json(raw)
        return [e.model_dump() for e in parsed.entities]
    except Exception:
        return []
    
async def main():
    question = "Can bedside assessment reliably exclude aspiration following acute stroke?"

    entities = await extract_entities(question)
    print(f"Extracted entities: {entities}")

if __name__ == "__main__":
    asyncio.run(main())