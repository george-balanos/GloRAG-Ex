from pydantic import BaseModel, Field
from typing import Literal
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from src.prompts.prompts import LLM_AS_A_JUDGE
from src.llm.utils import get_judge_llm

import json
import asyncio

# ---------------------------------------------
#  Schema
# ---------------------------------------------

class JudgeScore(BaseModel):
    score: Literal["0", "1"] = Field(
        description="This is the score of the LLM-as-a-Judge."
    )


# ---------------------------------------------
#  Core functions
# ---------------------------------------------

def get_binary_score(judge_output) -> int:
    try:
        return int(judge_output)
    except:
        return 0


async def judge_response(question: str, generated_answer: str, ground_truth: str) -> int:
    prompt = LLM_AS_A_JUDGE.format(
        question=question,
        system_generated_answer=generated_answer,
        ground_truth_answer=ground_truth,
    )

    llm       = get_judge_llm()
    tokenizer = llm.get_tokenizer()

    full_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        structured_outputs=StructuredOutputsParams(choice=["0", "1"]),  # ← this
    )

    loop = asyncio.get_event_loop()
    outputs = await loop.run_in_executor(
        None,
        lambda: llm.generate([full_prompt], sampling_params, use_tqdm=False),
    )

    content = outputs[0].outputs[0].text.strip()
    return int(content)