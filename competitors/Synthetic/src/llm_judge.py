from pydantic import BaseModel, Field
from typing import Literal
from ollama import chat
from src.prompts.prompts import LLM_AS_A_JUDGE

import json

llm_model = "qwen3:latest"

class JudgeScore(BaseModel):
    score: Literal["0", "1"] = Field(
        "This is the score of the LLM-as-a-Judge."
    )

def judge_response(question, generated_answer, ground_truth):
    prompt = LLM_AS_A_JUDGE.format(question=question, system_generated_answer=generated_answer, ground_truth_answer=ground_truth)

    response = chat(
        model=llm_model, 
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        },
        format=JudgeScore.model_json_schema(),
        think=False
    )

    try: 
        content = response["message"]["content"]
        parsed = json.loads(content)
        score = parsed.get("score", -1)
        return score
    except:
        print("Failed to parse json")
        return -1
    
def get_binary_score(judge_output):
    try:
        score = int(judge_output)
        return score
    except:
        return 0