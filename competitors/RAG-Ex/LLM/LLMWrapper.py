from typing import Dict, Any
import json
from ollama import chat
from prompts import LLM_AS_A_JUDGE_PROMPT


class LLMWrapper:
    def __init__(self, model: str = "mistral-small3.2:24b-instruct-2506-q4_K_M"):
        self.model = model

    def _call(self, query: Dict[str, Any], prompt: str = None):

        if prompt == "LLM_AS_A_JUDGE_PROMPT":
            prompt_text = LLM_AS_A_JUDGE_PROMPT.format(**query)
        else:
            prompt_text = str(query)

        response = chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            options={"temperature": 0},
        )

        content = response["message"]["content"]

        try:
            return json.loads(content)
        except Exception:
            return content
    
    def write_answer_to_file(self, filename, answer):
        pass

    def perturbe(self, query: Dict[str, Any], perturbation: str):
        results = []

        if perturbation == "sliding_window":
            results.append(result)    
        pass
    
    def answer_question(self, question: str):
        pass
    

lw = LLMWrapper()
answer = lw._call({
    "question": "What is the capital of France?",
})
print(answer)