from lightrag.prompt import PROMPTS
from src.retrieve import retrieve_subgraph, initialize_lightrag

import asyncio

def build_rag_system_prompt(context) -> str:
    return PROMPTS["rag_response"].format(
        context_data=context,
        response_type="Single Sentence, without references and extra explanations.",
        user_prompt=""
    )

async def query(rag, context, question: str):
    system_prompt = PROMPTS["rag_response"].format(
        context_data=context,
        response_type="Single Sentence, without references and extra explanations.",
        user_prompt=""
    )

    response = await rag.llm_model_func(
        question, 
        system_prompt=system_prompt
    )

    return response

async def main():
    rag = await initialize_lightrag()
    context = await retrieve_subgraph(rag)

    await query(rag, context)

if __name__ == "__main__":
    asyncio.run(main())