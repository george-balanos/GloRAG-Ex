from lightrag.prompt import PROMPTS
from src.retrieve import retrieve_subgraph, initialize_lightrag, QUERY

import asyncio

async def query(rag, context, question: str):
    # print(PROMPTS["rag_response"])

    system_prompt = PROMPTS["rag_response"].format(
        context_data=context,
        response_type="Single Sentence, without references and extra explanations.",
        user_prompt=""
    )

    response = await rag.llm_model_func(
        question, 
        system_prompt=system_prompt
    )

    # print(response)
    return response

async def main():
    rag = await initialize_lightrag()
    context = await retrieve_subgraph(rag)

    await query(rag, context)

if __name__ == "__main__":
    asyncio.run(main())