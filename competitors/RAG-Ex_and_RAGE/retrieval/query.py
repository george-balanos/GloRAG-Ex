from lightrag.prompt import PROMPTS
from retrieval.retrieve import retrieve_subgraph, initialize_lightrag, QUERY

import asyncio

async def query(rag, context):
    # print(PROMPTS["rag_response"])

    system_prompt = PROMPTS["rag_response"].format(
        context_data=context,
        response_type="Single sentence.",
        user_prompt=""
    )

    response = await rag.llm_model_func(
        QUERY, 
        system_prompt=system_prompt
    )

    print(response)

async def main():
    rag = await initialize_lightrag()
    context = await retrieve_subgraph(rag)

    await query(rag, context)

if __name__ == "__main__":
    asyncio.run(main())