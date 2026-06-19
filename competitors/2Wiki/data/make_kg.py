import os
import asyncio
import pandas as pd
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./2wiki_kg"
CSV_FILE_PATH = "2wiki_dev.csv"

CUSTOM_DELIMITER = "|||"

os.makedirs(WORKING_DIR, exist_ok=True)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="mistral-small3.2:24b-instruct-2506-q4_K_M",
        summary_max_tokens=8192,
        llm_model_kwargs={
            "host": "http://localhost:11434/",
            "options": {"temperature": 0},
            "timeout": 200,
        },
        embedding_func=ollama_embed,
        embedding_batch_num=1,
        embedding_func_max_async=1,
        enable_llm_cache=False,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    await rag.aclear_cache()

    return rag


def extract_supporting_chunks(cell: str):
    """
    Split supporting paragraph column into individual chunks
    """
    if not isinstance(cell, str):
        return []

    return [
        p.strip()
        for p in cell.split(CUSTOM_DELIMITER)
        if p.strip()
    ]


async def main():
    rag = await initialize_rag()

    print(f"Loading data from {CSV_FILE_PATH}...")

    df = pd.read_csv(CSV_FILE_PATH, delimiter="|")

    # optional: limit for testing
    df = df.head(300)

    print(f"Loaded {len(df)} rows.")

    all_chunks = set()

    print("Extracting supporting paragraphs...")

    for _, row in df.iterrows():
        chunks = extract_supporting_chunks(row.get("Supporting Paragraph", ""))
        all_chunks.update(chunks)

    all_chunks = list(all_chunks)

    print(f"Found {len(all_chunks)} unique text chunks.")

    print("Inserting into LightRAG...")

    try:
        await rag.ainsert(all_chunks)
        print("\nKnowledge Graph build complete!")
    except Exception as e:
        print(f"\nError during ingestion: {e}")


if __name__ == "__main__":
    asyncio.run(main())