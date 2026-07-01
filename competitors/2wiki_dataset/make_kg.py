import os
import asyncio
import pandas as pd
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./2wiki_fixed"
CSV_FILE_PATH = "2wiki_dev.csv"
CUSTOM_DELIMITER = "|||"
PREVIEW = 3

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
            "timeout": 600,
        },
        embedding_func=ollama_embed,
        # embedding_batch_num=1,
        # embedding_func_max_async=1,
        # enable_llm_cache=False,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    await rag.aclear_cache()
    return rag


def build_document(supporting_paragraph: str) -> str:
    if not isinstance(supporting_paragraph, str):
        return ""
    parts = [p.strip() for p in supporting_paragraph.split(CUSTOM_DELIMITER) if p.strip()]
    return "\n\n".join(parts)


async def main():
    rag = await initialize_rag()

    print(f"Loading data from {CSV_FILE_PATH}...")
    df = pd.read_csv(CSV_FILE_PATH, delimiter="|")
    df.columns = df.columns.str.strip()
    df = df.iloc[:300]
    print(f"Loaded {len(df)} rows.")

    documents = []
    for idx, row in df.iterrows():
        doc = build_document(row.get("Supporting Paragraph", ""))
        if doc:
            documents.append(doc)
            if idx < PREVIEW:
                print(f"\n{'='*60}")
                print(f"Row {idx} | Question: {row.get('Question', '').strip()}")
                print(f"{'-'*60}")
                print(doc)
                print(f"{'='*60}")

    print(f"\nTotal documents prepared: {len(documents)}")

    print("Inserting into LightRAG...")
    try:
        await rag.ainsert(documents)
        print("\nKnowledge Graph build complete!")
    except Exception as e:
        print(f"\nError during ingestion: {e}")


if __name__ == "__main__":
    asyncio.run(main())