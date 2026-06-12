import os
import asyncio
import pandas as pd
from tqdm import tqdm
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./musique_kg"
CSV_FILE_PATH = "musique_train.csv"
CUSTOM_DELIMITER = "|||"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="mistral:latest",
        summary_max_tokens=8192,
        llm_model_kwargs={
            "host": "http://localhost:11434/",
            "options": {'temperature': 0},
            "timeout": 200
        },
        embedding_func=ollama_embed,
    )
    
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def main():
    rag = await initialize_rag()

    print(f"Loading data from {CSV_FILE_PATH}...")
    full_df = pd.read_csv(CSV_FILE_PATH, delimiter="|")

    df = full_df.head(101)
    print(f"Loaded {len(full_df)} total rows. Sliced down to the first {len(df)} rows.")

    text_chunks_to_insert = set()

    print("Parsing rows and extracting all support paragraph lines...")
    for idx, row in df.iterrows():
        supporting_paragraph_cell = str(row.get("Supporting Paragraph", ""))
        
        paragraphs = [
            p.strip() 
            for p in supporting_paragraph_cell.split(CUSTOM_DELIMITER) 
            if p.strip()
        ]
        
        for para in paragraphs:
            text_chunks_to_insert.add(para)

    unique_chunks = list(text_chunks_to_insert)
    print(f"Found a total of {len(unique_chunks)} distinct support paragraphs across those 101 rows.")

    print("Submitting chunks to the LightRAG execution pipeline...")
    try:
        await rag.ainsert(unique_chunks)
        print("\nSuccessfully finished building the partial Knowledge Graph!")
    except Exception as e:
        print(f"\nAn error occurred during batch ingestion: {e}")

if __name__ == "__main__":
    asyncio.run(main())