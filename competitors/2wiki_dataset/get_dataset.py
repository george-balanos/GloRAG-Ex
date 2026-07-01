from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", "2wikimultihopqa")

ds["dev"].to_json("2wiki_dev.jsonl")