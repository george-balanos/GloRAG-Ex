import numpy as np
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from SHapRAG.rag_shap import ContextAttribution

print("Loading local lightweight chat model and tokenizer...")
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  

accelerator = Accelerator()
device = accelerator.device

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    device_map={"": device}
)

list_of_document_strings = [
    "Document 1: The primary color of a ripe banana is yellow.",
    "Document 2: Penguins are flightless birds living mostly south.",
    "Document 3: Bananas turn brown when left inside a cold fridge."
]

user_query = "What color are bananas and how do they change in the fridge?"

print("Initializing ContextAttribution framework...")
attributor = ContextAttribution(
    items=list_of_document_strings,
    query=user_query,
    prepared_model=model,
    prepared_tokenizer=tokenizer,
    accelerator=accelerator,
    utility_mode="log-perplexity", 
    verbose=True
)

print("Calculating EXACT Shapley Values (SV) on GPU...")
scores = attributor._calculate_exact(method='SV')

print("\n" + "="*40)
print("  ATTRIBUTION SCORES FOR EACH CONTEXT ITEM")
print("="*40)
for idx, score in enumerate(scores):
    print(f"Item [{idx}]: Score = {score:.4f} | Content: {list_of_document_strings[idx][:60]}...")
print("="*40)