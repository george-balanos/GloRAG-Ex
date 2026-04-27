from src.retrieve import sentence_transformer_embed
import numpy as np

def cosine_similarity(e1: np.array, e2: np.array):
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

def cosine_similarity_norm(e1: np.array, e2: np.array):
    '''Embeddings e1 and e2 are normalized.'''
    return float(np.dot(e1, e2))

async def compute_answer_similarity(original_answer: str, perturbed_answer: str):
    e1, e2 = await sentence_transformer_embed([original_answer, perturbed_answer])
    return cosine_similarity_norm(np.array(e1), np.array(e2))