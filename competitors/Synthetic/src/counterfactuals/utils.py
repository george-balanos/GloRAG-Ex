import numpy as np

def cosine_similarity(e1: np.array, e2: np.array):
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

def cosine_similarity_norm(e1: np.array, e2: np.array):
    '''Embeddings e1 and e2 are normalized.'''
    return float(np.dot(e1, e2))