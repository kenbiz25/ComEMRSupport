
# core/knowledge/store_faiss.py
import faiss, numpy as np
from typing import List, Tuple

class FaissStore:
    def __init__(self, dim: int = 384):
        self.index = faiss.IndexFlatIP(dim)   # inner product (cosine-like)
        self.docs = []

    def upsert(self, content: str, metadata: dict, embedding: List[float]):
        vec = np.array([embedding], dtype='float32')
        self.index.add(vec)
        self.docs.append((content, metadata))

    def search(self, query_emb: List[float], top_k: int = 6) -> List[Tuple[str, float, dict]]:
        vec = np.array([query_emb], dtype='float32')
        scores, idxs = self.index.search(vec, top_k)
        results = []
        for j, i in enumerate(idxs[0]):
            if i == -1:  # no more results
                continue
            content, meta = self.docs[i]
            results.append((content, float(scores[0][j]), meta))
        return results
