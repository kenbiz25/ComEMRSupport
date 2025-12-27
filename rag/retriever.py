
# rag/retriever.py
from typing import List
import os
import numpy as np

from config.settings import settings

class Retriever:
    """
    Embedding provider that switches automatically:
      - OpenAI embeddings when EMBED_MODEL starts with 'text-embedding'
      - Sentence-Transformers otherwise (HF model id)
    Returned vectors are L2-normalized (recommended for cosine distance).

    Usage:
      r = Retriever()
      vecs = r.embed(["hello", "world"])
      dim = r.get_dim()
    """

    def __init__(self):
        self.embed_model = settings.EMBED_MODEL

        # Decide provider
        self.use_openai = self.embed_model.startswith("text-embedding")

        if self.use_openai:
            # OpenAI embeddings branch
            from openai import OpenAI
            if not settings.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set in environment/.env")
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            # Sentence-Transformers branch
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(self.embed_model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Returns a list of L2-normalized vectors (List[List[float]]).
        """
        if self.use_openai:
            # NOTE: OpenAI SDK supports batching natively with 'input=list[str]'
            resp = self.client.embeddings.create(
                model=self.embed_model,
                input=texts
            )
            vecs = [d.embedding for d in resp.data]
        else:
            # If you keep ST path for other models
            vecs = self.st_model.encode(texts, normalize_embeddings=False).tolist()

        return self._normalize(vecs)

    def get_dim(self) -> int:
        """
        Returns the embedding dimension without mutating store.
        """
        if self.use_openai:
            # Known dims for text-embedding-3-*
            name = self.embed_model
            if "text-embedding-3-small" in name:
                return 1536
            if "text-embedding-3-large" in name:
                return 3072
            # Fallback probe (covers future models)
            resp = self.client.embeddings.create(model=name, input="probe")
            return len(resp.data[0].embedding)
        else:
            v = self.st_model.encode(["probe"], normalize_embeddings=False)[0]
            return len(v)

    @staticmethod
    def _normalize(vecs: List[List[float]]) -> List[List[float]]:
        """
        L2-normalize vectors for cosine similarity; safe for FAISS/cosine.
        """
        nvecs: List[List[float]] = []
        for v in vecs:
            arr = np.asarray(v, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm == 0:
                nvecs.append(arr.tolist())
            else:
                nvecs.append((arr / norm).tolist())
        return nvecs
