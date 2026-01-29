# rag/retriever.py
# ComEMR Knowledge Retriever – Production Scale

from typing import List, Dict, Any, Optional  # ✅ Added Optional
import logging
import numpy as np

from config.settings import settings
from core.knowledge.store_faiss import FaissStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Conversational retriever:
    - Embeds queries
    - FAISS ANN search
    - Intent & audience aware scoring
    - Returns ranked KB chunks for Composer
    """

    def __init__(self, top_k: Optional[int] = None):
        self.top_k = top_k or settings.TOP_K
        self.embed_model = settings.EMBED_MODEL
        self.use_openai = self.embed_model.startswith("text-embedding")

        # Embedding client
        if self.use_openai:
            from openai import OpenAI
            if not settings.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY missing")
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(settings.HF_EMBED_MODEL)

        # FAISS store
        self.store = FaissStore(
            dim=self.get_dim(),
            index_dir=settings.FAISS_INDEX_DIR
        )

        logger.info(f"Retriever ready | model={self.embed_model} | top_k={self.top_k}")
        logger.info(f"KB size: {self.store.size()} documents")

    # ─────────────────────────────────────────────
    # EMBEDDINGS
    # ─────────────────────────────────────────────
    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.use_openai:
            resp = self.client.embeddings.create(
                model=self.embed_model,
                input=texts
            )
            vectors = [d.embedding for d in resp.data]
        else:
            vectors = self.st_model.encode(
                texts, normalize_embeddings=False
            ).tolist()

        return self._normalize(vectors)

    # ─────────────────────────────────────────────
    # RETRIEVAL
    # ─────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        intent: Optional[str] = None,
        audience: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve KB chunks with optional intent & audience boosting.
        """
        if not query:
            return []

        top_k = top_k or self.top_k

        logger.info(f"KB search | q='{query[:60]}' | intent={intent} | audience={audience}")

        # Embed query
        query_vec = self.embed([query])[0]

        # Raw FAISS search
        results = self.store.search(query_vec, top_k=top_k * 2)

        if not results:
            logger.warning("No KB matches found")
            return []

        # Re-rank with metadata awareness
        ranked = self._rerank(results, intent, audience)

        # Trim to top_k
        final = ranked[:top_k]

        # Log confidence
        scores = [round(float(r.get("score", 0.0)), 3) for r in final]
        logger.info(f"KB confidence scores: {scores}")

        return final

    # Backward compatibility
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        return self.retrieve(query, top_k=top_k)

    # ─────────────────────────────────────────────
    # RERANKING LOGIC
    # ─────────────────────────────────────────────
    def _rerank(
        self,
        results: List[Dict[str, Any]],
        intent: Optional[str],
        audience: Optional[str],
    ) -> List[Dict[str, Any]]:
        scored = []

        for r in results:
            base = float(r.get("score", 0.0))
            meta = r.get("metadata") or {}

            boost = 0.0

            # Intent match boost
            if intent and meta.get("intent") == intent:
                boost += 0.15

            # Audience match boost
            if audience:
                aud = meta.get("audience")
                if aud == audience:
                    boost += 0.10
                elif aud == "both":
                    boost += 0.05

            r["score"] = min(base + boost, 1.0)
            scored.append(r)

        return sorted(scored, key=lambda x: x["score"], reverse=True)

    # ─────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────
    def get_dim(self) -> int:
        if self.use_openai:
            if "text-embedding-3-small" in self.embed_model:
                return 1536
            if "text-embedding-3-large" in self.embed_model:
                return 3072

            resp = self.client.embeddings.create(
                model=self.embed_model,
                input="probe"
            )
            return len(resp.data[0].embedding)
        else:
            vec = self.st_model.encode(
                ["probe"], normalize_embeddings=False
            )[0]
            return len(vec)

    @staticmethod
    def _normalize(vectors: List[List[float]]) -> List[List[float]]:
        out = []
        for v in vectors:
            arr = np.asarray(v, dtype=np.float32)
            norm = np.linalg.norm(arr)
            out.append((arr / norm).tolist() if norm else arr.tolist())
        return out
