# rag/retriever.py
from typing import List, Dict, Any
import os
import logging

from config.settings import settings
from core.knowledge.store_faiss import FaissStore

logger = logging.getLogger(__name__)

class Retriever:
    """
    Unified retriever that handles both embeddings and vector search.
    
    - Embeds queries using OpenAI or Sentence Transformers
    - Searches FAISS index for relevant documents
    - Returns results with confidence scores
    """

    def __init__(self):
        self.embed_model = settings.EMBED_MODEL
        self.use_openai = self.embed_model.startswith("text-embedding")
        
        # Initialize embedding client
        if self.use_openai:
            from openai import OpenAI
            if not settings.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set in environment/.env")
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(settings.HF_EMBED_MODEL)
        
        # Initialize FAISS store
        self.store = FaissStore(
            dim=self.get_dim(),
            index_dir=settings.FAISS_INDEX_DIR
        )
        
        logger.info(f"Retriever initialized with {self.embed_model}")
        logger.info(f"FAISS index contains {self.store.size()} documents")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts and return L2-normalized vectors
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (L2-normalized)
        """
        if self.use_openai:
            resp = self.client.embeddings.create(
                model=self.embed_model,
                input=texts
            )
            vecs = [d.embedding for d in resp.data]
        else:
            vecs = self.st_model.encode(texts, normalize_embeddings=False).tolist()
        
        return self._normalize(vecs)
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant documents for a query
        
        Args:
            query: User's question/query
            top_k: Number of results to return (default from settings)
            
        Returns:
            List of document dicts with text, score, metadata
        """
        if top_k is None:
            top_k = settings.TOP_K
        
        logger.info(f"Retrieving top-{top_k} documents for query: {query[:50]}...")
        
        # Embed the query
        query_embedding = self.embed([query])[0]
        
        # Search FAISS index
        results = self.store.search(query_embedding, top_k=top_k)
        
        # Log results
        if results:
            logger.info(f"Retrieved {len(results)} documents:")
            for i, res in enumerate(results, 1):
                logger.info(f"  [{i}] Score: {res['score']:.3f} | {res.get('title', 'Unknown')}")
        else:
            logger.warning("No documents found in index!")
        
        return results
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Alias for retrieve() for backward compatibility"""
        return self.retrieve(query, top_k)

    def get_dim(self) -> int:
        """Get embedding dimension"""
        if self.use_openai:
            if "text-embedding-3-small" in self.embed_model:
                return 1536
            if "text-embedding-3-large" in self.embed_model:
                return 3072
            # Fallback probe
            resp = self.client.embeddings.create(model=self.embed_model, input="probe")
            return len(resp.data[0].embedding)
        else:
            v = self.st_model.encode(["probe"], normalize_embeddings=False)[0]
            return len(v)

    @staticmethod
    def _normalize(vecs: List[List[float]]) -> List[List[float]]:
        """L2-normalize vectors for cosine similarity"""
        import numpy as np
        
        nvecs: List[List[float]] = []
        for v in vecs:
            arr = np.asarray(v, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm == 0:
                nvecs.append(arr.tolist())
            else:
                nvecs.append((arr / norm).tolist())
        return nvecs