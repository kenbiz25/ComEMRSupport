# core/memory/vector_memory.py
from __future__ import annotations
import os
import json
import logging
import hashlib
import math
from datetime import datetime
from typing import List, Dict, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


def _cosine(a: List[float], b: List[float]) -> float:
    # pure python cosine similarity
    try:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    except Exception:
        return 0.0


def _deterministic_embedding(text: str, dim: int = 128) -> List[float]:
    # Deterministic embedding fallback: use sha256 to seed an RNG and produce floats
    try:
        seed = int(hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16], 16)
    except Exception:
        seed = abs(hash(text)) & ((1 << 63) - 1)
    import random

    rnd = random.Random(seed)
    vec = [rnd.random() - 0.5 for _ in range(dim)]
    # normalize
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class VectorMemory:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(os.getcwd(), "core", "memory", "vectors")
        os.makedirs(self.base_dir, exist_ok=True)
        # fallback dims
        self.dim = getattr(settings, "EMBEDDING_DIM", None) or 128

        # Optionally use an encrypted SQLite-backed store
        self.use_sqlite = False
        try:
            if getattr(settings, "ENABLE_ENCRYPTED_VECTOR_DB", False):
                db_path = getattr(settings, "VECTOR_DB_PATH", None)
                if db_path:
                    try:
                        from core.memory.sqlite_vector_store import SQLiteVectorStore
                        key = getattr(settings, "VECTOR_DB_ENCRYPTION_KEY", None)
                        if key:
                            # Accept raw key string; if it's base64 urlsafe, Fernet expects it
                            key_bytes = key.encode("utf-8")
                        else:
                            key_bytes = None
                        self.sqlite = SQLiteVectorStore(path=db_path, encryption_key=key_bytes)
                        self.use_sqlite = True
                    except Exception:
                        logger.exception("Failed to initialize SQLite vector store; falling back to JSONL files")
        except Exception:
            pass

    def _session_path(self, session_id: str) -> str:
        safe = str(session_id).replace('/', '_')
        return os.path.join(self.base_dir, f"{safe}.jsonl")

    def _embed(self, text: str) -> List[float]:
        # Try using OpenAI embeddings; fall back to deterministic embedding
        try:
            from adapters.llm.openai_client import get_openai
            client = get_openai()
            model = getattr(settings, "EMBED_MODEL", "text-embedding-3-small")
            resp = client.embeddings.create(model=model, input=text)
            emb = resp.data[0].embedding
            # ensure floats
            return [float(x) for x in emb]
        except Exception:
            logger.debug("OpenAI embeddings unavailable; using deterministic fallback")
            return _deterministic_embedding(text, dim=self.dim)

    def add_entry(self, session_id: str, text: str, role: str = "user", ts: Optional[str] = None):
        try:
            emb = self._embed(text)
            # If sqlite store is enabled, store there
            if getattr(self, "use_sqlite", False):
                try:
                    self.sqlite.add_entry(session_id=session_id, text=text, embedding=emb, role=role, ts=ts)
                    return
                except Exception:
                    logger.exception("SQLite add_entry failed; falling back to JSONL")
            path = self._session_path(session_id)
            entry = {"ts": ts or datetime.utcnow().isoformat() + "Z", "role": role, "text": text, "embedding": emb}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to add vector memory entry")

    def get_similar(self, session_id: str, query: str, top_k: int = 3) -> List[Dict]:
        try:
            query_emb = self._embed(query)
        except Exception:
            logger.exception("Failed to embed query for memory retrieval")
            return []

        # If sqlite store enabled, delegate there
        if getattr(self, "use_sqlite", False):
            try:
                return self.sqlite.get_similar(session_id, query_emb, top_k=top_k)
            except Exception:
                logger.exception("SQLite get_similar failed; falling back to JSONL scan")

        path = self._session_path(session_id)
        if not os.path.exists(path):
            return []

        results = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        emb = obj.get("embedding")
                        if not emb:
                            continue
                        score = _cosine(query_emb, emb)
                        results.append({"score": float(score), "text": obj.get("text"), "role": obj.get("role"), "ts": obj.get("ts")})
                    except Exception:
                        continue
        except Exception:
            logger.exception("Failed to read vector memory file")
            return []

        results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return results[:top_k]

    def clear_session(self, session_id: str) -> bool:
        try:
            path = self._session_path(session_id)
            if os.path.exists(path):
                os.remove(path)
            return True
        except Exception:
            logger.exception("Failed to clear vector session")
            return False
