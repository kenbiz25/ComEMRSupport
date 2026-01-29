# core/memory/sqlite_vector_store.py
from __future__ import annotations
import os
import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None


def _ensure_db(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            ts TEXT,
            role TEXT,
            text BLOB,
            embedding BLOB
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_session ON vectors(session_id)")
    conn.commit()
    return conn


class SQLiteVectorStore:
    def __init__(self, path: str, encryption_key: Optional[bytes] = None):
        self.path = path
        self.conn = _ensure_db(path)
        self.fernet = None
        if encryption_key and Fernet:
            try:
                self.fernet = Fernet(encryption_key)
            except Exception:
                logger.exception("Failed to init Fernet with provided key; operating without encryption")
                self.fernet = None
        elif encryption_key and not Fernet:
            logger.warning("Encryption key provided but cryptography.fernet not available; storing unencrypted")

    def _maybe_encrypt(self, data: bytes) -> bytes:
        if self.fernet:
            return self.fernet.encrypt(data)
        return data

    def _maybe_decrypt(self, data: bytes) -> bytes:
        if self.fernet:
            try:
                return self.fernet.decrypt(data)
            except Exception:
                logger.exception("Failed to decrypt data")
                return data
        return data

    def add_entry(self, session_id: str, text: str, embedding: List[float], role: str = "user", ts: Optional[str] = None):
        try:
            ts = ts or datetime.utcnow().isoformat() + "Z"
            emb_blob = json.dumps(embedding, ensure_ascii=False).encode("utf-8")
            text_blob = text.encode("utf-8")
            emb_blob = self._maybe_encrypt(emb_blob)
            text_blob = self._maybe_encrypt(text_blob)
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO vectors (session_id, ts, role, text, embedding) VALUES (?, ?, ?, ?, ?)",
                (session_id, ts, role, text_blob, emb_blob),
            )
            self.conn.commit()
        except Exception:
            logger.exception("Failed to add entry to sqlite vector store")

    def get_similar(self, session_id: str, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT ts, role, text, embedding FROM vectors WHERE session_id = ?", (session_id,))
            rows = cur.fetchall()
            results = []
            for ts, role, text_blob, emb_blob in rows:
                try:
                    emb_bytes = self._maybe_decrypt(emb_blob)
                    emb = json.loads(emb_bytes.decode("utf-8"))
                    # compute cosine similarity
                    score = self._cosine_similarity(query_embedding, emb)
                    text_bytes = self._maybe_decrypt(text_blob)
                    text = text_bytes.decode("utf-8")
                    results.append({"score": float(score), "text": text, "role": role, "ts": ts})
                except Exception:
                    continue
            results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
            return results[:top_k]
        except Exception:
            logger.exception("Failed to query sqlite vector store")
            return []

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        try:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(y * y for y in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)
        except Exception:
            return 0.0

    def clear_session(self, session_id: str) -> bool:
        try:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM vectors WHERE session_id = ?", (session_id,))
            self.conn.commit()
            return True
        except Exception:
            logger.exception("Failed to clear session from sqlite vector store")
            return False
