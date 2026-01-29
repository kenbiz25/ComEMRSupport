import os
import tempfile
from core.memory.sqlite_vector_store import SQLiteVectorStore
from core.memory.vector_memory import _deterministic_embedding


def test_sqlite_add_get(tmp_path):
    db_path = str(tmp_path / "vectors.db")
    key = None
    # If cryptography available, generate a key for encryption test
    try:
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
    except Exception:
        key = None

    store = SQLiteVectorStore(path=db_path, encryption_key=key)
    emb = _deterministic_embedding("fever and cough", dim=64)
    store.add_entry("sess-x", "Patient: fever and cough", embedding=emb, role="user")

    results = store.get_similar("sess-x", emb, top_k=1)
    assert len(results) == 1
    assert "fever" in results[0]["text"].lower()
