
# core/knowledge/store_faiss.py
import faiss
import numpy as np
import pickle
import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Iterable
import logging

logger = logging.getLogger(__name__)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _doc_uid(metadata: Dict[str, Any]) -> str:
    """
    Stable unique ID for a chunk:
    namespace|document_id|chunk_id
    Falls back to namespace|source_path|chunk_id if document_id missing.
    """
    ns = metadata.get("namespace", "")
    doc_id = metadata.get("document_id") or metadata.get("source_path", "")
    chunk_id = str(metadata.get("chunk_id", "0"))
    return f"{ns}|{doc_id}|{chunk_id}"

class FaissStore:
    """
    FAISS-based vector store with persistence, deletions, and namespace-aware search.

    - Index: IndexIDMap2(IndexFlatIP) to support add_with_ids/remove_ids.
    - Vectors are assumed L2-normalized (so IP == cosine similarity).
    - Stores document records alongside FAISS IDs and a UID for idempotent upserts.
    """

    def __init__(
        self,
        dim: int = 1536,
        index_dir: str = "core/faiss_index",
        namespace: Optional[str] = None,
        overfetch: int = 5,   # how many extra to fetch to satisfy namespace filtering
    ):
        self.dim = dim
        self.index_dir = Path(index_dir)
        self.index_file = self.index_dir / "index.faiss"
        self.docs_file = self.index_dir / "documents.pkl"
        self.meta_file = self.index_dir / "meta.pkl"  # for counters etc.
        self.namespace_default = namespace  # used if missing in metadata
        self.overfetch = max(1, int(overfetch))

        # In-memory structures
        self.index: faiss.Index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.docs: List[Dict[str, Any]] = []  # parallel to FAISS vectors by ID (we also keep id maps)
        self.uid_to_faiss_id: Dict[str, int] = {}  # uid -> faiss_id
        self.faiss_id_to_pos: Dict[int, int] = {}  # faiss_id -> position in self.docs
        self.next_faiss_id: int = 1  # monotonically increasing int64 ids

        # Load persisted data if exists
        self._load_if_exists()

    # ---------------------------------------------------------------------
    # Persistence (load/save)
    # ---------------------------------------------------------------------

    def _load_if_exists(self):
        """Load existing FAISS index and documents if available."""
        try:
            if self.index_file.exists() and self.docs_file.exists():
                logger.info(f"Loading FAISS index from {self.index_file}")
                idx = faiss.read_index(str(self.index_file))

                # Ensure we have IDMap; if not, we keep as-is but warn that deletions won't work.
                if not isinstance(idx, faiss.IndexIDMap2):
                    logger.warning(
                        "Loaded FAISS index is not IndexIDMap2. "
                        "Deletions/upserts by UID will be disabled until you reindex fresh "
                        "(call clear() and reindex), or we wrap and rebuild IDs."
                    )
                    # We can wrap, but original IDs are lost. We *cannot* recreate IDs without embeddings.
                    # So we keep read-only for deletions.
                    self.index = idx
                    self._load_docs_only(read_only=True)
                    return

                self.index = idx
                self._load_docs_only(read_only=False)

                # Load meta (next id counter)
                if self.meta_file.exists():
                    try:
                        with open(self.meta_file, "rb") as f:
                            meta = pickle.load(f)
                            self.next_faiss_id = int(meta.get("next_faiss_id", self.next_faiss_id))
                    except Exception as e:
                        logger.warning(f"Failed to load meta: {e}")

                logger.info(f"Loaded {len(self.docs)} documents")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}. Starting fresh.")
            self._fresh_index()

    def _load_docs_only(self, read_only: bool):
        with open(self.docs_file, "rb") as f:
            self.docs = pickle.load(f)

        # Rebuild maps if possible
        self.uid_to_faiss_id.clear()
        self.faiss_id_to_pos.clear()

        if read_only:
            # We don't know FAISS vector IDs; we cannot rebuild deletion maps.
            logger.warning(
                "Operating in read-only deletion mode (legacy index without IDs). "
                "Upserts will append only; delete_* will raise until you reindex."
            )
            return

        # When saved via this class, each doc has 'faiss_id' and 'uid'
        missing = 0
        for pos, d in enumerate(self.docs):
            faiss_id = d.get("faiss_id")
            uid = d.get("uid")
            if faiss_id is None or uid is None:
                missing += 1
                continue
            self.faiss_id_to_pos[int(faiss_id)] = pos
            self.uid_to_faiss_id[str(uid)] = int(faiss_id)

        if missing:
            logger.warning(f"{missing} doc records missing (faiss_id/uid). Consider reindexing.")

    def _fresh_index(self):
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
        self.docs = []
        self.uid_to_faiss_id = {}
        self.faiss_id_to_pos = {}
        self.next_faiss_id = 1

    def save(self):
        """Persist index and documents to disk (non-atomic)."""
        try:
            _ensure_dir(self.index_dir)
            faiss.write_index(self.index, str(self.index_file))
            with open(self.docs_file, "wb") as f:
                pickle.dump(self.docs, f)
            with open(self.meta_file, "wb") as f:
                pickle.dump({"next_faiss_id": self.next_faiss_id}, f)
            logger.info(f"Saved FAISS index with {len(self.docs)} documents to {self.index_dir}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise

    def save_atomic(self):
        """Persist index and documents using temp files and atomic rename."""
        try:
            _ensure_dir(self.index_dir)

            # Save index atomically
            with tempfile.NamedTemporaryFile(delete=False, dir=str(self.index_dir), suffix=".faiss") as tf_idx:
                faiss.write_index(self.index, tf_idx.name)
                tmp_index = Path(tf_idx.name)

            # Save docs atomically
            with tempfile.NamedTemporaryFile(delete=False, dir=str(self.index_dir), suffix=".pkl") as tf_docs:
                pickle.dump(self.docs, tf_docs)
                tmp_docs = Path(tf_docs.name)

            # Save meta atomically
            with tempfile.NamedTemporaryFile(delete=False, dir=str(self.index_dir), suffix=".pkl") as tf_meta:
                pickle.dump({"next_faiss_id": self.next_faiss_id}, tf_meta)
                tmp_meta = Path(tf_meta.name)

            # Rename into place
            tmp_index.replace(self.index_file)
            tmp_docs.replace(self.docs_file)
            tmp_meta.replace(self.meta_file)

            logger.info(f"Atomically saved FAISS index with {len(self.docs)} documents to {self.index_dir}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index atomically: {e}")
            raise

    def clear(self):
        """Clear the index and documents (fresh, deletions enabled)."""
        self._fresh_index()
        logger.info("Cleared FAISS index")

    def size(self) -> int:
        """Return number of documents in index (docs list length)."""
        return len(self.docs)

    # ---------------------------------------------------------------------
    # Upsert / Delete
    # ---------------------------------------------------------------------

    def _assert_deletions_supported(self):
        if not isinstance(self.index, faiss.IndexIDMap2):
            raise RuntimeError(
                "This FAISS index does not support deletions (not an IndexIDMap2). "
                "Clear and reindex to enable deletions."
            )

    def upsert(self, content: str, metadata: Dict[str, Any], embedding: List[float]):
        """
        Add or replace a chunk by stable UID. If a record with the same UID exists, we delete it first.
        """
        # Default namespace if missing
        if "namespace" not in metadata and self.namespace_default:
            metadata["namespace"] = self.namespace_default

        uid = _doc_uid(metadata)
        self._assert_deletions_supported()

        # Remove existing record for this uid
        if uid in self.uid_to_faiss_id:
            old_id = self.uid_to_faiss_id[uid]
            self._remove_faiss_ids([old_id])

        # Assign new FAISS id
        faiss_id = int(self.next_faiss_id)
        self.next_faiss_id += 1

        # Add vector
        vec = np.asarray([embedding], dtype="float32")
        ids = np.asarray([faiss_id], dtype="int64")
        self.index.add_with_ids(vec, ids)

        # Store document record
        doc_record = {
            "uid": uid,
            "faiss_id": faiss_id,
            "text": content,
            "metadata": metadata,
            **metadata  # flatten for easy access
        }
        self.docs.append(doc_record)
        self.uid_to_faiss_id[uid] = faiss_id
        self.faiss_id_to_pos[faiss_id] = len(self.docs) - 1

    def _remove_faiss_ids(self, ids: Iterable[int]) -> int:
        """
        Remove ids from FAISS and prune docs and maps.
        Returns the number of removed items.
        """
        self._assert_deletions_supported()

        to_remove = [int(i) for i in ids if i in self.faiss_id_to_pos]
        if not to_remove:
            return 0

        id_vec = np.asarray(to_remove, dtype="int64")
        removed = self.index.remove_ids(id_vec)

        # Remove from in-memory structures.
        # Because FAISS keeps vector storage dense but we track doc records separately,
        # we'll logically mark removed docs by setting an empty record and keeping position indexes valid.
        count = 0
        for fid in to_remove:
            pos = self.faiss_id_to_pos.pop(fid, None)
            if pos is not None:
                # Keep a tombstone-like record to preserve other positions (optional)
                self.docs[pos] = {
                    "uid": f"deleted::{fid}",
                    "faiss_id": fid,
                    "text": "",
                    "metadata": {},
                }
                count += 1

        # Clear reverse map for these UIDs
        dead_uids = [u for u, fid in list(self.uid_to_faiss_id.items()) if fid in to_remove]
        for u in dead_uids:
            self.uid_to_faiss_id.pop(u, None)

        return count

    def delete_by_document_id(self, namespace: str, document_id: str) -> int:
        """
        Delete all chunks matching (namespace, document_id).
        """
        self._assert_deletions_supported()
        ids = []
        for rec in self.docs:
            if not rec or not isinstance(rec, dict):
                continue
            if rec.get("namespace") == namespace and rec.get("document_id") == document_id:
                fid = rec.get("faiss_id")
                if isinstance(fid, int):
                    ids.append(fid)
        return self._remove_faiss_ids(ids)

    def delete_by_source_path(self, namespace: str, source_path: str) -> int:
        """
        Delete all chunks matching (namespace, source_path).
        Useful when you don't track document_id externally.
        """
        self._assert_deletions_supported()
        ids = []
        for rec in self.docs:
            if not rec or not isinstance(rec, dict):
                continue
            if rec.get("namespace") == namespace and rec.get("source_path") == source_path:
                fid = rec.get("faiss_id")
                if isinstance(fid, int):
                    ids.append(fid)
        return self._remove_faiss_ids(ids)

    # ---------------------------------------------------------------------
    # Search
    # ---------------------------------------------------------------------

    def search(
        self,
        query_emb: List[float],
        top_k: int = 3,
        namespace: Optional[str] = None,
        filter_fn: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for top-k most similar documents. If namespace is provided, results
        are filtered to that namespace. If filter_fn is provided, it's applied to
        each doc record to decide inclusion.

        Returns list of dicts with keys: text, score, rank, metadata, title, source_path, chunk_id, id
        """
        if not self.size():
            logger.warning("FAISS index is empty. Run reindex first.")
            return []

        vec = np.asarray([query_emb], dtype="float32")

        # Overfetch to survive post-filtering (namespace/filter_fn)
        fetch = min(max(top_k * self.overfetch, top_k), self.size())
        scores, indices = self.index.search(vec, fetch)

        results: List[Dict[str, Any]] = []
        considered = 0
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx == -1:
                continue

            # Our docs list is position-based; we need to get the record by faiss_id -> pos
            pos = self.faiss_id_to_pos.get(idx)
            if pos is None or pos >= len(self.docs):
                continue

            doc = self.docs[pos]
            if not doc or not isinstance(doc, dict):
                continue

            # Filter by namespace (explicit or default)
            ns = namespace or self.namespace_default
            if ns:
                if doc.get("namespace") != ns:
                    continue

            if filter_fn and not filter_fn(doc):
                continue

            # Normalize IP score into [0..1], assuming vectors are normalized
            normalized_score = float(max(0.0, min(1.0, score)))

            results.append({
                "text": doc.get("text", ""),
                "chunk_text": doc.get("text", ""),
                "score": normalized_score,
                "rank": len(results) + 1,
                "metadata": doc.get("metadata", {}),
                "title": doc.get("title", ""),
                "source_path": doc.get("source_path", ""),
                "chunk_id": doc.get("chunk_id", 0),
                "id": doc.get("uid") or f"{doc.get('title', 'unknown')}#{doc.get('chunk_id', pos)}",
            })

            if len(results) >= top_k:
                break

        return results