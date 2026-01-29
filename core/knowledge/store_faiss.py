# core/knowledge/store_faiss.py
try:
    import faiss
except Exception:
    faiss = None

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Iterable
import logging

logger = logging.getLogger(__name__)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _doc_uid(metadata: Dict[str, Any]) -> str:
    ns = metadata.get("namespace", "")
    doc_id = metadata.get("document_id") or metadata.get("source_path", "")
    chunk_id = str(metadata.get("chunk_id", "0"))
    return f"{ns}|{doc_id}|{chunk_id}"

class FaissStore:
    """A FAISS-backed store with an in-memory fallback when the `faiss` package is not available."""

    def __init__(self, dim: int = 1536, index_dir: str = "core/faiss_index", namespace: str = None, overfetch: int = 5):
        self.dim = dim
        self.index_dir = Path(index_dir)
        self.index_file = self.index_dir / "index.faiss"
        self.docs_file = self.index_dir / "documents.pkl"
        self.meta_file = self.index_dir / "meta.pkl"
        self.namespace_default = namespace
        self.overfetch = max(1, overfetch)

        # If FAISS is available, create an IndexIDMap2 index for efficient similarity search.
        if faiss is not None:
            self.index: faiss.Index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        else:
            logger.warning("faiss is not installed; using in-memory search fallback for FaissStore")
            self.index = None

        # Document storage & mappings (works for both backends)
        self.docs: List[Dict[str, Any]] = []
        self.uid_to_faiss_id: Dict[str, int] = {}
        self.faiss_id_to_pos: Dict[int, int] = {}
        self.next_faiss_id: int = 1

        self._load_if_exists()

    def _load_if_exists(self):
        try:
            if self.docs_file.exists():
                with open(self.docs_file, "rb") as f:
                    self.docs = pickle.load(f)
                # Rebuild mappings
                self.uid_to_faiss_id.clear()
                self.faiss_id_to_pos.clear()
                for pos, d in enumerate(self.docs):
                    faiss_id = d.get("faiss_id")
                    uid = d.get("uid")
                    if faiss_id is None or uid is None:
                        continue
                    self.faiss_id_to_pos[int(faiss_id)] = pos
                    self.uid_to_faiss_id[str(uid)] = int(faiss_id)
                # Load next id if available
                if self.meta_file.exists():
                    with open(self.meta_file, "rb") as f:
                        meta = pickle.load(f)
                        self.next_faiss_id = int(meta.get("next_faiss_id", self.next_faiss_id))
                logger.info(f"Loaded {len(self.docs)} documents (in-memory)")

                # If FAISS exists and index file present, try to load it as well
                if faiss is not None and self.index_file.exists():
                    try:
                        idx = faiss.read_index(str(self.index_file))
                        if isinstance(idx, faiss.IndexIDMap2):
                            self.index = idx
                            logger.info("Loaded FAISS index file")
                        else:
                            logger.warning("FAISS index on disk is not IndexIDMap2; using in-memory docs only")
                    except Exception as e:
                        logger.warning(f"Failed to load FAISS index file: {e}")

        except Exception as e:
            logger.warning(f"Could not load index/docs: {e}. Starting fresh.")
            self._fresh_index()

    def _fresh_index(self):
        if faiss is not None:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
        else:
            self.index = None
        self.docs = []
        self.uid_to_faiss_id = {}
        self.faiss_id_to_pos = {}
        self.next_faiss_id = 1

    def save(self):
        _ensure_dir(self.index_dir)
        # Save FAISS index only if present
        if faiss is not None and self.index is not None:
            try:
                faiss.write_index(self.index, str(self.index_file))
            except Exception as e:
                logger.warning(f"Failed to write FAISS index: {e}")
        # Always save docs and next_faiss_id
        with open(self.docs_file, "wb") as f:
            pickle.dump(self.docs, f)
        with open(self.meta_file, "wb") as f:
            pickle.dump({"next_faiss_id": self.next_faiss_id}, f)
        logger.info(f"Saved store with {len(self.docs)} docs.")

    def upsert(self, content: str, metadata: Dict[str, Any], embedding: List[float]):
        if "namespace" not in metadata and self.namespace_default:
            metadata["namespace"] = self.namespace_default

        uid = _doc_uid(metadata)

        # Remove old doc if exists
        if uid in self.uid_to_faiss_id:
            old_id = self.uid_to_faiss_id[uid]
            self._remove_faiss_ids([old_id])

        # Assign new FAISS id
        faiss_id = self.next_faiss_id
        self.next_faiss_id += 1

        # Backend-specific storage
        if faiss is not None and self.index is not None:
            vec = np.asarray([embedding], dtype="float32")
            ids = np.asarray([faiss_id], dtype="int64")
            self.index.add_with_ids(vec, ids)
            stored_embedding = None
        else:
            # Store normalized embedding for in-memory similarity
            arr = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(arr)
            stored_embedding = (arr / norm).tolist() if norm else arr.tolist()

        # Record metadata
        doc_record = {
            "uid": uid,
            "faiss_id": faiss_id,
            "text": content,
            "metadata": metadata,
        }
        if stored_embedding is not None:
            doc_record["embedding"] = stored_embedding

        self.docs.append(doc_record)
        self.uid_to_faiss_id[uid] = faiss_id
        self.faiss_id_to_pos[faiss_id] = len(self.docs) - 1

        logger.info(f"Upserted document {uid} with FAISS id {faiss_id}. Total docs: {len(self.docs)}")

    def _remove_faiss_ids(self, ids: Iterable[int]) -> int:
        to_remove = [i for i in ids if i in self.faiss_id_to_pos]
        if not to_remove:
            return 0

        # Backend-specific removal
        if faiss is not None and self.index is not None:
            try:
                self.index.remove_ids(np.array(to_remove, dtype="int64"))
            except Exception as e:
                logger.warning(f"Failed removing ids from FAISS index: {e}")

        # Mark deleted in docs and remove mappings
        for fid in to_remove:
            pos = self.faiss_id_to_pos.pop(fid, None)
            if pos is not None:
                self.docs[pos] = {
                    "uid": f"deleted::{fid}",
                    "faiss_id": fid,
                    "text": "",
                    "metadata": {}
                }

        for u, fid in list(self.uid_to_faiss_id.items()):
            if fid in to_remove:
                self.uid_to_faiss_id.pop(u, None)

        logger.info(f"Removed {len(to_remove)} FAISS ids: {to_remove}. Total docs: {len(self.docs)}")
        return len(to_remove)

    def search(self, query_emb: List[float], top_k: int = 3, namespace: str = None, overfetch: int = None) -> List[Dict[str, Any]]:
        # Early return if no docs
        if not self.docs:
            return []

        ns = namespace or self.namespace_default
        fetch_count = overfetch or self.overfetch
        fetch = min(top_k * fetch_count, len(self.docs))

        results = []

        if faiss is not None and self.index is not None:
            vec = np.asarray([query_emb], dtype="float32")
            scores, idxs = self.index.search(vec, fetch)
            for score, idx in zip(scores[0], idxs[0]):
                if idx == -1:
                    continue
                pos = self.faiss_id_to_pos.get(idx)
                if pos is None:
                    continue
                doc = self.docs[pos]
                if ns and doc.get("namespace") != ns:
                    continue
                results.append({
                    "text": doc.get("text", ""),
                    "score": float(max(0, min(1, score))),
                    "metadata": doc.get("metadata", {}),
                })
                if len(results) >= top_k:
                    break
        else:
            # In-memory similarity (dot product of normalized vectors)
            q = np.asarray(query_emb, dtype=np.float32)
            qn = np.linalg.norm(q)
            if qn:
                q = q / qn
            scored = []
            for d in self.docs:
                if ns and d.get("metadata", {}).get("namespace") != ns:
                    continue
                emb = d.get("embedding")
                if not emb:
                    continue
                s = float(np.dot(q, np.asarray(emb, dtype=np.float32)))
                scored.append((s, d))
            scored.sort(key=lambda x: x[0], reverse=True)
            for s, d in scored[:fetch]:
                results.append({
                    "text": d.get("text", ""),
                    "score": float(max(0, min(1, s))),
                    "metadata": d.get("metadata", {}),
                })

        logger.info(f"Search returned {len(results)} results for top_k={top_k}, namespace={ns}")
        return results

    def size(self) -> int:
        """Return the number of active (non-deleted) documents."""
        return len([d for d in self.docs if not d["uid"].startswith("deleted::")])
