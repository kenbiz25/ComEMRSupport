
# core/indexing/pipeline.py
import os
import pathlib
from typing import List, Iterable, Dict, Any

import numpy as np
from openai import OpenAI

from config.settings import settings
from core.knowledge.store_faiss import FaissStore

# Optional readers (install: python-docx, PyPDF2, python-pptx)
try:
    from docx import Document  # python-docx
except Exception:
    Document = None

try:
    from PyPDF2 import PdfReader  # PyPDF2
except Exception:
    PdfReader = None

try:
    from pptx import Presentation  # python-pptx
except Exception:
    Presentation = None


# ------------ helpers ---------------------------------------------------------

def _is_openai_embed(model_name: str) -> bool:
    return str(model_name).startswith("text-embedding-")

def _embed_dim(model_name: str) -> int:
    if model_name == "text-embedding-3-small":
        return 1536
    if model_name == "text-embedding-3-large":
        return 3072
    # Fallback for other providers/models if ever used
    return 1536

def _l2_normalize(vecs: List[List[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for v in vecs:
        arr = np.asarray(v, dtype=np.float32)
        norm = np.linalg.norm(arr)
        out.append(arr.tolist() if norm == 0 else (arr / norm).tolist())
    return out

def _read_docx(path: pathlib.Path) -> str:
    if Document is None:
        return ""
    try:
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text).strip()
    except Exception:
        return ""

def _read_pdf(path: pathlib.Path) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(str(path))
        buf = []
        for page in getattr(reader, "pages", []):
            t = page.extract_text() or ""
            if t:
                buf.append(t)
        return "\n".join(buf).strip()
    except Exception:
        return ""

def _read_pptx(path: pathlib.Path) -> str:
    if Presentation is None:
        return ""
    try:
        prs = Presentation(str(path))
        buf = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    buf.append(shape.text)
        return "\n".join(buf).strip()
    except Exception:
        return ""


# ------------ pipeline --------------------------------------------------------

class IndexingPipeline:
    """
    Reindexes KB content using OpenAI embeddings and stores into FAISS.
    - Reads config from `settings`.
    - Supports .docx, .pdf, .pptx (skips Office temp files "~$...").
    - Batches embedding requests for throughput.
    """

    SUPPORTED_EXTS = {".docx", ".pdf", ".pptx"}

    def __init__(self, batch_size: int = 128, kb_dir: str | None = None, namespace: str | None = None):
        self.model_name = settings.EMBED_MODEL
        self.use_openai = _is_openai_embed(self.model_name)
        self.dim = _embed_dim(self.model_name)
        self.batch_size = batch_size

        # KB path and namespace (from settings)
        self.kb_dir = pathlib.Path(kb_dir or settings.KB_DIR)
        self.namespace = namespace or settings.KB_NAMESPACE

        # Vector store (FAISS)
        # If your FaissStore supports namespaces, you can pass it here.
        self.store = FaissStore(dim=self.dim)

        # OpenAI client (required for text-embedding-3-*)
        if self.use_openai:
            if not settings.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set in environment/.env")
            self.oa = OpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            raise RuntimeError(
                f"EMBED_MODEL='{self.model_name}' is not an OpenAI embedding model. "
                "Set EMBED_MODEL to 'text-embedding-3-small' or 'text-embedding-3-large'."
            )

    # --- text loading & chunking ------------------------------------------------

    def _chunk(self, text: str, size: int = 1800, overlap: int = 200) -> List[str]:
        """
        Character-based sliding window chunker.
        Increase `size` if your docs are short; keep overlap to preserve context.
        """
        if not text:
            return []
        chunks: List[str] = []
        i, n = 0, len(text)
        step = max(size - overlap, 1)
        while i < n:
            chunks.append(text[i: i + size])
            i += step
        return chunks

    def _extract_text(self, path: pathlib.Path) -> str:
        ext = path.suffix.lower()
        if ext == ".docx":
            return _read_docx(path)
        if ext == ".pdf":
            return _read_pdf(path)
        if ext == ".pptx":
            return _read_pptx(path)
        return ""

    def _iter_kb_files(self, root: pathlib.Path) -> Iterable[pathlib.Path]:
        """
        Yields supported KB files. Skips Office temporary files (~$...).
        """
        if not root.exists():
            print(f"[Indexing] KB dir not found: {root}")
            return []
        for p in root.glob("**/*"):
            if not p.is_file():
                continue
            name = p.name
            if name.startswith("~$"):  # skip Office temp/lock files
                continue
            if p.suffix.lower() in self.SUPPORTED_EXTS:
                yield p

    # --- embedding --------------------------------------------------------------

    def _embed_batch_openai(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of texts with OpenAI and returns L2-normalized vectors."""
        resp = self.oa.embeddings.create(model=self.model_name, input=texts)
        vecs = [d.embedding for d in resp.data]
        return _l2_normalize(vecs)

    def _embed_in_batches(self, texts: List[str], batch_size: int) -> List[List[float]]:
        all_vecs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            vecs = self._embed_batch_openai(batch)
            all_vecs.extend(vecs)
        return all_vecs

    # --- storage ----------------------------------------------------------------

    def _store_embeddings(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Upserts each chunk + vector into FAISS. Replace with pgvector logic if needed.
        Stores useful metadata for later citations.
        """
        for ch, vec in zip(chunks, embeddings):
            meta = {
                "title": ch["title"],
                "source_path": ch["source_path"],
                "chunk_id": ch["chunk_id"],
                "namespace": self.namespace,
            }
            # store.upsert(text, meta, vector)
            self.store.upsert(ch["text"], meta, vec)

    # --- public API -------------------------------------------------------------

    def reindex_all(self) -> int:
        """
        Walks KB_DIR, parses supported files, chunks, embeds in batches, and stores.
        Returns the number of chunks processed.
        """
        root = self.kb_dir
        files = list(self._iter_kb_files(root))
        print(f"[Indexing] KB_DIR={root} | namespace={self.namespace} | files={len(files)}")
        if not files:
            return 0

        # 1) Extract & chunk
        prepared_chunks: List[Dict[str, Any]] = []
        for path in files:
            try:
                text = self._extract_text(path)
                if not text:
                    print(f"[Indexing] Empty/failed parse: {path.name}")
                    continue

                pieces = self._chunk(text, size=1800, overlap=200)
                if not pieces:
                    print(f"[Indexing] No chunks produced: {path.name}")
                    continue

                title = path.stem
                for idx, t in enumerate(pieces, start=1):
                    prepared_chunks.append({
                        "text": t,
                        "title": title,
                        "source_path": str(path),
                        "chunk_id": idx,
                    })
            except Exception as e:
                print(f"[Indexing] Parse error for {path.name}: {e}")

        if not prepared_chunks:
            print("[Indexing] Extracted 0 chunks from KB.")
            return 0

        print(f"[Indexing] Prepared {len(prepared_chunks)} chunks. Embedding with {self.model_name}...")

        # 2) Embed in batches
        texts = [c["text"] for c in prepared_chunks]
        embeddings = self._embed_in_batches(texts, self.batch_size)

        # 3) Upsert into FAISS
        self._store_embeddings(prepared_chunks, embeddings)

        # (Optional) persist index to disk if your FaissStore supports saving
        try:
            # e.g., self.store.save(f"faiss_{self.namespace}.index")
            pass
        except Exception:
            pass

        print(f"[Indexing] Completed. Total chunks upserted: {len(prepared_chunks)}")
        return len(prepared_chunks)

    # Backward compatibility: if you call the old API
    def reindex_folder(self, folder: str = "kb") -> int:
        """
        Deprecated – use reindex_all(). Kept for backward compatibility.
        """
        # Use provided folder instead of settings if explicitly given
        self.kb_dir = pathlib.Path(folder)
        return self.reindex_all()
