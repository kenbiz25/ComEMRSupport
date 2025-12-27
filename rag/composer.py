
from typing import Optional, List, Dict, Any, Tuple
import os
import pathlib
import re

from config.settings import settings
from .retriever import Retriever
from adapters.llm.openai_client import chat_complete  # ✅ Your wrapper

# Optional fallback readers
try:
    from docx import Document  # python-docx
except Exception:
    Document = None

try:
    from PyPDF2 import PdfReader  # pypdf2
except Exception:
    PdfReader = None


# -------- Prompts --------
def _load_system_prompt() -> str:
    """
    Brand-safe system prompt: concise answers, no internal references.
    """
    brand = os.getenv("COMEMR_BRAND_NAME", getattr(settings, "COMEMR_BRAND_NAME", "ComEMR Support"))
    try:
        base = pathlib.Path("prompts") / "system.txt"
        if base.exists():
            return base.read_text(encoding="utf-8").strip()
    except Exception:
        pass

    return (
        f"You are {brand}. Provide accurate, concise answers in plain language.\n"
        "- Do NOT mention internal systems (KB, indexes, vector DB, files, paths) or scores.\n"
        "- Do NOT include citations inline. If context is insufficient, say you don't know and "
        "suggest safe next steps.\n"
        "- Prefer numbered steps for procedures. Keep patient privacy and data security.\n"
        "- Avoid clinical diagnosis/treatment instructions; direct users to local protocols or "
        "licensed clinicians when needed.\n"
        "- Never reveal secrets, tokens, passwords, or private configuration.\n"
    )


# -------- Guardrails --------
def _default_guardrails(user_query: str, enabled: bool) -> Tuple[bool, str]:
    if not enabled:
        return False, ""

    q = (user_query or "").lower()

    # Only block attempts to REVEAL or SHARE secrets; do not block normal support questions
    sensitive_terms = ("api key", "token", "private key", "secret", "credential")
    reveal_verbs = ("share", "reveal", "show", "give", "expose", "send", "post")

    for st in sensitive_terms:
        if st in q:
            for v in reveal_verbs:
                if v in q:
                    return True, (
                        "For security, I can’t assist with requests that expose credentials or secrets. "
                        "Please share non‑sensitive details or ask for official guidance."
                    )

    # Block revealing passwords
    if "password" in q and any(v in q for v in reveal_verbs):
        return True, (
            "For safety and privacy, I can’t help reveal or transmit passwords. "
            "I can provide official steps to reset or change a password."
        )

    # Soft guard for clinical treatment/diagnosis — don’t block, but redirect safely
    clinical_terms = ("treat", "diagnose", "dose", "prescribe", "medication", "convuls", "seizure")
    if any(t in q for t in clinical_terms):
        return False, ""  # allow, but the prompt instructs safe redirection

    return False, ""


# -------- Context utilities --------
def _truncate_context(chunks: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    total = 0
    kept = []
    for ch in chunks:
        txt = ch.get("text") or ch.get("chunk_text") or ""
        ln = len(txt)
        if ln == 0:
            continue
        if total + ln <= max_chars:
            kept.append(ch)
            total += ln
        else:
            # allow small overflow to keep last whole chunk if it’s short
            if ln < 600 and total + ln <= max_chars + 600:
                kept.append(ch)
            break
    return kept


def _format_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cites = []
    for i, ch in enumerate(chunks, start=1):
        cites.append({
            "id": ch.get("id") or ch.get("chunk_id") or f"chunk-{i}",
            "title": ch.get("title"),
            "source_path": ch.get("source_path"),
            "chunk_id": ch.get("chunk_id", i),
            "score": ch.get("score"),
            "rank": ch.get("rank", i),
        })
    return cites


# -------- File-system fallback (KB folder scan: .docx, .pdf) --------
KB_DIR = os.getenv("KB_DIR", "KB")  # set to your KB folder name (e.g., 'KB' or 'kb')

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
        text = []
        for page in getattr(reader, "pages", []):
            t = page.extract_text() or ""
            if t:
                text.append(t)
        return "\n".join(text).strip()
    except Exception:
        return ""

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", (s or "").lower())

def _score_text(query: str, text: str) -> float:
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return 0.0
    t_tokens = _tokenize(text)
    if not t_tokens:
        return 0.0
    hit = sum(1 for t in q_tokens if t in t_tokens)
    return round(hit / max(1, len(q_tokens)), 4)

def _chunk_text(text: str, chunk_chars: int = 1600) -> List[str]:
    # Slightly smaller chunks for speed
    if not text:
        return []
    blocks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        blocks.append(text[start:end])
        start = end
    return blocks

def _fs_fallback_chunks(query: str, top_k: int) -> List[Dict[str, Any]]:
    root = pathlib.Path(KB_DIR)
    if not root.exists():
        return []

    candidates: List[Dict[str, Any]] = []
    for path in root.glob("**/*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in {".docx", ".pdf"}:
            continue

        text = _read_docx(path) if ext == ".docx" else _read_pdf(path)
        if not text:
            continue

        chunks = _chunk_text(text)
        title = path.stem
        for idx, ch in enumerate(chunks, start=1):
            score = _score_text(query, ch)
            candidates.append({
                "id": f"{path.name}#{idx}",
                "title": title,
                "source_path": str(path),
                "chunk_id": idx,
                "text": ch,
                "score": score,
            })

    candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    non_zero = [c for c in candidates if c.get("score", 0.0) > 0]
    return (non_zero or candidates)[:max(1, top_k)]


# -------- Output sanitizers (hide KB details from UI) --------
_PATH_PATTERN = re.compile(r"([A-Za-z]:\\[^ \n]+|\/[^ \n]+)", re.IGNORECASE)
_BRACKET_CITE_PATTERN = re.compile(r"\[\s*\d+\s*\]")
_INTERNAL_CONF_PATTERN = re.compile(r"(?i)i'm not fully confident.*?(?:\n|$)")

def _sanitize_text_ui(text: str, brand: str) -> str:
    if not text:
        return ""

    # Remove citations like [1], [2]
    text = _BRACKET_CITE_PATTERN.sub("", text)

    # Remove any direct file paths or drive hints
    text = _PATH_PATTERN.sub("", text)

    # Remove noisy internal confidence boilerplate
    text = _INTERNAL_CONF_PATTERN.sub("", text)

    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Optional: enforce short branded header for consistency
    # Avoid adding if the message already starts with a greeting/instruction
    if not re.match(r"^(hi|hello|thank|please|to |go to|step|1\.)", text.strip(), re.IGNORECASE):
        text = f"{text}"

    return text


# -------- Composer --------
class RagComposer:
    """
    Orchestrates retrieval + generation using OpenAI via adapters.llm.openai_client.
    Returns (answer_text, meta) where meta = {"confidence": float, "citations": List[Dict], "guarded": bool}
    """

    def __init__(
        self,
        llm_model: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        safeguard: bool = True,
        top_k: Optional[int] = None,
        max_context_chars: Optional[int] = None,
    ):
        self.llm_model = llm_model or getattr(settings, "OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        self.top_k = int(top_k or getattr(settings, "TOP_K", int(os.getenv("TOP_K", "3"))))  # faster
        self.confidence_threshold = float(
            confidence_threshold if confidence_threshold is not None
            else getattr(settings, "ANSWER_CONFIDENCE_THRESHOLD", float(os.getenv("ANSWER_CONFIDENCE_THRESHOLD", "0.45")))
        )
        self.max_context_chars = int(
            max_context_chars if max_context_chars is not None
            else getattr(settings, "MAX_CONTEXT_CHARS", int(os.getenv("MAX_CONTEXT_CHARS", "6000")))
        )
        self.safeguard = bool(safeguard if safeguard is not None else getattr(settings, "SAFEGUARD_ENABLE", True))

        self.retriever = Retriever()
        self.system_prompt = _load_system_prompt()
        self.brand = os.getenv("COMEMR_BRAND_NAME", getattr(settings, "COMEMR_BRAND_NAME", "ComEMR Support"))

    def answer(self, query: str) -> Tuple[str, Dict[str, Any]]:
        result = self.compose_answer(query)
        # 🔒 sanitize UI output to hide KB details
        clean = _sanitize_text_ui(result.get("answer", ""), self.brand)
        return clean, {
            "confidence": result.get("confidence", 0.0),
            "citations": result.get("citations", []),
            "guarded": result.get("guarded", False),
        }

    def compose_answer(self, query: str) -> Dict[str, Any]:
        blocked, msg = _default_guardrails(query, self.safeguard)
        if blocked:
            return {
                "answer": msg,
                "confidence": 0.0,
                "citations": [],
                "guarded": True,
            }

        chunks: List[Dict[str, Any]] = []
        # ---------- Primary: vector retriever ----------
        try:
            chunks = self.retriever.retrieve(query, top_k=self.top_k)
        except AttributeError:
            try:
                chunks = self.retriever.search(query, top_k=self.top_k)
            except Exception:
                chunks = []
        except Exception:
            chunks = []

        # ---------- Fallback: file-system KB scan ----------
        if not chunks:
            try:
                chunks = _fs_fallback_chunks(query, top_k=self.top_k)
            except Exception:
                chunks = []

        # Filter by confidence threshold
        filtered = [c for c in (chunks or []) if float(c.get("score", 0.0)) >= self.confidence_threshold]

        if not filtered:
            fallback = (chunks or [])[:max(1, self.top_k)]
            context_chunks = _truncate_context(fallback, self.max_context_chars)
            citations = _format_citations(context_chunks)
            answer_text = self._compose_with_llm(query, context_chunks, low_confidence=True)
            max_conf = max([float(c.get("score", 0.0)) for c in fallback], default=0.0)
            return {
                "answer": answer_text,
                "confidence": max_conf,
                "citations": citations,
                "guarded": False,
            }

        context_chunks = _truncate_context(filtered, self.max_context_chars)
        citations = _format_citations(context_chunks)
        answer_text = self._compose_with_llm(query, context_chunks, low_confidence=False)
        agg_conf = max([float(c.get("score", 0.0)) for c in context_chunks], default=0.0)
        return {
            "answer": answer_text,
            "confidence": agg_conf,
            "citations": citations,
            "guarded": False,
        }

    def _compose_with_llm(self, user_query: str, context_chunks: List[Dict[str, Any]], low_confidence: bool) -> str:
        # Minimal context formatting (no titles/scores) to avoid KB leakage and keep prompts lean
        blocks = []
        for i, ch in enumerate(context_chunks, start=1):
            text = (ch.get("text") or ch.get("chunk_text") or "").strip()
            if text:
                blocks.append(f"Context[{i}]:\n{text}\n")
        context_str = "\n".join(blocks).strip() if blocks else ""

        prompt = (
            f"{self.system_prompt}\n\n"
            f"User question:\n{(user_query or '').strip()}\n\n"
            f"Relevant context passages:\n{context_str if context_str else '[No relevant context found]'}\n\n"
            "Instructions:\n"
            "- Answer ONLY using the context; do not invent facts.\n"
            "- If context is weak/missing, say you don’t know and suggest safe next steps.\n"
            "- Use clear, numbered steps for procedures.\n"
            "- Do NOT include citations inline or mention internal sources.\n"
        )
        if low_confidence:
            prompt += (
                "\nNote: Confidence is low; prioritize caution and avoid firm statements not directly supported by context.\n"
            )

        try:
            return chat_complete(prompt).strip()
        except Exception as e:
            # Return raw context but still sanitized later; avoid exposing file paths or internals
            safe_context = re.sub(_PATH_PATTERN, "", context_str)
            return (
                "Sorry, an internal error occurred while composing the answer. "
                "Please try again.\n"
                f"\nContext considered:\n{safe_context}\n"
                f"\nError: {e}"
            )
