from typing import Optional, List, Dict, Any, Tuple
import os
import pathlib
import re
import logging

from config.settings import settings
from .retriever import Retriever

# Prefer importing the client module lazily at call time to allow tests to monkeypatch it
import adapters.llm.openai_client as openai_client  # uses chat_complete via openai_client.chat_complete

logger = logging.getLogger(__name__)

# Optional fallback readers
try:
    from docx import Document
except Exception:
    Document = None

# Prefer pypdf (recommended) and fallback to PyPDF2 if installed
try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception:
        PdfReader = None

# Optional media processing
try:
    import openai  # NOTE: This may not be used if you rely on openai_client
except Exception:
    openai = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import pytesseract
except Exception:
    pytesseract = None


# -------- System Prompt --------
def _load_system_prompt() -> str:
    brand = os.getenv("COMEMR_BRAND_NAME", getattr(settings, "COMEMR_BRAND_NAME", "ComEMR Support"))
    try:
        base = pathlib.Path("prompts") / "system.txt"
        if base.exists():
            return base.read_text(encoding="utf-8").strip()
    except Exception:
        pass

    return (
        f"You are {brand}. Provide accurate, concise answers in plain language. Be empathetic and helpful.\n"
        "- Keep answers short and crisp: prefer max 2-3 sentences or numbered steps for procedures.\n"
        "- Use friendly, empathetic tone suitable for community health workers and clinicians.\n"
        "- Prioritize knowledge from the internal KB. If the KB does not contain a confident answer, say you do not know rather than guessing; offer to provide general guidance clearly labeled as such.\n"
        "- When unsure, ask 1 concise clarifying question instead of hallucinating.\n"
        "- Do NOT mention internal system details (indexes, file paths) or include raw scores in replies.\n"
        "- Never reveal secrets, tokens, passwords, or private configuration.\n"
    )


# -------- Guardrails --------
def _default_guardrails(user_query: str, enabled: bool) -> Tuple[bool, str]:
    if not enabled:
        return False, ""
    q = (user_query or "").lower()
    sensitive_terms = ("api key", "token", "private key", "secret", "credential")
    reveal_verbs = ("share", "reveal", "show", "give", "expose", "send", "post")
    for st in sensitive_terms:
        if st in q and any(v in q for v in reveal_verbs):
            return True, "For security, I can’t assist with requests that expose credentials or secrets."
    if "password" in q and any(v in q for v in reveal_verbs):
        return True, "For safety and privacy, I can’t help reveal or transmit passwords. I can provide safe reset steps."
    return False, ""


# -------- Context utils --------
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
            # small overflow allowance for short chunks
            if ln < 600 and total + ln <= max_chars + 600:
                kept.append(ch)
            break
    return kept


def _format_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Keep citations metadata-only; do not expose scores in UI.
    cites = []
    for i, ch in enumerate(chunks, start=1):
        cites.append({
            "id": ch.get("id") or f"chunk-{i}",
            "title": ch.get("title"),
            "score": ch.get("score"),  # kept for debug/meta; UI sanitizer strips score patterns
        })
    return cites


# -------- Media helpers --------
def _audio_to_text(audio_path: str) -> str:
    # Best-effort transcription (optional)
    if openai:
        try:
            with open(audio_path, "rb") as f:
                transcription = openai.Audio.transcriptions.create(file=f, model="whisper-1")
            return transcription.text.strip()
        except Exception:
            pass

    if sr:
        try:
            r = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = r.record(source)
            return r.recognize_google(audio_data)
        except Exception:
            pass

    return f"[Unable to transcribe audio: {audio_path}]"


def _image_to_text(image_path: str) -> str:
    # OCR first
    if pytesseract and Image:
        try:
            img = Image.open(image_path)
            return pytesseract.image_to_string(img).strip()
        except Exception:
            pass

    return f"[Unable to extract text from image: {image_path}]"


# -------- Micro-Templates for common procedures --------
MICRO_TEMPLATES = {
    "reset password": [
        "1. Open the ComEMR app and tap 'Forgot Password'.",
        "2. Enter your registered phone number and submit.",
        "3. You will receive an SMS with a secure link.",
        "4. Open the link and set a new password.",
        "5. Create a strong password (e.g., Jam332) and tap Submit.",
        "6. Login again using your new password."
    ],
}


def _match_micro_template(query: str) -> Optional[List[str]]:
    q = (query or "").lower()
    for key in MICRO_TEMPLATES.keys():
        if key in q:
            return MICRO_TEMPLATES[key]
    return None


# -------- Intent Detection --------
# Important: Use grouping to avoid regex precedence bugs.
_GREETING_RE = re.compile(r"\b(hello|hi|hey|good morning|good afternoon|good evening)\b", re.IGNORECASE)
_PROCEDURAL_RE = re.compile(r"\b(how to|steps?|procedure|guide|reset|install|update)\b", re.IGNORECASE)
_META_RE = re.compile(r"\b(are you a bot|are you a robot|can i ask you|outside comemr|other things)\b", re.IGNORECASE)
_FAQ_RE = re.compile(r"\b(what|who|when|where|why)\b", re.IGNORECASE)


def _detect_intent(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "general"
    if _GREETING_RE.search(q):
        return "greeting"
    if _PROCEDURAL_RE.search(q):
        return "procedural"
    if _META_RE.search(q):
        return "meta"
    if _FAQ_RE.search(q):
        return "faq"
    return "general"


# -------- Output Sanitizer --------
_PATH_PATTERN = re.compile(r"([A-Za-z]:\\[^ \n]+|\/[^ \n]+)", re.IGNORECASE)
_BRACKET_CITE_PATTERN = re.compile(r"\[\s*\d+\s*\]")
_INTERNAL_CONF_PATTERN = re.compile(r"(?i)i'm not fully confident.*?(?:\n|$)")
_SCORE_PATTERN = re.compile(r"(?i)\bscore\s*[:=]\s*\d+(\.\d+)?\b")


def _sanitize_text_ui(text: str, brand: str) -> str:
    if not text:
        return ""
    text = _BRACKET_CITE_PATTERN.sub("", text)
    text = _PATH_PATTERN.sub("", text)
    text = _INTERNAL_CONF_PATTERN.sub("", text)
    text = _SCORE_PATTERN.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# -------- Composer --------
class RagComposer:
    def __init__(
        self,
        llm_model: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        safeguard: bool = True,
        top_k: Optional[int] = None,
        max_context_chars: Optional[int] = None,
    ):
        self.llm_model = llm_model or getattr(settings, "OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o"))
        self.top_k = top_k or getattr(settings, "TOP_K", 3)
        self.confidence_threshold = confidence_threshold or getattr(settings, "ANSWER_CONFIDENCE_THRESHOLD", 0.45)
        self.max_context_chars = max_context_chars or 6000
        self.safeguard = safeguard
        self.retriever = Retriever(top_k=self.top_k)
        self.system_prompt = _load_system_prompt()
        self.brand = os.getenv("COMEMR_BRAND_NAME", "ComEMR Support")

        # If true, do NOT return a "reset-like" greeting response.
        self.suppress_hard_greeting = True

    def answer(
        self,
        query: Optional[str] = None,
        audio_path: Optional[str] = None,
        image_path: Optional[str] = None,
        language: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Produce an answer and return meta including confidence, citations, and detected intent.

        language: 'en' (default) or 'krio' to indicate user requested Krio mixing.
        session_id: optional session identifier used to fetch conversation memory summaries.
        """
        query = (query or "").strip()

        # 1️⃣ Handle audio/image
        if audio_path:
            query += f"\n[Audio transcription]: {_audio_to_text(audio_path)}"
        if image_path:
            query += f"\n[Image text]: {_image_to_text(image_path)}"

        # 1b️⃣ Conversation memory hint (best-effort)
        memory_summary = ""
        try:
            if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and session_id:
                try:
                    from core.memory.memory_service import ConversationMemory
                    mem = ConversationMemory()
                    memory_summary = mem.summarize(session_id) or ""
                    if memory_summary:
                        query = f"[Conversation summary]: {memory_summary}\n\n{query}"
                except Exception:
                    pass
        except Exception:
            pass

        # 2️⃣ Guardrails
        blocked, msg = _default_guardrails(query, self.safeguard)
        if blocked:
            return msg, {"confidence": 0.0, "citations": [], "guarded": True, "intent": None, "low_confidence": False}

        # 3️⃣ Micro-template check
        micro_steps = _match_micro_template(query)
        if micro_steps:
            answer_text = "\n".join(micro_steps)
            return answer_text, {
                "confidence": 1.0,
                "citations": [],
                "guarded": False,
                "intent": _detect_intent(query),
                "low_confidence": False
            }

        # 4️⃣ Intent detection
        intent = _detect_intent(query)

        # ✅ Fix: Greeting should NOT cause "reset"
        if intent == "greeting":
            if self.suppress_hard_greeting:
                # Forward-moving prompt (WhatsApp-friendly)
                msg = f"Hi! How can I help you with ComEMR today? (You can type 'menu' to see options.)"
                return msg, {"confidence": 1.0, "citations": [], "guarded": False, "intent": intent, "low_confidence": False}
            else:
                return f"Hello! I’m {self.brand}. How can I assist you today?", {
                    "confidence": 1.0, "citations": [], "guarded": False, "intent": intent, "low_confidence": False
                }

        # Meta intent: short direct answers
        if intent == "meta":
            meta_ans = (
                f"I’m a virtual assistant for {self.brand}. I can help with ComEMR support — troubleshooting, how-tos, and guidance. "
                "Tell me what you’re trying to do or what error you see."
            )
            return meta_ans, {"confidence": 1.0, "citations": [], "guarded": False, "intent": intent, "low_confidence": False}

        # 5️⃣ KB retrieval
        try:
            chunks = self.retriever.retrieve(query, top_k=self.top_k)
        except Exception:
            chunks = []

        # 6️⃣ Fallback to filesystem KB
        if not chunks:
            try:
                from .composer_fallback import _fs_fallback_chunks
                chunks = _fs_fallback_chunks(query, self.top_k)
            except Exception:
                chunks = []

        # 7️⃣ Filter by confidence
        filtered = [c for c in chunks if float(c.get("score", 0.0)) >= self.confidence_threshold]
        context_chunks = _truncate_context(filtered or chunks, self.max_context_chars)
        citations = _format_citations(context_chunks)

        # 8️⃣ Compose answer via LLM
        low_confidence = not bool(filtered) and not bool(context_chunks)

        # If absolutely no context and query is short/ambiguous, guide instead of hallucinating.
        if low_confidence and len(query) < 10:
            return (
                "I can help. Please describe the issue in one sentence (e.g., login error, sync failing, device problem), or type 'menu'.",
                {"confidence": 0.0, "citations": [], "guarded": False, "intent": intent, "low_confidence": True}
            )

        answer_text = self._compose_with_llm(
            query,
            context_chunks,
            low_confidence=low_confidence,
            language=language,
            session_id=session_id
        )

        clean_answer = _sanitize_text_ui(answer_text, self.brand)
        max_conf = max([float(c.get("score", 0.0)) for c in context_chunks], default=0.0)
        return clean_answer, {
            "confidence": max_conf,
            "citations": citations,
            "guarded": False,
            "intent": intent,
            "low_confidence": low_confidence
        }

    def _compose_with_llm(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        low_confidence: bool,
        language: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        context_text = "\n\n".join([c.get("text", "") for c in context_chunks if c.get("text")])

        # Retrieve vector memory items if enabled and session_id provided
        memory_texts = []
        try:
            if session_id and getattr(settings, "ENABLE_VECTOR_MEMORY", False):
                try:
                    from core.memory.vector_memory import VectorMemory
                    vec = VectorMemory()
                    mem_items = vec.get_similar(session_id, query, top_k=getattr(settings, "MEMORY_VECTOR_TOP_K", 3))
                    if mem_items:
                        memory_texts = [f"{m.get('role','')}: {m.get('text','')}" for m in mem_items if m.get("text")]
                except Exception:
                    pass
        except Exception:
            pass

        memory_section = ("Relevant memory:\n" + "\n".join(memory_texts) + "\n\n") if memory_texts else ""

        # Krio mixing instruction
        lang_instruction = ""
        if language == "krio":
            lang_instruction = (
                "\nNote: The user requested Krio. Reply mostly in English but include Krio so that roughly 40% of the reply is in Krio (by sentence count) and 60% in English. "
                "Keep critical guidance primarily in English to ensure clarity; use Krio for friendly phrasing or short clarifying sentences."
            )

        prompt = f"""
User question:
{query}

{memory_section}Relevant context:
{context_text or '[No relevant context found]'}

Instructions:
- Answer clearly and concisely.
- Use natural, conversational language.
- For short direct answers (yes/no or single-line replies), do NOT use numbered lists; use a single plain sentence.
- Provide plain numbered steps only for procedural answers with multiple steps.
- Be tolerant of minor typos and infer user intent; ask a clarifying question only if intent is unclear.
- Avoid internal paths, confidential info, or private data.
- If unsure, suggest safe next steps.
{lang_instruction}
""".strip()

        if low_confidence:
            prompt += (
                "\n\nNote: KB confidence is low; do NOT hallucinate. "
                "If you cannot answer confidently, say you don't know and ask ONE concise clarifying question (1 sentence). "
                "You may provide brief 'General guidance' if helpful."
            )

        try:
            answer = openai_client.chat_complete(
                prompt,
                model=self.llm_model,
                temperature=getattr(settings, "LLM_TEMPERATURE", 0.2),
                max_tokens=getattr(settings, "LLM_MAX_TOKENS", 450),
                system_prompt=self.system_prompt,
            )
            return (answer or "").strip()
        except Exception:
            logger.exception("LLM chat completion failed")
            return "Sorry, an internal error occurred while generating the answer. Please try again later."