from typing import Optional, List, Dict, Any, Tuple
import os
import pathlib
import re

from config.settings import settings
from .retriever import Retriever
from adapters.llm.openai_client import chat_complete  # ✅ Your wrapper

# Optional fallback readers
try:
    from docx import Document
except Exception:
    Document = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

# Optional media processing
try:
    import openai
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
            if ln < 600 and total + ln <= max_chars + 600:
                kept.append(ch)
            break
    return kept

def _format_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cites = []
    for i, ch in enumerate(chunks, start=1):
        cites.append({
            "id": ch.get("id") or f"chunk-{i}",
            "title": ch.get("title"),
            "score": ch.get("score"),
        })
    return cites

# -------- Media helpers --------
def _audio_to_text(audio_path: str) -> str:
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
    if pytesseract and Image:
        try:
            img = Image.open(image_path)
            return pytesseract.image_to_string(img).strip()
        except Exception:
            pass
    if openai:
        try:
            with open(image_path, "rb") as img_file:
                resp = openai.Image.create(model="gpt-image-caption-001", image=img_file)
                return resp.data[0].caption if resp.data else ""
        except Exception:
            pass
    return f"[Unable to extract text from image: {image_path}]"

# -------- Micro-Templates for common procedures --------
MICRO_TEMPLATES = {
    "reset password": [
        "Step 1: Open the comEMR app and click on 'Forgot Password'.",
        "Step 2: Enter your registered phone number and submit.",
        "Step 3: You will receive an SMS with a secure link.",
        "Step 4: Click the link to open the password reset screen.",
        "Step 5: Enter your new password. Must start with an uppercase letter, followed by lowercase letters and numbers (e.g., Jam332).",
        "Step 6: Click 'Submit' to update your password. Login with your new credentials."
    ],
    # Add more templates here for account unlock, registration, etc.
}

def _match_micro_template(query: str) -> Optional[List[str]]:
    q = (query or "").lower()
    for key in MICRO_TEMPLATES.keys():
        if key in q:
            return MICRO_TEMPLATES[key]
    return None

# -------- Intent Detection --------
def _detect_intent(query: str) -> str:
    q = (query or "").lower()
    # Greeting
    if re.search(r"\bhello|hi|hey\b", q):
        return "greeting"
    # Procedural / how-to
    elif re.search(r"how to|step|procedure|guide|reset|install", q):
        return "procedural"
    # Meta / yes-no or agent-capability questions
    elif re.search(r"\bare you a bot\b|\bare you a robot\b|\bcan i ask you\b|\boutside comemr\b|\bother things\b", q):
        return "meta"
    # FAQ style wh-questions
    elif re.search(r"\bwhat|who|when|where|why\b", q):
        return "faq"
    return "general"

# -------- Output Sanitizer --------
_PATH_PATTERN = re.compile(r"([A-Za-z]:\\[^ \n]+|\/[^ \n]+)", re.IGNORECASE)
_BRACKET_CITE_PATTERN = re.compile(r"\[\s*\d+\s*\]")
_INTERNAL_CONF_PATTERN = re.compile(r"(?i)i'm not fully confident.*?(?:\n|$)")

def _sanitize_text_ui(text: str, brand: str) -> str:
    if not text:
        return ""
    text = _BRACKET_CITE_PATTERN.sub("", text)
    text = _PATH_PATTERN.sub("", text)
    text = _INTERNAL_CONF_PATTERN.sub("", text)
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
        self.llm_model = llm_model or getattr(settings, "OPENAI_MODEL", "gpt-4o-mini")
        self.top_k = top_k or getattr(settings, "TOP_K", 3)
        self.confidence_threshold = confidence_threshold or 0.45
        self.max_context_chars = max_context_chars or 6000
        self.safeguard = safeguard
        self.retriever = Retriever(top_k=self.top_k)
        self.system_prompt = _load_system_prompt()
        self.brand = os.getenv("COMEMR_BRAND_NAME", "ComEMR Support")

    def answer(
        self,
        query: Optional[str] = None,
        audio_path: Optional[str] = None,
        image_path: Optional[str] = None,
        language: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Produce an answer and return meta including confidence, citations, and detected intent.

        language: 'en' (default) or 'krio' to indicate user requested Krio mixing.
        session_id: optional session identifier used to fetch conversation memory summaries.
        """
        query = query or ""

        # 1️⃣ Handle audio/image
        if audio_path:
            query += f"\n[Audio transcription]: {_audio_to_text(audio_path)}"
        if image_path:
            query += f"\n[Image text]: {_image_to_text(image_path)}"

        # 1b️⃣ Conversation memory hint (best-effort)
        memory_summary = ""
        try:
            from config.settings import settings
            if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and session_id:
                try:
                    from core.memory.memory_service import ConversationMemory
                    mem = ConversationMemory()
                    memory_summary = mem.summarize(session_id)
                    if memory_summary:
                        query = f"[Conversation summary]: {memory_summary}\n\n{query}"
                except Exception:
                    pass
        except Exception:
            pass

        # 2️⃣ Guardrails
        blocked, msg = _default_guardrails(query, self.safeguard)
        if blocked:
            return msg, {"confidence": 0.0, "citations": [], "guarded": True, "intent": None}

        # 3️⃣ Micro-template check
        micro_steps = _match_micro_template(query)
        if micro_steps:
            answer_text = "\n".join(micro_steps)
            return answer_text, {"confidence": 1.0, "citations": [], "guarded": False, "intent": _detect_intent(query)}

        # 4️⃣ Intent detection
        intent = _detect_intent(query)
        if intent == "greeting":
            return f"Hello! I’m {self.brand}. How can I assist you today?", {"confidence": 1.0, "citations": [], "guarded": False, "intent": intent}

        # Meta intent: short direct answers (do not use numbered lists)
        if intent == "meta":
            meta_ans = (
                f"I’m a virtual assistant for {self.brand}. I can help with ComEMR support — troubleshooting, how-tos, and guidance. "
                "You can ask me about the app, connectivity, or common procedures."
            )
            return meta_ans, {"confidence": 1.0, "citations": [], "guarded": False, "intent": intent}

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
        low_confidence = not bool(filtered)
        answer_text = self._compose_with_llm(query, context_chunks, low_confidence=low_confidence, language=language, session_id=session_id)

        clean_answer = _sanitize_text_ui(answer_text, self.brand)
        max_conf = max([float(c.get("score", 0.0)) for c in context_chunks], default=0.0)
        return clean_answer, {"confidence": max_conf, "citations": citations, "guarded": False, "intent": intent}

    def _compose_with_llm(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        low_confidence: bool,
        language: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        context_text = "\n\n".join([c.get("text", "") for c in context_chunks])

        # Retrieve vector memory items if enabled and session_id provided
        memory_texts = []
        try:
            from config.settings import settings
            if session_id and getattr(settings, "ENABLE_VECTOR_MEMORY", False):
                try:
                    from core.memory.vector_memory import VectorMemory
                    vec = VectorMemory()
                    mem_items = vec.get_similar(session_id, query, top_k=getattr(settings, "MEMORY_VECTOR_TOP_K", 3))
                    if mem_items:
                        memory_texts = [f"{m['role']}: {m['text']}" for m in mem_items]
                except Exception:
                    pass
        except Exception:
            pass

        memory_section = ("Relevant memory:\n" + "\n".join(memory_texts) + "\n\n") if memory_texts else ""

        # Additional guardrail instructions: prefer plain numbered lists, be tolerant of typos, and infer intent
        lang_instruction = ""
        if language == 'krio':
            lang_instruction = (
                "\nNote: The user requested Krio. Reply primarily in English but include short, natural Krio phrases where appropriate. "
                "Keep important facts in English to maintain clarity for clinicians; Krio should not exceed ~20% of the reply and should be used only when the user initiated it."
            )

        prompt = f"""
{self.system_prompt}

User question:
{query}

{memory_section}Relevant context:
{context_text or '[No relevant context found]'}

Instructions:
- Answer clearly and concisely.
- For short direct answers (yes/no or single-line replies), do NOT use numbered lists; use a single plain sentence.
- Provide plain numbered steps (e.g., '1. Step one') only for procedural answers with multiple steps; avoid markdown bullets or emphasis.
- Be tolerant of minor typos and infer user intent; ask a clarifying question only if intent is unclear.
- Use friendly language where appropriate.
- Avoid internal paths, confidential info, or private data.
- If unsure, suggest safe next steps.
{lang_instruction}
"""
        if low_confidence:
            prompt += "\nNote: KB confidence is low; do NOT hallucinate. If you cannot answer confidently, respond that you don't know and ask ONE concise clarifying question (1 sentence). You may provide brief general guidance labeled 'General guidance' if helpful.\n"

        try:
            # chat_complete signature: chat_complete(prompt, *, model=None, temperature=None, max_tokens=None)
            answer = chat_complete(prompt, model=self.llm_model, temperature=settings.LLM_TEMPERATURE, max_tokens=settings.LLM_MAX_TOKENS)
            return answer.strip()
        except Exception:
            # Log full exception server-side but return a generic safe message to the user
            logger = logging.getLogger(__name__)
            logger.exception("LLM chat completion failed")
            return "Sorry, an internal error occurred while generating the answer. Please try again later."