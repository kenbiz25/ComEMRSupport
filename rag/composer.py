from typing import Optional, List, Dict, Any, Tuple, Deque
import os
import pathlib
import re
import logging
import time
import textwrap
from collections import deque
from difflib import SequenceMatcher

from config.settings import settings, confidence_thresholds
from .retriever import Retriever
from adapters.llm.openai_client import chat_complete

logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================
MAX_PROMPT_CHARS = 12000
MAX_MEMORY_CHARS = 2000
MAX_SUMMARY_CHARS = 1000
WRAP_WIDTH = int(os.getenv("WHATSAPP_WRAP_WIDTH", "0"))  # 0 = no wrap

# ==================== Special Messages ====================
SPECIAL_MESSAGES = {
    "greetings": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"],
    "farewells": ["bye", "goodbye", "see you", "later"],
    "thanks": ["thanks", "thank you", "thx"],
    "exit": ["stop", "exit", "end chat", "reset", "cancel"],
}

# ==================== Special Message Checker ====================
def check_special_message(text: str, session_user_turns: int = 0) -> Optional[Dict[str, str]]:
    """
    Detects greetings, farewells, thanks, and exit commands.
    Exit continuity note only shows if user has had >2 messages in this session.
    """
    msg = text.strip().lower()

    # ----------------- EXIT -----------------
    exit_commands = SPECIAL_MESSAGES["exit"]
    if any(msg == word or msg.startswith(word + " ") or msg.endswith(" " + word) for word in exit_commands):
        msg_text = "Goodbye!"
        if session_user_turns > 2:
            msg_text += " Your conversation has been ended. Feel free to start a new chat anytime."
        return {"intent": "exit", "text": msg_text}

    # ----------------- GREETINGS -----------------
    greetings = SPECIAL_MESSAGES["greetings"]
    best_match = max(greetings, key=lambda g: SequenceMatcher(None, g, msg).ratio(), default=None)
    if best_match and SequenceMatcher(None, best_match, msg).ratio() > 0.7:
        return {"intent": "greeting", "text": f"Hello there! 👋 How are you doing today?"}

    # ----------------- FAREWELLS -----------------
    farewells = SPECIAL_MESSAGES["farewells"]
    best_match = max(farewells, key=lambda f: SequenceMatcher(None, f, msg).ratio(), default=None)
    if best_match and SequenceMatcher(None, best_match, msg).ratio() > 0.7:
        return {"intent": "farewell", "text": "Goodbye! Take care and chat with us anytime."}

    # ----------------- THANKS -----------------
    thanks = SPECIAL_MESSAGES["thanks"]
    best_match = max(thanks, key=lambda t: SequenceMatcher(None, t, msg).ratio(), default=None)
    if best_match and SequenceMatcher(None, best_match, msg).ratio() > 0.7:
        return {"intent": "thanks", "text": "You're welcome! Happy to assist you. 😊"}

    return None

# ==================== System Prompt ====================
def _load_system_prompt() -> str:
    brand = os.getenv("COMEMR_BRAND_NAME", getattr(settings, "COMEMR_BRAND_NAME", "ComEMR Support"))
    try:
        prompt_file = pathlib.Path("prompts") / "system.txt"
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return f"""You are {brand}, an AI assistant for the ComEMR healthcare platform.

CORE RULES:
- Answer ONLY questions about ComEMR/SPICE features, workflows, troubleshooting, and user support.
- ALWAYS ground your answers in the provided information without mentioning it explicitly.
- NEVER mention internal systems, documents, or sources.
- If unsure, be honest and suggest contacting support.

STYLE:
- WhatsApp-friendly
- Use short headings (## Heading)
- Numbered steps and bullet points
- Short paragraphs (1–3 lines)
"""

# ==================== Sanitization ====================
def _sanitize_response(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"(?i)\b(kb|knowledge base|vector|faiss|embedding)\b",
        r"(?i)according to .*",
        r"\[\d+\]",
        r"[A-Z]:\\[^\s]+",
        r"/[^\s]+\.(md|pdf|docx|txt)",
    ]
    for p in patterns:
        text = re.sub(p, "", text)
    text = re.sub(r"(?<!\n)(\d+)[\.\)]\s+", r"\n\1. ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ==================== Support Contact ====================
def _format_support_contact() -> str:
    lines = ["*Need more help? Contact ComEMR Support:*"]
    support_teams = [
        {"name": "General Support", "description": "For general platform issues and guidance", "whatsapp": "https://wa.me/1234567890"},
        {"name": "Technical Support", "description": "Troubleshooting errors, bugs, and technical problems", "whatsapp": "https://wa.me/1234567891"},
        {"name": "Training & Onboarding", "description": "Help with user training and onboarding workflows", "whatsapp": "https://wa.me/1234567892"},
    ]
    for team in support_teams:
        lines.append(f"*{team['name']}*\n_{team['description']}_\nWhatsApp: {team['whatsapp']}")
    if getattr(settings, "SUPPORT_EMAIL", None):
        lines.append(f"📧 Email: {settings.SUPPORT_EMAIL}")
    if getattr(settings, "SUPPORT_PHONE", None):
        lines.append(f"📞 Phone: {settings.SUPPORT_PHONE}")
    if getattr(settings, "SUPPORT_DOCS_URL", None):
        lines.append(f"📚 Docs: {settings.SUPPORT_DOCS_URL}")
    return "\n\n".join(lines)

# ==================== Memory ====================
class SessionState:
    def __init__(self, window_size: int):
        self.window: Deque[Dict[str, str]] = deque(maxlen=window_size)
        self.summary: str = ""
        self.turns_since_summary: int = 0
        self.last_updated_ts: float = time.time()

class MemoryStore:
    def __init__(self, window_size: int, ttl_seconds: int):
        self.window_size = window_size
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState:
        now = time.time()
        state = self._sessions.get(session_id)
        if not state or (now - state.last_updated_ts) > self.ttl_seconds:
            state = SessionState(self.window_size)
            self._sessions[session_id] = state
        return state

    def update_turn(self, session_id: str, role: str, content: str):
        state = self.get(session_id)
        state.window.append({"role": role, "content": content})
        state.turns_since_summary += 1
        state.last_updated_ts = time.time()

    def set_summary(self, session_id: str, summary: str):
        state = self.get(session_id)
        state.summary = summary[:MAX_SUMMARY_CHARS]
        state.turns_since_summary = 0
        state.last_updated_ts = time.time()

    def clear_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

# ==================== Intent Detection Helper ====================
def _detect_intent_style(query: str) -> str:
    q = query.lower().strip()
    if q.startswith(("how do", "how to", "how can", "steps", "guide me")):
        return "steps"
    if any(w in q for w in ["list", "what are", "options", "features", "types"]):
        return "list"
    return "explain"

# ==================== Composer ====================
class RagComposer:
    def __init__(self, llm_model: Optional[str] = None, top_k: Optional[int] = None, max_context_chars: Optional[int] = None):
        self.llm_model = llm_model or settings.LLM_MODEL
        self.top_k = top_k or settings.TOP_K
        self.max_context_chars = max_context_chars or 6000
        self.enable_memory = getattr(settings, "ENABLE_CONVERSATION_MEMORY", True)
        self.memory_window = int(getattr(settings, "MEMORY_WINDOW", 6))
        self.summary_every_turns = int(getattr(settings, "SUMMARY_EVERY_TURNS", 6))
        self.session_ttl = int(getattr(settings, "SESSION_TTL_MINUTES", 30)) * 60
        self._memory = MemoryStore(self.memory_window, self.session_ttl)
        self.retriever = Retriever()
        self.system_prompt = _load_system_prompt()

    # ---------------- Public API ----------------
    def answer(self, query: str, session_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        # count only user turns for exit continuity
        session_user_turns = 0
        if self.enable_memory and session_id:
            state = self._memory.get(session_id)
            session_user_turns = sum(1 for t in state.window if t["role"] == "user")

        special = check_special_message(query, session_user_turns=session_user_turns)
        if special:
            if special.get("intent") == "exit" and session_id:
                self._memory.clear_session(session_id)
                return special["text"], {"confidence": 1.0, "strategy": "exit"}
            return special["text"], {"confidence": 1.0, "strategy": "special_message"}

        result = self.compose_answer(query, session_id)
        answer = self._format_for_whatsapp(_sanitize_response(result["answer"]), user_query=query)

        if not answer:
            answer = "I’m having trouble answering that right now.\n\n" + _format_support_contact()

        if self.enable_memory and session_id:
            self._memory.update_turn(session_id, "assistant", answer)
            self._maybe_refresh_summary(session_id)

        return answer, {"confidence": result["confidence"], "strategy": result["strategy"]}

    # ---------------- Core Logic ----------------
    def compose_answer(self, query: str, session_id: Optional[str]) -> Dict[str, Any]:
        logger.info("Processing query: %s", query[:100])
        if self.enable_memory and session_id:
            self._memory.update_turn(session_id, "user", query)
        results = self.retriever.retrieve(query, top_k=self.top_k)
        if not results:
            return {"answer": "I couldn’t find relevant information.\n\n" + _format_support_contact(),
                    "confidence": 0.0, "strategy": "no_results"}

        confidence = max(r.get("score", 0.0) for r in results)
        strategy = confidence_thresholds.get_strategy(confidence)

        context = self._prepare_context(results)
        summary, recent = self._build_memory_context(session_id) if session_id else ("", "")
        answer = self._generate_llm_response(query=query, context=context, memory_summary=summary, memory_recent=recent)
        return {"answer": answer, "confidence": confidence, "strategy": strategy}

    # ---------------- Helpers ----------------
    def _prepare_context(self, results: List[Dict[str, Any]]) -> str:
        chunks, total = [], 0
        for r in results:
            t = (r.get("text") or "").strip()
            if not t:
                continue
            if total + len(t) > self.max_context_chars:
                break
            chunks.append(t)
            total += len(t)
        return "\n\n".join(chunks)

    def _build_memory_context(self, session_id: str) -> Tuple[str, str]:
        state = self._memory.get(session_id)
        recent_lines, total = [], 0
        for t in reversed(state.window):
            line = f"{'You' if t['role']=='user' else 'Assistant'}: {t['content']}"
            if total + len(line) > MAX_MEMORY_CHARS:
                break
            recent_lines.insert(0, line)
            total += len(line)
        return state.summary[:MAX_SUMMARY_CHARS], "\n".join(recent_lines)

    def _maybe_refresh_summary(self, session_id: str):
        state = self._memory.get(session_id)
        if state.turns_since_summary < self.summary_every_turns:
            return
        convo = "\n".join(f"{t['role']}: {t['content']}" for t in state.window if t["content"])
        if not convo:
            return
        summary = chat_complete(f"Summarize the following conversation briefly:\n\n{convo}\n\nSummary:", model="gpt-4.1-mini", temperature=0.2)
        if summary:
            self._memory.set_summary(session_id, summary)

    def _generate_llm_response(self, query: str, context: str, memory_summary: str, memory_recent: str) -> str:
        prompt = f"""{self.system_prompt}

Conversation Summary:
{memory_summary or "[None]"}

Recent Conversation:
{memory_recent or "[None]"}

User Question:
{query}

Relevant Information:
{context}

Answer format:
- Short 1–2 line overview
- Numbered steps for procedures
- Bullets for options
- Short paragraphs
- WhatsApp-friendly
Answer:"""

        if len(prompt) > MAX_PROMPT_CHARS:
            prompt = prompt[-MAX_PROMPT_CHARS:]

        try:
            return (chat_complete(prompt, model=self.llm_model) or "").strip()
        except Exception as e:
            logger.error("LLM failure: %s", e)
            return "I encountered a temporary issue.\n\n" + _format_support_contact()

    # ---------------- WhatsApp Formatting ----------------
    def _format_for_whatsapp(self, text: str, user_query: Optional[str] = None) -> str:
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\s*\n\s*", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        def heading_to_bold(line: str) -> str:
            m = re.match(r"^\s{0,3}(#{1,6})\s+(.*)$", line)
            if not m:
                return line
            return f"*{m.group(2).strip()}*"

        lines = [heading_to_bold(l) for l in text.split("\n")]
        for i, l in enumerate(lines):
            lines[i] = re.sub(r"^(\d+)[\.\)]\s*", r"\1. ", l)
            lines[i] = re.sub(r"^(\s*)[\*\-\•]\s*", r"\1• ", l)

        formatted = []
        for idx, l in enumerate(lines):
            formatted.append(l)
            if re.match(r"^\*.+\*$", l):
                nxt = lines[idx+1] if idx+1 < len(lines) else ""
                if nxt and not re.match(r"^\s*(•|\d+\.)\s+", nxt):
                    formatted.append("")

        style = _detect_intent_style(user_query or "")
        final_lines = []

        if style == "steps":
            num = 1
            for l in text.split("\n"):
                if l.startswith("• "):
                    final_lines.append(f"{num}. {l[2:].strip()}")
                    num += 1
                else:
                    final_lines.append(l)
            text = "\n".join(final_lines)
        elif style == "list":
            final_lines = []
            for l in text.split("\n"):
                if re.match(r"^\d+\.\s+", l):
                    final_lines.append(f"• {l.split('.', 1)[1].strip()}")
                else:
                    final_lines.append(l)
            text = "\n".join(final_lines)

        if WRAP_WIDTH and WRAP_WIDTH > 0:
            wrapped = []
            for l in text.split("\n"):
                wrapped.extend(textwrap.wrap(l, width=WRAP_WIDTH, replace_whitespace=False) if len(l) > WRAP_WIDTH else [l])
            text = "\n".join(wrapped)

        if style == "steps" and not text.strip().endswith("?"):
            text += "\n\nNeed help with the next part?"

        return text.strip()
