
from typing import Optional, List, Dict, Any, Tuple, Deque
import os
import pathlib
import re
import logging
import time
from collections import deque

from config.settings import settings, confidence_thresholds
from .retriever import Retriever
from adapters.llm.openai_client import chat_complete

logger = logging.getLogger(__name__)


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
- NEVER mention “knowledge base”, “documents”, “context”, vectors, or internal systems.
- NEVER display citations, file names, or sources.
- If unsure, be honest and suggest contacting support.

RESPONSE STYLE:
- Clear, friendly, helpful (WhatsApp style).
- Short paragraphs and bullet points.
- Use numbered steps for procedures.
"""


# ==================== Sanitization (no KB leakage) ====================
def _sanitize_response(text: str) -> str:
    if not text:
        return ""
    patterns_to_remove = [
        r"(?i)\b(kb|knowledge base|vector|faiss|index|database|embedding|embeddings)\b",
        r"(?i)according to .*",
        r"(?i)from the document.*",
        r"\[\d+\]", r"\[cite:\d+\]",
        r"[A-Z]:\\[^\s]+",            # Windows path
        r"/[^\s]+\.(md|pdf|docx|txt)" # Unix path w/ ext
    ]
    for p in patterns_to_remove:
        text = re.sub(p, "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==================== Support Contact ====================
def _format_support_contact() -> str:
    lines = ["If you need more help, contact ComEMR Support:"]
    if settings.SUPPORT_EMAIL:
        lines.append(f"📧 {settings.SUPPORT_EMAIL}")
    if settings.SUPPORT_PHONE:
        lines.append(f"📞 {settings.SUPPORT_PHONE}")
    if settings.SUPPORT_DOCS_URL:
        lines.append(f"📚 {settings.SUPPORT_DOCS_URL}")
    return "\n".join(lines)


# ==================== In‑Memory Session Store (per phone) ====================
class SessionState:
    """Holds memory for a single user (session_id = WhatsApp phone)."""
    def __init__(self, window_size: int):
        self.window: Deque[Dict[str, str]] = deque(maxlen=window_size)  # recent turns
        self.summary: str = ""                                          # rolling summary
        self.turns_since_summary: int = 0
        self.last_updated_ts: float = time.time()


class MemoryStore:
    """
    Simple in‑process memory store. For production, you can replace internals
    with Redis or a DB while keeping the same interface.
    """
    def __init__(self, window_size: int, ttl_seconds: int):
        self.window_size = window_size
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            state = SessionState(window_size=self.window_size)
            self._sessions[session_id] = state
        # TTL cleanup
        now = time.time()
        if self.ttl_seconds > 0 and (now - state.last_updated_ts) > self.ttl_seconds:
            state = SessionState(window_size=self.window_size)  # reset expired
            self._sessions[session_id] = state
        return state

    def update_turn(self, session_id: str, role: str, content: str):
        state = self.get(session_id)
        state.window.append({"role": role, "content": content})
        state.turns_since_summary += 1
        state.last_updated_ts = time.time()

    def set_summary(self, session_id: str, summary: str):
        state = self.get(session_id)
        state.summary = summary
        state.turns_since_summary = 0
        state.last_updated_ts = time.time()


# ==================== Guardrails / OOS (optional) ====================
def _apply_guardrails(user_query: str, enabled: bool) -> Tuple[bool, str]:
    if not enabled:
        return False, ""
    q = user_query.lower()
    sensitive_terms = ("api key", "token", "private key", "secret", "credential", "password")
    reveal_verbs = ("share", "reveal", "show", "give", "expose", "send", "post")
    for term in sensitive_terms:
        if term in q and any(v in q for v in reveal_verbs):
            return True, (
                "For security and privacy, I can’t help with credentials or passwords. "
                "I can guide you through secure reset steps instead."
            )
    return False, ""


def _is_out_of_scope(query: str) -> Tuple[bool, str]:
    q = query.lower()
    for keyword in settings.OUT_OF_SCOPE_KEYWORDS:
        if keyword in q:
            return True, (
                "I can help with ComEMR questions like features, registration, troubleshooting, "
                "and user management. Please ask about those topics."
            )
    return False, ""


# ==================== RAG Composer ====================
class RagComposer:
    """
    Hybrid memory RAG composer:
    - Query rewriting → better retrieval
    - Multi‑chunk grounding
    - Short‑term window + rolling summary per phone (session_id)
    - WhatsApp‑friendly, no citations
    """

    def __init__(
        self,
        llm_model: Optional[str] = None,
        safeguard: bool = True,
        top_k: Optional[int] = None,
        max_context_chars: Optional[int] = None,
        # Optional: enforce a minimum confidence inside the composer
        answer_min_confidence: Optional[float] = None,
    ):
        self.llm_model = llm_model or settings.LLM_MODEL
        self.safeguard = safeguard
        self.top_k = top_k or settings.TOP_K
        self.max_context_chars = max_context_chars or 6000

        # Optional internal min-confidence threshold (default from settings if provided)
        default_min = getattr(settings, "ANSWER_CONFIDENCE_THRESHOLD", None)
        self.answer_min_confidence = (
            float(answer_min_confidence)
            if answer_min_confidence is not None
            else (float(default_min) if default_min is not None else None)
        )

        # Feature flags / knobs (env‑configurable via settings with sensible defaults)
        self.enable_query_rewrite = getattr(settings, "ENABLE_QUERY_REWRITE", True)
        self.enable_conversation_memory = getattr(settings, "ENABLE_CONVERSATION_MEMORY", True)
        self.memory_window = int(getattr(settings, "MEMORY_WINDOW", 6))
        self.summary_every_turns = int(getattr(settings, "SUMMARY_EVERY_TURNS", 6))
        self.summary_max_chars = int(getattr(settings, "SUMMARY_MAX_CHARS", 1400))
        self.session_ttl_seconds = int(getattr(settings, "SESSION_TTL_MINUTES", 30)) * 60

        # Memory store (per phone/session)
        self._memory = MemoryStore(window_size=self.memory_window, ttl_seconds=self.session_ttl_seconds)

        # Retriever
        try:
            self.retriever = Retriever()
            logger.info("Retriever initialized")
        except Exception as e:
            logger.error(f"Retriever initialization failed: {e}")
            raise

        self.system_prompt = _load_system_prompt()

        # Helpful init log
        logger.info(
            "RagComposer initialized | model=%s | top_k=%s | memory=%s | rewrite=%s | answer_min_confidence=%s",
            self.llm_model, self.top_k, self.enable_conversation_memory,
            self.enable_query_rewrite, self.answer_min_confidence
        )

    # ---------------- Public API ----------------
    def answer(self, query: str, session_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        result = self.compose_answer(query, session_id=session_id)
        answer = _sanitize_response(result["answer"])
        meta = {
            "confidence": result["confidence"],
            "strategy": result["strategy"],
        }
        # Update memory with assistant turn
        if self.enable_conversation_memory and session_id:
            self._memory.update_turn(session_id, "assistant", answer)
            # Maybe produce/update summary
            self._maybe_refresh_summary(session_id)
        return answer, meta

    def compose_answer(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"Processing query: {query[:100]}...")

        # Guardrails & OOS
        blocked, block_msg = _apply_guardrails(query, self.safeguard)
        if blocked:
            return {"answer": block_msg, "confidence": 0.0, "strategy": "blocked"}
        is_oos, oos_msg = _is_out_of_scope(query)
        if is_oos:
            return {"answer": oos_msg, "confidence": 0.0, "strategy": "out_of_scope"}

        # Update memory with user turn
        if self.enable_conversation_memory and session_id:
            self._memory.update_turn(session_id, "user", query)

        # Query rewrite
        rewritten = query
        if self.enable_query_rewrite:
            rewritten = self._rewrite_query(query, session_id=session_id)
            if rewritten != query:
                logger.info(f"Rewritten query: {rewritten}")

        # Retrieve
        try:
            results = self.retriever.retrieve(rewritten, top_k=self.top_k)
        except Exception as e:
            logger.error(f"Retrieval error: {e}", exc_info=True)
            return {
                "answer": "I couldn’t retrieve information right now.\n\n" + _format_support_contact(),
                "strategy": "error",
                "confidence": 0.0,
            }

        if not results:
            return {
                "answer": "I couldn’t find anything related.\n\n" + _format_support_contact(),
                "strategy": "no_results",
                "confidence": 0.0,
            }

        # Confidence & strategy
        max_conf = max(r.get("score", 0.0) for r in results)
        strategy = confidence_thresholds.get_strategy(max_conf)
        logger.info(f"Max confidence: {max_conf:.3f} → {strategy}")

        if strategy == "reject":
            return {
                "answer": "I don’t have enough information to answer that.\n\n" + _format_support_contact(),
                "strategy": "reject",
                "confidence": max_conf,
            }

        # Context prep
        context_chunks = self._prepare_context(results)

        # Build memory context (summary + recent turns)
        mem_summary, mem_recent = ("", "")
        if self.enable_conversation_memory and session_id:
            mem_summary, mem_recent = self._build_memory_context(session_id)

        # Generate final answer
        answer = self._generate_llm_response(
            original_query=query,
            rewritten=rewritten,
            context_chunks=context_chunks,
            strategy=strategy,
            memory_summary=mem_summary,
            memory_recent=mem_recent,
        )

        # Soft guidance addendum based on strategy
        if strategy == "low_confidence":
            answer += "\n\n_If this doesn’t fully answer your question, " + _format_support_contact() + "_"
        elif strategy == "cautious":
            answer += "\n\n_For detailed guidance, " + _format_support_contact() + "_"

        # Optional internal min-confidence gate
        if (self.answer_min_confidence is not None) and (max_conf < float(self.answer_min_confidence)):
            wrapped = (
                "I’m not fully confident in the answer from the internal knowledge base. "
                "Here is an initial suggestion:\n"
                f"{answer}\n\n"
                "For official guidance, please contact Tech Support."
            )
            return {"answer": wrapped, "strategy": "low_conf_threshold", "confidence": max_conf}

        return {"answer": answer, "strategy": strategy, "confidence": max_conf}

    # ---------------- Memory helpers ----------------
    def _build_memory_context(self, session_id: str) -> Tuple[str, str]:
        state = self._memory.get(session_id)
        # Build recent window transcript
        recent_lines = []
        for t in list(state.window)[-self.memory_window:]:
            role = "You" if t["role"] == "user" else "Assistant"
            content = t["content"].strip()
            if content:
                recent_lines.append(f"{role}: {content}")
        recent_text = "\n".join(recent_lines)
        return state.summary or "", recent_text

    def _maybe_refresh_summary(self, session_id: str):
        """Refresh rolling summary every N turns or if recent transcript is long."""
        try:
            state = self._memory.get(session_id)
            if state.turns_since_summary < self.summary_every_turns:
                return
            # Build a compact transcript of the window to summarize
            recent_lines = []
            for t in list(state.window):
                role = "User" if t["role"] == "user" else "Assistant"
                content = t["content"].strip()
                if content:
                    recent_lines.append(f"{role}: {content}")
            recent_text = "\n".join(recent_lines)
            if not recent_text:
                return

            prompt = (
                "Summarize the following conversation turns into a concise, factual update. "
                "Focus on user goals, issues, and resolutions. Keep it under "
                f"{self.summary_max_chars} characters. Avoid pleasantries.\n\n"
                f"{recent_text}\n\nSummary:"
            )
            summary_piece = (chat_complete(prompt, model="gpt-4.1-mini", temperature=0.2) or "").strip()
            # Merge with previous summary (short and evergreen)
            merged = self._merge_summaries(state.summary, summary_piece, self.summary_max_chars)
            self._memory.set_summary(session_id, merged)
        except Exception as e:
            logger.debug(f"Summary refresh skipped: {e}")

    @staticmethod
    def _merge_summaries(prev: str, new: str, max_chars: int) -> str:
        if not prev:
            return new[:max_chars]
        combined = (prev + " | " + new).strip()
        if len(combined) <= max_chars:
            return combined
        # If too long, ask model to compress both (fallback: truncate)
        try:
            prompt = (
                "Compress the following two summaries into a single concise update "
                f"under {max_chars} characters. Be factual and task‑focused.\n\n"
                f"Prev: {prev}\nNew: {new}\n\nCompressed:"
            )
            return (chat_complete(prompt, model="gpt-4.1-mini", temperature=0.2) or "").strip()[:max_chars]
        except Exception:
            return combined[:max_chars]

    # ---------------- Chunking & LLM ----------------
    def _prepare_context(self, results: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for r in results:
            t = (r.get("text") or "").strip()
            if not t:
                continue
            if total + len(t) > self.max_context_chars:
                break
            out.append(t)
            total += len(t)
        return out

    def _rewrite_query(self, query: str, session_id: Optional[str] = None) -> str:
        try:
            # Include recent memory to resolve pronouns like "that", "it", "this"
            mem_summary, mem_recent = ("", "")
            if self.enable_conversation_memory and session_id:
                mem_summary, mem_recent = self._build_memory_context(session_id)
            prompt_parts = [
                "Rewrite the user's question so it is explicit and suitable for searching a knowledge base.",
                "Preserve meaning. Avoid pronouns; prefer concrete terms. Output only the rewritten query.",
                f"User: {query}",
            ]
            if mem_recent:
                prompt_parts.append(f"Recent Context:\n{mem_recent}")
            if mem_summary:
                prompt_parts.append(f"Conversation Summary:\n{mem_summary}")
            prompt_parts.append("Rewrite:")
            prompt = "\n\n".join(prompt_parts)
            rewritten = (chat_complete(prompt, model="gpt-4.1-mini", temperature=0.15) or "").strip()
            return rewritten if len(rewritten) >= 4 else query
        except Exception:
            return query

    def _generate_llm_response(
        self,
        original_query: str,
        rewritten: str,
        context_chunks: List[str],
        strategy: str,
        memory_summary: str = "",
        memory_recent: str = "",
    ) -> str:
        context = "\n\n".join(context_chunks) if context_chunks else "No relevant information available."

        if strategy == "direct":
            tone = "Give a confident, direct answer using the information."
        elif strategy == "cautious":
            tone = "You have good information. Answer clearly but be mindful about possible gaps."
        else:
            tone = "Give the best possible answer based on limited information."

        # Build prompt with memory
        prompt = f"""{self.system_prompt}

Conversation Summary (if any):
{memory_summary or "[None]"}

Recent Conversation (last turns):
{memory_recent or "[None]"}

User Question:
{original_query}

Rewritten for retrieval:
{rewritten}

Relevant Information:
{context}

Instructions:
- {tone}
- DO NOT mention context, sources, documents, KB, file names, or internal systems.
- DO NOT show citations.
- Use short paragraphs and bullet points.
- Be friendly and helpful.

Answer:"""

        try:
            return (chat_complete(prompt, model=self.llm_model) or "").strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            return "I encountered a temporary error.\n\n" + _format_support_contact()
