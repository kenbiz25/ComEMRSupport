# rag/flow.py
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any
if TYPE_CHECKING:
    from core.knowledge.store_faiss import FaissStore
    from core.llm.llm_service import LLMService
    from core.whatsapp.whatsapp_service import WhatsAppService

import logging
import re

from config.settings import settings

logger = logging.getLogger(__name__)

# Availability flag only — do NOT instantiate at import time (slow disk load)
try:
    from spellchecker import SpellChecker as _SpellChecker
    _SPELLCHECKER_AVAILABLE = True
except Exception:
    _SpellChecker = None  # type: ignore
    _SPELLCHECKER_AVAILABLE = False


class BotFlow:
    """High-level orchestration for handling incoming user messages.

    Behavior:
    - optionally spell-corrects user messages
    - uses RagComposer (RAG + LLM) to respond
    - uses ConversationMemory markers to avoid repeated menus and repeated follow-ups
    - supports Krio only when user explicitly requests it
    """

    def __init__(
        self,
        faiss_store: FaissStore,
        llm_service: LLMService,
        whatsapp_service: WhatsAppService,
        composer=None,
    ):
        self.faiss = faiss_store
        self.llm = llm_service
        self.whatsapp = whatsapp_service
        self._spell = None  # lazy-loaded on first use
        self._composer = composer

    @property
    def spell(self):
        if self._spell is None and _SPELLCHECKER_AVAILABLE:
            self._spell = _SpellChecker()
        return self._spell

    def _get_composer(self):
        if self._composer:
            return self._composer
        try:
            from rag.composer import RagComposer
            self._composer = RagComposer()
        except Exception as e:
            logger.error(f"Could not create RagComposer: {e}")
            self._composer = None
        return self._composer

    def preprocess(self, message: str) -> str:
        """Return a (possibly) corrected message for downstream processing."""
        if not message:
            return ""
        if not self.spell:
            return message

        # Safer correction: SpellChecker.correction can return None
        corrected_words = []
        for w in message.split():
            try:
                cw = self.spell.correction(w)
                corrected_words.append(cw if cw else w)
            except Exception:
                corrected_words.append(w)

        corrected = " ".join(corrected_words)
        return corrected.strip()

    def _format_outgoing(self, text: str) -> str:
        """Normalize outgoing text:
        - strip simple markdown asterisks
        - limit list items to 5 items
        - truncate by sentence boundary to ~800 chars
        """
        if not text:
            return text

        # Remove simple markdown emphasis (*bold*)
        try:
            text = re.sub(r"\*(.*?)\*", r"\1", text)
        except Exception:
            pass

        text = text.strip()

        # Detect and limit lists (numbered or bullet)
        lines = text.splitlines()
        list_start = None
        for i, line in enumerate(lines):
            if re.match(r"^\s*(?:\d+\.\s+|[-\*]\s+)", line):
                list_start = i
                break

        if list_start is not None:
            header = "\n".join(lines[:list_start]).strip()
            list_items = []
            for line in lines[list_start:]:
                m = re.match(r"^\s*(?:\d+\.\s+|[-\*]\s+)(.*)", line)
                if m:
                    item = m.group(1).strip()
                    if item:
                        list_items.append(item)
                else:
                    break

            if list_items:
                truncated = False
                if len(list_items) > 5:
                    list_items = list_items[:5]
                    truncated = True

                numbered = "\n".join([f"{i+1}. {it}" for i, it in enumerate(list_items)])
                if truncated:
                    numbered = numbered + "\n..."

                text = f"{header}\n\n{numbered}" if header else numbered

                # Flatten very short lists into plain sentence(s)
                try:
                    if len(list_items) <= 2 and sum(len(s) for s in list_items) < 200:
                        flat = " ".join(list_items).strip()
                        if flat and not flat.endswith((".", "!", "?")):
                            flat += "."
                        text = f"{header}\n\n{flat}" if header else flat
                except Exception:
                    pass

        # Safe truncation by sentence boundary (~800 chars)
        if len(text) > 800:
            try:
                sentences = re.split(r"(?<=[.!?])\s+", text)
                out = ""
                for s in sentences:
                    if len(out) + len(s) + 1 > 800:
                        break
                    out = out + (" " if out else "") + s
                text = out.strip()
                if text and not text.endswith((".", "!", "?")):
                    text += "."
            except Exception:
                text = text[:800]

        # Post-process: flatten short numbered lists to plain text
        try:
            lines2 = [l.strip() for l in text.splitlines() if l.strip()]
            if lines2 and re.match(r"^\d+\.\s+", lines2[0]) and len(lines2) <= 2 and len(text) < 200:
                flattened = " ".join([re.sub(r"^\d+\.\s+", "", l) for l in lines2]).strip()
                if flattened and not flattened.endswith((".", "!", "?")):
                    flattened += "."
                text = flattened
        except Exception:
            pass

        return text

    def _detect_krio(self, text: str) -> bool:
        """Enable Krio only when user explicitly asks for it."""
        if not text:
            return False
        t = text.lower()
        return bool(
            "krio" in t or "speak krio" in t or "in krio" in t or t.strip().startswith("krio:")
        )

    def _is_short_ack(self, text: str) -> bool:
        """Short acknowledgements that should not trigger follow-ups."""
        if not text:
            return False
        t = text.strip().lower()
        if len(t) > 60:
            return False
        if re.search(r"\b(thank(s| you)?|than you|ty|bye|goodbye|see you|thanks a lot|ok(ay)?|k)\b", t):
            return True
        if t in ("ok", "okay", "yes", "no", "sure"):
            return True
        return False

    def _is_closing_message(self, text: str) -> bool:
        """Explicit conversation-ending messages."""
        if not text:
            return False
        t = text.strip().lower()
        if len(t) > 120:
            return False
        return bool(re.search(r"\b(thank(s| you)?|than you|thanks a lot|bye|goodbye|see you|talk later)\b", t))

    def _is_resolution_message(self, text: str) -> bool:
        """User indicates issue is resolved or working now."""
        if not text:
            return False
        t = text.strip().lower()
        if len(t) > 200:
            return False
        return bool(
            re.search(r"\b(resolved|fixed|worked|now works|working now|it works|it worked|problem solved|solved|all good|okay now|ok now)\b", t)
        )

    def _mem(self):
        """Best-effort ConversationMemory loader."""
        try:
            from core.memory.memory_service import ConversationMemory
            return ConversationMemory()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Ticket helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_ticket_yes(text: str) -> bool:
        t = text.strip().lower()
        return t in ("yes", "y", "yeah", "yep", "sure", "ok", "okay", "yep", "confirm")

    @staticmethod
    def _is_ticket_no(text: str) -> bool:
        t = text.strip().lower()
        return t in ("no", "n", "nope", "nah", "cancel", "never mind", "nevermind")

    @staticmethod
    def _extract_pending_issue(recent: list) -> str:
        """Return the user message that triggered the ticket confirmation request."""
        # Find the index of the ticket_pending marker
        pending_idx = None
        for i, m in enumerate(recent):
            if m.get("role") == "system" and m.get("text") == "ticket_pending":
                pending_idx = i

        if pending_idx is None:
            return "Issue not captured"

        # Walk backwards to find the last user message before the marker
        for m in reversed(recent[:pending_idx]):
            if m.get("role") == "user":
                return (m.get("text") or "")[:500]

        return "Issue not captured"

    def handle_message(self, user_id: str, message: str, session_id: Optional[str] = None):
        """Main entrypoint called by the webhook to handle and respond to a message."""
        raw = message or ""
        cleaned = self.preprocess(raw)

        # ✅ Critical: if session_id not provided, default it to user_id (stable per WhatsApp user)
        # This prevents "memory never works" situations.
        if not session_id:
            session_id = user_id

        use_krio = self._detect_krio(raw)
        language = "krio" if use_krio else "en"

        # --- Memory: always save the user’s message (best-effort) ---
        mem = None
        try:
            if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and session_id:
                mem = self._mem()
                if mem:
                    mem.save_message(session_id, "user", cleaned)
        except Exception:
            mem = None

        # If conversation was previously closed and user sends only a short ACK, ignore.
        try:
            if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and mem and session_id:
                recent = mem.get_recent(session_id, limit=10)
                closed = any(m.get("role") == "system" and m.get("text") == "conversation_closed" for m in recent)
                if closed and self._is_short_ack(raw):
                    return "", {"ignored": True}
        except Exception:
            pass

        # --- Ticket confirmation: handle yes/no reply to a pending ticket request ---
        try:
            if (
                getattr(settings, "ENABLE_TICKETING", True)
                and getattr(settings, "ENABLE_CONVERSATION_MEMORY", False)
                and mem
                and session_id
            ):
                recent = mem.get_recent(session_id, limit=15)
                ticket_pending = any(
                    m.get("role") == "system" and m.get("text") == "ticket_pending"
                    for m in recent
                )
                if ticket_pending:
                    if self._is_ticket_yes(cleaned):
                        issue = self._extract_pending_issue(recent)
                        from core.tickets.ticket_manager import create_ticket
                        ticket_id = create_ticket(user_id, issue)
                        mem.save_message(session_id, "system", "ticket_created")
                        if ticket_id:
                            outgoing = (
                                f"Your support ticket has been created.\n"
                                f"Ticket ID: {ticket_id}\n"
                                "Our team will review it and get back to you shortly."
                            )
                        else:
                            outgoing = (
                                "I tried to create a ticket but something went wrong on our end. "
                                "Please contact support directly."
                            )
                        try:
                            self.whatsapp.send_message(user_id, message=outgoing)
                        except Exception:
                            pass
                        mem.save_message(session_id, "assistant", outgoing)
                        return outgoing, {"ticket_created": bool(ticket_id), "ticket_id": ticket_id, "language": language}

                    elif self._is_ticket_no(cleaned):
                        mem.save_message(session_id, "system", "ticket_declined")
                        outgoing = "No problem! Let me know if there's anything else I can help you with."
                        try:
                            self.whatsapp.send_message(user_id, message=outgoing)
                        except Exception:
                            pass
                        mem.save_message(session_id, "assistant", outgoing)
                        return outgoing, {"ticket_declined": True, "language": language}
                    # else: ambiguous reply — fall through to normal RAG flow
        except Exception:
            pass

        # Resolution messages -> single closing reply + close marker
        if self._is_resolution_message(cleaned):
            outgoing = "Great — glad it’s working now. If you need anything else, just message me anytime."
            try:
                if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and mem and session_id:
                    mem.save_message(session_id, "assistant", outgoing)
                    mem.save_message(session_id, "system", "conversation_closed")
            except Exception:
                pass

            try:
                self.whatsapp.send_message(user_id, message=outgoing)
            except Exception:
                pass

            return outgoing, {"conversation_closed": True, "resolution_ack": True, "language": language}

        # Closing messages -> single closing reply + close marker
        if self._is_closing_message(cleaned):
            outgoing = "You’re welcome — glad I could help. If you need anything else, just message me anytime."
            try:
                if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and mem and session_id:
                    mem.save_message(session_id, "assistant", outgoing)
                    mem.save_message(session_id, "system", "conversation_closed")
            except Exception:
                pass

            try:
                self.whatsapp.send_message(user_id, message=outgoing)
            except Exception:
                pass

            return outgoing, {"conversation_closed": True, "language": language}

        # First-touch menu (only once per session)
        try:
            if (
                getattr(settings, "ENABLE_FIRST_TOUCH_MENU", True)
                and getattr(settings, "ENABLE_CONVERSATION_MEMORY", False)
                and mem
                and session_id
            ):
                recent = mem.get_recent(session_id, limit=20)

                menu_shown = any(m.get("role") == "system" and m.get("text") == "menu_shown" for m in recent)
                menu_consumed = any(m.get("role") == "system" and m.get("text") == "menu_consumed" for m in recent)

                # show menu only if never shown before (and not consumed)
                if not menu_shown and not menu_consumed:
                    try:
                        from core.menu import menu_routing
                        menu_list = menu_routing.show_menu_options(None)
                        menu_text = "\n".join(menu_list)
                        self.whatsapp.send_message(user_id, message=menu_text)
                        mem.save_message(session_id, "assistant", menu_text)
                        mem.save_message(session_id, "system", "menu_shown")
                        return menu_text, {"menu_shown": True, "language": language}
                    except Exception:
                        pass

                # If menu shown, process numeric selection
                if menu_shown and not menu_consumed:
                    try:
                        from core.menu.menu_routing import process_menu_selection
                        sel = cleaned.strip().lower()
                        reply, meta = process_menu_selection(sel)

                        # Only treat as menu selection if menu_selected is not None
                        if meta.get("menu_selected") is not None:
                            try:
                                self.whatsapp.send_message(user_id, message=reply)
                            except Exception:
                                pass
                            try:
                                mem.save_message(session_id, "assistant", reply)
                                mem.save_message(session_id, "system", "menu_consumed")
                            except Exception:
                                pass
                            return reply, {**(meta or {}), "language": language}
                    except Exception:
                        pass
        except Exception:
            pass

        composer = self._get_composer()
        if composer is None:
            # Last resort fallback: embeddings + FAISS + LLM service
            try:
                from adapters.llm.openai_client import get_openai
                client = get_openai()
                resp = client.embeddings.create(model="text-embedding-3-small", input=cleaned)
                embedding = resp.data[0].embedding
            except Exception as e:
                logger.error(f"Embedding failed or OpenAI client unavailable: {e}")
                embedding = [0] * getattr(self.faiss, "dim", 1536)

            docs = self.faiss.search(embedding, top_k=5)
            answer = self.llm.generate_response(cleaned, docs, language=language)

            try:
                if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and mem and session_id:
                    mem.save_message(session_id, "assistant", answer)
            except Exception:
                pass

            try:
                self.whatsapp.send_message(user_id, message=answer)
            except Exception:
                pass

            logger.info(f"Sent reply to {user_id} (fallback LLM path)")
            return answer, {"confidence": 0.0, "language": language}

        # Use RagComposer
        try:
            answer_text, meta = composer.answer(query=cleaned, language=language, session_id=session_id)
        except Exception as e:
            logger.error(f"Composer failed: {e}")
            answer_text = "Sorry, I couldn't generate a reply right now."
            meta = {"confidence": 0.0, "citations": [], "low_confidence": True}

        # Handoff detection (kept as-is, but make sure it doesn't spam)
        try:
            user_lower = (cleaned or "").strip().lower()

            explicit_re = re.compile(r"\b(connect me to support|please connect.*support|connect me to an agent|escalate to support)\b", re.I)
            mild_re = re.compile(r"\b(talk to support|talk to a support agent|support agent|human|talk to an agent)\b", re.I)

            handoff = False
            if explicit_re.search(user_lower):
                handoff = True
            elif mild_re.search(user_lower):
                cond = False
                if meta.get("low_confidence"):
                    cond = True

                # if we already asked a clarifying question recently, allow handoff
                if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and mem and session_id:
                    recent_msgs = mem.get_recent(session_id, limit=6)
                    for m in recent_msgs:
                        if m.get("role") == "assistant" and re.search(r"could you provide|please provide|i don't have enough|one brief detail", m.get("text", ""), re.I):
                            cond = True
                            break

                if cond:
                    handoff = True

            if handoff:
                meta = meta or {}
                meta["handoff"] = True

                if getattr(settings, "ENABLE_TICKETING", True):
                    outgoing = (
                        "I wasn't able to fully resolve this for you. "
                        "Would you like me to create a support ticket so our team can follow up? "
                        "Reply Yes to confirm or No to cancel."
                    )
                    try:
                        if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and mem and session_id:
                            mem.save_message(session_id, "assistant", outgoing)
                            mem.save_message(session_id, "system", "ticket_pending")
                    except Exception:
                        pass
                else:
                    outgoing = "Please hold — connecting you to a support agent now. I'll include a short summary so they can help you faster."
                    try:
                        if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and mem and session_id:
                            mem.save_message(session_id, "assistant", outgoing)
                            mem.save_message(session_id, "system", "handoff_requested")
                    except Exception:
                        pass

                try:
                    self.whatsapp.send_message(user_id, message=outgoing)
                except Exception:
                    pass

                return outgoing, meta
        except Exception:
            pass

        # Low-confidence augmentation (kept, but avoid double-asking)
        try:
            threshold = getattr(settings, "ANSWER_CONFIDENCE_THRESHOLD", 0.55)
            fallback_note = getattr(settings, "FALLBACK_MESSAGE", "")
        except Exception:
            threshold = 0.55
            fallback_note = ""

        outgoing = answer_text
        confidence = float((meta or {}).get("confidence", 0.0))
        citations = (meta or {}).get("citations", [])

        if confidence < threshold:
            clarify = "Could you share one brief detail (e.g., exact error message, where you are stuck, or device type) so I can guide you accurately?"
            already_asked = False

            if (meta or {}).get("low_confidence"):
                already_asked = True
            if (meta or {}).get("citations"):
                already_asked = True
            if re.search(r"could you provide|please provide|i don't have enough|one brief detail|could you share", outgoing, re.I):
                already_asked = True

            # prevent repeating prompt if it was asked recently
            try:
                if mem and session_id:
                    recent_msgs = mem.get_recent(session_id, limit=6)
                    for m in recent_msgs:
                        if m.get("role") == "assistant" and re.search(r"could you provide|could you share|one brief detail", m.get("text", ""), re.I):
                            already_asked = True
                            break
            except Exception:
                pass

            if not already_asked:
                if fallback_note and fallback_note not in outgoing:
                    outgoing = f"{outgoing}\n\n{fallback_note}\n\n{clarify}"
                else:
                    outgoing = f"{outgoing}\n\n{clarify}"

        # Final formatting for WhatsApp
        outgoing = self._format_outgoing(outgoing)

        # Save assistant response to memory (best-effort)
        try:
            if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and mem and session_id:
                mem.save_message(session_id, "assistant", outgoing)
        except Exception:
            pass

        # Send message
        try:
            self.whatsapp.send_message(user_id, message=outgoing)
            logger.info(f"Sent reply to {user_id} | confidence={confidence:.3f} | citations={citations}")
        except Exception as e:
            logger.error(f"Failed sending message to {user_id}: {e}")

        meta = meta or {}
        meta.setdefault("language", language)
        return outgoing, meta