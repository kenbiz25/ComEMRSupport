# rag/flow.py
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.knowledge.store_faiss import FaissStore
    from core.llm.llm_service import LLMService
    from core.whatsapp.whatsapp_service import WhatsAppService

import logging
import re

try:
    from spellchecker import SpellChecker
except Exception:
    SpellChecker = None

logger = logging.getLogger(__name__)

class BotFlow:
    """High-level orchestration for handling incoming user messages.

    Behavior:
    - optionallly spell-corrects user messages
    - uses the RagComposer to retrieve KB context and use the LLM to compose an answer
    - if KB confidence is low, appends a fallback message and still uses LLM power to answer
    - logs confidence and citations for observability
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
        self.spell = SpellChecker() if SpellChecker else None
        # Accept an injected RagComposer for testability; instantiate lazily if not provided
        self._composer = composer

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
        if not self.spell:
            return message
        corrected = " ".join([self.spell.correction(w) for w in message.split()])
        return corrected

    def _format_outgoing(self, text: str) -> str:
        """Normalize outgoing text:
        - strip simple markdown asterisks (bold/italic)
        - limit list items (numbered or bulleted) to 5 items and indicate truncation
        - truncate by sentence boundary to ~800 chars to avoid mid-sentence cut-offs
        """
        import re

        if not text:
            return text

        # Remove simple markdown emphasis (e.g., *bold* or *italic*)
        try:
            text = re.sub(r"\*(.*?)\*", r"\1", text)
        except Exception:
            pass

        # Normalize whitespace
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
                    # if item empty, ignore (prevents "5." lone items)
                    if item:
                        list_items.append(item)
                else:
                    break

            if list_items:
                # Keep up to 5 items
                truncated = False
                if len(list_items) > 5:
                    list_items = list_items[:5]
                    truncated = True
                # Reconstruct as simple numbered list
                numbered = "\n".join([f"{i+1}. {it}" for i, it in enumerate(list_items)])
                if truncated:
                    numbered = numbered + "\n..."
                if header:
                    text = f"{header}\n\n{numbered}"
                else:
                    text = numbered

                # Flatten short 1-2 item lists into plain sentences to avoid '1.' for simple answers
                try:
                    if len(list_items) <= 2 and sum(len(s) for s in list_items) < 200:
                        flat = " ".join(list_items).strip()
                        if flat and not flat.endswith(('.', '!', '?')):
                            flat = flat + '.'
                        if header:
                            text = f"{header}\n\n{flat}"
                        else:
                            text = flat
                except Exception:
                    pass

        # Safe truncation by sentence boundary (approx 800 chars)
        if len(text) > 800:
            try:
                sentences = re.split(r'(?<=[.!?])\s+', text)
                out = ""
                for s in sentences:
                    if len(out) + len(s) + 1 > 800:
                        break
                    out = out + (" " if out else "") + s
                text = out.strip()
                if text and not text.endswith(('.', '!', '?')):
                    text = text + '.'
            except Exception:
                text = text[:800]

        # Post-process: flatten short numbered lists to plain text (avoid '1.' for simple responses)
        try:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if lines and re.match(r"^\d+\.\s+", lines[0]) and len(lines) <= 2 and len(text) < 200:
                flattened = " ".join([re.sub(r"^\d+\.\s+", "", l) for l in lines]).strip()
                if flattened and not flattened.endswith(('.', '!', '?')):
                    flattened = flattened + '.'
                text = flattened
        except Exception:
            pass

        return text

    def _detect_krio(self, text: str) -> bool:
        """Detect whether the user explicitly asked to use Krio.

        Requirements: KRIO must be initiated by the user. We only enable Krio if the user includes an explicit
        marker such as 'krio', 'speak krio', 'in krio', or prefixes with 'krio:'. This avoids accidental language
        switching. Optionally, if `langdetect` is installed, we use it as an additional heuristic when confidence is low.
        """
        if not text:
            return False
        t = text.lower()
        # explicit markers only - do NOT auto-enable Krio based on tokens alone.
        if 'krio' in t or 'speak krio' in t or 'in krio' in t or t.strip().startswith('krio:'):
            return True

        # Do not infer Krio from single tokens like 'wetin' to avoid accidental language switching.
        # If advanced auto-detection is desired in the future, make it opt-in via a setting.
        return False

    def _is_short_ack(self, text: str) -> bool:
        """Return True for short acknowledgements or simple closings that should not trigger follow-ups.

        Examples: 'thanks', 'thank you', 'than you' (typo), 'bye', 'ok', 'okay', 'k', 'yes', 'no'
        """
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
        """Return True for explicit conversation-ending messages (short 'thank you' / 'bye' forms).

        This is intentionally conservative to avoid misclassifying longer follow-up requests as closers.
        """
        if not text:
            return False
        t = text.strip().lower()
        if len(t) > 120:
            return False
        if re.search(r"\b(thank(s| you)?|than you|thanks a lot|bye|goodbye|see you|talk later)\b", t):
            return True
        return False

    def _is_resolution_message(self, text: str) -> bool:
        """Return True if the user indicates the issue is resolved or working now."""
        if not text:
            return False
        t = text.strip().lower()
        if len(t) > 200:
            return False
        return bool(
            re.search(
                r"\b(resolved|fixed|worked|now works|working now|it works|it worked|problem solved|solved|all good|okay now|ok now)\b",
                t,
            )
        )

    def handle_message(self, user_id: str, message: str, session_id: str | None = None):
        """Main entrypoint called by the webhook to handle and respond to a message.

        session_id: optional session identifier (e.g., phone number) to allow composer to use conversation memory.
        """
        cleaned = self.preprocess(message)
        # detect if user initiated Krio mode
        use_krio = self._detect_krio(message)
        language = 'krio' if use_krio else 'en'

        # Avoid sending repeated follow-ups: if the conversation was explicitly closed earlier, short acknowledgements should be ignored
        try:
            from config.settings import settings
            if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and session_id:
                try:
                    from core.memory.memory_service import ConversationMemory
                    mem = ConversationMemory()
                    recent = mem.get_recent(session_id, limit=10)
                    closed = any(m.get('role') == 'system' and m.get('text') == 'conversation_closed' for m in recent)
                    if closed and self._is_short_ack(message):
                        # persist the user's ack but do not send another message
                        mem.save_message(session_id, "user", message)
                        return "", {"ignored": True}
                except Exception:
                    pass
        except Exception:
            pass

        # Detect resolution messages (issue fixed/working) and send a friendly closing reply
        if self._is_resolution_message(cleaned):
            try:
                from config.settings import settings
                outgoing = "Great — glad it’s working now. If you need anything else, just message me anytime."
                if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and session_id:
                    try:
                        from core.memory.memory_service import ConversationMemory
                        mem = ConversationMemory()
                        mem.save_message(session_id, "user", cleaned)
                        mem.save_message(session_id, "assistant", outgoing)
                        mem.save_message(session_id, "system", "conversation_closed")
                    except Exception:
                        pass
            except Exception:
                outgoing = "Great — glad it’s working now."

            try:
                self.whatsapp.send_message(user_id, message=outgoing)
            except Exception:
                pass

            return outgoing, {"conversation_closed": True, "resolution_ack": True}

        # Detect explicit closing messages (thank you / bye) and send a single friendly closing reply
        if self._is_closing_message(cleaned):
            try:
                from config.settings import settings
                outgoing = "You're welcome — glad I could help. If you need anything else, just message me anytime."
                if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and session_id:
                    try:
                        from core.memory.memory_service import ConversationMemory
                        mem = ConversationMemory()
                        mem.save_message(session_id, "user", cleaned)
                        mem.save_message(session_id, "assistant", outgoing)
                        mem.save_message(session_id, "system", "conversation_closed")
                    except Exception:
                        pass
            except Exception:
                outgoing = "You're welcome — glad I could help."

            try:
                self.whatsapp.send_message(user_id, message=outgoing)
            except Exception:
                pass

            return outgoing, {"conversation_closed": True}

        # Feature: Show a short structured menu only on the first interaction to help CHWs
        try:
            from config.settings import settings
            if getattr(settings, "ENABLE_FIRST_TOUCH_MENU", True) and getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and session_id:
                try:
                    from core.memory.memory_service import ConversationMemory
                    mem = ConversationMemory()
                    recent = mem.get_recent(session_id, limit=10)

                    # If no prior messages, show menu and save a system marker so subsequent replies can be handled
                    if not recent:
                        try:
                            from core.menu import menu_routing
                            menu_list = menu_routing.show_menu_options(None)
                            menu_text = "\n".join(menu_list)
                            # Send and persist menu marker
                            self.whatsapp.send_message(user_id, message=menu_text)
                            mem.save_message(session_id, "assistant", menu_text)
                            mem.save_message(session_id, "system", "menu_shown")
                            return menu_text, {"menu_shown": True}
                        except Exception:
                            # Fall through to normal processing if menu send fails
                            pass

                    # If a menu has been shown and the current message looks like a menu selection, handle it here
                    menu_shown = any(m.get('role') == 'system' and m.get('text') == 'menu_shown' for m in recent)
                    if menu_shown:
                        try:
                            from core.menu.menu_routing import process_menu_selection
                            sel = cleaned.strip().lower()
                            reply, meta = process_menu_selection(sel)
                            # If process_menu_selection didn't understand, let the message go to composer
                            if meta.get('menu_selected') is not None:
                                # save user and assistant messages and mark menu as consumed
                                mem.save_message(session_id, "user", cleaned)
                                self.whatsapp.send_message(user_id, message=reply)
                                mem.save_message(session_id, "assistant", reply)
                                mem.save_message(session_id, "system", "menu_consumed")
                                return reply, meta
                        except Exception:
                            pass
                except Exception:
                    # Memory subsystem unavailable - ignore and continue
                    pass
        except Exception:
            pass

        # Save incoming user message to conversation memory (best-effort)

        composer = self._get_composer()
        if composer is None:
            # Last resort: original behavior (search + LLM service) to keep flow alive
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
            # Save assistant response to memory (best-effort)
            try:
                from config.settings import settings
                if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and session_id:
                    try:
                        from core.memory.memory_service import ConversationMemory
                        mem = ConversationMemory()
                        mem.save_message(session_id, "assistant", answer)
                    except Exception:
                        pass
            except Exception:
                pass

            self.whatsapp.send_message(user_id, answer)
            logger.info(f"Sent reply to {user_id} (fallback LLM path)")
            return answer, {"confidence": 0.0, "language": language}

        # Use RagComposer to produce answer + metadata
        try:
            answer_text, meta = composer.answer(cleaned, language=language, session_id=session_id)
        except Exception as e:
            logger.error(f"Composer failed: {e}")
            # fallback to LLM path
            answer_text = "Sorry, I couldn't generate a reply right now."
            meta = {"confidence": 0.0, "citations": []}

        # --- Subtle hotword-based human handoff (only when appropriate) ---
        try:
            from config.settings import settings
            # normalize incoming cleaned message for handoff detection
            user_lower = (cleaned or "").strip().lower()
            # strong explicit requests -> immediate handoff
            explicit_re = re.compile(r"\b(connect me to support|please connect.*support|connect me to an agent|escalate to support)\b", re.I)
            # mild requests (e.g., "talk to support") are accepted only after options exhausted
            mild_re = re.compile(r"\b(talk to support|talk to a support agent|support agent|human|talk to an agent)\b", re.I)

            handoff = False
            if explicit_re.search(user_lower):
                handoff = True

            # Mild requests: require low KB confidence or that we've already asked a clarifying question
            elif mild_re.search(user_lower):
                cond = False
                try:
                    if meta.get('low_confidence'):
                        cond = True
                except Exception:
                    pass
                try:
                    if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and session_id:
                        from core.memory.memory_service import ConversationMemory
                        mem = ConversationMemory()
                        recent_msgs = mem.get_recent(session_id, limit=6)
                        for m in recent_msgs:
                            if m.get('role') == 'assistant' and re.search(r"could you provide|please provide|i don't have enough|one brief detail", m.get('text',''), re.I):
                                cond = True
                                break
                except Exception:
                    pass
                if cond:
                    handoff = True

            if handoff:
                # friendly, concise handoff confirmation
                outgoing = "Please hold — connecting you to a support agent now. I'll include a short summary of this conversation so they can help you faster."
                meta = meta or {}
                meta['handoff'] = True
                try:
                    if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and session_id:
                        from core.memory.memory_service import ConversationMemory
                        mem = ConversationMemory()
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
            # If handoff detection fails, continue normal flow
            pass

        # Save assistant response to memory (best-effort)
        try:
            from config.settings import settings
            if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and session_id:
                try:
                    from core.memory.memory_service import ConversationMemory
                    mem = ConversationMemory()
                    mem.save_message(session_id, "assistant", answer_text)
                except Exception:
                    pass
        except Exception:
            pass

        # Record decided language in meta for observability/analytics
        try:
            meta = meta or {}
            meta.setdefault('language', language)
        except Exception:
            pass

        confidence = float(meta.get("confidence", 0.0))
        citations = meta.get("citations", [])
        intent = meta.get("intent")
        if intent:
            logger.debug(f"Detected intent for {user_id}: {intent}")
        # If confidence is low, append fallback guidance while still answering using LLM output
        try:
            from config.settings import settings
            threshold = getattr(settings, "ANSWER_CONFIDENCE_THRESHOLD", 0.55)
            fallback_note = getattr(settings, "FALLBACK_MESSAGE", "")
        except Exception:
            threshold = 0.55
            fallback_note = ""

        outgoing = answer_text

        if confidence < threshold:
            # If confidence is low, avoid hallucination: ask for a short clarification and include fallback note if configured
            clarify = (
                "I don't have enough information to be confident. Could you provide one brief detail (e.g., patient age, exact error message, or how long this has been happening)?"
            )

            # If the composer already asked for clarification, or the outgoing contains a clarifying phrase, or the answer includes KB citations, avoid appending an extra clarify prompt.
            already_asked = False
            try:
                if meta.get('low_confidence'):
                    already_asked = True
            except Exception:
                pass

            try:
                if meta.get('citations'):
                    # If there are citations, assume KB provided supporting context — don't force another clarification
                    already_asked = True
            except Exception:
                pass

            try:
                if re.search(r"could you provide|please provide|i don't have enough|one brief detail", outgoing, re.I):
                    already_asked = True
            except Exception:
                pass

            # Prevent repeating the same clarify prompt multiple times by checking recent assistant messages
            try:
                from config.settings import settings
                if getattr(settings, "ENABLE_CONVERSATION_MEMORY", False) and session_id:
                    from core.memory.memory_service import ConversationMemory
                    mem = ConversationMemory()
                    recent_msgs = mem.get_recent(session_id, limit=6)
                    for m in recent_msgs:
                        if m.get('role') == 'assistant' and re.search(r"could you provide|please provide|i don't have enough|one brief detail", m.get('text',''), re.I):
                            already_asked = True
                            break
            except Exception:
                pass

            if not already_asked:
                if fallback_note and fallback_note not in outgoing:
                    outgoing = f"{outgoing}\n\n{fallback_note}\n\n{clarify}"
                else:
                    if clarify not in outgoing:
                        outgoing = f"{outgoing}\n\n{clarify}"

        # Clean, limit and safely truncate outgoing message for WhatsApp
        try:
            outgoing = self._format_outgoing(outgoing)
            # If the outgoing is a short numbered list (1 or 2 items), flatten to plain sentences
            try:
                lines = [l.strip() for l in outgoing.splitlines() if l.strip()]
                if lines and re.match(r"^\d+\.\s+", lines[0]) and len(lines) <= 2 and len(outgoing) < 200:
                    # remove numeric prefixes
                    flattened = " ".join([re.sub(r"^\d+\.\s+", "", l) for l in lines]).strip()
                    # ensure it's punctuation-terminated
                    if flattened and not flattened.endswith(('.', '!', '?')):
                        flattened = flattened + '.'
                    outgoing = flattened
            except Exception:
                pass
        except Exception:
            # best-effort fallback
            try:
                if len(outgoing) > 800:
                    outgoing = outgoing[:800]
            except Exception:
                pass

        # Send message and log metadata for observability
        try:
            # send_message supports optional media via a media argument; here we send the final text
            self.whatsapp.send_message(user_id, message=outgoing)
            logger.info(f"Sent reply to {user_id} | confidence={confidence:.3f} | citations={citations}")
        except Exception as e:
            logger.error(f"Failed sending message to {user_id}: {e}")

        # Return result & meta for testability/observability
        return outgoing, meta

