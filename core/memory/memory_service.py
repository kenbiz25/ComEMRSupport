# core/memory/memory_service.py
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Keep this lightweight and dependency-free; we summarize using the LLM adapter if available
class ConversationMemory:
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.join(os.getcwd(), "core", "memory", "sessions")
        os.makedirs(self.base_dir, exist_ok=True)

    def _session_path(self, session_id: str) -> str:
        safe = str(session_id).replace('/', '_')
        return os.path.join(self.base_dir, f"{safe}.jsonl")

    def save_message(self, session_id: str, role: str, text: str, ts: Optional[str] = None) -> None:
        try:
            path = self._session_path(session_id)
            entry = {"ts": ts or datetime.utcnow().isoformat() + "Z", "role": role, "text": text}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Optionally add to vector memory for semantic recall
            try:
                from config.settings import settings
                if getattr(settings, "ENABLE_VECTOR_MEMORY", False):
                    try:
                        from core.memory.vector_memory import VectorMemory
                        vec = VectorMemory(base_dir=self.base_dir)
                        vec.add_entry(session_id, text=text, role=role, ts=entry["ts"])
                    except Exception:
                        logger.exception("Failed to add to vector memory")
            except Exception:
                pass
        except Exception:
            logger.exception("Failed to save conversation message")

    def get_recent(self, session_id: str, limit: int = 10) -> List[Dict]:
        path = self._session_path(session_id)
        if not os.path.exists(path):
            return []
        out = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            logger.exception("Failed to read session file")
            return []
        # return last N
        return out[-limit:]

    def summarize(self, session_id: str, force: bool = False) -> str:
        """Produce or return a short summary for the session using the LLM if available.
        Falls back to a naive join of the most recent messages when LLM is not available.
        """
        try:
            msgs = self.get_recent(session_id, limit=20)
            if not msgs:
                return ""
            # Compose a compact conversation log
            log = "\n".join([f"{m['role']}: {m['text']}" for m in msgs])
            # Attempt to use LLM summarization helper
            try:
                from adapters.llm.openai_client import chat_complete
                prompt = (
                    "Summarize the following conversation in 2-3 short bullet points for a support agent. "
                    "Keep the summary simple and usable as context for future answers. Do not add extra information.\n\n"
                    f"Conversation:\n{log}\n\nSummary:")
                resp = chat_complete(prompt, max_tokens=120)
                return resp.strip()
            except Exception:
                # LLM unavailable - return compact joined recent user messages
                users = [m['text'] for m in msgs if m.get('role') == 'user']
                return ('; '.join(users[-4:]))[:500]
        except Exception:
            logger.exception("Failed to summarize conversation")
            return ""

    def clear_session(self, session_id: str) -> bool:
        try:
            path = self._session_path(session_id)
            if os.path.exists(path):
                os.remove(path)
            return True
        except Exception:
            logger.exception("Failed to clear session")
            return False
