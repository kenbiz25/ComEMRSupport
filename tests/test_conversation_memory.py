import os
import tempfile
from core.memory.memory_service import ConversationMemory


def test_save_and_get_recent(tmp_path):
    session_id = "test_user_1"
    base = tmp_path / "sessions"
    mem = ConversationMemory(base_dir=str(base))

    mem.save_message(session_id, "user", "Hello there")
    mem.save_message(session_id, "assistant", "Hi, how can I help?")
    recent = mem.get_recent(session_id, limit=5)
    assert len(recent) == 2
    assert recent[0]["role"] == "user"
    assert recent[1]["role"] == "assistant"


def test_summarize_with_llm(monkeypatch, tmp_path):
    # Create session and messages
    session_id = "test_user_2"
    base = tmp_path / "sessions"
    mem = ConversationMemory(base_dir=str(base))
    mem.save_message(session_id, "user", "Patient reports fever and cough for 2 days")
    mem.save_message(session_id, "assistant", "Do they have difficulty breathing?")

    # Patch chat_complete
    def fake_chat_complete(prompt, **kwargs):
        return "- Fever and cough for 2 days; asked about breathing."

    monkeypatch.setattr("adapters.llm.openai_client.chat_complete", fake_chat_complete)
    summary = mem.summarize(session_id)
    assert "Fever" in summary or "fever" in summary.lower()
