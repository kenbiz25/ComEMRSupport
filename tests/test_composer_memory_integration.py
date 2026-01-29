from rag.composer import RagComposer


def test_composer_includes_memory_in_prompt(monkeypatch):
    # Prepare composer
    composer = RagComposer()

    # Fake VectorMemory.get_similar to return a clear memory
    def fake_get_similar(session_id, query, top_k=3):
        return [{"text": "Patient: high fever yesterday", "score": 0.9, "role": "user", "ts": "ts1"}]

    monkeypatch.setattr("core.memory.vector_memory.VectorMemory.get_similar", fake_get_similar)

    # Capture prompt passed to chat_complete
    captured = {}

    def fake_chat_complete(prompt, **kwargs):
        captured["prompt"] = prompt
        return "OK"

    monkeypatch.setattr("adapters.llm.openai_client.chat_complete", fake_chat_complete)

    # Run composer: the internal chat_complete should receive a prompt including the memory text
    composer._compose_with_llm("What should I do?", context_chunks=[], low_confidence=False, language="en", session_id="sess-1")

    assert "Patient: high fever yesterday" in captured.get("prompt", "")
