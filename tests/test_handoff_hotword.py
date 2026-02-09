from rag.flow import BotFlow


class DummyWhatsApp:
    def __init__(self):
        self.sent = []

    def send_message(self, user_id, message=None):
        self.sent.append((user_id, message))


class DummyComposerLow:
    # composer returns low confidence and asked clarify previously
    def answer(self, query, language=None, session_id=None):
        return ("I don't have enough information. Could you provide the patient's age?", {"confidence": 0.1, "citations": []})


class DummyComposerHigh:
    # composer returns high-confidence answer
    def answer(self, query, language=None, session_id=None):
        return ("All good. Follow these steps.", {"confidence": 0.9, "citations": []})


def test_handoff_when_options_exhausted(tmp_path, monkeypatch):
    from core.memory import memory_service

    # use tmp sessions dir
    orig_init = memory_service.ConversationMemory.__init__

    def fake_init(self, base_dir=None):
        return orig_init(self, base_dir=str(tmp_path / "sessions"))

    monkeypatch.setattr(memory_service.ConversationMemory, "__init__", fake_init)

    wa = DummyWhatsApp()
    composer = DummyComposerLow()
    bf = BotFlow(None, None, wa, composer=composer)

    session_id = "sess-handoff-1"
    # Simulate previous assistant clarification
    from core.memory.memory_service import ConversationMemory
    mem = ConversationMemory()
    mem.save_message(session_id, "assistant", "I don't have enough information. Could you provide the patient's age?")
    mem.save_message(session_id, "user", "I tried, still not working")

    out, meta = bf.handle_message("+100", "talk to support", session_id=session_id)
    assert meta.get('handoff') is True
    assert 'connecting you to a support agent' in out.lower()


def test_no_handoff_if_not_exhausted(tmp_path, monkeypatch):
    from core.memory import memory_service

    orig_init = memory_service.ConversationMemory.__init__

    def fake_init(self, base_dir=None):
        return orig_init(self, base_dir=str(tmp_path / "sessions"))

    monkeypatch.setattr(memory_service.ConversationMemory, "__init__", fake_init)

    wa = DummyWhatsApp()
    composer = DummyComposerHigh()
    bf = BotFlow(None, None, wa, composer=composer)

    session_id = "sess-handoff-2"
    out, meta = bf.handle_message("+200", "talk to support", session_id=session_id)
    # should not set handoff since nothing was exhausted
    assert not meta.get('handoff')
    # result should be normal composer output rather than handoff text
    assert 'connecting you to a support agent' not in out.lower()