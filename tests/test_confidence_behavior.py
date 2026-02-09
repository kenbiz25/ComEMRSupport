from rag.flow import BotFlow


class DummyWhatsApp:
    def __init__(self):
        self.sent = []

    def send_message(self, user_id, message=None):
        self.sent.append((user_id, message))


class DummyComposerLowClarify:
    def answer(self, query, language=None, session_id=None):
        return ("I can't answer that confidently. Could you provide the patient's age?", {"confidence": 0.1, "citations": []})


class DummyComposerLowNoClarify:
    def answer(self, query, language=None, session_id=None):
        return ("I think this might work, not certain.", {"confidence": 0.1, "citations": []})


def test_flow_does_not_duplicate_clarify_when_composer_already_asks():
    wa = DummyWhatsApp()
    composer = DummyComposerLowClarify()
    bf = BotFlow(None, None, wa, composer=composer)
    bf.handle_message("user-10", "How to X?")
    # Should not append another clarify sentence
    assert sum(1 for _, m in wa.sent if "could you provide" in (m or "").lower()) == 1


def test_flow_appends_clarify_when_needed():
    wa = DummyWhatsApp()
    composer = DummyComposerLowNoClarify()
    bf = BotFlow(None, None, wa, composer=composer)
    bf.handle_message("user-11", "How to X?")
    assert any("i don't have enough information" in (m or "").lower() for _, m in wa.sent)