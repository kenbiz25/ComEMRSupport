from types import SimpleNamespace

from rag.flow import BotFlow

class MockWhatsApp:
    def __init__(self):
        self.sent = []
    def send_message(self, to, message=None, media=None):
        self.sent.append((to, message))

class MockComposer:
    def answer(self, query, language='en', session_id=None):
        return ("Please try restarting your device and clearing the app cache.", {"confidence": 0.8})

class MockMem:
    def __init__(self):
        self._messages = []
    def get_recent(self, session_id, limit=10):
        return list(self._messages[-limit:])
    def save_message(self, session_id, role, text):
        self._messages.append({"role": role, "text": text})


def test_close_and_ignore_followups(monkeypatch):
    # Enable conversation memory
    monkeypatch.setenv('ENABLE_CONVERSATION_MEMORY', '1')
    # Monkeypatch settings read to return True for ENABLE_CONVERSATION_MEMORY
    class S:
        ENABLE_CONVERSATION_MEMORY = True
        ENABLE_FIRST_TOUCH_MENU = False
    monkeypatch.setattr('config.settings.settings', S)

    # Replace ConversationMemory with our MockMem
    mock_mem = MockMem()
    monkeypatch.setattr('core.memory.memory_service.ConversationMemory', lambda: mock_mem)

    whatsapp = MockWhatsApp()
    # BotFlow requires faiss and llm but we won't use them in this test
    bot = BotFlow(faiss_store=None, llm_service=None, whatsapp_service=whatsapp, composer=MockComposer())

    # 1) Initial user message -> normal reply
    out1, meta1 = bot.handle_message('+100', 'My phone is just loading when i open any page', session_id='+100')
    assert whatsapp.sent[-1][1].startswith('Please try restarting')

    # 2) User says 'Than you bye' -> closing reply
    out2, meta2 = bot.handle_message('+100', 'Than you bye', session_id='+100')
    assert meta2.get('conversation_closed') is True
    assert whatsapp.sent[-1][1].startswith("You're welcome")

    # 3) Later user sends 'Okay' -> should be ignored (no extra message sent)
    out3, meta3 = bot.handle_message('+100', 'Okay', session_id='+100')
    assert meta3.get('ignored') is True
    # Ensure only two messages were sent total
    assert len(whatsapp.sent) == 2