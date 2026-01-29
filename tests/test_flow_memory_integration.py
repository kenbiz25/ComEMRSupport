import os
import json
from rag.flow import BotFlow


class DummyFaiss:
    dim = 1536
    def search(self, embedding, top_k=5):
        return []


class DummyLLM:
    def generate_response(self, query, docs, language='en'):
        return "Fallback answer"


class DummyWhatsApp:
    def __init__(self):
        self.sent = []

    def send_message(self, user_id, message=None):
        self.sent.append((user_id, message))


class SpyComposer:
    def __init__(self):
        self.received = None

    def answer(self, query, language=None, session_id=None):
        self.received = {
            'query': query,
            'language': language,
            'session_id': session_id,
        }
        return "Composer reply", {"confidence": 0.9, "citations": []}


def test_flow_saves_messages(tmp_path, monkeypatch):
    # arrange
    # monkeypatch ConversationMemory to write into tmp_path
    from core.memory import memory_service

    orig_init = memory_service.ConversationMemory.__init__

    def fake_init(self, base_dir=None):
        return orig_init(self, base_dir=str(tmp_path / "sessions"))

    monkeypatch.setattr(memory_service.ConversationMemory, "__init__", fake_init)

    faiss = DummyFaiss()
    llm = DummyLLM()
    whatsapp = DummyWhatsApp()
    composer = SpyComposer()

    flow = BotFlow(faiss, llm, whatsapp, composer=composer)
    user_id = "+123"
    session_id = "sess-42"

    # act
    out, meta = flow.handle_message(user_id, "Hello, I have fever", session_id=session_id)

    # assert that composer received session_id
    assert composer.received is not None
    assert composer.received['session_id'] == session_id

    # assert session file exists and contains two messages (user + assistant)
    path = os.path.join(str(tmp_path / "sessions"), f"{session_id}.jsonl")
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as f:
        lines = [json.loads(l) for l in f if l.strip()]
    # we should have at least two saved messages
    roles = [l['role'] for l in lines]
    assert 'user' in roles
    assert 'assistant' in roles
