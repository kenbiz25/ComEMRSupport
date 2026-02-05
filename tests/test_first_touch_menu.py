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
    def answer(self, query, language=None, session_id=None):
        return "Composer reply", {"confidence": 0.9, "citations": []}


def test_menu_shown_on_first_interaction(tmp_path, monkeypatch):
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
    session_id = "sess-menu-1"

    out, meta = flow.handle_message(user_id, "Hello", session_id=session_id)

    # Should have returned the menu and meta indicating it
    assert meta.get('menu_shown') is True
    assert out is not None
    assert 'Report a System Issue' in out

    # Session file should include the assistant menu and a system marker
    path = os.path.join(str(tmp_path / "sessions"), f"{session_id}.jsonl")
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as f:
        lines = [json.loads(l) for l in f if l.strip()]
    roles = [l['role'] for l in lines]
    assert 'assistant' in roles
    assert 'system' in roles


def test_menu_selection_response(tmp_path, monkeypatch):
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
    session_id = "sess-menu-2"

    # First, simulate previous menu shown
    from core.memory.memory_service import ConversationMemory
    mem = ConversationMemory()
    menu_text = (
        "Welcome to ComEMR Support! Please choose an option below:\n"
        "1. Report a System Issue\n   App not working, login errors, sync issues\n"
        "2. Get Help Using ComEMR\n   How to register patients, submit reports, use features\n"
        "3. Check System Status\n   Downtime, known issues, maintenance updates\n"
        "4. Request a Feature or Improvement\n   Suggest changes or new functionality\n"
        "5. Training & User Guides\n   Manuals, videos, onboarding support"
    )
    mem.save_message(session_id, "assistant", menu_text)
    mem.save_message(session_id, "system", "menu_shown")

    # User selects '1' to report an issue
    out, meta = flow.handle_message(user_id, "1", session_id=session_id)

    assert meta.get('menu_selected') == '1'
    assert 'describe the problem' in out.lower() or 'please briefly describe' in out.lower()
    # Session should have assistant reply and menu_consumed marker
    path = os.path.join(str(tmp_path / "sessions"), f"{session_id}.jsonl")
    with open(path, 'r', encoding='utf-8') as f:
        lines = [json.loads(l) for l in f if l.strip()]
    texts = [l['text'] for l in lines]
    assert any('menu_consumed' in t for t in texts) or any(l['role']=='assistant' and 'describe' in l['text'].lower() for l in lines)
