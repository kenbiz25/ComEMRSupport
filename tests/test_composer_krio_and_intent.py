# tests/test_composer_krio_and_intent.py
from adapters.llm import openai_client
from rag.composer import RagComposer

class DummyChat:
    def __init__(self):
        self.calls = []
    def __call__(self, prompt, *, model=None, temperature=None, max_tokens=None):
        self.calls.append({'prompt': prompt, 'model': model})
        # return a deterministic short answer
        return 'Short answer.'


def test_compose_includes_krio_instruction(monkeypatch):
    dummy = DummyChat()
    monkeypatch.setattr(openai_client, 'chat_complete', dummy)
    rc = RagComposer(llm_model='gpt-test')
    rc._compose_with_llm('How are?', [], low_confidence=False, language='krio')
    assert any('Krio' in c['prompt'] or 'krio' in c['prompt'].lower() for c in dummy.calls)


def test_answer_returns_intent(monkeypatch):
    monkeypatch.setattr(openai_client, 'chat_complete', lambda *args, **kwargs: "Ok")
    rc = RagComposer()
    ans, meta = rc.answer('Hello there')
    assert meta.get('intent') == 'greeting'
