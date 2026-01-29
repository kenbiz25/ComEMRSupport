from adapters.llm import openai_client
from rag.composer import RagComposer

class DummyChat:
    def __init__(self):
        self.calls = []
    def __call__(self, prompt, *, model=None, temperature=None, max_tokens=None):
        self.calls.append({'prompt': prompt, 'model': model, 'temperature': temperature, 'max_tokens': max_tokens})
        return 'Short answer.'


def test_compose_calls_chat_complete_with_correct_args(monkeypatch):
    dummy = DummyChat()
    monkeypatch.setattr(openai_client, 'chat_complete', dummy)
    rc = RagComposer(llm_model='gpt-test')
    rc._compose_with_llm('How are?', [], low_confidence=False)
    assert len(dummy.calls) == 1
    c = dummy.calls[0]
    assert c['model'] == 'gpt-test'
    assert 'Do NOT hallucinate' not in c['prompt']


def test_compose_low_confidence_includes_clarify(monkeypatch):
    dummy = DummyChat()
    monkeypatch.setattr(openai_client, 'chat_complete', lambda *args, **kwargs: "I don't know. Could you provide more details?")
    rc = RagComposer(llm_model='gpt-test')
    ans = rc._compose_with_llm('What is X?', [], low_confidence=True)
    assert "Could you provide more details" in ans or "I don't know" in ans
