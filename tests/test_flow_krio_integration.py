# tests/test_flow_krio_integration.py
from rag.flow import BotFlow

class DummyComposer:
    def __init__(self):
        self.calls = []
    def answer(self, query, language=None, **kwargs):
        # Explicit language parameter ensures it's present when passed as kwarg
        self.calls.append({'query': query, 'language': language})
        return ('OK', {'confidence': 1.0, 'citations': [], 'intent': 'greeting'})


def test_flow_passes_language_to_composer():
    composer = DummyComposer()
    class DummyWhatsapp:
        def send_message(self, user_id, message=None, media=None):
            return {'ok': True}
    bf = BotFlow(None, None, None, composer=composer)
    bf.whatsapp = DummyWhatsapp()
    bf.handle_message('+100', 'krio: wetin de do')
    # composer.calls contains dicts with explicit 'language' key
    assert any(c.get('language') == 'krio' for c in composer.calls)
