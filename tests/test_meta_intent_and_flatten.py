# tests/test_meta_intent_and_flatten.py
from rag.composer import RagComposer
from rag.flow import BotFlow


def test_meta_intent_handler():
    rc = RagComposer()
    ans, meta = rc.answer('Are you a robot?')
    assert 'virtual assistant' in ans.lower() or 'i am' in ans.lower()
    assert meta.get('intent') == 'meta'


def test_flatten_short_numbered_list():
    bf = BotFlow(None, None, None)
    text = '1. Yes\n2. If needed, call support'
    out = bf._format_outgoing(text)
    # For short lists the output should be flattened into one or two sentences without numeric prefixes
    assert not out.lstrip().startswith('1.')
    assert 'Yes' in out


def test_handle_message_session_passes_session_id():
    composer_calls = {}
    class DummyComposer:
        def answer(self, query, language=None, session_id=None, **kwargs):
            composer_calls['sess'] = session_id
            return ('OK', {'confidence': 1.0, 'citations': [], 'intent': 'general'})
    class DummyWhatsapp:
        def send_message(self, user_id, message=None, media=None):
            return {'ok': True}

    comp = DummyComposer()
    bf = BotFlow(None, None, None, composer=comp)
    bf.whatsapp = DummyWhatsapp()
    bf.handle_message('+100', 'hello', session_id='+100')
    assert composer_calls.get('sess') == '+100'
