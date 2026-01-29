from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

class MockBot:
    def __init__(self):
        self.calls = []
    def handle_message(self, from_, body):
        self.calls.append((from_, body))


def test_whatsapp_webhook_top_level_messages():
    mock = MockBot()
    app.bot = mock
    payload = {"messages": [{"from": "+254700000000", "text": {"body": "Top"}}]}
    r = client.post("/whatsapp/webhook", json=payload)
    assert r.status_code == 200
    assert mock.calls == [("+254700000000", "Top")]


def test_whatsapp_webhook_unmatched_dumps_file(tmp_path, monkeypatch):
    # Ensure unmatched payload is dumped to logs/unmatched_webhooks.jsonl
    mock = MockBot()
    app.bot = mock
    logfile = tmp_path / "unmatched.jsonl"
    # Patch global path
    import app as appmod
    appmod._LOG_UNMATCHED = logfile

    r = client.post("/whatsapp/webhook", json={"foo": "bar"})
    assert r.status_code == 422
    # Check file written
    txt = logfile.read_text(encoding='utf-8')
    assert 'foo' in txt
