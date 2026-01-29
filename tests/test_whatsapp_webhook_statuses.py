# tests/test_whatsapp_webhook_statuses.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

class MockBot:
    def __init__(self):
        self.calls = []
    def handle_message(self, from_, body):
        self.calls.append((from_, body))


def test_whatsapp_webhook_statuses():
    mock = MockBot()
    app.bot = mock
    payload = {
        "entry": [
            {"changes": [{"value": {"statuses": [{"id": "msg_1", "status": "delivered"}]}}]}
        ]
    }
    r = client.post("/whatsapp/webhook", json=payload)
    assert r.status_code == 200
    assert r.json().get("status") == "acknowledged"
    # status-only payloads should not trigger bot.handle_message
    assert mock.calls == []
