from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

class MockBot:
    def __init__(self):
        self.calls = []
    def handle_message(self, from_, body):
        self.calls.append((from_, body))


def test_whatsapp_webhook_json():
    mock = MockBot()
    app.bot = mock
    payload = {
        "entry": [
            {"changes": [{"value": {"messages": [{"from": "+254700000000", "text": {"body": "Hello JSON"}}]}}]}
        ]
    }
    r = client.post("/whatsapp/webhook", json=payload)
    assert r.status_code == 200
    assert mock.calls == [("+254700000000", "Hello JSON")]


def test_whatsapp_webhook_query_params():
    mock = MockBot()
    app.bot = mock
    r = client.post("/whatsapp/webhook?from=+254700000000&body=Ping")
    assert r.status_code == 200
    assert mock.calls == [("+254700000000", "Ping")]


def test_whatsapp_webhook_form():
    mock = MockBot()
    app.bot = mock
    r = client.post("/whatsapp/webhook", data={"from": "+254700000000", "body": "FormPing"})
    assert r.status_code == 200
    assert mock.calls == [("+254700000000", "FormPing")]
