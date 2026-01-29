from fastapi.testclient import TestClient
import app


def test_ask_endpoint_handles_llm_exceptions(monkeypatch):
    client = TestClient(app.app)

    def fake_generate_response(q, docs):
        raise RuntimeError("openai failure: leaked-key-abc123")

    # Patch the LLMService instance used by the app
    monkeypatch.setattr("app.llm_service.generate_response", fake_generate_response)

    res = client.post("/ask", params={"query": "Tell me about fever"})
    assert res.status_code == 500
    body = res.json()
    # FastAPI returns {'detail': 'Internal server error'} for our generic error
    assert isinstance(body, dict)
    assert body.get("detail") == "Internal server error"
    assert "leaked-key-abc123" not in str(body).lower()


def test_webhook_handles_handler_exceptions(monkeypatch):
    client = TestClient(app.app)

    def fake_handle_message(from_, body, session_id=None):
        raise RuntimeError("bot handler crashed: secret-token-567")

    # Patch the bot instance used by the app
    monkeypatch.setattr(app.bot, "handle_message", fake_handle_message)

    res = client.post("/whatsapp/webhook", params={"from": "+15551234567", "body": "hello"})
    assert res.status_code == 500
    body = res.json()
    assert body.get("detail") == "Internal server error"
    assert "secret-token-567" not in str(body).lower()
