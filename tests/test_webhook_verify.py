
from fastapi.testclient import TestClient
from app import app
from config.settings import settings

client = TestClient(app)

def test_verify_ok():
    r = client.get("/whatsapp/webhook", params={
        "mode": "subscribe",
        "challenge": "12345",
        "verify_token": settings.META_VERIFY_TOKEN
    })
    assert r.status_code == 200
    assert r.text == "12345"