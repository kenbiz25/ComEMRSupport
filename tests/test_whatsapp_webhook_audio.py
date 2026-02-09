from fastapi.testclient import TestClient
from types import SimpleNamespace
import json

from app import app

client = TestClient(app)

class MockBot:
    def __init__(self):
        self.calls = []
    def handle_message(self, from_, body, session_id=None):
        # preserve signature compatibility
        self.calls.append((from_, body))


def test_whatsapp_webhook_audio_transcribed_success(monkeypatch):
    mock = MockBot()
    app.bot = mock

    payload = {
        "entry": [
            {"changes": [{"value": {"messages": [{"from": "+254700000000", "type": "audio", "audio": {"url": "https://example.com/audio.ogg", "mime_type": "audio/ogg"}}]}}]}
        ]
    }

    # Mock httpx.get to return a fake audio file
    def fake_get(url, timeout=15.0):
        class R:
            status_code = 200
            content = b"FAKE-AUDIO-BYTES"
            headers = {"content-type": "audio/ogg"}
            def raise_for_status(self):
                return None
        return R()

    monkeypatch.setattr("httpx.get", fake_get)

    # Mock the transcription function
    def fake_transcribe(path):
        return ("This is a transcribed voice note", 0.95, "en")

    monkeypatch.setattr("core.stt.whisper_client.transcribe_audio", fake_transcribe)

    r = client.post("/whatsapp/webhook", json=payload)
    assert r.status_code == 200
    assert mock.calls == [("+254700000000", "This is a transcribed voice note")]


def test_whatsapp_webhook_audio_transcribe_failure(monkeypatch):
    mock = MockBot()
    app.bot = mock

    payload = {
        "entry": [
            {"changes": [{"value": {"messages": [{"from": "+254700000001", "type": "audio", "audio": {"url": "https://example.com/audio2.ogg", "mime_type": "audio/ogg"}}]}}]}
        ]
    }

    # Mock httpx.get to return a fake audio file
    def fake_get(url, timeout=15.0):
        class R:
            status_code = 200
            content = b"FAKE-AUDIO-BYTES"
            headers = {"content-type": "audio/ogg"}
            def raise_for_status(self):
                return None
        return R()

    monkeypatch.setattr("httpx.get", fake_get)

    # Mock the transcription function to raise
    def fake_transcribe_fail(path):
        raise RuntimeError("transcription failed")

    monkeypatch.setattr("core.stt.whisper_client.transcribe_audio", fake_transcribe_fail)

    r = client.post("/whatsapp/webhook", json=payload)
    assert r.status_code == 200
    assert mock.calls == [("+254700000001", "[Audio received; transcription unavailable]")]


def test_whatsapp_webhook_audio_401_then_resolve_via_media_id(monkeypatch):
    mock = MockBot()
    app.bot = mock

    media_id = "880417654599234"
    payload = {
        "entry": [
            {"changes": [{"value": {"messages": [{"from": "+254705091683", "type": "audio", "audio": {"url": "https://lookaside.fbsbx.com/whatsapp_business/attachments/?mid=880417654599234&source=webhook", "mime_type": "audio/ogg; codecs=opus", "id": media_id}}]}}]}
        ]
    }

    # Mock httpx.get to return 401 for the lookaside URL, then return media metadata, then return actual audio bytes
    def fake_get(url, headers=None, timeout=15.0):
        class R401:
            status_code = 401
            headers = {}
            content = b""
            def raise_for_status(self):
                raise Exception("401 Unauthorized")
        class Rmeta:
            status_code = 200
            def raise_for_status(self):
                return None
            def json(self):
                return {"url": "https://example.com/resolved_audio.ogg"}
        class Raudio:
            status_code = 200
            content = b"REAL-AUDIO-BYTES"
            headers = {"content-type": "audio/ogg"}
            def raise_for_status(self):
                return None
        if "lookaside.fbsbx.com" in url:
            return R401()
        if url.endswith(media_id):
            return Rmeta()
        if "resolved_audio.ogg" in url:
            return Raudio()
        # default
        return R401()

    monkeypatch.setattr("httpx.get", fake_get)

    # Mock transcribe to return a phrase
    def fake_transcribe(path):
        return ("Transcribed after resolving media id", 0.9, "en")

    monkeypatch.setattr("core.stt.whisper_client.transcribe_audio", fake_transcribe)

    r = client.post("/whatsapp/webhook", json=payload)
    assert r.status_code == 200
    assert mock.calls == [("+254705091683", "Transcribed after resolving media id")]
