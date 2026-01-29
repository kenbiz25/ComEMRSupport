import pytest
import httpx
from core.whatsapp.whatsapp_service import WhatsAppService

class DummyResponse:
    def __init__(self, status_code=200, text='OK'):
        self.status_code = status_code
        self.text = text
    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError('error', request=None, response=self)


def test_send_retries_and_succeeds(monkeypatch):
    calls = {'count': 0}
    def fake_post(url, json=None, headers=None, timeout=None):
        calls['count'] += 1
        if calls['count'] < 3:
            # simulate transient network error
            raise httpx.RequestError('network down')
        return DummyResponse(200, 'OK')

    monkeypatch.setattr(httpx, 'post', fake_post)
    svc = WhatsAppService('123', 'token', max_retries=4, backoff=0.01)
    res = svc.send_message('+100', 'hello')
    assert res['ok'] is True
    assert calls['count'] == 3


def test_send_permanent_failure(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return DummyResponse(400, 'Bad Request')
    monkeypatch.setattr(httpx, 'post', fake_post)
    svc = WhatsAppService('123', 'token', max_retries=2, backoff=0.01)
    res = svc.send_message('+100', 'hello')
    assert res['ok'] is False
    assert 'error' in res
