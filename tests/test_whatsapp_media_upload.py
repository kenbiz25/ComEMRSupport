# tests/test_whatsapp_media_upload.py
import httpx
from core.whatsapp.whatsapp_service import WhatsAppService

class DummyResp:
    def __init__(self, status_code=200, json_data=None, text='OK'):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text
    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError('err', request=None, response=self)
    def json(self):
        return self._json


def test_upload_media_success(monkeypatch):
    def fake_post(url, files=None, data=None, headers=None, timeout=None):
        assert 'media' in url
        return DummyResp(200, json_data={'id': 'media_123'})
    monkeypatch.setattr(httpx, 'post', fake_post)
    svc = WhatsAppService('123', 'tok', max_retries=1)
    res = svc.upload_media(b'binary', 'photo.jpg', 'image/jpeg')
    assert res['ok'] is True
    assert res['id'] == 'media_123'


def test_send_media_by_id(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        assert 'messages' in url
        return DummyResp(200, json_data={'messages': [{'id': 'm1'}]})
    monkeypatch.setattr(httpx, 'post', fake_post)
    svc = WhatsAppService('123', 'tok', max_retries=1)
    res = svc.send_media_by_id('+100', 'media_123', 'image', caption='hi')
    assert res['ok'] is True
    assert res['body']['messages'][0]['id'] == 'm1'
