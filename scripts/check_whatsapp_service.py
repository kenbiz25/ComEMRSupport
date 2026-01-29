import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import httpx
from core.whatsapp.whatsapp_service import WhatsAppService

class DummyResponse:
    def __init__(self, status_code=200, text='OK'):
        self.status_code = status_code
        self.text = text
    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError('error', request=None, response=self)

# Test 1: transient errors then success
calls = {'count': 0}
def fake_post_1(url, json=None, headers=None, timeout=None):
    calls['count'] += 1
    if calls['count'] < 3:
        raise httpx.RequestError('network down')
    return DummyResponse(200, 'OK')

httpx.post = fake_post_1
svc = WhatsAppService('123', 'token', max_retries=4, backoff=0.01)
res = svc.send_message('+100', 'hello')
print('test1', res)
assert res['ok'] is True
assert calls['count'] == 3

# Test 2: permanent failure

def fake_post_2(url, json=None, headers=None, timeout=None):
    return DummyResponse(400, 'Bad Request')

httpx.post = fake_post_2
svc = WhatsAppService('123', 'token', max_retries=2, backoff=0.01)
res = svc.send_message('+100', 'hello')
print('test2', res)
assert res['ok'] is False
assert 'error' in res

print('whatsapp checks ok')
