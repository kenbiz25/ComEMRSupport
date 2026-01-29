from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

r = client.post('/whatsapp/webhook', data={'from': '+254700000000', 'body': 'FormPing'})
print('status', r.status_code)
print('text', r.text)
