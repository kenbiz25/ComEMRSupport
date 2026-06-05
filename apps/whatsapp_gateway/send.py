
import httpx
from config.settings import settings

def send_whatsapp_text(to_number: str, text: str):
    url = f"https://graph.facebook.com/v20.0/{settings.WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {settings.META_WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "text": {"body": text}
    }
    with httpx.Client(timeout=15) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()
