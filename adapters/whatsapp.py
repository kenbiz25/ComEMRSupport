
# whatsapp.py
"""
WhatsApp Cloud API adapter (Meta) for sending messages.
- Defaults to API version v22.0 (matching your working curl).
- Reads META_WHATSAPP_TOKEN and WHATSAPP_PHONE_ID from environment.
- Adds send_text, send_template, and send_media helpers.
- Secure logging: only prints token prefix on errors, never the full token.
"""

import os
import requests
from typing import Any, Dict, List, Optional

# ---- Environment ----
META_TOKEN: str = os.getenv("META_WHATSAPP_TOKEN", "")
PHONE_ID: str   = os.getenv("WHATSAPP_PHONE_ID", "")
API_VER: str    = os.getenv("WHATSAPP_API_VERSION", "v22.0")  # default to v22.0

# ---- Internal helpers ----
def _ensure_env() -> None:
    if not META_TOKEN or not PHONE_ID:
        raise RuntimeError(
            "Missing META_WHATSAPP_TOKEN or WHATSAPP_PHONE_ID in environment variables."
        )

def _messages_url() -> str:
    _ensure_env()
    return f"https://graph.facebook.com/{API_VER}/{PHONE_ID}/messages"

def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {META_TOKEN}",
        "Content-Type": "application/json",
    }

def _raise_for_status(resp: requests.Response) -> None:
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Avoid leaking full token
        auth_header = resp.request.headers.get("Authorization", "")
        print(f"ERROR Failed to send WhatsApp message: {e}")
        print("Request URL:", resp.request.url)
        print("Auth header prefix:", (auth_header[:18] + "...") if auth_header else "missing")
        print("Response body:", resp.text)
        raise

# ---- Public API ----
def send_whatsapp_text(to: str, message: str, timeout_sec: int = 20) -> Dict[str, Any]:
    """
    Sends a plain text message to a WhatsApp user via Meta Cloud API.

    Parameters
    ----------
    to : str
        Recipient number in international/E.164 format (e.g., "254705091683").
    message : str
        Message text to send.
    timeout_sec : int
        Request timeout (seconds).

    Returns
    -------
    dict
        JSON response from Meta API or raises on HTTP error.
    """
    _ensure_env()
    url = _messages_url()
    payload: Dict[str, Any] = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message},
    }
    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout_sec)
    _raise_for_status(resp)
    return resp.json()

def send_whatsapp_template(
    to: str,
    template_name: str,
    language_code: str = "en_US",
    components: Optional[List[Dict[str, Any]]] = None,
    timeout_sec: int = 20,
) -> Dict[str, Any]:
    """
    Sends a template message (e.g., 'hello_world').

    components example:
    [{"type":"body","parameters":[{"type":"text","text":"Keneth"}]}]
    """
    _ensure_env()
    url = _messages_url()
    payload: Dict[str, Any] = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": language_code},
        },
    }
    if components:
        payload["template"]["components"] = components

    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout_sec)
    _raise_for_status(resp)
    return resp.json()

def send_whatsapp_media(
    to: str,
    media_type: str,            # "image" | "document" | "audio" | "video"
    link: str,
    caption: Optional[str] = None,
    filename: Optional[str] = None,  # for document/PDF
    timeout_sec: int = 20,
) -> Dict[str, Any]:
    """
    Sends a media message (image/PDF/audio/video) using public link.

    For PDFs, use media_type="document" and include filename="Guide.pdf".
    """
    _ensure_env()
    if media_type not in {"image", "document", "audio", "video"}:
        raise ValueError("Unsupported media_type. Use image|document|audio|video.")

    url = _messages_url()
    media_obj: Dict[str, Any] = {"link": link}
    if caption:
        media_obj["caption"] = caption
    if filename and media_type == "document":
        media_obj["filename"] = filename

    payload: Dict[str, Any] = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": media_type,
        media_type: media_obj,
    }
    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout_sec)
    _raise_for_status(resp)
    return resp.json()
