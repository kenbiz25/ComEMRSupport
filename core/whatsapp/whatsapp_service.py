# core/whatsapp/whatsapp_service.py
import httpx
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class WhatsAppService:
    def __init__(self, phone_id: str, token: str, max_retries: int = 3, backoff: float = 0.5):
        self.phone_id = phone_id
        self.token = token
        api_version = getattr(settings, "WHATSAPP_API_VERSION", "v20.0")
        # base endpoints
        self.base_message_url = f"https://graph.facebook.com/{api_version}/{self.phone_id}/messages"
        self.base_media_url = f"https://graph.facebook.com/{api_version}/{self.phone_id}/media"
        self.max_retries = max_retries
        self.backoff = backoff

    def _make_headers(self, json_content: bool = True):
        headers = {"Authorization": f"Bearer {self.token}"}
        if json_content:
            headers["Content-Type"] = "application/json"
        return headers

    def _send_payload(self, payload: dict, url: str):
        headers = self._make_headers()
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Sending WhatsApp message attempt {attempt} to {payload.get('to')}")
                resp = httpx.post(url, json=payload, headers=headers, timeout=10.0)
                logger.info(f"WhatsApp API response status={resp.status_code} body={resp.text}")
                resp.raise_for_status()
                logger.info(f"Reply sent to {payload.get('to')}")
                return {"ok": True, "status_code": resp.status_code, "body": resp.json()}
            except httpx.RequestError as e:
                last_exc = e
                logger.warning(f"WhatsApp send request error (attempt {attempt}): {e}")
            except httpx.HTTPStatusError as e:
                last_exc = e
                try:
                    logger.error(f"WhatsApp HTTP error status={e.response.status_code} body={e.response.text}")
                except Exception:
                    logger.exception("WhatsApp HTTPStatusError without response body")
                if e.response.status_code == 429 and attempt < self.max_retries:
                    pass
                else:
                    break
            except Exception as e:
                last_exc = e
                logger.exception(f"Unexpected error sending WhatsApp message (attempt {attempt}): {e}")
            # backoff
            try:
                import time
                time.sleep(self.backoff * (2 ** (attempt - 1)))
            except Exception:
                pass

        logger.error(f"Failed to send WhatsApp message after {self.max_retries} attempts: {last_exc}")
        return {"ok": False, "error": str(last_exc)}

    def upload_media(self, file_bytes: bytes, filename: str, mime_type: str) -> dict:
        """Upload a media file to WhatsApp via the Graph API media endpoint and return media id.

        Returns: {"ok": True, "id": "<media_id>", "body": <response_json>} on success
                 {"ok": False, "error": "..."} on failure
        """
        headers = self._make_headers(json_content=False)
        files = {"file": (filename, file_bytes, mime_type)}
        data = {"messaging_product": "whatsapp"}
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Uploading media attempt {attempt} filename={filename}")
                resp = httpx.post(self.base_media_url, files=files, data=data, headers=headers, timeout=30.0)
                logger.info(f"WhatsApp media upload status={resp.status_code} body={resp.text}")
                resp.raise_for_status()
                body = resp.json()
                media_id = body.get("id")
                if not media_id:
                    return {"ok": False, "error": "no media id returned", "body": body}
                return {"ok": True, "id": media_id, "body": body}
            except httpx.RequestError as e:
                last_exc = e
                logger.warning(f"WhatsApp media upload request error (attempt {attempt}): {e}")
            except httpx.HTTPStatusError as e:
                last_exc = e
                try:
                    logger.error(f"WhatsApp media HTTP error status={e.response.status_code} body={e.response.text}")
                except Exception:
                    logger.exception("WhatsApp media HTTPStatusError without response body")
                if e.response.status_code == 429 and attempt < self.max_retries:
                    pass
                else:
                    break
            except Exception as e:
                last_exc = e
                logger.exception(f"Unexpected error uploading media (attempt {attempt}): {e}")
            try:
                import time
                time.sleep(self.backoff * (2 ** (attempt - 1)))
            except Exception:
                pass

        return {"ok": False, "error": str(last_exc)}

    def send_media_by_id(self, user_id: str, media_id: str, mtype: str, caption: str | None = None) -> dict:
        if mtype not in ("image", "audio", "video", "document"):
            return {"ok": False, "error": f"unsupported media type: {mtype}"}
        payload = {
            "messaging_product": "whatsapp",
            "to": user_id,
            "type": mtype,
            mtype: {"id": media_id},
        }
        if caption and mtype == "image":
            payload[mtype]["caption"] = caption
        return self._send_payload(payload, self.base_message_url)

    def send_message(self, user_id: str, message: str = None, media: list[dict] | None = None) -> dict:
        """Send a message or media to a WhatsApp user.

        message: optional text body to send
        media: optional list of media dicts, each with keys:
            - type: 'image'|'audio'|'video'|'document'
            - link: publicly accessible URL to the media OR
            - id: media id previously uploaded to WhatsApp
            - caption: optional caption (for images)

        If multiple media items are provided we send them one-by-one (WhatsApp API expects one media per request).
        """
        results = []

        # Send media items first (one request per item)
        if media:
            for item in media:
                mtype = item.get("type")
                link = item.get("link")
                media_id = item.get("id")
                caption = item.get("caption")
                if not mtype or (not link and not media_id):
                    logger.warning(f"Skipping invalid media item: {item}")
                    continue
                if mtype not in ("image", "audio", "video", "document"):
                    logger.warning(f"Unsupported media type: {mtype}")
                    continue
                if media_id:
                    results.append(self.send_media_by_id(user_id, media_id, mtype, caption))
                    continue
                # link path
                payload = {
                    "messaging_product": "whatsapp",
                    "to": user_id,
                    "type": mtype,
                    mtype: {"link": link},
                }
                if caption and mtype == "image":
                    payload[mtype]["caption"] = caption
                results.append(self._send_payload(payload, self.base_message_url))

        # Then send text message if provided
        if message:
            payload = {"messaging_product": "whatsapp", "to": user_id, "text": {"body": message}}
            results.append(self._send_payload(payload, self.base_message_url))

        # Return aggregated results in a backward-compatible way
        if not results:
            return {"ok": False, "error": "no message or media provided"}

        # Single result: return it directly (preserves prior shape used by tests/callers)
        if len(results) == 1 and isinstance(results[0], dict):
            return results[0]

        # Multiple results: if any succeeded, report ok with results array
        ok_any = any(r.get("ok") for r in results if isinstance(r, dict))
        if ok_any:
            return {"ok": True, "results": results}

        # All failed: include last error for compatibility
        last_err = None
        for r in reversed(results):
            if isinstance(r, dict) and not r.get("ok"):
                last_err = r.get("error")
                break
        return {"ok": False, "error": last_err, "results": results}
