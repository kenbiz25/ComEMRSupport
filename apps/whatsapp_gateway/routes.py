
from fastapi import APIRouter, Request, HTTPException, Query, Response
from config.settings import settings
from config.logging import get_logger
from config.rate_limit import allow
from apps.whatsapp_gateway.send import send_whatsapp_text
from rag.composer import RagComposer

router = APIRouter()
log = get_logger("whatsapp")

# Instantiate composer once; pass model + policy knobs
composer = RagComposer(
    llm_model=settings.LLM_MODEL,
    confidence_threshold=settings.ANSWER_CONFIDENCE_THRESHOLD,
    safeguard=settings.SAFEGUARD_ENABLE,
)

# --- Meta verification (GET) ---------------------------------------------------
@router.get("/webhook")
async def verify(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
):
    """
    Meta webhook verification:
    Respond with the raw hub.challenge string as text/plain when the verify token matches.
    Any other response will fail verification.
    """
    if hub_mode == "subscribe" and hub_verify_token == settings.META_VERIFY_TOKEN:
        return Response(content=(hub_challenge or ""), media_type="text/plain")
    raise HTTPException(status_code=403, detail="Verification failed")

# --- Inbound WhatsApp messages (POST) -----------------------------------------
@router.post("/webhook")
async def inbound(req: Request):
    """
    Handles inbound WhatsApp notifications:
      - Extracts message text from text or interactive payloads
      - Applies rate limiting
      - Composes an answer using RagComposer
      - Sends the reply via Meta WhatsApp API
    """
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    entry_list = body.get("entry") or []
    if not entry_list:
        return {"status": "ignored"}

    changes_list = entry_list[0].get("changes") or []
    if not changes_list:
        return {"status": "ignored"}

    value = changes_list[0].get("value", {}) or {}
    messages = value.get("messages") or []
    contacts = value.get("contacts") or []

    # Ignore delivery/read status updates
    if not messages:
        return {"status": "ignored"}

    msg = messages[0]
    from_number = msg.get("from") or (contacts[0].get("wa_id") if contacts else None)
    msg_type = msg.get("type", "text")

    # Extract text from supported message types
    text_body = ""
    if msg_type == "text":
        text_body = (msg.get("text") or {}).get("body", "") or ""
    elif msg_type == "interactive":
        interactive = msg.get("interactive") or {}
        text_body = (
            (interactive.get("list_reply") or {}).get("title")
            or (interactive.get("button_reply") or {}).get("title")
            or (interactive.get("button_reply") or {}).get("id")
            or ""
        )
    else:
        # Other types (image, audio, etc.) can be handled later
        text_body = ""

    if not from_number:
        log.warning("Inbound message missing 'from' number.")
        return {"status": "ignored"}

    # Rate-limit protection
    if not allow(from_number):
        try:
            send_whatsapp_text(from_number, "Too many requests. Please try again later.")
        except Exception as e:
            log.error(f"Rate-limit send failed: {e}")
        return {"status": "rate_limited"}

    # Compose answer using RAG
    try:
        answer, meta = composer.answer(text_body)
    except Exception as e:
        log.error(f"Composer error: {e}")
        answer = (
            "Sorry, an internal error occurred while preparing your answer. "
            "Please try again in a few moments."
        )
        meta = {"confidence": 0.0}

    # Confidence check and fallback message
    confidence = float(meta.get("confidence", 0.0) or 0.0)
    if confidence < settings.ANSWER_CONFIDENCE_THRESHOLD:
        answer = (
            "I’m not fully confident in the answer from the internal knowledge base. "
            "Here is an initial suggestion:\n"
            f"{answer}\n\n"
            "For official guidance, please contact Tech Support."
        )

    # apps/whatsapp_gateway/routes.py (inside webhook(), before sending)
    from adapters.whatsapp import send_whatsapp_text  # ensure explicit path
    print("[Webhook] using:", send_whatsapp_text.__module__, send_whatsapp_text.__name__)

    # Send reply back to the user
    try:
        send_whatsapp_text(from_number, answer)
        log.info(f"Reply sent to {from_number} | confidence={confidence:.2f}")
    except Exception as e:
        log.error(f"Failed to send WhatsApp message: {e}")
        return {"status": "send_error", "confidence": confidence}

    return {"status": "sent", "confidence": confidence}
