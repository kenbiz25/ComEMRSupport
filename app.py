# app.py
import os
import time
import logging
import random
import re
import textwrap
import asyncio
from typing import Dict, List, Tuple, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse

from config.settings import settings
from rag.composer import RagComposer, _detect_intent_style

# -------------------------------------------------------------------
# App & logging
# -------------------------------------------------------------------
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whatsapp")

# -------------------------------------------------------------------
# Environment & config
# -------------------------------------------------------------------
API_VERSION = os.getenv("WHATSAPP_API_VERSION") or "v24.0"
ACCESS_TOKEN = (
    os.getenv("WHATSAPP_ACCESS_TOKEN")
    or os.getenv("WHATSAPP_TOKEN")
    or os.getenv("META_WHATSAPP_TOKEN")
)
PHONE_NUMBER_ID = (
    os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    or os.getenv("WHATSAPP_PHONE_ID")
    or os.getenv("PHONE_ID")
)
VERIFY_TOKEN = (
    os.getenv("WHATSAPP_VERIFY_TOKEN")
    or getattr(settings, "WHATSAPP_VERIFY_TOKEN", None)
)
WRAP_WIDTH = int(os.getenv("WHATSAPP_WRAP_WIDTH", "72"))

def _env_ok() -> Tuple[bool, Optional[str]]:
    if not ACCESS_TOKEN:
        return False, "Missing ACCESS_TOKEN"
    if not PHONE_NUMBER_ID:
        return False, "Missing PHONE_NUMBER_ID"
    if not VERIFY_TOKEN:
        return False, "Missing VERIFY_TOKEN"
    return True, None

ok, why = _env_ok()
if not ok:
    logger.warning("WhatsApp env not ready: %s", why)

# -------------------------------------------------------------------
# RAG composer
# -------------------------------------------------------------------
rag = RagComposer(
    llm_model=settings.LLM_MODEL,
    top_k=getattr(settings, "TOP_K", 3),
)

# -------------------------------------------------------------------
# Lightweight in-memory tracking (short-term nudges)
# -------------------------------------------------------------------
MEMORY: Dict[str, List[Dict[str, str]]] = {}
MAX_TURNS = 6
STOP_WORDS = {"ok", "okay", "thanks", "thank you", "bye", "alright", "sure", "not now"}
COURTESY_NUDGES = ["Want the next step?", "Need help with the next part?", "Should I continue?"]

def remember(user: str, role: str, content: str):
    MEMORY.setdefault(user, []).append({"role": role, "content": content, "ts": time.time()})
    MEMORY[user] = MEMORY[user][-MAX_TURNS:]

def clear_memory(user: str):
    MEMORY.pop(user, None)
    if hasattr(rag, "_memory"):
        rag._memory.clear_session(user)

# -------------------------------------------------------------------
# Response hygiene helpers
# -------------------------------------------------------------------
def sanitize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\*{1,3}", "", text)        # strip markdown asterisks
    text = re.sub(r"\n{3,}", "\n\n", text)     # collapse blank lines
    lines = text.strip().splitlines()
    text = "\n".join(lines[:12])               # keep it reasonable for WA
    text = text[:4096]                          # WA hard cap
    return text.strip()

def maybe_add_nudge(text: str) -> str:
    if random.random() < 0.25:
        return f"{text}\n\n{random.choice(COURTESY_NUDGES)}"
    return text

# -------------------------------------------------------------------
# WhatsApp formatting merged from composer
# -------------------------------------------------------------------
def format_for_whatsapp(text: str, user_query: Optional[str] = None) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    def heading_to_bold(line: str) -> str:
        m = re.match(r"^\s{0,3}(#{1,6})\s+(.*)$", line)
        return f"*{m.group(2).strip()}*" if m else line

    lines = [heading_to_bold(l) for l in text.split("\n")]
    for i, l in enumerate(lines):
        lines[i] = re.sub(r"^(\d+)[\.\)]\s*", r"\1. ", l)
        lines[i] = re.sub(r"^(\s*)[\*\-\•]\s*", r"\1• ", l)

    formatted = []
    for idx, l in enumerate(lines):
        formatted.append(l)
        if re.match(r"^\*.+\*$", l):
            nxt = lines[idx + 1] if idx + 1 < len(lines) else ""
            if nxt and not re.match(r"^\s*(•|\d+\.)\s+", nxt):
                formatted.append("")

    style = _detect_intent_style(user_query or "")
    final_lines = []

    if style == "steps":
        num = 1
        for l in text.split("\n"):
            if l.startswith("• "):
                final_lines.append(f"{num}. {l[2:].strip()}")
                num += 1
            else:
                final_lines.append(l)
        text = "\n".join(final_lines)
    elif style == "list":
        final_lines = []
        for l in text.split("\n"):
            if re.match(r"^\d+\.\s+", l):
                final_lines.append(f"• {l.split('.', 1)[1].strip()}")
            else:
                final_lines.append(l)
        text = "\n".join(final_lines)

    # Wrap long lines
    if WRAP_WIDTH > 0:
        wrapped = []
        for l in text.split("\n"):
            wrapped.extend(textwrap.wrap(l, width=WRAP_WIDTH, replace_whitespace=False) if len(l) > WRAP_WIDTH else [l])
        text = "\n".join(wrapped)

    if style == "steps" and not text.strip().endswith("?"):
        text += "\n\nNeed help with the next part?"

    return text.strip()

# -------------------------------------------------------------------
# AI answering logic with automatic summary context
# -------------------------------------------------------------------
def ai_answer(prompt: str, user: str) -> str:
    normalized = prompt.lower().strip()
    if any(word in normalized for word in STOP_WORDS):
        clear_memory(user)
        return "Alright 👍"

    try:
        # Compose answer using RAG + conversation memory
        answer, meta = rag.answer(prompt, session_id=user)
        if not answer:
            return "Could you clarify that?"

        # Format, sanitize, and include memory-aware context
        answer = sanitize_text(answer)
        answer = format_for_whatsapp(answer, user_query=prompt)
        if meta.get("strategy") == "exit":
            clear_memory(user)
            return answer

        answer = maybe_add_nudge(answer)
        remember(user, "assistant", answer)
        return answer

    except Exception:
        logger.exception("RAG answering failed")
        return "I’m having trouble right now. Please try again later."

# -------------------------------------------------------------------
# WhatsApp send helper
# -------------------------------------------------------------------
async def send_whatsapp_text(to: str, text: str) -> None:
    if not text:
        text = "…"  # avoid empty body 400

    base = f"https://graph.facebook.com/{API_VERSION}"
    url = f"{base}/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}}

    async with httpx.AsyncClient(timeout=15) as client:
        for attempt in range(1, 4):
            try:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code >= 400:
                    logger.error(
                        "WA send failed (try %s): %s %s | body=%s",
                        attempt, resp.status_code, resp.reason_phrase, resp.text
                    )
                    resp.raise_for_status()
                logger.info("WA send OK: %s", resp.text)
                return
            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                logger.warning("WA send exception (try %s): %s", attempt, e)
                if attempt == 3:
                    raise
                await asyncio.sleep(2 ** attempt)

# -------------------------------------------------------------------
# Health & config endpoints
# -------------------------------------------------------------------
@app.get("/healthz")
async def healthz():
    ok, why = _env_ok()
    return JSONResponse({
        "ok": ok,
        "why": why,
        "api_version": API_VERSION,
        "phone_number_id": PHONE_NUMBER_ID,
        "has_token": bool(ACCESS_TOKEN),
    }, status_code=200 if ok else 500)

# -------------------------------------------------------------------
# Webhook verification (GET)
# -------------------------------------------------------------------
@app.get("/whatsapp/webhook")
async def verify(hub_mode: str | None = None, hub_verify_token: str | None = None, hub_challenge: str | None = None):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge or "")
    return PlainTextResponse("Forbidden", status_code=403)

# -------------------------------------------------------------------
# Webhook receiver (POST)
# -------------------------------------------------------------------
@app.post("/whatsapp/webhook")
async def webhook(request: Request):
    payload = await request.json()
    try:
        entry = payload.get("entry", [{}])[0]
        change = entry.get("changes", [{}])[0]
        value = change.get("value", {})

        if value.get("statuses"):
            logger.info("Status update: %s", value["statuses"])
            return {"status": "ok"}

        messages = value.get("messages")
        if not messages:
            return {"status": "ignored"}

        for message in messages:
            user = message["from"]
            text = message.get("text", {}).get("body", "").strip()
            if not text:
                logger.info("Non-text message from %s ignored", user)
                continue

            logger.info("Incoming message from %s: %s", user, text)
            remember(user, "user", text)

            # Memory-aware answer
            reply = ai_answer(text, user)
            await send_whatsapp_text(user, reply)
            logger.info("Reply sent to %s", user)

        return {"status": "ok"}

    except httpx.HTTPStatusError as e:
        logger.exception("Webhook send failed with HTTP error")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=400)
    except Exception:
        logger.exception("Webhook processing failed")
        return JSONResponse({"status": "error"}, status_code=500)
