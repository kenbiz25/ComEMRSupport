
# app.py
import os
import json
from typing import Any, Dict
from collections import defaultdict, deque
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from rag.composer import RagComposer
from core.indexing.pipeline import IndexingPipeline
from config.settings import settings

# Optional: your existing WhatsApp adapter
try:
    from whatsapp import (
        send_whatsapp_text as adapter_send_text,
        send_whatsapp_template as adapter_send_template,
        # If you’ve already implemented buttons in your adapter, import it here:
        # send_whatsapp_buttons as adapter_send_buttons
    )
    HAS_WHATSAPP_ADAPTER = True
except Exception:
    HAS_WHATSAPP_ADAPTER = False

# Optional: your existing routes (kept)
try:
    from apps.whatsapp_gateway.routes import router as whatsapp_router
    HAS_WHATSAPP_ROUTER = True
except Exception:
    HAS_WHATSAPP_ROUTER = False

# --- Config / Env ---
API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v22.0")
VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "")
WA_TOKEN     = os.getenv("META_WHATSAPP_TOKEN", "")
PHONE_ID     = os.getenv("WHATSAPP_PHONE_ID", "")
BOT_BRAND    = os.getenv("BOT_BRAND", "ComEMR Support")
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")

# NEW: Teams handover config
TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL", "")
TEAMS_USE_ADAPTIVE_CARDS = os.getenv("TEAMS_USE_ADAPTIVE_CARDS", "false").lower() == "true"

GRAPH_URL = f"https://graph.facebook.com/{API_VERSION}/{PHONE_ID}/messages"

# --- App ---
app = FastAPI(title="ComEMR Support", version="1.0.0")

# CORS (tighten allowed origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

# Shared components
rag = RagComposer(
    llm_model=settings.LLM_MODEL,
    safeguard=settings.SAFEGUARD_ENABLE,
    top_k=getattr(settings, "TOP_K", 3),
)
indexer = IndexingPipeline()

# --- In-memory webhook debug buffer ---
WEBHOOK_BUFFER = deque(maxlen=50)  # recent payloads for inspection

# --- Conversation memory (rolling, per user) ---
# Stores a short transcript: [{"ts": "...Z", "role": "user|assistant", "content": "..."}]
MEMORY = defaultdict(lambda: deque(maxlen=12))

def remember(user: str, role: str, content: str):
    MEMORY[user].append({
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "role": role,
        "content": (content or "").strip()[:2000],
    })

def last_user_message(user: str) -> str:
    for turn in reversed(MEMORY[user]):
        if turn["role"] == "user":
            return turn["content"]
    return ""

def _gather_recent_turns(user: str, max_turns: int = 12) -> list:
    return list(MEMORY[user])[-max_turns:]

# --- Startup diagnostics (safe) ---
@app.on_event("startup")
def startup_diag():
    print("[Startup] WhatsApp config:")
    print("  API version:", API_VERSION)
    print("  TOKEN prefix:", (WA_TOKEN[:8] + "...") if WA_TOKEN else "MISSING")
    print("  PHONE_ID:", PHONE_ID if PHONE_ID else "MISSING")
    if not WA_TOKEN or not PHONE_ID:
        print("⚠️ META_WHATSAPP_TOKEN or WHATSAPP_PHONE_ID missing. Sending will fail.")
    if TEAMS_WEBHOOK_URL:
        print("[Startup] Teams webhook configured ✓")
    else:
        print("[Startup] TEAMS_WEBHOOK_URL not set (handover will be skipped).")

# --- Low-level WhatsApp send helpers (used if adapter is absent) ---
def _wa_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json",
    }

def _wa_post(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(GRAPH_URL, headers=_wa_headers(), data=json.dumps(payload), timeout=30)
    try:
        data = r.json()
    except Exception:
        data = {"text": r.text}
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail={"status": r.status_code, "response": data})
    return data

def send_text(to: str, body: str) -> Dict[str, Any]:
    if HAS_WHATSAPP_ADAPTER:
        # Use your adapter if present
        return adapter_send_text(to, body)
    # Fallback direct Cloud API call
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": body},
    }
    return _wa_post(payload)

def send_buttons(to: str, body_text: str) -> Dict[str, Any]:
    # If you have adapter_send_buttons, prefer it. Otherwise, use raw API:
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body_text},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": "TALK_SUPPORT",   "title": "TALK TO SUPPORT"}},
                    {"type": "reply", "reply": {"id": "RESET_PASSWORD", "title": "Reset password"}},
                    {"type": "reply", "reply": {"id": "SYSTEM_STATUS",  "title": "System status"}},
                ]
            },
        },
    }
    return _wa_post(payload)

# --- AI answer: try RAG -> OpenAI -> fallback text ---
def ai_answer(prompt: str, user: str) -> str:
    # Build short rolling context (system + last 6 turns + new user prompt) for OpenAI path
    context_turns = _gather_recent_turns(user, max_turns=6)
    convo = []
    for t in context_turns:
        role = "user" if t["role"] == "user" else "assistant"
        convo.append({"role": role, "content": t["content"]})

    # 1) Try your RAG first
    try:
        if hasattr(rag, "answer") and callable(getattr(rag, "answer")):
            ans = rag.answer(prompt)
        else:
            ans = rag(prompt)  # type: ignore
        if isinstance(ans, str) and ans.strip():
            remember(user, "assistant", ans)
            return ans.strip()
        if isinstance(ans, dict) and ans.get("answer"):
            out = str(ans["answer"]).strip()
            remember(user, "assistant", out)
            return out
    except Exception as e:
        print("[AI:RAG] Error:", e)

    # 2) Fallback to OpenAI if available
    if OPENAI_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_KEY)
            system_prompt = (
                "You are ComEMR Support Assistant. Be concise, helpful, and factual. "
                "If unsure, ask for details. Provide step-by-step guidance for ComEMR users."
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}, *convo, {"role": "user", "content": prompt}],
                temperature=0.2,
            )
            out = resp.choices[0].message.content.strip()
            remember(user, "assistant", out)
            return out
        except Exception as e:
            print("[AI:OpenAI] Error:", e)

    # 3) Final fallback text
    fallback = "I’m having trouble reaching the AI right now. Please try again, or tap TALK TO SUPPORT."
    remember(user, "assistant", fallback)
    return fallback

# --- Conversation summarizer (AI) ---
def summarize_conversation(user: str) -> str:
    """
    Creates a short, human‑friendly summary from MEMORY.
    Output: 4–7 bullet points with problem, attempts, errors, and current ask.
    """
    turns = _gather_recent_turns(user, max_turns=12)
    if not turns:
        return "No conversation content available."

    convo_lines = []
    for t in turns:
        role = "User" if t["role"] == "user" else "Bot"
        convo_lines.append(f"{role}: {t['content']}")

    prompt = (
        "Summarize the following WhatsApp support conversation in 4–7 concise bullet points. "
        "Include: the user's problem, steps already tried, any error messages, device/app context if present, "
        "and the user's latest request. Use plain language:\n\n"
        + "\n".join(convo_lines)
    )
    summary = ai_answer(prompt, user)

    # Simple deterministic fallback if AI returns the standard failure message
    if summary.lower().startswith("i’m having trouble reaching the ai"):
        # Build a lightweight summary from last few user turns
        bullets = []
        for t in turns:
            if t["role"] == "user":
                bullets.append(f"- {t['content']}")
        summary = "Summary (fallback):\n" + "\n".join(bullets[-6:]) if bullets else "Summary unavailable."
    return summary

# --- Teams payload helpers ---
def _post_to_teams(payload: dict):
    if not TEAMS_WEBHOOK_URL:
        print("⚠️ TEAMS_WEBHOOK_URL missing")
        return
    try:
        r = requests.post(TEAMS_WEBHOOK_URL, json=payload, timeout=15)
        r.raise_for_status()
        print("[Handover] Summary posted to Teams")
    except Exception as e:
        print("[Handover] Teams post failed:", e)

def _teams_text_payload(user_e164: str, profile_name: str, summary: str, last_msg: str) -> dict:
    who = f"+{user_e164}" if not (user_e164 or "").startswith("+") else user_e164
    display = f"{profile_name} ({who})" if profile_name else who
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    text = (
        f"**WhatsApp → Human Escalation**\n\n"
        f"**User:** {display}\n"
        f"**Last message:** {last_msg or '_none_'}\n\n"
        f"**Summary:**\n{summary}\n\n"
        f"**Time:** {ts}"
    )
    return {"text": text}

def _teams_adaptive_card_payload(user_e164: str, profile_name: str, summary: str, last_msg: str) -> dict:
    who = f"+{user_e164}" if not (user_e164 or "").startswith("+") else user_e164
    display = f"{profile_name} ({who})" if profile_name else who
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    card = {
      "type": "message",
      "attachments": [{
        "contentType": "application/vnd.microsoft.card.adaptive",
        "content": {
          "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
          "type": "AdaptiveCard",
          "version": "1.5",
          "body": [
            {"type":"TextBlock","text":"WhatsApp → Human Escalation","weight":"Bolder","size":"Large"},
            {"type":"FactSet","facts":[
              {"title":"User","value": display},
              {"title":"Last message","value": (last_msg or "_none_")},
              {"title":"Time","value": ts}
            ]},
            {"type":"TextBlock","text":"Summary","weight":"Bolder","spacing":"Medium"},
            {"type":"TextBlock","text": summary[:4500], "wrap": True}
          ]
        }
      }]
    }
    return card

def send_summary_to_teams(user_e164: str, profile_name: str = ""):
    """
    Builds a summary using AI and sends to Teams (text or Adaptive Card).
    """
    summary = summarize_conversation(user_e164)
    last_msg = last_user_message(user_e164)

    payload = (
        _teams_adaptive_card_payload(user_e164, profile_name, summary, last_msg)
        if TEAMS_USE_ADAPTIVE_CARDS else
        _teams_text_payload(user_e164, profile_name, summary, last_msg)
    )
    _post_to_teams(payload)

# --- Health check ---
@app.get("/health")
def health():
    return {"status": "ok"}

# --- Optional: trigger KB reindex (protect in prod) ---
@app.post("/kb/reindex")
def reindex():
    count = indexer.reindex_all()
    return {"reindexed": count}

# --- Webhook verification (GET) ---
@app.get("/webhook")
def verify(
    hub_mode: str = Query(default=None, alias="hub.mode"),
    hub_verify_token: str = Query(default=None, alias="hub.verify_token"),
    hub_challenge: str = Query(default=None, alias="hub.challenge"),
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        # Meta requires the challenge to be returned verbatim with 200
        return hub_challenge
    raise HTTPException(status_code=403, detail="Forbidden")

# --- Webhook receive (POST) ---
@app.post("/webhook")
async def incoming(request: Request):
    data = await request.json()
    WEBHOOK_BUFFER.appendleft(data)  # capture for debugging

    entries = data.get("entry") or []
    if not entries:
        return {"status": "ignored_no_entry"}

    value = (entries[0].get("changes") or [{}])[0].get("value", {})

    # Extract contact display name when provided
    contacts = value.get("contacts") or []
    profile_name = ""
    if contacts and isinstance(contacts, list):
        profile_name = ((contacts[0].get("profile") or {}).get("name") or "").strip()

    # 1) Delivery/read statuses (debug-friendly)
    if "statuses" in value:
        for s in value.get("statuses", []):
            print(
                "[WA:Status]",
                "id=", s.get("id"),
                "status=", s.get("status"),
                "timestamp=", s.get("timestamp"),
                "recipient_id=", s.get("recipient_id"),
                "errors=", s.get("errors"),
            )
        return {"status": "ok_status"}

    # 2) Incoming messages we care about
    messages = value.get("messages", [])
    if not messages:
        return {"status": "ok_no_message"}

    msg = messages[0]
    from_e164 = msg.get("from")  # E.164 without +
    msg_type = msg.get("type")

    # a) Text → remember + Welcome + buttons
    if msg_type == "text":
        user_text = (msg.get("text") or {}).get("body", "").strip()
        if user_text:
            remember(from_e164, "user", user_text)

        welcome = f"Karibu! 👋 This is {BOT_BRAND}.\nPick an option below or ask your question:"
        try:
            send_buttons(from_e164, welcome)
        except Exception as e:
            # If buttons fail for any reason, send plain text fallback
            print("[Buttons] Error:", e)
            send_text(from_e164, welcome + "\n\n• TALK TO SUPPORT\n• Reset password\n• System status")
        return {"status": "ok_welcome"}

    # b) Interactive button replies
    if msg_type == "interactive":
        interactive = msg.get("interactive") or {}
        subtype = interactive.get("type")
        if subtype == "button_reply":
            reply = interactive.get("button_reply") or {}
            btn_id = reply.get("id")

            # Remember the user's selection as part of the transcript
            if btn_id:
                remember(from_e164, "user", f"[Button] {btn_id}")

            if btn_id == "TALK_SUPPORT":
                send_text(from_e164, "Okay, summarizing your issue and connecting you to a human agent…")
                # NEW: Summarize and send to Teams
                send_summary_to_teams(from_e164, profile_name)
                return {"status": "ok_escalated"}

            elif btn_id == "RESET_PASSWORD":
                send_text(
                    from_e164,
                    "To reset your ComEMR password:\n"
                    "1) Open ComEMR app → Login screen → 'Forgot password'\n"
                    "2) Enter phone number used for your ComEMR account\n"
                    "3) Check WhatsApp/SMS for OTP and follow prompts\n\n"
                    "Need a human? Tap TALK TO SUPPORT."
                )
                return {"status": "ok_reset_pw"}

            elif btn_id == "SYSTEM_STATUS":
                # Placeholder; wire to real status endpoint when ready
                send_text(from_e164, "All systems operational ✅\nNo active incidents reported.")
                return {"status": "ok_status"}

        elif subtype == "list_reply":
            list_reply = interactive.get("list_reply") or {}
            choice_title = list_reply.get("title") or ""
            remember(from_e164, "user", f"[List] {choice_title}")
            send_text(from_e164, f"You chose: {choice_title}")
            return {"status": "ok_list_reply"}

        # Any other interactive subtype
        send_text(from_e164, "Got your selection.")
        return {"status": "ok_interactive_other"}

    # c) Other content → remember + AI + re-show menu
    remember(from_e164, "user", f"[{msg_type}] content")
    fallback_prompt = "User sent a non-text message on WhatsApp. Provide helpful next steps for a ComEMR user."
    ai = ai_answer(fallback_prompt, user=from_e164)
    send_text(from_e164, ai)
    try:
        send_buttons(from_e164, "Pick an option or reply with your question:")
    except Exception as e:
        print("[Buttons:fallback] Error:", e)
    return {"status": "ok_fallback"}

# --- Debug helpers: inspect recent webhook payloads in dev ---
@app.get("/webhook/debug/last")
def webhook_debug_last(n: int = 1):
    n = max(1, min(n, len(WEBHOOK_BUFFER)))
    return {"count": n, "items": list(WEBHOOK_BUFFER)[:n]}

# --- Existing router (kept, if present) ---
if HAS_WHATSAPP_ROUTER:
    app.include_router(whatsapp_router, prefix="/whatsapp", tags=["WhatsApp Gateway"])

# --- Test send endpoints (kept) ---
@app.post("/whatsapp/send-test/text", tags=["WhatsApp Gateway"])
def send_test_text(
    to: str = Query(..., description="Recipient E.164, e.g. 254705091683"),
    body: str = Query("Auth OK – replying from backend (v22.0)")
):
    try:
        result = send_text(to, body)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/whatsapp/send-test/template", tags=["WhatsApp Gateway"])
def send_test_template(
    to: str = Query(..., description="Recipient E.164, e.g., 254705091683"),
    name: str = Query("hello_world", description="Template name"),
    lang: str = Query("en_US", description="Language code, e.g., en_US")
):
    if not HAS_WHATSAPP_ADAPTER:
        # You can still send templates without the adapter, but we kept parity with your setup.
        raise HTTPException(status_code=500, detail="WhatsApp adapter not available/importable.")
    try:
        result = adapter_send_template(to, name, lang)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
