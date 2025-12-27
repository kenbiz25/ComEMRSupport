
# app.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from apps.whatsapp_gateway.routes import router as whatsapp_router
from rag.composer import RagComposer
from core.indexing.pipeline import IndexingPipeline
from config.settings import settings

# If your WhatsApp adapter file is whatsapp.py (as shared), import helpers:
try:
    from whatsapp import (
        send_whatsapp_text,
        send_whatsapp_template,
    )
    HAS_WHATSAPP_ADAPTER = True
except Exception:
    HAS_WHATSAPP_ADAPTER = False

import os

# Initialize FastAPI app
app = FastAPI(title="ComEMR Support", version="1.0.0")

# CORS middleware (restrict origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Replace with allowed domains in production
    allow_headers=["*"],
    allow_methods=["*"],
)

# Shared components
rag = RagComposer(
    llm_model=settings.LLM_MODEL,
    confidence_threshold=settings.ANSWER_CONFIDENCE_THRESHOLD,
    safeguard=settings.SAFEGUARD_ENABLE,
)

# Indexing pipeline (reads settings internally)
indexer = IndexingPipeline()

# --- Startup diagnostics (secure) ---
@app.on_event("startup")
def startup_diag():
    # Only print prefixes for safety
    token = os.getenv("META_WHATSAPP_TOKEN", "")
    phone_id = os.getenv("WHATSAPP_PHONE_ID", "")
    api_ver = os.getenv("WHATSAPP_API_VERSION", "v22.0")

    print("[Startup] WhatsApp config:")
    print("  API version:", api_ver)
    print("  TOKEN prefix:", (token[:8] + "...") if token else "MISSING")
    print("  PHONE_ID:", phone_id if phone_id else "MISSING")

    if not token or not phone_id:
        print("⚠️ META_WHATSAPP_TOKEN or WHATSAPP_PHONE_ID missing. "
              "Sending will fail until environment is set.")

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# WhatsApp webhook routes
app.include_router(whatsapp_router, prefix="/whatsapp", tags=["WhatsApp Gateway"])

# Optional: Trigger KB reindex (protect with authentication in production)
@app.post("/kb/reindex")
def reindex():
    count = indexer.reindex_all()
    return {"reindexed": count}

# --- Test endpoints to verify backend send path ---

@app.post("/whatsapp/send-test/text", tags=["WhatsApp Gateway"])
def send_test_text(
    to: str = Query(..., description="Recipient E.164, e.g. 254705091683"),
    body: str = Query("Auth OK – replying from backend (v22.0)")
):
    if not HAS_WHATSAPP_ADAPTER:
        raise HTTPException(status_code=500, detail="WhatsApp adapter not available/importable.")
    try:
        result = send_whatsapp_text(to, body)
        return {"ok": True, "result": result}
    except Exception as e:
        # Return the error so you can see it in Swagger/terminal
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/whatsapp/send-test/template", tags=["WhatsApp Gateway"])
def send_test_template(
    to: str = Query(..., description="Recipient E.164, e.g. 254705091683"),
    name: str = Query("hello_world", description="Template name"),
    lang: str = Query("en_US", description="Language code, e.g., en_US")
):
    if not HAS_WHATSAPP_ADAPTER:
        raise HTTPException(status_code=500, detail="WhatsApp adapter not available/importable.")
    try:
        result = send_whatsapp_template(to, name, lang)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
