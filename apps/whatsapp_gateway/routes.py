from fastapi import APIRouter, Request, HTTPException, Query, Response, UploadFile, File
from config.settings import settings
from config.logging import get_logger
from config.rate_limit import allow
from apps.whatsapp_gateway.send import send_whatsapp_text  # keep only this import
from rag.composer import RagComposer
from core.whatsapp.whatsapp_service import WhatsAppService

# Helper to create a WhatsApp service using configured credentials
def _create_whatsapp_service():
    return WhatsAppService(settings.WHATSAPP_PHONE_ID, settings.META_WHATSAPP_TOKEN)

router = APIRouter()

# Include the webhook verification route
log = get_logger("whatsapp")

# Instantiate composer once; pass model + policy knobs
composer = RagComposer(
    llm_model=settings.LLM_MODEL,
    safeguard=settings.SAFEGUARD_ENABLE,
    top_k=getattr(settings, "TOP_K", 3),
)

# ...existing code...
