# ARCHIVED: moved to archive/apps/whatsapp_gateway/dto.py
# This file is retained here as an inert stub to preserve imports.
from pydantic import BaseModel

class InboundMessage(BaseModel):
    from_number: str = ""
    text: str = ""