
from pydantic import BaseModel

class InboundMessage(BaseModel):
    from_number: str
    text: str