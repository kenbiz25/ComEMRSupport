
from pydantic import BaseModel
from typing import Dict, List

class KBChunk(BaseModel):
    namespace: str
    content: str
    metadata: Dict
    embedding: List[float]
