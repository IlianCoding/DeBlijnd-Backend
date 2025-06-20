from pydantic import BaseModel
from typing import Optional

class BusOcrResponseBase(BaseModel):
    bus_number: str
    confidence: float
    raw_text: str
    cropped_image: str


class BusOcrResponse(BusOcrResponseBase):
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: float
