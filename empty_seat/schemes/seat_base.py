from pydantic import BaseModel
from typing import Optional


class SeatDetectionResponse(BaseModel):
    empty_seat: str


class SeatDetectionImgResponse(SeatDetectionResponse):
    img:str

class SeatDetectionResponseDebug(BaseModel):
    empty_seat: str
    visualization: Optional[str] = None
