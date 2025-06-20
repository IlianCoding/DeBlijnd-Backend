from fastapi import FastAPI
from api.seat_route import seat_router

app = FastAPI(
    title="Seat Detection API",
    description="API for detecting empty seats in public transportation",
    version="0.0.2"
)

app.include_router(seat_router)