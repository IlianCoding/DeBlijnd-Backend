from fastapi import FastAPI
from api.bus_number_route import bus_number_router

app = FastAPI(
    title="Bus Number OCR API",
    description="API for detecting and recognizing bus numbers from images",
    version="0.0.1"
)

app.include_router(bus_number_router)