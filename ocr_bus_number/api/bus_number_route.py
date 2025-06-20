import io
import time
import base64
import logging
import numpy as np
from PIL import Image, ExifTags
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from schemes.bus_number_base import BusOcrResponse, HealthResponse
from services.bus_number_service import pipeline
import logging

bus_number_router = APIRouter(prefix="/api", tags=["Bus Number OCR"])
logger = logging.getLogger(__name__)

def correct_image_orientation(img: Image.Image) -> Image.Image:
    """Fix image rotation based on EXIF data."""
    try:
        # Find the orientation tag
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break

        exif = img._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation, None)
            if orientation_value == 3:
                img = img.rotate(180, expand=True)
            elif orientation_value == 6:
                img = img.rotate(270, expand=True)
            elif orientation_value == 8:
                img = img.rotate(90, expand=True)
    except Exception as e:
        logger.error(f"Error correcting image orientation: {e}")
    return img

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode a numpy image (RGB) to a base64 string."""
    pil_img = Image.fromarray(image.astype('uint8'), 'RGB')
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_str

@bus_number_router.post("/predict", response_model=BusOcrResponse)
async def predict_bus_number(file: UploadFile = File(...)):
    """
    Process an image to detect and recognize bus numbers.
    Returns the detected bus number, confidence score, raw OCR text, and the cropped image (base64-encoded).
    """
    try:
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data))
        # Correct the orientation using EXIF data
        img = correct_image_orientation(img)
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        img_resized = img.resize((640, 640))
        if img_resized.mode != "RGB":
            img_resized = img_resized.convert("RGB")
        image_rgb = np.array(img_resized)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail="Unable to process image")

    # Run bus-number detection
    box = pipeline.detect_bus_number(image_rgb)
    if box is None:
        return BusOcrResponse(
            bus_number="null",
            confidence=0.0,
            raw_text="",
            cropped_image="",
            message="No bus number detected"
        )

    # Extract OCR text from the detected region, and get the enhanced (cropped) region
    ocr_text, enhanced_region, conf = pipeline.extract_text_from_led(image_rgb, box)
    valid_number = pipeline.validate_number(ocr_text)

    # Encode the cropped image to base64 if available
    encoded_cropped = encode_image_to_base64(enhanced_region) if enhanced_region is not None else ""

    return BusOcrResponse(
        bus_number=valid_number if valid_number else "null",
        confidence=float(conf) if valid_number else 0.0,
        raw_text=ocr_text,
        cropped_image=encoded_cropped,
        message="" if valid_number else "No valid bus number recognized"
    )


@bus_number_router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify the API is running."""
    return HealthResponse(status="healthy", timestamp=time.time())
