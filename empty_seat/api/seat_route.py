from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from schemes.seat_base import SeatDetectionResponseDebug, SeatDetectionImgResponse
from services.seat_service import process_image, draw_detections
from PIL import Image
import io
import logging
import base64
seat_router = APIRouter(prefix="/api", tags=["Seat Detection"])
logger = logging.getLogger(__name__)


def encode_image_to_base64(image) -> str:
    """Encode a PIL Image to a base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_str
@seat_router.post("/detect_seat" , response_model=SeatDetectionImgResponse)
async def detect_seat(
        image: UploadFile = File(...),
        debug: bool = Query(False, description="Enable debug visualization")
):
    # Read and validate the image
    try:
        contents = await image.read()
    except Exception as e:
        logger.error(f"Failed to read image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail="Unable to process image")

    # Process the image and perform inference
    quadrant, final_box, results, img_resized = process_image(img, debug)
    img_debug = draw_detections(img_resized.copy(), results, final_box)
    img = encode_image_to_base64(img_debug)
    response = SeatDetectionImgResponse(empty_seat=quadrant,img=img)

    # If debug mode is enabled, draw and save the visualization
    if debug:
        try:
            img_debug = draw_detections(img_resized.copy(), results, final_box)
            visualization_path = "detection_output.jpg"
            img_debug.save(visualization_path)
            return SeatDetectionResponseDebug(empty_seat=quadrant, visualization=visualization_path)
        except Exception as e:
            logger.error(f"Error during visualization: {e}")
            raise HTTPException(status_code=500, detail="Visualization failed")

    return response
