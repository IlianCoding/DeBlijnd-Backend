from PIL import Image, ImageDraw
from ultralytics import YOLO
from fastapi import HTTPException
import logging
import os

logger = logging.getLogger(__name__)

# Load the YOLO model once
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "own_seat.pt")
try:
    model = YOLO(MODEL_PATH)
    model.eval()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise e


def get_final_seat(image: Image.Image, results):
    """
    Determine the final seat detection using a weighted score that combines
    bounding box area and confidence, with a bonus for detections in the center.
    """
    width, height = image.size
    segment_width = width // 3

    best_score = 0
    best_quadrant = "none"
    best_box = None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            area = (x2 - x1) * (y2 - y1)
            score = area * conf
            x_center = (x1 + x2) / 2

            # Determine quadrant and boost score if detection is in the center
            if 0 <= x_center < segment_width:
                quadrant = "left"
            elif segment_width <= x_center < 2 * segment_width:
                quadrant = "center"
                score *= 1.2  # center bias boost
            elif 2 * segment_width <= x_center <= width:
                quadrant = "right"
            else:
                quadrant = "none"

            if score > best_score:
                best_score = score
                best_quadrant = quadrant
                best_box = [x1, y1, x2, y2]

    return best_quadrant, best_box


def process_image(image: Image.Image, debug: bool = False):
    """
    Resize the image, run model inference, and return the quadrant,
    bounding box, inference results, and resized image.
    """
    image_resized = image.resize((640, 640))
    try:
        results = model(image_resized)
    except Exception as e:
        logger.error(f"Error during model inference: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed")

    quadrant, final_box = get_final_seat(image_resized, results)
    return quadrant, final_box, results, image_resized


def draw_detections(image: Image.Image, results, final_box):
    """
    Draw all detections, region divisions, and highlight the final detection.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    segment_width = width // 3

    # Draw detection boxes and confidence scores
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"Conf: {conf:.2f}", fill="red")

    # Draw vertical lines for region divisions
    draw.line([(segment_width, 0), (segment_width, height)], fill="green", width=3)
    draw.line([(2 * segment_width, 0), (2 * segment_width, height)], fill="green", width=3)

    # Annotate regions
    draw.text((segment_width // 2 - 20, 10), "Left", fill="blue")
    draw.text((segment_width + segment_width // 2 - 30, 10), "Center", fill="blue")
    draw.text(((2 * segment_width) + segment_width // 2 - 20, 10), "Right", fill="blue")

    # Highlight the final detection
    if final_box is not None:
        x1, y1, x2, y2 = final_box
        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=5)
        draw.text((x1, y1 - 15), "Final Seat", fill="yellow")

    return image
