import os
import re
import logging
from collections import Counter
from difflib import SequenceMatcher
from typing import Optional, Tuple, Union, List
import cv2
import numpy as np
import requests
import functools
import onnxruntime as ort
import torch
from collections import Counter
import logging
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configuration (using os.path for compatibility with your existing structure)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "bus_number.onnx")
CONF_THRESHOLD = 0.75
OCR_PATH = os.path.join(BASE_DIR, "..", "model", "local_trocr")


def similarity(a: str, b: str) -> float:
    """Calculate string similarity ratio."""
    return SequenceMatcher(None, a, b).ratio()


def led_panel_enhancement(image: np.ndarray) -> np.ndarray:
    """
    Enhance LED panel image for better OCR results.
    Converts the image to LAB, applies CLAHE to the L channel, then converts back to RGB.
    """
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    lab_enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(lab_enhanced_bgr, cv2.COLOR_BGR2RGB)

class BusNumberOcrPipeline:
    def __init__(self):
        # Initialize ONNX session for bus-number detection
        try:
            self.number_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
            logger.info("Bus number detection ONNX session initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize detection session: {e}")
            raise

        # Initialize primary OCR model from local files
        try:
            self.processor = TrOCRProcessor.from_pretrained(OCR_PATH, local_files_only=True)
            self.ocr_model = VisionEncoderDecoderModel.from_pretrained(OCR_PATH, local_files_only=True)
            logger.info("OCR model loaded successfully from local path.")
        except Exception as e:
            logger.error(f"Failed to load OCR model: {e}")
            raise

        self.bus_number_pattern = re.compile(r'^[0-9]{1,3}[A-Za-z]?$')
        self.conf_threshold = CONF_THRESHOLD
        self.last_valid_numbers: List[str] = []
        self.smoothing_window = 5

    def detect_bus_number(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Run the bus-number detection model on a 640x640 RGB image.
        Returns a detection box [center_x, center_y, width, height] if a valid score is found.
        """
        img_input = image.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)           # Shape: (1, H, W, 3)
        img_input = np.transpose(img_input, (0, 3, 1, 2))        # Shape: (1, 3, H, W)

        try:
            input_name = self.number_session.get_inputs()[0].name
            outputs = self.number_session.run(None, {input_name: img_input})
        except Exception as e:
            logger.error(f"Error during bus-number detection inference: {e}")
            return None

        output_data = np.array(outputs[0])
        output_data = np.squeeze(output_data).T  # Each row: [x, y, w, h, score]
        if output_data.ndim != 2 or output_data.shape[1] < 5:
            logger.error("Unexpected output shape from detection model.")
            return None

        boxes = output_data[:, :4]
        scores = output_data[:, 4]
        valid_indices = np.where(scores > self.conf_threshold)[0]
        if valid_indices.size == 0:
            logger.info("No detection exceeds confidence threshold.")
            return None

        best_idx = valid_indices[np.argmax(scores[valid_indices])]
        return boxes[best_idx]

    def extract_text_from_led(self, image: np.ndarray, box: Union[List[float], np.ndarray, Tuple[float, float, float, float]]) -> Tuple[str, Optional[np.ndarray], float]:
        """
        Extract OCR text from the LED region defined by the detection box.
        Also enhances the cropped region for improved OCR.

        Returns:
            Tuple[str, Optional[np.ndarray], float]: (extracted text, enhanced image region, confidence score)
        """
        try:
            x, y, w, h = box
        except Exception as e:
            logger.error(f"Invalid box format: {box} - {e}")
            return "", None, 0.0

        margin = 0.15
        x1 = max(0, int(x - (w / 2) * (1 + margin)))
        y1 = max(0, int(y - (h / 2) * (1 + margin)))
        x2 = min(image.shape[1], int(x + (w / 2) * (1 + margin)))
        y2 = min(image.shape[0], int(y + (h / 2) * (1 + margin)))
        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0 or cropped.shape[0] < 1 or cropped.shape[1] < 1:
            logger.warning("Cropped region is empty or invalid.")
            return "", None, 0.0

        # Resize if region is too small
        if cropped.shape[0] < 30 or cropped.shape[1] < 30:
            scale_factor = max(30 / cropped.shape[0], 30 / cropped.shape[1])
            new_h = int(cropped.shape[0] * scale_factor)
            new_w = int(cropped.shape[1] * scale_factor)
            cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        enhanced = led_panel_enhancement(cropped)

        try:
            pixel_values = self.processor(images=enhanced, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids = self.ocr_model.generate(pixel_values)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().upper()
            text = self.clean_led_ocr_result(text)
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return "", enhanced, 0.0

        return text, enhanced, 0.9  # Confidence score hardcoded as 0.9

    def clean_led_ocr_result(self, text: str) -> str:
        """
        Clean the OCR output by removing unwanted characters and replacing similar-looking characters.
        """
        text = re.sub(r'[^A-Z0-9]', '', text)
        replacements = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'}
        if sum(c.isdigit() for c in text) > len(text) * 0.5:
            for old, new in replacements.items():
                text = text.replace(old, new)
        return text

    def validate_number(self, text: str, conf: Optional[float] = None) -> Optional[str]:
        """
        Validate the cleaned OCR result against the bus number pattern.
        A valid bus number consists of 1 to 3 digits followed by an optional letter.
        Uses temporal smoothing based on recent valid results if needed.

        Args:
            text (str): The OCR extracted text.
            conf (Optional[float]): Confidence score (unused for now).

        Returns:
            Optional[str]: Validated bus number if valid, else None.
        """
        text = text.strip().upper()
        # Remove any characters that are not A-Z or 0-9
        text = re.sub(r'[^A-Z0-9]', '', text)
        if not text:
            return None

        if self.bus_number_pattern.fullmatch(text):
            self.last_valid_numbers.append(text)
            # Keep only the last 'smoothing_window' valid detections
            if len(self.last_valid_numbers) > self.smoothing_window:
                self.last_valid_numbers = self.last_valid_numbers[-self.smoothing_window:]
            return text

        if self.last_valid_numbers:
            counts = Counter(self.last_valid_numbers)
            most_common, count = counts.most_common(1)[0]
            if count >= 3:
                return most_common

        return None

pipeline = BusNumberOcrPipeline()
