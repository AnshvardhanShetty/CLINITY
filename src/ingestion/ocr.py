"""OCR processing for handwritten and scanned notes."""

import cv2
import numpy as np
from PIL import Image
import pytesseract
from pathlib import Path
from typing import Union
import io

from ..config import config
from ..models import InputDocument, SourceType


class OCRProcessor:
    """Process handwritten and scanned documents using OCR."""

    def __init__(self):
        if config.model.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = config.model.tesseract_path

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR on handwritten notes.

        Applies:
        - Grayscale conversion
        - Noise reduction
        - Adaptive thresholding
        - Deskewing
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Adaptive thresholding for varying lighting
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Deskew
        coords = np.column_stack(np.where(binary < 255))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.5:
                (h, w) = binary.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                binary = cv2.warpAffine(
                    binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                )

        return binary

    def extract_text(
        self,
        image_source: Union[str, Path, bytes, np.ndarray, Image.Image],
        preprocess: bool = True,
    ) -> tuple[str, float]:
        """
        Extract text from an image.

        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        # Load image into numpy array
        if isinstance(image_source, (str, Path)):
            image = cv2.imread(str(image_source))
        elif isinstance(image_source, bytes):
            nparr = np.frombuffer(image_source, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_source, Image.Image):
            image = np.array(image_source)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = image_source

        if image is None:
            raise ValueError("Could not load image")

        # Preprocess if requested
        if preprocess and config.model.ocr_preprocessing:
            processed = self.preprocess_image(image)
        else:
            processed = image

        # Run OCR with confidence data
        ocr_data = pytesseract.image_to_data(
            processed,
            lang=config.model.ocr_language,
            output_type=pytesseract.Output.DICT,
        )

        # Extract text and calculate confidence
        words = []
        confidences = []
        for i, word in enumerate(ocr_data["text"]):
            if word.strip():
                words.append(word)
                conf = ocr_data["conf"][i]
                if conf > 0:  # -1 means no confidence available
                    confidences.append(conf / 100.0)

        text = " ".join(words)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return text, avg_confidence

    def process_document(
        self,
        image_source: Union[str, Path, bytes],
        document_id: str,
        source_type: SourceType = SourceType.HANDWRITTEN,
        filename: str | None = None,
    ) -> InputDocument:
        """
        Process an image and return an InputDocument.
        """
        # Read raw bytes if path
        if isinstance(image_source, (str, Path)):
            with open(image_source, "rb") as f:
                raw_bytes = f.read()
            filename = filename or Path(image_source).name
        else:
            raw_bytes = image_source

        text, confidence = self.extract_text(image_source)

        return InputDocument(
            id=document_id,
            source_type=source_type,
            content=text,
            raw_content=raw_bytes,
            filename=filename,
            confidence=confidence,
        )
