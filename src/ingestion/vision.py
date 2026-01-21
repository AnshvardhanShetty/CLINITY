"""Vision-based document processing using MedGemma multimodal."""

import os
from pathlib import Path
from typing import Union
from PIL import Image
import io

from ..config import config
from ..models import InputDocument, SourceType


class VisionProcessor:
    """Process documents using MedGemma's vision capabilities directly."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None

    def _get_device(self) -> str:
        """Determine best available device."""
        import torch
        if config.model.medgemma_device != "auto":
            return config.model.medgemma_device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self):
        """Load MedGemma multimodal model."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._device = self._get_device()
        model_id = config.model.medgemma_model_id

        print(f"Loading MedGemma vision on {self._device}...")

        self._processor = AutoProcessor.from_pretrained(
            model_id,
            token=config.hf_token,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=config.hf_token,
            torch_dtype=torch.bfloat16 if self._device != "cpu" else torch.float32,
            device_map=self._device,
        )

        print("✓ MedGemma vision loaded")

    def extract_from_image(
        self,
        image: Image.Image,
        extraction_prompt: str | None = None,
    ) -> str:
        """
        Extract text/information from an image using MedGemma vision.

        Args:
            image: PIL Image to process
            extraction_prompt: Custom prompt (uses default clinical extraction if None)

        Returns:
            Extracted text content
        """
        import torch

        self.load()

        if extraction_prompt is None:
            extraction_prompt = """Extract all text and clinical information visible in this image.
Include both typed and handwritten content.
Preserve the structure and organize by patient/section if applicable.
Note any information that is unclear or partially visible."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": extraction_prompt}
                ]
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=1500,
                temperature=0.1,
                do_sample=True,
            )

        response = self._processor.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def process_document(
        self,
        image_source: Union[str, Path, bytes, Image.Image],
        document_id: str,
        source_type: SourceType = SourceType.HANDWRITTEN,
        filename: str | None = None,
        custom_prompt: str | None = None,
    ) -> InputDocument:
        """
        Process an image document using vision and return an InputDocument.

        Args:
            image_source: Path, bytes, or PIL Image
            document_id: Unique ID for this document
            source_type: Type of document
            filename: Original filename
            custom_prompt: Custom extraction prompt

        Returns:
            InputDocument with extracted content
        """
        # Load image
        if isinstance(image_source, (str, Path)):
            image = Image.open(image_source)
            with open(image_source, "rb") as f:
                raw_bytes = f.read()
            filename = filename or Path(image_source).name
        elif isinstance(image_source, bytes):
            image = Image.open(io.BytesIO(image_source))
            raw_bytes = image_source
        elif isinstance(image_source, Image.Image):
            image = image_source
            # Convert to bytes for storage
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            raw_bytes = buf.getvalue()
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")

        # Extract content using vision
        content = self.extract_from_image(image, custom_prompt)

        return InputDocument(
            id=document_id,
            source_type=source_type,
            content=content,
            raw_content=raw_bytes,
            filename=filename,
            confidence=0.85,  # Vision typically higher confidence than OCR
        )


class HybridProcessor:
    """
    Hybrid processor that chooses the best method based on document type.

    - Images (handwritten, scanned) → MedGemma Vision
    - Text files → Direct text processing
    - Audio → MedASR
    """

    def __init__(self):
        self._vision = None
        self._text = None
        self._audio = None
        self._ocr = None

    @property
    def vision(self):
        if self._vision is None:
            self._vision = VisionProcessor()
        return self._vision

    @property
    def text(self):
        if self._text is None:
            from .text import TextProcessor
            self._text = TextProcessor()
        return self._text

    @property
    def audio(self):
        if self._audio is None:
            from .audio import AudioProcessor
            self._audio = AudioProcessor()
        return self._audio

    @property
    def ocr(self):
        """Fallback OCR for simple typed documents."""
        if self._ocr is None:
            from .ocr import OCRProcessor
            self._ocr = OCRProcessor()
        return self._ocr

    def process(
        self,
        source: Union[str, Path, bytes],
        document_id: str,
        source_type: SourceType | None = None,
        filename: str | None = None,
        use_vision: bool = True,
    ) -> InputDocument:
        """
        Process any document using the best available method.

        Args:
            source: File path, bytes, or text content
            document_id: Unique document ID
            source_type: Document type (auto-detected if None)
            filename: Original filename
            use_vision: Use MedGemma vision for images (recommended)

        Returns:
            InputDocument with extracted content
        """
        # Determine file type
        if isinstance(source, (str, Path)):
            path = Path(source)
            suffix = path.suffix.lower()
            filename = filename or path.name

            # Image files → Vision (preferred) or OCR
            if suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
                if use_vision:
                    return self.vision.process_document(
                        source, document_id,
                        source_type or SourceType.HANDWRITTEN,
                        filename
                    )
                else:
                    return self.ocr.process_document(
                        source, document_id,
                        source_type or SourceType.HANDWRITTEN,
                        filename
                    )

            # Audio files → ASR
            elif suffix in [".wav", ".mp3", ".m4a", ".flac", ".ogg"]:
                return self.audio.process_document(source, document_id, filename)

            # Text files → Direct processing
            else:
                return self.text.process_document(
                    source, document_id, source_type, filename
                )

        # Bytes - try to detect type
        elif isinstance(source, bytes):
            # Check for image magic bytes
            if source[:4] in [b'\x89PNG', b'\xff\xd8\xff\xe0', b'\xff\xd8\xff\xe1']:
                if use_vision:
                    return self.vision.process_document(
                        source, document_id,
                        source_type or SourceType.HANDWRITTEN,
                        filename
                    )
                else:
                    return self.ocr.process_document(
                        source, document_id,
                        source_type or SourceType.HANDWRITTEN,
                        filename
                    )
            else:
                # Assume text
                return self.text.process_document(
                    source.decode('utf-8', errors='ignore'),
                    document_id, source_type, filename
                )

        # String - direct text
        else:
            return self.text.process_document(
                source, document_id, source_type, filename
            )
