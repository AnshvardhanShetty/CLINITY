"""Configuration for Clinical State Compiler."""

from pathlib import Path
from pydantic import BaseModel
from typing import Optional
import os


class ModelConfig(BaseModel):
    """Model configuration."""

    # MedGemma settings
    medgemma_model_id: str = "google/medgemma-4b-it"
    medgemma_device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    medgemma_load_in_4bit: bool = True  # Quantization for smaller GPUs

    # MedASR settings (for dictation)
    medasr_model_id: str = "google/medasr-base"  # Update when available

    # OCR settings
    tesseract_path: Optional[str] = None  # Path to tesseract binary if not in PATH
    ocr_language: str = "eng"
    ocr_preprocessing: bool = True  # Apply image preprocessing for better OCR


class CompilerConfig(BaseModel):
    """Compiler output configuration."""

    # Output sections to include
    include_active_problems: bool = True
    include_current_status: bool = True
    include_timeline: bool = True
    include_pending_tasks: bool = True
    include_risks: bool = True
    include_unclear_items: bool = True
    include_provenance: bool = True

    # Safety settings
    flag_ambiguity: bool = True
    require_source_attribution: bool = True
    max_output_length: int = 2000  # Approximate character limit for snapshot


class Config(BaseModel):
    """Main application configuration."""

    model: ModelConfig = ModelConfig()
    compiler: CompilerConfig = CompilerConfig()

    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")

    # Hugging Face token (set via environment variable)
    hf_token: Optional[str] = None

    @classmethod
    def load(cls) -> "Config":
        """Load configuration with environment overrides."""
        config = cls()
        config.hf_token = os.environ.get("HF_TOKEN")
        return config


# Global config instance
config = Config.load()
