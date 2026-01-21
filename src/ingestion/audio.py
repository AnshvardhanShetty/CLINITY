"""Audio processing for clinical dictation using MedASR."""

from pathlib import Path
from typing import Union
import soundfile as sf
import numpy as np

from ..config import config
from ..models import InputDocument, SourceType


class AudioProcessor:
    """Process audio dictation using MedASR."""

    def __init__(self):
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load MedASR model."""
        if self._model is None:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            import torch

            device = self._get_device()
            dtype = torch.float16 if device != "cpu" else torch.float32

            # Note: Update model ID when MedASR is available on HuggingFace
            # For now, we'll use a placeholder that can be swapped
            model_id = config.model.medasr_model_id

            try:
                self._processor = AutoProcessor.from_pretrained(
                    model_id,
                    token=config.hf_token,
                )
                self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    token=config.hf_token,
                ).to(device)
            except Exception as e:
                # Fallback to Whisper if MedASR not available
                print(f"MedASR not available ({e}), falling back to Whisper")
                model_id = "openai/whisper-base"
                self._processor = AutoProcessor.from_pretrained(model_id)
                self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                ).to(device)

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

    def load_audio(self, audio_source: Union[str, Path, bytes]) -> tuple[np.ndarray, int]:
        """Load audio file and return (waveform, sample_rate)."""
        if isinstance(audio_source, bytes):
            import io
            audio_source = io.BytesIO(audio_source)

        waveform, sample_rate = sf.read(audio_source)

        # Convert to mono if stereo
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        return waveform, sample_rate

    def transcribe(self, audio_source: Union[str, Path, bytes]) -> tuple[str, float]:
        """
        Transcribe audio to text.

        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        self._load_model()

        waveform, sample_rate = self.load_audio(audio_source)

        # Resample to 16kHz if needed (standard for speech models)
        if sample_rate != 16000:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        # Process audio
        inputs = self._processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        device = self._get_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate transcription
        import torch
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=448,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode
        transcription = self._processor.batch_decode(
            generated_ids.sequences,
            skip_special_tokens=True,
        )[0]

        # Approximate confidence from generation scores
        # (simplified - real implementation would be more sophisticated)
        confidence = 0.85  # Placeholder

        return transcription.strip(), confidence

    def process_document(
        self,
        audio_source: Union[str, Path, bytes],
        document_id: str,
        filename: str | None = None,
    ) -> InputDocument:
        """Process audio and return an InputDocument."""
        # Read raw bytes if path
        if isinstance(audio_source, (str, Path)):
            with open(audio_source, "rb") as f:
                raw_bytes = f.read()
            filename = filename or Path(audio_source).name
        else:
            raw_bytes = audio_source

        text, confidence = self.transcribe(audio_source)

        return InputDocument(
            id=document_id,
            source_type=SourceType.DICTATION,
            content=text,
            raw_content=raw_bytes,
            filename=filename,
            confidence=confidence,
        )
