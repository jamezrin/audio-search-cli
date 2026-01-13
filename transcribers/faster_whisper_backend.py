"""
Faster-Whisper backend implementation (CTranslate2-based).
"""

import sys
from typing import Dict, Optional

from faster_whisper import WhisperModel

from .base import TranscriberBackend


class FasterWhisperBackend(TranscriberBackend):
    """Faster-Whisper transcription backend using CTranslate2"""

    def __init__(self, model_name: str = "base", language: Optional[str] = None):
        super().__init__(model_name, language)
        self.model = None

    def load_model(self):
        """Load the Faster-Whisper model"""
        print(f"Loading Faster-Whisper model '{self.model_name}'...", file=sys.stderr)

        # Auto-detect compute type based on available hardware
        # cuda for NVIDIA, auto for CPU
        try:
            import torch

            if torch.cuda.is_available():
                compute_type = "float16"
                device = "cuda"
            else:
                compute_type = "int8"  # Quantized for CPU
                device = "cpu"
        except ImportError:
            compute_type = "int8"
            device = "cpu"

        self.model = WhisperModel(
            self.model_name,
            device=device,
            compute_type=compute_type,
            download_root=None,  # Use default cache
        )

    def transcribe(self, audio_path: str) -> Dict:
        """
        Transcribe audio using Faster-Whisper.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcription dictionary with standard format
        """
        if self.model is None:
            self.load_model()

        print("Transcribing with Faster-Whisper (CTranslate2)...", file=sys.stderr)

        # Transcribe with word-level timestamps
        segments, info = self.model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=True,
            vad_filter=True,  # Voice activity detection for better accuracy
        )

        # Convert to standard format
        result = self._convert_to_standard_format(segments, info)

        return result

    def _convert_to_standard_format(self, segments, info) -> Dict:
        """
        Convert faster-whisper output to standard format.

        Args:
            segments: Segments from faster-whisper (generator)
            info: Transcription info

        Returns:
            Dictionary in standard format matching OpenAI Whisper
        """
        standard_segments = []
        full_text_parts = []
        total_duration = info.duration if hasattr(info, "duration") else 0.0

        # segments is a generator, so we need to iterate
        for segment in segments:
            segment_dict = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            }

            # Add word-level timestamps if available
            if hasattr(segment, "words") and segment.words:
                words = []
                for word in segment.words:
                    words.append(
                        {
                            "word": word.word.strip(),
                            "start": word.start,
                            "end": word.end,
                        }
                    )
                segment_dict["words"] = words

            standard_segments.append(segment_dict)
            full_text_parts.append(segment_dict["text"])

        return {
            "text": " ".join(full_text_parts),
            "segments": standard_segments,
            "duration": total_duration,
            "language": info.language if hasattr(info, "language") else self.language,
        }

    def get_backend_name(self) -> str:
        """Get backend name"""
        return "faster-whisper"
