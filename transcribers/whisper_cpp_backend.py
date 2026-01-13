"""
Whisper.cpp backend implementation (via pywhispercpp).
"""

import sys
from typing import Dict, Optional

from pywhispercpp.model import Model

from .base import TranscriberBackend


class WhisperCppBackend(TranscriberBackend):
    """Whisper.cpp transcription backend using pywhispercpp"""

    def __init__(self, model_name: str = "base", language: Optional[str] = None):
        super().__init__(model_name, language)
        self.model = None

    def load_model(self):
        """Load the Whisper.cpp model"""
        print(f"Loading Whisper.cpp model '{self.model_name}'...", file=sys.stderr)

        # pywhispercpp will automatically download the model if not present
        # Don't specify n_threads to let it use the default
        self.model = Model(model=self.model_name)

    def transcribe(self, audio_path: str) -> Dict:
        """
        Transcribe audio using Whisper.cpp.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcription dictionary with standard format
        """
        if self.model is None:
            self.load_model()

        print("Transcribing with Whisper.cpp (faster)...", file=sys.stderr)

        # Transcribe - pywhispercpp API doesn't have word_timestamps parameter
        # It returns segments with word-level timestamps automatically
        segments = self.model.transcribe(
            audio_path,
            language=self.language if self.language else None,
        )

        # Convert to standard format
        result = self._convert_to_standard_format(segments)

        return result

    def _convert_to_standard_format(self, segments) -> Dict:
        """
        Convert pywhispercpp output to standard format.

        Args:
            segments: Segments from pywhispercpp

        Returns:
            Dictionary in standard format matching OpenAI Whisper
        """
        standard_segments = []
        full_text_parts = []
        total_duration = 0.0

        for segment in segments:
            # Extract segment info
            segment_dict = {
                "start": segment.t0 / 100.0,  # Convert centiseconds to seconds
                "end": segment.t1 / 100.0,
                "text": segment.text.strip(),
            }

            # Add word-level timestamps if available
            if hasattr(segment, "words") and segment.words:
                words = []
                for word in segment.words:
                    words.append(
                        {
                            "word": word.word.strip(),
                            "start": word.t0 / 100.0,
                            "end": word.t1 / 100.0,
                        }
                    )
                segment_dict["words"] = words

            standard_segments.append(segment_dict)
            full_text_parts.append(segment_dict["text"])

            if segment_dict["end"] > total_duration:
                total_duration = segment_dict["end"]

        return {
            "text": " ".join(full_text_parts),
            "segments": standard_segments,
            "duration": total_duration,
            "language": self.language if self.language else "auto",
        }

    def get_backend_name(self) -> str:
        """Get backend name"""
        return "whisper.cpp"
