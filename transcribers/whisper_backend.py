"""
OpenAI Whisper backend implementation.
"""

import sys
from typing import Dict, Optional

try:
    import whisper
except ImportError:
    print(
        "Error: openai-whisper not installed. Install with: uv sync",
        file=sys.stderr,
    )
    whisper = None

from .base import TranscriberBackend


class WhisperBackend(TranscriberBackend):
    """OpenAI Whisper transcription backend"""

    def __init__(self, model_name: str = "base", language: Optional[str] = None):
        super().__init__(model_name, language)
        self.model = None

    def load_model(self):
        """Load the Whisper model"""
        if whisper is None:
            raise ImportError("openai-whisper is not installed")

        print(f"Loading Whisper model '{self.model_name}'...", file=sys.stderr)
        self.model = whisper.load_model(self.model_name)

    def transcribe(self, audio_path: str) -> Dict:
        """
        Transcribe audio using OpenAI Whisper.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcription dictionary with standard format
        """
        if self.model is None:
            self.load_model()

        print("Transcribing with OpenAI Whisper...", file=sys.stderr)

        # Transcribe with word-level timestamps
        result = self.model.transcribe(
            audio_path, language=self.language, word_timestamps=True, verbose=False
        )

        # OpenAI Whisper already returns the correct format
        return result

    def get_backend_name(self) -> str:
        """Get backend name"""
        return "whisper"
