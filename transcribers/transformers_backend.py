"""
Transformers (HuggingFace) backend implementation.
"""

import sys
from typing import Dict, Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from .base import TranscriberBackend


class TransformersBackend(TranscriberBackend):
    """HuggingFace Transformers transcription backend"""

    def __init__(self, model_name: str = "base", language: Optional[str] = None):
        super().__init__(model_name, language)
        self.model = None
        self.processor = None
        self.pipe = None
        self.device = None
        self.torch_dtype = None

        # Map Whisper model names to HuggingFace model IDs
        self.model_mapping = {
            "tiny": "openai/whisper-tiny",
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large-v3",
        }

    def load_model(self):
        """Load the Transformers Whisper model"""
        model_id = self.model_mapping.get(self.model_name, "openai/whisper-base")
        print(f"Loading Transformers model '{model_id}'...", file=sys.stderr)

        # Auto-detect device and dtype
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.torch_dtype = torch.float16
            print("Using CUDA GPU", file=sys.stderr)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.torch_dtype = torch.float16
            print("Using Apple Metal GPU", file=sys.stderr)
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32
            print("Using CPU", file=sys.stderr)

        # Load model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            return_timestamps="word",  # Word-level timestamps
        )

    def transcribe(self, audio_path: str) -> Dict:
        """
        Transcribe audio using HuggingFace Transformers.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcription dictionary with standard format
        """
        if self.pipe is None:
            self.load_model()

        print("Transcribing with Transformers (HuggingFace)...", file=sys.stderr)

        # Set generation parameters
        generate_kwargs = {}
        if self.language:
            # Map language code to full name if needed
            generate_kwargs["language"] = self.language

        # Transcribe
        result = self.pipe(
            audio_path,
            generate_kwargs=generate_kwargs,
            return_timestamps="word",
        )

        # Convert to standard format
        standard_result = self._convert_to_standard_format(result)

        return standard_result

    def _convert_to_standard_format(self, result: Dict) -> Dict:
        """
        Convert Transformers output to standard format.

        Args:
            result: Result from Transformers pipeline

        Returns:
            Dictionary in standard format matching OpenAI Whisper
        """
        text = result.get("text", "")
        chunks = result.get("chunks", [])

        # Group words into segments (by sentence or fixed intervals)
        segments = []
        current_segment = None

        for chunk in chunks:
            word_text = chunk.get("text", "").strip()
            timestamp = chunk.get("timestamp", (0.0, 0.0))

            # Handle timestamp format
            if isinstance(timestamp, tuple) and len(timestamp) == 2:
                start, end = timestamp
            else:
                start = timestamp if timestamp else 0.0
                end = start

            # Create word entry
            word_entry = {
                "word": word_text,
                "start": start if start is not None else 0.0,
                "end": end if end is not None else start if start is not None else 0.0,
            }

            # Start new segment or add to current
            if current_segment is None:
                current_segment = {
                    "start": word_entry["start"],
                    "end": word_entry["end"],
                    "text": word_text,
                    "words": [word_entry],
                }
            else:
                # Add to current segment
                current_segment["text"] += " " + word_text
                current_segment["end"] = word_entry["end"]
                current_segment["words"].append(word_entry)

                # Split segment after punctuation or reasonable length
                if len(current_segment["words"]) >= 20 or word_text.endswith((".", "!", "?")):
                    segments.append(current_segment)
                    current_segment = None

        # Add last segment if exists
        if current_segment is not None:
            segments.append(current_segment)

        # Calculate duration
        duration = 0.0
        if segments:
            duration = max(seg["end"] for seg in segments)

        return {
            "text": text.strip(),
            "segments": segments,
            "duration": duration,
            "language": self.language if self.language else "auto",
        }

    def get_backend_name(self) -> str:
        """Get backend name"""
        return "transformers"
