"""
Abstract base class for transcriber backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class TranscriberBackend(ABC):
    """Abstract base class for audio transcription backends"""
    
    def __init__(self, model_name: str = "base", language: Optional[str] = None):
        """
        Initialize the transcriber backend.
        
        Args:
            model_name: Model size (tiny, base, small, medium, large)
            language: Language code (e.g., 'en', 'es', 'fr'). None for auto-detection.
        """
        self.model_name = model_name
        self.language = language
    
    @abstractmethod
    def load_model(self):
        """Load the transcription model"""
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: str) -> Dict:
        """
        Transcribe an audio file with word-level timestamps.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with transcription data in standard format:
            {
                'text': str,
                'duration': float,
                'segments': [
                    {
                        'start': float,
                        'end': float,
                        'text': str,
                        'words': [
                            {'word': str, 'start': float, 'end': float},
                            ...
                        ]
                    },
                    ...
                ]
            }
        """
        pass
    
    @abstractmethod
    def get_backend_name(self) -> str:
        """Get the name of this backend"""
        pass
    
    def get_model_name(self) -> str:
        """Get the model name being used"""
        return self.model_name
    
    def get_language(self) -> Optional[str]:
        """Get the language setting"""
        return self.language
