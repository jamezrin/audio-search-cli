"""
Transcriber backends for audio transcription.

Provides a factory function to create the appropriate transcriber backend.
"""

from .base import TranscriberBackend
from .whisper_backend import WhisperBackend

# Try to import whisper.cpp backend
WHISPER_CPP_AVAILABLE = False
WhisperCppBackend = None

try:
    # Check if pywhispercpp is installed first
    import pywhispercpp  # noqa: F401
    from .whisper_cpp_backend import WhisperCppBackend
    WHISPER_CPP_AVAILABLE = True
except ImportError:
    WHISPER_CPP_AVAILABLE = False
    WhisperCppBackend = None


def create_transcriber(
    backend: str = "auto",
    model_name: str = "base",
    language: str = None
) -> TranscriberBackend:
    """
    Factory function to create a transcriber backend.
    
    Args:
        backend: Backend to use ("whisper", "whisper.cpp", or "auto")
        model_name: Model size to use
        language: Language code (optional)
        
    Returns:
        TranscriberBackend instance
        
    Raises:
        ValueError: If backend is not available or invalid
    """
    backend = backend.lower()
    
    if backend == "auto":
        # Prefer whisper.cpp if available
        if WHISPER_CPP_AVAILABLE:
            backend = "whisper.cpp"
        else:
            backend = "whisper"
    
    if backend == "whisper.cpp":
        if not WHISPER_CPP_AVAILABLE:
            raise ValueError(
                "whisper.cpp backend not available. "
                "Install with: pip install pywhispercpp"
            )
        return WhisperCppBackend(model_name, language)
    
    elif backend == "whisper":
        return WhisperBackend(model_name, language)
    
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Available: whisper, whisper.cpp"
        )


def list_available_backends():
    """List available transcriber backends"""
    backends = ["whisper"]
    if WHISPER_CPP_AVAILABLE:
        backends.append("whisper.cpp")
    return backends


__all__ = [
    'TranscriberBackend',
    'WhisperBackend',
    'WhisperCppBackend',
    'create_transcriber',
    'list_available_backends',
    'WHISPER_CPP_AVAILABLE'
]
