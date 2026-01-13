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

# Try to import faster-whisper backend
FASTER_WHISPER_AVAILABLE = False
FasterWhisperBackend = None

try:
    import faster_whisper  # noqa: F401

    from .faster_whisper_backend import FasterWhisperBackend

    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    FasterWhisperBackend = None

# Try to import transformers backend
TRANSFORMERS_AVAILABLE = False
TransformersBackend = None

try:
    import transformers  # noqa: F401

    from .transformers_backend import TransformersBackend

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TransformersBackend = None


def create_transcriber(
    backend: str = "auto", model_name: str = "base", language: str = None
) -> TranscriberBackend:
    """
    Factory function to create a transcriber backend.

    Args:
        backend: Backend to use ("whisper", "whisper.cpp", "faster-whisper", "transformers", or "auto")
        model_name: Model size to use
        language: Language code (optional)

    Returns:
        TranscriberBackend instance

    Raises:
        ValueError: If backend is not available or invalid
    """
    backend = backend.lower()

    if backend == "auto":
        # Priority: faster-whisper > whisper.cpp > transformers > whisper
        if FASTER_WHISPER_AVAILABLE:
            backend = "faster-whisper"
        elif WHISPER_CPP_AVAILABLE:
            backend = "whisper.cpp"
        elif TRANSFORMERS_AVAILABLE:
            backend = "transformers"
        else:
            backend = "whisper"

    if backend == "faster-whisper":
        if not FASTER_WHISPER_AVAILABLE:
            raise ValueError(
                "faster-whisper backend not available. Install with: pip install faster-whisper"
            )
        return FasterWhisperBackend(model_name, language)

    elif backend == "transformers":
        if not TRANSFORMERS_AVAILABLE:
            raise ValueError(
                "transformers backend not available. Install with: pip install transformers torch"
            )
        return TransformersBackend(model_name, language)

    elif backend == "whisper.cpp":
        if not WHISPER_CPP_AVAILABLE:
            raise ValueError(
                "whisper.cpp backend not available. Install with: pip install pywhispercpp"
            )
        return WhisperCppBackend(model_name, language)

    elif backend == "whisper":
        return WhisperBackend(model_name, language)

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Available: whisper, whisper.cpp, faster-whisper, transformers"
        )


def list_available_backends():
    """List available transcriber backends"""
    backends = ["whisper"]
    if FASTER_WHISPER_AVAILABLE:
        backends.append("faster-whisper")
    if WHISPER_CPP_AVAILABLE:
        backends.append("whisper.cpp")
    if TRANSFORMERS_AVAILABLE:
        backends.append("transformers")
    return backends


__all__ = [
    "TranscriberBackend",
    "WhisperBackend",
    "WhisperCppBackend",
    "FasterWhisperBackend",
    "TransformersBackend",
    "create_transcriber",
    "list_available_backends",
    "WHISPER_CPP_AVAILABLE",
    "FASTER_WHISPER_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
]
