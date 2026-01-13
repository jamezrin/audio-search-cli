#!/usr/bin/env python3
"""
Audio Search Tool - Search for words/phrases in audio files using OpenAI Whisper

This script transcribes audio files using Whisper and allows searching for specific
words or phrases, returning timestamps where they occur.

Usage:
    # Interactive mode (default)
    python audio_search.py audio.mp3
    
    # Headless mode (single search)
    python audio_search.py audio.mp3 --search "hello world" --headless
    
    # With custom model
    python audio_search.py audio.mp3 --model medium --language en
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from transcribers import create_transcriber, list_available_backends, WHISPER_CPP_AVAILABLE


class AudioTranscriber:
    """Handles audio transcription using pluggable backends"""
    
    def __init__(
        self, 
        model_name: str = "base", 
        language: Optional[str] = None,
        backend: str = "auto"
    ):
        """
        Initialize the transcriber with a backend.
        
        Args:
            model_name: Model size (tiny, base, small, medium, large)
            language: Language code (e.g., 'en', 'es', 'fr'). None for auto-detection.
            backend: Backend to use ("whisper", "whisper.cpp", or "auto")
        """
        self.backend = create_transcriber(backend, model_name, language)
        self.backend_name = self.backend.get_backend_name()
        print(f"Using backend: {self.backend_name}", file=sys.stderr)
        
        # Load model during initialization
        self.backend.load_model()
    
    def transcribe(self, audio_path: str, use_cache: bool = True) -> Dict:
        """
        Transcribe audio file with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            use_cache: Whether to use cached transcription if available
            
        Returns:
            Dictionary containing transcription with word-level timestamps
        """
        cache_path = self._get_cache_path(audio_path)
        
        # Check cache first
        if use_cache and cache_path.exists():
            print(f"Loading cached transcription from {cache_path}", file=sys.stderr)
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        print(f"Transcribing {audio_path}... (this may take a while)", file=sys.stderr)
        
        # Transcribe using backend
        result = self.backend.transcribe(audio_path)
        
        # Save to cache
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Transcription cached to {cache_path}", file=sys.stderr)
        
        return result
    
    def _get_cache_path(self, audio_path: str) -> Path:
        """Get the cache file path for a given audio file"""
        audio_file = Path(audio_path)
        cache_dir = Path.home() / '.cache' / 'audio_search'
        # Include backend name in cache to avoid conflicts
        return cache_dir / f"{audio_file.stem}.{self.backend_name}.transcription.json"


class TranscriptionSearcher:
    """Searches through transcription for words/phrases"""
    
    def __init__(self, transcription: Dict):
        """
        Initialize searcher with transcription data.
        
        Args:
            transcription: Whisper transcription result with word timestamps
        """
        self.transcription = transcription
        self._build_word_index()
    
    def _build_word_index(self):
        """Build an index of words with their timestamps"""
        self.words = []
        
        for segment in self.transcription.get('segments', []):
            # Use word-level timestamps if available
            if 'words' in segment:
                for word_info in segment['words']:
                    self.words.append({
                        'word': word_info.get('word', '').strip(),
                        'start': word_info.get('start', segment['start']),
                        'end': word_info.get('end', segment['end'])
                    })
            else:
                # Fall back to segment-level timestamps
                text = segment['text'].strip()
                self.words.append({
                    'word': text,
                    'start': segment['start'],
                    'end': segment['end']
                })
    
    def search(
        self, 
        query: str, 
        case_sensitive: bool = False,
        regex: bool = False,
        context_words: int = 3
    ) -> List[Dict]:
        """
        Search for a word or phrase in the transcription.
        
        Args:
            query: Word or phrase to search for
            case_sensitive: Whether search should be case-sensitive
            regex: Whether to treat query as a regex pattern
            context_words: Number of words to include before/after match for context
            
        Returns:
            List of matches with timestamps and context
        """
        matches = []
        
        if not query.strip():
            return matches
        
        # Prepare search pattern
        if regex:
            try:
                pattern = re.compile(query, 0 if case_sensitive else re.IGNORECASE)
            except re.error as e:
                print(f"Invalid regex pattern: {e}", file=sys.stderr)
                return matches
        else:
            # Escape special regex characters for literal search
            escaped_query = re.escape(query)
            pattern = re.compile(escaped_query, 0 if case_sensitive else re.IGNORECASE)
        
        # Search through words
        word_texts = [w['word'] for w in self.words]
        full_text = ' '.join(word_texts)
        
        # Find all matches in the full text
        for match in pattern.finditer(full_text):
            start_char = match.start()
            end_char = match.end()
            
            # Find which words correspond to these character positions
            word_start_idx, word_end_idx = self._char_to_word_indices(
                word_texts, start_char, end_char
            )
            
            if word_start_idx is not None and word_end_idx is not None:
                # Get context
                context_start_idx = max(0, word_start_idx - context_words)
                context_end_idx = min(len(self.words), word_end_idx + 1 + context_words)
                
                match_info = {
                    'query': query,
                    'matched_text': match.group(),
                    'start_time': self.words[word_start_idx]['start'],
                    'end_time': self.words[word_end_idx]['end'],
                    'context': ' '.join(
                        w['word'] for w in self.words[context_start_idx:context_end_idx]
                    ),
                    'word_index': word_start_idx
                }
                matches.append(match_info)
        
        return matches
    
    def _char_to_word_indices(
        self, 
        word_texts: List[str], 
        start_char: int, 
        end_char: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """Convert character positions to word indices"""
        current_pos = 0
        start_idx = None
        end_idx = None
        
        for i, word in enumerate(word_texts):
            word_start = current_pos
            word_end = current_pos + len(word)
            
            # Check if this word overlaps with the match
            if start_idx is None and start_char >= word_start and start_char < word_end + 1:
                start_idx = i
            
            if end_char > word_start and end_char <= word_end + 1:
                end_idx = i
                break
            
            current_pos = word_end + 1  # +1 for space
        
        # If end_idx not found, it might be the last word
        if start_idx is not None and end_idx is None:
            end_idx = len(word_texts) - 1
        
        return start_idx, end_idx


def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def print_search_results(matches: List[Dict], show_context: bool = True):
    """Print search results in a readable format"""
    if not matches:
        print("No matches found.")
        return
    
    print(f"\nFound {len(matches)} match(es):\n")
    
    for i, match in enumerate(matches, 1):
        start_ts = format_timestamp(match['start_time'])
        end_ts = format_timestamp(match['end_time'])
        
        print(f"Match {i}:")
        print(f"  Time: {start_ts} - {end_ts}")
        print(f"  Matched: \"{match['matched_text']}\"")
        
        if show_context:
            print(f"  Context: {match['context']}")
        
        print()


def interactive_mode(searcher: TranscriptionSearcher, args: argparse.Namespace):
    """Run in interactive mode allowing multiple searches"""
    print("\n" + "="*70)
    print("Interactive Search Mode")
    print("="*70)
    print("Type your search query and press Enter.")
    print("Type 'exit', 'quit', or 'q' to exit.")
    print("Type 'help' for available commands.")
    print("="*70 + "\n")
    
    while True:
        try:
            query = input("Search query: ").strip()
            
            if not query:
                continue
            
            # Check for exit commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("Exiting...")
                break
            
            # Check for help command
            if query.lower() == 'help':
                print("\nAvailable commands:")
                print("  exit, quit, q - Exit the program")
                print("  help          - Show this help message")
                print("\nSearch options (set via command-line flags):")
                print(f"  Case sensitive: {args.case_sensitive}")
                print(f"  Regex mode: {args.regex}")
                print(f"  Context words: {args.context}")
                print()
                continue
            
            # Perform search
            matches = searcher.search(
                query,
                case_sensitive=args.case_sensitive,
                regex=args.regex,
                context_words=args.context
            )
            
            print_search_results(matches, show_context=args.show_context)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except EOFError:
            print("\nExiting...")
            break


def headless_mode(searcher: TranscriptionSearcher, args: argparse.Namespace):
    """Run in headless mode with a single search"""
    if not args.search:
        print("Error: --search query is required in headless mode", file=sys.stderr)
        sys.exit(1)
    
    matches = searcher.search(
        args.search,
        case_sensitive=args.case_sensitive,
        regex=args.regex,
        context_words=args.context
    )
    
    print_search_results(matches, show_context=args.show_context)
    
    # Exit with code 0 if matches found, 1 if not
    sys.exit(0 if matches else 1)


def main():
    parser = argparse.ArgumentParser(
        description="Search for words/phrases in audio files using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  %(prog)s audio.mp3
  
  # Headless mode with search
  %(prog)s audio.mp3 --search "hello world" --headless
  
  # Case-sensitive search with larger model
  %(prog)s audio.mp3 --model medium --case-sensitive --search "API"
  
  # Regex search
  %(prog)s audio.mp3 --regex --search "\\b(hello|hi)\\b" --headless
        """
    )
    
    # Required arguments
    parser.add_argument(
        'audio_file',
        help='Path to the audio file (mp3, mp4, wav, etc.)'
    )
    
    # Mode selection
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run in headless mode (single search, then exit)'
    )
    
    parser.add_argument(
        '--search', '-s',
        help='Search query (required in headless mode)'
    )
    
    # Backend selection
    parser.add_argument(
        '--backend', '-b',
        default='auto',
        choices=['auto', 'whisper', 'whisper.cpp'],
        help='Transcription backend (default: auto - prefers whisper.cpp if available)'
    )
    
    # Whisper model options
    parser.add_argument(
        '--model', '-m',
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size (default: base)'
    )
    
    parser.add_argument(
        '--language', '-l',
        help='Language code (e.g., en, es, fr). Auto-detect if not specified.'
    )
    
    # Search options
    parser.add_argument(
        '--case-sensitive',
        action='store_true',
        help='Make search case-sensitive'
    )
    
    parser.add_argument(
        '--regex', '-r',
        action='store_true',
        help='Treat search query as a regex pattern'
    )
    
    parser.add_argument(
        '--context', '-c',
        type=int,
        default=3,
        help='Number of words to show before/after match (default: 3)'
    )
    
    parser.add_argument(
        '--no-context',
        dest='show_context',
        action='store_false',
        help='Do not show context around matches'
    )
    
    # Cache options
    parser.add_argument(
        '--no-cache',
        dest='use_cache',
        action='store_false',
        help='Do not use or create transcription cache'
    )
    
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear the cached transcription for this file and re-transcribe'
    )
    
    args = parser.parse_args()
    
    # Show available backends and GPU info
    available_backends = list_available_backends()
    print(f"Available backends: {', '.join(available_backends)}", file=sys.stderr)
    if WHISPER_CPP_AVAILABLE:
        print("✓ whisper.cpp is available (recommended for speed)", file=sys.stderr)
    else:
        print("ℹ whisper.cpp not available (install with: pip install pywhispercpp)", file=sys.stderr)
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ NVIDIA GPU available: {torch.cuda.get_device_name(0)}", file=sys.stderr)
        elif hasattr(torch, 'hip') and torch.hip.is_available():
            print(f"✓ AMD GPU (ROCm) available", file=sys.stderr)
        else:
            print("ℹ No GPU detected - using CPU", file=sys.stderr)
    except ImportError:
        print("ℹ PyTorch not installed - GPU detection unavailable", file=sys.stderr)
    
    # Validate audio file exists
    if not os.path.isfile(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize transcriber
    try:
        transcriber = AudioTranscriber(args.model, args.language, args.backend)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Clear cache if requested
    if args.clear_cache:
        cache_path = transcriber._get_cache_path(args.audio_file)
        if cache_path.exists():
            cache_path.unlink()
            print(f"Cleared cache: {cache_path}", file=sys.stderr)
    
    # Transcribe audio
    transcription = transcriber.transcribe(args.audio_file, use_cache=args.use_cache)
    
    # Initialize searcher
    searcher = TranscriptionSearcher(transcription)
    
    print(f"Transcription ready. Total segments: {len(transcription.get('segments', []))}", file=sys.stderr)
    print(f"Total duration: {format_timestamp(transcription.get('duration', 0))}", file=sys.stderr)
    
    # Run in appropriate mode
    if args.headless:
        headless_mode(searcher, args)
    else:
        interactive_mode(searcher, args)


if __name__ == '__main__':
    main()
