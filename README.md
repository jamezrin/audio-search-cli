# Audio Search Tool

Search for words and phrases in audio files using OpenAI Whisper for transcription.

## Features

- **Efficient handling of long audio files**: Transcriptions are cached to avoid re-processing
- **Word-level timestamps**: Precise timestamp information for each match
- **Interactive & Headless modes**: Use interactively for multiple searches or headless for automation
- **Flexible search options**: Case-sensitive, regex patterns, customizable context
- **Multiple Whisper models**: Choose from tiny to large based on accuracy needs
- **Multi-language support**: Auto-detect or specify language

## Installation

### Using uv (recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install with standard backend (openai-whisper)
uv sync

# Or install with fast backend (whisper.cpp) - recommended for long audio
uv sync --extra fast
```

### Using pip

```bash
# Standard installation
pip install -r requirements.txt

# Or with whisper.cpp backend (5-10x faster)
pip install openai-whisper pywhispercpp
```

Note: First run will download the Whisper model (varies in size based on model chosen).

## Backend Selection

The tool supports two backends:

| Backend | Speed | Memory | Installation |
|---------|-------|--------|--------------|
| **whisper.cpp** | 5-10x faster | 50% less | `pip install pywhispercpp` |
| **openai-whisper** | Baseline | Baseline | `pip install openai-whisper` |

By default, the tool will automatically use whisper.cpp if available, falling back to openai-whisper otherwise. You can explicitly choose a backend with `--backend`.

## Usage

### Interactive Mode (Default)

Run multiple searches in one session:

```bash
# With uv
uv run audio_search.py audio.mp3

# Or with python directly
python audio_search.py audio.mp3
```

Then type your search queries and press Enter. Type `exit`, `quit`, or `q` to exit.

### Headless Mode

Single search, then exit (useful for scripts):

```bash
# With uv
uv run audio_search.py audio.mp3 --search "hello world" --headless

# Or with python directly
python audio_search.py audio.mp3 --search "hello world" --headless
```

### Examples

**Use whisper.cpp backend (faster):**
```bash
uv run audio_search.py audio.mp3 --backend whisper.cpp
```

**Force standard whisper backend:**
```bash
uv run audio_search.py audio.mp3 --backend whisper
```

**Search with a larger model for better accuracy:**
```bash
python audio_search.py audio.mp3 --model medium
```

**Case-sensitive search:**
```bash
python audio_search.py audio.mp3 --search "API" --case-sensitive --headless
```

**Regex search for variations:**
```bash
python audio_search.py audio.mp3 --regex --search "\b(hello|hi|hey)\b" --headless
```

**Specify language (faster processing):**
```bash
python audio_search.py audio.mp3 --language en
```

**More context around matches:**
```bash
python audio_search.py audio.mp3 --context 10
```

**No context, just timestamps:**
```bash
python audio_search.py audio.mp3 --no-context
```

**Clebackend`, `-b` | Backend: auto, whisper, whisper.cpp (default: auto) |
| `--ar cache and re-transcribe:**
```bash
python audio_search.py audio.mp3 --clear-cache
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `audio_file` | Path to audio file (required) |
| `--headless` | Run in headless mode (single search) |
| `--search`, `-s` | Search query (required in headless mode) |
| `--model`, `-m` | Whisper model: tiny, base, small, medium, large (default: base) |
| `--language`, `-l` | Language code (e.g., en, es, fr) |
| `--case-sensitive` | Make search case-sensitive |
| `--regex`, `-r` | Treat query as regex pattern |
| `--context`, `-c` | Words to show before/after match (default: 3) |
| `--no-context` | Don't show context around matches |
| `--no-cache` | Don't use cached transcription |
| `--clear-cache` | Clear cache and re-transcribe |

## How It Works

1. **Transcription**: Uses OpenAI Whisper to transcribe audio with word-level timestamps
2. **Caching**: Saves transcription to `~/.cache/audio_search/` to avoid re-processing
3. **Searching**: Searches through transcription using regex patterns
4. **Results**: Returns timestamps and context for each match

## Model Selection

- **tiny**: Fastest, least accurate (~1GB RAM)
- **base**: Good balance (default) (~1GB RAM)
- **small**: Better accuracy (~2GB RAM)
- **medium**: High accuracy (~5GB RAM)
- **large**: Best accuracy (~10GB RAM)

## Performance Notes

- **First run**: Will be slow as audio is transcribed
- **Subsequent runs**: Fast, uses cached transcription
- **Long files**: Transcription time scales linearly with audio length
- **Memory**: Larger models require more RAM

## Exit Codes (Headless Mode)

- `0`: Matches found
- `1`: No matches found or error

## Examples of Searches

**Find specific phrases:**
```
Search query: thank you very much
```

**Find words with variations (regex):**
```
Search query: \b(car|vehicle|automobile)\b
```

**Find numbers (regex):**
```
Search query: \d+
```

**Find questions:**
```
Search query: \?
```
