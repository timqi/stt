# STT Server

OpenAI-compatible speech-to-text server, single-file Python. Supports the Whisper family (via `faster-whisper`) and FunASR family (Paraformer / SenseVoice).

## Features

- **OpenAI-compatible API** — drop-in replacement for `/v1/audio/transcriptions`, works with the official OpenAI SDK or plain `curl`.
- **Streaming** — `stream=true` returns Server-Sent Events (`transcript.text.delta` / `transcript.text.done`), matching OpenAI's protocol.
- **Multi-backend** — Whisper (CTranslate2) and FunASR under one abstraction; swap with `--model`.
- **Platform-aware** — auto-selects CUDA on Linux; leaves room for MLX on Apple Silicon (stubbed).
- **Manual model downloads** — prints the exact `hfd` command when a model is missing; no surprise network I/O.

## Requirements

- Python ≥ 3.10
- Linux with NVIDIA GPU + CUDA (current supported platform)
- `uv` for dependency management
- `hfd` for model downloads (<https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f>)

## Install

```bash
uv sync --extra cuda
```

macOS/MLX users: `uv sync --extra mlx` — installs deps but the MLX backend is not implemented yet.

## Usage

```bash
# List built-in model aliases
uv run python server.py --list-models

# Download a model (follow the printed hint)
hfd mobiuslabsgmbh/faster-whisper-large-v3-turbo \
    --local-dir ~/models/stt/whisper-large-v3-turbo

# Start the server
uv run python server.py --model whisper-large-v3-turbo --port 8000
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Model alias from registry |
| `--host` | `0.0.0.0` | Bind host |
| `--port` | `8000` | Bind port |
| `--models-dir` | `~/models/stt` | Where models live on disk |
| `--device` | `auto` | `auto` / `cuda` / `mlx` |
| `--compute-type` | `float16` | faster-whisper precision: `float16` / `int8_float16` / `int8` / `float32` |
| `--list-models` | — | Print the registry and exit |

## Model registry

| Alias | HF repo | Backend |
|---|---|---|
| `whisper-large-v3-turbo` | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | whisper |
| `whisper-large-v3` | `Systran/faster-whisper-large-v3` | whisper |
| `whisper-medium` | `Systran/faster-whisper-medium` | whisper |
| `sensevoice-small` | `FunAudioLLM/SenseVoiceSmall` | funasr |
| `paraformer-zh` | `funasr/paraformer-zh` | funasr |

Add your own in `MODELS` at the top of `server.py`.

## API

### `POST /v1/audio/transcriptions`

Form fields:

| Field | Type | Notes |
|---|---|---|
| `file` | file | Audio file (any format ffmpeg can decode) |
| `model` | str | Accepted for OpenAI compat, server is already bound to one model at startup |
| `language` | str | BCP-47 or ISO 639-1 (e.g. `zh`, `en`); `auto` for FunASR |
| `prompt` | str | Initial prompt for Whisper (ignored by FunASR) |
| `response_format` | str | `json` (default) / `text` / `verbose_json` |
| `temperature` | float | Default `0.0` |
| `stream` | bool | `true` to stream SSE |

### `GET /v1/models`

Returns the single currently-loaded model.

### `GET /health`

Liveness probe.

## Examples

### Non-streaming

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=whisper-large-v3-turbo \
  -F response_format=verbose_json
```

### Streaming (SSE)

```bash
curl -N http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=whisper-large-v3-turbo \
  -F stream=true
```

SSE events:

```
data: {"type":"transcript.text.delta","delta":"Hello "}
data: {"type":"transcript.text.delta","delta":"world."}
data: {"type":"transcript.text.done","text":"Hello world."}
data: [DONE]
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")

with open("sample.wav", "rb") as f:
    resp = client.audio.transcriptions.create(
        file=f,
        model="whisper-large-v3-turbo",
        response_format="text",
    )
print(resp)
```

## Limitations

- Single model per process. Switch models by restarting with a different `--model`.
- Requests are serialized on the GPU (no batching).
- **SenseVoice** raw model handles audio up to ~30s. Long audio needs a VAD model wired in — not implemented yet.
- FunASR doesn't emit segment timestamps through this server (yields the full text as one segment).
- `stream=true` streams *results* as the server transcribes an uploaded file; it does **not** accept live microphone input. For real-time streaming input, a WebSocket endpoint is planned but not implemented.
- MLX backends (`WhisperMLXBackend`, `FunASRMLXBackend`) are stubs that raise `NotImplementedError`.

## Roadmap

- WebSocket `/v1/realtime/transcription` for live input (mode B streaming)
- MLX backend for Apple Silicon
- VAD integration for long-audio SenseVoice
- Whisper word-level timestamps in `verbose_json`
