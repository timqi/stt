# STT Server

OpenAI-compatible speech-to-text server, single-file Python. Supports Whisper (via `faster-whisper`) and FunASR's SeACo-Paraformer with hotword biasing.

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
- `hfd` for Whisper model downloads (<https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f>)
- `modelscope` CLI for FunASR model downloads — already bundled as a transitive dep of `funasr`, invoke as `uv run modelscope ...` after `uv sync --extra cuda`

## Install

```bash
uv sync --extra cuda
```

macOS/MLX users: `uv sync --extra mlx` — installs deps but the MLX backend is not implemented yet.

## Usage

```bash
# List built-in model aliases
uv run python server.py --list-models

# Download a Whisper model from HuggingFace
hfd mobiuslabsgmbh/faster-whisper-large-v3-turbo \
    --local-dir ~/models/stt/whisper-large-v3-turbo

# Or download a FunASR model from ModelScope (modelscope CLI ships with funasr)
uv run modelscope download \
    --model iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
    --local_dir ~/models/stt/paraformer-zh

# Start the server (it will print the exact download command if the model is missing)
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
| `paraformer-zh` | `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` (ModelScope) | funasr |

Whisper repos are pulled from HuggingFace (with `hfd`); FunASR repos are pulled from ModelScope (with `modelscope download`). The server prints the correct command per model when a download is needed.

`paraformer-zh` points at **SeACo-Paraformer-large**, FunASR's contextual ASR model for Mandarin. It supports hotword biasing — see [Hotwords](#hotwords) below.

Add your own in `MODELS` at the top of `server.py`.

## API

### `POST /v1/audio/transcriptions`

Form fields:

| Field | Type | Notes |
|---|---|---|
| `file` | file | Audio file (any format ffmpeg can decode) |
| `model` | str | Accepted for OpenAI compat, server is already bound to one model at startup |
| `language` | str | BCP-47 or ISO 639-1 (e.g. `zh`, `en`); `auto` for FunASR |
| `prompt` | str | Whisper: initial prompt text. `paraformer-zh` (SeACo): hotword list — see [Hotwords](#hotwords) |
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

## Hotwords

Hotwords (热词) bias the recognizer toward specific proper nouns, domain terms, or rare words — names, product names, jargon, etc. This fixes the common problem where a generic model transcribes `f2pool` as `F2 pool`, or `达摩院` as `大魔院`.

### Which models support hotwords

| Model | Hotword support | Mechanism |
|---|---|---|
| `paraformer-zh` (SeACo) | Yes — hard bias | Dedicated bias encoder, high recall for in-list words |
| `whisper-*` | No native hotwords | Only soft biasing via `prompt` (see below) |

Only `paraformer-zh` implements true hotword biasing. For Whisper, passing `prompt` acts as a conditioning text, which nudges vocabulary/style but does not guarantee any word shows up — effectiveness varies.

### How to pass hotwords

Hotwords are sent via the existing OpenAI-compatible `prompt` form field. No new API surface.

Format: a plain string of words separated by whitespace. Each "word" is one hotword candidate.

```
prompt = "达摩院 通义千问 f2pool Paraformer"
```

Rules:

- Whitespace-separated; the server passes the string verbatim to SeACo's `hotword` parameter.
- Recommended count: up to a few dozen. Throwing hundreds in makes the bias noisy and slows decoding.
- Keep each entry to one word or short phrase. Full sentences degrade recall.
- Works for Chinese, English, and mixed-language terms.
- Omit the field entirely when you don't need biasing — an empty or missing `prompt` disables hotword bias.

### Examples

**curl**

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@meeting.wav \
  -F model=paraformer-zh \
  -F prompt="达摩院 通义千问 f2pool Paraformer CTranslate2" \
  -F response_format=text
```

**Python (OpenAI SDK)**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")

with open("meeting.wav", "rb") as f:
    resp = client.audio.transcriptions.create(
        file=f,
        model="paraformer-zh",
        prompt="达摩院 通义千问 f2pool Paraformer CTranslate2",
        response_format="text",
    )
print(resp)
```

**Python (requests)**

```python
import requests

with open("meeting.wav", "rb") as f:
    r = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        files={"file": f},
        data={
            "model": "paraformer-zh",
            "prompt": "达摩院 通义千问 f2pool Paraformer CTranslate2",
            "response_format": "text",
        },
    )
print(r.text)
```

### Tips & limitations

- Hotwords boost recall for listed terms, but do not suppress everything else — normal words still transcribe normally.
- Homophones and near-homophones compete. If two hotwords sound alike, the model picks one; you can't force a specific one without more context.
- Single characters or very short tokens (1–2 chars) are risky — they over-trigger on unrelated audio. Prefer full words.
- Biasing strength is fixed by the model; there's no per-word weight knob in this API.
- Hotword lookup is case-sensitive for Latin characters. Pass the casing you want back.
- The same `prompt` is applied to the entire request. For multi-segment audio with different topics, that's a feature (broad coverage) or a drawback (cross-talk) depending on your data.

## Limitations

- Single model per process. Switch models by restarting with a different `--model`.
- Requests are serialized on the GPU (no batching).
- **SeACo-Paraformer** (`paraformer-zh`) works best on audio up to ~30s. Long audio needs a VAD model wired in — not implemented yet.
- FunASR doesn't emit segment timestamps through this server (yields the full text as one segment).
- `stream=true` streams *results* as the server transcribes an uploaded file; it does **not** accept live microphone input. For real-time streaming input, a WebSocket endpoint is planned but not implemented.
- MLX backends (`WhisperMLXBackend`, `FunASRMLXBackend`) are stubs that raise `NotImplementedError`.

## Roadmap

- WebSocket `/v1/realtime/transcription` for live input (mode B streaming)
- MLX backend for Apple Silicon
- VAD + punctuation pipeline for long-audio FunASR
- Whisper word-level timestamps in `verbose_json`
