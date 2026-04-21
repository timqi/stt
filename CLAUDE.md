# STT Server — Project Notes for Claude

Single-file OpenAI-compatible STT server. Python + FastAPI. Linux/CUDA today; macOS/MLX stubbed for later.

## Layout

```
stt/
├── server.py         # all runtime code (backends, API, CLI)
├── pyproject.toml    # deps; extras: [cuda] for Linux, [mlx] for macOS
├── README.md         # user-facing docs
└── CLAUDE.md         # this file
```

Keep it a single runtime file. If `server.py` grows past ~600 lines, split along backend boundaries (`backends/whisper.py`, `backends/funasr.py`) before splitting anything else.

## Architecture

### Backend Protocol

Every backend implements:

```python
class STTBackend(Protocol):
    name: str
    def transcribe_stream(audio_path, language, prompt, temperature) -> Iterator[Segment]
```

`Segment` is a dict with `start`, `end`, `text`. FunASR backends return a single segment containing the full text (start=end=0.0).

Only **streaming** is part of the protocol. Non-streaming is just `list(transcribe_stream(...))`. Do **not** add a separate `transcribe()` method.

### Backends

| Class | Platform | Library | Status |
|---|---|---|---|
| `WhisperFasterBackend` | Linux/CUDA | `faster-whisper` (CTranslate2) | implemented |
| `FunASRBackend` | Linux/CUDA | `funasr` | implemented |
| `WhisperMLXBackend` | macOS | `mlx-whisper` | stub, raises `NotImplementedError` |
| `FunASRMLXBackend` | macOS | TBD (ONNX/CoreML bridge) | stub |

### Platform detection

`detect_device(requested)` in `server.py`:
- `auto` + Linux + `torch.cuda.is_available()` → `cuda`
- `auto` + Darwin → `mlx`
- else → raise

Never silently fall back to CPU. Inference on CPU is too slow to be useful and hides real configuration problems.

### Model registry

`MODELS` dict at the top of `server.py`. Each entry:

```python
"alias": {"repo": "org/repo-id", "backend": "whisper" | "funasr"}
```

Repo sources split by backend:
- `whisper` → HuggingFace. Must be **CTranslate2-converted** (e.g. `Systran/faster-whisper-*`, `mobiuslabsgmbh/faster-whisper-*`), not raw `openai/whisper-*` — faster-whisper can't load the raw PyTorch weights directly.
- `funasr` → ModelScope (`iic/...`). FunASR only publishes complete weights there; HF mirrors under `funasr/` exist but are empty placeholders.

## Model storage

Convention: `~/models/stt/<alias>/` — one directory per model, contents are whatever the downloader fetched.

When a model is missing, `create_backend()` prints the download command to stderr and `sys.exit(1)`. The command differs by backend: `hfd <repo> --local-dir <path>` for Whisper, `uv run modelscope download --model <repo> --local_dir <path>` for FunASR (the `modelscope` CLI is a transitive dep of `funasr`, so no separate install is needed). **Do not add auto-download.** The user wants downloads to be an explicit manual step. If you change this rule, get the user's sign-off first.

## API surface

OpenAI compatibility is a deliberate constraint — don't drift from their schema.

- `POST /v1/audio/transcriptions` — multipart form, matches OpenAI field names
- `GET /v1/models` — returns the single loaded model
- `GET /health` — internal liveness probe

`response_format` values: `json` (default), `text`, `verbose_json`. `srt`/`vtt` are not implemented.

### Streaming

`stream=true` returns SSE. Event shape follows OpenAI's gpt-4o-transcribe format:

```
data: {"type":"transcript.text.delta","delta":"..."}
data: {"type":"transcript.text.done","text":"..."}
data: [DONE]
```

Implementation detail: faster-whisper's `transcribe()` returns a blocking Python generator. `_run_in_thread_iter()` adapts it to an async iterator via `queue.Queue` + a daemon thread. Don't replace this with `asyncio.to_thread` wrapping the whole list — that defeats streaming.

FunASR is not natively streaming; it yields one big chunk. That's fine, the API still works.

## What NOT to add

- **Translations** (`/v1/audio/translations`) — user explicitly declined.
- **Auto-download** of models — user wants `hfd` command printed instead.
- **CPU fallback** — Linux path requires CUDA; fail loudly if absent.
- **SRT/VTT formats** — not requested; `verbose_json` has segments if needed.
- **Multi-model hot-swap** — single model per process is intentional.
- **Authentication** — out of scope; local-network tool.

## What's explicitly planned

- Mode B streaming: WebSocket endpoint for real-time input. The backend Protocol already accommodates this (it's streaming-first). When adding it, integrate `whisper-streaming` (LocalAgreement-2) for Whisper and `paraformer-zh-streaming` for FunASR.
- MLX backends — replace the stubs with `mlx-whisper` and a FunASR bridge.
- VAD + punctuation pipeline for long-audio FunASR.

## Commands

```bash
# Install (Linux/CUDA)
uv sync --extra cuda

# Run
uv run python server.py --model <alias>

# List models
uv run python server.py --list-models

# Syntax check without deps
python3 -c "import ast; ast.parse(open('server.py').read())"
```

## Testing notes

No test suite yet. When adding one, priorities:

1. Backend protocol contract — a mock backend that yields known segments, verify `/v1/audio/transcriptions` produces correct JSON/text/verbose_json.
2. SSE format — `stream=true` emits the right event sequence ending in `[DONE]`.
3. Missing-model path — `create_backend()` exits with code 1 and prints an `hfd ...` line.

Don't write tests that require downloading real models in CI; mock the backend.

## Dependency policy

- Base deps (fastapi, uvicorn, multipart, numpy) stay in `[project.dependencies]`.
- Platform-specific heavy deps go in `[project.optional-dependencies]` — `cuda` or `mlx`. Users install only one extra.
- Don't add `torch` to base deps; let the extras pull it in.
- `funasr` pulls in a lot — keep it in `cuda` extra only.

## Style

- No emojis in code or logs (global rule).
- Minimal comments. Only explain non-obvious *why*.
- All code and documentation are English — identifiers, comments, docstrings, CLI help, log messages, error strings, README, this file. No Chinese in committed files. (AI-to-user conversation may use either language.)
- Error messages printed to stderr with a `[stt]` prefix or a visible separator for download hints.
