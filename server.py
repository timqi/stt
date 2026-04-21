"""OpenAI-compatible STT server supporting Whisper (faster-whisper) and FunASR."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import queue
import re
import sys
import tempfile
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator, Optional, Protocol

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse


# ---------------------------------------------------------------------------
# CUDA 12 library preload
# ---------------------------------------------------------------------------
# CTranslate2 (faster-whisper's runtime) is built against CUDA 12 and looks
# for libcublas.so.12 / libcudnn.so.9. Newer torch wheels ship CUDA 13 libs
# instead, which don't satisfy that. Preload the cu12 libs installed via the
# nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels so faster-whisper can find
# them without the user having to set LD_LIBRARY_PATH.


def _preload_cuda12_libs() -> None:
    if platform.system() != "Linux":
        return
    import ctypes

    try:
        import nvidia.cublas
        import nvidia.cudnn
    except ImportError:
        return
    targets = [
        (Path(nvidia.cublas.__path__[0]) / "lib", "libcublas*.so.12"),
        (Path(nvidia.cudnn.__path__[0]) / "lib", "libcudnn*.so.9"),
    ]
    for d, pattern in targets:
        for lib in sorted(d.glob(pattern)):
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                print(f"[stt] failed to preload {lib.name}: {e}", file=sys.stderr)


_preload_cuda12_libs()


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS: dict[str, dict[str, str]] = {
    "whisper-large-v3-turbo": {
        "repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "backend": "whisper",
    },
    "whisper-large-v3": {
        "repo": "Systran/faster-whisper-large-v3",
        "backend": "whisper",
    },
    "paraformer-zh": {
        "repo": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "backend": "funasr",
    },
}

# Auxiliary models (punctuation, VAD, speaker, ...) shared across ASR backends.
# Directory convention mirrors MODELS: `~/models/stt/<alias>/`.
AUX_MODELS: dict[str, str] = {
    "ct-punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
}


# ---------------------------------------------------------------------------
# Backend protocol & implementations
# ---------------------------------------------------------------------------


class Segment(dict):
    """A transcription segment: {start, end, text}."""


class STTBackend(Protocol):
    name: str

    def transcribe_stream(
        self,
        audio_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Iterator[Segment]:
        ...


class WhisperFasterBackend:
    """faster-whisper (CTranslate2) backend, CUDA."""

    name = "whisper-faster"

    def __init__(self, model_path: str, device: str = "cuda", compute_type: str = "float16"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_path, device=device, compute_type=compute_type)

    def transcribe_stream(self, audio_path, language=None, prompt=None, temperature=0.0):
        segments, _info = self.model.transcribe(
            audio_path,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            vad_filter=True,
        )
        for seg in segments:
            yield Segment(start=seg.start, end=seg.end, text=seg.text)


# Paraformer tokenises at character level, so raw output looks like
# "你 好 世 界". Collapse spaces between two non-ASCII chars — covers CJK
# ideographs, CJK punctuation (U+3000–303F), and full-width forms
# (U+FF00–FFEF) emitted by ct-punc. ASCII words (English, digits) keep
# their surrounding spaces.
_CJK = r"[　-〿㐀-鿿＀-￯]"
_CJK_SPACE = re.compile(rf"(?<={_CJK})\s+(?={_CJK})")


class FunASRBackend:
    """FunASR backend (SeACo-Paraformer + ct-punc), CUDA."""

    name = "funasr"

    def __init__(self, model_path: str, punc_path: str, device: str = "cuda"):
        from funasr import AutoModel

        self.model = AutoModel(
            model=model_path,
            punc_model=punc_path,
            device=device,
            disable_update=True,
        )

    def transcribe_stream(self, audio_path, language=None, prompt=None, temperature=0.0):
        kwargs = {"input": audio_path}
        if language:
            kwargs["language"] = language
        # For SeACo / Contextual Paraformer, `prompt` is repurposed as a hotword
        # string (space-separated). The base paraformer-zh model ignores it.
        if prompt:
            kwargs["hotword"] = prompt
        results = self.model.generate(**kwargs)
        if not results:
            return
        text = _CJK_SPACE.sub("", results[0].get("text", ""))
        yield Segment(start=0.0, end=0.0, text=text)


class WhisperMLXBackend:
    """Placeholder for Apple Silicon MLX implementation."""

    name = "whisper-mlx"

    def __init__(self, model_path: str):
        raise NotImplementedError("MLX backend not implemented yet")

    def transcribe_stream(self, *args, **kwargs):
        raise NotImplementedError


class FunASRMLXBackend:
    """Placeholder — FunASR has no official MLX port; may need ONNX + CoreML bridge."""

    name = "funasr-mlx"

    def __init__(self, model_path: str):
        raise NotImplementedError("FunASR on MLX not implemented yet")

    def transcribe_stream(self, *args, **kwargs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Device detection & backend factory
# ---------------------------------------------------------------------------


def detect_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    system = platform.system()
    if system == "Linux":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        raise RuntimeError(
            "No CUDA device detected on Linux. Run `uv sync --extra cuda` "
            "and make sure NVIDIA driver + CUDA runtime are installed."
        )
    if system == "Darwin":
        return "mlx"
    raise RuntimeError(f"Unsupported platform: {system}")


def print_download_hint(
    model_name: str, repo_id: str, model_path: Path, backend_type: str
) -> None:
    bar = "=" * 64
    print(bar, file=sys.stderr)
    print(f"Model `{model_name}` not found at: {model_path}", file=sys.stderr)
    print("", file=sys.stderr)
    # FunASR models live on ModelScope (repo ids like `iic/...`); Whisper
    # CTranslate2 weights live on HuggingFace.
    if backend_type == "funasr":
        print("Download it first with modelscope (bundled with funasr):", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            f"    uv run modelscope download --model {repo_id} --local_dir {model_path}",
            file=sys.stderr,
        )
    else:
        print("Download it first with hfd (HuggingFace CLI download accelerator):", file=sys.stderr)
        print("", file=sys.stderr)
        print(f"    hfd {repo_id} --local-dir {model_path}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Then restart this service.", file=sys.stderr)
    print(bar, file=sys.stderr)


def _warmup(backend: STTBackend) -> None:
    """Run 1s of low-amplitude noise to trigger CUDA kernel compile, cuDNN autotune,
    and lazy sub-model loading. Pure silence gets filtered out by VAD and never
    reaches the decoder, so we use low-amplitude noise instead."""
    import wave

    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        rng = np.random.default_rng(0)
        samples = (rng.standard_normal(16000) * 500).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(samples.tobytes())
        list(backend.transcribe_stream(path))
    except Exception as e:
        print(f"[stt] warmup failed (ignored): {e}", file=sys.stderr)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _missing(path: Path) -> bool:
    return not path.exists() or not any(path.iterdir())


def create_backend(
    model_name: str, models_dir: Path, device: str, compute_type: str
) -> STTBackend:
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODELS.keys())}"
        )

    info = MODELS[model_name]
    repo_id, backend_type = info["repo"], info["backend"]
    model_path = models_dir / model_name

    # Collect required paths: main ASR + backend-specific aux models.
    required = [(model_name, repo_id, model_path, backend_type)]
    punc_path = models_dir / "ct-punc"
    if backend_type == "funasr":
        required.append(("ct-punc", AUX_MODELS["ct-punc"], punc_path, "funasr"))

    missing = [r for r in required if _missing(r[2])]
    if missing:
        for name, repo, path, btype in missing:
            print_download_hint(name, repo, path, btype)
        sys.exit(1)

    if device == "cuda":
        if backend_type == "whisper":
            return WhisperFasterBackend(str(model_path), device="cuda", compute_type=compute_type)
        if backend_type == "funasr":
            return FunASRBackend(str(model_path), str(punc_path), device="cuda")
    elif device == "mlx":
        if backend_type == "whisper":
            return WhisperMLXBackend(str(model_path))
        if backend_type == "funasr":
            return FunASRMLXBackend(str(model_path))

    raise ValueError(f"Cannot create backend for device={device}, backend={backend_type}")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


# Global backend handle; populated by main() before uvicorn.run
_backend: Optional[STTBackend] = None
_model_name: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _backend is None:
        raise RuntimeError("Backend not initialized — call main() before serving.")
    yield


app = FastAPI(title="STT Server", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": _model_name}


@app.get("/v1/models")
async def list_loaded_models():
    return {
        "object": "list",
        "data": [{"id": _model_name, "object": "model", "owned_by": "local"}],
    }


def _run_in_thread_iter(gen_fn):
    """Convert a blocking generator into an async iterator via a background thread."""
    q: queue.Queue = queue.Queue(maxsize=32)
    SENTINEL = object()

    def producer():
        try:
            for item in gen_fn():
                q.put(item)
        except Exception as e:
            q.put(e)
        finally:
            q.put(SENTINEL)

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    async def aiter():
        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is SENTINEL:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    return aiter()


async def _sse_transcribe(
    audio_path: str, language: Optional[str], prompt: Optional[str], temperature: float
):
    full_text = ""
    try:
        async for seg in _run_in_thread_iter(
            lambda: _backend.transcribe_stream(audio_path, language, prompt, temperature)
        ):
            delta = seg["text"]
            full_text += delta
            event = {"type": "transcript.text.delta", "delta": delta}
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        done = {"type": "transcript.text.done", "text": full_text}
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        traceback.print_exc()
        err = {"type": "error", "error": {"message": str(e)}}
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    stream: bool = Form(False),
):
    if model and model != _model_name:
        # OpenAI protocol requires a model field, but this process only loads one.
        # Serve the request anyway; the client just can't switch models here.
        pass

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    async def cleanup_after(gen):
        try:
            async for chunk in gen:
                yield chunk
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if stream:
        return StreamingResponse(
            cleanup_after(_sse_transcribe(tmp_path, language, prompt, temperature)),
            media_type="text/event-stream",
        )

    try:
        segments = await asyncio.to_thread(
            lambda: list(_backend.transcribe_stream(tmp_path, language, prompt, temperature))
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    full_text = "".join(s["text"] for s in segments)

    if response_format == "text":
        return PlainTextResponse(full_text)
    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": language,
            "text": full_text,
            "segments": [
                {"id": i, "start": s["start"], "end": s["end"], "text": s["text"]}
                for i, s in enumerate(segments)
            ],
        }
    return {"text": full_text}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible STT server")
    parser.add_argument("--model", help="Model alias, e.g. whisper-large-v3-turbo / paraformer-zh")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--models-dir",
        default=str(Path.home() / "models" / "stt"),
        help="Model root directory (default ~/models/stt)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mlx"],
        help="Inference device; auto-detected per platform by default",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="faster-whisper precision: float16 / int8_float16 / int8 / float32",
    )
    parser.add_argument("--list-models", action="store_true", help="List built-in models")
    args = parser.parse_args()

    if args.list_models:
        print(f"{'NAME':<28} {'BACKEND':<10} REPO")
        for name, info in MODELS.items():
            print(f"{name:<28} {info['backend']:<10} {info['repo']}")
        return

    if not args.model:
        parser.error("Missing --model (use --list-models to see available models)")

    models_dir = Path(args.models_dir).expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)

    device = detect_device(args.device)
    print(f"[stt] device: {device}  model: {args.model}  dir: {models_dir}", file=sys.stderr)

    global _backend, _model_name
    _backend = create_backend(args.model, models_dir, device, args.compute_type)
    _model_name = args.model
    print("[stt] model loaded, starting warmup...", file=sys.stderr)
    _warmup(_backend)
    print(f"[stt] warmup done, listening on {args.host}:{args.port}", file=sys.stderr)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
