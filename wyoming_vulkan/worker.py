"""Text-to-speech models in their own processes: one worker process per model.

Why: every process gets its own GPU context, so each model can use any GPU (ONNX Runtime's WebGPU plug-in runs all
sessions of one process on one GPU); a crash in a model's runtime or driver takes down only its worker; and stopping a
worker really returns its memory (on an iGPU that is the host's RAM). Cost: ~0.2 s and ~45 MB per worker (measured).

`WorkerEngine` stands in for the engine inside the server: it starts `python -m wyoming_vulkan.worker`, which loads
the real engine (with its GPU proof and warm-up), and forwards each sentence to it. Protocol over the worker's
stdin/stdout, one JSON line per message, audio as raw bytes after its header line:
  server -> worker   {"config": {...}, "gpu": {...}}                     once, then per sentence
                     {"text": ..., "voice": ..., "speaker": ..., "settings": {...}, "stream": bool}
  worker -> server   {"type": "ready", "runtime": {...}, "sample_rate": n}  or  {"type": "error", "gpu": bool, ...}
                     per sentence: {"type": "audio", "size": n} + n bytes of int16 PCM (repeated), then
                     {"type": "done"} or {"type": "error", "message": ...}
The worker exits when its stdin closes (the server stopped it or died), closing the engine first.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterator, Optional

from .config import Attribution, EngineConfig, GpuConfig
from .engines.base import GpuUnavailable, RuntimeReport, SynthesisOptions, TtsPack

_LOGGER = logging.getLogger(__name__)

STOP_TIMEOUT_S = 20.0


def _config_to_json(config: EngineConfig, gpu: GpuConfig) -> dict:
    data = asdict(config)
    data["model"] = str(config.model)
    return {"config": data, "gpu": asdict(gpu)}


def _config_from_json(message: dict) -> tuple[EngineConfig, GpuConfig]:
    data = dict(message["config"])
    data["model"] = Path(data["model"])
    if data.get("attribution"):
        data["attribution"] = Attribution(**data["attribution"])
    return EngineConfig(**data), GpuConfig(**message["gpu"])


class WorkerEngine(TtsPack):
    """A TTS model (plain voice or voice pack) running in a worker process."""

    def __init__(self, config: EngineConfig, gpu: GpuConfig) -> None:
        from .engines import create_engine

        super().__init__(config, gpu.select(config.gpu))
        self.base_gpu = gpu  # the worker applies the engine's selector itself
        self.inner = create_engine(config, gpu)  # only for what is known without loading (voices, languages)
        self.proc: Optional[subprocess.Popen] = None
        self._rate: Optional[int] = None
        self.lock = threading.Lock()  # one sentence at a time per worker

    @property
    def languages(self) -> list[str]:
        return self.inner.languages

    @property
    def sample_rate(self) -> int:
        return self._rate or 22050

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def list_voices(self) -> list[tuple[str, list[str]]]:
        return self.inner.list_voices() if isinstance(self.inner, TtsPack) else [(self.name, self.languages)]

    def load(self) -> None:
        """Start the worker and wait until it has loaded and warmed up its model (the caller applies a timeout)."""
        self.close()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "wyoming_vulkan.worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, cwd="/",  # the worker logs to the server's stderr (the container log)
            env={**os.environ, "WYOMING_WORKER_PARENT": str(os.getpid())},
        )
        self._send(_config_to_json(self.config, self.base_gpu))
        message = self._receive()
        if message.get("type") != "ready":
            self.close()
            text = message.get("message") or "worker stopped while loading"
            raise GpuUnavailable(text) if message.get("gpu") else RuntimeError(text)
        self._rate = int(message["sample_rate"])
        known = {f.name for f in fields(RuntimeReport)}
        runtime = RuntimeReport(**{k: v for k, v in message["runtime"].items() if k in known})
        runtime.detail = f"{runtime.detail}; worker pid {self.proc.pid}".lstrip("; ")
        runtime.memory_mb = self.runtime.memory_mb
        self.runtime = runtime

    def _load(self, device: str) -> None:  # load() does it all in the worker
        raise NotImplementedError

    def warm_up(self) -> None:
        pass  # the worker warms up before it reports ready

    def _send(self, message: dict) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message).encode() + b"\n")
        self.proc.stdin.flush()

    def _receive(self) -> dict:
        assert self.proc is not None and self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            return {"type": "error", "message": f"worker exited (code {self.proc.poll()})"}
        return json.loads(line)

    def _read(self, size: int) -> bytes:
        assert self.proc is not None and self.proc.stdout is not None
        data = self.proc.stdout.read(size)
        if len(data) != size:
            raise RuntimeError("worker exited mid-sentence")
        return data

    def _speak(self, text: str, options: SynthesisOptions, stream: bool) -> Iterator[bytes]:
        with self.lock:
            if not self.alive:
                raise RuntimeError(f"worker of {self.name} is not running")
            self._send({"text": text, "voice": options.voice, "speaker": options.speaker,
                        "settings": options.settings, "stream": stream})
            finished = False
            try:
                while True:
                    message = self._receive()
                    if message["type"] == "audio":
                        yield self._read(message["size"])
                    elif message["type"] == "done":
                        finished = True
                        return
                    else:
                        finished = True
                        raise RuntimeError(message.get("message") or "worker error")
            finally:
                if not finished:  # the caller stopped reading (client gone): drain, so the next request starts clean
                    while self.alive:
                        message = self._receive()
                        if message["type"] == "audio":
                            self._read(message["size"])
                        else:
                            break

    @property
    def stream_audio(self):
        """Frame streaming if the model streams (Pocket, CosyVoice3), else None (whole sentences)."""
        if getattr(self.inner, "stream_audio", None) is None:
            return None
        return lambda text, options: self._speak(text, options, stream=True)

    def synthesize(self, text: str, options: SynthesisOptions) -> bytes:
        return b"".join(self._speak(text, options, stream=False))

    def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()  # the worker closes its engine and exits
            proc.wait(STOP_TIMEOUT_S)
        except (subprocess.TimeoutExpired, OSError):
            proc.kill()
            proc.wait()


def make_worker(config: EngineConfig, gpu: GpuConfig) -> WorkerEngine:
    """Engine factory for the voice registry and the benchmark: every TTS model in its own worker."""
    return WorkerEngine(config, gpu)


# ---------------------------------------------------------------- the worker process


def _die_with_parent() -> None:
    """Exit when the server dies (Linux: PR_SET_PDEATHSIG), even if it could not close our stdin."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # 1 = PR_SET_PDEATHSIG
        # the server may itself be PID 1 (in a container), so compare with the pid it passed, not with 1
        parent = os.environ.get("WYOMING_WORKER_PARENT")
        if parent and os.getppid() != int(parent):
            sys.exit(0)  # the server died before prctl took effect
    except OSError:
        pass


def main() -> int:
    _die_with_parent()
    # The protocol owns the original stdout; anything else writing to fd 1 (a library's printf) goes to stderr.
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    stdin = sys.stdin.buffer

    def send(message: dict, payload: bytes = b"") -> None:
        out.write(json.dumps(message).encode() + b"\n" + payload)
        out.flush()

    first = stdin.readline()
    if not first:
        return 0
    config, gpu = _config_from_json(json.loads(first))
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format=f"%(asctime)s %(levelname)s worker[{config.name}] %(name)s: %(message)s")
    from .engines import create_engine

    engine = create_engine(config, gpu)
    try:
        engine.load()
        engine.warm_up()
    except GpuUnavailable as err:
        send({"type": "error", "gpu": True, "message": str(err)})
        return 3
    except Exception as err:  # bad model file, missing voices, ...
        _LOGGER.exception("loading failed")
        send({"type": "error", "gpu": False, "message": str(err) or type(err).__name__})
        return 1
    send({"type": "ready", "runtime": asdict(engine.runtime), "sample_rate": engine.sample_rate})
    try:
        while line := stdin.readline():
            request = json.loads(line)
            options = SynthesisOptions(speaker=request.get("speaker"), voice=request.get("voice"),
                                       settings=request.get("settings") or {})
            try:
                stream = getattr(engine, "stream_audio", None)
                chunks = stream(request["text"], options) if request.get("stream") and stream else \
                    [engine.synthesize(request["text"], options)]
                for chunk in chunks:
                    if chunk:
                        send({"type": "audio", "size": len(chunk)}, chunk)
                send({"type": "done"})
            except Exception as err:
                _LOGGER.exception("synthesis failed")
                send({"type": "error", "message": str(err) or type(err).__name__})
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
