"""CosyVoice3 (FunAudioLLM Fun-CosyVoice3-0.5B-2512, Apache-2.0) through cosyvoice.cpp (MIT, ggml), as a voice pack.

cosyvoice.cpp brings its own ggml build, and two ggml copies in one process crash (same library names, different
versions). So it runs as a child process, `cosyvoice-server --api` on 127.0.0.1, and this engine asks it for raw PCM.
Voices are voice prompts: a reference clip plus its transcript, encoded once by the frontend (two ONNX models, CPU)
into a prompt file. CosyVoice3 has no built-in voices; every voice speaks every supported language.

GPU proof: the server gets an explicit ggml device name (`--backend Vulkan0`) and exits if that device does not exist,
so it cannot fall back to the CPU silently. The image's VK_DRIVER_FILES exposes only the Intel driver.

Measured on an Intel N305 (ADL-N iGPU): RTF ~12 (the flow model runs 10 diffusion steps of ~2.2 s each per short
sentence): too slow for a voice assistant here, meant for stronger GPUs. On the CPU it is slower still (RTF ~70).
Files (Lourdle/Fun-CosyVoice3-0.5B-2512-GGUF; scripts/fetch-models.sh cosyvoice):
  <dir>/CosyVoice3-2512_Q8_0.gguf                                          the model
  <dir>/frontend/speech_tokenizer_v3.int8.onnx, <dir>/frontend/campplus.onnx  frontend (only for .wav voices)
  <dir>/voices/<name>.gguf                                                 an encoded voice prompt, or
  <dir>/voices/<name>.wav + <name>.txt                                     a clip (5-15 s) and its exact transcript
Options: voices (folder, default <model dir>/voices), frontend (folder, default <model dir>/frontend), cache (folder
for encoded .wav voices, default /tmp/cosyvoice-voices), gpu_device (ggml device, default Vulkan0), threads,
stream (false: whole sentences; the server's chunked streaming is slower in total), seed, start_timeout (s),
length_scale.
"""

import hashlib
import http.client
import json
import logging
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

from .base import GpuUnavailable, SynthesisOptions, TtsPack

_LOGGER = logging.getLogger(__name__)

HOME = Path(os.environ.get("COSYVOICE_HOME", "/opt/cosyvoice"))
SAMPLE_RATE = 24000
# the languages of CosyVoice3-0.5B-2512 (plus Chinese dialects), as HA language codes
LANGUAGES = ["en_US", "de_DE", "fr_FR", "es_ES", "it_IT", "ru_RU", "ja_JP", "ko_KR", "zh_CN"]
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class CosyVoiceEngine(TtsPack):
    auto_languages = LANGUAGES
    proc: Optional[subprocess.Popen] = None

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def _dir(self, key: str, default: Path) -> Path:
        value = self.config.options.get(key)
        return Path(value) if value else default

    def _voice_files(self) -> dict[str, Path]:
        """name -> .gguf prompt or .wav clip (a .wav needs its .txt transcript); a .gguf wins over a .wav."""
        folder = self._dir("voices", self.config.model.parent / "voices")
        found: dict[str, Path] = {}
        for path in sorted(folder.iterdir()):
            if path.suffix == ".wav" and path.with_suffix(".txt").is_file():
                found.setdefault(path.stem, path)
            elif path.suffix == ".gguf":
                found[path.stem] = path
        return found

    def list_voices(self) -> list[tuple[str, list[str]]]:
        if not self.config.model.is_file():
            raise FileNotFoundError(self.config.model)
        return [(name, self.languages) for name in self._voice_files()]

    def _encode(self, name: str, wav: Path) -> Path:
        """Encode a clip + transcript into a prompt file once (cached by the files' contents)."""
        text = wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
        digest = hashlib.sha256(wav.read_bytes() + text.encode()).hexdigest()[:16]
        cache = self._dir("cache", Path("/tmp/cosyvoice-voices"))
        out = cache / f"{name}-{digest}.gguf"
        if out.is_file():
            return out
        cache.mkdir(parents=True, exist_ok=True)
        frontend = self._dir("frontend", self.config.model.parent / "frontend")
        cmd = [str(HOME / "bin/cosyvoice-cli"), "-q", "--frontend-only",
               "--speech-tokenizer", str(frontend / "speech_tokenizer_v3.int8.onnx"),
               "--campplus", str(frontend / "campplus.onnx"),
               "--prompt-audio", str(wav), "--prompt-text", text, "--prompt-speech-output", str(out)]
        t = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=HOME / "bin")
        if result.returncode != 0 or not out.is_file():
            raise RuntimeError(f"voice {name}: encoding {wav.name} failed: {_ANSI.sub('', result.stderr or result.stdout).strip()[-300:]}")
        _LOGGER.info("tts pack %s: encoded voice %s from %s in %.1f s", self.name, name, wav.name, time.perf_counter() - t)
        return out

    def _load(self, device: str) -> None:
        prompts = {name: (self._encode(name, p) if p.suffix == ".wav" else p) for name, p in self._voice_files().items()}
        if not prompts:
            raise RuntimeError("no voices")
        gpu_device = str(self.config.options.get("gpu_device") or "Vulkan0")
        backend = "CPU" if device == "cpu" else gpu_device
        with socket.socket() as s:  # a free port on the loopback interface
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        cmd = [str(HOME / "bin/cosyvoice-server"), "--api", "--model", str(self.config.model), "--backend", backend,
               "--host", "127.0.0.1", "--port", str(self.port), "--served-model-name", "cosyvoice",
               "--inference-buffer-policy", "dedicated", "--concurrency", "1"]
        if self.config.options.get("threads"):
            cmd += ["--threads", str(self.config.options["threads"])]
        for name, path in prompts.items():
            cmd += ["--voice-prompt", f"{name}={path}"]
        self.output: list[str] = []
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=HOME / "bin")
        threading.Thread(target=self._log_output, args=(self.proc,), daemon=True, name=f"{self.name}-log").start()
        deadline = time.monotonic() + float(self.config.options.get("start_timeout", 60))
        while True:
            if self.proc.poll() is not None:
                time.sleep(0.2)  # let the log thread catch the last lines
                message = " / ".join(self.output[-3:]) or f"exit code {self.proc.returncode}"
                self.proc = None
                if device == "igpu" and "failed to initialize backend" in message:
                    raise GpuUnavailable(f"cosyvoice-server has no ggml device {gpu_device!r}: {message}")
                raise RuntimeError(f"cosyvoice-server stopped: {message}")
            try:
                status = json.loads(self._request("GET", "/healthz").read())
                if status.get("status") == "ok":
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                self.close()
                raise RuntimeError("cosyvoice-server did not start in time")
            time.sleep(0.5)
        self.lock = threading.Lock()  # one sentence at a time (the server has one slot)
        self.runtime.actual = device
        if device == "cpu":
            self.runtime.device_name, self.runtime.detail = "CPU", "cosyvoice-server --backend CPU"
        else:
            self.runtime.device_name = _ggml_description(gpu_device) or gpu_device
            self.runtime.detail = f"cosyvoice-server --backend {gpu_device} (pid {self.proc.pid})"
        _LOGGER.info("tts pack %s: cosyvoice-server on port %d with %d voices on %s", self.name, self.port, len(prompts),
                     self.runtime.summary())

    def _log_output(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            line = _ANSI.sub("", line).rstrip()
            if line:
                self.output = (self.output + [line])[-20:]
                _LOGGER.debug("cosyvoice-server: %s", line)

    def _request(self, method: str, path: str, body: Optional[dict] = None, timeout: float = 5.0):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        conn.request(method, path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"cosyvoice-server {path}: HTTP {response.status}: {response.read()[:300]!r}")
        return response

    def stream_audio(self, text: str, options: SynthesisOptions) -> Iterator[bytes]:
        """int16 PCM as the server sends it (whole sentences unless the stream option is set)."""
        assert self.proc is not None, "engine not loaded"
        body = {"model": "cosyvoice", "input": text, "voice": options.voice or self.list_voices()[0][0],
                "response_format": "pcm", "stream": bool(self.config.options.get("stream", False)),
                "speed": self.speed(options)}
        if self.config.options.get("seed") not in (None, ""):
            body["seed"] = int(self.config.options["seed"])
        with self.lock:
            response = self._request("POST", "/v1/audio/speech", body, timeout=600)
            rest = b""
            while chunk := response.read1(65536):
                chunk, rest = rest + chunk, b""
                if len(chunk) % 2:
                    chunk, rest = chunk[:-1], chunk[-1:]
                if chunk:
                    yield chunk

    def synthesize(self, text: str, options: SynthesisOptions) -> bytes:
        return b"".join(self.stream_audio(text, options))

    def warm_up(self) -> None:
        self.synthesize("Hi.", SynthesisOptions())

    def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()


def _ggml_description(name: str) -> str:
    """The description of a ggml device of this process (the server's ggml sees the same Vulkan devices)."""
    try:
        from .ggml import list_devices

        return next((d.description for d in list_devices() if d.name == name), "")
    except OSError:
        return ""
