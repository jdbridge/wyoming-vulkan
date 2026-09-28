"""Pocket TTS (Kyutai, 100M, CC-BY-4.0) through ONNX Runtime, as a voice pack with frame-by-frame streaming.

Runtime: the vendored single-file runtime of thewh1teagle/pocket-tts-onnx (third_party/pocket_tts_onnx): one .onnx
holds the graph, the SentencePiece tokenizer and the built-in voices (no espeak). The model is autoregressive and
decodes 80 ms frames one after another; `stream_audio()` hands each frame out as soon as it exists, so the first
audio of a sentence is ready after ~0.1 s instead of after the whole sentence.
Measured on an Intel N305: CPU RTF 0.42 (first frame 0.07 s); the WebGPU EP is slower (RTF 1.84, too many tiny
steps), so the stack runs it with device = "cpu".
Files: <dir>/pocket-tts-english.onnx (thewh1teagle/pocket-tts-onnx; scripts/fetch-models.sh pocket).
Options: threads (4), temperature, decode_steps (the model's defaults otherwise), gain (1.5: the voices peak around
0.5-0.6 and frames cannot be peak-normalised per sentence while streaming). English.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .base import SynthesisOptions, TtsPack
from .ort_session import make_session

_LOGGER = logging.getLogger(__name__)

_VOICE_PREFIX = "pocket_tts_voice/"


def onnx_metadata(path: Path) -> dict[str, str]:
    """The metadata_props of an ONNX file, read without loading the model: walk the top-level protobuf fields of
    ModelProto and skip everything else (the graph is most of the file)."""

    def varint(f) -> Optional[int]:
        shift = result = 0
        while True:
            b = f.read(1)
            if not b:
                return None
            result |= (b[0] & 0x7F) << shift
            if not b[0] & 0x80:
                return result
            shift += 7

    def entry(data: bytes) -> tuple[str, str]:
        import io

        f, key, value = io.BytesIO(data), "", ""
        while (tag := varint(f)) is not None:
            n = varint(f)
            chunk = f.read(n)
            if tag >> 3 == 1:
                key = chunk.decode()
            elif tag >> 3 == 2:
                value = chunk.decode()
        return key, value

    out: dict[str, str] = {}
    with open(path, "rb") as f:
        while (tag := varint(f)) is not None:
            field, wire = tag >> 3, tag & 7
            if wire == 0:
                varint(f)
            elif wire == 1:
                f.seek(8, 1)
            elif wire == 5:
                f.seek(4, 1)
            elif wire == 2:
                n = varint(f)
                if field == 14:  # ModelProto.metadata_props
                    key, value = entry(f.read(n))
                    out[key] = value
                else:
                    f.seek(n, 1)
            else:
                raise ValueError(f"{path}: not an ONNX model (wire type {wire})")
    return out


class PocketEngine(TtsPack):
    tts = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    _sample_rate = 24000

    def list_voices(self) -> list[tuple[str, list[str]]]:
        meta = onnx_metadata(self.config.model)
        if "pocket_tts_config" in meta:
            self._sample_rate = int(json.loads(meta["pocket_tts_config"]).get("sample_rate", 24000))
        return [(key[len(_VOICE_PREFIX):], ["en_US"]) for key in meta if key.startswith(_VOICE_PREFIX)]

    def _load(self, device: str) -> None:
        from pocket_tts_onnx import PocketTTS

        self.tts = PocketTTS(self.config.model, num_threads=int(self.config.options.get("threads", 4)))
        if device == "igpu":  # replace the runtime's CPU session by one on the WebGPU EP (with the GPU proof)
            self.tts.session = make_session(self.config.model, device, self.gpu, self.runtime)
        else:
            self.runtime.device_name, self.runtime.detail = "CPU", "providers ['CPUExecutionProvider']"
        self._sample_rate = self.tts.sample_rate
        self.gain = float(self.config.options.get("gain", 1.5))
        self.lock = threading.Lock()  # the runtime keeps decoding state per call; one sentence at a time
        self.runtime.actual = device
        _LOGGER.info("tts pack %s: loaded %s on %s", self.name, self.config.model.name, self.runtime.summary())

    def _kwargs(self) -> dict:
        o = self.config.options
        kw = {}
        if o.get("temperature") not in (None, ""):
            kw["temperature"] = float(o["temperature"])
        if o.get("decode_steps") not in (None, ""):
            kw["decode_steps"] = int(o["decode_steps"])
        return kw

    def stream_audio(self, text: str, options: SynthesisOptions) -> Iterator[bytes]:
        """int16 PCM per 80 ms frame, as the model produces them (holds the engine lock while iterating)."""
        assert self.tts is not None, "engine not loaded"
        voice = options.voice or self.list_voices()[0][0]
        with self.lock:
            for frame in self.tts.stream(text, voice=voice, **self._kwargs()):
                audio = np.clip(np.asarray(frame, dtype=np.float32).reshape(-1) * self.gain, -1.0, 1.0)
                yield (audio * 32767.0).astype(np.int16).tobytes()

    def synthesize(self, text: str, options: SynthesisOptions) -> bytes:
        return b"".join(self.stream_audio(text, options))

    def warm_up(self) -> None:
        voices = self.list_voices()
        if voices:
            self.synthesize("Warming up.", SynthesisOptions(voice=voices[0][0]))

    def close(self) -> None:
        lock = getattr(self, "lock", None)
        if lock is None:
            self.tts = None
            return
        with lock:
            self.tts = None
