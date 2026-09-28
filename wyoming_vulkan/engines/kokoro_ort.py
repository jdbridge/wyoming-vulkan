"""Kokoro-82M (hexgrad, Apache-2.0) through ONNX Runtime: WebGPU EP (Dawn -> Vulkan) or the CPU EP.

One model, many voices (a TtsPack). Text front end: kokoro-onnx's tokenizer (espeak-ng via phonemizer, the standard
front end of Kokoro's ONNX builds); the ONNX session is ours, so it runs on the GPU and is proven like the others.
Files (onnx-community/Kokoro-82M-v1.0-ONNX; scripts/fetch-models.sh kokoro):
  <dir>/model.onnx          fp32 (the fp16 builds give NaN on the WebGPU EP)
  <dir>/tokenizer.json      phoneme vocabulary
  <dir>/voices/<id>.bin     one style table per voice (510 x 256 float32)
Options: voices (folder, default <model dir>/voices), include (list of voice-id patterns, default: every voice whose
language espeak can phonemise; Japanese and Chinese voices need other front ends and are left out), length_scale.
"""

import fnmatch
import json
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .base import ESPEAK_LOCK, SynthesisOptions, TtsPack, to_pcm16
from .ort_session import make_session

_LOGGER = logging.getLogger(__name__)
# phonemizer warns "words count mismatch" for nearly every sentence with punctuation; harmless for synthesis
logging.getLogger("phonemizer").setLevel(logging.ERROR)

SAMPLE_RATE = 24000
MAX_TOKENS = 510
# first letter of a voice id -> (HA language, espeak language); j (Japanese) and z (Chinese) are not supported
LANGUAGES = {
    "a": ("en_US", "en-us"),
    "b": ("en_GB", "en-gb"),
    "e": ("es_ES", "es"),
    "f": ("fr_FR", "fr-fr"),
    "h": ("hi_IN", "hi"),
    "i": ("it_IT", "it"),
    "p": ("pt_BR", "pt-br"),
}


class KokoroEngine(TtsPack):
    session = None

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def _voices_dir(self) -> Path:
        return Path(self.config.options.get("voices", self.config.model.parent / "voices"))

    def list_voices(self) -> list[tuple[str, list[str]]]:
        include = self.config.options.get("include") or ["*"]
        out = []
        for path in sorted(self._voices_dir().glob("*.bin")):
            voice = path.stem
            lang = LANGUAGES.get(voice[:1])
            if lang and any(fnmatch.fnmatch(voice, p) for p in include):
                out.append((voice, [lang[0]]))
        return out

    def _load(self, device: str) -> None:
        from kokoro_onnx.tokenizer import Tokenizer

        vocab_file = self.config.model.parent / "tokenizer.json"
        vocab = json.loads(vocab_file.read_text(encoding="utf-8"))["model"]["vocab"] if vocab_file.exists() else None
        with ESPEAK_LOCK:
            self.tokenizer = Tokenizer(vocab=vocab)
        self.session = make_session(self.config.model, device, self.gpu, self.runtime)
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.styles: dict[str, np.ndarray] = {}
        self.lock = threading.Lock()
        self.runtime.actual = device
        _LOGGER.info("tts pack %s: loaded %s on %s, %s", self.name, self.config.model.name, self.runtime.summary(), self.runtime.detail)

    def _style(self, voice: str) -> np.ndarray:
        if voice not in self.styles:
            self.styles[voice] = np.fromfile(self._voices_dir() / f"{voice}.bin", dtype=np.float32).reshape(-1, 1, 256)
        return self.styles[voice]

    def synthesize(self, text: str, options: SynthesisOptions) -> bytes:
        assert self.session is not None, "engine not loaded"
        voice = options.voice or self.list_voices()[0][0]
        espeak_lang = LANGUAGES[voice[:1]][1]
        with ESPEAK_LOCK:
            phonemes = self.tokenizer.phonemize(text, espeak_lang)
        ids = self.tokenizer.tokenize(phonemes, limit=None)
        table = self._style(voice)
        speed = np.array([self.speed(options)], dtype=np.float32)
        parts = []
        with self.lock:
            for start in range(0, max(len(ids), 1), MAX_TOKENS - 2):  # the model takes at most 510 tokens
                chunk = ids[start : start + MAX_TOKENS - 2]
                if not chunk:
                    break
                feed = {"input_ids": np.array([[0, *chunk, 0]], dtype=np.int64),
                        "style": table[min(len(chunk), len(table) - 1)], "speed": speed}
                if "tokens" in self.inputs:  # older exports name the first input "tokens"
                    feed["tokens"] = feed.pop("input_ids")
                audio = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
                parts.append(to_pcm16(audio))
        return b"".join(parts)

    def warm_up(self) -> None:
        voices = self.list_voices()
        if voices:
            self.synthesize("Warming up.", SynthesisOptions(voice=voices[0][0]))

    def close(self) -> None:
        lock: Optional[threading.Lock] = getattr(self, "lock", None)
        if lock is None:
            self.session = None
            return
        with lock:
            self.session = None
