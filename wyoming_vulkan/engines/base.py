"""Engine interfaces. The handler only talks to these; backends live in their own modules."""

import logging
from abc import ABC, abstractmethod
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..config import EngineConfig, GpuConfig

_LOGGER = logging.getLogger(__name__)


class GpuUnavailable(RuntimeError):
    """The GPU path was requested but is not what the engine really got."""


@dataclass
class RuntimeReport:
    """What an engine actually runs on, as opposed to what was asked for."""

    backend: str
    requested: str  # igpu | cpu | auto
    actual: str = "unloaded"  # igpu | cpu
    device_name: str = ""
    detail: str = ""
    fell_back: bool = False

    def label(self) -> str:
        """Short execution label for `info`: the GPU's name, "CPU", or "CPU fallback"."""
        if self.fell_back:
            return "CPU fallback"
        if self.actual == "cpu":
            return "CPU"
        return self.device_name or self.actual

    def summary(self) -> str:
        """Longer form for the logs."""
        text = f"{self.actual} ({self.device_name})" if self.device_name else self.actual
        if self.fell_back:
            text += " [FALLBACK: GPU unavailable]"
        return text


class Engine(ABC):
    def __init__(self, config: EngineConfig, gpu: GpuConfig) -> None:
        self.config = config
        self.gpu = gpu
        self.runtime = RuntimeReport(backend=config.backend, requested=config.device)

    @property
    def name(self) -> str:
        return self.config.name

    auto_languages: list[str] = []  # what the engine supports when the config says languages = "auto"

    @property
    def languages(self) -> list[str]:
        return self.config.languages if self.config.languages is not None else (self.auto_languages or ["en"])

    def load(self) -> None:
        """Load on the configured device; 'auto' falls back to the CPU loudly."""
        device = self.config.device
        if device == "cpu":
            self._load("cpu")
            return
        try:
            self._load("igpu")
        except GpuUnavailable as err:
            if device == "igpu":
                raise
            _LOGGER.warning("!!! %s %s: GPU unavailable (%s); FALLING BACK TO CPU", self.config.kind, self.name, err)
            self.close()
            self._load("cpu")
            self.runtime.fell_back = True
            self.runtime.detail = f"GPU unavailable: {err}"

    @abstractmethod
    def _load(self, device: str) -> None:
        """Load on exactly this device ('igpu' or 'cpu'); raise GpuUnavailable if the GPU path is not real."""

    @abstractmethod
    def warm_up(self) -> None:
        """Run once so shader compilation etc. happens before the first real request."""

    def close(self) -> None:
        pass


class SttEngine(Engine):
    sample_rate = 16000

    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: Optional[str]) -> str:
        """audio: float32 mono in [-1, 1] at self.sample_rate. Thread-safe (engines lock internally)."""


@dataclass
class SynthesisOptions:
    speaker: Optional[str] = None
    voice: Optional[str] = None  # a voice pack's voice id (TtsPack)
    settings: dict[str, Any] = field(default_factory=dict)  # a pack voice's speech settings ([voice_settings])


def to_pcm16(audio, normalize: bool = True) -> bytes:
    """Float audio -> int16 PCM bytes, peak-normalised per sentence like piper-tts (so every engine is as loud)."""
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if normalize:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        audio = np.zeros_like(audio) if peak < 1e-8 or not np.isfinite(peak) else audio / peak
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


# espeak-ng keeps process-global state: phonemising from two threads at once corrupts the phonemes. Every engine
# that uses espeak (Piper, Kokoro, KittenTTS) holds this lock while phonemising (only then, not during inference).
ESPEAK_LOCK = threading.Lock()


class TtsEngine(Engine):
    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @abstractmethod
    def synthesize(self, text: str, options: SynthesisOptions) -> bytes:
        """int16 mono PCM at self.sample_rate for one sentence (the handler splits sentences). Thread-safe."""


class TtsPack(TtsEngine):
    """One model with many voices (Kokoro, KittenTTS). Offered to HA as "<pack name>_<voice id>"."""

    @abstractmethod
    def list_voices(self) -> list[tuple[str, list[str]]]:
        """(voice id, languages) for every voice, without loading the model (reads only small files)."""

    def speed(self, options: SynthesisOptions) -> float:
        """The model's speed input from the speech settings: speed = 1 / length_scale (Piper's convention)."""
        value = options.settings.get("length_scale", self.config.options.get("length_scale"))
        return 1.0 / float(value) if value not in (None, "") else 1.0
