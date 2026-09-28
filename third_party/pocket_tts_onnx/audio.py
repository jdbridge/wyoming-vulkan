"""Reading and resampling a voice prompt, with numpy as the only requirement.

`scipy.signal.resample_poly` is what upstream pocket-tts uses; when scipy is
installed we use it too, so a cloned voice matches upstream exactly. Otherwise
we fall back to bandlimited interpolation with a Kaiser-windowed sinc, which is
perceptually equivalent for a voice prompt.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

_TAPS = 32
_KAISER_BETA = 8.6


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Return `(mono float32 samples, sample_rate)`."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                if handle.getsampwidth() == 2:
                    raw = handle.readframes(-1)
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    return to_mono(samples, handle.getnchannels()), handle.getframerate()
        except wave.Error:
            pass
    try:
        import soundfile as sf
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            f"reading {path.suffix} needs soundfile: pip install soundfile"
        ) from error
    data, sample_rate = sf.read(str(path), dtype="float32")
    return (data if data.ndim == 1 else data.mean(axis=1)).astype(np.float32), sample_rate


def to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples.astype(np.float32)


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate:
        return samples.astype(np.float32)
    try:
        from scipy.signal import resample_poly
    except ImportError:
        return _resample_sinc(samples, from_rate, to_rate)
    gcd = math.gcd(int(from_rate), int(to_rate))
    return resample_poly(samples, to_rate // gcd, from_rate // gcd).astype(np.float32)


def _resample_sinc(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Bandlimited interpolation: one windowed sinc per output sample."""
    ratio = to_rate / from_rate
    cutoff = min(1.0, ratio)
    count = int(math.floor(len(samples) * ratio))
    centres = np.arange(count, dtype=np.float64) / ratio
    base = np.floor(centres).astype(np.int64)

    taps = np.arange(-_TAPS // 2 + 1, _TAPS // 2 + 1, dtype=np.int64)
    indices = base[:, None] + taps[None, :]
    offsets = (centres[:, None] - indices) * cutoff

    window = np.kaiser(_TAPS, _KAISER_BETA)[None, :]
    weights = np.sinc(offsets) * window
    weights /= weights.sum(axis=1, keepdims=True)

    padded = np.pad(samples.astype(np.float64), (_TAPS, _TAPS))
    return (padded[indices + _TAPS] * weights).sum(axis=1).astype(np.float32)


def pad_to_frames(samples: np.ndarray, frame_size: int) -> np.ndarray:
    """The encoder consumes whole frames, so round the prompt up to one."""
    remainder = len(samples) % frame_size
    if remainder:
        samples = np.pad(samples, (0, frame_size - remainder))
    return samples.astype(np.float32)
