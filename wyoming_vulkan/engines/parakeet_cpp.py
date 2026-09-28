"""Parakeet speech-to-text through whisper.cpp's libparakeet (ggml; Vulkan on the iGPU, or the CPU).

Struct layouts follow third_party/parakeet.h of the pinned whisper.cpp v1.9.4 and must match the libparakeet in
the image. Backend options in the config: threads (default 4).
"""

import ctypes as C
import logging
import threading
from typing import Optional

import numpy as np

from .. import devices
from . import ggml
from .base import GpuUnavailable, SttEngine

_LOGGER = logging.getLogger(__name__)

PARAKEET_SAMPLING_GREEDY = 0
# Parakeet TDT 0.6b v3 per its model card (detects the language itself; the requested language is not passed on)
V3_LANGUAGES = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt", "nl",
                "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk"]


class ContextParams(C.Structure):
    _fields_ = [("use_gpu", C.c_bool), ("gpu_device", C.c_int)]


class FullParams(C.Structure):
    _fields_ = [
        ("strategy", C.c_int),
        ("n_threads", C.c_int),
        ("offset_ms", C.c_int),
        ("duration_ms", C.c_int),
        ("no_context", C.c_bool),
        ("audio_ctx", C.c_int),
        ("new_segment_callback", C.c_void_p),
        ("new_segment_callback_user_data", C.c_void_p),
        ("new_token_callback", C.c_void_p),
        ("new_token_callback_user_data", C.c_void_p),
        ("progress_callback", C.c_void_p),
        ("progress_callback_user_data", C.c_void_p),
        ("encoder_begin_callback", C.c_void_p),
        ("encoder_begin_callback_user_data", C.c_void_p),
        ("abort_callback", C.c_void_p),
        ("abort_callback_user_data", C.c_void_p),
    ]


_LIB: Optional[C.CDLL] = None


def _lib() -> C.CDLL:
    global _LIB
    if _LIB is None:
        L = C.CDLL("libparakeet.so")
        L.parakeet_version.restype = C.c_char_p
        L.parakeet_context_default_params.restype = ContextParams
        L.parakeet_init_from_file_with_params.argtypes = [C.c_char_p, ContextParams]
        L.parakeet_init_from_file_with_params.restype = C.c_void_p
        L.parakeet_full_default_params.argtypes = [C.c_int]
        L.parakeet_full_default_params.restype = FullParams
        L.parakeet_full.argtypes = [C.c_void_p, FullParams, C.POINTER(C.c_float), C.c_int]
        L.parakeet_full.restype = C.c_int
        L.parakeet_full_n_segments.argtypes = [C.c_void_p]
        L.parakeet_full_n_segments.restype = C.c_int
        L.parakeet_full_get_segment_text.argtypes = [C.c_void_p, C.c_int]
        L.parakeet_full_get_segment_text.restype = C.c_char_p
        L.parakeet_free.argtypes = [C.c_void_p]
        ggml.install_logging(L.parakeet_log_set)
        _LOGGER.info("libparakeet %s", L.parakeet_version().decode())
        _LIB = L
    return _LIB


class ParakeetEngine(SttEngine):
    ctx: Optional[int] = None
    auto_languages = V3_LANGUAGES  # for languages = "auto" (valid for the v3 models)

    def _load(self, device: str) -> None:
        L = _lib()
        cp = L.parakeet_context_default_params()
        cp.use_gpu = device == "igpu"
        cp.gpu_device = 0
        expected = ""
        if device == "igpu":
            problem = devices.vulkan_icd_problem()
            if problem:
                raise GpuUnavailable(problem)
            all_devices = ggml.list_devices()
            gpus = [d for d in all_devices if d.type in ("gpu", "igpu")]
            matching = [i for i, d in enumerate(gpus) if self.gpu.name_contains in d.description]
            if not matching:
                raise GpuUnavailable(
                    f"no ggml GPU device contains {self.gpu.name_contains!r}; devices: "
                    + ", ".join(f"{d.name}={d.description} ({d.type})" for d in all_devices)
                )
            cp.gpu_device = matching[0]
            expected = gpus[matching[0]].name
            self.runtime.device_name = gpus[matching[0]].description

        mark = len(ggml.LOG.lines)
        ctx = L.parakeet_init_from_file_with_params(str(self.config.model).encode(), cp)
        if not ctx:
            raise RuntimeError(f"libparakeet could not load {self.config.model}")
        self.ctx = ctx

        if device == "igpu":
            # libparakeet logs "parakeet_backend_init_gpu: using Vulkan0 backend" when the GPU backend is really used
            used = [l for l in ggml.LOG.lines[mark:] if "backend_init_gpu: using" in l]
            if not any(f"using {expected} backend" in l for l in used):
                raise GpuUnavailable(f"libparakeet did not report using {expected}; saw {used or 'no GPU backend line'}")
        else:
            self.runtime.device_name = f"CPU, {self._threads()} threads"

        self.params = L.parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY)
        self.params.n_threads = self._threads()
        self.lock = threading.Lock()  # parakeet_full is not thread-safe for one context
        self.runtime.actual = device
        _LOGGER.info("stt %s: loaded %s on %s", self.name, self.config.model.name, self.runtime.summary())

    def _threads(self) -> int:
        return int(self.config.options.get("threads", 4))

    def transcribe(self, audio: np.ndarray, language: Optional[str]) -> str:
        assert self.ctx, "engine not loaded"
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        L = _lib()
        with self.lock:
            rc = L.parakeet_full(self.ctx, self.params, audio.ctypes.data_as(C.POINTER(C.c_float)), len(audio))
            if rc != 0:
                raise RuntimeError(f"parakeet_full returned {rc}")
            n = L.parakeet_full_n_segments(self.ctx)
            parts = [L.parakeet_full_get_segment_text(self.ctx, i).decode("utf-8", "replace").strip() for i in range(n)]
        return " ".join(p for p in parts if p)

    def warm_up(self) -> None:
        # ggml-vulkan compiles some pipelines only when an input length first needs them: measured 2026-09-28, the
        # first 1, 2 and 3 s clips each cost 0.4-3.6 s extra, from 4 s on nothing new. Cover those lengths up front.
        for seconds in (1, 2, 3, 5):
            self.transcribe(np.zeros(seconds * self.sample_rate, dtype=np.float32), None)

    def close(self) -> None:
        if self.ctx:
            _lib().parakeet_free(self.ctx)
            self.ctx = None
