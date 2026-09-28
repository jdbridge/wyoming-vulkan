"""Whisper speech-to-text through whisper.cpp's libwhisper (ggml; Vulkan on the iGPU, or the CPU).

Struct layouts follow third_party/whisper.h of the pinned whisper.cpp v1.9.4 and must match the libwhisper in the
image; tests/check_abi.sh compares them with what a C compiler makes of the header.
Backend options in the config:
  threads (4), audio_ctx (0 = full 30 s window; e.g. 512 = ~10 s, much faster for short commands; longer audio falls
  back to the full window automatically), beam_size (0 = greedy), language (fallback when the request's language is
  not one the model knows; default: auto-detect for multilingual models, "en" for .en models),
  max_seconds (time budget per request, default max(15, 3 x audio length): Whisper can fall into a repetition loop;
  the decode is then aborted through whisper.cpp's abort_callback and the text so far is returned).
"""

import ctypes as C
import logging
import re
import threading
import time
from typing import Optional

import numpy as np

from .. import devices
from . import ggml
from .base import GpuUnavailable, SttEngine

_LOGGER = logging.getLogger(__name__)

WHISPER_SAMPLING_GREEDY, WHISPER_SAMPLING_BEAM_SEARCH = 0, 1
FRAMES_PER_SECOND = 50  # encoder frames per second of audio (1500 per 30 s window)
ABORT_CALLBACK = C.CFUNCTYPE(C.c_bool, C.c_void_p)  # ggml_abort_callback: return true to abort


class Aheads(C.Structure):
    _fields_ = [("n_heads", C.c_size_t), ("heads", C.c_void_p)]


class ContextParams(C.Structure):
    _fields_ = [
        ("use_gpu", C.c_bool),
        ("flash_attn", C.c_bool),
        ("gpu_device", C.c_int),
        ("dtw_token_timestamps", C.c_bool),
        ("dtw_aheads_preset", C.c_int),
        ("dtw_n_top", C.c_int),
        ("dtw_aheads", Aheads),
        ("dtw_mem_size", C.c_size_t),
    ]


class Greedy(C.Structure):
    _fields_ = [("best_of", C.c_int)]


class BeamSearch(C.Structure):
    _fields_ = [("beam_size", C.c_int), ("patience", C.c_float)]


class VadParams(C.Structure):
    _fields_ = [
        ("threshold", C.c_float),
        ("min_speech_duration_ms", C.c_int),
        ("min_silence_duration_ms", C.c_int),
        ("max_speech_duration_s", C.c_float),
        ("speech_pad_ms", C.c_int),
        ("samples_overlap", C.c_float),
    ]


class FullParams(C.Structure):
    _fields_ = [
        ("strategy", C.c_int),
        ("n_threads", C.c_int),
        ("n_max_text_ctx", C.c_int),
        ("offset_ms", C.c_int),
        ("duration_ms", C.c_int),
        ("translate", C.c_bool),
        ("no_context", C.c_bool),
        ("no_timestamps", C.c_bool),
        ("single_segment", C.c_bool),
        ("print_special", C.c_bool),
        ("print_progress", C.c_bool),
        ("print_realtime", C.c_bool),
        ("print_timestamps", C.c_bool),
        ("token_timestamps", C.c_bool),
        ("thold_pt", C.c_float),
        ("thold_ptsum", C.c_float),
        ("max_len", C.c_int),
        ("split_on_word", C.c_bool),
        ("max_tokens", C.c_int),
        ("debug_mode", C.c_bool),
        ("audio_ctx", C.c_int),
        ("tdrz_enable", C.c_bool),
        ("suppress_regex", C.c_char_p),
        ("initial_prompt", C.c_char_p),
        ("carry_initial_prompt", C.c_bool),
        ("prompt_tokens", C.c_void_p),
        ("prompt_n_tokens", C.c_int),
        ("language", C.c_char_p),
        ("detect_language", C.c_bool),
        ("suppress_blank", C.c_bool),
        ("suppress_nst", C.c_bool),
        ("temperature", C.c_float),
        ("max_initial_ts", C.c_float),
        ("length_penalty", C.c_float),
        ("temperature_inc", C.c_float),
        ("entropy_thold", C.c_float),
        ("logprob_thold", C.c_float),
        ("no_speech_thold", C.c_float),
        ("greedy", Greedy),
        ("beam_search", BeamSearch),
        ("new_segment_callback", C.c_void_p),
        ("new_segment_callback_user_data", C.c_void_p),
        ("progress_callback", C.c_void_p),
        ("progress_callback_user_data", C.c_void_p),
        ("encoder_begin_callback", C.c_void_p),
        ("encoder_begin_callback_user_data", C.c_void_p),
        ("abort_callback", C.c_void_p),
        ("abort_callback_user_data", C.c_void_p),
        ("logits_filter_callback", C.c_void_p),
        ("logits_filter_callback_user_data", C.c_void_p),
        ("grammar_rules", C.c_void_p),
        ("n_grammar_rules", C.c_size_t),
        ("i_start_rule", C.c_size_t),
        ("grammar_penalty", C.c_float),
        ("vad", C.c_bool),
        ("vad_model_path", C.c_char_p),
        ("vad_params", VadParams),
    ]


_LIB: Optional[C.CDLL] = None


def _lib() -> C.CDLL:
    global _LIB
    if _LIB is None:
        L = C.CDLL("libwhisper.so")
        L.whisper_version.restype = C.c_char_p
        L.whisper_context_default_params.restype = ContextParams
        L.whisper_init_from_file_with_params.argtypes = [C.c_char_p, ContextParams]
        L.whisper_init_from_file_with_params.restype = C.c_void_p
        L.whisper_full_default_params.argtypes = [C.c_int]
        L.whisper_full_default_params.restype = FullParams
        L.whisper_full.argtypes = [C.c_void_p, FullParams, C.POINTER(C.c_float), C.c_int]
        L.whisper_full.restype = C.c_int
        L.whisper_full_n_segments.argtypes = [C.c_void_p]
        L.whisper_full_n_segments.restype = C.c_int
        L.whisper_full_get_segment_text.argtypes = [C.c_void_p, C.c_int]
        L.whisper_full_get_segment_text.restype = C.c_char_p
        L.whisper_is_multilingual.argtypes = [C.c_void_p]
        L.whisper_is_multilingual.restype = C.c_int
        L.whisper_lang_id.argtypes = [C.c_char_p]
        L.whisper_lang_id.restype = C.c_int
        L.whisper_model_type_readable.argtypes = [C.c_void_p]
        L.whisper_model_type_readable.restype = C.c_char_p
        L.whisper_free.argtypes = [C.c_void_p]
        L.whisper_lang_max_id.restype = C.c_int
        L.whisper_lang_str.argtypes = [C.c_int]
        L.whisper_lang_str.restype = C.c_char_p
        ggml.install_logging(L.whisper_log_set)
        _LOGGER.info("libwhisper %s", L.whisper_version().decode())
        _LIB = L
    return _LIB


# Whole segments that are only a sound annotation, e.g. "[BLANK_AUDIO]", "(silence)", "[Music]"
_ANNOTATION = re.compile(r"^\s*[\[(][^\])]*[\])]\s*$")


def _collapse_repeats(text: str) -> str:
    """"A. A. A." -> "A.": drop sentences that repeat the one before (Whisper's repetition loop)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out: list[str] = []
    for s in sentences:
        if not out or s.strip().lower() != out[-1].strip().lower():
            out.append(s)
    # an aborted loop usually ends in a fragment of a sentence it is repeating ("... ab. Schalte a")
    if len(out) > 1 and out[-1][-1:] not in ".!?" and any(x.lower().startswith(out[-1].lower()) for x in out[:-1]):
        out.pop()
    return " ".join(out)


class WhisperEngine(SttEngine):
    ctx: Optional[int] = None

    def _load(self, device: str) -> None:
        L = _lib()
        cp = L.whisper_context_default_params()
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
        ctx = L.whisper_init_from_file_with_params(str(self.config.model).encode(), cp)
        if not ctx:
            raise RuntimeError(f"libwhisper could not load {self.config.model}")
        self.ctx = ctx
        if device == "igpu":
            # libwhisper logs "whisper_backend_init_gpu: using Vulkan0 backend" when the GPU backend is really used
            used = [l for l in ggml.LOG.lines[mark:] if "backend_init_gpu: using" in l]
            if not any(f"using {expected} backend" in l for l in used):
                raise GpuUnavailable(f"libwhisper did not report using {expected}; saw {used or 'no GPU backend line'}")
        else:
            self.runtime.device_name = f"CPU, {self._threads()} threads"

        self.multilingual = bool(L.whisper_is_multilingual(ctx))
        # for languages = "auto": every language Whisper knows, or English for the .en models
        self.auto_languages = (
            [L.whisper_lang_str(i).decode() for i in range(L.whisper_lang_max_id() + 1)] if self.multilingual else ["en"]
        )
        o = self.config.options
        beam = int(o.get("beam_size", 0))
        self.params = L.whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH if beam > 1 else WHISPER_SAMPLING_GREEDY)
        p = self.params
        p.n_threads = self._threads()
        p.no_context = True
        p.no_timestamps = False  # without timestamps whisper.cpp falls into repetition loops more easily
        p.single_segment = False
        # time budget: whisper.cpp calls this before each graph computation; True aborts the decode
        self._deadline = float("inf")
        self._abort_cb = ABORT_CALLBACK(lambda _data: time.monotonic() > self._deadline)  # keep a reference
        p.abort_callback = C.cast(self._abort_cb, C.c_void_p)
        self.max_seconds = o.get("max_seconds")
        p.print_progress = p.print_realtime = p.print_timestamps = p.print_special = False
        if beam > 1:
            p.beam_search.beam_size = beam
        self.audio_ctx = int(o.get("audio_ctx", 0))
        self.fallback_language = o.get("language") or (None if self.multilingual else "en")
        self.lock = threading.Lock()  # whisper_full is not thread-safe for one context
        self.runtime.actual = device
        _LOGGER.info(
            "stt %s: loaded %s (%s, %s) on %s, audio_ctx %s",
            self.name, self.config.model.name, L.whisper_model_type_readable(ctx).decode(),
            "multilingual" if self.multilingual else "English-only", self.runtime.summary(), self.audio_ctx or "full",
        )

    def _threads(self) -> int:
        return int(self.config.options.get("threads", 4))

    def _language(self, requested: Optional[str]) -> Optional[str]:
        """Whisper's language code for a request ("en-US" -> "en"); None = auto-detect."""
        if not self.multilingual:
            return "en"
        if requested:
            base = requested.replace("_", "-").split("-")[0].lower()
            if _lib().whisper_lang_id(base.encode()) >= 0:
                return base
        return self.fallback_language

    def transcribe(self, audio: np.ndarray, language: Optional[str]) -> str:
        assert self.ctx, "engine not loaded"
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        L = _lib()
        lang_bytes = (self._language(language) or "auto").encode()  # "auto": detect, then transcribe
        # a reduced audio context only covers audio_ctx / 50 s; for longer audio use the full window
        seconds = len(audio) / self.sample_rate
        audio_ctx = self.audio_ctx if self.audio_ctx and seconds < self.audio_ctx / FRAMES_PER_SECOND - 0.5 else 0
        budget = float(self.max_seconds) if self.max_seconds else max(15.0, 3 * seconds)
        with self.lock:  # the params struct is shared, so it is filled in under the lock too
            p = self.params
            p.language = lang_bytes
            p.detect_language = False  # True would only detect, not transcribe
            p.audio_ctx = audio_ctx
            self._deadline = time.monotonic() + budget
            try:
                rc = L.whisper_full(self.ctx, p, audio.ctypes.data_as(C.POINTER(C.c_float)), len(audio))
            finally:
                self._deadline = float("inf")
            n = L.whisper_full_n_segments(self.ctx)
            parts = [L.whisper_full_get_segment_text(self.ctx, i).decode("utf-8", "replace").strip() for i in range(n)]
        if rc != 0:
            if not any(parts):
                raise RuntimeError(f"whisper_full returned {rc}")
            _LOGGER.warning("stt %s: decode aborted after %.0f s (rc %d); returning the text so far", self.name, budget, rc)
        return _collapse_repeats(" ".join(p for p in parts if p and not _ANNOTATION.match(p)))

    def warm_up(self) -> None:
        # as for Parakeet: some ggml-vulkan pipelines are only compiled when an input length first needs them
        for seconds in (1, 2, 3, 5):
            self.transcribe(np.zeros(seconds * self.sample_rate, dtype=np.float32), "en")

    def close(self) -> None:
        if self.ctx:
            _lib().whisper_free(self.ctx)
            self.ctx = None
