"""ggml helpers shared by the whisper.cpp-family backends (libparakeet now, libwhisper later).

- routes ggml / libparakeet log output into Python logging and keeps it for device checks
- lists the ggml backend devices (the Vulkan loader decides which GPUs exist; see devices.py)
Constants and signatures: ggml.h / ggml-backend.h of the pinned whisper.cpp v1.9.4.
"""

import ctypes as C
import logging
import threading
from dataclasses import dataclass

_LOGGER = logging.getLogger("wyoming_vulkan.ggml")

# enum ggml_log_level
_LEVELS = {0: logging.DEBUG, 1: logging.DEBUG, 2: logging.DEBUG, 3: logging.WARNING, 4: logging.ERROR}
_CONT = 5
# enum ggml_backend_dev_type
DEV_TYPES = {0: "cpu", 1: "gpu", 2: "igpu", 3: "accel"}

LOG_CALLBACK = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)


class _LogSink:
    """Collects ggml's line fragments into lines; keeps every line for later checks."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._partial = ""
        self._level = logging.DEBUG
        self._lock = threading.Lock()
        self.callback = LOG_CALLBACK(self._on_log)  # keep a reference: ctypes must not free it

    def _on_log(self, level: int, text: bytes, _user: int) -> None:
        with self._lock:
            if level != _CONT:
                self._level = _LEVELS.get(level, logging.DEBUG)
            self._partial += (text or b"").decode("utf-8", "replace")
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                if line.strip():
                    self.lines.append(line)
                    if len(self.lines) > 2000:
                        del self.lines[:1000]
                    _LOGGER.log(self._level, "%s", line)

    def find(self, needle: str) -> list[str]:
        with self._lock:
            return [l for l in self.lines if needle in l]


LOG = _LogSink()


def install_logging(*setters) -> None:
    """Point ggml's and each library's log setter (e.g. parakeet_log_set) at the Python sink."""
    base = C.CDLL("libggml-base.so.0")
    base.ggml_log_set.argtypes = [LOG_CALLBACK, C.c_void_p]
    base.ggml_log_set(LOG.callback, None)
    for setter in setters:
        setter.argtypes = [LOG_CALLBACK, C.c_void_p]
        setter(LOG.callback, None)


@dataclass
class GgmlDevice:
    name: str
    description: str
    type: str


def list_devices() -> list[GgmlDevice]:
    """All ggml backend devices of this process (CPU plus whatever the Vulkan loader exposes)."""
    lib = C.CDLL("libggml.so.0")
    lib.ggml_backend_dev_count.restype = C.c_size_t
    lib.ggml_backend_dev_get.argtypes = [C.c_size_t]
    lib.ggml_backend_dev_get.restype = C.c_void_p
    for fn in ("ggml_backend_dev_name", "ggml_backend_dev_description"):
        getattr(lib, fn).argtypes = [C.c_void_p]
        getattr(lib, fn).restype = C.c_char_p
    lib.ggml_backend_dev_type.argtypes = [C.c_void_p]
    lib.ggml_backend_dev_type.restype = C.c_int
    devices = []
    for i in range(lib.ggml_backend_dev_count()):
        dev = lib.ggml_backend_dev_get(i)
        devices.append(
            GgmlDevice(
                name=(lib.ggml_backend_dev_name(dev) or b"").decode(),
                description=(lib.ggml_backend_dev_description(dev) or b"").decode(),
                type=DEV_TYPES.get(lib.ggml_backend_dev_type(dev), "?"),
            )
        )
    return devices
