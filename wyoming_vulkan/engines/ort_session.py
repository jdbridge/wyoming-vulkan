"""ONNX Runtime sessions on the WebGPU EP (Dawn -> Vulkan) or the CPU EP, with the GPU path proven (never silent).

Shared by the ONNX-based TTS engines (Piper, Kokoro, KittenTTS).
"""

import logging
import threading
from pathlib import Path

import onnxruntime as ort

from .. import devices
from ..config import GpuConfig
from .base import GpuUnavailable, RuntimeReport

_LOGGER = logging.getLogger(__name__)

# ORT warns about every small node it deliberately keeps on the CPU EP; expected and harmless (DESIGN.md §4.3).
ort.set_default_logger_severity(3)

_WEBGPU_LOCK = threading.Lock()
_WEBGPU_REGISTERED = False


def _register_webgpu() -> str:
    """Register the WebGPU plugin EP once per process; return its EP name."""
    global _WEBGPU_REGISTERED
    import onnxruntime_ep_webgpu as webgpu

    with _WEBGPU_LOCK:
        if not _WEBGPU_REGISTERED:
            ort.register_execution_provider_library("webgpu", webgpu.get_library_path())
            _WEBGPU_REGISTERED = True
    return webgpu.get_ep_name()


_PROCESS_GPU: str | None = None  # the GPU the first WebGPU session got; every later session gets the same one


def power_preference(target, gpus) -> str | None:
    """The WebGPU powerPreference that makes Dawn pick `target` among `gpus` (ggml devices of this process).

    The plug-in (0.4.0) ignores which EP device is passed and lets Dawn choose the adapter: "low-power" gives the
    integrated GPU, "high-performance" (the default) the discrete one (measured with an Intel iGPU + an RTX 4060).
    None: only one GPU, nothing to choose. Raises GpuUnavailable when the target cannot be told apart that way."""
    if len(gpus) <= 1:
        return None
    same_type = [d for d in gpus if d.type == target.type]
    if len(same_type) > 1:
        kind = "integrated" if target.type == "igpu" else "discrete"
        raise GpuUnavailable(f"ONNX Runtime's WebGPU plug-in cannot choose between several {kind} GPUs "
                             f"({', '.join(d.description for d in same_type)})")
    return "low-power" if target.type == "igpu" else "high-performance"


def make_session(model: Path, device: str, gpu: GpuConfig, runtime: RuntimeReport) -> ort.InferenceSession:
    """A session on `device` ("igpu" = the GPU `gpu` selects, or "cpu"); fills runtime.device_name / detail. Raises
    GpuUnavailable when the WebGPU path is not really used (ONNX Runtime would otherwise quietly return a CPU-only
    session) or would land on another GPU than asked for."""
    global _PROCESS_GPU
    so = ort.SessionOptions()
    providers = None
    if device == "igpu":
        problem = devices.vulkan_icd_problem()
        if problem:
            raise GpuUnavailable(problem)
        gpus = devices.gpu_devices()
        target = next((d for d in gpus if gpu.name_contains in d.description), None)
        if target is None:
            raise GpuUnavailable(f"no Vulkan GPU contains {gpu.name_contains!r}; found "
                                 + (", ".join(d.description for d in gpus) or "none"))
        if _PROCESS_GPU is not None and _PROCESS_GPU != target.description:
            # the plug-in would silently run this session on the first session's GPU
            raise GpuUnavailable(f"ONNX Runtime already runs on {_PROCESS_GPU} in this process and cannot use a "
                                 f"second GPU ({target.description}); give all ONNX engines the same gpu")
        preference = power_preference(target, gpus)
        ep_name = _register_webgpu()
        candidates = [d for d in ort.get_ep_devices() if d.ep_name == ep_name]
        if not candidates:
            raise GpuUnavailable(f"{ep_name} offers no device")
        # the plug-in ignores which device is passed (it follows powerPreference); pass the matching one anyway
        chosen = next((d for d in candidates if d.device.vendor_id == gpu.vendor_id), candidates[0])
        so.add_provider_for_devices([chosen], {"powerPreference": preference} if preference else {})
        runtime.device_name = target.description
        how = f"powerPreference {preference} of {len(gpus)} GPUs" if preference else "the only GPU"
    else:
        providers = ["CPUExecutionProvider"]
        runtime.device_name = "CPU"

    session = ort.InferenceSession(str(model), sess_options=so, providers=providers)
    got = session.get_providers()
    if device == "igpu" and (not got or got[0] != "WebGpuExecutionProvider"):
        # The known silent failure: Dawn finds no Vulkan device and ORT runs everything on the CPU EP.
        raise GpuUnavailable(f"ONNX Runtime session providers are {got}, not WebGPU")
    if device == "igpu":
        _PROCESS_GPU = target.description
        runtime.detail = f"WebGPU EP on {target.description} ({how}), providers {got}"
    else:
        runtime.detail = f"providers {got}"
    return session
