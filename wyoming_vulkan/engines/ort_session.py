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


def make_session(model: Path, device: str, gpu: GpuConfig, runtime: RuntimeReport) -> ort.InferenceSession:
    """A session on `device` ("igpu" or "cpu"); fills runtime.device_name / detail. Raises GpuUnavailable when the
    WebGPU path is not really used (ONNX Runtime would otherwise quietly return a CPU-only session)."""
    so = ort.SessionOptions()
    providers = None
    pci = ""
    if device == "igpu":
        problem = devices.vulkan_icd_problem()
        if problem:
            raise GpuUnavailable(problem)
        ep_name = _register_webgpu()
        candidates = [d for d in ort.get_ep_devices() if d.ep_name == ep_name]
        chosen = [d for d in candidates if d.device.vendor_id == gpu.vendor_id]
        if not chosen:
            raise GpuUnavailable(
                f"no {ep_name} device with vendor {gpu.vendor_id:#06x}; found "
                + ", ".join(f"{d.device.vendor_id:#06x}:{d.device.device_id:#06x}" for d in candidates)
            )
        hw = chosen[0].device
        so.add_provider_for_devices([chosen[0]], {})
        pci = f"PCI {hw.vendor_id:04x}:{hw.device_id:04x} {hw.metadata.get('pci_bus_id', '')}".strip()
        runtime.device_name = devices.vulkan_gpu_name(gpu.name_contains) or pci
    else:
        providers = ["CPUExecutionProvider"]
        runtime.device_name = "CPU"

    session = ort.InferenceSession(str(model), sess_options=so, providers=providers)
    got = session.get_providers()
    if device == "igpu" and (not got or got[0] != "WebGpuExecutionProvider"):
        # The known silent failure: Dawn finds no Vulkan device and ORT runs everything on the CPU EP.
        raise GpuUnavailable(f"ONNX Runtime session providers are {got}, not WebGPU")
    runtime.detail = f"WebGPU EP on {pci}, providers {got}" if device == "igpu" else f"providers {got}"
    return session
