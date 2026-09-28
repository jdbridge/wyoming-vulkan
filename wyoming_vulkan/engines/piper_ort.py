"""Piper text-to-speech through ONNX Runtime: WebGPU EP (Dawn -> Vulkan) on the iGPU, or the CPU EP.

Backend options in the config: config (voice .onnx.json, default <model>.json), length_scale, noise_scale,
noise_w_scale (default: the voice's own values).
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import onnxruntime as ort
from piper.config import PiperConfig, SynthesisConfig
from piper.voice import PiperVoice

from .. import devices
from .base import GpuUnavailable, SynthesisOptions, TtsEngine

_LOGGER = logging.getLogger(__name__)

# ORT warns about every small node it deliberately keeps on the CPU EP; expected and harmless (DESIGN.md §4.2).
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


class PiperEngine(TtsEngine):
    voice: Optional[PiperVoice] = None

    @property
    def sample_rate(self) -> int:
        assert self.voice is not None, "engine not loaded"
        return self.voice.config.sample_rate

    def _load(self, device: str) -> None:
        model = self.config.model
        config_path = Path(self.config.options.get("config", f"{model}.json"))
        piper_config = PiperConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))

        so = ort.SessionOptions()
        providers = None
        if device == "igpu":
            problem = devices.vulkan_icd_problem()
            if problem:
                raise GpuUnavailable(problem)
            ep_name = _register_webgpu()
            candidates = [d for d in ort.get_ep_devices() if d.ep_name == ep_name]
            chosen = [d for d in candidates if d.device.vendor_id == self.gpu.vendor_id]
            if not chosen:
                raise GpuUnavailable(
                    f"no {ep_name} device with vendor {self.gpu.vendor_id:#06x}; found "
                    + ", ".join(f"{d.device.vendor_id:#06x}:{d.device.device_id:#06x}" for d in candidates)
                )
            hw = chosen[0].device
            so.add_provider_for_devices([chosen[0]], {})
            pci = f"PCI {hw.vendor_id:04x}:{hw.device_id:04x} {hw.metadata.get('pci_bus_id', '')}".strip()
            self.runtime.device_name = devices.vulkan_gpu_name(self.gpu.name_contains) or pci
        else:
            providers = ["CPUExecutionProvider"]
            self.runtime.device_name = "CPU"

        session = ort.InferenceSession(str(model), sess_options=so, providers=providers)
        got = session.get_providers()
        if device == "igpu" and (not got or got[0] != "WebGpuExecutionProvider"):
            # The known silent failure: Dawn finds no Vulkan device and ORT runs everything on the CPU EP.
            raise GpuUnavailable(f"ONNX Runtime session providers are {got}, not WebGPU")
        self.voice = PiperVoice(session=session, config=piper_config)
        self.lock = threading.Lock()
        self.runtime.actual = device
        self.runtime.detail = f"WebGPU EP on {pci}, providers {got}" if device == "igpu" else f"providers {got}"
        _LOGGER.info(
            "tts %s: loaded %s (%d Hz) on %s, %s",
            self.name, model.name, piper_config.sample_rate, self.runtime.summary(), self.runtime.detail,
        )

    def synthesize(self, text: str, options: SynthesisOptions) -> bytes:
        voice = self.voice
        assert voice is not None, "engine not loaded"
        o = self.config.options
        syn = SynthesisConfig(
            length_scale=o.get("length_scale"), noise_scale=o.get("noise_scale"), noise_w_scale=o.get("noise_w_scale")
        )
        if options.speaker is not None:
            syn.speaker_id = voice.config.speaker_id_map.get(options.speaker)
            if syn.speaker_id is None and options.speaker.isdigit():
                syn.speaker_id = int(options.speaker)
        with self.lock:
            return b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(text, syn))

    def warm_up(self) -> None:
        self.synthesize("Warming up.", SynthesisOptions())

    def close(self) -> None:
        """Drop the session (frees its GPU buffers); waits for a synthesis in progress."""
        lock = getattr(self, "lock", None)
        if lock is None:
            self.voice = None
            return
        with lock:
            self.voice = None
