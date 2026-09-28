"""Backend registry. Backends are imported only when configured, so each one's dependencies are optional."""

import importlib

from ..config import ConfigError, EngineConfig, GpuConfig
from .base import Engine

# backend name -> (kind, "module:Class")
BACKENDS = {
    "parakeet.cpp": ("stt", "wyoming_vulkan.engines.parakeet_cpp:ParakeetEngine"),
    "whisper.cpp": ("stt", "wyoming_vulkan.engines.whisper_cpp:WhisperEngine"),
    "piper": ("tts", "wyoming_vulkan.engines.piper_ort:PiperEngine"),
    "kokoro": ("tts_pack", "wyoming_vulkan.engines.kokoro_ort:KokoroEngine"),
    "kitten": ("tts_pack", "wyoming_vulkan.engines.kitten_ort:KittenEngine"),
}


def create_engine(config: EngineConfig, gpu: GpuConfig) -> Engine:
    if config.backend not in BACKENDS:
        raise ConfigError(f"{config.kind} {config.name}: unknown backend {config.backend!r} (have {sorted(BACKENDS)})")
    kind, target = BACKENDS[config.backend]
    if kind.split("_")[0] != config.kind:
        raise ConfigError(f"{config.kind} {config.name}: backend {config.backend!r} is a {kind} backend")
    module, cls = target.split(":")
    return getattr(importlib.import_module(module), cls)(config, gpu)
