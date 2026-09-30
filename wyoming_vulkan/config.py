"""TOML configuration: server, GPU recognition, STT and TTS engines, and folders of voices (TTS libraries)."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

DEVICES = ("gpu", "igpu", "cpu", "auto")  # "gpu" and "igpu" mean the same: the GPU chosen by `gpu`, never the CPU
DEFAULT_CONFIG = Path("/etc/wyoming-vulkan/config.toml")


class ConfigError(ValueError):
    pass


@dataclass
class ServerConfig:
    uri: str = "tcp://0.0.0.0:10310"  # the main endpoint
    stt: Optional[list[str]] = None  # STT engines offered on the main endpoint (default: all), in routing order
    tts: bool = True  # voices offered on the main endpoint
    name: Optional[str] = None  # program name HA shows for the main endpoint's STT (default: first engine's label)
    samples_per_chunk: int = 1024  # audio-chunk size sent to clients, as in wyoming-piper
    max_audio_seconds: float = 60.0  # longer STT input is cut off (HA's VAD ends commands long before)
    max_warm_models: int = 1  # TTS models kept loaded (each in its own worker); the least recently used one is stopped
    web_port: int = 10312  # diagnostics web page (wyoming_vulkan/web.py); 0 = off
    web_host: str = "0.0.0.0"


# `gpu = "..."` shortcuts: name substring of the Vulkan device and PCI vendor id
GPU_ALIASES = {"intel": ("Intel", 0x8086), "nvidia": ("NVIDIA", 0x10de), "amd": ("AMD", 0x1002)}


@dataclass
class GpuConfig:
    """Which GPU an engine uses: the Vulkan device whose name contains `name_contains`. [gpu] is the default for every
    engine; an engine's `gpu = "intel" | "nvidia" | "amd" | "<part of the device name>"` overrides it."""

    name_contains: str = "Intel"  # substring of the ggml / Vulkan device description
    vendor_id: Optional[int] = 0x8086  # PCI vendor (None: any)

    def select(self, selector: Optional[str]) -> "GpuConfig":
        """The GPU for an engine whose config says `gpu = selector` (None or "": this default)."""
        if not selector:
            return self
        alias = GPU_ALIASES.get(selector.strip().lower())
        return GpuConfig(*alias) if alias else GpuConfig(name_contains=selector.strip(), vendor_id=None)


@dataclass
class Attribution:
    name: str
    url: str


@dataclass
class EngineConfig:
    kind: str  # "stt" or "tts"
    name: str
    backend: str
    model: Path
    device: str = "igpu"
    gpu: Optional[str] = None  # which GPU (GpuConfig.select); None: the [gpu] section
    languages: Optional[list[str]] = field(default_factory=lambda: ["en"])  # None ("auto"): the engine decides
    description: Optional[str] = None
    attribution: Optional[Attribution] = None
    options: dict[str, Any] = field(default_factory=dict)  # backend-specific keys


@dataclass
class EndpointConfig:
    """A Wyoming port. HA makes one STT entity per port and never picks a model itself, so every STT choice that
    should be selectable in HA gets its own endpoint. All endpoints share the same loaded engines."""

    uri: str
    stt: Optional[list[str]] = None  # engine names in routing order (None: all); a request goes to the first engine
    #                                  that supports its language, else to the first engine
    tts: bool = False
    name: Optional[str] = None


@dataclass
class LibraryConfig:
    """A folder of voices: every <name>.onnx with a <name>.onnx.json next to it is offered, loaded on first use."""

    path: Path
    backend: str = "piper"
    device: str = "auto"
    gpu: Optional[str] = None  # which GPU for these voices (GpuConfig.select); None: the [gpu] section
    optional: bool = False  # missing or unreadable folder: warning instead of refusing to start
    recursive: bool = False  # also scan subfolders (e.g. piper/); the first file with a name wins
    min_age_seconds: float = 60.0  # skip files changed more recently (a training run may still be writing them)
    languages: Optional[list[str]] = None  # default: from the file name (en_US-...), else the voice's espeak voice
    options: dict[str, Any] = field(default_factory=dict)  # backend options for every voice, as in [[tts]]


@dataclass
class Config:
    server: ServerConfig
    gpu: GpuConfig
    stt: list[EngineConfig]
    tts: list[EngineConfig]
    tts_library: list[LibraryConfig] = field(default_factory=list)
    # voice packs ([[tts_pack]]): one model with many voices (Kokoro, KittenTTS); name = the voice-id prefix
    tts_pack: list[EngineConfig] = field(default_factory=list)
    endpoint: list[EndpointConfig] = field(default_factory=list)  # extra endpoints ([[endpoint]])
    # speech settings by voice name, "*" = every voice: {"*": {"noise_w": 0.6}, "en_US-x-high": {"length_scale": 1.1}}
    voice_settings: dict[str, dict[str, Any]] = field(default_factory=dict)

    def speech_options(self, voice: str, entry_options: dict[str, Any]) -> dict[str, Any]:
        """A voice's engine options with its speech settings merged in. Precedence (last wins): [voice_settings."*"],
        the voice's [[tts]] / [[tts_library]] entry, [voice_settings."<voice>"]; the voice's own .onnx.json values
        apply to whatever is still unset. Empty values count as unset."""
        merged: dict[str, Any] = {}
        for source in (self.voice_settings.get("*", {}), entry_options, self.voice_settings.get(voice, {})):
            merged.update({k: v for k, v in source.items() if v != ""})
        return merged

    @property
    def endpoints(self) -> list[EndpointConfig]:
        """The main endpoint ([server]) followed by the extra ones."""
        main = EndpointConfig(uri=self.server.uri, stt=self.server.stt, tts=self.server.tts, name=self.server.name)
        return [main, *self.endpoint]


_ENGINE_KEYS = {"name", "backend", "model", "device", "gpu", "languages", "description", "attribution"}


def _device(where: str, value: str) -> str:
    if value not in DEVICES:
        raise ConfigError(f"{where}: device must be one of {DEVICES}, not {value!r}")
    return "igpu" if value == "gpu" else value


def _section(raw: dict, key: str, cls):
    data = dict(raw.get(key, {}))
    if key == "server" and "max_loaded_voices" in data:  # the name before 0.10.0
        data.setdefault("max_warm_models", data.pop("max_loaded_voices"))
    unknown = set(data) - set(cls.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"[{key}]: unknown keys {sorted(unknown)}")
    return cls(**data)


def _languages(value) -> Optional[list[str]]:
    """A list, "auto" (the engine decides: None), or a comma-separated string such as "en,de" (handy in .env)."""
    if value == "auto":
        return None
    if isinstance(value, str):
        return [v.strip() for v in value.replace(" ", ",").split(",") if v.strip()]
    return list(value)


def _engine(kind: str, i: int, data: dict) -> EngineConfig:
    where = f"[[{kind}]] #{i + 1}"
    for key in ("name", "backend", "model"):
        if not data.get(key):
            raise ConfigError(f"{where}: '{key}' is required")
    device = _device(where, data.get("device", "igpu"))
    attribution = data.get("attribution")
    if attribution is not None:
        attribution = Attribution(**attribution)
    return EngineConfig(
        kind=kind,
        name=data["name"],
        backend=data["backend"],
        model=Path(data["model"]),
        device=device,
        gpu=data.get("gpu") or None,
        languages=_languages(data.get("languages", ["en"])),
        description=data.get("description"),
        attribution=attribution,
        options={k: v for k, v in data.items() if k not in _ENGINE_KEYS},
    )


_LIBRARY_KEYS = {"path", "backend", "device", "gpu", "optional", "recursive", "min_age_seconds", "languages"}


def _libraries(i: int, data: dict) -> list[LibraryConfig]:
    """One [[tts_library]] entry; `path` is a folder or a list of folders that share the entry's settings."""
    where = f"[[tts_library]] #{i + 1}"
    paths = data.get("path")
    paths = [paths] if isinstance(paths, str) else paths
    if not paths or not all(isinstance(p, str) and p for p in paths):
        raise ConfigError(f"{where}: 'path' is required (a folder or a list of folders)")
    known = {k: data[k] for k in _LIBRARY_KEYS - {"path"} if k in data}
    known["device"] = _device(where, data.get("device", "auto"))
    known["gpu"] = data.get("gpu") or None
    options = {k: v for k, v in data.items() if k not in _LIBRARY_KEYS}
    return [LibraryConfig(path=Path(p), **known, options=dict(options)) for p in paths]


SPEECH_SETTINGS = {"length_scale", "noise_scale", "noise_w", "noise_w_scale"}  # noise_w: Piper's .onnx.json name


def _voice_settings(data: dict) -> dict[str, dict[str, Any]]:
    out = {}
    for voice, settings in data.items():
        if not isinstance(settings, dict):
            raise ConfigError(f"[voice_settings.\"{voice}\"] must be a table")
        unknown = set(settings) - SPEECH_SETTINGS
        if unknown:
            raise ConfigError(f"[voice_settings.\"{voice}\"]: unknown keys {sorted(unknown)} (have {sorted(SPEECH_SETTINGS)})")
        out[voice] = dict(settings)
    return out


def _endpoint(i: int, data: dict) -> EndpointConfig:
    where = f"[[endpoint]] #{i + 1}"
    unknown = set(data) - set(EndpointConfig.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    if not data.get("uri"):
        raise ConfigError(f"{where}: 'uri' is required")
    return EndpointConfig(**data)


def load_config(path: Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    unknown = set(raw) - {"server", "gpu", "stt", "tts", "tts_library", "tts_pack", "endpoint", "voice_settings"}
    if unknown:
        raise ConfigError(f"unknown sections {sorted(unknown)}")
    config = Config(
        server=_section(raw, "server", ServerConfig),
        gpu=_section(raw, "gpu", GpuConfig),
        stt=[_engine("stt", i, d) for i, d in enumerate(raw.get("stt", []))],
        tts=[_engine("tts", i, d) for i, d in enumerate(raw.get("tts", []))],
        tts_library=[lib for i, d in enumerate(raw.get("tts_library", [])) for lib in _libraries(i, d)],
        tts_pack=[_engine("tts", i, {"device": "auto", **d}) for i, d in enumerate(raw.get("tts_pack", []))],
        endpoint=[_endpoint(i, d) for i, d in enumerate(raw.get("endpoint", []))],
        voice_settings=_voice_settings(raw.get("voice_settings", {})),
    )
    if not (config.stt or config.tts or config.tts_library or config.tts_pack):
        raise ConfigError("no [[stt]], [[tts]], [[tts_library]] or [[tts_pack]] configured")
    for engines in (config.stt, config.tts, config.tts_pack):
        names = [e.name for e in engines]
        if len(names) != len(set(names)):
            raise ConfigError(f"duplicate engine names: {names}")
    stt_names = {e.name for e in config.stt}
    uris = [ep.uri for ep in config.endpoints]
    if len(uris) != len(set(uris)):
        raise ConfigError(f"duplicate endpoint uris: {uris}")
    for ep in config.endpoints:
        unknown_stt = set(ep.stt or []) - stt_names
        if unknown_stt:
            raise ConfigError(f"endpoint {ep.uri}: unknown stt engines {sorted(unknown_stt)} (have {sorted(stt_names)})")
        if ep.stt == [] and not ep.tts:
            raise ConfigError(f"endpoint {ep.uri} offers neither stt nor tts")
    return config

