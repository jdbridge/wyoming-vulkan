"""Start-up: read config -> load engines on their devices -> verify -> warm up -> open the Wyoming port."""

import argparse
import asyncio
import logging
import platform
import sys
import time
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from wyoming.server import AsyncServer

from . import __version__, devices
from .config import DEFAULT_CONFIG, DEVICES, ConfigError, load_config
from .engines import BACKENDS, create_engine
from .engines.base import GpuUnavailable
from .handler import Services, WyomingHandler
from .voices import VoiceRegistry

_LOGGER = logging.getLogger("wyoming_vulkan")


def _versions() -> str:
    out = [f"wyoming-vulkan {__version__}", f"python {platform.python_version()}"]
    for pkg in ("wyoming", "onnxruntime", "onnxruntime-ep-webgpu", "piper-tts", "numpy"):
        try:
            out.append(f"{pkg} {version(pkg)}")
        except PackageNotFoundError:
            pass
    return ", ".join(out)


async def main() -> int:
    parser = argparse.ArgumentParser(prog="wyoming_vulkan", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML config (default: %(default)s)")
    parser.add_argument("--uri", help="override [server] uri, e.g. tcp://0.0.0.0:10310")
    parser.add_argument("--device", choices=DEVICES, help="override the device of every engine")
    parser.add_argument("--debug", action="store_true", help="debug logging, incl. ggml output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _LOGGER.info("%s", _versions())

    try:
        config = load_config(args.config)
    except (OSError, ConfigError) as err:
        _LOGGER.error("Config %s: %s", args.config, err)
        return 2
    if args.uri:
        config.server.uri = args.uri
    if args.device:
        for e in config.stt + config.tts + config.tts_library:
            e.device = args.device
    for lib in config.tts_library:
        if BACKENDS.get(lib.backend, ("",))[0] != "tts":
            _LOGGER.error("Config %s: [[tts_library]] %s: %r is not a TTS backend", args.config, lib.path, lib.backend)
            return 2

    devices.log_environment()

    started = time.perf_counter()
    stt, tts = [], []
    try:
        for cfg in config.tts:  # merge the speech settings ([voice_settings]) into each fixed voice's options
            cfg.options = config.speech_options(cfg.name, cfg.options)
        for cfg in config.stt + config.tts:
            engine = create_engine(cfg, config.gpu)
            t = time.perf_counter()
            engine.load()
            loaded = time.perf_counter() - t
            t = time.perf_counter()
            engine.warm_up()
            _LOGGER.info("%s %s: load %.1f s, warm-up %.1f s", cfg.kind, cfg.name, loaded, time.perf_counter() - t)
            (stt if cfg.kind == "stt" else tts).append(engine)
    except GpuUnavailable as err:
        _LOGGER.error("!!! iGPU requested (device = \"igpu\") but not available: %s", err)
        _LOGGER.error("!!! Refusing to start. Use device = \"auto\" to fall back to the CPU, or \"cpu\".")
        return 3
    except (ConfigError, OSError, RuntimeError) as err:
        _LOGGER.error("Start-up failed: %s", err)
        return 1

    for engine in stt + tts:
        level = logging.WARNING if engine.runtime.fell_back else logging.INFO
        _LOGGER.log(level, "%s %-28s %s", engine.config.kind.upper(), engine.name, engine.runtime.summary())
    if any(e.runtime.fell_back for e in stt + tts):
        _LOGGER.warning("!!! Running with CPU FALLBACK for at least one engine (see above); voice is slower.")

    voices = VoiceRegistry(
        tts, config.tts_library, config.gpu, config.server.max_loaded_voices, create_engine, config.speech_options
    )
    await voices.refresh()
    for i, lib in enumerate(config.tts_library):
        problem = voices.problem(i)
        if problem and not lib.optional:
            _LOGGER.error("Start-up failed: tts library %s: %s (set optional = true to allow this)", lib.path, problem)
            return 1
    library_names = [n for n in voices.names() if n not in {e.name for e in tts}]
    _LOGGER.info(
        "Voices: %d fixed, %d from %d library folder(s), loaded on first use (max %d at once)",
        len(tts), len(library_names), len(config.tts_library), config.server.max_loaded_voices,
    )
    # One Wyoming server per endpoint, all sharing the loaded engines (HA: one STT entity per endpoint)
    by_name = {e.name: e for e in stt}
    servers = []
    for ep in config.endpoints:
        ep_stt = [by_name[n] for n in ep.stt] if ep.stt is not None else list(stt)
        services = Services(server=config.server, stt=ep_stt, tts=voices if ep.tts else None, stt_name=ep.name)
        servers.append((AsyncServer.from_uri(ep.uri), partial(WyomingHandler, services)))
        _LOGGER.info(
            "Endpoint %s: stt %s, tts %s", ep.uri, " > ".join(e.name for e in ep_stt) or "none",
            "all voices" if ep.tts else "none",
        )
    _LOGGER.info("Ready on %s after %.1f s", ", ".join(ep.uri for ep in config.endpoints), time.perf_counter() - started)
    await asyncio.gather(*(server.run(factory) for server, factory in servers))
    return 0


def run() -> None:
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    run()
