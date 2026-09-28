"""TTS voices: the fixed [[tts]] engines (loaded at start) plus folders of voices ([[tts_library]]), loaded on
first use and unloaded again when more than `max_loaded_voices` library voices are in memory.

Library folders may be on NFS with a `hard` mount, where a NAS outage makes file access hang instead of fail. So
every scan and every load of a library voice runs in a worker thread with a timeout: a hung NAS costs that voice
(the default voice is used instead), never the event loop, and never speech-to-text.
"""

import asyncio
import fnmatch
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

from .config import EngineConfig, GpuConfig, LibraryConfig
from .engines.base import SynthesisOptions, TtsEngine, TtsPack

_LOGGER = logging.getLogger(__name__)

SCAN_TIMEOUT_S = 5.0
LOAD_TIMEOUT_S = 90.0
# A voice that appears (or changes) while the server runs is offered only after its files have looked the same (mtime
# and sizes of .onnx and .onnx.json) for this long. mtime alone is not enough: an SMB copy from Windows can give the
# new file the source's old mtime while it is still being written. The first scan of a folder offers files at once.
SETTLE_S = 10.0

Signature = tuple[float, int, int]  # newest mtime, size of .onnx, size of .onnx.json
_LANG_FROM_NAME = re.compile(r"^([a-z]{2,3}_[A-Z]{2})-")


@dataclass
class VoiceInfo:
    """What `info` advertises for one voice."""

    name: str
    description: str
    languages: list[str]
    backend: str


@dataclass
class _LibraryVoice:
    name: str
    path: Path
    sig: Signature
    languages: list[str]
    library: LibraryConfig
    engine: Optional[TtsEngine] = None
    loaded_sig: Optional[Signature] = None
    active: int = 0
    last_used: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _languages(path: Path, config_path: Path, library: LibraryConfig) -> list[str]:
    if library.languages:
        return list(library.languages)
    m = _LANG_FROM_NAME.match(path.stem)
    if m:
        return [m.group(1)]
    try:
        voice = json.loads(config_path.read_text(encoding="utf-8"))
        code = (voice.get("language") or {}).get("code")
        if code:
            return [code]
        espeak = (voice.get("espeak") or {}).get("voice", "")
        if espeak:
            lang, _, region = espeak.partition("-")
            return [f"{lang}_{region.upper()}" if region else lang]
    except (OSError, ValueError, AttributeError):
        pass
    return ["en"]


def scan_library(library: LibraryConfig, now: Optional[float] = None) -> dict[str, tuple[Path, Signature, list[str]]]:
    """name -> (model path, signature, languages) for voices whose .onnx and .onnx.json both exist and were not changed
    within min_age_seconds. Raises OSError if the folder is unusable."""
    now = time.time() if now is None else now
    found = {}
    for model in sorted(library.path.glob("**/*.onnx" if library.recursive else "*.onnx")):
        if model.stem in found:
            continue  # same voice name in another subfolder: the first (in path order) wins
        config_path = model.with_name(model.name + ".json")
        try:
            m, c = model.stat(), config_path.stat()
        except FileNotFoundError:
            continue  # no voice config (yet): not a Piper voice, or not finished
        sig = (max(m.st_mtime, c.st_mtime), m.st_size, c.st_size)
        if now - sig[0] < library.min_age_seconds or not (m.st_size and c.st_size):
            continue  # possibly still being written
        found[model.stem] = (model, sig, _languages(model, config_path, library))
    if not found and not library.path.is_dir():
        raise OSError(f"{library.path} is not a folder")
    return found


def _strip_language(name: str) -> str:
    m = _LANG_FROM_NAME.match(name)
    return name[m.end():] if m else name


ENGINE_TITLES = {"piper": "Piper", "kokoro": "Kokoro", "kitten": "Kitten", "pocket": "Pocket"}  # HA shows "<Engine> <voice> [<device>]"


def _title(backend: str) -> str:
    return ENGINE_TITLES.get(backend, backend.capitalize())


@dataclass
class _Pack:
    """A voice pack (one model, many voices), loaded on first use and then kept."""

    engine: TtsPack
    voices: dict[str, list[str]]  # voice id -> languages
    loaded: bool = False
    failed: Optional[str] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class _PackVoice:
    """What `use()` hands out for a pack voice: the pack's engine bound to one voice and its speech settings."""

    def __init__(self, engine: TtsPack, voice: str, name: str, settings: dict) -> None:
        self.engine, self.voice, self.name, self.settings = engine, voice, name, settings
        self.config = engine.config
        self.runtime = engine.runtime

    @property
    def sample_rate(self) -> int:
        return self.engine.sample_rate

    def _options(self, options: SynthesisOptions) -> SynthesisOptions:
        return SynthesisOptions(speaker=options.speaker, voice=self.voice, settings=self.settings)

    def synthesize(self, text: str, options: SynthesisOptions) -> bytes:
        return self.engine.synthesize(text, self._options(options))

    @property
    def stream_audio(self):
        """The engine's frame streaming bound to this voice, or None if the engine only does whole sentences."""
        stream = getattr(self.engine, "stream_audio", None)
        return (lambda text, options: stream(text, self._options(options))) if stream else None


class VoiceRegistry:
    def __init__(
        self,
        fixed: list[TtsEngine],
        libraries: list[LibraryConfig],
        gpu: GpuConfig,
        max_loaded: int,
        factory: Callable[[EngineConfig, GpuConfig], TtsEngine],
        speech_options: Optional[Callable[[str, dict], dict]] = None,
    ) -> None:
        # (voice name, the folder's options) -> the voice's options with its speech settings (Config.speech_options)
        self.speech_options = speech_options or (lambda _name, options: dict(options))
        self.fixed = {e.name: e for e in fixed}
        self.libraries = libraries
        self.gpu = gpu
        self.max_loaded = max(1, max_loaded)
        self.factory = factory
        self.library_voices: dict[str, _LibraryVoice] = {}
        self.packs: dict[str, _Pack] = {}  # pack name -> pack
        self.pack_voices: dict[str, tuple[_Pack, str]] = {}  # "<pack>_<voice>" -> (pack, voice id)
        self._scanned: dict[int, dict] = {}  # library index -> last good scan
        self._scan_pool = [ThreadPoolExecutor(1, thread_name_prefix=f"scan{i}") for i in range(len(libraries))]
        self._scan_futures: dict[int, asyncio.Future] = {}
        self._problems: dict[int, Optional[str]] = {}
        self._failed: dict[str, Signature] = {}  # name -> signature of voice files that failed to load
        self._seen: dict[tuple[int, str], tuple[Signature, float]] = {}  # (library, name) -> (signature, first seen)
        self._scanned_once: set[int] = set()  # libraries with at least one successful scan
        self._last_refresh = 0.0

    # ---- voice packs ----

    async def add_packs(self, engines: list[TtsPack]) -> None:
        """List each pack's voices (small files only, in a thread with a timeout: they may be on a network share).
        A pack whose files cannot be read is left out with a warning; it never stops the server."""
        for engine in engines:
            try:
                voices = await asyncio.wait_for(asyncio.to_thread(engine.list_voices), SCAN_TIMEOUT_S * 2)
            except Exception as err:  # missing files, NAS not answering
                _LOGGER.warning("tts pack %s: cannot list voices (%s); its voices are not offered", engine.name, err or type(err).__name__)
                continue
            # pack options include / exclude: voice-id patterns (fnmatch), e.g. exclude = ["cosette"]
            opts = engine.config.options
            include, exclude = opts.get("include") or ["*"], opts.get("exclude") or []
            voices = [(v, langs) for v, langs in voices
                      if any(fnmatch.fnmatch(v, p) for p in include) and not any(fnmatch.fnmatch(v, p) for p in exclude)]
            if not voices:
                _LOGGER.warning("tts pack %s: no voices found; nothing offered", engine.name)
                continue
            pack = _Pack(engine, dict(voices))
            self.packs[engine.name] = pack
            for voice, _languages in voices:
                self.pack_voices[f"{engine.name}_{voice}"] = (pack, voice)
            _LOGGER.info("tts pack %s: %d voices, loaded on first use", engine.name, len(voices))

    async def _ensure_pack(self, pack: _Pack) -> bool:
        async with pack.lock:
            if pack.loaded:
                return True
            if pack.failed:
                return False
            t = time.perf_counter()
            try:
                await asyncio.wait_for(asyncio.to_thread(pack.engine.load), LOAD_TIMEOUT_S)
                await asyncio.wait_for(asyncio.to_thread(pack.engine.warm_up), LOAD_TIMEOUT_S)
            except Exception as err:
                pack.failed = str(err) or type(err).__name__
                _LOGGER.error("tts pack %s could not be loaded: %s; its voices use the default voice", pack.engine.name, pack.failed)
                return False
            pack.loaded = True
            _LOGGER.info("tts pack %s loaded on demand in %.1f s on %s", pack.engine.name, time.perf_counter() - t, pack.engine.runtime.summary())
            return True

    # ---- scanning ----

    async def refresh(self, min_interval: float = 0.0) -> None:
        """Rescan the library folders (at most every `min_interval` s). Never blocks longer than SCAN_TIMEOUT_S."""
        if time.monotonic() - self._last_refresh < min_interval:
            return
        self._last_refresh = time.monotonic()
        loop = asyncio.get_running_loop()
        for i, library in enumerate(self.libraries):
            future = self._scan_futures.get(i)
            if future is not None and not future.done():
                # an earlier scan is still stuck (hung NFS): do not wait again, keep the library off the list
                self._scanned[i] = {}
                continue
            future = loop.run_in_executor(self._scan_pool[i], scan_library, library)
            self._scan_futures[i] = future
            try:
                self._scanned[i] = self._settled(i, await asyncio.wait_for(asyncio.shield(future), SCAN_TIMEOUT_S))
                self._set_problem(i, None)
            except asyncio.TimeoutError:
                self._set_problem(i, f"scan did not finish within {SCAN_TIMEOUT_S:.0f} s (NAS not answering?)")
                self._scanned[i] = {}
            except OSError as err:
                self._set_problem(i, str(err))
                self._scanned[i] = {}
        self._merge()

    def _settled(self, i: int, found: dict) -> dict:
        """Keep the voices whose files have stopped changing (SETTLE_S); all of them on a folder's first scan."""
        now = time.monotonic()
        first_scan = i not in self._scanned_once
        self._scanned_once.add(i)
        settled = {}
        for name, (path, sig, languages) in found.items():
            seen = self._seen.get((i, name))
            if seen is None or seen[0] != sig:
                # on a folder's first scan, count the files as settled already, so they stay offered on the next scans
                self._seen[(i, name)] = seen = (sig, now - SETTLE_S if first_scan else now)
            if first_scan or now - seen[1] >= SETTLE_S:
                settled[name] = (path, sig, languages)
            else:
                _LOGGER.debug("voice %s in %s is new or changing; offered once its files settle", name, self.libraries[i].path)
        for key in [k for k in self._seen if k[0] == i and k[1] not in found]:
            del self._seen[key]
        return settled

    def _set_problem(self, i: int, problem: Optional[str]) -> None:
        library = self.libraries[i]
        if problem == self._problems.get(i, "unset"):
            return  # log changes only, not every 30 s
        self._problems[i] = problem
        if problem is None:
            _LOGGER.info("tts library %s: available", library.path)
        elif library.optional:
            _LOGGER.warning("tts library %s (optional): %s; its voices are not offered", library.path, problem)
        else:
            _LOGGER.error("tts library %s: %s", library.path, problem)

    def problem(self, i: int) -> Optional[str]:
        return self._problems.get(i)

    def _merge(self) -> None:
        seen = set(self.fixed)
        current: dict[str, _LibraryVoice] = {}
        for i, library in enumerate(self.libraries):
            for name, (path, sig, languages) in self._scanned.get(i, {}).items():
                if name in seen:
                    continue  # a fixed voice or an earlier library wins
                if self._failed.get(name) == sig:
                    continue  # failed to load; offered again once the files change
                seen.add(name)
                entry = self.library_voices.get(name)
                if entry is None or entry.path != path:
                    entry = _LibraryVoice(name, path, sig, languages, library)
                else:
                    entry.sig, entry.languages = sig, languages
                current[name] = entry
        for name, gone in self.library_voices.items():
            if name not in current and gone.engine is not None and gone.active == 0:
                gone.engine.close()
        self.library_voices = current

    # ---- what info advertises ----

    def _predicted_label(self, device: str) -> str:
        """Where a not-yet-loaded voice would run: as the loaded voices on the same kind of device do."""
        if device == "cpu":
            return "CPU"
        loaded = list(self.fixed.values()) + [v.engine for v in self.library_voices.values() if v.engine]
        loaded += [p.engine for p in self.packs.values() if p.loaded]
        for e in loaded:
            if e.config.device != "cpu":
                return e.runtime.label()
        return "not loaded"

    def voices(self) -> list[VoiceInfo]:
        out = [
            VoiceInfo(e.name, f"{e.config.description or _title(e.config.backend) + ' ' + _strip_language(e.name)} [{e.runtime.label()}]",
                      e.languages, e.config.backend)
            for e in self.fixed.values()
        ]
        for v in self.library_voices.values():
            label = v.engine.runtime.label() if v.engine else self._predicted_label(v.library.device)
            out.append(VoiceInfo(v.name, f"{_title(v.library.backend)} {_strip_language(v.name)} [{label}]", v.languages, v.library.backend))
        for name, (pack, voice) in self.pack_voices.items():
            if pack.failed:
                continue
            label = pack.engine.runtime.label() if pack.loaded else self._predicted_label(pack.engine.config.device)
            title = pack.engine.config.description or _title(pack.engine.config.backend)
            out.append(VoiceInfo(name, f"{title} {voice} [{label}]", pack.voices[voice], pack.engine.config.backend))
        return out

    def names(self) -> list[str]:
        return list(self.fixed) + list(self.library_voices) + list(self.pack_voices)

    # ---- using a voice ----

    def default(self) -> TtsEngine:
        if self.fixed:
            return next(iter(self.fixed.values()))
        loaded = [v.engine for v in self.library_voices.values() if v.engine]
        if loaded:
            return loaded[0]
        raise RuntimeError("no voice available")

    @asynccontextmanager
    async def use(self, name: Optional[str]) -> AsyncIterator[TtsEngine]:
        """The engine for a voice, loading a library voice if needed. Unknown or failing voices -> the default."""
        if name in self.fixed:
            yield self.fixed[name]
            return
        if name in self.pack_voices:
            pack, voice = self.pack_voices[name]
            if await self._ensure_pack(pack):
                yield _PackVoice(pack.engine, voice, name, self.speech_options(name, pack.engine.config.options))
            else:
                yield self.default()
            return
        entry = self.library_voices.get(name) if name else None
        if entry is None:
            if name and name not in self.fixed:
                _LOGGER.warning("voice %r is not available; using the default voice", name)
            if not self.fixed and self.library_voices:
                entry = next(iter(self.library_voices.values()))
            else:
                yield self.default()
                return
        entry.active += 1
        try:
            engine = await self._ensure_loaded(entry)
            entry.last_used = time.monotonic()
            yield engine if engine is not None else self.default()
        finally:
            entry.active -= 1
        await self._evict()

    async def _ensure_loaded(self, entry: _LibraryVoice) -> Optional[TtsEngine]:
        async with entry.lock:
            if entry.engine is not None and entry.loaded_sig == entry.sig:
                return entry.engine
            if entry.engine is not None:  # the file changed on disk: load the new version
                _LOGGER.info("voice %s changed on disk; reloading", entry.name)
                old, entry.engine = entry.engine, None
                await asyncio.to_thread(old.close)
            library = entry.library
            config = EngineConfig(
                kind="tts", name=entry.name, backend=library.backend, model=entry.path, device=library.device,
                languages=entry.languages, options=self.speech_options(entry.name, library.options),
            )
            t = time.perf_counter()
            try:
                engine = self.factory(config, self.gpu)
                await asyncio.wait_for(asyncio.to_thread(engine.load), LOAD_TIMEOUT_S)
            except Exception as err:  # bad file, NAS gone, timeout: this voice is skipped until its file changes
                _LOGGER.error("voice %s could not be loaded from %s: %s; using the default voice", entry.name, entry.path, err or type(err).__name__)
                self._failed[entry.name] = entry.sig
                return None
            entry.engine, entry.loaded_sig = engine, entry.sig
            _LOGGER.info("voice %s loaded on demand in %.1f s on %s", entry.name, time.perf_counter() - t, engine.runtime.summary())
            return engine

    async def _evict(self) -> None:
        loaded = [v for v in self.library_voices.values() if v.engine is not None]
        excess = len(loaded) - self.max_loaded
        for v in sorted(loaded, key=lambda v: v.last_used):
            if excess <= 0:
                break
            if v.active or v.lock.locked():
                continue
            engine, v.engine = v.engine, None
            await asyncio.to_thread(engine.close)
            _LOGGER.info("voice %s unloaded (max_loaded_voices = %d)", v.name, self.max_loaded)
            excess -= 1

    def loaded(self) -> list[str]:
        return [v.name for v in self.library_voices.values() if v.engine is not None]
