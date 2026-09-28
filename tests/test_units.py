"""Unit tests without GPU or models: config, device checks, fallback logic, info, and the protocol handler
(driven over a real TCP socket with fake engines). Standard library only:
  python -m unittest discover -s tests -p "test_*.py"      (inside the image: see tests/run.sh)
"""

import asyncio
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from functools import partial
from pathlib import Path
from unittest import mock

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.error import Error
from wyoming.info import Describe, Info
from wyoming.server import AsyncTcpServer
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeVoice

from wyoming_vulkan import devices
from wyoming_vulkan.config import ConfigError, EngineConfig, GpuConfig, ServerConfig, load_config
from wyoming_vulkan.engines import create_engine
from wyoming_vulkan.engines.base import GpuUnavailable, SttEngine, SynthesisOptions, TtsEngine, TtsPack
from wyoming_vulkan.handler import Services, WyomingHandler
from wyoming_vulkan.info import build_info
from wyoming_vulkan import voices as voices_module
from wyoming_vulkan.config import LibraryConfig
from wyoming_vulkan.voices import VoiceInfo, VoiceRegistry, scan_library

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = next(p for p in (ROOT / "config/config.example.toml", Path("/etc/wyoming-vulkan/config.toml")) if p.exists())


def write(text: str) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
    f.write(text)
    f.close()
    return Path(f.name)


MINIMAL = """
[[stt]]
name = "a"
backend = "parakeet.cpp"
model = "/m.bin"
"""


class ConfigTests(unittest.TestCase):
    def test_example_config(self):
        c = load_config(EXAMPLE)
        self.assertEqual(c.server.uri, "tcp://0.0.0.0:10310")
        self.assertEqual([e.device for e in c.stt + c.tts], ["auto", "auto"], "shipped default is auto (decision 2)")
        self.assertEqual(c.stt[0].options, {"threads": 4})
        self.assertEqual(c.gpu.vendor_id, 0x8086)

    def test_stack_config(self):
        """The stack's inline config as rendered by tests/run.sh (deploy/compose.yaml + deploy/.env)."""
        path = Path("/deploy/config.toml")
        if not path.exists():
            self.skipTest("run via tests/run.sh, which renders the stack config")
        c = load_config(path)
        self.assertEqual(
            [(str(l.path), l.optional, l.recursive) for l in c.tts_library],
            [("/data/models/piper", False, False), ("/data/voices", True, True), ("/voices-extra", True, False)],
        )
        self.assertEqual([(p.name, p.backend, str(p.model.parent)) for p in c.tts_pack],
                         [("kokoro", "kokoro", "/data/models/kokoro"), ("kitten", "kitten", "/data/models/kitten")])
        self.assertEqual([(ep.stt, ep.tts) for ep in c.endpoints], [(["parakeet"], True), (["whisper"], False)])
        self.assertEqual({e.device for e in c.stt + c.tts + c.tts_library}, {"auto"})
        self.assertEqual(c.tts[0].model, Path(f"/data/models/piper/{c.tts[0].name}.onnx"))

    def test_library_config(self):
        c = load_config(write(MINIMAL + '[[tts_library]]\npath = "/v"\nlength_scale = 1.1\n'))
        lib = c.tts_library[0]
        self.assertEqual((lib.path, lib.backend, lib.device, lib.optional, lib.options), (Path("/v"), "piper", "auto", False, {"length_scale": 1.1}))
        with self.assertRaises(ConfigError):
            load_config(write(MINIMAL + '[[tts_library]]\noptional = true\n'))
        many = load_config(write(MINIMAL + '[[tts_library]]\npath = ["/a", "/b"]\noptional = true\n[[tts_library]]\npath = "/c"\n'))
        self.assertEqual([(str(l.path), l.optional) for l in many.tts_library], [("/a", True), ("/b", True), ("/c", False)])

    def test_minimal_defaults(self):
        c = load_config(write(MINIMAL))
        self.assertEqual((c.stt[0].device, c.stt[0].languages, c.tts), ("igpu", ["en"], []))

    def test_errors(self):
        cases = {
            "unknown section": MINIMAL + "\n[extra]\n",
            "unknown server key": "[server]\nport = 1\n" + MINIMAL,
            "bad device": MINIMAL + 'device = "gpu"\n',
            "missing model": '[[tts]]\nname = "v"\nbackend = "piper"\n',
            "no engines": '[server]\nuri = "tcp://0.0.0.0:1"\n',
            "duplicate": MINIMAL + MINIMAL,
        }
        for what, text in cases.items():
            with self.subTest(what), self.assertRaises(ConfigError):
                load_config(write(text))

    def test_backend_registry(self):
        gpu = GpuConfig()
        with self.assertRaises(ConfigError):
            create_engine(EngineConfig(kind="stt", name="x", backend="nope", model=Path("/m")), gpu)
        with self.assertRaises(ConfigError):  # a TTS backend in an STT slot; fails before importing it
            create_engine(EngineConfig(kind="stt", name="x", backend="piper", model=Path("/m")), gpu)


class VulkanIcdTests(unittest.TestCase):
    def icd(self, library: str) -> str:
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"ICD": {"library_path": library}}, f)
        f.close()
        return f.name

    def check(self, env):
        clean = {k: v for k, v in os.environ.items() if k not in ("VK_DRIVER_FILES", "VK_ICD_FILENAMES")}
        with mock.patch.dict(os.environ, {**clean, **env}, clear=True):
            return devices.vulkan_icd_problem()

    def test_cases(self):
        good = self.icd("libvulkan_intel.so")
        self.assertIsNone(self.check({"VK_DRIVER_FILES": good, "VK_ICD_FILENAMES": good}))
        self.assertIn("neither", self.check({}))
        self.assertIn("does not exist", self.check({"VK_DRIVER_FILES": "/nope.json"}))
        self.assertIn("software", self.check({"VK_DRIVER_FILES": self.icd("libvulkan_lvp.so")}))


# ---------------------------------------------------------------- fake engines


class FakeStt(SttEngine):
    def __init__(self, name="fake-stt", fail_gpu=False, delay=0.0, reply=None, **kw):
        super().__init__(EngineConfig(kind="stt", name=name, backend="fake", model=Path("/m"), **kw), GpuConfig())
        self.fail_gpu, self.delay, self.reply, self.calls, self.loads = fail_gpu, delay, reply, [], []
        self.runtime.actual = "igpu"

    def _load(self, device):
        self.loads.append(device)
        if device == "igpu" and self.fail_gpu:
            raise GpuUnavailable("no test GPU")
        self.runtime.actual, self.runtime.device_name = device, f"fake {device}"

    def warm_up(self):
        pass

    def transcribe(self, audio, language):
        self.calls.append((len(audio), audio.dtype, language))
        time.sleep(self.delay)
        if self.reply is Exception:
            raise RuntimeError("engine broke")
        return self.reply or f"{self.name} heard {len(audio)} samples"


class FakeTts(TtsEngine):
    def __init__(self, name="fake-tts", rate=22050, config=None, fail_load=False, **kw):
        config = config or EngineConfig(kind="tts", name=name, backend="fake", model=Path("/m"), **kw)
        super().__init__(config, GpuConfig())
        self.rate, self.texts, self.fail_load, self.closed = rate, [], fail_load, False
        self.runtime.actual, self.runtime.device_name = "igpu", "Fake GPU"

    @property
    def sample_rate(self):
        return self.rate

    def _load(self, device):
        if self.fail_load:
            raise RuntimeError("broken voice file")
        self.runtime.actual = device
        if device == "cpu":
            self.runtime.device_name = "CPU"

    def close(self):
        self.closed = True

    def warm_up(self):
        pass

    def synthesize(self, text, options):
        self.texts.append((text, options.speaker))
        return b"\x01\x00" * int(0.01 * len(text) * self.rate)  # 10 ms of audio per character


class FakeFactory:
    """Stands in for engines.create_engine; records the engines it made. Names in `broken` fail to load."""

    def __init__(self, broken=()):
        self.made, self.broken = [], set(broken)

    def __call__(self, config, gpu):
        engine = FakeTts(config=config, fail_load=config.name in self.broken)
        self.made.append(engine)
        return engine


def make_library(voices, age=3600.0, espeak="en-us"):
    """A temp folder with <name>.onnx + <name>.onnx.json per voice, `age` seconds old."""
    folder = Path(tempfile.mkdtemp())
    for name in voices:
        add_voice(folder, name, age, espeak)
    return folder


def add_voice(folder, name, age=3600.0, espeak="en-us", json_file=True):
    old = time.time() - age
    files = [folder / f"{name}.onnx"] + ([folder / f"{name}.onnx.json"] if json_file else [])
    files[0].write_bytes(b"onnx")
    if json_file:
        files[1].write_text(json.dumps({"espeak": {"voice": espeak}}))
    for f in files:
        os.utime(f, (old, old))


class FallbackTests(unittest.TestCase):
    def test_igpu_refuses(self):
        e = FakeStt(fail_gpu=True, device="igpu")
        with self.assertRaises(GpuUnavailable):
            e.load()

    def test_auto_falls_back_and_says_so(self):
        e = FakeStt(fail_gpu=True, device="auto")
        with self.assertLogs("wyoming_vulkan.engines.base", "WARNING") as logs:
            e.load()
        self.assertEqual(e.loads, ["igpu", "cpu"])
        self.assertTrue(e.runtime.fell_back)
        self.assertIn("FALLBACK", e.runtime.summary())
        self.assertIn("FALLING BACK TO CPU", logs.output[0])

    def test_auto_uses_gpu_when_there(self):
        e = FakeStt(device="auto")
        e.load()
        self.assertEqual((e.loads, e.runtime.fell_back), (["igpu"], False))

    def test_cpu_never_touches_gpu(self):
        e = FakeStt(fail_gpu=True, device="cpu")
        e.load()
        self.assertEqual(e.loads, ["cpu"])

    def test_info_always_states_execution(self):
        ok, fb = FakeStt(name="ok", device="auto"), FakeStt(name="fb", fail_gpu=True, device="auto")
        ok.load()
        fb.load()
        cpu = FakeStt(name="cpu", device="cpu")
        cpu.load()
        info = build_info([ok, fb, cpu], [VoiceInfo("v", "v [Fake GPU]", ["en_US"], "piper")])
        descriptions = [m.description for m in info.asr[0].models]
        self.assertEqual(descriptions, ["ok [fake igpu]", "fb [CPU fallback]", "cpu [CPU]"])
        self.assertTrue(info.tts[0].supports_synthesize_streaming)
        self.assertTrue(info.asr[0].installed and info.asr[0].models[0].installed)


# ---------------------------------------------------------------- handler over TCP


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def audio_events(seconds, rate=16000, width=2, channels=1):
    n = int(seconds * rate)
    pcm = (np.sin(np.arange(n * channels) / 7) * 8000).astype(np.int16).tobytes()
    step = int(rate * 0.02) * width * channels  # 20 ms chunks, like a satellite / HA stream
    chunks = [AudioChunk(audio=pcm[i : i + step], rate=rate, width=width, channels=channels).event() for i in range(0, len(pcm), step)]
    return [AudioStart(rate=rate, width=width, channels=channels).event(), *chunks, AudioStop().event()]


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stt = [FakeStt("stt-a"), FakeStt("stt-b")]
        self.tts = [FakeTts("voice-a"), FakeTts("voice-b", rate=16000)]
        self.server = None

    async def start(self):
        """Start the server after the test has configured the fakes."""
        self.registry = VoiceRegistry(self.tts, getattr(self, "libraries", []), GpuConfig(), 2, FakeFactory())
        await self.registry.refresh()
        services = Services(ServerConfig(max_audio_seconds=5), self.stt, self.registry)
        self.port = free_port()
        self.server = AsyncTcpServer("127.0.0.1", self.port)
        await self.server.start(partial(WyomingHandler, services))

    async def asyncTearDown(self):
        if self.server is not None:
            await self.server.stop()

    async def converse(self, events, until, timeout=5.0):
        """Send events, read until an event whose type is in `until` or the connection closes."""
        got = []
        async with AsyncTcpClient("127.0.0.1", self.port) as c:
            for e in events:
                await c.write_event(e)
            while True:
                e = await asyncio.wait_for(c.read_event(), timeout)
                if e is None:
                    break
                got.append(e)
                if e.type in until:
                    break
        return got

    # -- describe / STT --

    async def test_describe(self):
        await self.start()
        [e] = await self.converse([Describe().event()], {"info"})
        info = Info.from_event(e)
        self.assertEqual([m.name for m in info.asr[0].models], ["stt-a", "stt-b"])
        self.assertEqual([v.name for v in info.tts[0].voices], ["voice-a", "voice-b"])

    async def test_transcribe_default_engine(self):
        await self.start()
        got = await self.converse([Transcribe(language="de").event(), *audio_events(1.0)], {"transcript"})
        t = Transcript.from_event(got[-1])
        self.assertEqual((t.text, t.language), ("stt-a heard 16000 samples", "de"))
        self.assertEqual(self.stt[0].calls, [(16000, np.float32, "de")])

    async def test_transcribe_selects_by_name_and_converts_audio(self):
        await self.start()
        got = await self.converse([Transcribe(name="stt-b").event(), *audio_events(0.5, rate=48000, channels=2)], {"transcript"})
        self.assertEqual(Transcript.from_event(got[-1]).text, "stt-b heard 8000 samples")  # 48 kHz stereo -> 16 kHz mono

    async def test_unknown_stt_name_uses_default(self):
        await self.start()
        got = await self.converse([Transcribe(name="nope").event(), *audio_events(0.25)], {"transcript"})
        self.assertTrue(Transcript.from_event(got[-1]).text.startswith("stt-a"))

    async def test_no_audio_gives_empty_transcript(self):
        await self.start()
        got = await self.converse([AudioStart(rate=16000, width=2, channels=1).event(), AudioStop().event()], {"transcript"})
        self.assertEqual(Transcript.from_event(got[-1]).text, "")
        self.assertEqual(self.stt[0].calls, [])

    async def test_audio_without_audio_start(self):
        await self.start()
        got = await self.converse(audio_events(0.5)[1:], {"transcript"})
        self.assertEqual(Transcript.from_event(got[-1]).text, "stt-a heard 8000 samples")

    async def test_audio_is_capped(self):
        await self.start()
        got = await self.converse(audio_events(7.0), {"transcript"})
        self.assertEqual(Transcript.from_event(got[-1]).text, "stt-a heard 80000 samples")  # max_audio_seconds = 5

    async def test_connection_closes_after_transcript(self):
        await self.start()
        got = await self.converse([*audio_events(0.2), Describe().event()], {"__never__"})
        self.assertEqual([e.type for e in got], ["transcript"])

    async def test_engine_error_is_reported(self):
        self.stt[0].reply = Exception
        await self.start()
        got = await self.converse(audio_events(0.2), {"error"})
        self.assertEqual(Error.from_event(got[-1]).code, "RuntimeError")

    async def test_event_loop_not_blocked_by_inference(self):
        self.stt[0].delay = 1.0
        await self.start()
        stt = asyncio.create_task(self.converse(audio_events(0.2), {"transcript"}))
        await asyncio.sleep(0.2)
        t = time.perf_counter()
        await self.converse([Describe().event()], {"info"})
        self.assertLess(time.perf_counter() - t, 0.3, "describe must not wait for a running transcription")
        await stt

    # -- TTS --

    def audio_summary(self, got):
        types, rates = [], set()
        for e in got:
            if e.type == "audio-chunk":
                c = AudioChunk.from_event(e)
                rates.add(c.rate)
                self.assertLessEqual(len(c.audio), 1024 * 2, "chunk size = samples_per_chunk")
                if types and types[-1] == "audio-chunk":
                    continue
            types.append(e.type)
        return types, rates

    async def test_oneshot(self):
        await self.start()
        got = await self.converse([Synthesize(text="Hello there. How are you").event()], {"audio-stop"})
        self.assertEqual(self.audio_summary(got), (["audio-start", "audio-chunk", "audio-stop"], {22050}))
        self.assertEqual([t for t, _ in self.tts[0].texts], ["Hello there.", "How are you."])  # auto punctuation

    async def test_oneshot_voice_and_speaker(self):
        await self.start()
        got = await self.converse([Synthesize(text="Hi.", voice=SynthesizeVoice(name="voice-b", speaker="3")).event()], {"audio-stop"})
        self.assertEqual(self.audio_summary(got)[1], {16000})
        self.assertEqual(self.tts[1].texts, [("Hi.", "3")])

    async def test_unknown_voice_uses_default(self):
        await self.start()
        await self.converse([Synthesize(text="Hi.", voice=SynthesizeVoice(name="nope")).event()], {"audio-stop"})
        self.assertEqual(len(self.tts[0].texts), 1)

    async def test_empty_text(self):
        await self.start()
        got = await self.converse([Synthesize(text="   ").event()], {"audio-stop"})
        self.assertEqual([e.type for e in got], ["audio-start", "audio-stop"])
        self.assertEqual(self.tts[0].texts, [])

    async def test_streaming(self):
        await self.start()
        v = SynthesizeVoice(name="voice-a")
        events = [SynthesizeStart(voice=v).event()]
        events += [SynthesizeChunk(text=t).event() for t in ("Tomorrow will be ", "cloudy. The ", "high is ", "eighteen")]
        events += [Synthesize(text="Tomorrow will be cloudy. The high is eighteen", voice=v).event(), SynthesizeStop().event()]
        got = await self.converse(events, {"synthesize-stopped"})
        self.assertEqual(self.audio_summary(got)[0], ["audio-start", "audio-chunk", "audio-stop"] * 2 + ["synthesize-stopped"])
        self.assertEqual([t for t, _ in self.tts[0].texts], ["Tomorrow will be cloudy.", "The high is eighteen."])

    async def test_streaming_then_oneshot_on_same_connection(self):
        await self.start()
        events = [SynthesizeStart().event(), SynthesizeChunk(text="One.").event(), SynthesizeStop().event(), Synthesize(text="Two.").event()]
        got = []
        async with AsyncTcpClient("127.0.0.1", self.port) as c:
            for e in events:
                await c.write_event(e)
            stops = 0
            while stops < 2:
                e = await asyncio.wait_for(c.read_event(), 5)
                got.append(e.type)
                stops += e.type == "audio-stop"
        self.assertEqual([t for t, _ in self.tts[0].texts], ["One.", "Two."])
        self.assertIn("synthesize-stopped", got)


class LibraryScanTests(unittest.TestCase):
    def test_scan(self):
        folder = make_library(["en_US-a-high", "de_DE-b-medium"])
        add_voice(folder, "c-plain", espeak="de")  # language from the espeak voice
        add_voice(folder, "en_US-new-high", age=5)  # still being written
        add_voice(folder, "en_US-nojson-high", json_file=False)  # not a Piper voice
        (folder / "notes.wav").write_bytes(b"x")
        found = scan_library(LibraryConfig(path=folder))
        self.assertEqual({n: v[2] for n, v in found.items()}, {"en_US-a-high": ["en_US"], "de_DE-b-medium": ["de_DE"], "c-plain": ["de"]})
        self.assertIn("en_US-new-high", scan_library(LibraryConfig(path=folder, min_age_seconds=0)))

    def test_recursive(self):
        folder = make_library(["top"])
        (folder / "piper" / "de").mkdir(parents=True)
        add_voice(folder / "piper", "sub")
        add_voice(folder / "piper" / "de", "de_DE-deep-low")
        self.assertEqual(sorted(scan_library(LibraryConfig(path=folder))), ["top"])
        self.assertEqual(sorted(scan_library(LibraryConfig(path=folder, recursive=True))), ["de_DE-deep-low", "sub", "top"])

    def test_missing_folder(self):
        with self.assertRaises(OSError):
            scan_library(LibraryConfig(path=Path("/nonexistent/voices")))


class RegistryTests(unittest.IsolatedAsyncioTestCase):
    async def registry(self, libraries, fixed=None, max_loaded=2, broken=()):
        self.factory = FakeFactory(broken)
        fixed = [FakeTts("en_US-fixed-high")] if fixed is None else fixed
        r = VoiceRegistry(fixed, libraries, GpuConfig(), max_loaded, self.factory)
        await r.refresh()
        return r

    async def test_list_dedupe_and_labels(self):
        a = make_library(["en_US-fixed-high", "en_US-one-high"])
        b = make_library(["en_US-one-high", "en_US-two-medium"])
        r = await self.registry([LibraryConfig(path=a), LibraryConfig(path=b), LibraryConfig(path=make_library(["en_US-cpu-low"]), device="cpu")])
        self.assertEqual(r.names(), ["en_US-fixed-high", "en_US-one-high", "en_US-two-medium", "en_US-cpu-low"])
        self.assertEqual(
            [v.description for v in r.voices()],
            ["Fake fixed-high [Fake GPU]", "Piper one-high [Fake GPU]", "Piper two-medium [Fake GPU]", "Piper cpu-low [CPU]"],
        )  # not-yet-loaded voices show where they will run, predicted from the loaded ones
        self.assertEqual(r.library_voices["en_US-one-high"].path.parent, a)  # first folder wins

    async def test_lazy_load_lru_and_active_protection(self):
        r = await self.registry([LibraryConfig(path=make_library(["v1", "v2", "v3"]))], max_loaded=2)
        self.assertEqual((self.factory.made, r.loaded()), ([], []))
        async with r.use("v1") as e1:
            self.assertEqual(e1.name, "v1")
        async with r.use("v2") as e2:
            async with r.use("v3"):
                pass  # v1 is unloaded here (least recently used); v2 is in use and must stay
        self.assertEqual(sorted(r.loaded()), ["v2", "v3"])
        self.assertTrue(e1.closed and not e2.closed)
        async with r.use("v1"):
            pass
        self.assertEqual(len(self.factory.made), 4, "v1 was loaded again")

    async def test_unknown_and_broken_voices_use_default(self):
        folder = make_library(["good", "bad"])
        r = await self.registry([LibraryConfig(path=folder)], broken={"bad"})
        async with r.use("nope") as e:
            self.assertEqual(e.name, "en_US-fixed-high")
        with self.assertLogs("wyoming_vulkan.voices", "ERROR"):
            async with r.use("bad") as e:
                self.assertEqual(e.name, "en_US-fixed-high")
        await r.refresh()
        self.assertNotIn("bad", r.names(), "a voice that failed to load is not offered again...")
        add_voice(folder, "bad", age=1000)  # new mtime
        self.factory.broken.clear()
        with mock.patch.object(voices_module, "SETTLE_S", 0):
            await r.refresh()
        self.assertIn("bad", r.names(), "...until its file changes")
        async with r.use("bad") as e:
            self.assertEqual(e.name, "bad")

    async def test_changed_file_is_reloaded(self):
        folder = make_library(["v1"])
        r = await self.registry([LibraryConfig(path=folder)])
        async with r.use("v1") as first:
            pass
        add_voice(folder, "v1", age=1000)
        with mock.patch.object(voices_module, "SETTLE_S", 0):
            await r.refresh()
        async with r.use("v1") as second:
            pass
        self.assertIsNot(first, second)
        self.assertTrue(first.closed)

    async def test_new_voice_waits_until_files_settle(self):
        folder = make_library(["old"])
        with mock.patch.object(voices_module, "SETTLE_S", 0.3):
            r = await self.registry([LibraryConfig(path=folder)])
            self.assertIn("old", r.names(), "the first scan offers settled-looking files at once")
            add_voice(folder, "copied", age=3600)  # an SMB copy that kept the source's old mtime
            await r.refresh()
            self.assertNotIn("copied", r.names(), "new while running: wait until it stops changing")
            self.assertIn("old", r.names(), "files offered by the first scan stay offered on the next one")
            (folder / "copied.onnx").write_bytes(b"onnx, still growing")  # size changes -> the wait starts again
            old = time.time() - 3600
            os.utime(folder / "copied.onnx", (old, old))
            await asyncio.sleep(0.2)
            await r.refresh()
            self.assertNotIn("copied", r.names())
            await asyncio.sleep(0.35)
            await r.refresh()
            self.assertIn("copied", r.names(), "unchanged for SETTLE_S: offered")
            (folder / "half.onnx").write_bytes(b"onnx")  # config file missing
            os.utime(folder / "half.onnx", (old, old))
            await asyncio.sleep(0.35)
            await r.refresh()
            await asyncio.sleep(0.35)
            await r.refresh()
            self.assertNotIn("half", r.names(), "never without its .onnx.json")

    async def test_optional_missing_folder(self):
        lib = LibraryConfig(path=Path("/nonexistent/voices-nas"), optional=True)
        with self.assertLogs("wyoming_vulkan.voices", "WARNING") as logs:
            r = await self.registry([lib])
        self.assertEqual(r.names(), ["en_US-fixed-high"])
        self.assertIn("optional", logs.output[0])
        self.assertIsNotNone(r.problem(0))

    async def test_hung_scan_does_not_block(self):
        release = threading.Event()

        def hanging_scan(library, now=None):
            release.wait(10)
            return {}

        with mock.patch.object(voices_module, "scan_library", hanging_scan), mock.patch.object(voices_module, "SCAN_TIMEOUT_S", 0.3):
            t = time.perf_counter()
            r = await self.registry([LibraryConfig(path=Path("/nas"), optional=True)])
            first = time.perf_counter() - t
            t = time.perf_counter()
            await r.refresh()
            second = time.perf_counter() - t
            release.set()
        self.assertLess(first, 1.0)
        self.assertLess(second, 0.1, "a scan that is still stuck is not waited for again")
        self.assertIn("did not finish", r.problem(0))
        self.assertEqual(r.names(), ["en_US-fixed-high"])


class HandlerLibraryTests(HandlerTests.__base__):
    """Library voices through the protocol: listed in info, loaded on first use, new files appear."""

    async def asyncSetUp(self):
        self.stt, self.tts = [], [FakeTts("en_US-fixed-high")]
        self.folder = make_library(["en_US-lib-medium"])
        self.libraries = [LibraryConfig(path=self.folder)]
        self.server = None

    start, asyncTearDown, converse = HandlerTests.start, HandlerTests.asyncTearDown, HandlerTests.converse

    async def test_library_voice(self):
        await self.start()
        [e] = await self.converse([Describe().event()], {"info"})
        voices = Info.from_event(e).tts[0].voices
        self.assertEqual([(v.name, v.description, v.languages) for v in voices], [
            ("en_US-fixed-high", "Fake fixed-high [Fake GPU]", ["en"]), ("en_US-lib-medium", "Piper lib-medium [Fake GPU]", ["en_US"])])
        await self.converse([Synthesize(text="Hi.", voice=SynthesizeVoice(name="en_US-lib-medium")).event()], {"audio-stop"})
        self.assertEqual(self.registry.loaded(), ["en_US-lib-medium"])
        self.assertEqual(self.registry.library_voices["en_US-lib-medium"].engine.texts, [("Hi.", None)])
        self.assertEqual(self.tts[0].texts, [])

    async def test_new_voice_appears(self):
        await self.start()
        add_voice(self.folder, "en_US-later-high")
        self.registry._last_refresh = 0  # skip the 5 s rate limit
        with mock.patch.object(voices_module, "SETTLE_S", 0):
            [e] = await self.converse([Describe().event()], {"info"})
        self.assertIn("en_US-later-high", [v.name for v in Info.from_event(e).tts[0].voices])


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    """Several endpoints sharing engines; routing by name and language; per-endpoint info."""

    def test_config(self):
        base = MINIMAL + '[[stt]]\nname = "w"\nbackend = "whisper.cpp"\nmodel = "/w.bin"\nlanguages = "auto"\n'
        c = load_config(write('[server]\nstt = ["w", "a"]\n' + base + '[[endpoint]]\nuri = "tcp://0.0.0.0:10311"\nstt = ["w"]\nname = "Whisper"\n'))
        self.assertEqual([(e.uri, e.stt, e.tts, e.name) for e in c.endpoints], [
            ("tcp://0.0.0.0:10310", ["w", "a"], True, None), ("tcp://0.0.0.0:10311", ["w"], False, "Whisper")])
        self.assertIsNone(c.stt[1].languages)  # "auto": the engine decides
        self.assertEqual(load_config(write(MINIMAL + 'languages = "en, de"\n')).stt[0].languages, ["en", "de"])
        for bad in ('[[endpoint]]\nuri = "tcp://0.0.0.0:10311"\nstt = ["nope"]\n',
                    '[[endpoint]]\nuri = "tcp://0.0.0.0:10310"\n',  # same uri as the main endpoint
                    '[[endpoint]]\nuri = "tcp://0.0.0.0:10311"\nstt = []\n',  # nothing offered
                    '[[endpoint]]\nstt = ["a"]\n'):
            with self.subTest(bad), self.assertRaises(ConfigError):
                load_config(write(base + bad))

    def test_routing(self):
        en, multi = FakeStt("parakeet", languages=["en"]), FakeStt("whisper", languages=None)
        multi.auto_languages = ["en", "de", "ja"]
        s = Services(ServerConfig(), [en, multi], None)
        self.assertEqual(s.stt_engine(None, "en").name, "parakeet")
        self.assertEqual(s.stt_engine(None, "de-DE").name, "whisper")  # region codes match the base language
        self.assertEqual(s.stt_engine(None, "ja").name, "whisper")
        self.assertEqual(s.stt_engine("whisper", "en").name, "whisper")  # an explicit name wins
        with self.assertLogs("wyoming_vulkan.handler", "WARNING"):
            self.assertEqual(s.stt_engine(None, "xx").name, "parakeet")  # nothing fits: the first engine
        self.assertEqual(s.stt_engine(None, None).name, "parakeet")

    async def test_endpoint_info_and_no_tts(self):
        a, b = FakeStt("a"), FakeStt("b", description="Whisper small")
        b.load()
        registry = VoiceRegistry([FakeTts("voice")], [], GpuConfig(), 2, FakeFactory())
        port = free_port()
        server = AsyncTcpServer("127.0.0.1", port)
        await server.start(partial(WyomingHandler, Services(ServerConfig(), [b], None)))
        try:
            async with AsyncTcpClient("127.0.0.1", port) as c:
                await c.write_event(Describe().event())
                info = Info.from_event(await c.read_event())
            self.assertEqual(info.asr[0].name, "Whisper small [fake igpu]")  # HA's STT entity name
            self.assertEqual([m.name for m in info.asr[0].models], ["b"])
            self.assertEqual(info.tts, [])
            async with AsyncTcpClient("127.0.0.1", port) as c:
                await c.write_event(Synthesize(text="Hi.").event())
                e = await asyncio.wait_for(c.read_event(), 5)
            self.assertEqual(Error.from_event(e).code, "NoTts")
        finally:
            await server.stop()
        info = build_info([a, b], registry.voices(), "Speech [custom]")
        self.assertEqual((info.asr[0].name, len(info.tts[0].voices)), ("Speech [custom]", 1))
        a.runtime.device_name = "fake igpu"
        self.assertEqual(build_info([a, b], []).asr[0].name, "a + Whisper small [fake igpu]")
        cpu = FakeStt("c", device="cpu")
        cpu.load()
        self.assertEqual(build_info([b, cpu], []).asr[0].name, "Whisper small [fake igpu] + c [CPU]")


class WhisperTextTests(unittest.TestCase):
    def test_collapse_repeats(self):
        from wyoming_vulkan.engines.whisper_cpp import _collapse_repeats
        self.assertEqual(_collapse_repeats("Turn it on. Turn it on. Turn it on."), "Turn it on.")
        self.assertEqual(_collapse_repeats("Schalte alle aus. Schalte alle aus. Schalte a"), "Schalte alle aus.")
        self.assertEqual(_collapse_repeats("One. Two. One."), "One. Two. One.")
        self.assertEqual(_collapse_repeats("no punctuation"), "no punctuation")


class SpeechSettingsTests(unittest.TestCase):
    def test_precedence_and_empty_values(self):
        c = load_config(write(MINIMAL + '''
[voice_settings."*"]
length_scale = ""
noise_w = 0.6
[voice_settings."v1"]
length_scale = "1.2"
noise_w = ""
'''))
        # "*" < the entry's options < the voice's own table; empty values never erase a value
        self.assertEqual(c.speech_options("v1", {"noise_scale": 0.5}), {"noise_w": 0.6, "noise_scale": 0.5, "length_scale": "1.2"})
        self.assertEqual(c.speech_options("other", {}), {"noise_w": 0.6})
        with self.assertRaises(ConfigError):
            load_config(write(MINIMAL + '[voice_settings."v"]\nspeed = 2\n'))

    def test_piper_values(self):
        from wyoming_vulkan.engines.piper_ort import speech_settings
        self.assertEqual(speech_settings({}), {"length_scale": None, "noise_scale": None, "noise_w_scale": None})
        self.assertEqual(
            speech_settings({"length_scale": "1.2", "noise_scale": "", "noise_w": 0.6}),
            {"length_scale": 1.2, "noise_scale": None, "noise_w_scale": 0.6},
        )
        self.assertEqual(speech_settings({"noise_w_scale": 0.7})["noise_w_scale"], 0.7)

    async def _registry_options(self):
        c = load_config(write(MINIMAL + '[voice_settings."v1"]\nlength_scale = 1.3\n'))
        factory = FakeFactory()
        r = VoiceRegistry([], [LibraryConfig(path=make_library(["v1"]), options={"noise_scale": 0.4})], GpuConfig(), 2,
                          factory, c.speech_options)
        await r.refresh()
        async with r.use("v1"):
            pass
        return factory.made[0].config.options

    def test_folder_voice_gets_its_settings(self):
        self.assertEqual(asyncio.run(self._registry_options()), {"noise_scale": 0.4, "length_scale": 1.3})


class FakePack(TtsPack):
    """A voice pack for tests: two voices, records what it synthesised."""

    def __init__(self, name="kokoro", fail_load=False, voices=("af_x", "bm_y")):
        super().__init__(EngineConfig(kind="tts", name=name, backend="kokoro", model=Path("/m"), device="auto"), GpuConfig())
        self.fail_load, self._voices, self.calls, self.loads = fail_load, voices, [], 0

    @property
    def sample_rate(self):
        return 24000

    def list_voices(self):
        return [(v, ["en_US" if v.startswith("a") else "en_GB"]) for v in self._voices]

    def _load(self, device):
        self.loads += 1
        if self.fail_load:
            raise RuntimeError("model file broken")
        self.runtime.actual, self.runtime.device_name = device, "Fake GPU"

    def warm_up(self):
        pass

    def synthesize(self, text, options):
        self.calls.append((text, options.voice, dict(options.settings)))
        return b"\x01\x00" * 2400


class VoicePackTests(unittest.IsolatedAsyncioTestCase):
    async def registry(self, packs, settings=None):
        c = load_config(write(MINIMAL + (settings or "")))
        r = VoiceRegistry([FakeTts("en_US-fixed-high")], [], GpuConfig(), 2, FakeFactory(), c.speech_options)
        await r.add_packs(packs)
        return r

    async def test_listing_and_names(self):
        pack = FakePack()
        r = await self.registry([pack])
        info = {v.name: (v.description, v.languages) for v in r.voices()}
        self.assertEqual(info["kokoro_af_x"], ("Kokoro af_x [Fake GPU]", ["en_US"]))  # predicted before loading
        self.assertEqual(info["kokoro_bm_y"][1], ["en_GB"])
        self.assertEqual(pack.loads, 0, "packs load on first use, not at start")

    async def test_use_loads_once_and_passes_voice_and_settings(self):
        pack = FakePack()
        r = await self.registry([pack], '[voice_settings."kokoro_bm_y"]\nlength_scale = 1.2\n')
        for name in ("kokoro_af_x", "kokoro_bm_y", "kokoro_af_x"):
            async with r.use(name) as e:
                self.assertEqual(e.sample_rate, 24000)
                e.synthesize("Hi.", SynthesisOptions())
        self.assertEqual(pack.loads, 1)
        self.assertEqual([(v, s) for _, v, s in pack.calls], [("af_x", {}), ("bm_y", {"length_scale": 1.2}), ("af_x", {})])

    async def test_broken_pack_uses_default_voice_and_disappears(self):
        pack = FakePack(fail_load=True)
        r = await self.registry([pack])
        with self.assertLogs("wyoming_vulkan.voices", "ERROR"):
            async with r.use("kokoro_af_x") as e:
                self.assertEqual(e.name, "en_US-fixed-high")
        self.assertNotIn("kokoro_af_x", [v.name for v in r.voices()])

    async def test_unreadable_pack_is_left_out(self):
        class Unreadable(FakePack):
            def list_voices(self):
                raise FileNotFoundError("/data/models/kokoro/voices")

        with self.assertLogs("wyoming_vulkan.voices", "WARNING"):
            r = await self.registry([Unreadable()])
        self.assertEqual(r.names(), ["en_US-fixed-high"])

    def test_pack_speed_from_length_scale(self):
        pack = FakePack()
        self.assertEqual(pack.speed(SynthesisOptions()), 1.0)
        self.assertAlmostEqual(pack.speed(SynthesisOptions(settings={"length_scale": "1.25"})), 0.8)


class FrontEndTests(unittest.TestCase):
    def test_kitten_symbols_follow_the_reference(self):
        from wyoming_vulkan.engines.kitten_ort import _PUNCTUATION, SYMBOL_IDS
        # pad, then the 16 punctuation characters, then the letters (reference: TextCleaner in kittentts)
        self.assertEqual((SYMBOL_IDS["$"], SYMBOL_IDS[";"], SYMBOL_IDS["A"]), (0, 1, 1 + len(_PUNCTUATION)))
        self.assertEqual(SYMBOL_IDS["ˈ"], max(i for s, i in SYMBOL_IDS.items() if s == "ˈ"))

    def test_kokoro_languages(self):
        from wyoming_vulkan.engines.kokoro_ort import LANGUAGES
        self.assertNotIn("j", LANGUAGES)
        self.assertEqual(LANGUAGES["b"], ("en_GB", "en-gb"))


if __name__ == "__main__":
    unittest.main()
