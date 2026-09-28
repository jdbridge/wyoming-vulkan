"""Per-connection Wyoming protocol handling. Knows the engine interfaces, not the backends.

Event sequences follow OHF-Voice/wyoming-faster-whisper 3.8 (STT) and wyoming-piper 2.5 (TTS, incl. streaming):
  STT: [transcribe] -> audio-start -> audio-chunk* -> audio-stop  => transcript (then the connection closes)
  TTS one-shot: synthesize  => audio-start -> audio-chunk* -> audio-stop
  TTS streaming: synthesize-start -> synthesize-chunk* -> [synthesize] -> synthesize-stop
                 => per sentence: audio-start -> audio-chunk* -> audio-stop; finally synthesize-stopped
The `synthesize` inside a stream is only sent for older servers and is ignored.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sentence_stream import SentenceBoundaryDetector
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped, SynthesizeVoice

from .config import ServerConfig
from .engines.base import SttEngine, SynthesisOptions
from .info import build_info
from .voices import VoiceRegistry

_LOGGER = logging.getLogger(__name__)


@dataclass
class Services:
    """Everything a connection needs; shared by all connections."""

    server: ServerConfig
    stt: list[SttEngine]  # this endpoint's engines, in routing order
    tts: Optional[VoiceRegistry]  # None: this endpoint offers no voices
    stt_name: Optional[str] = None  # program name for HA (default: the first engine's label)

    def stt_engine(self, name: Optional[str], language: Optional[str] = None) -> SttEngine:
        """By name if given and known; else the first engine that supports the language; else the first engine."""
        if not self.stt:
            raise RuntimeError("no stt engine on this endpoint")
        if name:
            for e in self.stt:
                if e.name == name:
                    return e
            _LOGGER.warning("stt %r requested but not on this endpoint; routing by language", name)
        if language:
            base = language.replace("_", "-").split("-")[0].lower()
            for e in self.stt:
                if any(l.replace("_", "-").split("-")[0].lower() == base for l in e.languages):
                    return e
            _LOGGER.warning("no stt engine for language %r; using %s", language, self.stt[0].name)
        return self.stt[0]

    async def info_event(self) -> Event:
        """Info as of now: library folders are rescanned (at most every 5 s), so new voices appear."""
        voices = []
        if self.tts is not None:
            await self.tts.refresh(min_interval=5.0)
            voices = self.tts.voices()
        return build_info(self.stt, voices, self.stt_name).event()


def _prepare_text(text: str, auto_punctuation: str) -> str:
    text = " ".join(text.strip().splitlines()).strip()
    if text and auto_punctuation and text[-1] not in auto_punctuation:
        text += auto_punctuation[0]
    return text


class WyomingHandler(AsyncEventHandler):
    def __init__(self, services: Services, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.services = services
        # STT state
        self._stt_name: Optional[str] = None
        self._language: Optional[str] = None
        self._converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self._audio: list[bytes] = []
        self._audio_bytes = 0
        self._truncated = False
        # TTS state
        self._streaming = False
        self._sbd = SentenceBoundaryDetector()
        self._voice: Optional[SynthesizeVoice] = None
        self._stream_started = 0.0

    async def handle_event(self, event: Event) -> bool:
        try:
            return await self._handle(event)
        except Exception as err:  # report to the client, then drop the connection
            _LOGGER.exception("Error handling %s", event.type)
            try:
                await self.write_event(Error(text=str(err), code=err.__class__.__name__).event())
            except (ConnectionError, AttributeError, RuntimeError):
                pass  # the client is already gone
            return False

    async def _handle(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(await self.services.info_event())
            return True

        # ---- speech-to-text ----
        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self._stt_name = transcribe.name
            self._language = transcribe.language
            return True
        if AudioStart.is_type(event.type):
            self._reset_audio()
            return True
        if AudioChunk.is_type(event.type):
            self._add_audio(AudioChunk.from_event(event))
            return True
        if AudioStop.is_type(event.type):
            await self._transcribe()
            self._reset_audio()
            return False  # like wyoming-faster-whisper: one utterance per connection

        # ---- text-to-speech ----
        if self.services.tts is None and event.type in ("synthesize", "synthesize-start", "synthesize-chunk", "synthesize-stop"):
            await self.write_event(Error(text="this endpoint offers no text-to-speech", code="NoTts").event())
            return False
        if Synthesize.is_type(event.type):
            if self._streaming:
                return True  # compatibility copy of the streamed text
            synthesize = Synthesize.from_event(event)
            await self._synthesize_oneshot(synthesize.text, synthesize.voice)
            return True
        if SynthesizeStart.is_type(event.type):
            start = SynthesizeStart.from_event(event)
            self._streaming = True
            self._sbd = SentenceBoundaryDetector()
            self._voice = start.voice
            self._stream_started = time.perf_counter()
            return True
        if SynthesizeChunk.is_type(event.type):
            if not self._streaming:
                _LOGGER.warning("synthesize-chunk outside a stream; ignored")
                return True
            for sentence in self._sbd.add_chunk(SynthesizeChunk.from_event(event).text):
                await self._speak(sentence, self._voice, send_start=True, send_stop=True)
            return True
        if SynthesizeStop.is_type(event.type):
            if self._streaming:
                rest = self._sbd.finish()
                if rest.strip():
                    await self._speak(rest, self._voice, send_start=True, send_stop=True)
                _LOGGER.info("tts stream done in %.2f s", time.perf_counter() - self._stream_started)
            self._streaming = False
            await self.write_event(SynthesizeStopped().event())
            return True

        _LOGGER.debug("Ignoring event %s", event.type)
        return True

    # ---- speech-to-text ----

    def _reset_audio(self) -> None:
        self._audio = []
        self._audio_bytes = 0
        self._truncated = False

    def _add_audio(self, chunk: AudioChunk) -> None:
        chunk = self._converter.convert(chunk)
        limit = int(self.services.server.max_audio_seconds * 16000) * 2
        if self._audio_bytes + len(chunk.audio) > limit:
            if not self._truncated:
                _LOGGER.warning("Audio longer than %.0f s; the rest is dropped", self.services.server.max_audio_seconds)
                self._truncated = True
            return
        self._audio.append(chunk.audio)
        self._audio_bytes += len(chunk.audio)

    async def _transcribe(self) -> None:
        engine = self.services.stt_engine(self._stt_name, self._language)
        language = self._language or engine.languages[0]
        if not self._audio_bytes:
            _LOGGER.warning("audio-stop without audio")
            await self.write_event(Transcript(text="", language=language).event())
            return
        pcm = np.frombuffer(b"".join(self._audio), dtype=np.int16).astype(np.float32) / 32768.0
        t = time.perf_counter()
        text = await asyncio.to_thread(engine.transcribe, pcm, self._language)
        _LOGGER.info(
            "stt %s: %.2f s audio -> %.3f s: %r", engine.name, len(pcm) / 16000, time.perf_counter() - t, text
        )
        await self.write_event(Transcript(text=text, language=language).event())

    # ---- text-to-speech ----

    async def _speak_streaming(self, engine, stream, text, voice, rate, send_start: bool, send_stop: bool) -> None:
        """Run the engine's frame generator in a worker thread and send every frame as it arrives."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        done = object()
        options = SynthesisOptions(speaker=voice.speaker if voice else None)

        def produce() -> None:
            try:
                for pcm in stream(text, options):
                    loop.call_soon_threadsafe(queue.put_nowait, pcm)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        t = time.perf_counter()
        worker = loop.run_in_executor(None, produce)
        if send_start:
            await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
        first, total = None, 0
        step = self.services.server.samples_per_chunk * 2
        while (pcm := await queue.get()) is not done:
            if first is None:
                first = time.perf_counter() - t
            total += len(pcm)
            for offset in range(0, len(pcm), step):
                await self.write_event(AudioChunk(audio=pcm[offset : offset + step], rate=rate, width=2, channels=1).event())
        await worker  # re-raises an engine error
        took, seconds = time.perf_counter() - t, total / 2 / rate
        _LOGGER.info("tts %s: %r -> %.2f s audio in %.3f s (RTF %.2f, first audio %.3f s, streamed)",
                     engine.name, text, seconds, took, took / max(seconds, 1e-6), first or took)
        if send_stop:
            await self.write_event(AudioStop().event())

    async def _synthesize_oneshot(self, text: str, voice: Optional[SynthesizeVoice]) -> None:
        sbd = SentenceBoundaryDetector()
        sentences = list(sbd.add_chunk(text))
        rest = sbd.finish()
        if rest.strip():
            sentences.append(rest)
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            async with self.services.tts.use(voice.name if voice else None) as engine:
                rate = engine.sample_rate
            await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
            await self.write_event(AudioStop().event())
            return
        for i, sentence in enumerate(sentences):
            await self._speak(sentence, voice, send_start=(i == 0), send_stop=(i == len(sentences) - 1))

    async def _speak(self, sentence: str, voice: Optional[SynthesizeVoice], send_start: bool, send_stop: bool) -> None:
        # `use` loads a library voice on first use (or falls back to the default voice) and keeps it from being
        # unloaded while this sentence is synthesised
        async with self.services.tts.use(voice.name if voice else None) as engine:
            text = _prepare_text(sentence, engine.config.options.get("auto_punctuation", ".?!"))
            rate = engine.sample_rate
            stream = getattr(engine, "stream_audio", None)
            if text and stream is not None:  # the engine produces frames: send each one as soon as it exists
                await self._speak_streaming(engine, stream, text, voice, rate, send_start, send_stop)
                return
            pcm = b""
            if text:
                t = time.perf_counter()
                pcm = await asyncio.to_thread(engine.synthesize, text, SynthesisOptions(speaker=voice.speaker if voice else None))
                took, seconds = time.perf_counter() - t, len(pcm) / 2 / rate
                _LOGGER.info("tts %s: %r -> %.2f s audio in %.3f s (RTF %.2f)", engine.name, text, seconds, took, took / max(seconds, 1e-6))
        if send_start:
            await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
        step = self.services.server.samples_per_chunk * 2
        for offset in range(0, len(pcm), step):
            await self.write_event(AudioChunk(audio=pcm[offset : offset + step], rate=rate, width=2, channels=1).event())
        if send_stop:
            await self.write_event(AudioStop().event())
