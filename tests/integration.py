"""Integration test against a running server with the real engines, playing Home Assistant's side of the protocol.

  python tests/integration.py --uri tcp://127.0.0.1:10310 --audio tests/data/en --expect igpu
Checks: info (programs, streaming, the "[execution]" label of every engine, expected device), STT on testdata/cmdN.wav sent in
20 ms chunks like HA (accuracy against truth.txt), one-shot and streamed TTS (event order, audio, time to first
audio, RTF), edge cases with the real engines, and STT + TTS at the same moment. Latency limits depend on where the
engines run (read from info). Exit code 1 if any check fails.
"""

import argparse
import asyncio
import re
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.error import Error
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeVoice


@dataclass
class Limits:
    """Regression bars, with headroom over what was measured on an Intel i3-N305 iGPU (2026-09-28)."""

    stt_s: float  # per command (1.8-3.4 s of audio)
    tts_first_s: float  # time to first audio, one short sentence
    tts_rtf: float  # synthesis time / audio time
    stt_concurrent_s: float
    tts_concurrent_s: float


LIMITS = {
    # measured: STT 0.44-0.71, first audio 0.61-0.80 (one spike of 1.06 in ~8 runs), RTF 0.39-0.45, concurrent 1.33 / 1.88
    "igpu": Limits(stt_s=1.0, tts_first_s=1.2, tts_rtf=0.6, stt_concurrent_s=2.0, tts_concurrent_s=3.0),
    # measured (host busy with Piper training, load 4-6 on 7 vCPUs): STT 0.43-1.01, first audio 1.15-1.43,
    # concurrent STT 3.1-4.4 s (follows host load; a sanity bar for the fallback path, not a performance target)
    "cpu": Limits(stt_s=2.0, tts_first_s=2.5, tts_rtf=1.2, stt_concurrent_s=6.0, tts_concurrent_s=6.0),
}

FAILS: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        FAILS.append(what)


# Counted as equal: the models write digits, truth.txt spells numbers out.
EQUIVALENT = {"twenty one": "21", "seven pm": "7 pm", "7pm": "7 pm"}


def norm(text: str) -> str:
    text = " ".join(re.sub(r"[^a-z0-9 ]", "", text.lower()).split())
    for words, digits in EQUIVALENT.items():
        text = text.replace(words, digits)
    return text


def runs_on(description: str) -> str:
    """info descriptions end in "[<GPU name>]", "[CPU]" or "[CPU fallback]"; return igpu | cpu | ?"""
    m = re.search(r"\[([^\]]+)\]$", description or "")
    if not m:
        return "?"
    return "cpu" if m.group(1).startswith("CPU") else "igpu"


async def read_until(c: AsyncClient, until: set[str]) -> list:
    got = []
    while True:
        e = await asyncio.wait_for(c.read_event(), 30)
        if e is None:
            return got
        if Error.is_type(e.type):
            raise RuntimeError(f"server error: {Error.from_event(e).text}")
        got.append(e)
        if e.type in until:
            return got


async def describe(uri: str) -> Info:
    async with AsyncClient.from_uri(uri) as c:
        await c.write_event(Describe().event())
        return Info.from_event((await read_until(c, {"info"}))[-1])


def load_wav(path: Path) -> tuple[int, int, int, bytes]:
    with wave.open(str(path)) as w:
        return w.getframerate(), w.getsampwidth(), w.getnchannels(), w.readframes(w.getnframes())


async def transcribe(uri: str, rate, width, channels, pcm: bytes, model=None, language="en") -> tuple[str, float]:
    """Like HA: transcribe, audio-start, 20 ms audio-chunks, audio-stop; latency = audio-stop -> transcript."""
    async with AsyncClient.from_uri(uri) as c:
        await c.write_event(Transcribe(name=model, language=language).event())
        await c.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
        step = int(rate * 0.02) * width * channels
        for i in range(0, len(pcm), step):
            await c.write_event(AudioChunk(audio=pcm[i : i + step], rate=rate, width=width, channels=channels).event())
        t = time.perf_counter()
        await c.write_event(AudioStop().event())
        got = await read_until(c, {"transcript"})
        return Transcript.from_event(got[-1]).text, time.perf_counter() - t


@dataclass
class TtsResult:
    types: list[str]
    seconds: float
    rate: int
    first_s: float
    total_s: float


async def synthesize(uri: str, events: list, until: str) -> TtsResult:
    async with AsyncClient.from_uri(uri) as c:
        t = time.perf_counter()
        for e in events:
            await c.write_event(e)
        first, types, nbytes, rate = 0.0, [], 0, 0
        while True:  # timed as events arrive (not after reading them all)
            e = await asyncio.wait_for(c.read_event(), 30)
            if e is None:
                raise RuntimeError("connection closed")
            if Error.is_type(e.type):
                raise RuntimeError(f"server error: {Error.from_event(e).text}")
            if e.type == until:
                types.append(e.type)
                break
            if AudioChunk.is_type(e.type):
                chunk = AudioChunk.from_event(e)
                if not nbytes:
                    first = time.perf_counter() - t
                nbytes += len(chunk.audio)
                rate = chunk.rate
                if types and types[-1] == "audio-chunk":
                    continue
            types.append(e.type)
        return TtsResult(types, nbytes / 2 / max(rate, 1), rate, first, time.perf_counter() - t)


ONE_BLOCK = ["audio-start", "audio-chunk", "audio-stop"]


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default="tcp://127.0.0.1:10310")
    p.add_argument("--audio", type=Path, default=Path("tests/data/en"))
    p.add_argument("--expect", choices=("igpu", "cpu", "any"), default="any", help="where the engines must run")
    p.add_argument("--min-voices", type=int, default=1, help="at least this many voices in info (library folders)")
    p.add_argument("--min-exact", type=int, default=5, help="STT: at least this many of the six commands exact")
    p.add_argument("--stt-limit", type=float, help="STT: seconds per command (default: by device)")
    p.add_argument("--audio-de", type=Path, help="German clips (tests/data/de): sent with language de, must be routed "
                   "to an engine that gets at least --min-exact-de of them right")
    p.add_argument("--min-exact-de", type=int, default=3)
    args = p.parse_args()
    uri = args.uri

    print("info")
    info = await describe(uri)
    check(bool(info.asr and info.asr[0].models), "info has an ASR model")
    if args.min_voices > 0:
        check(bool(info.tts and info.tts[0].voices), "info has a TTS voice")
        check(bool(info.tts and info.tts[0].supports_synthesize_streaming), "TTS advertises streaming")
    else:
        check(not info.tts, "no voices on this endpoint")
    items = [*(m for prog in info.asr for m in prog.models), *(v for prog in info.tts for v in prog.voices)]
    devices = {}
    for item in items:
        devices[item.name] = runs_on(item.description)
        print(f"       {item.name}: {item.description}")
        check(devices[item.name] in ("igpu", "cpu"), f"{item.name}: info states where it runs")
    if args.expect != "any":
        check(all(d == args.expect for d in devices.values()), f"all engines run on {args.expect}: {devices}")

    stt_name = info.asr[0].models[0].name if info.asr else None
    voice = info.tts[0].voices[0].name if info.tts else None
    stt_limits = LIMITS.get(devices.get(stt_name, "cpu"), LIMITS["cpu"])
    if args.stt_limit:
        stt_limits = Limits(**{**stt_limits.__dict__, "stt_s": args.stt_limit})
    tts_limits = LIMITS.get(devices.get(voice, "cpu"), LIMITS["cpu"])

    if info.asr:
        print(f"speech-to-text (20 ms chunks, limit {stt_limits.stt_s} s)")
        truth = dict(l.strip().split("|", 1) for l in (args.audio / "truth.txt").read_text().splitlines() if "|" in l)
        exact = 0
        solo = {}
        for key in sorted(truth):
            text, took = await transcribe(uri, *load_wav(args.audio / f"{key}.wav"), model=stt_name)
            solo[key] = text
            exact += norm(text) == norm(truth[key])
            print(f"       {key} {took:.3f} s  {text!r}")
            check(took <= stt_limits.stt_s, f"{key} transcript within {stt_limits.stt_s} s ({took:.3f})")
        check(exact >= args.min_exact, f"{exact}/{len(truth)} transcripts exact (at least {args.min_exact})")

        if args.audio_de:
            print("speech-to-text, German (language routing)")
            truth_de = dict(l.strip().split("|", 1) for l in (args.audio_de / "truth.txt").read_text().splitlines() if "|" in l)
            exact_de = 0
            for key in sorted(truth_de):
                text, took = await transcribe(uri, *load_wav(args.audio_de / f"{key}.wav"), language="de")
                exact_de += norm(text).replace("21", "einundzwanzig") == norm(truth_de[key])
                print(f"       {key} {took:.3f} s  {text!r}")
            check(exact_de >= args.min_exact_de, f"{exact_de}/{len(truth_de)} German transcripts exact (at least {args.min_exact_de})")

        print("speech-to-text edge cases")
        rate, width, channels, pcm = load_wav(args.audio / "cmd1.wav")
        text, _ = await transcribe(uri, rate, width, channels, b"")
        check(text == "", f"no audio -> empty transcript ({text!r})")
        text, _ = await transcribe(uri, rate, width, channels, pcm[: int(0.1 * rate) * width])
        check(isinstance(text, str), f"0.1 s of audio -> a transcript without error ({text!r})")
        text, _ = await transcribe(uri, rate, width, channels, pcm, model="no-such-model")
        check(norm(text) == norm(truth["cmd1"]), f"unknown model name -> default model ({text!r})")

    if info.tts:
        v = SynthesizeVoice(name=voice)
        print(f"text-to-speech, one-shot (first audio <= {tts_limits.tts_first_s} s, RTF <= {tts_limits.tts_rtf})")
        r = await synthesize(uri, [Synthesize(text="Turned on the kitchen lights. The living room is now set to twenty one degrees.", voice=v).event()], "audio-stop")
        print(f"       {r.types}, {r.seconds:.2f} s audio at {r.rate} Hz, first audio {r.first_s:.3f} s, total {r.total_s:.3f} s")
        check(r.types == ONE_BLOCK, "one-shot: one audio block")
        check(r.seconds > 3.0, f"one-shot audio length ({r.seconds:.2f} s)")
        check(r.first_s <= tts_limits.tts_first_s, f"one-shot first audio ({r.first_s:.3f} s)")
        check(r.total_s / r.seconds <= tts_limits.tts_rtf, f"one-shot RTF ({r.total_s / r.seconds:.2f})")

        print("text-to-speech, streamed")
        events = [SynthesizeStart(voice=v).event()]
        events += [SynthesizeChunk(text=t).event() for t in ("Tomorrow will be ", "mostly cloudy. ", "The high ", "is eighteen degrees.")]
        events += [Synthesize(text="Tomorrow will be mostly cloudy. The high is eighteen degrees.", voice=v).event(), SynthesizeStop().event()]
        r = await synthesize(uri, events, "synthesize-stopped")
        print(f"       {r.types}, {r.seconds:.2f} s audio, first audio {r.first_s:.3f} s")
        check(r.types == ONE_BLOCK * 2 + ["synthesize-stopped"], "streamed: one audio block per sentence, then synthesize-stopped")
        check(r.seconds > 2.0, f"streamed audio length ({r.seconds:.2f} s)")
        check(r.first_s <= tts_limits.tts_first_s, f"streamed first audio ({r.first_s:.3f} s)")

        print(f"voice library ({len(info.tts[0].voices)} voices)")
        check(len(info.tts[0].voices) >= args.min_voices, f"at least {args.min_voices} voices offered")
        others = [x.name for x in info.tts[0].voices if x.name != voice and not x.name.startswith(("kokoro_", "kitten_"))]
        if others:
            lib_voice = next((n for n in others if "medium" in n), others[0])
            text = "The front door is locked and the lights are off."
            cold = await synthesize(uri, [Synthesize(text=text, voice=SynthesizeVoice(name=lib_voice)).event()], "audio-stop")
            warm = await synthesize(uri, [Synthesize(text=text, voice=SynthesizeVoice(name=lib_voice)).event()], "audio-stop")
            print(f"       {lib_voice}: first use {cold.total_s:.2f} s (loads the voice), then {warm.total_s:.3f} s for {warm.seconds:.2f} s audio (RTF {warm.total_s / max(warm.seconds, 1e-6):.2f})")
            check(cold.types == ONE_BLOCK and cold.seconds > 1.0, f"library voice {lib_voice} speaks")
            check(cold.total_s <= 15.0, f"first use incl. loading within 15 s ({cold.total_s:.2f})")
            check(warm.first_s <= tts_limits.tts_first_s, f"library voice warm first audio ({warm.first_s:.3f} s)")
            again = await describe(uri)
            desc = next(x.description for x in again.tts[0].voices if x.name == lib_voice)
            check(runs_on(desc) in ("igpu", "cpu"), f"loaded library voice states where it runs ({desc})")

        for prefix in ("kokoro_", "kitten_"):
            pack_voices = [x.name for x in info.tts[0].voices if x.name.startswith(prefix)]
            if not pack_voices:
                continue
            name = pack_voices[0]
            print(f"voice pack {prefix[:-1]} ({len(pack_voices)} voices), {name}")
            text = "The front door is locked and the lights are off."
            cold = await synthesize(uri, [Synthesize(text=text, voice=SynthesizeVoice(name=name)).event()], "audio-stop")
            warm = await synthesize(uri, [Synthesize(text=text, voice=SynthesizeVoice(name=name)).event()], "audio-stop")
            print(f"       first use {cold.total_s:.2f} s (loads the pack), then {warm.total_s:.3f} s for {warm.seconds:.2f} s audio at {warm.rate} Hz (RTF {warm.total_s / max(warm.seconds, 1e-6):.2f})")
            check(cold.types == ONE_BLOCK and cold.seconds > 1.0, f"{name} speaks")
            check(cold.total_s <= 30.0, f"{name}: first use incl. loading within 30 s ({cold.total_s:.2f})")
            desc = next(x.description for x in (await describe(uri)).tts[0].voices if x.name == name)
            check(runs_on(desc) in ("igpu", "cpu") and (args.expect == "any" or runs_on(desc) == args.expect),
                  f"{name} states where it runs ({desc})")

        print("text-to-speech edge cases")
        r = await synthesize(uri, [Synthesize(text="  ", voice=v).event()], "audio-stop")
        check(r.types == ["audio-start", "audio-stop"], f"empty text -> audio-start, audio-stop ({r.types})")
        r = await synthesize(uri, [Synthesize(text="Hello", voice=SynthesizeVoice(name="no-such-voice")).event()], "audio-stop")
        check(r.types == ONE_BLOCK and r.seconds > 0.3, "unknown voice -> default voice")
        r = await synthesize(uri, [Synthesize(text="It's 21°C & sunny: 100% (really)!\nNew line.", voice=v).event()], "audio-stop")
        check(r.types == ONE_BLOCK and r.seconds > 1.0, f"numbers, symbols and a line break ({r.seconds:.2f} s)")

    if info.asr and info.tts:
        lim = LIMITS["cpu"] if "cpu" in (devices[stt_name], devices[voice]) else LIMITS["igpu"]
        print(f"STT and TTS at the same moment (limits {lim.stt_concurrent_s} / {lim.tts_concurrent_s} s)")
        results = []
        for _ in range(3):
            stt_task = transcribe(uri, *load_wav(args.audio / "cmd4.wav"), model=stt_name)
            tts_task = synthesize(uri, [Synthesize(text="Tomorrow will be mostly cloudy with a high of eighteen degrees.", voice=v).event()], "audio-stop")
            (text, stt_s), r = await asyncio.gather(stt_task, tts_task)
            results.append((stt_s, r.total_s, text, r.types))
        stt_s = sorted(x[0] for x in results)[1]
        tts_s = sorted(x[1] for x in results)[1]
        print(f"       median STT {stt_s:.3f} s, TTS {tts_s:.3f} s (research: 1.33 / 1.93)")
        check(all(norm(x[2]) == norm(solo["cmd4"]) for x in results), "concurrent transcripts same as alone")
        check(all(x[3] == ONE_BLOCK for x in results), "concurrent TTS complete")
        check(stt_s <= lim.stt_concurrent_s, f"concurrent STT median ({stt_s:.3f} s)")
        check(tts_s <= lim.tts_concurrent_s, f"concurrent TTS median ({tts_s:.3f} s)")

    print(("FAILED: " + "; ".join(FAILS)) if FAILS else "all checks passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    asyncio.run(main())
