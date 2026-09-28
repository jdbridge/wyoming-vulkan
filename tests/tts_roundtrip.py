"""TTS round trip: every engine speaks a set of sentences, Parakeet transcribes them back, word error rate per voice.

Run in the image (models mounted), e.g.:
  python tests/tts_roundtrip.py --parakeet /models/parakeet/ggml-parakeet-tdt-0.6b-v3-q8_0.bin \
      --kokoro /models/kokoro/model.onnx --kokoro-voices af_heart,bm_george --kitten /models/kitten/kitten_tts_mini_v0_8.onnx \
      --piper /voices/en_US-ljspeech-high.onnx --device igpu --gpu-vs-cpu 30
--gpu-vs-cpu N additionally synthesises N sentences with the first Kokoro voice on the GPU and on the CPU and compares
them (catches GPU-only corruption such as onnxruntime issue #29807). Exit code 1 if a voice's WER exceeds --max-wer.
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

from wyoming_vulkan.config import EngineConfig, GpuConfig
from wyoming_vulkan.engines.base import SynthesisOptions
from wyoming_vulkan.engines.kitten_ort import KittenEngine
from wyoming_vulkan.engines.kokoro_ort import KokoroEngine
from wyoming_vulkan.engines.parakeet_cpp import ParakeetEngine
from wyoming_vulkan.engines.piper_ort import PiperEngine

SENTENCES = [
    "Turned on the kitchen lights.",
    "The living room is now set to twenty one degrees.",
    "Tomorrow will be mostly cloudy with a high of eighteen degrees.",
    "All downstairs lights are off and the front door is locked.",
    "Your package from the pharmacy will arrive on Thursday afternoon.",
    "I have started the vacuum in the bedroom.",
    "The garage door has been open for fifteen minutes.",
    "Would you like me to turn off the television as well?",
    "The washing machine finished its cycle ten minutes ago.",
    "It is currently raining in Munich, so take an umbrella.",
    "Good morning, the coffee machine is warming up now.",
    "The battery of the smoke detector in the hallway is low.",
    "Your next appointment is at half past three with the dentist.",
    "I could not find a device called garden lamp.",
    "The dishwasher still needs about forty minutes.",
    "Motion was detected at the back door a moment ago.",
    "The thermostat in the office is set to nineteen degrees.",
    "All windows are closed and the alarm is armed.",
    "The weather forecast calls for snow later this evening.",
    "Remind me to call my mother on Sunday.",
    "The air quality in the kids room is good.",
    "Playing your evening playlist in the living room.",
    "The mailbox sensor says there is new mail.",
    "Energy use today is lower than yesterday.",
    "The robot mower has returned to its base.",
    "Sunset today is at seven forty two.",
    "The front porch light will switch off at midnight.",
    "Humidity in the bathroom is above seventy percent.",
    "I set a timer for twelve minutes.",
    "The guest room heating is switched off.",
]
NUMBERS = {"twenty one": "21", "eighteen": "18", "fifteen": "15", "ten": "10", "forty": "40", "nineteen": "19",
           "twelve": "12", "seventy": "70", "seven": "7", "forty two": "42", "three": "3"}


def words(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9% ]", " ", text.lower().replace("-", " "))
    text = " ".join(text.split())
    for w, d in sorted(NUMBERS.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"\b{w}\b", d, text)
    return text.replace("percent", "%").replace("%", " %").split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(len(r), 1)


def to16k(pcm: bytes, rate: int) -> np.ndarray:
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    n = int(len(x) * 16000 / rate)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def engine(cls, name, model, device, **options):
    e = cls(EngineConfig(kind="tts", name=name, backend=name, model=Path(model), device=device, options=options), GpuConfig())
    e.load()
    return e


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--parakeet", required=True)
    p.add_argument("--kokoro")
    p.add_argument("--kokoro-voices", default="af_heart")
    p.add_argument("--kitten")
    p.add_argument("--kitten-voices", default="")
    p.add_argument("--piper")
    p.add_argument("--device", default="igpu")
    p.add_argument("--sentences", type=int, default=len(SENTENCES))
    p.add_argument("--gpu-vs-cpu", type=int, default=0)
    p.add_argument("--max-wer", type=float, default=0.15)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    sentences = SENTENCES[: args.sentences]

    stt = ParakeetEngine(EngineConfig(kind="stt", name="p", backend="parakeet.cpp", model=Path(args.parakeet), device=args.device), GpuConfig())
    stt.load()
    stt.warm_up()

    voices = []  # (label, engine, options)
    if args.piper:
        e = engine(PiperEngine, "piper", args.piper, args.device)
        voices.append((f"piper {Path(args.piper).stem}", e, SynthesisOptions()))
    if args.kokoro:
        e = engine(KokoroEngine, "kokoro", args.kokoro, args.device)
        for v in args.kokoro_voices.split(","):
            voices.append((f"kokoro {v}", e, SynthesisOptions(voice=v)))
    if args.kitten:
        e = engine(KittenEngine, "kitten", args.kitten, args.device)
        names = args.kitten_voices.split(",") if args.kitten_voices else [v for v, _ in e.list_voices()]
        for v in names:
            voices.append((f"kitten {v}", e, SynthesisOptions(voice=v)))

    failed = False
    print(f"{'voice':28s} {'WER':>6s} {'exact':>7s} {'RTF':>5s} {'first':>6s}  worst")
    for label, e, opts in voices:
        e.synthesize("Warming up.", opts)
        total_err, exact, synth_s, audio_s, firsts, worst = 0.0, 0, 0.0, 0.0, [], (0.0, "", "")
        for text in sentences:
            t = time.perf_counter()
            pcm = e.synthesize(text, opts)
            took = time.perf_counter() - t
            synth_s += took
            firsts.append(took)
            audio_s += len(pcm) / 2 / e.sample_rate
            heard = stt.transcribe(to16k(pcm, e.sample_rate), "en")
            w = wer(text, heard)
            total_err += w
            exact += w == 0
            if w > worst[0]:
                worst = (w, text, heard)
        mean = total_err / len(sentences)
        failed |= mean > args.max_wer
        print(f"{label:28s} {mean:6.1%} {exact:3d}/{len(sentences):<3d} {synth_s / audio_s:5.2f} {np.median(firsts):5.2f}s"
              f"  {worst[0]:.0%}: {worst[1]!r} -> {worst[2]!r}")

    if args.gpu_vs_cpu and args.kokoro and args.device == "igpu":
        gpu = next(e for label, e, _ in voices if label.startswith("kokoro"))
        cpu = engine(KokoroEngine, "kokoro", args.kokoro, "cpu")
        v = args.kokoro_voices.split(",")[0]
        bad = []
        for text in (SENTENCES * 3)[: args.gpu_vs_cpu]:
            a = np.frombuffer(gpu.synthesize(text, SynthesisOptions(voice=v)), dtype=np.int16).astype(np.float32)
            b = np.frombuffer(cpu.synthesize(text, SynthesisOptions(voice=v)), dtype=np.int16).astype(np.float32)
            n = min(len(a), len(b))
            corr = float(np.corrcoef(a[:n], b[:n])[0, 1]) if n else 0.0
            # energy above 6 kHz relative to the CPU result (the reported corruption is added high-frequency noise)
            def hf(x):
                spec = np.abs(np.fft.rfft(x[:n]))
                return spec[int(6000 / 12000 * len(spec)):].sum() / max(spec.sum(), 1e-9)
            ratio = hf(a) / max(hf(b), 1e-9)
            if corr < 0.98 or ratio > 1.5 or len(a) != len(b):
                bad.append((text, round(corr, 4), round(ratio, 2), len(a), len(b)))
        print(f"kokoro {v} GPU vs CPU: {args.gpu_vs_cpu - len(bad)}/{args.gpu_vs_cpu} sentences match"
              + (f"; differing: {bad}" if bad else ""))
        failed |= bool(bad)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
