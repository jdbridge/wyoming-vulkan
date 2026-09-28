"""Piper text-to-speech through ONNX Runtime: WebGPU EP (Dawn -> Vulkan) on the iGPU, or the CPU EP.

Backend options in the config: config (voice .onnx.json, default <model>.json) and the speech settings
  length_scale  speaking speed: > 1 slower, < 1 faster
  noise_scale   variation / expressiveness of the audio
  noise_w       variation of phoneme durations, i.e. rhythm (also accepted as noise_w_scale, piper-tts's name)
A setting that is not configured (missing or empty) uses the voice's own value from the "inference" section of its
.onnx.json. Per-voice values come from [voice_settings."<voice name>"] in the config (merged into these options).
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from piper.config import PiperConfig, SynthesisConfig
from piper.voice import PiperVoice

from .base import ESPEAK_LOCK, SynthesisOptions, TtsEngine, to_pcm16
from .ort_session import make_session

_LOGGER = logging.getLogger(__name__)


def speech_settings(options: dict) -> dict[str, Optional[float]]:
    """The configured speech settings (None = the voice's own value); empty strings count as not configured."""

    def number(*keys: str) -> Optional[float]:
        for key in keys:
            value = options.get(key)
            if value not in (None, ""):
                return float(value)
        return None

    return {
        "length_scale": number("length_scale"),
        "noise_scale": number("noise_scale"),
        "noise_w_scale": number("noise_w", "noise_w_scale"),
    }


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

        session = make_session(model, device, self.gpu, self.runtime)
        self.voice = PiperVoice(session=session, config=piper_config)
        self.lock = threading.Lock()
        self.runtime.actual = device
        self.settings = speech_settings(self.config.options)
        own = {"length_scale": piper_config.length_scale, "noise_scale": piper_config.noise_scale,
               "noise_w_scale": piper_config.noise_w_scale}
        effective = ", ".join(
            f"{k.replace('_scale', '') if k == 'noise_w_scale' else k} {v if v is not None else own[k]:g}"
            f" ({'config' if v is not None else 'voice'})" for k, v in self.settings.items()
        )
        _LOGGER.info(
            "tts %s: loaded %s (%d Hz) on %s, %s; %s",
            self.name, model.name, piper_config.sample_rate, self.runtime.summary(), self.runtime.detail, effective,
        )

    def synthesize(self, text: str, options: SynthesisOptions) -> bytes:
        voice = self.voice
        assert voice is not None, "engine not loaded"
        syn = SynthesisConfig(**self.settings)  # None: piper uses the voice's own value
        if options.speaker is not None:
            syn.speaker_id = voice.config.speaker_id_map.get(options.speaker)
            if syn.speaker_id is None and options.speaker.isdigit():
                syn.speaker_id = int(options.speaker)
        with ESPEAK_LOCK:  # phonemise under the shared espeak lock, synthesise outside it
            sentences = [p for p in voice.phonemize(text) if p]
        parts = []
        with self.lock:
            for phonemes in sentences:
                audio = voice.phoneme_ids_to_audio(voice.phonemes_to_ids(phonemes), syn)
                if isinstance(audio, tuple):  # (audio, alignments) in some piper-tts versions
                    audio = audio[0]
                parts.append(to_pcm16(audio))  # as piper-tts's synthesize(): normalise, clip, int16
        return b"".join(parts)

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
