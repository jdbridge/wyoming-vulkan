"""KittenTTS (KittenML, Apache-2.0) through ONNX Runtime: WebGPU EP (Dawn -> Vulkan) or the CPU EP.

One model, several voices (a TtsPack). The text front end follows KittenTTS's reference code (kittentts/onnx_model.py,
v0.8): espeak-ng (en-us, punctuation and stress kept) via phonemizer, words and punctuation separated by spaces, its
symbol table, tokens framed as [0, ..., 10, 0], the style row chosen by the text length, the model's per-voice
speed_priors (config.json) multiplied into the speed, the last 5000 samples cut.
Files (KittenML/kitten-tts-nano-0.8-fp32; scripts/fetch-models.sh kitten):
  <dir>/kitten_tts_nano_v0_8.onnx, <dir>/voices.npz, <dir>/config.json (voice names such as "Bella")
  On an Intel N305 iGPU nano ran at RTF 0.16 (CPU 0.29); mini (80M) and micro (40M) were 5-10x slower.
Options: voices (npz, default <model dir>/voices.npz), length_scale. English only.
"""

import json
import logging
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .base import ESPEAK_LOCK, SynthesisOptions, TtsPack, to_pcm16
from .ort_session import make_session

_LOGGER = logging.getLogger(__name__)
# phonemizer warns "words count mismatch" for nearly every sentence with punctuation; harmless for synthesis
logging.getLogger("phonemizer").setLevel(logging.ERROR)

SAMPLE_RATE = 24000
TRIM_SAMPLES = 5000  # the reference code drops the model's trailing samples

# KittenTTS's symbol table (TextCleaner in kittentts/onnx_model.py)
_PAD = "$"
_PUNCTUATION = ';:,.!?¡¿—…"«»"" '
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_LETTERS_IPA = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
_SYMBOLS = [_PAD, *_PUNCTUATION, *_LETTERS, *_LETTERS_IPA]
SYMBOL_IDS = {s: i for i, s in enumerate(_SYMBOLS)}  # later duplicates win, as in the reference dict


class KittenEngine(TtsPack):
    session = None

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def _voices_file(self) -> Path:
        return Path(self.config.options.get("voices", self.config.model.parent / "voices.npz"))

    def _model_config(self) -> dict:
        cfg = self.config.model.parent / "config.json"
        return json.loads(cfg.read_text(encoding="utf-8")) if cfg.exists() else {}

    def _aliases(self) -> dict[str, str]:
        """Friendly name -> voice key in voices.npz (from config.json), e.g. "Bella" -> "expr-voice-2-f"."""
        return dict(self._model_config().get("voice_aliases", {}))

    def list_voices(self) -> list[tuple[str, list[str]]]:
        aliases = self._aliases()
        if aliases:
            return [(name, ["en_US"]) for name in aliases]
        with np.load(self._voices_file()) as npz:
            return [(key, ["en_US"]) for key in npz.files]

    def _load(self, device: str) -> None:
        import phonemizer

        from kokoro_onnx.tokenizer import Tokenizer  # sets up espeak-ng for phonemizer (espeakng-loader)

        with ESPEAK_LOCK:
            Tokenizer()
            self.phonemizer = phonemizer.backend.EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
        with np.load(self._voices_file()) as npz:
            self.styles = {key: npz[key] for key in npz.files}
        self.alias_map = self._aliases()
        self.speed_priors = dict(self._model_config().get("speed_priors", {}))
        self.session = make_session(self.config.model, device, self.gpu, self.runtime)
        self.lock = threading.Lock()
        self.runtime.actual = device
        _LOGGER.info("tts pack %s: loaded %s on %s, %s", self.name, self.config.model.name, self.runtime.summary(), self.runtime.detail)

    def synthesize(self, text: str, options: SynthesisOptions) -> bytes:
        assert self.session is not None, "engine not loaded"
        voice = options.voice or self.list_voices()[0][0]
        key = self.alias_map.get(voice, voice)
        with ESPEAK_LOCK:
            phonemes = self.phonemizer.phonemize([text])[0]
        spaced = " ".join(re.findall(r"\w+|[^\w\s]", phonemes))
        ids = [0, *(SYMBOL_IDS[c] for c in spaced if c in SYMBOL_IDS), 10, 0]
        table = self.styles[key]
        ref = min(len(text), table.shape[0] - 1)
        feed = {"input_ids": np.array([ids], dtype=np.int64), "style": table[ref : ref + 1],
                "speed": np.array([self.speed(options) * self.speed_priors.get(key, 1.0)], dtype=np.float32)}
        with self.lock:
            audio = self.session.run(None, feed)[0]
        audio = np.asarray(audio).reshape(-1)
        return to_pcm16(audio[:-TRIM_SAMPLES] if audio.size > TRIM_SAMPLES else audio)

    def warm_up(self) -> None:
        voices = self.list_voices()
        if voices:
            self.synthesize("Warming up.", SynthesisOptions(voice=voices[0][0]))

    def close(self) -> None:
        lock: Optional[threading.Lock] = getattr(self, "lock", None)
        if lock is None:
            self.session = None
            return
        with lock:
            self.session = None
