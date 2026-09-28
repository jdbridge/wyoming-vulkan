"""The Wyoming Info that Home Assistant sees: one ASR program over the STT engines, one TTS program over all voices.

Built per `describe`, because library voices come and go (HA re-reads info every 30 s)."""

from wyoming.info import AsrModel, AsrProgram, Attribution, Info, TtsProgram, TtsVoice

from . import __version__
from .engines.base import Engine, SttEngine
from .voices import VoiceInfo

_PROJECT = Attribution(name="wyoming-vulkan (local build)", url="")
_DEFAULT_ATTRIBUTION = {
    "parakeet.cpp": Attribution(name="NVIDIA Parakeet via ggml-org/whisper.cpp", url="https://github.com/ggml-org/whisper.cpp"),
    "piper": Attribution(name="Piper", url="https://github.com/OHF-Voice/piper1-gpl"),
}


def _attribution(engine: Engine) -> Attribution:
    a = engine.config.attribution
    if a is not None:
        return Attribution(name=a.name, url=a.url)
    return _DEFAULT_ATTRIBUTION.get(engine.config.backend, _PROJECT)


def _description(engine: Engine) -> str:
    """"Engine [execution]", e.g. "Parakeet [Intel(R) Graphics (ADL-N)]" or "Piper [CPU fallback]". Kept short for UIs."""
    return f"{engine.config.description or engine.name} [{engine.runtime.label()}]"


def stt_program_name(stt: list[SttEngine]) -> str:
    """Default name of an endpoint's STT program; HA uses it as the STT entity's name.
    "Parakeet [GPU]", or "Parakeet + Whisper small [GPU]" when the engines run on the same device."""
    if not stt:
        return "wyoming-vulkan"
    labels = {e.runtime.label() for e in stt}
    if len(labels) == 1:
        return " + ".join(e.config.description or e.name for e in stt) + f" [{labels.pop()}]"
    return " + ".join(_description(e) for e in stt)


def build_info(stt: list[SttEngine], voices: list[VoiceInfo], stt_name: str | None = None) -> Info:
    info = Info()
    if stt:
        info.asr = [
            AsrProgram(
                name=stt_name or stt_program_name(stt),
                description="wyoming-vulkan speech-to-text",
                attribution=_PROJECT,
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=e.name,
                        description=_description(e),
                        attribution=_attribution(e),
                        installed=True,
                        version=None,
                        languages=e.languages,
                    )
                    for e in stt
                ],
            )
        ]
    if voices:
        info.tts = [
            TtsProgram(
                name="wyoming-vulkan-tts",
                description="wyoming-vulkan text-to-speech",
                attribution=_PROJECT,
                installed=True,
                version=__version__,
                voices=[
                    TtsVoice(
                        name=v.name,
                        description=v.description,  # "<voice> [<where it runs>]"; HA shows it as the voice's name
                        attribution=_DEFAULT_ATTRIBUTION.get(v.backend, _PROJECT),
                        installed=True,
                        version=None,
                        languages=v.languages,
                    )
                    for v in voices
                ],
                supports_synthesize_streaming=True,
            )
        ]
    return info
