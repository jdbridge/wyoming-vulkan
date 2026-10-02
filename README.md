# wyoming-vulkan

Local speech-to-text and text-to-speech for [Home Assistant](https://www.home-assistant.io/)'s voice assistant, in one
container, on the GPU you already have: the integrated graphics of a mini PC or NAS box that usually sits idle.

- **Speech-to-text:** NVIDIA **Parakeet** TDT 0.6b v3 and OpenAI **Whisper**, both through
  [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (ggml, Vulkan backend).
- **Text-to-speech:** [**Piper**](https://github.com/OHF-Voice/piper1-gpl), [**Kokoro-82M**](https://huggingface.co/hexgrad/Kokoro-82M)
  (42 voices, 7 languages) and [**KittenTTS**](https://github.com/KittenML/KittenTTS) nano (8 voices), all through
  ONNX Runtime's WebGPU execution provider (Dawn on Vulkan), with sentence streaming; [**Pocket TTS**](https://github.com/kyutai-labs/pocket-tts)
  (5 voices) on the CPU, streaming 80 ms frames; optionally [**CosyVoice3**](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
  (voice cloning from a short clip, 9 languages) through [cosyvoice.cpp](https://github.com/Lourdle/cosyvoice.cpp)
  on Vulkan, for GPUs stronger than a small iGPU. Every voice of every engine appears in one list in Home Assistant,
  e.g. `Kokoro af_heart [Intel(R) Graphics (ADL-N)]`, loaded on first use; whole folders of Piper voices are picked
  up automatically.
- **One stack, several choices in Home Assistant:** every port is its own speech-to-text entity (Parakeet on one,
  Whisper on the other; the voices are offered with Parakeet), and all of them share one process and one GPU.
- **Only Vulkan / Mesa:** no CUDA, no Intel compute runtime, no OpenVINO. If the GPU is missing, the server either
  refuses to start or falls back to the CPU loudly, and Home Assistant shows where each model runs:
  `Parakeet [Intel(R) Graphics (ADL-N)]`, `Whisper small [CPU fallback]`.
- Speaks the [Wyoming protocol](https://github.com/OHF-Voice/wyoming), so it works with Home Assistant's Wyoming
  integration and any other Wyoming client.

> Status: working and tested on one machine (Intel Core i3-N305, Alder Lake-N iGPU with 32 EUs). Other GPUs with a
> Mesa Vulkan driver (other Intel iGPUs from Gen9 on, Intel Arc, AMD Radeon iGPUs) should work but are untested; see
> [Other GPUs](#other-gpus).

## Measurements

Two GPUs in one machine: the Intel i3-N305's integrated GPU (Alder Lake-N, 32 EUs, single-channel DDR5) and an
NVIDIA RTX 4060 (8 GB), both through Vulkan, plus the N305's 8 cores for comparison. Measured 2026-09-29 on a
shared host (load average 3–5 from other containers). Speed-up = CPU time / GPU time (> 1: the GPU is faster).

**Speech-to-text**, six English voice commands of 1.5–3.4 s (`tests/data/en`), median time from the end of speech to
the transcript:

| Model | CPU (8 cores) | Intel iGPU | RTX 4060 | iGPU vs CPU | 4060 vs CPU |
|---|---|---|---|---|---|
| Parakeet v3 q8_0 | 0.66 s | 0.67 s | ~0.09 s | 1.0× | ~7× |
| Whisper small, ~10 s window | 3.66 s | 0.85 s | ~0.07 s | 4.3× | ~50× |
| Whisper large-v3-turbo q5_0, 30 s window | 42.6 s | 15.7 s | 0.17 s | 2.7× | ~250× |

Every model got 5–6 of 6 commands word for word. Whisper tiny.en / base.en with the ~10 s window: 0.13 / 0.25 s on
the iGPU. German (`tests/data/de`): Whisper small 4/4.

**Text-to-speech**, 10 sentences per voice, real-time factor (RTF = synthesis time / audio length; lower is faster),
each voice transcribed back by Parakeet (`tests/tts_roundtrip.py`):

| Model | CPU (8 cores) | Intel iGPU | RTX 4060 | iGPU vs CPU | 4060 vs CPU | Notes |
|---|---|---|---|---|---|---|
| KittenTTS nano (8 voices) | 0.14 | 0.14 | 0.027 | 1.0× | 5× | 0 % word error |
| Piper, high voice | 0.47 | 0.39 | 0.069 | 1.2× | 7× | first audio after the first sentence |
| Kokoro-82M fp32 (42 voices) | 0.56 | 0.76 | 0.17 | 0.7× | 3× | fp16 exports give NaN on WebGPU |
| Pocket TTS (5 voices) | **0.36** | 2.19 | 3.32 | 0.2× | 0.1× | streams 80 ms frames: first audio 0.07 s on the CPU; many tiny sequential steps suit the CPU |
| CosyVoice3 0.5B Q8_0 | ~72 | ~10 | **0.17–0.19** | ~7× | ~400× | voice cloning; streaming on the 4060: first audio 0.35 s at RTF ~0.4 |

What this means: small models (Kitten, Piper, Kokoro, Parakeet) are limited by per-operation overhead, so a small
iGPU barely beats 8 CPU cores and a big GPU gives 3–7×; large transformers (Whisper large, CosyVoice3) are limited by
compute, where the RTX 4060 (with matrix cores) is 50–400× faster than the CPU and the only one that runs CosyVoice3
in real time. On the iGPU the value is mostly that it takes the work off the CPU.

A folder voice or a voice pack loads on first use (Piper voice ~2.5 s, Kokoro ~3–7 s, Kitten ~2.6 s, Pocket ~3.6 s,
CosyVoice3 ~16 s on the 4060). Start-up (loading and warming up Parakeet, Whisper small and one voice): ~17 s.
Speech-to-text and text-to-speech at the same moment share the GPU (each ~1.5–2× slower).

## Requirements

- Linux with Docker and a recent Compose v2 (tested with 2.40; the stack uses an inline `configs:` `content:`) and a GPU with a Mesa Vulkan driver, reachable as a
  render node (`/dev/dri/renderD*`). The GPU can reach the host in any way: bare metal, a GPU passed through to a VM,
  an SR-IOV virtual function, or `/dev/dri` bound into an LXC container.
- x86-64. Disk: ~2 GB for the image, plus models (Parakeet q8_0 669 MB, Whisper small 488 MB, a Piper voice 60–115 MB).
- RAM: 2.3 GB measured with Parakeet, Whisper small and two loaded voices (the GPU memory of an iGPU is system RAM).

## Quick start

```bash
git clone https://github.com/jdbridge/wyoming-vulkan.git && cd wyoming-vulkan
docker build -t wyoming-vulkan:latest .
scripts/fetch-models.sh -d deploy/data/models   # Parakeet, Whisper small, a Piper voice, Kokoro, KittenTTS
cp deploy/.env.example deploy/.env
```

Edit `deploy/.env`: at least `RENDER_GID` (`getent group render`), `RENDER_DEVICE` if the GPU is not
`/dev/dri/renderD128`, and `PUID`/`PGID` (a user that can read the data folder). `DATA_DIR` (default
`deploy/data`) holds `models/<engine>/` and `voices/` (your own Piper voices); it can be a network share. Then:

```bash
cd deploy && docker compose up -d && docker compose logs -f
```

Wait for `Ready on tcp://0.0.0.0:10310, tcp://0.0.0.0:10311`; the log lists where each model runs. For a Dockge
stack, copy `compose.yaml` and `.env` into the stack folder and use absolute paths in `.env`.

**Home Assistant:** Settings → Devices & services → Add integration → **Wyoming Protocol**, host = the Docker host,
port `10310` (Parakeet and the voices); add it again with port `10311` (Whisper). Then Settings → Voice assistants: pick the
speech-to-text entity and a voice. Wyoming has no authentication: keep the ports on a trusted network.

## Configuration

Everyday settings are in `deploy/.env` (documented in `.env.example`): ports, device mode, models, default voice,
voice folders. The structure (which engines, endpoints and voice folders exist) is the server config written inline
in `deploy/compose.yaml`. `config/config.example.toml` documents every option; it is also the image's built-in
config when it runs without Compose.

| Topic | How |
|---|---|
| Device | `WYOMING_DEVICE=auto` (GPU, else CPU with a warning), `gpu` (refuse to start without the GPU), `cpu` |
| Which GPU | `PARAKEET_GPU`, `WHISPER_GPU`, `PIPER_GPU` (default voice and voice folders), `KOKORO_GPU`, `KITTEN_GPU`, `POCKET_GPU`, `COSYVOICE_GPU`: `intel`, `nvidia`, `amd` or part of the GPU's Vulkan name; empty = `GPU_NAME_CONTAINS`. See [Several GPUs](#several-gpus) |
| Warm models | `MAX_WARM_MODELS` (default 1): text-to-speech models kept loaded, each in its own process. Using another one stops the least recently used first (its memory is freed), then starts the new one (~2–7 s, CosyVoice3 ~15 s). Set it to the number of models Home Assistant uses regularly |
| Other models | `PARAKEET_MODEL`, `WHISPER_MODEL` in `DATA_DIR/models/parakeet` and `…/whisper` (`scripts/fetch-models.sh --list`) |
| Languages | `PARAKEET_LANGUAGES` (default `en`, `auto` = all 25 of Parakeet v3); Whisper offers all of its languages |
| Whisper speed | `WHISPER_AUDIO_CTX=512` (~10 s window, ~3× faster; longer audio uses the full window automatically), `0` = always 30 s |
| Voices | the default Piper voice `DEFAULT_VOICE` in `DATA_DIR/models/piper`; every `<name>.onnx` + `<name>.onnx.json` there, in `DATA_DIR/voices` (subfolders too) and in `EXTRA_VOICES_DIR` is offered to Home Assistant, plus the Kokoro, Kitten and Pocket voices (`[[tts_pack]]` in the inline config). All load on first use; new files appear within ~30 s, no restart |
| Speaking speed and style | `PIPER_LENGTH_SCALE` (speed: > 1 slower), `PIPER_NOISE_SCALE` (expressiveness), `PIPER_NOISE_W` (rhythm) for every voice; empty = each voice's own value from its `.onnx.json`. Per voice: `[voice_settings."<voice>"]` in the inline config |
| Own Kokoro model | `KOKORO_CUSTOM_ENABLED=true` and `KOKORO_CUSTOM_DIR` (a folder in `DATA_DIR/models` laid out like `models/kokoro`: fp32 `model.onnx`, `tokenizer.json`, `voices/<id>.bin`), e.g. a fine-tune. Voice ids start with the language letter (`a` = American English) |
| CosyVoice3 | `scripts/fetch-models.sh -d <models> cosyvoice`, then `COSYVOICE_ENABLED=true`. Voices: a clip of 5–15 s as `models/cosyvoice/voices/<name>.wav` plus its exact transcript as `<name>.txt` (encoded once at load), or an encoded `<name>.gguf`; every voice speaks all 9 languages. Only for voices you have the right to clone |
| More endpoints | add `[[endpoint]]` blocks to the inline config (and their ports). An endpoint may also list several engines: a request then goes to the first one that supports its language |

Voice folders on network shares are fine: they are optional (a missing or hung share only removes its voices), and a
voice is offered only once its files have stopped changing.

## Diagnostics page

`http://<docker host>:10312/` (`DIAG_PORT`; `web_port = 0` in the inline config turns it off). No authentication:
keep it on a trusted network, like the Wyoming ports.

- **Overview:** version, uptime, process and container memory; the Wyoming endpoints; every speech-to-text engine
  and text-to-speech model with where it runs, whether it is loaded and roughly how much memory loading it took;
  the voices per model; a form that speaks any text with any voice in the browser (with first-audio time and RTF).
- **Benchmark** (`/bench`): runs a text through every selected engine, model and voice with a separate, freshly
  loaded copy of each model: cold time to the first sentence (load + first sentence), warm time to the first
  sentence, total time, real-time factor, a word error rate (the audio transcribed back), and for the STT engines
  the same on the text spoken by the default voice. Sortable table, CSV download.
- JSON API: `GET /api/status`, `POST /api/synthesize` (`{"text", "voice"}` → WAV), `POST /api/bench`, `GET /api/bench`.

## Testing

```bash
tests/run.sh          # build, unit tests, then a real server on the GPU: both STT ports, TTS, voices, concurrency
tests/run.sh cpu      # the same without a GPU: everything must fall back to the CPU and still pass
tests/check_abi.sh    # the ctypes structs against the pinned C headers (after changing a binding or the pin)
python tests/tts_roundtrip.py --help   # (in the image) every voice speaks, Parakeet transcribes, word error rate
```

`tests/run.sh` uses `deploy/.env` (or `.env.example`), renders the stack's inline config and tests exactly that.
Test audio: `tests/data/en` (public-domain LJ Speech voice) and `tests/data/de` (CC0 Thorsten voice).

## Other GPUs

The image lists Mesa's Intel driver and NVIDIA's (`VK_DRIVER_FILES`), which also hides `llvmpipe` (software Vulkan
on the CPU); a listed file that is not in the container is skipped. For an AMD GPU, set `VK_DRIVER_FILES` and
`VK_ICD_FILENAMES` in the compose file to its driver file (`/usr/share/vulkan/icd.d/radeon_icd.json`) and
`GPU_NAME_CONTAINS` / `GPU_VENDOR_ID` in `.env` (`AMD` / `0x1002`). Reports of what works are welcome.

**NVIDIA** works through its Vulkan driver (tested: RTX 4060, driver 580, NVIDIA Container Toolkit 1.18): set
`DOCKER_RUNTIME=nvidia` and `NVIDIA_VISIBLE_DEVICES` (`all`, an index or the GPU's UUID from `nvidia-smi -L`) in
`.env`. The compose file adds the `graphics` capability, which brings NVIDIA's Vulkan driver file into the container
(`--gpus all` alone does not). If the NVIDIA runtime cannot start (driver update, toolkit problem), the whole
container does not start, including the engines on another GPU.

### Several GPUs

Each engine can use a different GPU, or the CPU: `device = "gpu" | "cpu" | "auto"` and `gpu = "intel" | "nvidia" |
"amd" | "<part of the name>"` per engine in the inline config (`*_GPU` in `.env`). For example: Parakeet, Whisper
small and the Piper voices on the iGPU (always there), CosyVoice3 on the NVIDIA card. The log lists the Vulkan GPUs
and where each engine runs, and Home Assistant shows it: `CosyVoice zh_female [NVIDIA GeForce RTX 4060]`.

- Every engine can use any GPU: speech-to-text runs in the server, every text-to-speech model in its own worker
  process (ONNX Runtime's WebGPU plug-in can use only one GPU per process, so this is what lets Piper, Kokoro, Kitten
  and Pocket models sit on different GPUs).
- The WebGPU plug-in (0.4.0) picks its GPU only by power preference (integrated vs discrete), so the ONNX models cannot
  choose between two discrete GPUs.
- A voice pack with `device = "gpu"` whose GPU is missing is not offered (warning in the log).

## Troubleshooting

| Log | Meaning |
|---|---|
| `!!! … FALLING BACK TO CPU` / `[CPU fallback]` in HA | the GPU path is not usable; the reason is in the same line |
| `no ggml GPU device contains 'Intel'` | no Vulkan GPU in the container: render node not mapped, wrong `RENDER_GID`, or no Mesa driver for this GPU |
| `ONNX Runtime session providers are ['CPUExecutionProvider'], not WebGPU` | ONNX Runtime found no Vulkan device (the silent CPU fallback it would otherwise do) |
| `no Vulkan driver file of … exists` | wrong `VK_DRIVER_FILES` path (a missing NVIDIA file alone is fine: it is only there with the NVIDIA runtime) |
| `tts library … (optional): … not offered` | an optional voice folder is missing or not answering |

## Project layout

| Path | Contents |
|---|---|
| `wyoming_vulkan/` | the server (design: `DESIGN.md`) |
| `deploy/` | `compose.yaml`, `.env.example` |
| `config/config.example.toml` | every config option, commented |
| `scripts/` | `fetch-models.sh` (pinned downloads with sha256), `lock-requirements.sh` |
| `tests/` | `run.sh`, unit and integration tests, ABI check, test audio |
| `third_party/` | whisper.cpp v1.9.4 headers (MIT) for the ctypes bindings; the Pocket TTS ONNX runtime (CC-BY-4.0) |

## Ideas

- **Pronunciation list:** Piper never sees letters; espeak-ng turns the text into phonemes first, so a mispronounced
  name cannot be fixed by training a voice. A small, case-insensitive whole-word replacement table (a file mounted
  into the container, applied to the text before Piper) would fix such words for every voice, e.g. respelling a
  brand name or "Bichon Frise" as "Beeshon Freezay", and could also normalise symbols an LLM emits ("21°C", "7:30",
  "%", "&").

## Licence

GPL-3.0 (see `LICENSE`), matching piper-tts, which the image includes. Components: whisper.cpp and ggml (MIT), ONNX
Runtime (MIT), the wyoming library (MIT), piper-tts (GPL-3.0), kokoro-onnx (MIT), phonemizer and espeak-ng (GPL-3.0),
cosyvoice.cpp (MIT, built into the image with its own ggml), pocket-tts-onnx runtime (CC-BY-4.0), Parakeet models
(CC-BY-4.0, NVIDIA), Whisper models (MIT, OpenAI), Kokoro-82M, KittenTTS and CosyVoice3 models (Apache-2.0), Pocket
TTS model (CC-BY-4.0, Kyutai). Every Piper voice has its own licence; the image contains no models or voices.
