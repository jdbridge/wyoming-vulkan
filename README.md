# wyoming-vulkan

Local speech-to-text and text-to-speech for [Home Assistant](https://www.home-assistant.io/)'s voice assistant, in one
container, on the GPU you already have: the integrated graphics of a mini PC or NAS box that usually sits idle.

- **Speech-to-text:** NVIDIA **Parakeet** TDT 0.6b v3 and OpenAI **Whisper**, both through
  [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (ggml, Vulkan backend).
- **Text-to-speech:** [**Piper**](https://github.com/OHF-Voice/piper1-gpl) voices through ONNX Runtime's WebGPU
  execution provider (Dawn on Vulkan), with streaming, and whole folders of voices offered to Home Assistant.
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

## Measurements (Intel i3-N305 iGPU)

Six English voice commands of 1.5–3.4 s (`tests/data/en`), time from the end of speech to the transcript:

| Speech-to-text | Per command | Exact | Notes |
|---|---|---|---|
| Parakeet v3 q8_0 | 0.45–0.74 s | 6/6 | 25 European languages |
| Whisper small, ~10 s window | 0.84–0.92 s | 6/6 | multilingual; German test set (`tests/data/de`) 4/4 |
| Whisper tiny.en, ~10 s window* | 0.13–0.18 s | 4/6 | |
| Whisper base.en, ~10 s window* | 0.25–0.29 s | 4/6 | 0.82–0.87 s with the full 30 s window |
| Whisper large-v3-turbo q5_0* | 4.4–4.5 s | 4/6 | too slow for commands on this GPU |

\* measured on an earlier set of the same six sentences, spoken by a different synthetic voice (Parakeet and Whisper
small got 5/6 on that set).

| Text-to-speech (Piper) | Result |
|---|---|
| high-quality voice | first audio after 0.6–0.7 s, real-time factor 0.39–0.45 (about 1.8× faster than the CPU) |
| medium-quality voice | real-time factor ~0.12 |
| a voice from a folder, first use | loads in ~2.5 s once |

Start-up (loading and warming up Parakeet, Whisper small and one voice): ~17 s. Speech-to-text and text-to-speech at
the same moment share the GPU (each ~1.5–2× slower). Without the GPU (CPU fallback on the same machine): Parakeet
0.45–0.8 s, Whisper small ~3.7 s, first audio ~1.3 s.

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
scripts/fetch-models.sh -d deploy/models parakeet-q8_0 whisper-small
scripts/fetch-models.sh -d deploy/voices piper-en_US-ljspeech-high
cp deploy/.env.example deploy/.env
```

Edit `deploy/.env`: at least `RENDER_GID` (`getent group render`), `RENDER_DEVICE` if the GPU is not
`/dev/dri/renderD128`, and `PUID`/`PGID` (a user that can read the model and voice folders). Then:

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
| Device | `WYOMING_DEVICE=auto` (GPU, else CPU with a warning), `igpu` (refuse to start without the GPU), `cpu` |
| Other models | `PARAKEET_MODEL`, `WHISPER_MODEL` (`scripts/fetch-models.sh --list`) |
| Languages | `PARAKEET_LANGUAGES` (default `en`, `auto` = all 25 of Parakeet v3); Whisper offers all of its languages |
| Whisper speed | `WHISPER_AUDIO_CTX=512` (~10 s window, ~3× faster; longer audio uses the full window automatically), `0` = always 30 s |
| Voices | the default voice in `VOICES_DIR`; every `<name>.onnx` + `<name>.onnx.json` in `VOICES_DIR`, `VOICE_LIBRARY_DIR` (subfolders too) and `EXTRA_VOICES_DIR` is offered to Home Assistant and loaded on first use (`MAX_LOADED_VOICES` at once). New files appear within ~30 s, no restart |
| More endpoints | add `[[endpoint]]` blocks to the inline config (and their ports). An endpoint may also list several engines: a request then goes to the first one that supports its language |

Voice folders on network shares are fine: they are optional (a missing or hung share only removes its voices), and a
voice is offered only once its files have stopped changing.

## Testing

```bash
tests/run.sh          # build, unit tests, then a real server on the GPU: both STT ports, TTS, voices, concurrency
tests/run.sh cpu      # the same without a GPU: everything must fall back to the CPU and still pass
tests/check_abi.sh    # the ctypes structs against the pinned C headers (after changing a binding or the pin)
```

`tests/run.sh` uses `deploy/.env` (or `.env.example`), renders the stack's inline config and tests exactly that.
Test audio: `tests/data/en` (public-domain LJ Speech voice) and `tests/data/de` (CC0 Thorsten voice).

## Other GPUs

The image restricts Vulkan to Mesa's Intel driver (`VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.json`), which
also hides `llvmpipe` (software Vulkan on the CPU). For another GPU, set `VK_DRIVER_FILES` and `VK_ICD_FILENAMES` in
the compose file to its driver file (e.g. `radeon_icd.json` for AMD) and `GPU_NAME_CONTAINS` / `GPU_VENDOR_ID` in
`.env` (AMD: `AMD` / `0x1002`). Reports of what works are welcome.

## Troubleshooting

| Log | Meaning |
|---|---|
| `!!! … FALLING BACK TO CPU` / `[CPU fallback]` in HA | the GPU path is not usable; the reason is in the same line |
| `no ggml GPU device contains 'Intel'` | no Vulkan GPU in the container: render node not mapped, wrong `RENDER_GID`, or no Mesa driver for this GPU |
| `ONNX Runtime session providers are ['CPUExecutionProvider'], not WebGPU` | ONNX Runtime found no Vulkan device (the silent CPU fallback it would otherwise do) |
| `Vulkan driver file … does not exist` | wrong `VK_DRIVER_FILES` path |
| `tts library … (optional): … not offered` | an optional voice folder is missing or not answering |

## Project layout

| Path | Contents |
|---|---|
| `wyoming_vulkan/` | the server (design: `DESIGN.md`) |
| `deploy/` | `compose.yaml`, `.env.example` |
| `config/config.example.toml` | every config option, commented |
| `scripts/` | `fetch-models.sh` (pinned downloads with sha256), `lock-requirements.sh` |
| `tests/` | `run.sh`, unit and integration tests, ABI check, test audio |
| `third_party/` | whisper.cpp v1.9.4 headers (MIT) for the ctypes bindings |

## Licence

GPL-3.0 (see `LICENSE`), matching piper-tts, which the image includes. Components: whisper.cpp and ggml (MIT), ONNX Runtime (MIT), the
wyoming library (MIT), piper-tts (GPL-3.0), Parakeet models (CC-BY-4.0, NVIDIA), Whisper models (MIT, OpenAI). Every
Piper voice has its own licence; none is included.
