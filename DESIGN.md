# Design

How the server is built and why. For setup and use, see `README.md`.

## 1. Architecture

```
Home Assistant                       Wyoming over TCP (JSON header line + optional JSON data + optional binary payload)
     │  one integration per port
     ▼
┌──────────────────────────── container ────────────────────────────────────────┐
│ asyncio Wyoming servers, one per endpoint (wyoming Python library)           │
│   ├─ describe → info {asr: this endpoint's models, tts: voices, streaming}    │
│   ├─ STT: transcribe, audio-start/chunk/stop → transcript                     │
│   │     ├─ Parakeet: ctypes → libparakeet.so ─┐                               │
│   │     └─ Whisper:  ctypes → libwhisper.so ──┴─ ggml (Vulkan backend) ─┐     │
│   └─ TTS: synthesize | synthesize-start/chunk/stop                        │     │
│         → audio-start/chunk/stop (+ synthesize-stopped)                   │     │
│         └─ Piper (piper-tts) → ONNX Runtime → WebGPU EP → Dawn ─┐         │     │
│                                                  Vulkan loader ◄┴─────────┘     │
│                                                  Mesa driver (one ICD file only) │
└─────────────────────────────────────────────────────────── /dev/dri/renderD128 ─┘
                                                   kernel driver (i915 / xe) → GPU
```

All engines meet at **Vulkan / Mesa**: no vendor compute runtime (no Intel compute runtime, no OpenVINO, no CUDA). ONNX Runtime's OpenVINO execution provider was measured and rejected for Piper: OpenVINO's GPU plugin compiles a kernel set per input shape, and Piper's input length changes with every sentence, so every new sentence length cost 20–65 s. The WebGPU EP runs operator by operator with ready-made GPU programs that take sizes at run time.

How the GPU reaches the container does not matter: bare metal, a whole GPU passed through to a VM, an SR-IOV virtual function, or `/dev/dri` bound into an LXC container all work, as long as the container gets a render node and a working Mesa Vulkan driver.

## 2. Container

**Base:** the whisper.cpp **v1.9.4** release image with Vulkan, pinned by digest (`Dockerfile`; Ubuntu 24.04.4, Mesa 25.2.8). It contains Mesa's Vulkan drivers, `libvulkan`, `libparakeet.so` and `libwhisper.so` in `/app/build/bin` (added to `ldconfig`; newer whisper.cpp images use `/usr/local/lib`). Python 3.12 is in the base.

**Added**, all pinned: `python3-venv` and `vulkan-tools` (exact versions from a dated Ubuntu snapshot), a venv from `requirements.txt` with hashes (`onnxruntime` 1.30.0, `onnxruntime-ep-webgpu` 0.4.0, `piper-tts` 1.8.0, `wyoming` 1.10.2, `sentence-stream` 1.3.0, `numpy` 2.5.3; regenerate with `scripts/lock-requirements.sh`). The ctypes bindings follow the headers of the same whisper.cpp commit (`third_party/`); `tests/check_abi.sh` compiles them with gcc and compares every struct size and field offset with the ctypes definitions.

**Runtime settings:**

| Setting | Why |
|---|---|
| the GPU's render node as `/dev/dri/renderD128`, plus the host's `render` group | access to the GPU; with several GPUs, map it by PCI address (`/dev/dri/by-path/pci-…-render`) so a renumbering cannot hand over another card |
| `VK_DRIVER_FILES` / `VK_ICD_FILENAMES` = `/usr/share/vulkan/icd.d/intel_icd.json` (set in the image) | hides `llvmpipe` (software Vulkan on the CPU) and the other drivers. A wrong path means "no Vulkan device" and ONNX Runtime silently runs on the CPU, which the server detects (§5) |
| models and voices mounted read-only | the image contains no models |
| non-root user | only needs to read the model and voice folders |
| healthcheck | a Wyoming `describe` → `info` round trip on every endpoint (`python -m wyoming_vulkan.healthcheck`) |

## 3. Wyoming server

### 3.1 Protocol

Wire format, one event: a JSON line `{"type": ..., "data_length": N, "payload_length": M}` + `\n`, then N bytes of JSON data, then M bytes of payload (audio). The `wyoming` library does all of it: framing, typed events, the TCP server with one `AsyncEventHandler` per connection, `AudioChunkConverter`.

- **`Info`** has independent lists `asr`, `tts`, `handle`, `intent`, `wake`, …; one server may fill both `asr` and `tts`. Programs and models must be `installed=True`.
- **STT sequence:** `transcribe {name?, language?}` (optional) → `audio-start {rate,width,channels}` → `audio-chunk` × n → `audio-stop` → `transcript {text, language}`; the connection then closes (as in wyoming-faster-whisper). Latency that matters = `audio-stop` → `transcript`.
- **TTS, one-shot:** `synthesize {text, voice{name, speaker}}` → `audio-start` → `audio-chunk` × n → `audio-stop` (sentences split with `sentence-stream`, one start/stop around all of them).
- **TTS, streaming** (`TtsProgram.supports_synthesize_streaming = true`): `synthesize-start` → `synthesize-chunk {text}` × n → (`synthesize`, a compatibility copy of the whole text, ignored inside a stream) → `synthesize-stop`; each completed sentence is answered with its own `audio-start` → `audio-chunk` × n → `audio-stop`, and the stream ends with `synthesize-stopped` (as in wyoming-piper 2.5).

### 3.2 Home Assistant behaviour (checked in `home-assistant/core`)

- One Wyoming config entry gives one STT entity (named after the first ASR program) and one TTS entity.
- **STT:** HA sends only `Transcribe(language=…)`, never a model name, and cannot choose between models of one server. Every speech-to-text choice that should be selectable in HA therefore gets its own endpoint (port), see §3.4.
- **TTS:** HA builds its voice list per language from the installed voices in `info` and shows each voice's **description** as its name; the chosen voice's `name` is sent with every request. Newer HA versions re-read `info` every 30 s, so new voices appear without re-adding the integration.
- With `supports_synthesize_streaming`, HA plays audio as it arrives (WAV header first, then each `audio-chunk`).

### 3.3 Handler

- **Audio in:** everything is converted to 16 kHz, 16-bit mono (`AudioChunkConverter`), then to float32 for the engines; input beyond `max_audio_seconds` is dropped.
- **Audio out:** int16 at the voice's sample rate, split into `audio-chunk` events of `samples_per_chunk` (1024).
- **Concurrency:** inference runs in worker threads (`asyncio.to_thread`); ctypes and ONNX Runtime release the GIL. One lock per engine (a whisper.cpp context is not thread-safe). STT and TTS may run at the same time; on a small iGPU they time-slice (each ~1.5–2× slower), within one voice command they do not overlap.
- Errors are reported to the client as an `error` event; a client that already left is ignored.

### 3.4 Modules, configuration, endpoints

```
wyoming_vulkan/
  __main__.py        read config -> load engines -> verify devices -> warm up -> one server per endpoint
  config.py          TOML config (stdlib tomllib), validation
  handler.py         protocol only; per endpoint: its STT engines (routing) and optionally the voices
  info.py            the Wyoming Info per endpoint (what HA sees)
  voices.py          voice registry: fixed voices + voice folders, loaded on first use
  devices.py         Vulkan driver-file check, environment report
  healthcheck.py     describe -> info on every endpoint
  engines/
    base.py          SttEngine / TtsEngine, RuntimeReport (requested vs actual device), CPU fallback
    ggml.py          ggml log capture and device list (shared by the whisper.cpp-family backends)
    parakeet_cpp.py  libparakeet via ctypes     device: igpu (Vulkan) | cpu
    whisper_cpp.py   libwhisper via ctypes      device: igpu (Vulkan) | cpu
    piper_ort.py     Piper via ONNX Runtime      device: igpu (WebGPU EP) | cpu
```

- **Engines** are found by backend name in a registry and imported lazily. Interface: `load()`, `warm_up()`, `runtime`, `languages`, and `transcribe(float32 16 kHz mono, language) -> str` or `synthesize(sentence, options) -> int16 PCM`.
- **Device per engine:** `igpu` (refuse to start unless the GPU path is really used), `cpu`, or `auto` (the GPU, else the CPU with a loud warning). Where each engine really runs is always in `info`: model and voice descriptions read `<description> [<GPU name> | CPU | CPU fallback]`, e.g. `Parakeet [Intel(R) Graphics (ADL-N)]`.
- **Endpoints:** `[server]` is the main endpoint, `[[endpoint]]` adds more; each lists its STT engines (`stt = [...]`, in routing order) and whether it offers the voices (`tts`). All endpoints share the loaded engines. The STT program name (HA's entity name) defaults to `<descriptions> [<device>]`.
- **Routing on an endpoint** with several engines: a `transcribe` name that matches an engine wins; else the first engine whose `languages` contain the request's language (region codes match the base language, `de-DE` → `de`); else the first engine. The shipped stack keeps it simple: one engine per port (Parakeet with the voices on 10310, Whisper on 10311).
- **Languages:** a list, a comma-separated string, or `"auto"`: Parakeet v3's 25 languages, every Whisper language (`en` only for `.en` models).
- **Models are data:** the config points at files; `scripts/fetch-models.sh` downloads known-good revisions and checks their hashes, but any compatible file can be configured.
- The complete, commented option list is `config/config.example.toml`. The Compose deployment keeps this TOML inline in `deploy/compose.yaml` (`configs:` → `content:`) with the host-specific values from `.env`; `tests/run.sh` renders exactly that config and tests with it.

### 3.5 Voices (`voices.py`)

- **Fixed voices** (`[[tts]]`) are loaded and warmed up at start; the first is the default voice.
- **Voice folders** (`[[tts_library]]`, `path` = a folder or a list; `recursive`, `optional`, `min_age_seconds`): every `<name>.onnx` with a `<name>.onnx.json` is offered. `describe` rescans the folders at most every 5 s. A voice loads on first use (in a worker thread, 90 s timeout), is reloaded when its files change, and at most `max_loaded_voices` folder voices stay loaded (least recently used is unloaded; never one that is in use). Unknown or broken voices fall back to the default voice; a broken voice stays hidden until its files change.
- **Half-written files:** both files must exist, be non-empty and older than `min_age_seconds`; a voice that appears or changes while the server runs is offered only after its mtime and both sizes stayed the same for 10 s (a copy over SMB can keep the source's old mtime).
- **Network folders:** a folder on a `hard` NFS mount can hang instead of failing. Each folder is scanned in its own worker thread with a 5 s timeout, and a folder whose previous scan is still stuck is skipped immediately, so a hung share never blocks the event loop or speech-to-text. `optional = true` turns a missing folder into a warning.
- Language of a folder voice: the `ll_CC-` prefix of the file name, else `language.code` or the espeak voice in its config, unless the folder sets `languages`. Its `info` label before it is loaded is predicted from the loaded voices on the same kind of device.

## 4. Engines

### 4.1 Parakeet (`libparakeet`)

- whisper.cpp's C API for NVIDIA Parakeet TDT: `parakeet_init_from_file_with_params`, `parakeet_full`, `parakeet_full_n_segments`, `parakeet_full_get_segment_text`, `parakeet_log_set`. Models: `ggml-org/parakeet-GGUF` (v3: 25 European languages; it detects the language itself).
- GPU proof: ggml's device list (`ggml_backend_dev_*` in `libggml.so.0`) must contain a GPU device whose description contains `[gpu] name_contains` (the device *type* matters: a CPU's description may also say "Intel"), and after init the library's log must say `using Vulkan0 backend`.

### 4.2 Whisper (`libwhisper`)

- Same library family and GPU proof. `whisper_full_params` (304 bytes on x86-64) is mirrored field by field and checked by `tests/check_abi.sh`.
- **`audio_ctx`:** Whisper always encodes a 30 s window (1,500 frames of 20 ms), padding short commands with silence. `audio_ctx = 512` encodes only ~10 s, about 3× less encoder work (base.en: 0.82 → 0.25 s per command, same transcripts). For longer audio the server uses the full window automatically.
- **Repetition loops:** Whisper can repeat a sentence until it runs out of tokens (seen with large-v3-turbo and no timestamps: 40 s for a 4 s command). Timestamps stay on (fewer loops), each request has a time budget (`max_seconds`, default max(15 s, 3× the audio), enforced through whisper.cpp's `abort_callback`), and repeated sentences are collapsed.
- Language: the request's language if the model knows it, else the configured fallback, else auto-detection; `.en` models always `en`. Sound annotations such as `[BLANK_AUDIO]` are dropped.

### 4.3 Piper (ONNX Runtime WebGPU EP)

- The WebGPU plugin EP is registered once (`ort.register_execution_provider_library`), the device picked from `ort.get_ep_devices()` by `[gpu] vendor_id`, and the session built directly (`PiperVoice(session, config)`, which skips piper's own CPU session).
- GPU proof: `session.get_providers()[0] == "WebGpuExecutionProvider"`. Without a usable Vulkan device ONNX Runtime silently returns a CPU-only session; that case is caught here. The EP picks the physical GPU itself, so the Vulkan driver restriction (§2) is what guarantees the right one; the GPU name shown in `info` comes from the Vulkan device list.
- Small shape and cast nodes run on the CPU EP on purpose; expected.

### 4.4 CPU fallback

Parakeet and Whisper with `use_gpu = false`, Piper with the CPU EP. Everything keeps working, slower (see `README.md`).

## 5. Start-up

1. Log versions, the Vulkan driver restriction and the render nodes in the container.
2. Load each engine and prove its GPU path (§4); `igpu`: exit code 3 on failure, `auto`: warnings and `[CPU fallback]` in `info`.
3. Warm up: ggml-vulkan compiles some pipelines only when an input length first needs them (the first 1, 2 and 3 s clips each cost 0.4–3.6 s extra on an Intel iGPU, nothing new from 4 s), so each STT engine transcribes 1, 2, 3 and 5 s of silence; Piper speaks one sentence.
4. Scan the voice folders (a required folder that is unusable stops the start), then open all endpoints. The healthcheck passes from here.

## 6. Deployment (`deploy/`)

- `compose.yaml` + `.env` (template `.env.example`); works as a Dockge stack. The image is built locally (`pull_policy: never`).
- Render node by `RENDER_DEVICE` (long `devices` syntax, because by-path names contain colons), the host's render group by `RENDER_GID`, user by `PUID`/`PGID`.
- Optional voice folders as read-only binds with `create_host_path` (a missing folder never stops the container) and, for network mounts, `propagation: rslave` (a mount that appears after the container started still shows up inside it).
- Log rotation 3 × 10 MB.
