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
│         (each TTS model in its own worker process, ≤ max_warm_models warm)│     │
│         ├─ Piper / Kokoro / KittenTTS → ONNX Runtime → WebGPU EP → Dawn ┐ │     │
│         ├─ Pocket TTS → ONNX Runtime, CPU EP                             │ │     │
│         └─ CosyVoice3 → HTTP → cosyvoice-server (child process, own ggml) ┤ │     │
│                                                  Vulkan loader ◄┴─────────┘     │
│                                   Mesa (Intel) / NVIDIA drivers (listed ICD files) │
└──────────────────────────────────────── /dev/dri/renderD128 (+ NVIDIA runtime) ─┘
                                        kernel drivers (i915 / xe, nvidia) → GPUs
```

All engines meet at **Vulkan / Mesa**: no vendor compute runtime (no Intel compute runtime, no OpenVINO, no CUDA). ONNX Runtime's OpenVINO execution provider was measured and rejected for Piper: OpenVINO's GPU plugin compiles a kernel set per input shape, and Piper's input length changes with every sentence, so every new sentence length cost 20–65 s. The WebGPU EP runs operator by operator with ready-made GPU programs that take sizes at run time.

How the GPU reaches the container does not matter: bare metal, a whole GPU passed through to a VM, an SR-IOV virtual function, or `/dev/dri` bound into an LXC container all work, as long as the container gets a render node and a working Mesa Vulkan driver.

## 2. Container

**Base:** the whisper.cpp **v1.9.4** release image with Vulkan, pinned by digest (`Dockerfile`; Ubuntu 24.04.4, Mesa 25.2.8). It contains Mesa's Vulkan drivers, `libvulkan`, `libparakeet.so` and `libwhisper.so` in `/app/build/bin` (added to `ldconfig`; newer whisper.cpp images use `/usr/local/lib`). Python 3.12 is in the base.

**Added**, all pinned: `python3-venv` and `vulkan-tools` (exact versions from a dated Ubuntu snapshot), a venv from `requirements.txt` with hashes (`onnxruntime` 1.30.0, `onnxruntime-ep-webgpu` 0.4.0, `piper-tts` 1.8.0, `wyoming` 1.10.2, `sentence-stream` 1.3.0, `numpy` 2.5.3; regenerate with `scripts/lock-requirements.sh`). The ctypes bindings follow the headers of the same whisper.cpp commit (`third_party/`); `tests/check_abi.sh` compiles them with gcc and compares every struct size and field offset with the ctypes definitions.

**CosyVoice3 build stage:** cosyvoice.cpp (pinned commit) is compiled in a separate stage of the same base image, with its own ggml (the v0.23.0 tag, Vulkan backend), the ONNX Runtime 1.30.0 C library (sha256-checked) for its voice-prompt frontend, gcc 14 (C++20), API-only server. It lands in `/opt/cosyvoice` with its libraries next to it (RUNPATH `$ORIGIN`); the build fails if `ldd` would resolve anything to whisper.cpp's ggml. Two ggml builds in one process crash (same library names, different versions), which is why it runs as a separate process.

**Runtime settings:**

| Setting | Why |
|---|---|
| the GPU's render node as `/dev/dri/renderD128`, plus the host's `render` group | access to the GPU; with several GPUs, map it by PCI address (`/dev/dri/by-path/pci-…-render`) so a renumbering cannot hand over another card |
| `VK_DRIVER_FILES` / `VK_ICD_FILENAMES` = Intel's and NVIDIA's driver files (set in the image) | hides `llvmpipe` (software Vulkan on the CPU) and the other drivers; NVIDIA's file exists only with the NVIDIA runtime (§7). With no existing file there is no Vulkan device and ONNX Runtime silently runs on the CPU, which the server detects (§5) |
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
  voices.py          voice registry: fixed voices, voice folders, packs; at most max_warm_models warm
  worker.py          each TTS model in its own process (WorkerEngine in the server, `python -m wyoming_vulkan.worker`)
  devices.py         Vulkan driver-file check, environment report
  healthcheck.py     describe -> info on every endpoint
  web.py             diagnostics page (aiohttp, same event loop): status, sample speech, benchmark job
  engines/
    base.py          SttEngine / TtsEngine, RuntimeReport (requested vs actual device), CPU fallback
    ggml.py          ggml log capture and device list (shared by the whisper.cpp-family backends)
    parakeet_cpp.py  libparakeet via ctypes     device: igpu (Vulkan) | cpu
    whisper_cpp.py   libwhisper via ctypes      device: igpu (Vulkan) | cpu
    ort_session.py   ONNX Runtime session on the WebGPU EP or the CPU, with the GPU proof (shared)
    piper_ort.py     Piper via ONNX Runtime      device: igpu (WebGPU EP) | cpu
    kokoro_ort.py    Kokoro-82M voice pack       device: igpu (WebGPU EP) | cpu
    kitten_ort.py    KittenTTS voice pack        device: igpu (WebGPU EP) | cpu
```

- **Engines** are found by backend name in a registry and imported lazily. Interface: `load()`, `warm_up()`, `runtime`, `languages`, and `transcribe(float32 16 kHz mono, language) -> str` or `synthesize(sentence, options) -> int16 PCM`.
- **Device per engine:** `igpu` (refuse to start unless the GPU path is really used), `cpu`, or `auto` (the GPU, else the CPU with a loud warning). Where each engine really runs is always in `info`: model and voice descriptions read `<description> [<GPU name> | CPU | CPU fallback]`, e.g. `Parakeet [Intel(R) Graphics (ADL-N)]`.
- **Endpoints:** `[server]` is the main endpoint, `[[endpoint]]` adds more; each lists its STT engines (`stt = [...]`, in routing order) and whether it offers the voices (`tts`). All endpoints share the loaded engines. The STT program name (HA's entity name) defaults to `<descriptions> [<device>]`.
- **Routing on an endpoint** with several engines: a `transcribe` name that matches an engine wins; else the first engine whose `languages` contain the request's language (region codes match the base language, `de-DE` → `de`); else the first engine. The shipped stack keeps it simple: one engine per port (Parakeet with the voices on 10310, Whisper on 10311).
- **Languages:** a list, a comma-separated string, or `"auto"`: Parakeet v3's 25 languages, every Whisper language (`en` only for `.en` models).
- **Models are data:** the config points at files; `scripts/fetch-models.sh` downloads known-good revisions and checks their hashes, but any compatible file can be configured.
- The complete, commented option list is `config/config.example.toml`. The Compose deployment keeps this TOML inline in `deploy/compose.yaml` (`configs:` → `content:`) with the host-specific values from `.env`; `tests/run.sh` renders exactly that config and tests with it.

### 3.5 Voices (`voices.py`)

- **Names in HA:** every voice's description is `<Engine> <voice> [<where it runs>]` (`Piper ljspeech-high [...]`, `Kokoro af_heart [...]`, `Kitten Bella [...]`), so the engines stay grouped in HA's voice list. Piper voice ids are the file names; pack voice ids are `<pack>_<voice>`.
- **One process per model** (0.10.0, `worker.py`): every text-to-speech model (a Piper voice file, a pack) runs in its own worker process; speech-to-text stays in the server. `WorkerEngine` stands in for the engine: it starts `python -m wyoming_vulkan.worker`, which loads the real engine (GPU proof, warm-up) and then speaks sentence by sentence over stdin/stdout (one JSON line per message, PCM bytes after an audio header; frame streaming is passed through). Why: each process has its own GPU context, so every model can use any GPU (ONNX Runtime's WebGPU plug-in allows one GPU per process, §7); stopping a worker returns its memory (an iGPU's memory is the host's RAM); a crash in a runtime or driver takes down one worker, which the next request starts again. Measured cost: ~0.2 s and ~45 MB per worker, and ~0.6–1.2 s more for a cold start than in-process; warm speed is unchanged (Pocket's first frame even improved from 0.27 to 0.07 s without the other engines in its process). A worker exits when its stdin closes and, via `PR_SET_PDEATHSIG`, when the server dies.
- **Warm models:** every model loads on first use; at most `max_warm_models` stay warm. Before another one starts, the least recently used idle one is stopped (so its memory is free first); models in use are never stopped (the limit is exceeded for a moment instead). The first `max_warm_models` `[[tts]]` voices are started at start-up, proving their GPU (with `device = "gpu"` the server refuses to start without it); the first `[[tts]]` voice is the default (requests without a known voice), started again when needed. `info` shows where a model runs, or ran when it was last warm, or would run.
- **Voice packs** (`[[tts_pack]]`, a `TtsPack` engine): one model with many voices (Kokoro, KittenTTS, Pocket, CosyVoice3). At start only the voice list is read (small files, thread with timeout; unreadable files leave the pack out with a warning); a pack that fails to load disappears from `info` and its requests use the default voice. `use()` hands out a small handle binding the pack engine to one voice and that voice's speech settings.
- **Voice folders** (`[[tts_library]]`, `path` = a folder or a list; `recursive`, `optional`, `min_age_seconds`): every `<name>.onnx` with a `<name>.onnx.json` is offered. `describe` rescans the folders at most every 5 s. A voice loads on first use (90 s timeout) and is reloaded when its files change; it counts towards `max_warm_models` like any model. Unknown or broken voices fall back to the default voice; a broken voice stays hidden until its files change.
- **Half-written files:** both files must exist, be non-empty and older than `min_age_seconds`; a voice that appears or changes while the server runs is offered only after its mtime and both sizes stayed the same for 10 s (a copy over SMB can keep the source's old mtime).
- **Network folders:** a folder on a `hard` NFS mount can hang instead of failing. Each folder is scanned in its own worker thread with a 5 s timeout, and a folder whose previous scan is still stuck is skipped immediately, so a hung share never blocks the event loop or speech-to-text. `optional = true` turns a missing folder into a warning.
- **Speech settings** (`length_scale` speed, `noise_scale` expressiveness, `noise_w` rhythm): unset values use the voice's own `.onnx.json` ("inference"); `[voice_settings."*"]` sets them for every voice (the Compose stack fills it from `PIPER_*` in `.env`), a voice's `[[tts]]`/`[[tts_library]]` entry overrides that, and `[voice_settings."<voice>"]` overrides both. Empty values never erase a more specific one (`Config.speech_options`); the load log shows each effective value and its source.
- Language of a folder voice: the `ll_CC-` prefix of the file name, else `language.code` or the espeak voice in its config, unless the folder sets `languages`. Its `info` label before it is loaded is predicted from loaded models on the same GPU, else that GPU's Vulkan name.

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
- GPU proof: `session.get_providers()[0] == "WebGpuExecutionProvider"`. Without a usable Vulkan device ONNX Runtime silently returns a CPU-only session; that case is caught here. The EP picks the physical GPU itself (by power preference when there are several, §7); the GPU name shown in `info` comes from the Vulkan device list.
- Small shape and cast nodes run on the CPU EP on purpose; expected.

### 4.4 Kokoro-82M and KittenTTS (voice packs)

- **Every engine converts text itself** (the ONNX models take phoneme ids, not text): Piper through piper-tts; Kokoro through kokoro-onnx's tokenizer (espeak-ng via phonemizer, the standard front end of Kokoro's ONNX builds; the vocabulary comes from the model's `tokenizer.json`); KittenTTS exactly as its reference code does it (espeak en-us with punctuation and stress, words and punctuation space-separated, its symbol table, tokens `[0, …, 10, 0]`, style row by text length, per-voice `speed_priors` from `config.json`, the last 5000 samples cut). Only the ONNX session is ours, so all of them run on the WebGPU EP with the same GPU proof (`ort_session.make_session`).
- **espeak-ng keeps global state:** two threads phonemising at once corrupt each other's phonemes, so all engines hold one shared `ESPEAK_LOCK` while phonemising (never during inference). Piper therefore phonemises and synthesises in separate steps (same result, including piper-tts's per-sentence peak normalisation, which all engines share via `to_pcm16`).
- **Kokoro:** `model.onnx` (fp32; fp16 exports give NaN on the WebGPU EP), `voices/<id>.bin` (510 × 256 style table, row by token count), at most 510 tokens per call; voices whose language espeak can phonemise (a/b English, e, f, h, i, p); Japanese and Chinese need other front ends and are left out. 24 kHz. Verified: GPU output matches the CPU on 29 of 30 sentences (the one difference is small numeric drift, no added high-frequency noise, so not onnxruntime issue #29807).
- **KittenTTS:** the nano model (15M); mini (80M) and micro (40M) were 5–10× slower on this iGPU and CPU. 24 kHz, English.
- Speed (`length_scale`): packs take `speed = 1 / length_scale`.

### 4.5 Pocket TTS (voice pack, CPU)

- Kyutai's 100M model (CC-BY-4.0) through the single-file ONNX runtime of thewh1teagle/pocket-tts-onnx (vendored in `third_party/pocket_tts_onnx`): one `.onnx` holds the graph, the SentencePiece tokenizer and the voices. Perfect round trip.
- Autoregressive: it decodes 80 ms frames one after another, and the engine's `stream_audio()` hands each frame to Home Assistant as soon as it exists (first audio after ~0.07 s instead of after the sentence). CPU RTF 0.42; on the WebGPU EP RTF 1.84 (too many tiny steps), so the stack runs it with `device = "cpu"`.
- Frames cannot be peak-normalised per sentence while streaming, so a fixed `gain` (1.5) brings it to the level of the other engines. The voice `cosette` is excluded in the stack (so quiet it fails half of the round trips).

### 4.6 CosyVoice3 (voice pack, optional, child process)

- Fun-CosyVoice3-0.5B-2512 (Apache-2.0): text → Qwen2.5-0.5B language model (speech tokens, 25 per second) → flow-matching DiT (10 Euler steps with classifier-free guidance) → HiFT vocoder, 24 kHz, 9 languages. [cosyvoice.cpp](https://github.com/Lourdle/cosyvoice.cpp) (MIT) runs all of it on ggml, including Vulkan; its models are GGUF (`Lourdle/Fun-CosyVoice3-0.5B-2512-GGUF`, Q8_0).
- **Child process:** the engine starts `cosyvoice-server --api` on a free loopback port with every voice, waits for `/healthz`, and requests raw PCM from `/v1/audio/speech` (whole sentences; the server's chunked streaming was slower in total without its DiT cache, and that cache crashed it). One request at a time. Closing the engine stops the process.
- **GPU proof:** the server gets an explicit ggml device (`--backend VulkanN` for the engine's `gpu`, or the option `gpu_device`) and exits if it does not exist ("failed to initialize backend"), so it cannot fall back silently; that exit becomes `GpuUnavailable` (`igpu`: the pack fails; `auto`: CPU fallback, loud). The device name shown in `info` is the description of the same ggml device in the server process's own ggml list.
- **Voices** are voice prompts, not built in: `voices/<name>.wav` + `<name>.txt` (the clip's exact transcript) is encoded once at load by `cosyvoice-cli --frontend-only` (speech tokenizer + CAM++ speaker model, ONNX on the CPU, ~4 s) into a cache keyed by the files' contents; `<name>.gguf` is used as is. Every voice speaks every language (cross-lingual). The fetch script provides the upstream example clip (Apache-2.0, a Chinese speaker) as `zh_female`.
- **Measured on the N305 iGPU:** ~20 s for a 2 s sentence (RTF 9–12); the profile shows ~2.2 s per flow step (matrix products at ~175 GFLOPS, near what ggml gets from this GPU), so it is the GPU's compute, not a build problem. The language model alone runs at ~33 tokens/s. On the CPU RTF ~70. Q4_K_M is no faster than Q8_0. Output is intelligible (0 % word error on a two-sentence test with an English prompt). Off by default (`COSYVOICE_ENABLED`). **On an RTX 4060:** RTF 0.17–0.19 (~0.7 s per sentence), with `stream = true` first audio after ~0.35 s at RTF ~0.4: usable, with the rest of the stack on the iGPU (§7).

### 4.7 Engines tested and not added

- **Fish Speech 1.5, XTTS v2:** not attempted (non-commercial weights, far too large for this class of GPU).

### 4.8 CPU fallback

Parakeet and Whisper with `use_gpu = false`; Piper, Kokoro and KittenTTS with the CPU EP; CosyVoice3 with ggml's CPU backend. Everything keeps working, slower (see `README.md`).

## 5. Start-up

1. Log versions, the Vulkan driver restriction and the render nodes in the container.
2. Load each engine and prove its GPU path (§4); `igpu`: exit code 3 on failure, `auto`: warnings and `[CPU fallback]` in `info`.
3. Warm up: ggml-vulkan compiles some pipelines only when an input length first needs them (the first 1, 2 and 3 s clips each cost 0.4–3.6 s extra on an Intel iGPU, nothing new from 4 s), so each STT engine transcribes 1, 2, 3 and 5 s of silence; Piper speaks one sentence.
4. Scan the voice folders (a required folder that is unusable stops the start), then open all endpoints. The healthcheck passes from here.

## 6. Deployment (`deploy/`)

- `compose.yaml` + `.env` (template `.env.example`); works as a Dockge stack. The image is built locally (`pull_policy: never`).
- Render node by `RENDER_DEVICE` (long `devices` syntax, because by-path names contain colons), the host's render group by `RENDER_GID`, user by `PUID`/`PGID`.
- **One data folder** `DATA_DIR` → `/data` (read-only, `rslave`, `create_host_path`): `models/<engine>/` (`scripts/fetch-models.sh -d <DATA_DIR>/models`) and `voices/` (own Piper voices). It may be a network share: if it is not mounted yet when the container starts, the models are missing, the server exits, Docker restarts it, and the share appears inside the container once the host mounts it (`rslave`). An optional extra voice folder (`EXTRA_VOICES_DIR` → `/voices-extra`).
- Log rotation 3 × 10 MB.
- **Diagnostics page** (`web.py`, port 10312): runs in the same event loop and reads the live engines and the voice registry. The benchmark creates separate engine instances with `create_engine` (so "cold" includes loading, and the serving engines are not disturbed apart from sharing the GPU), one model at a time, and closes each afterwards. Memory per model is the growth of the container's memory (cgroup) during load and warm-up: on an iGPU the model's buffers are system RAM and are charged there, whereas the process RSS misses them.

## 7. Several GPUs (0.9.0)

Each engine picks its GPU with `gpu = "intel" | "nvidia" | "amd" | "<part of the Vulkan device name>"` (`GpuConfig.select`; empty = the `[gpu]` default) and its device with `device = "gpu" | "cpu" | "auto"` (`gpu` = `igpu`, the old name).

**Getting an NVIDIA GPU into the container through Vulkan** (NVIDIA Container Toolkit ≥ 1.12, headless Vulkan): the NVIDIA runtime with `NVIDIA_DRIVER_CAPABILITIES` including `graphics` (it replaces the default `compute,utility`); the toolkit then mounts the host's `nvidia_icd.json` at `/etc/vulkan/icd.d/` and the driver libraries. `--gpus all` alone does not. The ICD needs `libXext.so.6` and `libEGL.so.1`, both already in the base image. The image lists both drivers in `VK_DRIVER_FILES` (`intel_icd.json:/etc/vulkan/icd.d/nvidia_icd.json`, still no `llvmpipe`); the loader skips a listed file that does not exist, and the server also removes such files from the variables at start-up (so child processes get a clean list) and accepts the list as long as one file exists.

**How each engine chooses:**

| Engine | Mechanism | Proof |
|---|---|---|
| Parakeet, Whisper (ggml in-process) | whisper.cpp's `gpu_device` = the index of the matching device among ggml's GPUs (ggml v0.23 lists every Vulkan GPU, integrated ones too) | the library's log line `using VulkanN backend` must name that device |
| CosyVoice3 (child process) | `--backend VulkanN`, the ggml name of the matching device in this process's list (same loader, same driver files) | the server exits if the device does not exist |
| Piper, Kokoro, Kitten, Pocket (ONNX Runtime WebGPU plug-in 0.4.0) | `powerPreference`: `low-power` → the integrated GPU, `high-performance` → the discrete one (only when there are several GPUs) | WebGPU EP active; the adapter itself cannot be read back from the plug-in, so the device name shown is the predicted one |

**Measured limits of the WebGPU plug-in** (Intel iGPU + RTX 4060, KittenTTS): the EP device passed to `add_provider_for_devices` is ignored (with no preference Dawn takes the discrete GPU whichever device is passed); `powerPreference` decides. All sessions of a process share WebGPU context 0: a second session asking for the other GPU silently ran on the first session's GPU; a separate context (`deviceId` > 0) fails without a custom WebGPU instance, and the session silently becomes a CPU session. Hence (0.10.0) every text-to-speech model runs in its own worker process (§3.5), which gives each its own WebGPU context and so its own GPU; `make_session` still refuses a second GPU within one process. Two discrete GPUs cannot be told apart by power preference. `adapterIndex` (merged upstream 2026-09-25, not in 0.4.0) would lift this. Concurrent `Run()` on two WebGPU sessions has been reported to crash (onnxruntime #32561); with one model per process that cannot happen, and a crash would only stop one worker.

**A pack with `device = "gpu"` whose GPU is missing** is not offered at all (warning), since it could only fail on first use.

**Measurements** on the Intel iGPU, the RTX 4060 and the CPU: `README.md` (Measurements).
