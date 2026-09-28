# wyoming-vulkan image: Parakeet and Whisper (whisper.cpp, ggml Vulkan) + Piper (ONNX Runtime WebGPU EP) on a Vulkan GPU.
# Every layer is pinned: the base by digest, apt by an Ubuntu snapshot date plus exact versions, Python by requirements.txt
# (hashes). After changing a pin, run tests/check_abi.sh and tests/run.sh (both modes).

# whisper.cpp v1.9.4 release (commit 927cfce, 2026-09-11), Vulkan build: Ubuntu 24.04.4, Mesa 25.2.8 (ANV), Vulkan loader
# 1.3.275, libparakeet/libwhisper 1.9.4, ggml-vulkan 0.23.0. Libraries live in /app/build/bin in this release's image.
# third_party/parakeet.h is the header of this same commit; the ctypes struct layouts must match it.
ARG BASE_IMAGE=ghcr.io/ggml-org/whisper.cpp:main-vulkan-927cfce34f31707e17f2bff35c349632fb9e2c3a@sha256:f54a86a01b2315ae0ce609b8b6f274259a779016830edddec96c9d35296eac07

FROM ${BASE_IMAGE} AS base
# Ubuntu snapshot service: apt sees the archive as it was at this moment, so the pinned versions stay installable.
ARG APT_SNAPSHOT=20260928T000000Z
RUN apt-get update --snapshot ${APT_SNAPSHOT} \
 && apt-get install -y --no-install-recommends \
      python3-venv=3.12.3-0ubuntu2.1 \
      python3.12-venv=3.12.3-1ubuntu0.17 \
      vulkan-tools=1.3.275.0+dfsg1-1 \
 && rm -rf /var/lib/apt/lists/* \
 && test -e /app/build/bin/libparakeet.so.1.9.4 \
 && echo /app/build/bin > /etc/ld.so.conf.d/whisper-cpp.conf && ldconfig \
 && python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# The base image's entrypoint is `bash -c`; cleared here, the runtime stage sets the server.
ENTRYPOINT []
WORKDIR /

# CosyVoice3 (optional voice pack): cosyvoice.cpp, built here with its own ggml (v0.23.0 tag, Vulkan) and the
# ONNX Runtime C library for its voice-prompt frontend. It runs as a separate process (two ggml builds in one
# process crash), from /opt/cosyvoice with its libraries next to it (RUNPATH $ORIGIN), never whisper.cpp's.
FROM ${BASE_IMAGE} AS cosyvoice-build
ARG APT_SNAPSHOT=20260928T000000Z
RUN apt-get update --snapshot ${APT_SNAPSHOT} \
 && apt-get install -y --no-install-recommends \
      g++=4:13.2.0-7ubuntu1 g++-14=14.2.0-4ubuntu2~24.04.1 gcc-14=14.2.0-4ubuntu2~24.04.1 \
      ninja-build=1.11.1-2 git=1:2.43.0-1ubuntu7.3 ca-certificates=20260601~24.04.1 curl=8.5.0-2ubuntu10.15 \
      libvulkan-dev=1.3.275.0-1build1 glslc=2023.8-1build1 spirv-headers=1.6.1+1.4.309.0-1~ubuntu0.24.04.2 \
      libicu-dev=74.2-1ubuntu3.1 pkg-config=1.8.1-2build1 \
 && rm -rf /var/lib/apt/lists/*
# cosyvoice.cpp main 2026-09-16; ggml v0.23.0 (the commit cosyvoice.cpp pins for its Metal build); ORT 1.30.0 (sha256)
ARG COSYVOICE_COMMIT=f46496b7a1f9da6cd0b0b8401f7e941235c6a413
ARG GGML_COMMIT=e91ded11bdcd78c42f9c8d3978ff6686eb4c1226
ARG ORT_VERSION=1.30.0
ARG ORT_SHA256=a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd
WORKDIR /src
RUN git init -q cosyvoice && cd cosyvoice \
 && git fetch -q --depth 1 https://github.com/Lourdle/cosyvoice.cpp ${COSYVOICE_COMMIT} && git checkout -q FETCH_HEAD \
 && git submodule update -q --init --depth 1 vendor/pcre2 \
 && git init -q vendor/ggml && git -C vendor/ggml fetch -q --depth 1 https://github.com/ggml-org/ggml.git ${GGML_COMMIT} \
 && git -C vendor/ggml checkout -q FETCH_HEAD \
 && curl -sSfL -o /tmp/ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz \
 && echo "${ORT_SHA256}  /tmp/ort.tgz" | sha256sum -c - \
 && mkdir -p /opt/ort && tar xzf /tmp/ort.tgz -C /opt/ort --strip-components=1 && rm /tmp/ort.tgz
# gcc 14 (C++20; its module support crashes on the server, hence PCH); no AVX10 tier (gcc 14 cannot build it);
# API-only server (the web UI needs C23 #embed); CPU code for x86-64 with AVX2, like the base image's ggml.
RUN cd cosyvoice && cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
      -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF \
      -DCOSYVOICE_SERVER_NO_WEBUI=ON -DCOSYVOICE_SERVER_DEFAULT_MODE=API -DCOSYVOICE_CLI_NO_PLAYBACK=ON \
      -DCOSYVOICE_HAS_AVX10_1=OFF -DCOSYVOICE_USE_PCH=ON \
      -DORT_PREBUILT_DIR=/opt/ort -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/opt/ort/lib \
      -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,/opt/ort/lib \
      -DCMAKE_INSTALL_RPATH='$ORIGIN:$ORIGIN/../lib' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
 && cmake --build build -j"$(nproc)" \
 && mkdir -p /opt/cosyvoice/bin /opt/cosyvoice/lib \
 && cp build/bin/cosyvoice-server build/bin/cosyvoice-cli /opt/cosyvoice/bin/ \
 && cp -P build/lib/*.so* /opt/ort/lib/libonnxruntime.so* /opt/cosyvoice/lib/ \
 && cp LICENSE /opt/cosyvoice/LICENSE.cosyvoice.cpp && cp /opt/ort/LICENSE /opt/cosyvoice/LICENSE.onnxruntime

FROM base AS runtime
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes --only-binary=:all: -r /tmp/requirements.txt && rm /tmp/requirements.txt
# Only the Intel Vulkan driver: hides llvmpipe (CPU) and the other drivers. The path is specific to this base image;
# a wrong one means "no Vulkan device" (the server refuses to start with device = "igpu").
ENV VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.json \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/wyoming-vulkan
COPY config/config.example.toml /etc/wyoming-vulkan/config.toml
COPY wyoming_vulkan /opt/wyoming-vulkan/wyoming_vulkan
# Pocket TTS runtime (vendored, CC-BY-4.0; see third_party/pocket_tts_onnx/__init__.py)
COPY third_party/pocket_tts_onnx /opt/wyoming-vulkan/pocket_tts_onnx
# CosyVoice3 server + CLI with their own ggml; fail the build if a library would come from elsewhere
COPY --from=cosyvoice-build /opt/cosyvoice /opt/cosyvoice
RUN ! ldd /opt/cosyvoice/bin/cosyvoice-server /opt/cosyvoice/bin/cosyvoice-cli | grep -E "not found|/app/build/bin"
EXPOSE 10310 10311 10312
# Healthy = port open after load, GPU check and warm-up (~15 s) and a describe/info round trip works.
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 CMD ["python", "-m", "wyoming_vulkan.healthcheck"]
ENTRYPOINT ["python", "-m", "wyoming_vulkan"]
