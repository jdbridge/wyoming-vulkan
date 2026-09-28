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
EXPOSE 10310 10311 10312
# Healthy = port open after load, GPU check and warm-up (~15 s) and a describe/info round trip works.
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 CMD ["python", "-m", "wyoming_vulkan.healthcheck"]
ENTRYPOINT ["python", "-m", "wyoming_vulkan"]
