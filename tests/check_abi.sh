#!/usr/bin/env bash
# Check that the ctypes structs in wyoming_vulkan/engines match the pinned C headers (third_party/, whisper.cpp v1.9.4):
# the image writes a C program from the ctypes definitions, gcc compiles it against the headers, and every struct
# size and field offset is compared. Run after changing a binding or the whisper.cpp pin.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-wyoming-vulkan:dev}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
docker run --rm --entrypoint python --user "$(id -u):$(id -g)" -v "$PWD/tests:/tests:ro" -v "$WORK:/work" "$IMAGE" /tests/check_abi.py emit /work
docker run --rm -v "$PWD/third_party:/h:ro" -v "$WORK:/work" alpine:3.20@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc sh -c \
  "apk add -q gcc musl-dev >/dev/null && gcc -std=c11 -I/h -I/h/ggml -o /tmp/layout /work/layout.c && /tmp/layout > /work/c.json && chmod 666 /work/c.json"
docker run --rm --entrypoint python -v "$PWD/tests:/tests:ro" -v "$WORK:/work:ro" "$IMAGE" /tests/check_abi.py compare /work
