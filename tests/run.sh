#!/usr/bin/env bash
# Full test run: build the image, run the unit tests in it, then the integration test against a real server container
# with the stack's own config (the inline config of deploy/compose.yaml, values from the env file). The container
# publishes no port (the tests run inside it) and is removed afterwards.
#   tests/run.sh          GPU: every engine must really run on the GPU; the optional voice folders are mounted
#   tests/run.sh cpu      no GPU in the container, device auto: every engine must fall back to the CPU and pass anyway;
#                         the optional voice folders are left out (missing optional folders must not matter)
# Env file: deploy/.env if it exists, else deploy/.env.example (override with ENV_FILE=...). It needs the models and
# the default voice in place: scripts/fetch-models.sh (README.md). Test audio: tests/data (override with AUDIO=...).
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-igpu}
IMAGE=${IMAGE:-wyoming-vulkan:dev}
NAME=wyoming-vulkan-test-$MODE
ENV_FILE=${ENV_FILE:-deploy/.env}
[[ -f $ENV_FILE ]] || ENV_FILE=deploy/.env.example
AUDIO=${AUDIO:-$PWD/tests/data/en}
AUDIO_DE=${AUDIO_DE:-$PWD/tests/data/de}

# a value from the env file (default if unset); relative paths are relative to deploy/, as in compose
val() { local v; v=$(grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//'); echo "${v:-$2}"; }
path() { local p; p=$(val "$1" "$2"); [[ $p == /* ]] && echo "$p" || echo "$PWD/deploy/${p#./}"; }
MODELS=$(path MODELS_DIR ./models)
VOICES=$(path VOICES_DIR ./voices)
LIBRARY=$(path VOICE_LIBRARY_DIR ./voice-library)
EXTRA=$(path EXTRA_VOICES_DIR ./voices-extra)
RENDER_DEVICE=$(val RENDER_DEVICE /dev/dri/renderD128)
RENDER_GID=$(val RENDER_GID 993)
USER_ID="$(val PUID 1000):$(val PGID 1000)"
echo "== env file $ENV_FILE: models $MODELS, voices $VOICES, GPU $RENDER_DEVICE (group $RENDER_GID), user $USER_ID"

# The stack's server config: the inline config of deploy/compose.yaml with the env file filled in
CONFIG=$(mktemp -d)/config.toml
docker compose -f deploy/compose.yaml --env-file "$ENV_FILE" config --format json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["configs"]["wyoming_vulkan_config"]["content"], end="")' > "$CONFIG"

echo "== build $IMAGE"
docker build -q -t "$IMAGE" . >/dev/null

echo "== unit tests"
docker run --rm --entrypoint python -v "$PWD/tests:/tests:ro" -v "$CONFIG:/deploy/config.toml:ro" "$IMAGE" -m unittest discover -s /tests -p "test_*.py"

echo "== integration ($MODE)"
# (docker run --device cannot take by-path names, their colons clash with its syntax: resolve the link first)
GPU=(--device "$(readlink -f "$RENDER_DEVICE"):/dev/dri/renderD128" --group-add "$RENDER_GID")
FOLDERS=()
for dir_target in "$LIBRARY:/voices-library" "$EXTRA:/voices-extra"; do
  [[ -d ${dir_target%%:/*} ]] && FOLDERS+=(--mount "type=bind,source=${dir_target%%:/*},target=/${dir_target#*:/},readonly,bind-propagation=rslave")
done
MIN_VOICES=1
WHISPER_LIMIT=1.5   # Whisper small with a ~10 s window: 0.82-0.97 s on an Intel N305 iGPU (2026-09-28)
if [[ $MODE == cpu ]]; then GPU=(); FOLDERS=(); WHISPER_LIMIT=6; fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
docker run -d --name "$NAME" "${GPU[@]}" --health-interval=3s --user "$USER_ID" \
  -v "$MODELS:/models:ro" -v "$VOICES:/voices:ro" "${FOLDERS[@]}" -v "$CONFIG:/etc/wyoming-vulkan/config.toml:ro" \
  -v "$AUDIO:/testdata:ro" -v "$AUDIO_DE:/testdata-de:ro" -v "$PWD/tests:/tests:ro" "$IMAGE" >/dev/null
for _ in $(seq 1 60); do
  status=$(docker inspect -f '{{.State.Status}} {{.State.Health.Status}}' "$NAME")
  [[ $status == "running healthy" ]] && break
  [[ $status == running* ]] || { docker logs "$NAME"; echo "server exited"; exit 1; }
  sleep 3
done
docker logs "$NAME" 2>&1 | grep -E "WARNING|ERROR|STT |TTS |Voices:|Endpoint|Ready" || true
[[ $status == "running healthy" ]] || { echo "server not healthy: $status"; exit 1; }
echo "-- Parakeet endpoint (with the voices)"
docker exec "$NAME" python /tests/integration.py --audio /testdata --expect "$MODE" --min-voices "$MIN_VOICES"
echo "-- Whisper endpoint (English and German)"
docker exec "$NAME" python /tests/integration.py --uri tcp://127.0.0.1:10311 --audio /testdata --expect "$MODE" --min-voices 0 \
  --stt-limit "$WHISPER_LIMIT" --audio-de /testdata-de --min-exact-de 4
