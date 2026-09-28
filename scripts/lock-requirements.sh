#!/usr/bin/env bash
# Regenerate requirements.txt (full lock with hashes) from requirements.in, resolved with the image's own Python.
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -q --target base -t wyoming-vulkan-lockenv:local . >/dev/null
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/w" -w /w wyoming-vulkan-lockenv:local sh -c '
  python3 -m venv /tmp/pt && /tmp/pt/bin/pip install -q pip-tools==7.6.1 &&
  /tmp/pt/bin/pip-compile -q --generate-hashes --allow-unsafe --strip-extras --no-emit-index-url \
    --output-file requirements.txt requirements.in'
echo "requirements.txt updated"
