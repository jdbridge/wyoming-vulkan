#!/usr/bin/env bash
# Download models and voices at pinned Hugging Face revisions and verify their sha256 (known-good defaults; the
# server itself takes any compatible file the config names).
# Usage: scripts/fetch-models.sh [-d dir] [model ...]      (default dir ./models, default model parakeet-q8_0)
#   scripts/fetch-models.sh -d deploy/models parakeet-q8_0 whisper-small
#   scripts/fetch-models.sh -d deploy/voices piper-en_US-ljspeech-high      (a voice: .onnx + .onnx.json)
#   scripts/fetch-models.sh --list
set -euo pipefail
DIR=./models
PARAKEET="ggml-org/parakeet-GGUF@35156454d1a39de06863303dd209fd2bed6ee079"   # repo head 2026-06-16
WHISPER="ggerganov/whisper.cpp@5359861c739e955e79d9a303bcbc70fb988958b1"     # repo head 2024-10-29
PIPER="rhasspy/piper-voices@c10ece1aade47bb51c153c893d14e5bf8e5b7117"        # Piper voice catalogue
# key -> "repo@revision path-in-repo sha256 size" (the file is saved under its base name)
declare -A MODELS=(
  [piper-en_US-ljspeech-high.onnx]="$PIPER en/en_US/ljspeech/high/en_US-ljspeech-high.onnx 5d4f08ba6a2a48c44592eed3ce56bf85e9de3dd4e20df90541ae68a8310c029a 114MB"
  [piper-en_US-ljspeech-high.json]="$PIPER en/en_US/ljspeech/high/en_US-ljspeech-high.onnx.json 7e1f4634af596d83cca997fb7a931ba80b70f8a316a2655ee69c55365e0ace14 5kB"
  [parakeet-q8_0]="$PARAKEET ggml-parakeet-tdt-0.6b-v3-q8_0.bin 4d64e9e96c2792186d072fde0034df0ad670cf680a2f53069052ead827fd600e 669MB"
  [parakeet-f16]="$PARAKEET ggml-parakeet-tdt-0.6b-v3-f16.bin 833bffc9513b2cae867ee9e51633cfd11e4d51aaa5597c8ac02159385a2b426f 1.26GB"
  [parakeet-q4_0]="$PARAKEET ggml-parakeet-tdt-0.6b-v3-q4_0.bin aa7fe2f5fb47d863ca23e8b1d490632d63a2599f515268b6d6bd656158dad45e 356MB"
  [whisper-tiny.en]="$WHISPER ggml-tiny.en.bin 921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f 78MB"
  [whisper-base.en]="$WHISPER ggml-base.en.bin a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002 148MB"
  [whisper-base]="$WHISPER ggml-base.bin 60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe 148MB"
  [whisper-small.en]="$WHISPER ggml-small.en.bin c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d 488MB"
  [whisper-small]="$WHISPER ggml-small.bin 1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b 488MB"
  [whisper-small-q8_0]="$WHISPER ggml-small-q8_0.bin 49c8fb02b65e6049d5fa6c04f81f53b867b5ec9540406812c643f177317f779f 264MB"
  [whisper-medium-q5_0]="$WHISPER ggml-medium-q5_0.bin 19fea4b380c3a618ec4723c3eef2eb785ffba0d0538cf43f8f235e7b3b34220f 539MB"
  [whisper-large-v3-turbo-q5_0]="$WHISPER ggml-large-v3-turbo-q5_0.bin 394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2 574MB"
  [whisper-large-v3-turbo-q8_0]="$WHISPER ggml-large-v3-turbo-q8_0.bin 317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1 874MB"
)
# a Piper voice is two files
declare -A VOICE_FILES=(
  [piper-en_US-ljspeech-high]="piper-en_US-ljspeech-high.onnx piper-en_US-ljspeech-high.json"  # public domain (LJ Speech)
)
KEYS=()
while (($#)); do
  case $1 in
    -d) DIR=$2; shift 2 ;;
    --list) for k in $(printf '%s\n' "${!MODELS[@]}" | sort); do read -r _ f _ size <<<"${MODELS[$k]}"; printf '%-34s %-38s %s\n' "$k" "${f##*/}" "$size"; done
            printf '%s\n' "${!VOICE_FILES[@]}" | sort | sed 's/$/  (voice: .onnx + .onnx.json)/'; exit 0 ;;
    *) if [[ -n ${VOICE_FILES[$1]:-} ]]; then read -r -a g <<<"${VOICE_FILES[$1]}"; KEYS+=("${g[@]}"); else KEYS+=("$1"); fi; shift ;;
  esac
done
((${#KEYS[@]})) || KEYS=(parakeet-q8_0)
mkdir -p "$DIR"; cd "$DIR"
for key in "${KEYS[@]}"; do
  [[ -n ${MODELS[$key]:-} ]] || { echo "unknown model $key (see --list)" >&2; exit 1; }
  read -r src path sha _ <<<"${MODELS[$key]}"
  repo=${src%@*}; rev=${src#*@}; f=${path##*/}
  if ! [[ -f $f ]] || ! echo "$sha  $f" | sha256sum -c --quiet - 2>/dev/null; then
    curl -fsSL -o "$f.part" "https://huggingface.co/$repo/resolve/$rev/$path"
    echo "$sha  $f.part" | sha256sum -c --quiet - && mv "$f.part" "$f"
  fi
  echo "$DIR/$f ok"
done
