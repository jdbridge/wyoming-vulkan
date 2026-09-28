#!/usr/bin/env bash
# Download models and voices at pinned Hugging Face revisions into <models>/<engine>/ and verify them (sha256 for the
# large files; the small ones are pinned by the revision). Known-good defaults; the server takes any compatible file.
# Usage: scripts/fetch-models.sh [-d models-dir] [item ...]     (default dir ./models, default items: the stack's set)
#   scripts/fetch-models.sh -d deploy/data/models                    parakeet, whisper small, Piper LJ Speech, Kokoro, Kitten
#   scripts/fetch-models.sh -d deploy/data/models whisper-base.en    one more model
#   scripts/fetch-models.sh --list
set -euo pipefail
DIR=./models
PARAKEET="ggml-org/parakeet-GGUF@35156454d1a39de06863303dd209fd2bed6ee079"   # repo head 2026-06-16
WHISPER="ggerganov/whisper.cpp@5359861c739e955e79d9a303bcbc70fb988958b1"     # repo head 2024-10-29
PIPER="rhasspy/piper-voices@c10ece1aade47bb51c153c893d14e5bf8e5b7117"        # Piper voice catalogue
KOKORO="onnx-community/Kokoro-82M-v1.0-ONNX@1939ad2a8e416c0acfeecc08a694d14ef25f2231"  # Apache-2.0
KITTEN="KittenML/kitten-tts-nano-0.8-fp32@7a1db645b1f3ab9420761d87428e042b9cec3f26"  # Apache-2.0, 15M
KITTEN_MINI="KittenML/kitten-tts-mini-0.8@c02725660cea441db4c383af69f1f26f5cd00947"  # 80M, ~10x slower here

# item -> lines of "repo@revision path-in-repo sha256|- target-under-models size" (one line per file)
declare -A ITEMS
add() { ITEMS[$1]+="$2"$'\n'; }
add parakeet-q8_0 "$PARAKEET ggml-parakeet-tdt-0.6b-v3-q8_0.bin 4d64e9e96c2792186d072fde0034df0ad670cf680a2f53069052ead827fd600e parakeet/ggml-parakeet-tdt-0.6b-v3-q8_0.bin 669MB"
add parakeet-f16 "$PARAKEET ggml-parakeet-tdt-0.6b-v3-f16.bin 833bffc9513b2cae867ee9e51633cfd11e4d51aaa5597c8ac02159385a2b426f parakeet/ggml-parakeet-tdt-0.6b-v3-f16.bin 1.26GB"
add parakeet-q4_0 "$PARAKEET ggml-parakeet-tdt-0.6b-v3-q4_0.bin aa7fe2f5fb47d863ca23e8b1d490632d63a2599f515268b6d6bd656158dad45e parakeet/ggml-parakeet-tdt-0.6b-v3-q4_0.bin 356MB"
for w in "tiny.en 921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f 78MB" \
         "base.en a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002 148MB" \
         "base 60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe 148MB" \
         "small.en c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d 488MB" \
         "small 1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b 488MB" \
         "small-q8_0 49c8fb02b65e6049d5fa6c04f81f53b867b5ec9540406812c643f177317f779f 264MB" \
         "medium-q5_0 19fea4b380c3a618ec4723c3eef2eb785ffba0d0538cf43f8f235e7b3b34220f 539MB" \
         "large-v3-turbo-q5_0 394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2 574MB" \
         "large-v3-turbo-q8_0 317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1 874MB"; do
  read -r name sha size <<<"$w"
  add "whisper-$name" "$WHISPER ggml-$name.bin $sha whisper/ggml-$name.bin $size"
done
add piper-en_US-ljspeech-high "$PIPER en/en_US/ljspeech/high/en_US-ljspeech-high.onnx 5d4f08ba6a2a48c44592eed3ce56bf85e9de3dd4e20df90541ae68a8310c029a piper/en_US-ljspeech-high.onnx 114MB"
add piper-en_US-ljspeech-high "$PIPER en/en_US/ljspeech/high/en_US-ljspeech-high.onnx.json 7e1f4634af596d83cca997fb7a931ba80b70f8a316a2655ee69c55365e0ace14 piper/en_US-ljspeech-high.onnx.json 5kB"
add kokoro "$KOKORO onnx/model.onnx - kokoro/model.onnx 326MB"
add kokoro "$KOKORO tokenizer.json - kokoro/tokenizer.json 3kB"
# every voice whose language espeak can phonemise (a/b English, e Spanish, f French, h Hindi, i Italian, p Portuguese)
for v in af af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky \
         am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa \
         bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis \
         ef_dora em_alex em_santa ff_siwis hf_alpha hf_beta hm_omega hm_psi if_sara im_nicola pf_dora pm_alex pm_santa; do
  add kokoro "$KOKORO voices/$v.bin - kokoro/voices/$v.bin 0.5MB"
done
add kitten "$KITTEN kitten_tts_nano_v0_8.onnx 320564d2615f235de972ca27a7f39551c94185cfa24ca85b07a29084135f1e5e kitten/kitten_tts_nano_v0_8.onnx 57MB"
add kitten "$KITTEN voices.npz 8aa7cee235abb0739cb51e6559685f65a4dacd95568833d05699b1633f519b3f kitten/voices.npz 3MB"
add kitten "$KITTEN config.json - kitten/config.json 1kB"
# the larger mini model (same voices; its own folder, because voices.npz differs per model)
add kitten-mini "$KITTEN_MINI kitten_tts_mini_v0_8.onnx 0f5bbae4fc4800c98dbc544a87ecfa79510de2fb8222db30d12e5bfe9177df91 kitten-mini/kitten_tts_mini_v0_8.onnx 78MB"
add kitten-mini "$KITTEN_MINI voices.npz 40ad2638952b77b7b2f30127e2608e169fc69dd256b53bd8aaa3409a33193c42 kitten-mini/voices.npz 3MB"
add kitten-mini "$KITTEN_MINI config.json - kitten-mini/config.json 1kB"
DEFAULT=(parakeet-q8_0 whisper-small piper-en_US-ljspeech-high kokoro kitten)

KEYS=()
while (($#)); do
  case $1 in
    -d) DIR=$2; shift 2 ;;
    --list) for k in $(printf '%s\n' "${!ITEMS[@]}" | sort); do
              n=$(grep -c . <<<"${ITEMS[$k]}"); first=${ITEMS[$k]%%$'\n'*}
              extra=""; ((n > 1)) && extra=" (+$((n - 1)) files)"
              printf '%-30s %-50s %s\n' "$k" "$(awk '{print $4}' <<<"$first")$extra" "$(awk '{print $5}' <<<"$first")"
            done; exit 0 ;;
    *) KEYS+=("$1"); shift ;;
  esac
done
((${#KEYS[@]})) || KEYS=("${DEFAULT[@]}")
for key in "${KEYS[@]}"; do
  [[ -n ${ITEMS[$key]:-} ]] || { echo "unknown item $key (see --list)" >&2; exit 1; }
  count=0
  while read -r src path sha target _; do
    [[ -n $src ]] || continue
    repo=${src%@*}; rev=${src#*@}; out="$DIR/$target"
    mkdir -p "$(dirname "$out")"
    if [[ -f $out ]] && { [[ $sha == - ]] || echo "$sha  $out" | sha256sum -c --quiet - 2>/dev/null; }; then
      count=$((count + 1)); continue
    fi
    curl -fsSL -o "$out.part" "https://huggingface.co/$repo/resolve/$rev/$path"
    [[ $sha == - ]] || echo "$sha  $out.part" | sha256sum -c --quiet -
    mv "$out.part" "$out"; count=$((count + 1))
  done <<<"${ITEMS[$key]}"
  echo "$key: $count file(s) ok in $DIR"
done
