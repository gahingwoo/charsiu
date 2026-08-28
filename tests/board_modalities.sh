#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# The first board round for seeing, matching and hearing.
#
# ⚠ NONE OF THESE THREE USES THE NPU YET. vision.c, clip.c and whisper.c call
# gguf_matvec directly; only llama.c's projections are routed. So this round is
# not asking whether the hardware is fast at them. It is asking two things:
#
#   1. do they give the SAME ANSWERS on aarch64 that they give on a host --
#      the offline tests need numpy and the board may not have it, so the
#      checkable part here is the known clip, the known picture, the known
#      ranking;
#   2. what they COST on the CPU, which is the denominator for routing them.
#
# It downloads what it needs and skips what is already there.
#
#   sh tests/board_modalities.sh
set -eu

DIR=${CHARSIU_BOARD_DIR:-$HOME/charsiu-board}
mkdir -p "$DIR"

find_one() {
	for d in /usr/bin /opt/charsiu "$PWD/build" "$PWD"; do
		[ -x "$d/$1" ] && { echo "$d/$1"; return 0; }
	done
	return 1
}
RUN=$(find_one charsiu_run) || { echo "charsiu_run not found" >&2; exit 1; }
WSP=$(find_one charsiu_whisper) || { echo "charsiu_whisper not found -- update to dev" >&2; exit 1; }
CLP=$(find_one charsiu_clip) || { echo "charsiu_clip not found -- update to dev" >&2; exit 1; }

MODELS=${CHARSIU_MODELS:-$HOME/.charsiu/models}
[ -d "$MODELS" ] || MODELS=/opt/charsiu/models

get() {   # url, destination
	[ -f "$2" ] && return 0
	echo "  fetching $(basename "$2")"
	curl -fsSL -o "$2.part" "$1" && mv "$2.part" "$2"
}

ms() { date +%s%3N 2>/dev/null || echo 0; }
took() { echo "$(( $(ms) - $1 ))"; }

echo "== what this round needs =="
HF=https://huggingface.co
get "$HF/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin" "$DIR/ggml-tiny.en.bin"
get "$HF/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf" "$DIR/smolvlm.gguf"
get "$HF/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf" "$DIR/mmproj.gguf"
get "$HF/mys/ggml_clip-vit-base-patch32/resolve/main/clip-vit-base-patch32_ggml-model-q4_0.gguf" "$DIR/clip-b32.gguf"
get "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav" "$DIR/jfk.wav"
get "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/media/llama0-logo.png" "$DIR/logo.png"
echo

FAIL=0
say_result() {  # what, got, want
	case "$2" in
	*"$3"*) echo "  PASS  $1" ;;
	*)      echo "  FAIL  $1"; echo "        got: $2"; FAIL=1 ;;
	esac
}

echo "== 1. hearing: whisper tiny.en on jfk.wav =="
t0=$(ms)
OUT=$("$WSP" "$DIR/ggml-tiny.en.bin" --transcribe --audio "$DIR/jfk.wav")
T_WSP=$(took "$t0")
echo "  $OUT"
say_result "the words" "$OUT" "ask not what your country can do for you"
echo "  ${T_WSP} ms for 11 s of audio, all of it on the CPU"
echo

echo "== 2. seeing: SmolVLM-256M on the llama.cpp logo =="
t0=$(ms)
OUT=$("$RUN" "$DIR/smolvlm.gguf" --mmproj "$DIR/mmproj.gguf" --image "$DIR/logo.png" \
	-p "User:<image>What animal is this?<end_of_utterance>
Assistant:" -n 8 -c 1024 -t 4 2>/dev/null | sed -n '2p')
T_VIS=$(took "$t0")
echo "  $OUT"
say_result "the animal" "$OUT" "Llama"
echo "  ${T_VIS} ms, of which the tower is most of it"
echo

echo "== 3. matching: CLIP against the same picture =="
t0=$(ms)
OUT=$("$CLP" "$DIR/clip-b32.gguf" --image "$DIR/logo.png" --text \
	"a drawing of a llama" "the statue of liberty" "a dog running on grass")
T_CLP=$(took "$t0")
echo "$OUT" | sed 's/^/  /'
BEST=$(echo "$OUT" | sort -rn | head -1)
say_result "the best match" "$BEST" "llama"
echo "  ${T_CLP} ms"
echo

echo "== 4. the decode this all sits on, unchanged? =="
t0=$(ms)
OUT=$("$RUN" "$DIR/smolvlm.gguf" -p "The capital of France is" -n 16 --ignore-eos \
	-c 512 -t 4 2>/dev/null | tail -3 | head -1)
echo "  $OUT"
echo

echo "======================================================================="
echo "  whisper  ${T_WSP} ms   vision ${T_VIS} ms   clip ${T_CLP} ms"
echo "  ⚠ all three are CPU only today. These are the numbers that say what"
echo "    routing them to the NPU would be worth."
[ "$FAIL" = 0 ] && echo "  PASS" || echo "  FAILED"
echo "======================================================================="
exit $FAIL
