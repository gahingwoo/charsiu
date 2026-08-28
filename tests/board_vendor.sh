#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# charsiu against Rockchip's own published numbers, on their protocol.
#
# The table below is copied from
#   https://raw.githubusercontent.com/airockchip/rknn-llm/main/benchmark.md
# fetched 2026-08-28, RK3576 section, and it is the vendor's runtime on the
# vendor's driver. Their protocol is a 128 token prompt and 64 new tokens, so
# that is what this runs.
#
# ⚠⚠ THEIR NUMBERS ARE AT MAXIMUM CPU AND NPU FREQUENCY. Their own header says
# so: "collected based on the maximum CPU and NPU frequencies of each platform",
# with a script in their repo to set them. This runs at whatever the governor is
# doing, which on an idle board is ondemand. That is not a small difference and
# it is not one to quietly leave out of a comparison -- CHARSIU_BENCH_PERF=1
# sets the performance governor first and says it did.
#
# ⚠ AND w4a16 IS THEIR int4, WHICH IS OURS. w4a16_g128 is a finer group size
# and w8a8 is int8; the rows compared here are w4a16 against CHARSIU_NPU_W4V=1.
#
#   sh tests/board_vendor.sh
set -eu

DIR=${CHARSIU_BOARD_DIR:-$HOME/charsiu-board}
mkdir -p "$DIR"
MODELS=${CHARSIU_MODELS:-$HOME/.charsiu/models}
[ -d "$MODELS" ] || MODELS=/opt/charsiu/models

RUN=""
for d in /usr/bin /opt/charsiu "$PWD/build"; do
	[ -x "$d/charsiu_run" ] && { RUN="$d/charsiu_run"; break; }
done
[ -n "$RUN" ] || { echo "charsiu_run not found" >&2; exit 1; }

# ⚠ THE PULL NAME IS IN THE ROW, because "not here" without it is a dead end:
# the zoo's name and the file's name are not the same string and nobody should
# have to grep charsiu-get to find out which to type.
#
# pull name | file | vendor tok/s (w4a16) | vendor TTFT ms | vendor MB | label
rows() {
cat <<'ROWS'
qwen3-0.6b-q4|Qwen3-0.6B-Q4_0.gguf|24.85|468.61|512.71|Qwen3 0.6B
tinyllama-1.1b-q4|tinyllama-1.1b-chat-v1.0.Q4_0.gguf|19.71|543.68|601.09|TinyLLAMA 1.1B
phi3.5-mini-q4|Phi-3.5-mini-instruct-Q4_0.gguf|6.58|1829.12|1995.78|Phi3 3.8B
gemma4-e2b-q4|gemma-4-E2B-it-Q4_0.gguf|9.23|1219.25|1463.42|Gemma4 E2B
ROWS
}

# ⚠ 128 TOKENS IS THEIR SEQLEN, so the prompt has to be about that and the
# actual count is printed rather than assumed -- a comparison whose left column
# ran 40 tokens against their 128 is not a comparison.
PROMPT="The history of computing begins long before the first electronic machine.
Mechanical calculators, punched cards and tabulating engines each solved a part
of the problem, and each was built by somebody who wanted an answer faster than
a room full of clerks could produce one. What changed with the stored program
computer was not arithmetic but instruction: the machine could be told what to
do next by the same medium that held its data. Explain in your own words why
that distinction matters, and what it made possible that had not been possible."

if [ "${CHARSIU_BENCH_PERF:-0}" = 1 ]; then
	echo "setting the performance governor, as their protocol does"
	for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
		[ -w "$g" ] && echo performance > "$g" || true
	done
	echo "  governors now: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
else
	echo "⚠ the governor is left alone; their numbers are at MAXIMUM frequency."
	echo "  CHARSIU_BENCH_PERF=1 sets it, and this line changes when you do."
fi
echo "  governor: $(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
echo

printf '%-16s %10s %10s   %10s %10s   %8s %8s\n' \
	model "ours t/s" "theirs" "ours TTFT" "theirs" "ours MB" "theirs"

rows | while IFS='|' read -r name file vt vttft vmb label; do
	M=""
	for d in "$MODELS" "$DIR"; do
		[ -f "$d/$file" ] && { M="$d/$file"; break; }
	done
	if [ -z "$M" ]; then
		printf '%-16s %10s %10s   %10s %10s   %8s %8s   charsiu pull %s\n' \
			"$label" "-" "$vt" "-" "$vttft" "-" "$vmb" "$name"
		continue
	fi
	# ⚠ STDERR IS KEPT. The refusal that decides whether the batched prefill
	# does anything at all -- "int4 computes one row" -- is on it, and a
	# comparison that throws it away cannot tell a batched run from a run
	# that fell back to exactly what it was being compared against.
	# ⚠⚠ A FAILING RUN USED TO END THE WHOLE SCRIPT, IN SILENCE. `set -e`
	# plus OUT=$(cmd) means one model that will not load takes every model
	# after it with it -- three rounds of this table printed exactly one row
	# and nobody, me included, asked where the other four went. A row that
	# fails is a row, and it says so.
	OUT=$(CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
	      CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
	      CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 \
	      "$RUN" "$M" -p "$PROMPT" -n 64 --ignore-eos -c 512 -t 4 \
	      2>"$DIR/.v.err" | grep '^\[load') || OUT=""
	if [ -z "$OUT" ]; then
		printf '%-16s %10s %10s   %10s %10s   %8s %8s   THE RUN FAILED\n' \
			"$label" "-" "$vt" "-" "$vttft" "-" "$vmb"
		tail -3 "$DIR/.v.err" | sed 's/^/     /'
		continue
	fi
	NP=$(echo "$OUT" | sed 's/.*| *prompt \([0-9]*\) tok.*/\1/')
	TT=$(echo "$OUT" | sed 's/.*prompt [0-9]* tok in \([0-9]*\) ms.*/\1/')
	TS=$(echo "$OUT" | sed 's/.*| *gen [0-9]* tok in [0-9]* ms, \([0-9.]*\) tok.*/\1/')
	MB=$(echo "$OUT" | sed 's/.*peak \([0-9]*\) MB.*/\1/')
	printf '%-16s %10s %10s   %10s %10s   %8s %8s   (%s tok prompt)\n' \
		"$label" "$TS" "$vt" "$TT" "$vttft" "$MB" "$vmb" "$NP"
	grep -E "prompt batched|prompt a token|int4 computes one row|NPU pool" \
		"$DIR/.v.err" | sed 's/^/     /' | sort -u || true
done

echo
echo "⚠ TTFT AND PROMPT TIME ARE NOT THE SAME THING. Theirs is time to the first"
echo "  token, ours is the prompt's forward passes; the first token's own step is"
echo "  in theirs and not in ours, which is one token's worth in our favour."
echo
echo "SmolVLM-256M, the one multimodal model both sides publish, RK3576 w4a16:"
echo "  theirs   img-encoder 512x512   768 ms   prefill(128) 180 ms   decode 57.73 t/s"
if [ -f "$DIR/mmproj.gguf" ] && [ -f "$DIR/smolvlm.gguf" ]; then
	VIS=""
	for d in /usr/bin /opt/charsiu "$PWD/build"; do
		[ -x "$d/charsiu_vision" ] && { VIS="$d/charsiu_vision"; break; }
	done
	if [ -n "$VIS" ]; then
		t0=$(awk '{printf "%d", $1 * 1000}' /proc/uptime)
		CHARSIU_NPU=1 CHARSIU_STAGES=1 "$VIS" "$DIR/mmproj.gguf" \
			--encode >/dev/null 2>"$DIR/.vis2.err"
		t1=$(awk '{printf "%d", $1 * 1000}' /proc/uptime)
		echo "  ours     img-encoder 512x512   $((t1 - t0)) ms   (staging included)"
		echo
		# ⚠ WHERE, not just how much. 768 ms against ours is a number to
		# chase and the table is the only thing that says which part of
		# it to chase.
		sed -n '/charsiu vision:/,/pixel shuffle/p' "$DIR/.vis2.err" \
			| sed 's/^/  /'
		grep -E "NPU pool|ON THE HARDWARE" "$DIR/.vis2.err" | sed 's/^/  /' || true
	fi
fi
