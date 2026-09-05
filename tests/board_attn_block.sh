#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# How many rows should the batched prompt's attention take in one pass over
# the cache?
#
# ⚠ THE NUMBER 8 WAS NEVER SWEPT. attn_block_rows() returns 8 and llama.c's
# note beside it explains why blocking won and why the pool won, but the two
# arms it compares are 8-on-the-pool against one-row-at-a-time. Nothing has
# ever asked whether 8 is the right 8: a wider block reads the same K and V
# rows for more query rows, which is the axpy traffic through L1 that the same
# note names as the next lever, and a block too wide spills the scores.
#
# Phase 9 on 2026-09-05 is why it is worth asking: attention is the largest
# row of the batched stage table on four of seven models -- 52.2% of Qwen3's
# prompt, 50.2% of SmolLM2-135M's, 41.9% of gemma-3-1b's -- and every matmul
# lever left is worth less than that.
#
# Same discipline as board_ab.sh: one binary, one session, the performance
# governor pinned, arms alternating, several repeats so the spread is visible
# next to the difference. The prompt is phase 9's, so a row here can be read
# against that table.
#
#   CHARSIU_ATTN_VALUES="4 8 16 32"   the blocks to try (8 is today's default)
#   CHARSIU_ATTN_MODELS=              ggufs, space separated (default: three)
#   CHARSIU_ATTN_REPEATS=2
set -u
VALUES=${CHARSIU_ATTN_VALUES:-4 8 16 32}
N=${CHARSIU_ATTN_REPEATS:-2}
RUN=${CHARSIU_RUN_BIN:-}
[ -n "$RUN" ] || for d in /opt/charsiu ./build .; do
	[ -x "$d/charsiu_run" ] && { RUN=$d/charsiu_run; break; }
done
[ -n "${RUN:-}" ] || { echo "no charsiu_run" >&2; exit 1; }

MODELS=${CHARSIU_ATTN_MODELS:-}
if [ -z "$MODELS" ]; then
	for p in '*Qwen3-0.6B*Q4_0*.gguf' '*SmolLM2-135M*Q4_0*.gguf' '*gemma-3-1b*Q4_0*.gguf'; do
		for d in "$HOME/.charsiu/models" /opt/charsiu/models; do
			for f in $d/$p; do
				[ -r "$f" ] && { MODELS="$MODELS $f"; break 2; }
			done
		done
	done
fi
[ -n "$MODELS" ] || { echo "no models" >&2; exit 1; }

# ⚠ THE GOVERNOR IS THE WHOLE POINT, and attention is pure CPU: under ondemand
# this measures the governor. Put it back on the way out.
OLD=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "")
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
	[ -w "$g" ] && echo performance > "$g" 2>/dev/null
done
trap '[ -n "$OLD" ] && for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do [ -w "$g" ] && echo "$OLD" > "$g" 2>/dev/null; done' EXIT

P=$(i=1; while [ $i -le 256 ]; do printf '%d ' "$i"; i=$((i+1)); done)
P=${P% }
W4="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 CHARSIU_STAGES=1"

echo "governor  $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) (was $OLD)"
echo "blocks    $VALUES ($N repeats each, alternating), prompt 256 words"
echo

for M in $MODELS; do
	[ -r "$M" ] || { printf '  %s: not readable, skipped\n' "$M"; continue; }
	b=$(basename "$M" .gguf)
	printf '  %s\n' "$b"
	printf '    %-7s %-10s %-12s %-12s %s\n' block pass "prompt ms" "attn ms/row" "text"
	# ⚠ THE TEXT IS PART OF THE MEASUREMENT. A block that is faster and
	# wrong is the fault this project has shipped twice; the first arm's
	# output is the reference and every later one is compared to it.
	ref=""
	i=1
	while [ "$i" -le "$N" ]; do
		for B in $VALUES; do
			# ⚠ THE PROMPT LINE IS ON EITHER STREAM depending on the
			# runner, and phase 9 greps both. stdout is kept apart
			# because the generated text is compared out of it.
			err=${TMPDIR:-/tmp}/attn_block.err
			out=$(env $W4 CHARSIU_ATTN_BLOCK="$B" \
			      "$RUN" "$M" -p "$P" -n 4 --ignore-eos 2>"$err")
			ms=$({ printf '%s\n' "$out"; cat "$err"; } \
			     | grep -oE 'prompt [0-9]+ tok in [0-9.]+ ms' \
			     | head -1 | grep -oE '[0-9.]+ ms$' | cut -d' ' -f1)
			at=$(printf '%s' "$out" | awk '/charsiu batched stages:/{f=1}
			     f && $1=="attention" {print $2; exit}')
			# the generated text is everything before the first report line
			tx=$(printf '%s' "$out" | awk '/^(charsiu |\[)/{exit} {print}')
			if [ -z "$ref" ]; then
				ref=$tx; v=ref
			elif [ "$tx" = "$ref" ]; then
				v=same
			else
				v="⚠ DIFFERS"
			fi
			printf '    %-7s %-10s %-12s %-12s %s\n' \
			    "$B" "$i" "${ms:-?}" "${at:-?}" "$v"
		done
		i=$((i + 1))
	done
	echo
done
echo "the block to ship is the smallest one whose attention row is inside the"
echo "spread of the best, with the text unchanged."
