#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# pad_identity.sh: does a padded prefill chunk compute the unpadded answer?
#
# CHARSIU_PREFILL_PAD runs the last chunk of a prompt at the next multiple of
# four, the extra rows a copy of the last real token, instead of splitting off
# an odd token for the token loop. Two things have to hold for that to be
# free: the head must read the last REAL row, and the pad rows' cache entries
# must be given back (llama_cache_truncate) so nothing after the prompt reads
# them. This puts the two arms side by side:
#
#   - one shot runs at prompt lengths chosen to land on every residue mod 4,
#     below, inside and above one chunk: the text (the stdout before the
#     `[load` line) and the whole first logit vector (CHARSIU_LOGIT_DUMP),
#   - one conversation of four turns (-i), every turn fed at the position the
#     last one left, which is what a pad row left in the cache would break.
#
# ON THE DESK THE TWO ARMS MUST AGREE TO THE BIT: the CPU computes each row of
# a batched matmul alone, so a chunk's width changes no row's arithmetic. On
# the board it may not -- a different width is a different set of matmul
# shapes and the note over CHARSIU_PREFILL_CHUNK in charsiu_run.c has a text
# that moved with the chunk -- so there a DIFF is something to read, not a
# verdict: look at whether the two logit vectors' argmaxes are a near tie.
#
# The fp16 attention mirror (CHARSIU_ATTN_NPU) only exists where the NPU does,
# so its half of the truncate step is exercised on the board and nowhere else.
#
# Usage: pad_identity.sh RUN MODEL.gguf TEXT [N_GEN]
set -eu

RUN=${1:?usage: pad_identity.sh RUN MODEL.gguf TEXT [N_GEN]}
MODEL=${2:?usage: pad_identity.sh RUN MODEL.gguf TEXT [N_GEN]}
TEXT=${3:?usage: pad_identity.sh RUN MODEL.gguf TEXT [N_GEN]}
NGEN=${4:-32}
LENS=${PAD_LENS:-"81 82 84 87 114 115 124 125 131 139 157 158 163 164 165 167"}

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT INT TERM

ntok() {
	"$RUN" "$MODEL" -p "$1" --tokens 2>/dev/null |
		awk '$1 ~ /^[0-9]+$/ { n = $1 + 1 } END { print n + 0 }'
}

# THE TEXT BEFORE THE SUMMARY LINE, because the summary carries timings
text_of() { sed '/^\[load /,$d' "$1" | md5sum | cut -c1-12; }

# A prompt of exactly n tokens, cut from TEXT at a word boundary, or nothing
# if no word count lands there. One walk over the text for all lengths.
words=$(wc -w < "$TEXT")
k=20
fail=0
printf '%5s  %-12s %-12s  %-7s  %s\n' tokens 'PAD=1 text' 'PAD=0 text' logits 'PAD=1 widths'
for n in $LENS; do
	p=""
	while [ "$k" -le "$words" ]; do
		p=$(tr -s ' \n' '  ' < "$TEXT" | cut -d' ' -f1-"$k")
		t=$(ntok "$p")
		[ "$t" -ge "$n" ] && break
		k=$((k + 1))
	done
	if [ "${t:-0}" -ne "$n" ]; then
		printf '%5s  (no word count lands on it; skipped)\n' "$n"
		continue
	fi
	for a in 1 0; do
		CHARSIU_PREFILL_PAD=$a "$RUN" "$MODEL" -p "$p" -n "$NGEN" \
			--temp 0 -c 1024 > "$W/o$a" 2> "$W/e$a"
		CHARSIU_PREFILL_PAD=$a CHARSIU_LOGIT_DUMP="$W/l$a" "$RUN" \
			"$MODEL" -p "$p" -n 1 --temp 0 -c 1024 >/dev/null 2>&1
	done
	h1=$(text_of "$W/o1"); h0=$(text_of "$W/o0")
	if cmp -s "$W/l1" "$W/l0"; then lg=same; else lg=DIFF; fi
	wd=$(grep -oE 'widths [0-9x+]+\)( \([0-9]+ pad rows?\))?' "$W/e1" |
		head -1)
	grep -q 'rest a token' "$W/e1" && wd="$wd, a token alone"
	v=ok
	[ "$h1" = "$h0" ] && [ "$lg" = same ] || { v=DIFF; fail=$((fail + 1)); }
	printf '%5s  %-12s %-12s  %-7s  %s  %s\n' "$n" "$h1" "$h0" "$lg" \
		"$wd" "$v"
done

# --- a conversation -------------------------------------------------------
first=$(tr -s ' \n' '  ' < "$TEXT" | cut -d' ' -f1-110)
for a in 1 0; do
	printf '%s\n%s\n%s\n' "Summarise that in one sentence." \
		"What is the main character's name, and what does he do?" \
		"Now say it again, but shorter." |
		CHARSIU_PREFILL_PAD=$a "$RUN" "$MODEL" -i -p "$first" \
			-n "$NGEN" --temp 0 -c 2048 2>/dev/null |
		grep -v '^\[' > "$W/c$a"
done
c1=$(md5sum < "$W/c1" | cut -c1-12); c0=$(md5sum < "$W/c0" | cut -c1-12)
if [ "$c1" = "$c0" ]; then v=ok; else v=DIFF; fail=$((fail + 1)); fi
printf '\nfour turns (-i):  PAD=1 %s  PAD=0 %s  %s\n' "$c1" "$c0" "$v"

if [ "$fail" -ne 0 ]; then
	echo "pad_identity: $fail case(s) differ"
	exit 1
fi
echo "pad_identity: every case identical"
