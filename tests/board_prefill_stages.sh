#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# WHERE THE PROMPT'S TIME GOES, per model, on the hardware.
#
# ⚠⚠ THIS EXISTS BECAUSE THE GAP IS UNATTRIBUTED. charsiu is behind the
# vendor's published TTFT on all four models -- 1.31x, 1.64x, 1.63x, 1.82x --
# and the obvious explanation does not fit: the model with the MOST per-call
# dispatch overhead has the SMALLEST gap (Qwen3, 23% and 1.31x) and the one
# with the least has nearly the largest (Phi3, 4% and 1.63x).
#
# And the counters that gave those percentages cannot attribute prompt time
# anyway: charsiu_npu_report covers the whole process, and at 64 generated
# tokens a run the calls are mostly DECODE's. Reaching for them is a category
# error, which is exactly why the number is easy to reach for.
#
# llama.c already stamps every stage of the BATCHED prompt separately from the
# token loop's -- `bstage_ms`, added after phase 9 found the batched stage
# table summing to twice the NPU's own entry with a third of the prompt
# unnamed. Nothing had ever run it across the four scoreboard models.
#
# ⚠ -n 1, SO THE TABLE IS THE PROMPT'S. One generated token, so decode cannot
# contribute anything but its own single step, and CHARSIU_STAGES keeps the
# batched rows apart from the token loop's regardless.
#
# ⚠ THE MATMUL SPLIT IS THE LINE TO READ FIRST. "N calls: X ms a row inside
# the NPU entry" against what is in its wrapper and what fell to the CPU: a
# projection the hardware refuses falls back to a matvec a row at a time IN
# SILENCE, and that is the shape of an unnamed third of a prompt.
#
#   sh tests/board_prefill_stages.sh [PROMPT_WORDS]
set -eu

D=$(dirname "$0")
BIN=$D
[ -x "$D/../build/charsiu_run" ] && BIN=$D/../build
[ -x "$BIN/charsiu_run" ] || BIN=/opt/charsiu
[ -x "$BIN/charsiu_run" ] || { echo "no charsiu_run"; exit 2; }

find_model() {
	for _d in "$HOME/.charsiu/models" /opt/charsiu/models; do
		for _f in "$_d"/$1; do
			[ -f "$_f" ] && { echo "$_f"; return 0; }
		done
	done
	return 1
}

# board_vendor.sh's own protocol prompt, so the rows measured here are the rows
# the TTFT column is measured over. A different prompt length is a different
# chunking and a different table.
P="The history of computing begins long before the first electronic machine. Merchants kept accounts on clay, astronomers ruled tables by hand, and the abacus moved beads along a wire for two thousand years before anyone thought to make the beads move themselves. What changed was not arithmetic but who did it: a machine that could be told the order of operations once and would then repeat them without tiring, without a wage, and without the small drift of attention that makes a long column of figures a gamble. Explain, in plain words, why that shift mattered more than the speed:"

echo "== where the prompt's time goes, $(date -Is)"
echo "   binary $BIN/charsiu_run"
echo "   -n 1, so the table below is the PROMPT's and not the token loop's"
echo

for spec in "Qwen3 0.6B:Qwen3-0.6B-Q4_0.gguf" \
            "TinyLLAMA:tinyllama-1.1b*Q4_0.gguf" \
            "Phi3 3.8B:Phi-3.5-mini*Q4_0.gguf" \
            "Gemma4 E2B:gemma-4-E2B*Q4_0.gguf"; do
	lbl=${spec%%:*}; pat=${spec#*:}
	M=$(find_model "$pat" || true)
	if [ -z "$M" ]; then
		echo "-- $lbl: not on this card"
		echo
		continue
	fi
	echo "-- $lbl  $(basename "$M")"
	env CHARSIU_STAGES=1 CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
	    CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 \
	    "$BIN/charsiu_run" "$M" -p "$P" -n 1 --ignore-eos -c 512 -t 4 \
	    2>&1 >/dev/null |
	    grep -E 'batched stages|ms a row|matmul rows|attention:|fell|NPU entry' |
	    sed 's/^/   /'
	echo
done

echo "⚠ READ THE MATMUL SPLIT FIRST. Rows that fell to the CPU are a silent"
echo "  fallback and are the shape of an unnamed third of a prompt; the stage"
echo "  percentages below them are only meaningful once that line is zero."
