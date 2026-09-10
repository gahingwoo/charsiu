#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
#
# THE AWQ BOARD ROUND, AS ONE COMMAND.
#
# Four things about AWQ were built on a machine with no NPU and none of them
# has a number from the hardware. Three of the four are IDENTITY questions --
# two arms that must produce the same tokens -- and one is a price.
#
#   arm            question                                    decides
#   batch          does applying AWQ's factor at the batched    ship the batch,
#                  gather give the same tokens as applying      or keep the
#                  it a row at a time, and what does it save    refusal
#   share          does one packed input shared by a group      AWQ_SHARE on by
#                  give the same tokens as one pack each,       default
#                  and what does it save
#   layers         is AWQ on the first three blocks worth       AWQ_LAYERS as
#                  most of AWQ everywhere                       the default form
#   alpha          where is the exponent's minimum on this      the number the
#                  board's own quantiser                        README prints
#
# ⚠⚠ THE IDENTITY ARMS COMPARE charsiu TO charsiu, AND HERE THAT IS THE RIGHT
# QUESTION. It is the wrong question when both arms are the same graph -- a
# shared bug is invisible to it. These two are not that: the control arm in
# each is the path that has ALWAYS applied the factor and has evidence behind
# it (the single matvec, decode, shipped since 09-07), and the arm under test
# is a new place to apply the same factor. So "identical" means the new place
# computes what the old one does, which is exactly the claim being made.
#
# ⚠⚠ AND AN ARM THAT NEVER RAN IS NOT A NULL RESULT. Every identity here can
# pass vacuously: if the batched path refuses for some OTHER reason, arm B is
# arm A and the tokens match for no reason at all. So each identity arm also
# checks a positive tell that the path it is about actually ran, and says so
# when it did not. The tells:
#
#   batch   the batched matmul entry is non-zero in the stage report AND the
#           "applied a row at a time by request" refusal appears in arm A's
#           diagnostics and NOT in arm B's
#   share   "does not share a packed input unless CHARSIU_NPU_AWQ_SHARE=1"
#           appears in arm A's diagnostics and not in arm B's
#
#   usage: tests/board_awq.sh MODEL.gguf [CALIB.txt]
#
# The model must be one AWQ can help: four bits (CHARSIU_NPU_W4V=1) and a
# calibration file beside it, or one recorded here.
set -u
D=$(cd "$(dirname "$0")" && pwd)
M=${1:?usage: board_awq.sh MODEL.gguf [CALIB.txt]}
C=${2:-$D/corpus/calib.txt}
E=$D/corpus/long.txt

RUN=${CHARSIU_RUN_BIN:-}
PPL=${CHARSIU_PPL_BIN:-}
for c in "$D/../build/charsiu_run" "$D/charsiu_run" /opt/charsiu/charsiu_run; do
	[ -z "$RUN" ] && [ -x "$c" ] && RUN=$c
done
for c in "$D/../build/charsiu_ppl" "$D/charsiu_ppl" /opt/charsiu/charsiu_ppl; do
	[ -z "$PPL" ] && [ -x "$c" ] && PPL=$c
done
[ -x "${RUN:-/nonexistent}" ] || { echo "no charsiu_run"; exit 2; }
[ -x "${PPL:-/nonexistent}" ] || { echo "no charsiu_ppl"; exit 2; }
[ -r "$C" ] || { echo "no calibration text at $C"; exit 2; }

T=${TMPDIR:-/tmp}/charsiu-awq.$$
mkdir -p "$T"
trap 'rm -rf "$T"' EXIT

# ⚠ THE GOVERNOR. Every timing below is CPU work as much as hardware work and
# the board ships ondemand; three phase 9 runs disagreed by more than the
# change they were measuring because of it. Put it back on the way out.
OLD=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "")
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
	[ -w "$g" ] && echo performance > "$g" 2>/dev/null
done
trap 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do [ -n "$OLD" ] && [ -w "$g" ] && echo "$OLD" > "$g" 2>/dev/null; done; rm -rf "$T"' EXIT

#
# ⚠ CHARSIU_AWQ_BASE EXISTS SO THIS SCRIPT CAN BE RUN BEFORE THE BOARD IS.
# Setting it to "CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1" runs
# every arm against the host CPU reference on the SAME quantised weights:
# arms 3 and 4 are quality questions and that is a complete answer to them,
# arms 1 and 2 are about the hardware paths and will correctly report
# themselves vacuous. A script that has never been run is not a board round,
# it is a plan for one, and this tree has lost a round to a dead loop before.
#
W4=${CHARSIU_AWQ_BASE:-"CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1"}
# ⚠ 0.15 IS MEASURED, NOT CONVENTIONAL. On tests/corpus at 300 tokens,
# Llama-3.2-1B: off 41.5289, 0.10 33.7566, 0.15 32.5094, 0.20 35.2041. It is
# also inside the vendor's own per-tensor range (0.03-0.15 typical, 0.401
# max), which is a second line of evidence for the same place. Arm 4 sweeps it
# anyway, because the optimum is per model.
ALPHA=${CHARSIU_AWQ_ALPHA:-0.15}
NTOK=${CHARSIU_AWQ_NTOK:-32}
# long enough that the prompt is BATCHED, which is the whole subject of arm 1
PROMPT=${CHARSIU_AWQ_PROMPT:-"The keeper of the lighthouse wrote down the barometer and the wind every morning for eleven years, and what he remembered afterwards was not the storms but the particular quality of the light in the hour before one arrived. Explain, in plain words, why a written record outlasts a memory:"}

STATS=$T/awq.stats
echo "== recording calibration statistics"
env $W4 CHARSIU_NPU=0 CHARSIU_CALIB="$STATS" \
	"$PPL" "$M" "$C" -n 150 >/dev/null 2>&1
[ -s "$STATS" ] || { echo "FAIL: the calibration pass wrote nothing"; exit 1; }
echo "   $(wc -c < "$STATS") bytes"

#
# ⚠⚠ THE SUMMARY IS ON STDOUT AND IT DOES NOT SAY "TTFT". charsiu_run prints
#
#   [load N ms | staging N ms | prompt N tok in N ms, N tok/s
#                             | gen N tok in N ms, N tok/s | peak N MB]
#
# on STDOUT, next to the generated text; the refusals and the stage report go
# to stderr. The first version of this read timings out of the .err file and
# grepped for "TTFT", so every timing it printed was blank -- which the host
# dry run showed, and which on the board would have been a round that measured
# nothing and said so in a column of empty fields.
#
# And `head -1` on "tok/s" takes the PROMPT rate, not the decode one. They are
# different quantities wearing the same unit.
#
text()      { sed '/^\[load /,$d'; }
prompt_ms() { grep -o 'prompt [0-9]* tok in [0-9.]* ms' | head -1 | awk '{print $5}'; }
gen_tps()   { grep -o 'gen [0-9]* tok in [0-9.]* ms, [0-9.]* tok/s' | head -1 | awk '{print $7}'; }

# run one arm: $1 extra env, writes $T/$2.out and $T/$2.err
arm() {
	_e=$1; _n=$2
	# shellcheck disable=SC2086
	env $W4 CHARSIU_NPU_AWQ="$ALPHA" CHARSIU_AWQ_STATS="$STATS" \
		CHARSIU_STAGES=1 $_e \
		"$RUN" "$M" -p "$PROMPT" -n "$NTOK" \
		> "$T/$_n.out" 2> "$T/$_n.err"
}

fail=0
say_same() {
	if [ "$1" = "$2" ]; then echo "   tokens IDENTICAL"; else
		echo "   ⚠⚠ TOKENS DIFFER"
		printf '      A: %.90s\n      B: %.90s\n' "$1" "$2"
		fail=$((fail + 1))
	fi
}

echo
echo "== arm 1: AWQ's factor on the batched path"
arm "CHARSIU_NPU_AWQ_BATCH=0" b0
arm "CHARSIU_NPU_AWQ_BATCH=1" b1
A=$(text < "$T/b0.out"); B=$(text < "$T/b1.out")
say_same "$A" "$B"
# the tell: A refused and B did not
if ! grep -q "applied a row at a time by request" "$T/b0.err"; then
	echo "   ⚠⚠ arm A never refused -- this identity proved nothing"
	echo "      (no kscale on any tensor? AWQ off? then the whole arm is vacuous)"
	fail=$((fail + 1))
fi
if grep -q "applied a row at a time by request" "$T/b1.err"; then
	echo "   ⚠⚠ arm B refused too -- the batched path did not run"
	fail=$((fail + 1))
fi
if ! grep -q "charsiu NPU batched" "$T/b1.err"; then
	echo "   ⚠ no batched matmul entry in arm B's report -- nothing was batched"
	fail=$((fail + 1))
fi
printf '   prompt   refuse %s ms   batch %s ms\n' \
	"$(prompt_ms < "$T/b0.out")" "$(prompt_ms < "$T/b1.out")"
grep -h "a different AWQ factor" "$T/b1.err" | sed 's/^/   reuse: /'

echo
echo "== arm 2: one packed input shared by a group"
arm "CHARSIU_NPU_AWQ_SHARE=0" s0
arm "CHARSIU_NPU_AWQ_SHARE=1" s1
A=$(text < "$T/s0.out"); B=$(text < "$T/s1.out")
say_same "$A" "$B"
# ⚠ THE TELL, because identical tokens is what BOTH "the knob works" and "the
# knob never ran" look like. npudev whines this only where a factored tensor
# meets the group path with sharing off.
SH="does not share a packed input unless"
if ! grep -q "$SH" "$T/s0.err"; then
	echo "   ⚠⚠ arm A never refused to share -- this identity proved nothing"
	fail=$((fail + 1))
fi
if grep -q "$SH" "$T/s1.err"; then
	echo "   ⚠⚠ arm B refused to share too -- the knob did not take effect"
	fail=$((fail + 1))
fi
grep -h "cannot share one packed input" "$T/s1.err" | sed 's/^/   /'
printf '   decode %s tok/s (no share)   %s tok/s (shared)\n' \
	"$(gen_tps < "$T/s0.out")" "$(gen_tps < "$T/s1.out")"

echo
# ⚠ AT WHATEVER CHARSIU_AWQ_ALPHA IS, WHICH ARM 4 IS WHAT CHOOSES. The two
# arms are in this order because arm 3 only needs the comparison to be
# internally consistent, but a LAYERS number quoted without its alpha is not
# comparable to anything -- Llama's optimum measures 0.15 on the tree corpus,
# not the 0.20 the README used to name, and the whole surface moves with it.
echo "== arm 3: AWQ on the first three blocks only, at alpha $ALPHA"
for L in "" "0-2"; do
	# shellcheck disable=SC2086
	P=$(env $W4 CHARSIU_NPU_AWQ="$ALPHA" CHARSIU_AWQ_STATS="$STATS" \
		${L:+CHARSIU_NPU_AWQ_LAYERS=$L} \
		"$PPL" "$M" "$E" -n 300 2>/dev/null | tail -1 |
		grep -o 'ppl [0-9.]*' | awk '{print $2}')
	printf '   AWQ_LAYERS=%-6s ppl %s\n' "${L:-all}" "${P:-?}"
done

echo
echo "== arm 4: the exponent, on this board's own quantiser"
for a in 0.00 0.05 0.10 0.15 0.20 0.35 0.50; do
	# shellcheck disable=SC2086
	# ⚠ alpha 0.00 IS "AWQ OFF", not "AWQ at zero". The knob's absence is
	# the control arm; setting it to 0 still walks the factor code with a
	# vector of ones and rounds differently, which is a third thing.
	[ "$a" = 0.00 ] && A="" || A="CHARSIU_NPU_AWQ=$a"
	P=$(env $W4 CHARSIU_AWQ_STATS="$STATS" $A \
		"$PPL" "$M" "$E" -n 300 2>/dev/null | tail -1 |
		grep -o 'ppl [0-9.]*' | awk '{print $2}')
	printf '   alpha %s   ppl %s\n' "$a" "${P:-?}"
done
echo
echo "⚠ THE SURFACE MUST BE MONOTONE ON EACH SIDE OF ITS MINIMUM. A grid that
   is not is a grid that cannot rank its own cells: re-run the two best at
   -n 500 before believing either."

echo
echo "board_awq: $fail identity checks failed"
exit "$fail"
