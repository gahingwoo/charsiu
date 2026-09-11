#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# WHAT HAS TWO POSITIONS? The decode rate on this board is BIMODAL, and that
# is where the project's headline came from.
#
# 2026-09-11, one boot, performance governor, seven runs a model, every reading
# printed for the first time:
#
#   Phi3   7.04  5.96  5.98  5.96  5.96  5.96  7.04
#
# -- the same two values exactly, five low and two high, and the same 5/2 split
# on all four models. best-of-N lands on the high cluster every time, which
# reproduces the +5.8% / +16.1% over the vendor that is on record. On the
# median every model is BEHIND. So what switches is now the most valuable
# question here, and the ratios say where to look first:
#
#   Qwen3 1.26   TinyLLAMA 1.18   Phi3 1.18   Gemma4 1.28
#
# RK3576 is four A72 and four A53. That is the shape of a thread landing on one
# cluster or the other, so the first arm is affinity.
#
# ⚠⚠ THE ARMS ALTERNATE, THEY ARE NOT RUN IN BLOCKS. A board that warms over
# ten minutes gives its last block the worst numbers whatever the arm, and this
# tree has read that as a property of the arm before. One pass runs every arm
# once, in order, and the passes repeat.
#
# ⚠ AND THE FIRST PASS IS DISCARDED. The first run of anything here is cold --
# page cache, the NPU's own clocks, the governor settling -- and a cold first
# point has been read as a 38% win in this tree before.
#
#   sh tests/board_bimodal.sh [MODEL.gguf] [PASSES]
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
M=${1:-$(find_model 'Qwen3-0.6B-Q4_0.gguf' || true)}
[ -n "$M" ] && [ -f "$M" ] || { echo "no model"; exit 2; }
PASSES=${2:-8}
P="The keeper of the lighthouse wrote down the barometer and the wind every morning for eleven years. Explain in plain words why a written record outlasts a memory:"

#
# ⚠ READ THE TOPOLOGY, DO NOT ASSUME IT. "cpu0-3 is the little cluster" is true
# on most Rockchip parts and is not a fact about this one until it is read.
#
echo "== the cores, from the board rather than from memory"
for c in /sys/devices/system/cpu/cpu[0-9]*; do
	n=$(basename "$c")
	f=$(cat "$c/cpufreq/cpuinfo_max_freq" 2>/dev/null || echo '?')
	p=$(cat "$c/regs/identification/midr_el1" 2>/dev/null || echo '?')
	echo "   $n  max ${f} kHz   midr $p"
done
echo
BIG=$(for c in /sys/devices/system/cpu/cpu[0-9]*; do
	f=$(cat "$c/cpufreq/cpuinfo_max_freq" 2>/dev/null || echo 0)
	echo "$f $(basename "$c" | tr -dc 0-9)"; done | sort -rn | head -4 |
	awk '{printf "%s%s", (NR>1?",":""), $2}')
LIT=$(for c in /sys/devices/system/cpu/cpu[0-9]*; do
	f=$(cat "$c/cpufreq/cpuinfo_max_freq" 2>/dev/null || echo 0)
	echo "$f $(basename "$c" | tr -dc 0-9)"; done | sort -n | head -4 |
	awk '{printf "%s%s", (NR>1?",":""), $2}')
echo "   fast four: $BIG      slow four: $LIT"
command -v taskset >/dev/null || { echo "⛔ no taskset on this board"; exit 2; }
echo

#
# ⚠⚠ THE THREAD COUNT IS PINNED IN EVERY ARM. taskset does not change what
# sysconf(_SC_NPROCESSORS_ONLN) reports, so an unpinned run would start eight
# threads and a pinned one would start eight threads on four cores -- two
# differences at once, and the knob would not be affinity.
#
run_one() {   # $1 = label, $2 = taskset prefix or empty
	# shellcheck disable=SC2086
	env CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_THREADS=4 \
		$2 "$BIN/charsiu_run" "$M" -p "$P" -n 48 --ignore-eos -c 512 \
		2>/dev/null | grep -o 'gen [0-9]* tok in [0-9]* ms, [0-9.]* tok/s' |
		head -1 | awk '{print $7}'
}

echo "== $(basename "$M"), $PASSES passes, arms ALTERNATING, pass 1 discarded"
echo "   every arm runs CHARSIU_THREADS=4 so affinity is the only difference"
echo
ALL=; BG=; LT=
i=0
while [ $i -lt "$PASSES" ]; do
	i=$((i + 1))
	a=$(run_one all "")
	b=$(run_one big "taskset -c $BIG")
	c=$(run_one lit "taskset -c $LIT")
	printf '   pass %d   any4 %-7s big4 %-7s little4 %-7s%s\n' \
		"$i" "${a:-FAIL}" "${b:-FAIL}" "${c:-FAIL}" \
		"$([ $i = 1 ] && echo '   <- cold, discarded')"
	if [ $i -gt 1 ]; then
		ALL="$ALL $a"; BG="$BG $b"; LT="$LT $c"
	fi
done

echo
echo "   any four   $ALL"
echo "   big four   $BG"
echo "   little4    $LT"
echo
#
# ⚠ WHAT WOULD MAKE THIS RED. If all three arms stay bimodal, affinity is not
# the switch and the ratio was a coincidence of scale. If the pinned arms are
# each UNIMODAL and sit at the two levels, it is the scheduler, and the fix is
# a policy rather than a finding. Say which before reading the numbers.
#
echo "⚠ read it as: pinned arms UNIMODAL and at the two levels  -> it is affinity"
echo "              all three still bimodal                     -> it is not, look elsewhere"
echo "              pinned arms unimodal but at the SAME level   -> it is neither cluster"
