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
PASSES=${2:-10}
#
# ⚠⚠ THE INVOCATION IS board_vendor.sh's, VERBATIM, AND THAT IS THE POINT.
# The first version of this probe used its own prompt, -n 48, and neither of
# the two environment variables the scoreboard sets -- so it changed three
# things away from the configuration where the bimodality was SEEN, and then
# reported levels that did not match it (19.7/11.4 against the observed
# 20.5/26.1). An arm that is supposed to explain an observation has to start
# from that observation's configuration and move ONE thing.
#
# Taken from tests/board_vendor.sh: the 128-token protocol prompt, -n 64,
# --ignore-eos, -c 512, -t 4, MAXN and COEF_ELEMS. -t 4 beats CHARSIU_THREADS
# because llama.c's pool_start only consults the environment when the flag is
# absent, so the scoreboard has always run FOUR threads.
#
P="The history of computing begins long before the first electronic machine. Merchants kept accounts on clay, astronomers ruled tables by hand, and the abacus moved beads along a wire for two thousand years before anyone thought to make the beads move themselves. What changed was not arithmetic but who did it: a machine that could be told the order of operations once and would then repeat them without tiring, without a wage, and without the small drift of attention that makes a long column of figures a gamble. Explain, in plain words, why that shift mattered more than the speed:"

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
	env CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
		CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 \
		$2 "$BIN/charsiu_run" "$M" -p "$P" -n 64 --ignore-eos -c 512 -t 4 \
		2>/dev/null | grep -o 'gen [0-9]* tok in [0-9]* ms, [0-9.]* tok/s' |
		head -1 | awk '{print $7}'
}

echo "== $(basename "$M"), $PASSES passes, arms ALTERNATING, pass 1 discarded"
echo "   board_vendor.sh's own invocation; -t 4, so four threads in every arm"
echo "   ⚠ the DEFAULT arm is the scoreboard exactly: no taskset at all"
echo
DEF=; ALL8=; BG=; LT=
i=0
while [ $i -lt "$PASSES" ]; do
	i=$((i + 1))
	d=$(run_one def "")
	e=$(run_one all8 "taskset -c 0-7")
	b=$(run_one big "taskset -c $BIG")
	c=$(run_one lit "taskset -c $LIT")
	printf '   pass %2d   default %-7s all8 %-7s big4 %-7s little4 %-7s%s\n' \
		"$i" "${d:-FAIL}" "${e:-FAIL}" "${b:-FAIL}" "${c:-FAIL}" \
		"$([ $i = 1 ] && echo '  <- cold, discarded')"
	if [ $i -gt 1 ]; then
		DEF="$DEF $d"; ALL8="$ALL8 $e"; BG="$BG $b"; LT="$LT $c"
	fi
done

echo
echo "   default    $DEF"
echo "   all eight  $ALL8"
echo "   big four   $BG"
echo "   little4    $LT"
echo
#
# ⚠ WHAT WOULD MAKE THIS RED. If all three arms stay bimodal, affinity is not
# the switch and the ratio was a coincidence of scale. If the pinned arms are
# each UNIMODAL and sit at the two levels, it is the scheduler, and the fix is
# a policy rather than a finding. Say which before reading the numbers.
#
#
# ⚠⚠ THE FIRST THING TO CHECK IS THAT THE DEFAULT ARM REPRODUCED THE THING.
# If `default` comes back unimodal, this round did not observe the effect at
# all and NOTHING below it can be read -- not as a confirmation and not as a
# refutation. A probe that cannot see the phenomenon cannot rule on its cause.
#
echo "⚠ FIRST: is the DEFAULT arm bimodal? If not, this round saw nothing and"
echo "  none of the other arms mean anything either way."
echo
echo "  then: pinned arms unimodal, default bimodal   -> placement IS the switch"
echo "        every arm bimodal, including all8       -> not placement; the two"
echo "                                                   cores, the NPU clock or"
echo "                                                   the governor's own steps"
echo "        all8 unimodal but default bimodal       -> the scheduler's choice,"
echo "                                                   not the cores themselves"
