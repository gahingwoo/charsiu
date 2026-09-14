#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# Is a one-chunk prefill width that is a multiple of 16 cheaper than its
# neighbours?
#
# ⚠⚠ WHERE THE QUESTION CAME FROM. The one-chunk widening wins 5.6% on
# Qwen3-0.6B and loses ~7% on Llama-3.2-1B and tinyllama, at the same prompt
# length and with clean controls. The widths it produces are 1x112 on Qwen3
# and 1x114 on the other two. 112 is 16 x 7 and 114 is not a multiple of 16,
# and the FEATURE ATOM on this hardware is 16 -- the rule the mesa work landed
# on for conv input channels.
#
# So this sweeps the width itself rather than the knob. The prompt is built to
# an EXACT token count: "A record." is 4 tokens and " cat" is exactly 1, so
# 4 + N gives any length, and inside 81..160 the default runs it as one chunk
# of that width. A dip at 112, 128 and 144 against a smooth rise is the atom;
# no dip is the atom not applying here.
#
# ⚠ THE READOUT IS THE SECOND DIFFERENCE, not the raw time. TTFT rises with
# length whatever the width does, so what a 16-wide periodicity looks like is
# a dip in ms-per-token, and the eye is bad at that on a rising line.
set -u

# ⚠ SOURCED HERE AND NOT FURTHER DOWN: board_clk.sh is what makes
# CHARSIU_RUN and CHARSIU_RUN_BIN two names for one knob, and this
# script picks its binary below. Sourcing it after that point set the
# alias too late to be read -- which is how round 414 measured the
# INSTALLED binary for twenty minutes while believing otherwise.
. "$(dirname "$0")/board_clk.sh"

N=${ATOM_N:-3}
RUN=${CHARSIU_RUN:-/root/charsiu_run_lhd}
M=${CHARSIU_MODEL:-/opt/charsiu/models/Llama-3.2-1B-Instruct-Q4_0.gguf}
LENS=${ATOM_LENS:-106 108 110 112 114 116 118 120 122 124 126 128 130 132 134 136 138 140 142 144 146 148 150}
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"

[ -e /dev/accel/accel0 ] || { echo "no /dev/accel"; exit 1; }
for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo userspace > "$p/scaling_governor" 2>/dev/null || true
	tr ' ' '\n' < "$p/scaling_available_frequencies" | grep -v '^$' | sort -n \
		| tail -1 > "$p/scaling_setspeed" 2>/dev/null || true
done
sleep 1

echo "== one-chunk prefill width against the 16 atom"
echo "   boot   $(cat /proc/sys/kernel/random/boot_id)"
echo "   model  $(basename "$M")   binary $RUN"
# ⚠ SOURCED FOR charsiu_build ONLY, and npu_clk is deliberately NOT called:
# it refuses when debugfs is unmounted, and turning this probe into one that
# refuses to start is a different change from making it say which build
# produced its numbers.
echo "   build  $(charsiu_build "$RUN")"
echo "   $N repeats a length, one warm-up discarded, clock pinned"
echo

mid() { printf '%s\n' $1 | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$0} END{if(NR%2)printf "%s\n",a[(NR+1)/2]; else printf "%.0f\n",(a[NR/2]+a[NR/2+1])/2}'; }

printf '   %6s %6s  %8s %10s  %7s %s\n' want got 'TTFT ms' 'ms/tok' 'd(ms)' 'width'
printf '   %6s %6s  %8s %10s  %7s %s\n' ---- --- ------- ------ ----- -----
PREV=; PT=
for T in $LENS; do
	K=$((T - 4))
	P="A record."
	i=0; while [ $i -lt $K ]; do P="$P cat"; i=$((i+1)); done
	W=$(env $E "$RUN" "$M" -p "$P" -n 1 --ignore-eos -c 1024 -t 4 2>&1 >/dev/null \
	    | grep -oE 'widths [0-9x+]+' | head -1 | sed 's/widths //')
	env $E "$RUN" "$M" -p "$P" -n 1 --ignore-eos -c 1024 -t 4 >/dev/null 2>&1
	R=; G=; n=0
	while [ $n -lt "$N" ]; do
		n=$((n+1))
		O=$(env $E "$RUN" "$M" -p "$P" -n 1 --ignore-eos -c 1024 -t 4 2>/dev/null | grep '^\[load')
		[ -n "$O" ] || continue
		G=$(echo "$O" | sed 's/.*prompt \([0-9]*\) tok in.*/\1/')
		R="$R $(echo "$O" | sed 's/.*prompt [0-9]* tok in \([0-9]*\) ms.*/\1/')"
	done
	[ -n "$R" ] || { echo "   $T: NO OUTPUT"; continue; }
	MT=$(mid "$R")
	D=-
	# ⚠ TWO ROWS CAN TIE. $G is what the prompt TOKENISED to, not what was
	# asked for, and the whole point of this sweep is that it quantises --
	# so consecutive rows share a $G routinely and the slope's denominator
	# is zero. mawk prints "+inf" into the column and busybox awk fails the
	# whole substitution, leaving it BLANK, which reads as a measured zero.
	# board_ttft_curve.sh guards the same division on the same quantity.
	[ -n "$PREV" ] && [ "$G" != "$PREV" ] && \
		D=$(awk "BEGIN{printf \"%+.1f\", ($MT-$PT)/(($G)-($PREV))*2}")
	# ⚠ mark the multiples of 16 so the eye does not have to find them
	MARK=""; [ $((G % 16)) = 0 ] && MARK="  <- 16 x $((G / 16))"
	printf '   %6s %6s  %8s %10s  %7s %s%s\n' "$T" "$G" "$MT" \
		"$(awk "BEGIN{printf \"%.3f\", $MT/$G}")" "$D" "$W" "$MARK"
	PREV=$G; PT=$MT
done
echo
echo "d(ms) is the per-token cost of the two tokens since the previous row."
echo "restoring schedutil"
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil > "$p/scaling_governor" 2>/dev/null || true; done
