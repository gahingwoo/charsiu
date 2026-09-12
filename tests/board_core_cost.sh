#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# WHAT ONE NPU CORE IS WORTH, at whatever clock and rail the board booted with.
#
# ⚠⚠ WHY THIS EXISTS. v12's 13/14 argued "losing 192 MHz costs less than
# losing a core" from a single row that compared "one core at 786 MHz" against
# "both cores at 594". Three things were wrong with it and all three are the
# same mistake, which is putting two variables in one cell:
#
#   - the baseline was not one core. 1037/1565/5073/3604 is the SERIAL default
#     (two cores, never at once) -- npudev.c says so in as many words. A real
#     one-core arm is far slower: phi3 decode 9.71 tok/s against 14.70.
#   - the decode figures beside it, 24.3/20.3/6.8/8.7, were taken on 2026-09-04
#     and RETRACTED by this project on 09-11: they predate the affinity pin, so
#     what they measure is the cost of not pinning, not the cost of anything
#     else.
#   - every figure was the minimum of three runs with the spread dropped, which
#     is the same best-of-N this project spent 09-11 removing from the paper.
#
# So the question is split into the two quantities it was always made of, and
# this measures the one that needs no reflash:
#
#   what a CORE is worth   = one core vs two, at THIS clock       <- here
#   what 192 MHz is worth  = two cores at 594 vs two at 786       <- needs
#                                                                    the other
#                                                                    device tree
#
# ⚠ BOTH ARMS NAME THE KNOB. ("", "ONEDEV=1") is not two arms, it is the
# default against one value, and the default is not written down anywhere the
# round can print.
#
# ⚠ THE FIRST READING OF A ROUND IS COLD. One warm-up per model is taken and
# thrown away, and the arms ALTERNATE rather than running one after the other,
# because a drift across the round otherwise lands entirely on the second arm.
#
# ⚠⚠ EVERY READING IS PRINTED. The median and the full range are the report;
# a minimum is not. A round whose individual readings were discarded cannot be
# re-read later with a different statistic.
#
#   sh board_core_cost.sh [MODEL-substring ...]   (default: the four 13/14 used)
#
# ⚠ ONE MODEL AN INVOCATION IS FINE AND IS HOW THIS GETS DRIVEN OVER A SERIAL
# LINE. Every run reloads the model, so four models times five repeats times
# two arms does not fit in one command's timeout. What must NOT be split is the
# two arms: they alternate inside a model so that a drift across the round does
# not land on one of them.
#
set -u

D=$(dirname "$0")
REPEAT=${CHARSIU_CC_REPEAT:-5}
NTOK=${CHARSIU_CC_NTOK:-64}
PROMPT=${CHARSIU_CC_PROMPT:-"Explain in plain words why a written record outlasts a memory."}

RUN=
for d in "$D/../build" /opt/charsiu/bin /usr/local/bin /usr/bin; do
	[ -x "$d/charsiu_run" ] && { RUN="$d/charsiu_run"; break; }
done
[ -n "$RUN" ] || { echo "charsiu_run not found" >&2; exit 1; }

MODELDIRS="$HOME/.charsiu/models /opt/charsiu/models $D/../models"
find_model() {
	for d in $MODELDIRS; do
		[ -d "$d" ] || continue
		f=$(ls "$d" 2>/dev/null | grep -i "$1" | grep -v mmproj | head -1)
		[ -n "$f" ] && { echo "$d/$f"; return 0; }
	done
	return 1
}

#
# ⚠⚠ THE CONDITIONS ARE PART OF THE ANSWER AND THE ROUND PRINTS THEM. 13/14
# quoted numbers whose clock and rail lived only in the prose around them, and
# the prose was wrong about both. A reading here carries its own conditions.
#
echo "== what one NPU core is worth"
echo "   boot id    $(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
R=$(cat /sys/kernel/debug/clk/clk_rknn_dsu0/clk_rate 2>/dev/null)
echo "   dsu0       ${R:-unknown} Hz"
V=$(awk '/vdd_npu_s0/{print $6; exit}' /sys/kernel/debug/regulator/regulator_summary 2>/dev/null)
echo "   vdd_npu    ${V:-unknown}"
echo "   cores      $(ls /dev/accel/ 2>/dev/null | tr '\n' ' ')"
echo "   governor   $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
echo "   repeats    $REPEAT per arm, alternating, one warm-up discarded"
echo

mid() {
	printf '%s\n' $1 | sort -g |
		awk '{a[NR]=$0}
		     END{ if (NR == 0) exit
		          if (NR % 2) printf "%g\n", a[(NR+1)/2]
		          else         printf "%g\n", (a[NR/2] + a[NR/2+1]) / 2 }'
}
lo() { printf '%s\n' $1 | sort -g | head -1; }
hi() { printf '%s\n' $1 | sort -g | tail -1; }

# one reading; $1 is the ONEDEV value, $2 the model path
one() {
	env CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
	    CHARSIU_NPU_ONEDEV="$1" \
	    CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 \
	    "$RUN" "$2" -p "$PROMPT" -n "$NTOK" --ignore-eos -c 512 -t 4 \
	    2>/dev/null | grep '^\[load'
}

WANT=${*:-"qwen3 tinyllama phi-3.5 gemma-4"}
for want in $WANT; do
	M=$(find_model "$want") || { echo "-- $want: NOT FOUND under [$MODELDIRS]"; echo; continue; }
	echo "-- $(basename "$M")"

	# ⚠ discarded on purpose, and the round says so rather than folding it in
	one 0 "$M" >/dev/null 2>&1

	T2=; S2=; T1=; S1=; n=0
	while [ "$n" -lt "$REPEAT" ]; do
		n=$((n + 1))
		for arm in 0 1; do
			O=$(one "$arm" "$M") || O=""
			if [ -z "$O" ]; then
				echo "   run $n arm ONEDEV=$arm: NO OUTPUT"
				continue
			fi
			TT=$(echo "$O" | sed 's/.*prompt [0-9]* tok in \([0-9]*\) ms.*/\1/')
			TS=$(echo "$O" | sed 's/.*gen [0-9]* tok in [0-9]* ms, \([0-9.]*\) tok.*/\1/')
			if [ "$arm" = 0 ]; then T2="$T2 $TT"; S2="$S2 $TS"
			else                    T1="$T1 $TT"; S1="$S1 $TS"; fi
		done
	done

	printf '   %-22s %9s %9s   %s\n' "arm" "TTFT ms" "tok/s" "every reading"
	printf '   %-22s %9s %9s   TTFT:%s\n' "ONEDEV=0 (two cores)" \
		"$(mid "$T2")" "$(mid "$S2")" "$T2"
	printf '   %-22s %9s %9s   tok/s:%s\n' "" "" "" "$S2"
	printf '   %-22s %9s %9s   TTFT:%s\n' "ONEDEV=1 (one core)" \
		"$(mid "$T1")" "$(mid "$S1")" "$T1"
	printf '   %-22s %9s %9s   tok/s:%s\n' "" "" "" "$S1"
	printf '   %-22s TTFT %s..%s and %s..%s, tok/s %s..%s and %s..%s\n' "range" \
		"$(lo "$T2")" "$(hi "$T2")" "$(lo "$T1")" "$(hi "$T1")" \
		"$(lo "$S2")" "$(hi "$S2")" "$(lo "$S1")" "$(hi "$S1")"
	echo
done

echo "⚠ This is what a CORE is worth at the clock and rail printed above, and"
echo "  nothing else. What 192 MHz is worth needs the other device tree, and"
echo "  the two must not be quoted as one comparison."
