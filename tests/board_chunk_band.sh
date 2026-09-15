#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# Where inside its own band does the one-chunk widening pay?
#
# THE KNOB HAS A NARROW DOMAIN AND NOBODY HAD NAMED IT. charsiu_run runs a
# nominal chunk of 80 and onechunk_on() widens the whole prompt into ONE chunk
# when `n_ids > chunk && n_ids <= cap`. cap is 160 for Llama-3.2-1B at KMAX
# 1024. So the widening only ever engages for prompts of 81..160 tokens:
# shorter is one chunk anyway and longer is refused. Every measurement of this
# knob so far has been a single point inside that band.
#
# It went in against 128 token prompts, where it won 5.1 and 5.7% on the two
# vendor-protocol models whose baselines repeat. Evidence pack 1h then measured
# 102 tokens and found the default 5.9% SLOWER than letting it run 1x80+1x22.
# Both are inside the band. So the knob is non-monotone in prompt length and
# two points cannot say where it turns.
#
# THE CONTROLS ARE OUTSIDE THE BAND, and they are the point of including
# them: below 80 and above the cap the knob cannot act, so the two arms must
# read the SAME. An arm that differs there is measuring the board, not the knob.
#
#   CHARSIU_BAND_REPS="..."   clause counts (the clause is about 6 tokens)
#   CHARSIU_BAND_N=3
. "$(dirname "$0")/board_clk.sh"
NPUCLK=$(npu_clk) || exit 1
set -u

REPS=${CHARSIU_BAND_REPS:-10 14 16 18 20 22 24 32}
# EVEN, so each arm runs first exactly as often as it runs second.
N=${CHARSIU_BAND_N:-4}
RUN=${CHARSIU_RUN:-/root/charsiu_run_lhd}
M=${CHARSIU_MODEL:-/opt/charsiu/models/Llama-3.2-1B-Instruct-Q4_0.gguf}
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"
CLAUSE="A record can be copied and checked. "

[ -e /dev/accel/accel0 ] || { echo "no /dev/accel -- this needs the rocket arm"; exit 1; }
for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo userspace > "$p/scaling_governor" 2>/dev/null || true
	tr ' ' '\n' < "$p/scaling_available_frequencies" | grep -v '^$' | sort -n \
		| tail -1 > "$p/scaling_setspeed" 2>/dev/null || true
done
sleep 1

echo "== the one-chunk widening, inside and outside its own band"
echo "   boot    $(cat /proc/sys/kernel/random/boot_id)"
echo "   npu     $NPUCLK Hz"
echo "   cpu     $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq)/$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq) kHz"
echo "   binary  $RUN  $(ls -l --full-time "$RUN" 2>/dev/null | awk '{print $6}')"
echo "   build     $(charsiu_build "$RUN")"
case $((N % 2)) in 1) echo "   N=$N is odd; the order bias does not cancel";; esac
echo "   $N repeats an arm a length, arms AND ORDER alternating, one warm-up discarded"
echo "   the band is 81..160 tokens; outside it the two arms must agree"
echo

mid() { printf '%s\n' $1 | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$0} END{if(NR%2)printf "%s\n",a[(NR+1)/2]; else printf "%.0f\n",(a[NR/2]+a[NR/2+1])/2}'; }

ttft() { env $E $1 "$RUN" "$M" -p "$2" -n 1 --ignore-eos -c 1024 -t 4 2>/dev/null \
	| grep '^\[load' | sed 's/.*prompt \([0-9]*\) tok in \([0-9]*\) ms.*/\1 \2/'; }
widths() { env $E $1 "$RUN" "$M" -p "$2" -n 1 --ignore-eos -c 1024 -t 4 2>&1 >/dev/null \
	| grep -oE 'widths [0-9x+]+' | head -1; }

printf '   %7s  %9s %9s %8s   %-16s %s\n' tokens 'onechunk' 'off' 'ratio' 'widths on' 'widths off'
printf '   %7s  %9s %9s %8s   %-16s %s\n' ------ -------- --- ----- --------- ----------
for R in $REPS; do
	P=""; i=0
	while [ $i -lt "$R" ]; do P="$P$CLAUSE"; i=$((i+1)); done
	WA=$(widths "" "$P"); WB=$(widths "CHARSIU_PREFILL_ONECHUNK=0" "$P")
	ttft "" "$P" >/dev/null 2>&1
	ttft "CHARSIU_PREFILL_ONECHUNK=0" "$P" >/dev/null 2>&1
	# THE ORDER ALTERNATES TOO, NOT JUST THE ARMS. The first draft ran
	# "on" then "off" inside every repeat, so the on arm always took the
	# colder of the pair. gemma-3-1b caught it: the knob never engages on
	# that model at any length here -- both arms ran identical widths --
	# and the two arms still differed by 1.2% at 82 tokens rising to 3.8%
	# at 274. That is the order, not the knob, and it was large enough to
	# account for most of what the first run called a loss.
	#
	# A control that MUST read 1.000 is the only reason this was visible.
	A=; B=; TOK=; n=0
	while [ $n -lt "$N" ]; do
		n=$((n+1))
		if [ $((n % 2)) = 1 ]; then
			r=$(ttft "" "$P"); TOK=${r%% *}; A="$A ${r##* }"
			r=$(ttft "CHARSIU_PREFILL_ONECHUNK=0" "$P"); B="$B ${r##* }"
		else
			r=$(ttft "CHARSIU_PREFILL_ONECHUNK=0" "$P"); B="$B ${r##* }"
			r=$(ttft "" "$P"); TOK=${r%% *}; A="$A ${r##* }"
		fi
	done
	[ -n "$A" ] && [ -n "$B" ] || { echo "   $R clauses: NO OUTPUT"; continue; }
	MA=$(mid "$A"); MB=$(mid "$B")
	printf '   %7s  %9s %9s %8s   %-16s %s\n' "$TOK" "$MA" "$MB" \
		"$(awk "BEGIN{printf \"%.3f\", $MA/$MB}")" \
		"$(echo "$WA" | sed 's/widths //')" "$(echo "$WB" | sed 's/widths //')"
done
echo
echo "ratio above 1.000 means the shipped default is the slower arm"
echo "restoring schedutil"
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil > "$p/scaling_governor" 2>/dev/null || true; done
