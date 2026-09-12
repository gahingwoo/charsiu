#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# The other half of the withdrawn rule: how long a prompt does fp16 attention
# on the NPU need before it pays?
#
# ⚠⚠ THE KNOB EXISTS, IT IS CORRECT, AND IT IS OFF FOR A REASON THAT IS NOW
# ANSWERABLE. attn_npu_want_for() says so itself:
#
#   CHARSIU_ATTN_NPU=auto is withdrawn -- head_dim >= 128 is only half the
#   rule and the other half (how long a prompt) has not been measured;
#   attention stays on the CPU
#
# r393 is why it is worth measuring now. Fitting both runtimes' TTFT curves put
# charsiu's quadratic term at 0.004483 ms/tok^2 against the vendor's 0.000743 --
# six times theirs -- and the quadratic term IS attention. The batched stage
# table agrees from the other direction: attention is 42.1% of an 852 token
# prompt, measured, against 40.2% inferred from the fit.
#
# So the question is not whether attention is worth attacking. It is whether
# this arm, which already exists and already agrees with the CPU to fp16
# rounding, is faster -- and from what length.
#
# ⚠ CORRECTNESS IS NOT ASSUMED HERE EITHER. CHARSIU_ATTN_NPU_CHECK=1 runs both
# arms on the same rows and prints the worst disagreement per layer; on
# Llama-3.2-1B at 202 tokens that is 0.0024 relative and the generated text is
# byte-identical. This script re-checks it at the LONGEST length it sweeps,
# because a fp16 accumulator's error grows with how much it accumulates.
#
# Same discipline as board_attn_block.sh: one binary, one session, the clock
# pinned, arms ALTERNATING at each length, repeats so the spread sits beside
# the difference.
#
#   CHARSIU_ATTN_REPS="2 4 8 18 34"   clause counts (about 25 tokens each)
#   CHARSIU_ATTN_N=2                  repeats an arm a length
set -u

REPS=${CHARSIU_ATTN_REPS:-2 4 8 18 34}
N=${CHARSIU_ATTN_N:-2}
RUN=${CHARSIU_RUN:-/root/charsiu_run_maxn}
M=${CHARSIU_MODEL:-/opt/charsiu/models/Llama-3.2-1B-Instruct-Q4_0.gguf}
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"
CLAUSE="A written record can be copied, checked against other records, and read by someone who never met the person who wrote it. "

[ -e /dev/accel/accel0 ] || { echo "no /dev/accel -- this needs the rocket arm"; exit 1; }

for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo userspace > "$p/scaling_governor" 2>/dev/null || true
	tr ' ' '\n' < "$p/scaling_available_frequencies" | grep -v '^$' | sort -n \
		| tail -1 > "$p/scaling_setspeed" 2>/dev/null || true
done
sleep 1

echo "== fp16 attention on the NPU, against the CPU arm, by prompt length"
echo "   boot      $(cat /proc/sys/kernel/random/boot_id)"
echo "   npu clk   $(cat /sys/kernel/debug/clk/clk_rknn_dsu0/clk_rate 2>/dev/null) Hz"
echo "   cpu       $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq)/$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq) kHz"
echo "   binary    $RUN  $(ls -l --full-time "$RUN" 2>/dev/null | awk '{print $6}')"
echo "   model     $(basename "$M")"
echo "   $N repeats an arm a length, arms alternating, one warm-up discarded"
echo

mid() { printf '%s\n' $1 | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$0} END{if(NR%2)printf "%s\n",a[(NR+1)/2]; else printf "%.0f\n",(a[NR/2]+a[NR/2+1])/2}'; }
lo()  { printf '%s\n' $1 | tr ' ' '\n' | grep -v '^$' | sort -n | head -1; }
hi()  { printf '%s\n' $1 | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1; }

ttft() { # $1 = extra env, $2 = prompt
	env $E $1 "$RUN" "$M" -p "$2" -n 1 --ignore-eos -c 1024 -t 4 2>/dev/null \
		| grep '^\[load' | sed 's/.*prompt \([0-9]*\) tok in \([0-9]*\) ms.*/\1 \2/'
}

printf '   %7s  %11s %13s  %11s %13s   %s\n' tokens 'CPU ms' range 'NPU ms' range 'NPU/CPU'
printf '   %7s  %11s %13s  %11s %13s   %s\n' ------ ------ ----- ------ ----- -------

LAST=""
for R in $REPS; do
	P=""; i=0
	while [ $i -lt "$R" ]; do P="$P$CLAUSE"; i=$((i+1)); done
	# one warm-up an arm, discarded
	ttft "" "$P" >/dev/null 2>&1
	ttft "CHARSIU_ATTN_NPU=1" "$P" >/dev/null 2>&1
	A=; B=; TOK=; n=0
	while [ $n -lt "$N" ]; do
		n=$((n+1))
		# ⚠ ALTERNATE. All of A then all of B measures the board warming
		# up as much as it measures the arms.
		r=$(ttft "" "$P"); TOK=${r%% *}; A="$A ${r##* }"
		r=$(ttft "CHARSIU_ATTN_NPU=1" "$P"); B="$B ${r##* }"
	done
	[ -n "$A" ] && [ -n "$B" ] || { echo "   $R clauses: NO OUTPUT"; continue; }
	MA=$(mid "$A"); MB=$(mid "$B")
	RAT=$(awk "BEGIN{printf \"%.3f\", $MB/$MA}")
	printf '   %7s  %11s %5s..%-5s  %11s %5s..%-5s   %s\n' \
		"$TOK" "$MA" "$(lo "$A")" "$(hi "$A")" "$MB" "$(lo "$B")" "$(hi "$B")" "$RAT"
	LAST="$P"
done

echo
echo "== correctness at the longest length, both arms on the same rows"
env $E CHARSIU_ATTN_NPU=1 CHARSIU_ATTN_NPU_CHECK=1 "$RUN" "$M" -p "$LAST" \
	-n 1 --ignore-eos -c 1024 -t 4 2>&1 >/dev/null \
	| grep 'fp16 attention' | awk '{print "   " $0}' | tail -4

echo
echo "restoring schedutil"
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil > "$p/scaling_governor" 2>/dev/null || true; done
