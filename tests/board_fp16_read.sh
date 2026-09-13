#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# What it costs to ask "did this op write nothing" by counting all of it.
#
# The readback poisons the output buffer with 0xdeadbeef and then checks
# whether the hardware wrote anything. It did that by summing the matches over
# EVERY output word and comparing the sum against the cell count -- a full
# scalar pass over the answer, on top of the memcpy that follows it. For the
# scores shape that is m * npad words an op and thirty two ops a group.
#
# ⚠⚠ AND THE READ SIDE WAS THE CHEAP HALF. The buffer is POISONED the same way
# before every submit -- another full scalar pass over the answer, in the one
# region of that function with no clock on it. At 852 tokens the readback was
# 581 ms and the poisoning 668, together more than the hardware`s own 915.
#
# One sentinel a row answers both questions instead: a job writes its whole
# output or none of it, so a row whose first word survived is a row that was
# not written. That is strictly more than the every-cell form could say -- it
# could only ever answer all or not-all, so a half written output was accepted
# in silence -- and it costs m words instead of m * n. tests/sentinel.c drives
# all three verdicts on the desk, because the hardware will not.
#
# CHARSIU_FP16_FULLSCAN=1 is the every-cell form on BOTH sides, =0 the row
# sentinel on both. Same outcome for every output the board has ever produced,
# so the control that matters is the text.
#
#   CHARSIU_READ_REPS="4 12 24 34"   clause counts (about 25 tokens each)
#   CHARSIU_READ_N=2                 repeats an arm a length
. "$(dirname "$0")/board_clk.sh"
NPUCLK=$(npu_clk) || exit 1
set -u

REPS=${CHARSIU_READ_REPS:-4 12 24 34}
N=${CHARSIU_READ_N:-2}
RUN=${CHARSIU_RUN:-/opt/charsiu/charsiu_run}
M=${CHARSIU_MODEL:-/opt/charsiu/models/Llama-3.2-1B-Instruct-Q4_0.gguf}
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 CHARSIU_ATTN_NPU=1 CHARSIU_STAGES=1"
CLAUSE="A written record can be copied, checked against other records, and read by someone who never met the person who wrote it. "
ERR=/tmp/fp16read.$$

[ -e /dev/accel/accel0 ] || { echo "no /dev/accel -- this needs the rocket arm"; exit 1; }
[ -x "$RUN" ] || { echo "no $RUN"; exit 1; }

for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo userspace > "$p/scaling_governor" 2>/dev/null || true
	tr ' ' '\n' < "$p/scaling_available_frequencies" | grep -v '^$' | sort -n \
		| tail -1 > "$p/scaling_setspeed" 2>/dev/null || true
done
sleep 1

echo "== the poison check: every cell, or one sentinel a row"
echo "   boot      $(cat /proc/sys/kernel/random/boot_id)"
echo "   npu clk   $NPUCLK Hz"
echo "   cpu       $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq)/$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq) kHz"
echo "   binary    $RUN  $(ls -l --full-time "$RUN" 2>/dev/null | awk '{print $6}')"
echo "   model     $(basename "$M")"
echo "   $N repeats an arm a length, arms alternating, one warm-up discarded"
echo

mid() { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$0} END{if(NR%2)printf "%s\n",a[(NR+1)/2]; else printf "%.0f\n",(a[NR/2]+a[NR/2+1])/2}'; }
lo()  { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | head -1; }
hi()  { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1; }

# $1 = arm, $2 = prompt.  prints: tokens ttft_ms read_ms
one() {
	t=$(env $E "CHARSIU_FP16_FULLSCAN=$1" "$RUN" "$M" -p "$2" -n 1 \
	    --ignore-eos -c 1024 -t 4 2>"$ERR" \
	    | grep '^\[load' \
	    | sed 's/.*prompt \([0-9]*\) tok in \([0-9]*\) ms.*/\1 \2/')
	l=$(grep 'charsiu fp16:  plan' "$ERR")
	rd=$(printf '%s' "$l" | sed 's/.*read \([0-9]*\) .*/\1/')
	po=$(printf '%s' "$l" | sed 's/.*poison \([0-9]*\) .*/\1/')
	printf '%s %s\n' "$t" "$(( ${rd:-0} + ${po:-0} ))"
}

printf '   %6s  %9s %11s   %9s %11s   %s\n' \
	tokens 'cell ms' 'poison+read' 'row ms' 'poison+read' 'TTFT r/c'
printf '   %6s  %9s %11s   %9s %11s   %s\n' \
	------ --------- ----------- --------- ----------- --------

LAST=""
for R in $REPS; do
	P=""; i=0
	while [ $i -lt "$R" ]; do P="$P$CLAUSE"; i=$((i+1)); done
	one 1 "$P" >/dev/null 2>&1
	one 0 "$P" >/dev/null 2>&1
	A=; B=; RA=; RB=; TOK=; n=0
	while [ $n -lt "$N" ]; do
		n=$((n+1))
		if [ $((n % 2)) -eq 1 ]; then
			r=$(one 1 "$P"); TOK=$(echo "$r"|cut -d' ' -f1); A="$A $(echo "$r"|cut -d' ' -f2)"; RA="$RA $(echo "$r"|cut -d' ' -f3)"
			r=$(one 0 "$P"); B="$B $(echo "$r"|cut -d' ' -f2)"; RB="$RB $(echo "$r"|cut -d' ' -f3)"
		else
			r=$(one 0 "$P"); B="$B $(echo "$r"|cut -d' ' -f2)"; RB="$RB $(echo "$r"|cut -d' ' -f3)"
			r=$(one 1 "$P"); TOK=$(echo "$r"|cut -d' ' -f1); A="$A $(echo "$r"|cut -d' ' -f2)"; RA="$RA $(echo "$r"|cut -d' ' -f3)"
		fi
	done
	[ -n "$A" ] && [ -n "$B" ] || { echo "   $R clauses: NO OUTPUT"; continue; }
	MA=$(mid "$A"); MB=$(mid "$B")
	RAT=$(awk "BEGIN{printf \"%.3f\", $MB/$MA}" 2>/dev/null || echo "?")
	printf '   %6s  %9s %11s   %9s %11s   %s\n' \
		"$TOK" "$MA" "$(mid "$RA")" "$MB" "$(mid "$RB")" "$RAT"
	echo "          TTFT every cell $(lo "$A")..$(hi "$A")   one a row $(lo "$B")..$(hi "$B")"
	LAST="$P"
done

echo
echo "== the same tokens out of both arms, at the longest length"
a=$(env $E CHARSIU_FP16_FULLSCAN=1 "$RUN" "$M" -p "$LAST" -n 24 --ignore-eos -q -c 1024 -t 4 2>/dev/null | grep -v '^\[')
b=$(env $E CHARSIU_FP16_FULLSCAN=0 "$RUN" "$M" -p "$LAST" -n 24 --ignore-eos -q -c 1024 -t 4 2>/dev/null | grep -v '^\[')
if [ "$a" = "$b" ]; then
	echo "   identical: $(printf '%s' "$a" | md5sum | cut -c1-12)"
else
	echo "   ⛔ DIFFER"
	echo "   every cell $a"
	echo "   one a row  $b"
fi

echo
echo "== the whole stage table at the longest length, row sentinel on"
env $E CHARSIU_FP16_FULLSCAN=0 "$RUN" "$M" -p "$LAST" -n 1 --ignore-eos \
	-c 1024 -t 4 2>&1 >/dev/null | grep 'charsiu fp16' | awk '{print "   " $0}'

rm -f "$ERR"
echo
echo "restoring schedutil"
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil > "$p/scaling_governor" 2>/dev/null || true; done
