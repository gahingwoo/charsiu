#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# What the pack saves by not converting the zeros it is told about.
#
# Attention's values matmul contracts over k = THE CONTEXT LENGTH, because the
# V surface is packed at that k and a buffer written at one k and run at
# another is a different permutation of the same weights. So row r of the
# probabilities has pos+1 entries that can be nonzero and zeros to the end of
# k -- and on an 852 token prompt at k = 1024 that is 58% of every element the
# pack touches, converted from a float zero into a half zero.
#
# charsiu_fp16_op.xtri0 is the caller saying so, and the pack memsets the tail.
#
# A CALLER THAT SETS IT AND IS WRONG GETS A SILENTLY WRONG ANSWER, which
# is the whole reason this script runs CHARSIU_FP16_TRI_CHECK=1 at the longest
# length: that arm READS BACK every element it was told to skip and counts the
# ones that are not zero. A pack that quietly zeroes live probabilities still
# produces fluent text.
#
# THE KNOB THIS SCRIPT SWEEPS WAS REMOVED THE SAME DAY IT MEASURED IT, and
# this script now REFUSES rather than quietly measuring one arm twice.
#
# r400 section 2 priced CHARSIU_FP16_TRI at 8% of the pack and nothing outside
# TTFT's own spread. What made it worth keeping was the next result: the caller
# was zeroing that tail itself, in the one stage of the path with no pool behind
# it, and it only did so in order that this could convert a zero into a zero.
# With the caller's zeroing gone, what lies past the triangle is the RAW scores
# and the memset is the answer, not a saving -- so there is no off arm.
#
# Kept because r400 is reproducible against charsiu at 235635d, where the knob
# still existed. Against anything newer the two arms are the same arm, which is
# exactly the null arm this tree has been caught by before.
#
#   CHARSIU_TRI_REPS="4 12 24 34"   clause counts (about 25 tokens each)
#   CHARSIU_TRI_N=2                 repeats an arm a length
. "$(dirname "$0")/board_clk.sh"
NPUCLK=$(npu_clk) || exit 1
set -u

REPS=${CHARSIU_TRI_REPS:-4 12 24 34}
N=${CHARSIU_TRI_N:-2}
RUN=${CHARSIU_RUN:-/opt/charsiu/charsiu_run}
M=${CHARSIU_MODEL:-/opt/charsiu/models/Llama-3.2-1B-Instruct-Q4_0.gguf}
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 CHARSIU_ATTN_NPU=1 CHARSIU_STAGES=1 CHARSIU_FP16_PACK=1"
CLAUSE="A written record can be copied, checked against other records, and read by someone who never met the person who wrote it. "
ERR=/tmp/fp16tri.$$

[ -e /dev/accel/accel0 ] || { echo "no /dev/accel -- this needs the rocket arm"; exit 1; }
[ -x "$RUN" ] || { echo "no $RUN"; exit 1; }
# A NULL ARM IS NOT A NULL RESULT. If the binary does not carry the knob,
# both arms below are the default and the table would read 1.000 everywhere.
if ! strings "$RUN" 2>/dev/null | grep -q '^CHARSIU_FP16_TRI$'; then
	echo "$RUN has no CHARSIU_FP16_TRI: the knob was removed on 09-13"
	echo "   when the caller stopped zeroing the tail, which made the"
	echo "   triangle load bearing rather than optional. This script only"
	echo "   reproduces r400 section 2 against charsiu at 235635d."
	exit 1
fi

for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo userspace > "$p/scaling_governor" 2>/dev/null || true
	tr ' ' '\n' < "$p/scaling_available_frequencies" | grep -v '^$' | sort -n \
		| tail -1 > "$p/scaling_setspeed" 2>/dev/null || true
done
sleep 1

echo "== the causal tail: memset it, or convert it"
echo "   boot      $(cat /proc/sys/kernel/random/boot_id)"
echo "   npu clk   $NPUCLK Hz"
echo "   cpu       $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq)/$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq) kHz"
echo "   binary    $RUN  $(ls -l --full-time "$RUN" 2>/dev/null | awk '{print $6}')"
echo "   build     $(charsiu_build "$RUN")"
echo "   model     $(basename "$M")"
echo "   $N repeats an arm a length, arms alternating, one warm-up discarded"
echo "   the vector pack is on in BOTH arms"
echo

mid() { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$0} END{if(NR%2)printf "%s\n",a[(NR+1)/2]; else printf "%.0f\n",(a[NR/2]+a[NR/2+1])/2}'; }
lo()  { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | head -1; }
hi()  { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1; }

# $1 = arm, $2 = prompt.  prints: tokens ttft_ms pack_ms skipped_pct
one() {
	t=$(env $E "CHARSIU_FP16_TRI=$1" "$RUN" "$M" -p "$2" -n 1 \
	    --ignore-eos -c 1024 -t 4 2>"$ERR" \
	    | grep '^\[load' \
	    | sed 's/.*prompt \([0-9]*\) tok in \([0-9]*\) ms.*/\1 \2/')
	pk=$(grep 'charsiu fp16:  wcopy' "$ERR" | sed 's/.*pack \([0-9]*\).*/\1/')
	sk=$(grep 'as the causal tail' "$ERR" | sed 's/.*(\([0-9]*\)%).*/\1/')
	printf '%s %s %s\n' "$t" "${pk:-?}" "${sk:-0}"
}

printf '   %6s  %9s %8s   %9s %8s %7s   %s\n' \
	tokens 'convert ms' 'pack ms' 'memset ms' 'pack ms' 'skipped' 'TTFT t/c'
printf '   %6s  %9s %8s   %9s %8s %7s   %s\n' \
	------ ---------- ------- --------- ------- ------- --------

LAST=""
for R in $REPS; do
	P=""; i=0
	while [ $i -lt "$R" ]; do P="$P$CLAUSE"; i=$((i+1)); done
	one 0 "$P" >/dev/null 2>&1
	one 1 "$P" >/dev/null 2>&1
	A=; B=; PA=; PB=; SK=; TOK=; n=0
	while [ $n -lt "$N" ]; do
		n=$((n+1))
		# alternate the ORDER as well as the arms
		if [ $((n % 2)) -eq 1 ]; then
			r=$(one 0 "$P"); TOK=$(echo "$r"|cut -d' ' -f1); A="$A $(echo "$r"|cut -d' ' -f2)"; PA="$PA $(echo "$r"|cut -d' ' -f3)"
			r=$(one 1 "$P"); B="$B $(echo "$r"|cut -d' ' -f2)"; PB="$PB $(echo "$r"|cut -d' ' -f3)"; SK=$(echo "$r"|cut -d' ' -f4)
		else
			r=$(one 1 "$P"); B="$B $(echo "$r"|cut -d' ' -f2)"; PB="$PB $(echo "$r"|cut -d' ' -f3)"; SK=$(echo "$r"|cut -d' ' -f4)
			r=$(one 0 "$P"); TOK=$(echo "$r"|cut -d' ' -f1); A="$A $(echo "$r"|cut -d' ' -f2)"; PA="$PA $(echo "$r"|cut -d' ' -f3)"
		fi
	done
	[ -n "$A" ] && [ -n "$B" ] || { echo "   $R clauses: NO OUTPUT"; continue; }
	MA=$(mid "$A"); MB=$(mid "$B")
	RAT=$(awk "BEGIN{printf \"%.3f\", $MB/$MA}" 2>/dev/null || echo "?")
	printf '   %6s  %9s %8s   %9s %8s %6s%%   %s\n' \
		"$TOK" "$MA" "$(mid "$PA")" "$MB" "$(mid "$PB")" "$SK" "$RAT"
	echo "          TTFT convert $(lo "$A")..$(hi "$A")   memset $(lo "$B")..$(hi "$B")"
	LAST="$P"
done

echo
echo "== what the promise skipped, read back"
env $E CHARSIU_FP16_TRI=1 CHARSIU_FP16_TRI_CHECK=1 "$RUN" "$M" -p "$LAST" \
	-n 1 --ignore-eos -c 1024 -t 4 2>&1 >/dev/null \
	| grep 'causal tail' | awk '{print "   " $0}'

echo
echo "== the same tokens out of both arms, at the longest length"
a=$(env $E CHARSIU_FP16_TRI=0 "$RUN" "$M" -p "$LAST" -n 24 --ignore-eos -q -c 1024 -t 4 2>/dev/null | grep -v '^\[')
b=$(env $E CHARSIU_FP16_TRI=1 "$RUN" "$M" -p "$LAST" -n 24 --ignore-eos -q -c 1024 -t 4 2>/dev/null | grep -v '^\[')
if [ "$a" = "$b" ]; then
	echo "   identical: $(printf '%s' "$a" | md5sum | cut -c1-12)"
else
	echo "   DIFFER"
	echo "   convert $a"
	echo "   memset  $b"
fi

rm -f "$ERR"
echo
echo "restoring schedutil"
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil > "$p/scaling_governor" 2>/dev/null || true; done
