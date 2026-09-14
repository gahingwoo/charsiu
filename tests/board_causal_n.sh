#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# ⭐ THE SCORES MATMUL WAS WRITING A RECTANGLE WHERE THE ANSWER IS A TRIANGLE.
#
# A chunk of rows starting at position p0 attends to nothing after its own last
# row. Every column past that is computed, written to memory, and then memset by
# the values pack, which xtri0 has been telling since r400 -- while the scores
# matmul still ran at the whole prompt's width.
#
# r407 measured what a column costs, holding m and k still: n = 864 is 174 us,
# 432 is 94, 224 is 61, 112 is 42. About 22 us fixed and 0.176 us a column, so
# at the shipped width three quarters of the fence is the output write and the
# weight fetch, and both scale with n.
#
# ⚠ THE ARM IS NAMED IN BOTH DIRECTIONS. CHARSIU_ATTN_NPU_CAUSAL_N=0 is the
# rectangle this replaces, 1 is the triangle, and the script refuses a binary
# that does not carry the knob rather than measuring one arm twice.
#
# ⚠⚠ AND THE TEXT MUST BE IDENTICAL. This changes what the hardware computes,
# not how it is read: the columns dropped are ones no reader ever looked at. If
# that is wrong anywhere, the text moves -- and a wrong answer here is fluent.
#
#   CHARSIU_CN_REPS="4 12 24 34"   clause counts (about 25 tokens each)
#   CHARSIU_CN_N=2                 repeats an arm a length
. "$(dirname "$0")/board_clk.sh"
NPUCLK=$(npu_clk) || exit 1
set -u

REPS=${CHARSIU_CN_REPS:-4 12 24 34}
N=${CHARSIU_CN_N:-2}
RUN=${CHARSIU_RUN:-/opt/charsiu/charsiu_run}
M=${CHARSIU_MODEL:-/opt/charsiu/models/Llama-3.2-1B-Instruct-Q4_0.gguf}
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 CHARSIU_ATTN_NPU=1 CHARSIU_STAGES=1"
CLAUSE="A written record can be copied, checked against other records, and read by someone who never met the person who wrote it. "
ERR=/tmp/cn.$$

[ -e /dev/accel/accel0 ] || { echo "no /dev/accel -- this needs the rocket arm"; exit 1; }
[ -x "$RUN" ] || { echo "no $RUN"; exit 1; }
if ! strings "$RUN" 2>/dev/null | grep -q '^CHARSIU_ATTN_NPU_CAUSAL_N$'; then
	echo "⛔ $RUN has no CHARSIU_ATTN_NPU_CAUSAL_N: both arms would be the"
	echo "   default and the table would read 1.000 everywhere"
	exit 1
fi

for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo userspace > "$p/scaling_governor" 2>/dev/null || true
	tr ' ' '\n' < "$p/scaling_available_frequencies" | grep -v '^$' | sort -n \
		| tail -1 > "$p/scaling_setspeed" 2>/dev/null || true
done
sleep 1

echo "== the scores matmul at the chunk's own width, against the prompt's"
echo "   boot      $(cat /proc/sys/kernel/random/boot_id)"
echo "   npu clk   $NPUCLK Hz"
echo "   cpu       $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq)/$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq) kHz"
echo "   binary    $RUN  $(ls -l --full-time "$RUN" 2>/dev/null | awk '{print $6}')"
echo "   build     $(charsiu_build "$RUN")"
echo "   model     $(basename "$M")"
echo "   $N repeats an arm a length, arms alternating, one warm-up discarded"
echo

mid() { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$0} END{if(NR%2)printf "%s\n",a[(NR+1)/2]; else printf "%.0f\n",(a[NR/2]+a[NR/2+1])/2}'; }
lo()  { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | head -1; }
hi()  { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1; }

# $1 = arm, $2 = prompt.  prints: tokens ttft_ms attn_fence_ms
one() {
	t=$(env $E "CHARSIU_ATTN_NPU_CAUSAL_N=$1" "$RUN" "$M" -p "$2" -n 1 \
	    --ignore-eos -c 1024 -t 4 2>"$ERR" \
	    | grep '^\[load' \
	    | sed 's/.*prompt \([0-9]*\) tok in \([0-9]*\) ms.*/\1 \2/')
	fe=$(grep 'charsiu fp16:  plan' "$ERR" | head -1 | sed 's/.*fence \([0-9]*\).*/\1/')
	printf '%s %s\n' "$t" "${fe:-?}"
}

printf '   %6s  %9s %10s   %9s %10s   %s\n' \
	tokens 'rect ms' 'scores ms' 'tri ms' 'scores ms' 'TTFT t/r'
printf '   %6s  %9s %10s   %9s %10s   %s\n' \
	------ --------- ---------- --------- ---------- --------

LAST=""
for R in $REPS; do
	P=""; i=0
	while [ $i -lt "$R" ]; do P="$P$CLAUSE"; i=$((i+1)); done
	one 0 "$P" >/dev/null 2>&1
	one 1 "$P" >/dev/null 2>&1
	A=; B=; FA=; FB=; TOK=; n=0
	while [ $n -lt "$N" ]; do
		n=$((n+1))
		if [ $((n % 2)) -eq 1 ]; then
			r=$(one 0 "$P"); TOK=$(echo "$r"|cut -d' ' -f1); A="$A $(echo "$r"|cut -d' ' -f2)"; FA="$FA $(echo "$r"|cut -d' ' -f3)"
			r=$(one 1 "$P"); B="$B $(echo "$r"|cut -d' ' -f2)"; FB="$FB $(echo "$r"|cut -d' ' -f3)"
		else
			r=$(one 1 "$P"); B="$B $(echo "$r"|cut -d' ' -f2)"; FB="$FB $(echo "$r"|cut -d' ' -f3)"
			r=$(one 0 "$P"); TOK=$(echo "$r"|cut -d' ' -f1); A="$A $(echo "$r"|cut -d' ' -f2)"; FA="$FA $(echo "$r"|cut -d' ' -f3)"
		fi
	done
	[ -n "$A" ] && [ -n "$B" ] || { echo "   $R clauses: NO OUTPUT"; continue; }
	MA=$(mid "$A"); MB=$(mid "$B")
	RAT=$(awk "BEGIN{printf \"%.3f\", $MB/$MA}" 2>/dev/null || echo "?")
	printf '   %6s  %9s %10s   %9s %10s   %s\n' \
		"$TOK" "$MA" "$(mid "$FA")" "$MB" "$(mid "$FB")" "$RAT"
	echo "          TTFT rect $(lo "$A")..$(hi "$A")   tri $(lo "$B")..$(hi "$B")"
	LAST="$P"
done

echo
echo "== the same tokens out of both arms, at the longest length"
a=$(env $E CHARSIU_ATTN_NPU_CAUSAL_N=0 "$RUN" "$M" -p "$LAST" -n 24 --ignore-eos -q -c 1024 -t 4 2>/dev/null | grep -v '^\[')
b=$(env $E CHARSIU_ATTN_NPU_CAUSAL_N=1 "$RUN" "$M" -p "$LAST" -n 24 --ignore-eos -q -c 1024 -t 4 2>/dev/null | grep -v '^\[')
if [ "$a" = "$b" ]; then
	echo "   identical: $(printf '%s' "$a" | md5sum | cut -c1-12)"
else
	echo "   ⛔ DIFFER"
	echo "   rect $a"
	echo "   tri  $b"
fi

echo
echo "== and the refusals, which a width that is not a multiple of 16 would show"
env $E CHARSIU_ATTN_NPU_CAUSAL_N=1 "$RUN" "$M" -p "$LAST" -n 1 --ignore-eos \
	-c 1024 -t 4 2>&1 >/dev/null | grep -E "fp16 attention (ran|layer)" | head -5 | awk '{print "   " $0}'

rm -f "$ERR"
echo "restoring schedutil"
for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo schedutil > "$p/scaling_governor" 2>/dev/null || true
done
