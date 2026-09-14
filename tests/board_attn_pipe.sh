#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# ⭐⭐ THE FENCE IS A SLEEPING ioctl AND THE SOFTMAX IS FOUR BUSY CORES, AND
# THEY HAPPEN ONE AFTER THE OTHER.
#
# r407 read the layer at 852 tokens: scores fence 504 ms, values pack 477 (the
# softmax, fused into it since r409), values fence 385. Every millisecond of
# the two fences is four idle cores and every millisecond of the pack is an
# idle NPU, and nothing makes them exclusive except that one function did both
# halves of a group.
#
# CHARSIU_ATTN_PIPE=2 splits the heads over two unit pairs: group 1's scores go
# to the hardware BEFORE group 0's softmax runs, so the CPU works while the NPU
# does. The hardware order is unchanged -- S0 S1 V0 V1, one job a submit,
# serialised on one file descriptor -- so this is not the two cores in flight
# together, which corrupts.
#
# ⚠ 1 IS NOT QUITE TODAY, and the log should say so: at 1 the next call's
# sentinels are written while the values job runs rather than after it. The arm
# to read is 1 against 2 in THIS binary.
#
# ⚠⚠ AND THE TEXT MUST BE IDENTICAL. Splitting a group changes which op index a
# head has, and a head that reads another head's V surface is fluent and wrong.
#
#   CHARSIU_PIPE_REPS="4 12 24 34"   clause counts (about 25 tokens each)
#   CHARSIU_PIPE_N=3                 repeats an arm a length
#   CHARSIU_PIPE_G=2                 the second arm's group count (1, 2, 4, 8)
. "$(dirname "$0")/board_clk.sh"
NPUCLK=$(npu_clk) || exit 1
set -u

REPS=${CHARSIU_PIPE_REPS:-4 12 24 34}
N=${CHARSIU_PIPE_N:-3}
PG=${CHARSIU_PIPE_G:-2}
RUN=${CHARSIU_RUN:-/opt/charsiu/charsiu_run}
M=${CHARSIU_MODEL:-/opt/charsiu/models/Llama-3.2-1B-Instruct-Q4_0.gguf}
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 CHARSIU_ATTN_NPU=1 CHARSIU_STAGES=1"
CLAUSE="A written record can be copied, checked against other records, and read by someone who never met the person who wrote it. "
ERR=/tmp/pipe.$$

[ -e /dev/accel/accel0 ] || { echo "no /dev/accel -- this needs the rocket arm"; exit 1; }
[ -x "$RUN" ] || { echo "no $RUN"; exit 1; }
if ! strings "$RUN" 2>/dev/null | grep -q '^CHARSIU_ATTN_PIPE$'; then
	echo "⛔ $RUN has no CHARSIU_ATTN_PIPE: both arms would be the default"
	echo "   and the table would read 1.000 everywhere"
	exit 1
fi

for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo userspace > "$p/scaling_governor" 2>/dev/null || true
	tr ' ' '\n' < "$p/scaling_available_frequencies" | grep -v '^$' | sort -n \
		| tail -1 > "$p/scaling_setspeed" 2>/dev/null || true
done
sleep 1

echo "== the softmax against the next group's scores, by prompt length"
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

# $1 = pipe groups, $2 = prompt.  prints: tokens ttft_ms layer_ms
one() {
	t=$(env $E "CHARSIU_ATTN_PIPE=$1" "$RUN" "$M" -p "$2" -n 1 \
	    --ignore-eos -c 1024 -t 4 2>"$ERR" \
	    | grep '^\[load' \
	    | sed 's/.*prompt \([0-9]*\) tok in \([0-9]*\) ms.*/\1 \2/')
	l=$(grep 'in the layer' "$ERR" | head -1 | sed 's/.*of \([0-9]*\) ms in the layer.*/\1/')
	printf '%s %s\n' "$t" "${l:-?}"
}

printf '   %6s  %9s %12s %9s   %9s %12s %9s   %s\n' \
	tokens 'one ms' range 'layer' "$PG ms" range 'layer' "$PG / 1"
printf '   %6s  %9s %12s %9s   %9s %12s %9s   %s\n' \
	------ --------- ------------ --------- --------- ------------ --------- -----

LAST=""
for R in $REPS; do
	P=""; i=0
	while [ $i -lt "$R" ]; do P="$P$CLAUSE"; i=$((i+1)); done
	one 1 "$P" >/dev/null 2>&1
	one $PG "$P" >/dev/null 2>&1
	A=; B=; LA=; LB=; TOK=; n=0
	while [ $n -lt "$N" ]; do
		n=$((n+1))
		if [ $((n % 2)) -eq 1 ]; then
			r=$(one 1 "$P"); TOK=$(echo "$r"|cut -d' ' -f1); A="$A $(echo "$r"|cut -d' ' -f2)"; LA="$LA $(echo "$r"|cut -d' ' -f3)"
			r=$(one $PG "$P"); B="$B $(echo "$r"|cut -d' ' -f2)"; LB="$LB $(echo "$r"|cut -d' ' -f3)"
		else
			r=$(one $PG "$P"); B="$B $(echo "$r"|cut -d' ' -f2)"; LB="$LB $(echo "$r"|cut -d' ' -f3)"
			r=$(one 1 "$P"); TOK=$(echo "$r"|cut -d' ' -f1); A="$A $(echo "$r"|cut -d' ' -f2)"; LA="$LA $(echo "$r"|cut -d' ' -f3)"
		fi
	done
	[ -n "$A" ] && [ -n "$B" ] || { echo "   $R clauses: NO OUTPUT"; continue; }
	MA=$(mid "$A"); MB=$(mid "$B")
	printf '   %6s  %9s %5s..%-5s %9s   %9s %5s..%-5s %9s   %s\n' \
		"$TOK" "$MA" "$(lo "$A")" "$(hi "$A")" "$(mid "$LA")" \
		"$MB" "$(lo "$B")" "$(hi "$B")" "$(mid "$LB")" \
		"$(awk "BEGIN{printf \"%.3f\", $MB/$MA}")"
	LAST="$P"
done

echo
echo "== where the two arms spend the layer, at the longest length"
for g in 1 $PG; do
	echo "   CHARSIU_ATTN_PIPE=$g"
	env $E "CHARSIU_ATTN_PIPE=$g" "$RUN" "$M" -p "$LAST" -n 1 --ignore-eos \
	    -q -c 1024 -t 4 2>&1 >/dev/null \
	    | grep -E "in the layer|fp16 (scores|values)|^charsiu fp16: " \
	    | head -12 | awk '{print "      " $0}'
done

echo
echo "== the same tokens out of both arms, at the longest length"
a=$(env $E CHARSIU_ATTN_PIPE=1 "$RUN" "$M" -p "$LAST" -n 24 --ignore-eos -q -c 1024 -t 4 2>/dev/null | grep -v '^\[')
b=$(env $E CHARSIU_ATTN_PIPE=$PG "$RUN" "$M" -p "$LAST" -n 24 --ignore-eos -q -c 1024 -t 4 2>/dev/null | grep -v '^\[')
if [ "$a" = "$b" ]; then
	echo "   identical: $(printf '%s' "$a" | md5sum | cut -c1-12)"
else
	echo "   ⛔ DIFFER"
	echo "   one $a"
	echo "   two $b"
fi

rm -f "$ERR"
echo
echo "restoring schedutil"
for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo schedutil > "$p/scaling_governor" 2>/dev/null || true
done
