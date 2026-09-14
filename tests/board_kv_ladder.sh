#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# ⭐ THE V SURFACE'S LADDER DOUBLES, AND AT 852 TOKENS THAT FINISHES ON 1024.
#
# kv is the values matmul's reduction extent, so the cost of a prompt is the
# sum of kv over its positions. A doubling ladder pays 523k units where an
# extent that tracked the position exactly would pay 363k: 1.44x. The vendor's
# own file grows this length in steps of 32.
#
# ⛔ THE OBVIOUS FIX IS SLOWER, WHICH IS WHY THIS IS A SWEEP AND NOT A PATCH.
# Starting kv at the prompt total charges the first 512 rows 864 apiece where
# the ladder charges them 32, 64, ... 512 -- 736k against 523k. The hint is a
# CEILING, not a floor. Two knobs, named separately:
#
#   CHARSIU_ATTN_KV_CAP=1    stop the ladder at ceil(prompt/32) instead of
#                            climbing past it        (523k -> 469k, 1.29x)
#   CHARSIU_ATTN_KV_STEP=n   grow by n percent instead of 200
#                            150 -> 439k, 133 -> 420k, 125 -> 412k
#
# ⚠⚠ AND THE COUNTER-COST IS REAL AND IS NOT IN THAT ARITHMETIC. Every growth
# frees and re-allocates one buffer object per layer per kv head, memsets the
# whole of each, and syncs it twice -- then packs every live position again.
# A finer step buys extent and pays repacks: 5 repacks at 200, 11 at 125, and
# the rows repacked go 992 -> 3232. Which side wins is a board question.
#
# ⚠ THE LADDER ITSELF IS PRINTED, because a count of repacks does not say
# which rungs, and the rungs are what the arithmetic above predicts. If a row
# of this table does not show the ladder the model predicted, the knob did not
# reach the code and the timing is two copies of one arm.
#
# ⭐ AND THE THIRD FIELD IS THE FIX FOR THE REPACK. The padded extent appears
# in ONE term of the packed offset, so a rung's bytes are the next rung's bytes
# at another base: a growth is a block copy per output channel group, not a
# conversion per position. r411 measured the old way at about 68 ms a rung,
# which is why every finer step lost. tests/fp16_regrow.c holds the copy against
# the packer over 1584 shapes, byte for byte.
#
#   CHARSIU_KVL_ARMS="200:0:1 200:1:1 133:1:1 133:1:0 125:1:1"
#                     step:cap:copy, copy defaulting to 1
#   CHARSIU_KVL_N=3                repeats an arm
#   CHARSIU_KVL_REPS=34            clauses (about 25 tokens each)
. "$(dirname "$0")/board_clk.sh"
NPUCLK=$(npu_clk) || exit 1
set -u

ARMS=${CHARSIU_KVL_ARMS:-200:0:1 200:1:1 150:1:1 133:1:1 133:1:0 125:1:1}
N=${CHARSIU_KVL_N:-3}
REPS=${CHARSIU_KVL_REPS:-34}
RUN=${CHARSIU_RUN:-/opt/charsiu/charsiu_run}
M=${CHARSIU_MODEL:-/opt/charsiu/models/Llama-3.2-1B-Instruct-Q4_0.gguf}
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 CHARSIU_ATTN_NPU=1 CHARSIU_STAGES=1"
CLAUSE="A written record can be copied, checked against other records, and read by someone who never met the person who wrote it. "
ERR=/tmp/kvl.$$

[ -e /dev/accel/accel0 ] || { echo "no /dev/accel -- this needs the rocket arm"; exit 1; }
[ -x "$RUN" ] || { echo "no $RUN"; exit 1; }
for k in CHARSIU_ATTN_KV_STEP CHARSIU_ATTN_KV_CAP CHARSIU_ATTN_KV_COPY; do
	strings "$RUN" 2>/dev/null | grep -q "^$k\$" && continue
	echo "⛔ $RUN has no $k: every arm would be the default and the"
	echo "   table would be one arm measured five times"
	exit 1
done

for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo userspace > "$p/scaling_governor" 2>/dev/null || true
	tr ' ' '\n' < "$p/scaling_available_frequencies" | grep -v '^$' | sort -n \
		| tail -1 > "$p/scaling_setspeed" 2>/dev/null || true
done
sleep 1

P=""; i=0
while [ $i -lt "$REPS" ]; do P="$P$CLAUSE"; i=$((i+1)); done

echo "== the V surface's ladder: how steep, and how far past the prompt"
echo "   boot      $(cat /proc/sys/kernel/random/boot_id)"
echo "   npu clk   $NPUCLK Hz"
echo "   cpu       $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq)/$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq) kHz"
echo "   binary    $RUN  $(ls -l --full-time "$RUN" 2>/dev/null | awk '{print $6}')"
echo "   model     $(basename "$M")"
echo "   $N repeats an arm, arms rotated, order reversed on even passes"
echo "   arm 200:0:1 is today's ladder; the third field is the block copy"
echo

st() { echo "$1" | cut -d: -f1; }
cp_() { echo "$1" | cut -d: -f2; }
cy() { c=$(echo "$1" | cut -d: -f3); echo "${c:-1}"; }

mid() { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$0} END{if(NR%2)printf "%s\n",a[(NR+1)/2]; else printf "%.0f\n",(a[NR/2]+a[NR/2+1])/2}'; }
lo()  { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | head -1; }
hi()  { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1; }

# $1 = step, $2 = cap, $3 = copy.  prints: tokens ttft_ms values_ms layer_ms
one() {
	t=$(env $E "CHARSIU_ATTN_KV_STEP=$1" "CHARSIU_ATTN_KV_CAP=$2" \
	    "CHARSIU_ATTN_KV_COPY=$3" \
	    "$RUN" "$M" -p "$P" -n 1 --ignore-eos -c 1024 -t 4 2>"$ERR" \
	    | grep '^\[load' \
	    | sed 's/.*prompt \([0-9]*\) tok in \([0-9]*\) ms.*/\1 \2/')
	v=$(grep 'in the layer' "$ERR" | head -1 | sed 's/.*values group \([0-9]*\).*/\1/')
	l=$(grep 'in the layer' "$ERR" | head -1 | sed 's/.*of \([0-9]*\) ms in the layer.*/\1/')
	printf '%s %s %s\n' "$t" "${v:-?}" "${l:-?}"
}

ladder() {
	env $E "CHARSIU_ATTN_KV_STEP=$1" "CHARSIU_ATTN_KV_CAP=$2" \
	    "CHARSIU_ATTN_KV_COPY=$3" \
	    "$RUN" "$M" -p "$P" -n 1 --ignore-eos -q -c 1024 -t 4 2>&1 >/dev/null \
	    | grep 'V surface ladder' | head -1 | sed 's/^charsiu: *//'
}

for a in $ARMS; do one "$(st "$a")" "$(cp_ "$a")" "$(cy "$a")" >/dev/null 2>&1; done

TOK=""
for a in $ARMS; do
	eval "TT_$(echo "$a" | tr ':' '_')=''"
	eval "VV_$(echo "$a" | tr ':' '_')=''"
	eval "LL_$(echo "$a" | tr ':' '_')=''"
done

n=0
while [ $n -lt "$N" ]; do
	n=$((n+1))
	if [ $((n % 2)) -eq 1 ]; then ORDER="$ARMS"; else
		ORDER=$(printf '%s\n' $ARMS | sed '1!G;h;$!d' | tr '\n' ' '); fi
	for a in $ORDER; do
		v=$(echo "$a" | tr ':' '_')
		r=$(one "$(st "$a")" "$(cp_ "$a")" "$(cy "$a")")
		TOK=$(echo "$r" | cut -d' ' -f1)
		eval "TT_$v=\"\$TT_$v $(echo "$r"|cut -d' ' -f2)\""
		eval "VV_$v=\"\$VV_$v $(echo "$r"|cut -d' ' -f3)\""
		eval "LL_$v=\"\$LL_$v $(echo "$r"|cut -d' ' -f4)\""
	done
done

echo "   $TOK tokens"
echo
printf '   %-10s %9s %12s %9s %9s   %s\n' arm 'TTFT ms' range 'values ms' 'layer ms' 'vs today'
printf '   %-10s %9s %12s %9s %9s   %s\n' ---------- --------- ------------ --------- --------- --------
BASE=""
for a in $ARMS; do
	v=$(echo "$a" | tr ':' '_')
	eval "T=\$TT_$v"; eval "V=\$VV_$v"; eval "L=\$LL_$v"
	MT=$(mid "$T")
	[ -n "$MT" ] || { echo "   $a: NO OUTPUT"; continue; }
	[ -n "$BASE" ] || BASE=$MT
	printf '   %-10s %9s %5s..%-5s %9s %9s   %s\n' \
		"$a" "$MT" "$(lo "$T")" "$(hi "$T")" "$(mid "$V")" "$(mid "$L")" \
		"$(awk "BEGIN{printf \"%.3f\", $MT/$BASE}")"
done

echo
echo "== the rungs each arm actually stood on"
for a in $ARMS; do
	printf '   %-10s %s\n' "$a" "$(ladder "$(st "$a")" "$(cp_ "$a")" "$(cy "$a")")"
done

echo
echo "== the same tokens out of every arm"
REF=""
for a in $ARMS; do
	o=$(env $E "CHARSIU_ATTN_KV_STEP=$(st "$a")" "CHARSIU_ATTN_KV_CAP=$(cp_ "$a")" \
	    "CHARSIU_ATTN_KV_COPY=$(cy "$a")" \
	    "$RUN" "$M" -p "$P" -n 24 --ignore-eos -q -c 1024 -t 4 2>/dev/null \
	    | grep -v '^\[')
	h=$(printf '%s' "$o" | md5sum | cut -c1-12)
	if [ -z "$REF" ]; then REF="$h"; RT="$o"; printf '   %-10s %s  (reference)\n' "$a" "$h"
	elif [ "$h" = "$REF" ]; then printf '   %-10s %s\n' "$a" "$h"
	else
		printf '   %-10s %s  ⛔ DIFFERS\n' "$a" "$h"
		echo "      ref $RT"
		echo "      arm $o"
	fi
done

rm -f "$ERR"
echo
echo "restoring schedutil"
for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo schedutil > "$p/scaling_governor" 2>/dev/null || true
done
