#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# What the fp16 attention path spends on CONVERTING ITS INPUT, and what a
# vector conversion is worth.
#
# r399 printed the counters this path had been keeping since it was written.
# On Llama-3.2-1B the pack was 1357 ms of 2943 -- 46% -- and it is the
# float-to-fp16 conversion of the activation, one element at a time. The
# reading taken from it then was 235 ns an element, which is fifty times what
# fourteen instructions can cost, and it was wrong for a boring reason: the
# element count was guessed from `calls` when the two differ by a factor of
# the CONTEXT LENGTH. The values matmul contracts over k = kv, so one head of
# one chunk is m * kv elements, not m * head_dim. The counter is in the report
# now and this script reads it rather than dividing.
#
# ⚠ THE CONVERSION IS NOT vcvt_f16_f32 AND MUST NOT BECOME IT. charsiu_f2h
# truncates the mantissa and flushes subnormals; the hardware instruction
# rounds to nearest even and keeps them. tests/pack_f16run.c walks all 2^32
# float bit patterns against the scalar definition, on the desk, because the
# development host is itself aarch64 -- so what this script is measuring is a
# change already proved to move no byte.
#
# Both arms name the knob: CHARSIU_FP16_PACK=1 is the vector run, =0 the per
# element loop. Alternating, one boot, clock pinned.
#
#   CHARSIU_PACK_REPS="4 12 24 34"   clause counts (about 25 tokens each)
#   CHARSIU_PACK_N=2                 repeats an arm a length
. "$(dirname "$0")/board_clk.sh"
NPUCLK=$(npu_clk) || exit 1
set -u

REPS=${CHARSIU_PACK_REPS:-4 12 24 34}
N=${CHARSIU_PACK_N:-2}
RUN=${CHARSIU_RUN:-/opt/charsiu/charsiu_run}
M=${CHARSIU_MODEL:-/opt/charsiu/models/Llama-3.2-1B-Instruct-Q4_0.gguf}
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 CHARSIU_ATTN_NPU=1 CHARSIU_STAGES=1"
CLAUSE="A written record can be copied, checked against other records, and read by someone who never met the person who wrote it. "
ERR=/tmp/fp16pack.$$

[ -e /dev/accel/accel0 ] || { echo "no /dev/accel -- this needs the rocket arm"; exit 1; }
[ -x "$RUN" ] || { echo "no $RUN"; exit 1; }

for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo userspace > "$p/scaling_governor" 2>/dev/null || true
	tr ' ' '\n' < "$p/scaling_available_frequencies" | grep -v '^$' | sort -n \
		| tail -1 > "$p/scaling_setspeed" 2>/dev/null || true
done
sleep 1

echo "== the fp16 activation pack, vector against per element"
echo "   boot      $(cat /proc/sys/kernel/random/boot_id)"
echo "   npu clk   $NPUCLK Hz"
echo "   cpu       $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq)/$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq) kHz"
echo "   binary    $RUN  $(ls -l --full-time "$RUN" 2>/dev/null | awk '{print $6}')"
echo "   build     $(charsiu_build "$RUN")"
echo "   model     $(basename "$M")"
echo "   $N repeats an arm a length, arms alternating, one warm-up discarded"
echo "   CHARSIU_ATTN_NPU=1 throughout, or none of these lines is reached"
echo

mid() { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$0} END{if(NR%2)printf "%s\n",a[(NR+1)/2]; else printf "%.0f\n",(a[NR/2]+a[NR/2+1])/2}'; }
lo()  { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | head -1; }
hi()  { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1; }

# $1 = arm, $2 = prompt.  prints: tokens ttft_ms pack_ms psync_ms ns_each
one() {
	t=$(env $E "CHARSIU_FP16_PACK=$1" "$RUN" "$M" -p "$2" -n 1 \
	    --ignore-eos -c 1024 -t 4 2>"$ERR" \
	    | grep '^\[load' \
	    | sed 's/.*prompt \([0-9]*\) tok in \([0-9]*\) ms.*/\1 \2/')
	pk=$(grep 'charsiu fp16:  wcopy' "$ERR" | sed 's/.*pack \([0-9]*\).*/\1/')
	ps=$(grep 'charsiu fp16:  wcopy' "$ERR" | sed 's/.*psync \([0-9]*\).*/\1/')
	ns=$(grep 'elements,' "$ERR" | sed 's/.*elements, \([0-9.]*\) ns.*/\1/')
	el=$(grep 'elements,' "$ERR" | sed 's/.*pack \([0-9]*\) elements.*/\1/')
	printf '%s %s %s %s %s\n' "$t" "${pk:-?}" "${ps:-?}" "${ns:-?}" "${el:-?}"
}

printf '   %6s %6s  %9s %9s %6s   %9s %9s %6s   %s\n' \
	tokens elems 'scalar ms' 'pack ms' 'ns/el' 'vector ms' 'pack ms' 'ns/el' 'TTFT v/s'
printf '   %6s %6s  %9s %9s %6s   %9s %9s %6s   %s\n' \
	------ ----- --------- ------- ----- --------- ------- ----- --------

LAST=""
for R in $REPS; do
	P=""; i=0
	while [ $i -lt "$R" ]; do P="$P$CLAUSE"; i=$((i+1)); done
	one 0 "$P" >/dev/null 2>&1
	one 1 "$P" >/dev/null 2>&1
	A=; B=; PA=; PB=; NA=; NB=; TOK=; EL=; n=0
	while [ $n -lt "$N" ]; do
		n=$((n+1))
		# ⚠ ALTERNATE, and run the SCALAR arm first at odd repeats and
		# the vector one first at even, because running one arm before
		# the other inside every repeat penalises whichever goes first
		if [ $((n % 2)) -eq 1 ]; then
			r=$(one 0 "$P"); TOK=$(echo "$r"|cut -d' ' -f1); A="$A $(echo "$r"|cut -d' ' -f2)"; PA="$PA $(echo "$r"|cut -d' ' -f3)"; NA="$NA $(echo "$r"|cut -d' ' -f5)"
			r=$(one 1 "$P"); B="$B $(echo "$r"|cut -d' ' -f2)"; PB="$PB $(echo "$r"|cut -d' ' -f3)"; NB="$NB $(echo "$r"|cut -d' ' -f5)"; EL=$(echo "$r"|cut -d' ' -f6)
		else
			r=$(one 1 "$P"); B="$B $(echo "$r"|cut -d' ' -f2)"; PB="$PB $(echo "$r"|cut -d' ' -f3)"; NB="$NB $(echo "$r"|cut -d' ' -f5)"; EL=$(echo "$r"|cut -d' ' -f6)
			r=$(one 0 "$P"); TOK=$(echo "$r"|cut -d' ' -f1); A="$A $(echo "$r"|cut -d' ' -f2)"; PA="$PA $(echo "$r"|cut -d' ' -f3)"; NA="$NA $(echo "$r"|cut -d' ' -f5)"
		fi
	done
	[ -n "$A" ] && [ -n "$B" ] || { echo "   $R clauses: NO OUTPUT"; continue; }
	MA=$(mid "$A"); MB=$(mid "$B")
	RAT=$(awk "BEGIN{printf \"%.3f\", $MB/$MA}" 2>/dev/null || echo "?")
	printf '   %6s %6s  %9s %9s %6s   %9s %9s %6s   %s\n' \
		"$TOK" "$EL" "$MA" "$(mid "$PA")" "$(mid "$NA")" \
		"$MB" "$(mid "$PB")" "$(mid "$NB")" "$RAT"
	echo "          TTFT scalar $(lo "$A")..$(hi "$A")   vector $(lo "$B")..$(hi "$B")"
	LAST="$P"
done

echo
echo "== the same tokens out of both arms, at the longest length"
# the conversion is bit identical by construction and by tests/pack_f16run.c;
# this is the end to end control that says the runtime really uses it
a=$(env $E CHARSIU_FP16_PACK=0 "$RUN" "$M" -p "$LAST" -n 24 --ignore-eos -q -c 1024 -t 4 2>/dev/null | grep -v '^\[')
b=$(env $E CHARSIU_FP16_PACK=1 "$RUN" "$M" -p "$LAST" -n 24 --ignore-eos -q -c 1024 -t 4 2>/dev/null | grep -v '^\[')
if [ "$a" = "$b" ]; then
	echo "   identical: $(printf '%s' "$a" | md5sum | cut -c1-12)"
else
	echo "   ⛔ DIFFER"
	echo "   scalar $a"
	echo "   vector $b"
fi

rm -f "$ERR"
echo
echo "restoring schedutil"
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil > "$p/scaling_governor" 2>/dev/null || true; done
