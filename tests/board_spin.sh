#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# THE FENCE SLEEPS, AND UNTIL NOW NOBODY HAS PRICED NOT SLEEPING.
#
# charsiu_bo_prep blocks in the kernel, so every fence pays a wake-up. The int4
# projections fence 2464 times in an 852 token prompt (1232 calls, two devices)
# and r407 put that wait at 1.78 ms a row, 1516 ms of the prompt, with all four
# cores idle inside it. CHARSIU_NPU_SPIN_US polls instead, up to that many
# microseconds, before falling back to the blocking wait. device.c has carried
# it since phase 20 under "off unless asked, until phase 21 has priced it", and
# phase 21 never did.
#
# IT IS NOT FREE ANY MORE. Since r411 the attention fences have CPU work
# happening beside them, so a poll there competes with the softmax rather than
# filling an idle core. That is why the table prints the attention layer as
# well as TTFT: a knob that buys the projections and sells attention is two
# results, not one.
#
# AND EACH POLL IS AN ioctl. A spin that never wins is a syscall storm with
# the blocking wait still at the end of it, so the won/lost counts are printed
# and a round where `won` is near zero has answered the question.
#
#   CHARSIU_SPIN_ARMS="0 50 200 1000"
#   CHARSIU_SPIN_N=3                  repeats an arm
#   CHARSIU_SPIN_REPS=34              clauses (about 25 tokens each)
. "$(dirname "$0")/board_clk.sh"
NPUCLK=$(npu_clk) || exit 1
set -u

ARMS=${CHARSIU_SPIN_ARMS:-0 50 200 1000}
N=${CHARSIU_SPIN_N:-3}
REPS=${CHARSIU_SPIN_REPS:-34}
RUN=${CHARSIU_RUN:-/opt/charsiu/charsiu_run}
M=${CHARSIU_MODEL:-/opt/charsiu/models/Llama-3.2-1B-Instruct-Q4_0.gguf}
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 CHARSIU_ATTN_NPU=1 CHARSIU_STAGES=1"
CLAUSE="A written record can be copied, checked against other records, and read by someone who never met the person who wrote it. "
ERR=/tmp/spin.$$

[ -e /dev/accel/accel0 ] || { echo "no /dev/accel -- this needs the rocket arm"; exit 1; }
[ -x "$RUN" ] || { echo "no $RUN"; exit 1; }
if ! strings "$RUN" 2>/dev/null | grep -q '^CHARSIU_NPU_SPIN_US$'; then
	echo "$RUN has no CHARSIU_NPU_SPIN_US: every arm would be the"
	echo "   blocking wait and the table would be one arm four times"
	exit 1
fi

for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo userspace > "$p/scaling_governor" 2>/dev/null || true
	tr ' ' '\n' < "$p/scaling_available_frequencies" | grep -v '^$' | sort -n \
		| tail -1 > "$p/scaling_setspeed" 2>/dev/null || true
done
sleep 1

P=""; i=0
while [ $i -lt "$REPS" ]; do P="$P$CLAUSE"; i=$((i+1)); done

echo "== polling the fence instead of sleeping on it"
echo "   boot      $(cat /proc/sys/kernel/random/boot_id)"
echo "   npu clk   $NPUCLK Hz"
echo "   cpu       $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq)/$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq) kHz"
echo "   binary    $RUN  $(ls -l --full-time "$RUN" 2>/dev/null | awk '{print $6}')"
echo "   build     $(charsiu_build "$RUN")"
echo "   model     $(basename "$M")"
echo "   $N repeats an arm, arms rotated, order reversed on even passes"
echo

mid() { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$0} END{if(NR%2)printf "%s\n",a[(NR+1)/2]; else printf "%.0f\n",(a[NR/2]+a[NR/2+1])/2}'; }
lo()  { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | head -1; }
hi()  { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1; }

# $1 = spin us.  prints: tokens ttft_ms layer_ms fence_ms
one() {
	t=$(env $E "CHARSIU_NPU_SPIN_US=$1" "$RUN" "$M" -p "$P" -n 1 \
	    --ignore-eos -c 1024 -t 4 2>"$ERR" \
	    | grep '^\[load' \
	    | sed 's/.*prompt \([0-9]*\) tok in \([0-9]*\) ms.*/\1 \2/')
	l=$(grep 'in the layer' "$ERR" | head -1 | sed 's/.*of \([0-9]*\) ms in the layer.*/\1/')
	f=$(grep 'waiting for the fence' "$ERR" | head -1 | sed 's/.*, \([0-9]*\) ms waiting for the fence.*/\1/')
	printf '%s %s %s\n' "$t" "${l:-?}" "${f:-?}"
}

for a in $ARMS; do one "$a" >/dev/null 2>&1; done
for a in $ARMS; do eval "T_$a=''"; eval "L_$a=''"; eval "F_$a=''"; done

TOK=""; n=0
while [ $n -lt "$N" ]; do
	n=$((n+1))
	if [ $((n % 2)) -eq 1 ]; then ORDER="$ARMS"; else
		ORDER=$(printf '%s\n' $ARMS | sed '1!G;h;$!d' | tr '\n' ' '); fi
	for a in $ORDER; do
		r=$(one "$a")
		TOK=$(echo "$r" | cut -d' ' -f1)
		eval "T_$a=\"\$T_$a $(echo "$r"|cut -d' ' -f2)\""
		eval "L_$a=\"\$L_$a $(echo "$r"|cut -d' ' -f3)\""
		eval "F_$a=\"\$F_$a $(echo "$r"|cut -d' ' -f4)\""
	done
done

echo "   $TOK tokens"
echo
printf '   %8s %9s %12s %9s %9s   %s\n' 'spin us' 'TTFT ms' range 'attn ms' 'int4 fence' 'vs 0'
printf '   %8s %9s %12s %9s %9s   %s\n' -------- --------- ------------ --------- ---------- ------
BASE=""
for a in $ARMS; do
	eval "T=\$T_$a"; eval "L=\$L_$a"; eval "F=\$F_$a"
	MT=$(mid "$T")
	[ -n "$MT" ] || { echo "   $a: NO OUTPUT"; continue; }
	[ -n "$BASE" ] || BASE=$MT
	printf '   %8s %9s %5s..%-5s %9s %10s   %s\n' \
		"$a" "$MT" "$(lo "$T")" "$(hi "$T")" "$(mid "$L")" "$(mid "$F")" \
		"$(awk "BEGIN{printf \"%.3f\", $MT/$BASE}")"
done

echo
echo "== how often the poll actually won"
for a in $ARMS; do
	[ "$a" = 0 ] && continue
	printf '   spin %s us\n' "$a"
	env $E "CHARSIU_NPU_SPIN_US=$a" "$RUN" "$M" -p "$P" -n 1 --ignore-eos \
	    -q -c 1024 -t 4 2>&1 >/dev/null | grep -i "spin" | head -3 \
	    | awk '{print "      " $0}'
done

echo
echo "== the same tokens out of the fastest arm and the sleeping one"
a=$(env $E CHARSIU_NPU_SPIN_US=0 "$RUN" "$M" -p "$P" -n 24 --ignore-eos -q -c 1024 -t 4 2>/dev/null | grep -v '^\[')
for k in $ARMS; do
	[ "$k" = 0 ] && continue
	b=$(env $E "CHARSIU_NPU_SPIN_US=$k" "$RUN" "$M" -p "$P" -n 24 --ignore-eos -q -c 1024 -t 4 2>/dev/null | grep -v '^\[')
	if [ "$a" = "$b" ]; then
		printf '   spin %-6s identical: %s\n' "$k" "$(printf '%s' "$b" | md5sum | cut -c1-12)"
	else
		printf '   spin %-6s DIFFERS\n' "$k"
	fi
done

rm -f "$ERR"
echo
echo "restoring schedutil"
for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo schedutil > "$p/scaling_governor" 2>/dev/null || true
done
