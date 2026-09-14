#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# ⭐⭐ HOW MUCH ONE ARM MOVES INSIDE ONE BOOT, WITH NOTHING CHANGED.
#
# Every margin in this tree is read against two bounds: the 2.2% cross-boot
# drift, and the arm's own spread inside a run. Neither of those is the bound
# that matters when two cells of one table are measured twenty minutes apart.
#
# ⚠⚠ AND THAT BOUND HAS A READING NOW. r411 measured the CPU attention arm at
# 302 tokens twice on boot 2f21a2e1, same binary, clock pinned: 2274 ms in its
# section 8 and 2138 ms in r412's re-run. 6.4% apart, WITHIN one boot, larger
# than the cross-boot number this tree quotes -- and each run reported a spread
# of about 2%, so neither run could see it.
#
# This runs ONE arm at ONE length many times, back to back, and prints the CPU
# frequency and every thermal zone beside each reading. Three outcomes and they
# are different problems:
#
#   flat                     the r411/r412 gap was something else, go find it
#   climbs with temperature  the board throttles under a long sweep and every
#                            multi-cell table in this tree has a slow arm at
#                            the bottom of it
#   random                   the bound on a within-boot comparison is whatever
#                            this spread is, and tables have to alternate arms
#                            rather than finish one before starting the other
#
# ⚠ The governor is pinned exactly as every other board script pins it, so a
# frequency that still moves is the SoC refusing, not the governor choosing.
#
#   CHARSIU_DRIFT_N=12        readings
#   CHARSIU_DRIFT_CLAUSES=12  about 25 tokens each
. "$(dirname "$0")/board_clk.sh"
NPUCLK=$(npu_clk) || exit 1
set -u

N=${CHARSIU_DRIFT_N:-12}
R=${CHARSIU_DRIFT_CLAUSES:-12}
RUN=${CHARSIU_RUN:-/opt/charsiu/charsiu_run}
M=${CHARSIU_MODEL:-/opt/charsiu/models/Llama-3.2-1B-Instruct-Q4_0.gguf}
ARM=${CHARSIU_DRIFT_ARM:-CHARSIU_ATTN_NPU=0}
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"
CLAUSE="A written record can be copied, checked against other records, and read by someone who never met the person who wrote it. "

[ -n "$(ls /dev/accel/accel* 2>/dev/null)" ] || { echo "no /dev/accel"; exit 1; }
[ -x "$RUN" ] || { echo "no $RUN"; exit 1; }

for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo userspace > "$p/scaling_governor" 2>/dev/null || true
	tr ' ' '\n' < "$p/scaling_available_frequencies" | grep -v '^$' | sort -n \
		| tail -1 > "$p/scaling_setspeed" 2>/dev/null || true
done
sleep 1

ZONES=$(ls -d /sys/class/thermal/thermal_zone* 2>/dev/null)

echo "== one arm, one length, $N times, nothing changed"
echo "   boot      $(cat /proc/sys/kernel/random/boot_id)"
echo "   npu clk   $NPUCLK Hz"
echo "   arm       $ARM"
echo "   binary    $RUN"
echo "   model     $(basename "$M")"
printf '   zones    '
for z in $ZONES; do printf ' %s' "$(cat "$z/type" 2>/dev/null)"; done
echo
echo

P=""; i=0
while [ $i -lt "$R" ]; do P="$P$CLAUSE"; i=$((i+1)); done

printf '   %3s  %6s %8s  %9s %9s  %s\n' n tok 'TTFT ms' 'A53 kHz' 'A72 kHz' 'temps C'
env $E $ARM "$RUN" "$M" -p "$P" -n 1 --ignore-eos -c 1024 -t 4 >/dev/null 2>&1
n=0; T=
while [ $n -lt "$N" ]; do
	n=$((n+1))
	r=$(env $E $ARM "$RUN" "$M" -p "$P" -n 1 --ignore-eos -c 1024 -t 4 2>/dev/null \
	    | grep '^\[load' | sed 's/.*prompt \([0-9]*\) tok in \([0-9]*\) ms.*/\1 \2/')
	f0=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq 2>/dev/null)
	f4=$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq 2>/dev/null)
	t=""
	for z in $ZONES; do
		t="$t $(awk '{printf "%.1f", $1/1000}' "$z/temp" 2>/dev/null)"
	done
	T="$T ${r##* }"
	printf '   %3s  %6s %8s  %9s %9s  %s\n' "$n" "${r%% *}" "${r##* }" "$f0" "$f4" "$t"
done

echo
printf '%s\n' $T | sort -n | awk '{a[NR]=$0}
	END{ m=(NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2;
	     printf "   median %.0f  range %s..%s  spread %.1f%% of the median\n",
	            m, a[1], a[NR], 100*(a[NR]-a[1])/m }'
echo "   ⚠ THAT SPREAD IS THE FLOOR ON ANY WITHIN-BOOT COMPARISON IN THIS TREE."

echo
echo "restoring schedutil"
for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo schedutil > "$p/scaling_governor" 2>/dev/null || true
done
