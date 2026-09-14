#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# ⚠⚠⚠ DOES int8 WEIGHTS AGAINST fp16 ACTIVATIONS COMPUTE THE RIGHT ANSWER.
#
# r404 measured that arrangement at 2.4x the fp16 one on the scores shape and
# that measurement is the whole basis of the int8 KV project. It was taken
# with charsiu_int4 --cost, whose own header says it CHECKS NOTHING: the
# weight buffer holds whatever the setup left in it and the output is never
# read. Fast is not right.
#
# The precision register composes this out of two halves that are each
# confirmed on the vendor's own file -- 0x00000000 for int8 weights,
# 0x20000000 for a 16 bit activation -- and their SUM appears in none of its
# 8308 dispatches. So nothing outside this script says the combination works.
#
# Three arms at each shape, differing in one field each:
#
#   fp16 weights                          the shipped path, the control
#   int8 weights, every scale 1.0         vs the raw integer MAC
#   int8 weights, one scale per channel   vs the dequantised reference
#
# The second and third differ ONLY in job.weight_scales, so whether the per
# output channel fp16 table reaches the output in acc_out mode is a ratio of
# two hardware readings rather than a reading of the output stage.
#
# ⚠ AND THAT ANSWER DECIDES THE DESIGN, not just a constant: if the table does
# not apply, the scale has to be folded into the softmax on the CPU, which is
# a different patch from the one the handoff describes.
#
#   CHARSIU_W8_SHAPES="k:n ..."   default: the two attention shapes + a square
#   CHARSIU_W8_M=78               rows a dispatch
#   CHARSIU_W8_REPS=8             submits an arm
. "$(dirname "$0")/board_clk.sh"
NPUCLK=$(npu_clk) || exit 1
set -u

SHAPES=${CHARSIU_W8_SHAPES:-64:864 128:864 64:64 1024:64 1024:1024}
M=${CHARSIU_W8_M:-78}
REPS=${CHARSIU_W8_REPS:-8}
#
# ⚠⚠ NO BUILD LINE HERE, AND charsiu_build MUST NOT BE POINTED AT THIS ONE.
# npu_fp16_test is compiled with the same -DCHARSIU_BUILD as charsiu_run and
# ⚠ THIS COMMENT DESCRIBED THE OPPOSITE FOR A WHILE. It said the tool "has
# no --version to print it back" and warned that its first argument is read
# through atoi, so asking would submit a dimension of zero. Both halves were
# true when written; the flag went into every tool afterwards and is checked
# before any positional argument, so asking is safe now.
# has no --version to print it back. Worse, its first argument is K, read
# through atoi: asking it for --version asks the hardware for a K = 0 shape.
# What this round ran is $T and its mtime, and nothing finer, until the tool
# grows the flag.
#
T=${CHARSIU_W8_TEST:-/root/npu_fp16_test}

[ -e /dev/accel/accel0 ] || { echo "no /dev/accel -- this needs the rocket arm"; exit 1; }
[ -x "$T" ] || { echo "no $T"; exit 1; }
# ⚠ A NULL ARM IS NOT A NULL RESULT: an older binary has no --w8 and would run
# the default layout sweep at every shape and print a table that looks fine.
if ! strings "$T" 2>/dev/null | grep -q -- '--w8'; then
	echo "⛔ $T has no --w8 arm: it predates this probe"
	exit 1
fi

for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo userspace > "$p/scaling_governor" 2>/dev/null || true
	tr ' ' '\n' < "$p/scaling_available_frequencies" | grep -v '^$' | sort -n \
		| tail -1 > "$p/scaling_setspeed" 2>/dev/null || true
done
sleep 1

echo "== int8 weights, fp16 activations: does it compute, and does the scale apply"
echo "   boot      $(cat /proc/sys/kernel/random/boot_id)"
echo "   npu clk   $NPUCLK Hz"
echo "   cpu       $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq)/$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq) kHz"
echo "   binary    $T  $(ls -l --full-time "$T" 2>/dev/null | awk '{print $6}')"
echo "   m         $M rows, $REPS reps an arm, arms alternating"
echo

for S in $SHAPES; do
	K=${S%%:*}; N=${S##*:}
	echo "---------------------------------------------------------------"
	echo "K=$K N=$N"
	CHARSIU_TEST_M="$M" "$T" "$K" "$N" --w8 "$REPS" 2>&1
	echo
	dmesg 2>/dev/null | tail -3 | grep -i "timed out\|iommu\|rocket" && \
		echo "   ⚠ the kernel said something -- read the whole dmesg"
done

echo "restoring schedutil"
for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo schedutil > "$p/scaling_governor" 2>/dev/null || true
done
