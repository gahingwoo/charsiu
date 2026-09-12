#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# What the CPU clock is worth -- the half of "maximum frequency" r390 did not
# touch.
#
# r390 swept the NPU clock and found a 32.4% rise buys the vendor's decode
# -0.9%. Its own closing note says the obvious extension is NOT supported:
# the vendor's published figures are "at the maximum CPU *and* NPU
# frequencies", and a decode that ignores the NPU clock is a decode that may
# well be bound by the CPU. This is that measurement.
#
# 🔑 UNLIKE THE NPU CLOCK, THIS ONE MOVES WITHIN ONE BOOT. The NPU rate lives
# in the device tree, so r390 had to spend a reboot per arm and then measure
# the boot-to-boot drift it was warned about. cpufreq is sysfs, so every
# frequency point here is the same boot, the same load, the same binary --
# which is a much stronger experiment than r390 could run.
#
# ⚠⚠ THE FREQUENCY IS CONFIRMED, NOT ASSUMED. Writing scaling_setspeed is a
# request; scaling_cur_freq is what happened. Every block prints what it got,
# and a block that did not get what it asked for says so and is still printed,
# because a silently-ignored knob is exactly how a null result gets
# manufactured.
#
# ⚠⚠ THE CPU-ONLY ARM IS A POSITIVE CONTROL AND IT IS NOT OPTIONAL. If the
# NPU arms come back flat, "the CPU clock does not matter here" and "my knob
# never took effect" are the same reading. A pure CPU decode MUST scale with
# this knob; if it does not, nothing else on the page can be believed.
#
# ⚠ UP THEN DOWN. The points are swept low->high->low so that anything
# monotone in time -- warming silicon, a background job, a drifting rail --
# shows up as a disagreement between a point and its own repeat rather than as
# a slope. This kernel exposes no thermal zone, so that disagreement is the
# only throttling detector there is.
#
# ⚠ WHICH BINARY. A number belongs to the binary that produced it. The image's
# /opt/charsiu/charsiu_run is dated 2026-08-24 and predates the affinity pin;
# it read 14.8% spread and one core faster than two. This script refuses to
# run against a charsiu_run older than the affinity pin and prints the date of
# the one it used.
#
#   sh tests/board_cpu_clock.sh
set -u

REPEAT=${CPU_REPEAT:-3}
NTOK=${CPU_NTOK:-64}
RUN=${CHARSIU_RUN:-/root/charsiu_run_new}
VBIN=${VENDOR_BENCH:-/opt/vendor/bin/vendor_bench}
VLIB=${VENDOR_LIB:-/opt/vendor/lib}
VMODEL=${VENDOR_MODEL:-/opt/vendor/model/Llama-3.2-1B-Instruct-rk3576-w4a16.rkllm}
CMODEL=${CHARSIU_MODEL:-/opt/charsiu/models/Llama-3.2-1B-Instruct-Q4_0.gguf}
CTRL=${CONTROL_MODEL:-/opt/charsiu/models/SmolLM2-135M-Instruct-Q4_0.gguf}
P="Explain in plain words why a written record outlasts a memory."

mid() { printf '%s\n' $1 | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$0} END{if(NR==0){print "-"}else if(NR%2)printf "%s\n",a[(NR+1)/2]; else printf "%.2f\n",(a[NR/2]+a[NR/2+1])/2}'; }
lo()  { printf '%s\n' $1 | tr ' ' '\n' | grep -v '^$' | sort -n | head -1; }
hi()  { printf '%s\n' $1 | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1; }

POL=$(ls -d /sys/devices/system/cpu/cpufreq/policy* 2>/dev/null)
[ -n "$POL" ] || { echo "no cpufreq policies"; exit 1; }

# ⚠ THE TOP POINT IS PER-POLICY MAX, THE OTHERS ARE EQUAL ON BOTH. The
# vendor's condition is "maximum CPU frequency", which on this SoC is 2016 MHz
# for the A53 cluster and 2208 for the A72 -- so the top point is deliberately
# NOT one number. Every lower point is a frequency both clusters have, so the
# sweep below the top is symmetric.
maxof() { cat "$1/scaling_available_frequencies" | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1; }

setfreq() {  # $1 = frequency, or the word "max"
	for p in $POL; do
		echo userspace > "$p/scaling_governor" 2>/dev/null || true
		if [ "$1" = max ]; then echo "$(maxof "$p")" > "$p/scaling_setspeed" 2>/dev/null || true
		else echo "$1" > "$p/scaling_setspeed" 2>/dev/null || true; fi
	done
	sleep 1
	OK=1; GOT=
	for p in $POL; do
		want=$([ "$1" = max ] && maxof "$p" || echo "$1")
		got=$(cat "$p/scaling_cur_freq")
		GOT="$GOT $(basename "$p")=$got"
		[ "$got" = "$want" ] || OK=0
	done
	echo "   asked $1 -> got$GOT $([ "$OK" = 1 ] || echo '  ⚠ THE KNOB DID NOT TAKE')"
}

echo "== what the CPU clock is worth"
echo "   boot id   $(cat /proc/sys/kernel/random/boot_id)"
echo "   npu clk   $(cat /sys/kernel/debug/clk/clk_rknn_dsu0/clk_rate 2>/dev/null || cat /sys/kernel/debug/clk/aclk_rknn_root/clk_rate 2>/dev/null) Hz  (unchanged all round)"
echo "   rail      $(awk '/vdd_npu_s0/{print $6; exit}' /sys/kernel/debug/regulator/regulator_summary 2>/dev/null)"
echo "   accel     $(ls /dev/accel/ 2>/dev/null | tr '\n' ' ')$(ls /sys/bus/platform/drivers/RKNPU/ 2>/dev/null | grep -q npu && echo '(RKNPU bound)')"
echo "   charsiu   $RUN  $(ls -l --full-time "$RUN" 2>/dev/null | awk '{print $6}')"
echo "   thermal   $(ls /sys/class/thermal/ 2>/dev/null | tr '\n' ' ')"
echo "   repeats   $REPEAT per point, one warm-up discarded, swept up then down"
echo

case "$(ls -l --full-time "$RUN" 2>/dev/null | awk '{print $6}')" in
2026-08-2*) echo "⚠⚠ $RUN predates the affinity pin. Refusing."; exit 1;;
esac

vendor_arm() {
	# ⚠ THE VENDOR RUNTIME FINDS THE NPU THROUGH THE DRM RENDER NODE, which
	# only their driver publishes. On the rocket arm the binary and the model
	# are both still mounted and it will start, spend its load time and then
	# fail -- so the arm is skipped on the driver, not on the files.
	ls /sys/bus/platform/drivers/RKNPU 2>/dev/null | grep -q npu || { echo "   vendor:  not this arm (RKNPU unbound)"; return; }
	[ -x "$VBIN" ] && [ -f "$VMODEL" ] || { echo "   vendor:  MISSING ($VBIN / $VMODEL)"; return; }
	O=$(cd "$(dirname "$VBIN")" && LD_LIBRARY_PATH="$VLIB" "$VBIN" "$VMODEL" "$REPEAT" "$NTOK" "$P" 2>/dev/null)
	echo "$O" | grep -E '^   (TTFT|tok/s)' | sed 's/^   /   vendor  /'
}

# ⚠⚠ THIS IS THE TREE'S BOARD ENVIRONMENT AND THE SHORT VERSION IS NOT IT.
# The first run of this script copied clock_cost.sh's three knobs and read
# 13.85 tok/s where r388 recorded 17.85. The missing knob was
# CHARSIU_NPU_MAXN: its C default is 8192, the model's output head is 128256
# rows wide, and under 8192 that head silently stays on the CPU. It is worth
# 55% of decode. board_verify.sh, board_chunk_sweep.sh and board_intermittent.sh
# all spell this line; anything measuring decode has to spell it too.
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"

charsiu_npu_arm() {
	[ -e /dev/accel/accel0 ] || { echo "   charsiu: no /dev/accel -- not this arm"; return; }
	env $E "$RUN" "$CMODEL" -p "$P" -n "$NTOK" --ignore-eos -c 512 -t 4 >/dev/null 2>&1
	T=; S=; n=0
	while [ $n -lt "$REPEAT" ]; do
		n=$((n+1))
		O=$(env $E \
		    "$RUN" "$CMODEL" -p "$P" -n "$NTOK" --ignore-eos -c 512 -t 4 2>/dev/null | grep '^\[load')
		[ -n "$O" ] || { echo "   charsiu run $n: NO OUTPUT"; continue; }
		T="$T $(echo "$O" | sed 's/.*prompt [0-9]* tok in \([0-9]*\) ms.*/\1/')"
		S="$S $(echo "$O" | sed 's/.*gen [0-9]* tok in [0-9]* ms, \([0-9.]*\) tok.*/\1/')"
	done
	printf '   charsiu TTFT  median %s ms   range %s..%s\n' "$(mid "$T")" "$(lo "$T")" "$(hi "$T")"
	printf '   charsiu tok/s median %s      range %s..%s\n' "$(mid "$S")" "$(lo "$S")" "$(hi "$S")"
}

# ⚠⚠ THE CONTROL. A pure CPU decode, same harness, same parser. It is here to
# prove the knob bites, so its numbers are meaningless on their own and MUST
# move with the frequency.
control_arm() {
	[ -f "$CTRL" ] || { echo "   control: MISSING"; return; }
	env CHARSIU_NPU=0 "$RUN" "$CTRL" -p "$P" -n "$NTOK" --ignore-eos -c 512 -t 4 >/dev/null 2>&1
	S=; n=0
	while [ $n -lt "$REPEAT" ]; do
		n=$((n+1))
		O=$(env CHARSIU_NPU=0 "$RUN" "$CTRL" -p "$P" -n "$NTOK" --ignore-eos -c 512 -t 4 2>/dev/null | grep '^\[load')
		[ -n "$O" ] && S="$S $(echo "$O" | sed 's/.*gen [0-9]* tok in [0-9]* ms, \([0-9.]*\) tok.*/\1/')"
	done
	printf '   control tok/s median %s      range %s..%s   (CPU only, must move)\n' "$(mid "$S")" "$(lo "$S")" "$(hi "$S")"
}

for F in ${CPU_POINTS:-1008000 1416000 max max 1416000 1008000}; do
	echo "-- $F"
	setfreq "$F"
	vendor_arm
	charsiu_npu_arm
	control_arm
	echo
done

echo "restoring schedutil"
for p in $POL; do echo schedutil > "$p/scaling_governor" 2>/dev/null || true; done
