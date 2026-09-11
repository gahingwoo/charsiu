#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# THE ROUND OF RECORD: every number a paper quotes, from ONE boot, with the
# environment that produced them printed beside them.
#
# ⚠⚠ WHY THIS EXISTS. The figures currently in circulation are spread across
# 09-04, 09-06, 09-07, 09-08 and 09-10. Mixing two rounds has already produced
# one defect in this project that no checker could catch, because both figures
# were true in their own round and nothing on the page said they were different
# rounds. Five rounds is that failure with more surface.
#
# So this runs the speed table, the quality table and the gemma4 spread in one
# session, stamps the boot id at both ends, and REFUSES to be read as one round
# if the two stamps differ.
#
#   sh tests/board_record.sh [OUTFILE]
#
# ⚠ It is long -- four models at REPEAT runs each, plus a 20-reading sweep.
# CHARSIU_RECORD_REPEAT and CHARSIU_RECORD_N cut it for a dry run, and the
# header says when they have been cut, because a round of record that was
# quietly shortened is not one.
set -eu

OUT=${1:-$HOME/charsiu-board/record-$(date +%Y%m%d-%H%M%S).txt}
D=$(dirname "$0")
BIN=$D
[ -x "$BIN/../build/charsiu_ppl" ] && BIN=$D/../build
[ -x "$BIN/charsiu_ppl" ] || BIN=/opt/charsiu
[ -x "$BIN/charsiu_ppl" ] || { echo "no charsiu_ppl beside $0, ../build or /opt/charsiu"; exit 2; }
CORPUS=$D/corpus/long.txt
[ -f "$CORPUS" ] || CORPUS=/opt/charsiu/corpus/long.txt
REPEAT=${CHARSIU_RECORD_REPEAT:-7}
NSWEEP=${CHARSIU_RECORD_N:-20}
mkdir -p "$(dirname "$OUT")"

#
# ⚠⚠ THE MODEL FILE IS NAMED, NOT THE MODEL. charsiu re-quantises whatever it
# loads, so the source format is inside every perplexity: the same
# Llama-3.2-1B reads 41.5289 from Q4_0 and 34.6888 from Q8_0 with one scale a
# row. A quality number without its file is a 20% error with nothing on the
# page to show it.
#
#
# ⚠⚠ MODELS LIVE IN TWO PLACES AND A ROUND THAT PICKS ONE FINDS HALF OF THEM.
# `charsiu pull` puts them in $HOME/.charsiu/models; the installer puts them in
# /opt/charsiu/models. On this board gemma4 is in the first and the other three
# are in the second, so a script that resolves a DIRECTORY and then looks
# inside it reports "no gemma4" while gemma4 is sitting on the card -- which
# cost two rounds on 2026-09-11, once as a skipped sweep and once as a missing
# row in the speed table.
#
# Resolve per FILE, over every directory.
#
MODELDIRS=${CHARSIU_RECORD_MODELS:-"$HOME/.charsiu/models /opt/charsiu/models"}
find_model() {
	for _d in $MODELDIRS; do
		for _f in "$_d"/$1; do
			[ -f "$_f" ] && { echo "$_f"; return 0; }
		done
	done
	return 1
}
QMODEL=${CHARSIU_RECORD_QMODEL:-$(find_model 'Llama-3.2-1B-Instruct-Q4_0.gguf' || true)}

env_block() {
	echo "== environment, $1"
	echo "   boot id   $(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
	echo "   uptime    $(cut -d' ' -f1 /proc/uptime 2>/dev/null) s"
	echo "   kernel    $(uname -r)"
	echo "   governor  $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo '?')" \
	     "(cpu0) / $(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor 2>/dev/null || echo '?') (cpu4)"
	# ⚠ THE RAIL, ALWAYS. m=8 was read as a property of the width for a
	# fortnight and it was the NPU rail at 750 mV: 871 of 904 correct at
	# 750, 904 of 904 at 800. A round that does not say which voltage it
	# ran at cannot be compared with either reading.
	for _r in /sys/class/regulator/regulator.*; do
		_n=$(cat "$_r/name" 2>/dev/null) || continue
		case $_n in
		*npu*|*NPU*) echo "   rail      $_n $(cat "$_r/microvolts" 2>/dev/null) uV" ;;
		esac
	done
	for _d in /sys/class/devfreq/*; do
		[ -e "$_d/cur_freq" ] || continue
		case $_d in
		*npu*|*NPU*) echo "   npu clk   $(cat "$_d/cur_freq" 2>/dev/null) Hz" \
		                  "(governor $(cat "$_d/governor" 2>/dev/null))" ;;
		esac
	done
	for _t in /sys/class/thermal/thermal_zone*; do
		[ -e "$_t/temp" ] || continue
		echo "   $(basename "$_t")  $(cat "$_t/type" 2>/dev/null)" \
		     "$(awk '{printf "%.1f", $1/1000}' "$_t/temp" 2>/dev/null) C"
	done
}

{
echo "================================================================"
echo " charsiu round of record   $(date -Is)"
echo "================================================================"
echo " charsiu   $(cd "$D/.." 2>/dev/null && git log --oneline -1 2>/dev/null || echo 'not a git tree here')"
#
# ⚠⚠ THE FINGERPRINT, NOT THE FILENAME. Two machines can hold different bytes
# under one name, and charsiu re-quantises whatever it loads, so the source
# file is inside every perplexity it prints. The desk and this board disagree
# by about 1.3% on the same nominal model, and an md5 on each side is what
# decides whether that is the FILE or the BINARY -- thread count is already
# ruled out, since 8, 4 and 1 threads give 33.8071 on the desk bit for bit.
#
echo " quality model file:  ${QMODEL:-NOT FOUND}"
[ -n "$QMODEL" ] && echo " quality model md5:   $(md5sum "$QMODEL" 2>/dev/null | cut -c1-32)  ($(stat -Lc%s "$QMODEL" 2>/dev/null) bytes)"
echo " charsiu_ppl binary:  $BIN/charsiu_ppl  md5 $(md5sum "$BIN/charsiu_ppl" 2>/dev/null | cut -c1-32)"
echo " threads:             ${CHARSIU_THREADS:-all $(nproc 2>/dev/null) cores}"
echo " corpus:              $CORPUS  ($(md5sum "$CORPUS" 2>/dev/null | cut -c1-32))"
echo " speed REPEAT:        $REPEAT      gemma4 sweep N: $NSWEEP"
[ "$REPEAT" -ge 7 ] || echo " ⚠⚠ REPEAT WAS CUT -- this is a dry run, not a round of record"
[ "$NSWEEP" -ge 20 ] || echo " ⚠⚠ SWEEP WAS CUT -- this is a dry run, not a round of record"
echo

BOOT0=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)
env_block "start"
echo

#
# 1. THE SPEED TABLE. At the performance governor, because the vendor's
#    published column is at maximum CPU and NPU frequency and a comparison
#    across two governors is not one.
#
# ⚠⚠ AND THE VENDOR COLUMN IS STILL A CITATION, NOT AN ARM. It is copied from
# their benchmark.md; nothing here runs their runtime. It has no N and no
# spread, so "inside the noise" cannot be said of it in either direction. The
# only honest form of the claim names both: our median of N with its range,
# against their published point estimate at maximum frequency, dated.
#
echo "== 1. the speed table, performance governor, median of $REPEAT"
CHARSIU_BENCH_PERF=1 CHARSIU_BENCH_REPEAT="$REPEAT" \
	sh "$D/board_vendor.sh" 2>&1 || echo "  ⚠⚠ THE SPEED TABLE FAILED"
echo

#
# 2. THE QUALITY TABLE, on the CPU reference so the hardware path is out of
#    it, at the group the board actually runs. llama_auto_kmax pins that to
#    1024 and is called only when the NPU is on, so CHARSIU_NPU=0 has to be
#    told by hand or it measures one absmax a row -- a quantiser the board
#    never runs, and the configuration every desk number before 2026-09-10
#    was taken at.
#
echo "== 2. quality, CPU reference, group 1024 (what the board runs)"
if [ -f "$QMODEL" ]; then
	#
	# ⚠⚠ AWQ NEEDS ITS STATISTICS AND DECLINES WITHOUT THEM. The first
	# version of this set CHARSIU_NPU_AWQ=0.20 and nothing else, and the
	# board printed `int4 34.2425` and `int4+AWQ 34.2425` -- the same
	# number to the last digit, because with no calibration charsiu
	# refuses the method and says so on a stderr this was discarding.
	# An arm equal to its control to the last digit never ran.
	#
	# ⚠ AND THE CALIBRATION TEXT IS NOT THE EVALUATION TEXT. Calibrating
	# on the passage being scored measures how well the statistics fit
	# that passage, which is not the question.
	#
	CAL=$D/corpus/calib.txt
	[ -f "$CAL" ] || CAL=/opt/charsiu/corpus/calib.txt
	STATS=$(mktemp)
	env CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
	    CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
	    CHARSIU_CALIB="$STATS" \
	    "$BIN/charsiu_ppl" "$QMODEL" "$CAL" -n 150 >/dev/null 2>&1 || true
	if [ -s "$STATS" ]; then
		echo "   calibration      $(wc -c < "$STATS") bytes from $(basename "$CAL")"
	else
		echo "   ⚠⚠ THE CALIBRATION PASS WROTE NOTHING -- the AWQ arm below"
		echo "      cannot run and will equal the int4 arm. Do not read it."
	fi
	INT4=; AWQ=
	for arm in "int4:CHARSIU_NPU_W4V=1" \
	           "int4+AWQ 0.20:CHARSIU_NPU_W4V=1 CHARSIU_NPU_AWQ=0.20 CHARSIU_AWQ_STATS=$STATS" \
	           "int8:CHARSIU_NPU_W4V=0"; do
		lbl=${arm%%:*}; ev=${arm#*:}
		# shellcheck disable=SC2086
		P=$(env CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1 \
		        CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 $ev \
		        "$BIN/charsiu_ppl" "$QMODEL" "$CORPUS" -n 300 2>/dev/null |
		    tail -1 | grep -o 'ppl [0-9.]*' | tail -1 | awk '{print $2}')
		printf '   %-16s %s\n' "$lbl" "${P:-FAILED}"
		case $lbl in
		int4) INT4=$P ;;
		int4+*) AWQ=$P ;;
		esac
	done
	rm -f "$STATS"
	# ⚠⚠ THE TELL. Identical tokens is what BOTH "the knob works" and "the
	# knob never ran" look like, and so is an identical perplexity.
	if [ -n "$INT4" ] && [ "$INT4" = "$AWQ" ]; then
		echo "   ⚠⚠ THE AWQ ARM EQUALS THE int4 ARM TO THE LAST DIGIT."
		echo "      It did not run. AWQ declines without statistics, and a"
		echo "      null arm is not a null result. This table's AWQ row is"
		echo "      VOID for this round."
	fi
else
	echo "   ⚠ no Llama-3.2-1B-Instruct-Q4_0.gguf under [$MODELDIRS]"
	echo "     -- quality table SKIPPED"
fi
echo

#
# 3. GEMMA4'S SPREAD. Seven readings in one build ran 2133, 2182, 2185, 2325,
#    2408, 2707 and 3221 -- which is not a tidy distribution. Two clusters
#    would mean something is SWITCHING, and that is a finding rather than a
#    caveat; a plain tail means the row is unquotable and the paper says so.
#    Either way it needs more than seven, with the temperature beside each one
#    so thermal drift can be told from whatever else.
#
echo "== 3. gemma4 TTFT, $NSWEEP readings, timestamped with temperature"
GM=$(find_model 'gemma-4*Q4_0.gguf' || true)
if [ -n "$GM" ] && [ -x "$BIN/charsiu_run" ]; then
	i=0
	while [ $i -lt "$NSWEEP" ]; do
		i=$((i + 1))
		T=$(for z in /sys/class/thermal/thermal_zone*/temp; do
			[ -e "$z" ] && awk '{printf "%.1f ", $1/1000}' "$z"; done)
		MS=$(env CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
		     "$BIN/charsiu_run" "$GM" -p "Explain in plain words why a written record outlasts a memory." \
		     -n 8 --ignore-eos -c 512 -t 4 2>/dev/null |
		     grep -o 'prompt [0-9]* tok in [0-9]* ms' | head -1 | awk '{print $5}')
		printf '   %2d  %s  ttft %6s ms   temp %s\n' \
			"$i" "$(date +%H:%M:%S)" "${MS:-FAIL}" "$T"
	done
else
	echo "   ⚠ no gemma4 under [$MODELDIRS] or no charsiu_run -- sweep SKIPPED"
	echo "     (charsiu pull gemma4-e2b-q4 puts it in ~/.charsiu/models)"
fi
echo

env_block "end"
BOOT1=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)
echo
#
# ⚠⚠ THE WHOLE POINT, CHECKED. If the board rebooted in the middle -- a panic,
# a watchdog, a power blip on a long sweep -- then this file is two rounds
# wearing one header, which is the exact defect it was written to prevent.
#
if [ "$BOOT0" = "$BOOT1" ]; then
	echo "🏁 ONE BOOT: boot id unchanged, $BOOT0"
else
	echo "⛔⛔ THE BOARD REBOOTED DURING THIS ROUND."
	echo "   start $BOOT0"
	echo "   end   $BOOT1"
	echo "   This file is TWO rounds under one header. Do not quote across it."
fi
} 2>&1 | tee "$OUT"

echo
echo "written to $OUT"
