#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# Does an int4 weight matmul do more than one row when M is on the WIDTH axis?
#
# Five rounds established that w4a16 computes exactly one row, and every one of
# them ran with M on the HEIGHT: one column, M rows. The vendor's own
# Llama-3.2-1B .rkllm turns out to carry 3328 int4 streams of which 2816 are
# batched, at M of 16, 24, 32, 40, 48, 64 and 80, all of them ONE ROW HIGH and
# M PIXELS WIDE. Those streams read as M = 1 to anything that takes the row
# count, which is how the file was made to say the vendor never batches int4.
#
# tools/cmp_vendor.py compares charsiu's stream to theirs on a desktop, and on
# the width axis it is now 2 registers away at M = 32 and 64 -- fewer than at
# M = 1, where this hardware is known to be right. This is the board's half.
#
# ⚠ IT CHECKS BEFORE IT TIMES. llama_batch_probe compares every row against the
# one row path and prints which rows agree, so a fast wrong answer cannot pass:
# this tree has already shipped one, an int4 prompt that came back
# "ITES  (un- a- -  ( -  ' \ l'" at a very respectable 37.46 tok/s.
#
# ⚠ AND IT RUNS THE HEIGHT AXIS FIRST, as the control. If the height arm also
# reports rows agreeing then the probe is not discriminating and neither arm
# means anything -- five rounds say it must fail.
#
# `charsiu update dev` installs this at /opt/charsiu/board_w4_axis.sh.
#
# Usage: board_w4_axis.sh [MODEL.gguf]
set -u

MODEL=${1:-}

# --- the binary ------------------------------------------------------------
RUN=${CHARSIU_RUN_BIN:-}
[ -n "$RUN" ] || RUN=$(command -v charsiu_run 2>/dev/null || true)
[ -n "$RUN" ] || for d in /opt/charsiu /usr/bin "$HOME/charsiu" ./build .; do
	[ -x "$d/charsiu_run" ] && { RUN="$d/charsiu_run"; break; }
done
[ -n "${RUN:-}" ] || { echo "board_w4_axis: charsiu_run not found" >&2; exit 1; }

# --- the model -------------------------------------------------------------
# ⚠ int4 AND llama, the same file prefill_control.sh uses, so the two rounds
# are about one model.
DIRS="$HOME/.charsiu/models $HOME/models /opt/charsiu/models"
if [ -z "$MODEL" ]; then
	for pat in '*Llama-3.2*Q4_0*.gguf' '*llama*Q4_0*.gguf' '*Q4_0*.gguf'; do
		for d in $DIRS; do
			for f in $d/$pat; do
				[ -r "$f" ] && { MODEL="$f"; break 3; }
			done
		done
	done
fi
[ -n "${MODEL:-}" ] && [ -r "$MODEL" ] || {
	echo "board_w4_axis: no int4 gguf found in $DIRS" >&2
	echo "  pass one:  board_w4_axis.sh /path/to/model-Q4_0.gguf" >&2
	exit 1
}

OUTDIR=${CHARSIU_BOARD_DIR:-$HOME/charsiu-board}
mkdir -p "$OUTDIR"
# ⚠ 80, NOT 32. The probe's widths are 2 4 8 16 32 48 64 80 and this argument
# CAPS them. It said 32 for one round after the widths were extended, so the
# round that existed to reach 48, 64 and 80 never left 32.
MMAX=${CHARSIU_PROBE_MMAX:-80}

echo "model    $MODEL"
echo "binary   $RUN"
echo "probe    --batch-probe $MMAX   (widths up to that, checked then timed)"
echo

# ⚠⚠ THE WHOLE int4 ENVIRONMENT, NOT JUST THE AXIS. This is the set
# board_vendor.sh runs and it is the one that puts the run on the int4 path at
# all; CHARSIU_NPU_W4V=1 alone with no CHARSIU_NPU=1 stages nothing, and the
# probe then says "no NPU staged" and exits 1 -- a round that fails loudly is
# the good case, and a round that quietly measures the CPU is the one this
# tree keeps shipping.
W4_ENV="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"

# ⚠ AND THE READ ORDER'S SECOND HALF, which the in place scan pointed at.
# Row 0 agrees on EXACTLY half its channels and the first it misses is 16;
# charsiu_acc_index splits the channel as a = (c%32)/16, so a = 0 is exactly
# half of them and 16 is the first of the other half. `a * 4` is the only term
# that places that half. Of the whole family only a*4 and the two halves
# SWAPPED are permutations at all, so this is a two horse race and one of them
# is the control.
for AXIS in h w; do
	case $AXIS in
	h) label="height -- the control, and it must still disagree"
	   AX=h; ACC= ;;
	w) label="width -- the DEFAULT read order now picks roleswap2 for w4a16"
	   AX=w; ACC= ;;
	esac
	echo "===== M axis: $AXIS -- $label ====="
	out="$OUTDIR/w4-axis-$AXIS.txt"
	# ⚠ THE CONTROL MUST REACH THE HARDWARE. Round one set W4_BATCH=1 on
	# both arms, and on the height arm the gate in npudev.c refused it --
	# so the arm that had to fail failed in software and said nothing about
	# the silicon. "height" is the value that lets the wrong axis through
	# on purpose.
	# ⚠ int4 batching is ON BY DEFAULT now, so only the control needs a
	# switch: "height" is what lets the arm that must fail reach the
	# hardware at all.
	case $AX in h) GATE=height ;; *) GATE= ;; esac
	# shellcheck disable=SC2086
	env $W4_ENV CHARSIU_M_AXIS="$AX" ${GATE:+CHARSIU_NPU_W4_BATCH="$GATE"} \
	    ${ACC:+CHARSIU_ACC_A="$ACC"} \
	    "$RUN" "$MODEL" --batch-probe "$MMAX" >"$out" 2>&1
	rc=$?
	if [ $rc -ne 0 ]; then
		echo "  THE RUN FAILED (exit $rc), last lines:"
		tail -12 "$out" | sed 's/^/    /'
		echo
		continue
	fi
	if grep -q "int4 computes one row" "$out"; then
		echo "  THE GATE REFUSED THIS ARM -- it never reached the NPU."
		echo "  A control that cannot run is not a control; nothing here"
		echo "  is a statement about the hardware."
		echo
		continue
	fi
	if ! grep -q "batching" "$out"; then
		echo "  THE PROBE NEVER RAN -- no batching table in the output."
		echo "  That is the silent case: exit 0 and nothing measured."
		tail -12 "$out" | sed 's/^/    /'
		echo
		continue
	fi
	# ⚠ NO head CAP. This had one at 18 lines and the landing table is
	# exactly 18 lines long, so the round came back with row 0 sampled to
	# channel 384 and nothing else -- no row 1, and none of the timing
	# table either. The round before that lost m = 8 to a head -80 in
	# board_acc_map.sh. Twice is a pattern: the deciding output is at the
	# BOTTOM of what these tools print, and a cap is a silent cut.
	sed -n '/batching .* layers/,$p' "$out" | sed 's/^/  /'
	echo
done

echo "======================================================================"
echo "what to read: the HEIGHT arm must show rows DISAGREEING. It is the"
echo "control -- if it passes, the probe is not discriminating and the width"
echo "arm means nothing either."
echo
echo "if the width arm is close but not exact, the knob to sweep next is"
echo "CHARSIU_DPU_40B8: the vendor writes 3*M on its int4 projections and 7*M"
echo "on its int8 head, and this tree writes M for both. job.c already takes"
echo "the override, so it costs one more pass and no rebuild."
echo
echo "  the line that decides it is 'rowN in place: X of 2048 channels agree'."
echo "  1024 of 2048 is ONE QUADRANT of (row, half) and nothing else. 2048 of"
echo "  2048 on BOTH rows is the read order solved."
echo
echo "  roleswap2 is the read order and it is the DEFAULT now: charsiu_acc_index"
echo "  takes the format, and w4a16 on the width axis is the one case that"
echo "  reads differently. int8 keeps a*4 on both axes, which is what its own"
echo "  raw surface map said. No environment variable is needed for either."
echo
echo "  Last round, with the switch set by hand: 226 of 226 at m=2, 452 of 452"
echo "  at m=4, 1808 of 1808 at m=16 and 3616 of 3616 at m=32, worst relative"
echo "  5.1e-05 at every one. This round should reproduce that with nothing"
echo "  set, and it reaches 48, 64 and 80 as well -- the widths a real prompt"
echo "  actually hands it, which have never been asked."
echo
echo "  ⚠ m = 8 IS THE ONE THAT BENDS: 871 of 904, worst 3.1e+04, where every"
echo "  other width is exact. It has been four to six orders out in every arm"
echo "  of every round. Nothing here explains it and the refusal stays until"
echo "  something does."
echo
echo "  full logs: $OUTDIR/w4-axis-{h,w}.txt"
echo "======================================================================"
