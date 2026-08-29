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

echo "model    $MODEL"
echo "binary   $RUN"
echo "probe    --batch-probe 32   (m = 2, 4, 8, 16, 32, checked then timed)"
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

for AXIS in h w; do
	case $AXIS in
	h) label="height -- the axis five rounds proved writes ONE row" ;;
	w) label="width  -- the axis the vendor's own streams use" ;;
	esac
	echo "===== M axis: $AXIS -- $label ====="
	out="$OUTDIR/w4-axis-$AXIS.txt"
	# ⚠ THE CONTROL MUST REACH THE HARDWARE. Round one set W4_BATCH=1 on
	# both arms, and on the height arm the gate in npudev.c refused it --
	# so the arm that had to fail failed in software and said nothing about
	# the silicon. "height" is the value that lets the wrong axis through
	# on purpose.
	case $AXIS in h) GATE=height ;; *) GATE=1 ;; esac
	# shellcheck disable=SC2086
	env $W4_ENV CHARSIU_M_AXIS="$AXIS" CHARSIU_NPU_W4_BATCH="$GATE" \
	    "$RUN" "$MODEL" --batch-probe 32 >"$out" 2>&1
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
	sed -n '/batching .* layers/,$p' "$out" | head -18 | sed 's/^/  /'
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
echo "  full logs: $OUTDIR/w4-axis-h.txt and $OUTDIR/w4-axis-w.txt"
echo "======================================================================"
