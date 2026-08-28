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
# M = 1, where the hardware is known to be right. This is the board's half.
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
#   sh tests/board_w4_axis.sh
set -u

DIR=${CHARSIU_BOARD_DIR:-$HOME/charsiu-board}
mkdir -p "$DIR"
MODELS=${CHARSIU_MODELS:-$HOME/.charsiu/models}
[ -d "$MODELS" ] || MODELS=/opt/charsiu/models
RUN=${CHARSIU_RUN:-/opt/charsiu/charsiu_run}
MODEL=${CHARSIU_MODEL:-}

if [ -z "$MODEL" ]; then
	for f in "$MODELS"/*.gguf; do
		[ -e "$f" ] || continue
		MODEL="$f"
		break
	done
fi
[ -n "$MODEL" ] || { echo "no model in $MODELS"; exit 1; }
[ -x "$RUN" ] || { echo "no $RUN"; exit 1; }

echo "model  $MODEL"
echo "run    $RUN"
echo

# ⚠ CHARSIU_NPU_W4V=1 IS THE INT4 PATH. Without it this measures int8 and the
# question is not asked at all.
for AXIS in h w; do
	case $AXIS in
	h) label="height (the axis five rounds proved writes one row)" ;;
	w) label="width  (the axis the vendor's own stream uses)" ;;
	esac
	echo "===== M axis: $AXIS -- $label ====="
	out="$DIR/w4-axis-$AXIS.txt"
	env CHARSIU_NPU_W4V=1 \
	    CHARSIU_M_AXIS="$AXIS" \
	    CHARSIU_NPU_W4_BATCH=1 \
	    "$RUN" -m "$MODEL" --batch-probe 32 >"$out" 2>&1
	rc=$?
	if [ $rc -ne 0 ]; then
		echo "  THE RUN FAILED (exit $rc), last lines:"
		tail -8 "$out" | sed 's/^/    /'
		echo
		continue
	fi
	# the probe's own table; "rows that agree" is the column that decides
	sed -n '/batching .* layers/,$p' "$out" | head -14 | sed 's/^/  /'
	echo
done

echo "======================================================================"
echo "what to read: the HEIGHT arm must report rows disagreeing -- that is"
echo "the control, and if it passes the probe is not discriminating. Only"
echo "then does the WIDTH arm mean anything."
echo "  full logs: $DIR/w4-axis-h.txt and $DIR/w4-axis-w.txt"
echo "======================================================================"
