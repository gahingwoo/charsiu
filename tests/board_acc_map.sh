#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# Where does the accumulator put a value when M is on the WIDTH axis?
#
# The first width axis round answered the question it was asked and opened a
# different one. On the width axis the hardware BATCHES: at m = 2 it returns
# 2526 of 4096 wanted values, row 0's leading channels are bit identical to the
# one row path, and the speedup is real -- 1.58x at m = 2 rising to 3.47x at
# m = 32. On the height axis the same probe never got past the refusal, and
# five rounds before it had row 1 matching row 0 in 1 of 2048.
#
# So the arithmetic is happening and the values land somewhere this tree does
# not look. charsiu_acc_index() is the read order, and it was solved on the
# HEIGHT axis: P = m/2, super groups of 32, from maps printed at m = 2 and 4.
# The width axis swaps the two image axes, so it needs its own map -- and the
# int8 sweep says the same thing from the other side, differing at EVERY row
# count from 4 up where the height axis is exact to 80.
#
# npu_gemm_test --read prints the permutation itself rather than a candidate
# for it. That is what solved the height axis and it is what this asks for.
#
# ⚠ THE HEIGHT ARM IS THE CONTROL AND IT IS SOLVED. It must come back EXACT.
# If it does not, the shape or the build moved and the width map cannot be
# read against anything.
#
# ⚠ THE WIDTH ARM WROTE A SHORT SURFACE THE FIRST TIME THIS RAN, 92 of 128
# words, and that was 0x40b8: it is 3 * the batch count, and `rows` -- which is
# what the code multiplied -- is 1 on the width axis. Swept on the board, 6 at
# m = 2 writes all 128 with nothing absent. The default now computes it from
# whichever of ow and rows carries M, so this run should come back FULL, and
# the map underneath it is the thing that was never legible before.
#
# ⚠ AND IT RUNS m = 8 TOO. 3 * M is confirmed at m = 1, 2 and 4 on the height
# axis and at m = 2 on the width. m = 8 is a width it has not been seen at,
# which is the whole point of asking.
#
# ⚠ K = 64 N = 64 ON PURPOSE. locate() reports the FIRST index holding a value,
# so a reference with few distinct values answers a question it was not asked.
# At this shape it has 112 distinct in 128, and it prints that count itself.
#
# `charsiu update dev` installs this at /opt/charsiu/board_acc_map.sh.
#
# Usage: board_acc_map.sh
set -u

GEMM=${CHARSIU_GEMM_BIN:-}
[ -n "$GEMM" ] || GEMM=$(command -v npu_gemm_test 2>/dev/null || true)
[ -n "$GEMM" ] || for d in /opt/charsiu /usr/bin ./build .; do
	[ -x "$d/npu_gemm_test" ] && { GEMM="$d/npu_gemm_test"; break; }
done
[ -n "${GEMM:-}" ] || {
	echo "board_acc_map: npu_gemm_test not found." >&2
	echo "  it ships on the dev channel: charsiu update dev" >&2
	exit 1
}

OUTDIR=${CHARSIU_BOARD_DIR:-$HOME/charsiu-board}
mkdir -p "$OUTDIR"
K=${CHARSIU_MAP_K:-64}
N=${CHARSIU_MAP_N:-64}
MS=${CHARSIU_MAP_MS:-"2 4 8"}

echo "binary   $GEMM"
echo "shape    K=$K N=$N   (112 distinct values in 128 -- the map is readable)"
echo

for AXIS in h w; do
	case $AXIS in
	h) label="height -- SOLVED, this arm must come back EXACT" ;;
	w) label="width  -- the map to be read" ;;
	esac
	for M in $MS; do
		echo "===== axis $AXIS, m=$M -- $label ====="
		out="$OUTDIR/accmap-$AXIS-m$M.txt"
		env CHARSIU_M_AXIS="$AXIS" "$GEMM" "$K" "$N" --read "$M" \
			>"$out" 2>&1
		rc=$?
		if [ $rc -ne 0 ]; then
			echo "  THE RUN FAILED (exit $rc), last lines:"
			tail -12 "$out" | sed 's/^/    /'
			echo
			continue
		fi
		if ! grep -q "distinct values in" "$out"; then
			echo "  NO MAP WAS PRINTED -- exit 0 and nothing measured."
			tail -12 "$out" | sed 's/^/    /'
			echo
			continue
		fi
		# ⚠ THE VERDICT BEFORE THE MAP, NOT AFTER IT. This printed the
		# map with head -80 and the summary lines come AFTER the map,
		# so at m = 8 -- where the map is 512 words -- the counts were
		# cut off on BOTH arms and that width was silently not
		# measured. The three lines that decide the round are grepped
		# out first now, and the map is what gets truncated.
		grep "distinct values in" "$out" | sed 's/^/  /'
		grep "are absent from the buffer" "$out" | sed 's/^/  /'
		grep "the board wrote" "$out" | sed 's/^/  /'
		grep "never computed, so no read order" "$out" | sed 's/^/  ⚠ /'
		echo "  --- the map (first 40 rows of it) ---"
		sed -n '/the whole output, as the/,$p' "$out" \
			| head -44 | sed 's/^/  /'
		echo
	done
done

echo "======================================================================"
echo "the height arm is solved and must read EXACT. What to look for on the"
echo "width arm, in order:"
echo "  1. does it write the FULL surface now -- m*n words, 0 absent -- at"
echo "     m = 2, 4 AND 8? 3*M is confirmed at 1, 2, 4 on the height axis and"
echo "     at 2 on the width; m = 8 is the width it has not been asked at."
echo "  2. if it does, the map underneath IS the read order, and that is the"
echo "     last thing between here and a batched int4 prefill."
echo "a full surface is necessary and not sufficient: every value being"
echo "somewhere is not the same as this tree knowing where."
echo
echo "  full logs: $OUTDIR/accmap-{h,w}-m{2,4,8}.txt"
echo "  send all six; the height maps at m=2 and m=4 are what the current"
echo "  expression was fitted on, so they are the check on the check."
echo "======================================================================"
