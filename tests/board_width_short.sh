#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# The width axis writes a SHORT surface. Which register says how long it is?
#
# board_acc_map.sh settled what the width axis is and is not doing. The height
# arm, which is solved, came back exact: 128 of 128 words at m = 2 and 256 of
# 256 at m = 4, nothing absent. The width arm wrote 92 of 128 and 148 of 256,
# and npu_gemm_test's own verdict on it is the opposite of the height one:
#
#   31 values were never computed, so no read order recovers them.
#
# So it is not a permutation. The shape of the shortfall says the same thing:
# inside every (row, 32 channel super group) the width axis writes the FIRST
# FIVE four word runs -- channels 0-3, 16-19, 4-7, 20-23, 8-11, in the same
# interleave the height axis uses -- and then goes to the next group, dropping
# 24-27, 12-15 and 28-31 every time. Five runs of eight. Words written are
# 36 + 28m against the height axis's 64m.
#
# A regular fraction of every group, at every group, is a size or stride
# register and not an address. This sweeps the candidates one field at a time,
# which is the method that worked on 0x4050 and on 0x40b8, and scores each by
# the one number that cannot be argued with: how many words the board wrote.
#
# ⚠ THE BASELINE IS THE CONTROL AND IT RUNS FIRST. It must reproduce 92 of 128.
# If it does not, the build or the shape moved and no row below means anything.
#
# `charsiu update dev` installs this at /opt/charsiu/board_width_short.sh.
#
# Usage: board_width_short.sh
set -u

GEMM=${CHARSIU_GEMM_BIN:-}
[ -n "$GEMM" ] || GEMM=$(command -v npu_gemm_test 2>/dev/null || true)
[ -n "$GEMM" ] || for d in /opt/charsiu /usr/bin ./build .; do
	[ -x "$d/npu_gemm_test" ] && { GEMM="$d/npu_gemm_test"; break; }
done
[ -n "${GEMM:-}" ] || { echo "board_width_short: npu_gemm_test not found" >&2; exit 1; }

OUTDIR=${CHARSIU_BOARD_DIR:-$HOME/charsiu-board}
mkdir -p "$OUTDIR"
K=64; N=64; M=2
WANT=128                          # m * n, the number of words a full surface has

# how many words the board wrote, and how many reference values never appeared.
#
# ⚠ THE ABSENT COUNT IS ON ITS OWN LINE. The first version read it off the
# "of N reference values:" line, which ends at the comma before it, and got the
# empty string for every row -- a whole column of "?" that would have been read
# as the tool not reporting it.
score() {
	if [ -n "$1" ]; then
		env CHARSIU_M_AXIS=w CHARSIU_OVERRIDE="$1" \
		    "$GEMM" "$K" "$N" --read "$M" >"$OUTDIR/ws.txt" 2>&1
	else
		env CHARSIU_M_AXIS=w \
		    "$GEMM" "$K" "$N" --read "$M" >"$OUTDIR/ws.txt" 2>&1
	fi
	rc=$?
	if [ $rc -ne 0 ]; then echo "FAILED -"; return; fi
	w=$(sed -n 's/.*the board wrote \([0-9]*\) of .*/\1/p' "$OUTDIR/ws.txt" | head -1)
	a=$(sed -n 's/^ *\([0-9][0-9]*\) are absent from the buffer altogether.*/\1/p' \
	    "$OUTDIR/ws.txt" | head -1)
	[ -n "$w" ] || { echo "NOMAP -"; return; }
	echo "$w ${a:-?}"
}

echo "binary   $GEMM"
echo "shape    K=$K N=$N m=$M on the WIDTH axis; a full surface is $WANT words"
echo

set -- $(score "")
BASE=$1
# ⚠ THE EXPECTED BASELINE MOVED WHEN THE SWEEP SUCCEEDED. It was 92 of 128
# because 0x40b8 was 3 at every M on this axis. It is 3 * M now, so a correct
# build writes the full surface with no override at all and this script's own
# answer became its baseline. CHARSIU_WS_BASE=92 restores the old expectation
# for a build from before the fix.
EXPECT=${CHARSIU_WS_BASE:-$WANT}
echo "baseline (no override):  wrote $1 of $WANT words, $2 values absent"
if [ "$BASE" != "$EXPECT" ]; then
	echo
	echo "⚠ STOP. The baseline must be $EXPECT of $WANT and it is $BASE."
	echo "  Every row below is measured against it, so nothing here can be"
	echo "  read. If this is a build from before 0x40b8 was fixed, the old"
	echo "  expectation is CHARSIU_WS_BASE=92."
	exit 1
fi
echo
printf '  %-9s %-11s %-9s %-9s %s\n' register value wrote absent verdict
printf '  %-9s %-11s %-9s %-9s %s\n' -------- ----- ----- ------ -------

# ⚠ ONE FIELD AT A TIME, FROM THE BASELINE. Two at once and a pair that helps
# and hurts reads as no change.
#
# The values are ours at m=2 on the width axis, then multiples: 0x1090 is 8,
# 0x1094 is 2, 0x1098 is 4, 0x401c is 2, 0x40b8 is 2, DPU 0x4028 is 0 and
# RDMA 0x5010 is 1. Mesa's generic encoder computes the first three from inw
# and full_inh, which on this axis are M and 1 -- the swap is exactly what is
# under suspicion.
for spec in \
	"0x1090 4 8 16 32 64 128 256" \
	"0x1094 1 2 4 8 16 32 64" \
	"0x1098 1 2 4 8 16 32 64" \
	"0x401c 1 2 4 8 16 32 64" \
	"0x4028 0 1 2 4 8 16 32" \
	"0x40b8 1 2 3 4 6 8 16" \
	"0x5010 1 2 3 4 8 16 32" \
	; do
	# shellcheck disable=SC2086
	set -- $spec
	R=$1; shift
	for V in "$@"; do
		set -- $(score "$R=$V")
		v="-"
		[ "$1" != "FAILED" ] && [ "$1" != "NOMAP" ] && {
			if [ "$1" -gt "$BASE" ] 2>/dev/null; then v="MORE"; fi
			if [ "$1" = "$WANT" ]; then v="FULL SURFACE"; fi
			if [ "$1" -lt "$BASE" ] 2>/dev/null; then v="less"; fi
		}
		printf '  %-9s %-11s %-9s %-9s %s\n' "$R" "$V" "$1" "$2" "$v"
	done
	echo
done

echo "======================================================================"
echo "a row that writes MORE than 92 is the field, and one that reaches 128"
echo "with nothing absent is the answer. A register that changes nothing at"
echo "any value is excluded, which is worth as much and is why the whole"
echo "range is printed rather than only what helped."
echo "  last raw log: $OUTDIR/ws.txt"
echo "======================================================================"
