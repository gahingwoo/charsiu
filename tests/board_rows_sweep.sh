#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# How many rows can the batched matmul actually do?
#
# ⚠ THE ANSWER IS 32 AND THE QUESTION HAS NEVER BEEN ASKED ABOVE IT.
# charsiu_npu_matmul was checked value for value at m = 2 to 32. The towers hand
# it 1024 and 1500, and the first board round that did so produced a picture the
# model called "I am not sure" and an EMPTY transcript -- while CLIP, whose tower
# is fifty rows, was right. Fifty is inside the range and a thousand is not.
#
# Mesa's arithmetic says where it should break: the CBUF budget test fires above
# m = 320 and the row split above m = 640. That is a prediction, and this is the
# measurement.
#
# ⚠⚠ THE ORACLE IS THE NPU AT ONE ROW, NOT THE CPU. The tower on the hardware is
# int8 and on the CPU it is f32, so the two differ by the quantisation whatever
# the batch does -- the board's CLIP scored 0.2662 against the CPU's 0.2745 and
# was RIGHT. Comparing against the CPU would call that a failure at every width
# and hide the one that matters. One row at a time is the same quantisation, the
# same weights and the same kernel, so a difference against IT is the batch.
#
# The CPU line is still printed, as the size of the quantisation for scale.
#
#   sh tests/board_rows_sweep.sh MMPROJ.gguf
set -eu

MM=${1:-}
if [ -z "$MM" ]; then
	for d in "$HOME/charsiu-board" "$HOME/.charsiu/models" /opt/charsiu/models; do
		[ -f "$d/mmproj.gguf" ] && { MM="$d/mmproj.gguf"; break; }
	done
fi
[ -n "$MM" ] && [ -f "$MM" ] || { echo "usage: board_rows_sweep.sh MMPROJ.gguf" >&2; exit 1; }

VIS=""
for d in /usr/bin /opt/charsiu "$PWD/build"; do
	[ -x "$d/charsiu_vision" ] && { VIS="$d/charsiu_vision"; break; }
done
[ -n "$VIS" ] || { echo "charsiu_vision not found" >&2; exit 1; }

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
ms() { awk '{printf "%d", $1 * 1000}' /proc/uptime; }

echo "the same tower on the CPU, for scale"
t0=$(ms)
"$VIS" "$MM" --encode > "$T/cpu.txt"
echo "  $(awk -v m="$(( $(ms) - t0 ))" 'BEGIN{printf "%.1f", m/1000}') s, $(wc -l < "$T/cpu.txt") values"

# ⚠⚠ TWO ROWS, NOT ONE. charsiu_npu_matmul REFUSES m = 1 -- a decode has its own
# path -- so ROWS_MAX=1 falls back to the CPU for every chunk, and the first run
# of this sweep printed "int8 against the CPU's f32: 0.000e+00" and did not stop.
# A reference that is secretly the thing it is a reference FOR is worse than no
# reference: every row then DIFFERS by the quantisation and the real break is
# just another number in the column.
#
# The check below is what catches it: the reference must NOT equal the CPU.
echo "the reference: the same tower on the NPU, TWO rows at a time"
t0=$(ms)
CHARSIU_NPU=1 CHARSIU_NPU_ROWS_MAX=2 "$VIS" "$MM" --encode > "$T/ref.txt" || {
	echo "  the NPU run failed; there is nothing to sweep" >&2; exit 1; }
T1=$(( $(ms) - t0 ))
echo "  $(awk -v m="$T1" 'BEGIN{printf "%.1f", m/1000}') s"
Q=$(paste "$T/cpu.txt" "$T/ref.txt" | awk 'NR==1{next}
	{d = $1 - $2; if (d < 0) d = -d; if (d > w) w = d} END{printf "%.6f", w}')
echo "  int8 against the CPU's f32: $Q"
if [ "$Q" = "0.000000" ]; then
	echo >&2
	echo "  STOP. int8 cannot equal f32 to the last bit, so this reference is" >&2
	echo "  the CPU: the hardware refused every chunk and nothing said so." >&2
	echo "  There is nothing to sweep until that is fixed." >&2
	exit 1
fi
echo
printf '%8s  %10s  %12s  %s\n' rows seconds "vs one row" verdict

for R in 4 8 16 32 48 64 80 96 112 128 160 256 512 1024; do
	t0=$(ms)
	if ! CHARSIU_NPU=1 CHARSIU_NPU_ROWS_MAX=$R "$VIS" "$MM" --encode \
			> "$T/npu.txt" 2>"$T/err.txt"; then
		printf '%8s  %10s  %12s  %s\n' "$R" "-" "-" "the run failed"
		continue
	fi
	SEC=$(awk -v m="$(( $(ms) - t0 ))" 'BEGIN{printf "%.1f", m/1000}')
	# ⚠ line 1 is the shape; compare the numbers only
	D=$(paste "$T/ref.txt" "$T/npu.txt" | awk 'NR==1{next}
		{d = $1 - $2; if (d < 0) d = -d; if (d > w) w = d}
		END{printf "%.6f", w}')
	# ⚠ EXACT. Same weights, same quantisation, same kernel: only the number
	# of rows in a submit differs, and that must not change an output at all.
	V=$(awk -v d="$D" 'BEGIN{print (d == 0) ? "identical" : "DIFFERS"}')
	printf '%8s  %10s  %12s  %s\n' "$R" "$SEC" "$D" "$V"
done
echo
echo "The first row that DIFFERS is the bound. The sweep before this one put it"
echo "between 64 and 128 -- 2, 8, 32 and 64 gave the SAME output and 128 did not"
echo "-- which is nowhere near the 320 and 640 Mesa's arithmetic predicts, so the"
echo "prediction is about something else. This one steps 16 at a time through"
echo "that gap to say which width is the last good one."
