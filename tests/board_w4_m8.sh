#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# m = 8 is the one width the batched int4 matmul gets wrong. Which of the two
# numbers is doing it?
#
# The fault, as the board reported it: every width the probe sweeps is exact to
# 5.10e-05 -- 2, 4, 16, 32, 48, 64, 80 -- and m = 8 returns 871 rows of 904.
# The 33 that miss are ROW 0 of the n = 8192 tensors, every ffn_gate and
# ffn_up, at that width and no other.
#
# ⚠ IT NEEDS BOTH NUMBERS. m = 8 is exact at n = 512 and n = 2048, and n = 8192
# is exact at every other m. A desktop round retired everything that is a
# function of only one of them:
#
#   the read order       a bijection at all 32 (m, n) the probe runs, and it
#                        does not take n at all, so it cannot be n-selective
#   the input packing    a function of m and k, and gate/up share k = 2048 with
#                        attn_q, attn_o and the head, all exact at m = 8
#   the register stream  separable: of its 148 words none moves with m and n
#                        together, and not one word of the m=8 n=8192 stream is
#                        a word the board has not already run correctly
#   0x40b8               4*T - W on 3328 of 3328 vendor int4 streams, which is
#                        3*M for a single chunk -- confirmed, not fitted
#
# What is left is the output surface, and not its SIZE either: (m=8, n=8192)
# and (m=32, n=2048) are both 262144 bytes and the second is exact.
#
# So this asks the two questions a desktop cannot, one variable each, and both
# arms can come back either way:
#
#   ONEDEV   both K slices of a tensor go to ONE core instead of running
#            concurrently on two. The batched path submits both devices before
#            waiting on either -- concurrency is its design -- and round 362
#            measured two cores corrupting each other through the shared CBUF
#            three times in four. If m = 8 is exact on one core the fault is
#            the PAIR and not the shape.
#
#   NMAX     n = 8192 in two slices of 4096. The widest int4 dispatch anywhere
#            in the vendor's own Llama-3.2-1B .rkllm is 4096 output channels,
#            and ours is the only shape that asks for twice that. If m = 8 is
#            exact the fault is the WIDTH.
#
# ⚠ THE BASELINE RUNS FIRST AND IT MUST FAIL. Both arms are read against it,
# and the refusal in npudev.c means m = 8 does not reach the hardware unless
# CHARSIU_NPU_W4_M8 is set -- a control that cannot reach the thing it is
# controlling for is not a control, which this tree has already paid for once.
#
# ⚠ AND READ THE where-did-it-go LINE, not just the row count. The probe now
# scans the row that missed and says whether its wanted values are SOMEWHERE in
# the batch or absent from it. Absent means the block never wrote them and no
# reading recovers them; somewhere means they were written and misplaced, and
# the deduction above is wrong. Those want different next rounds.
#
# `charsiu update dev` installs this at /opt/charsiu/board_w4_m8.sh.
#
# Usage: board_w4_m8.sh [MODEL.gguf]
set -u

MODEL=${1:-}

RUN=${CHARSIU_RUN_BIN:-}
[ -n "$RUN" ] || RUN=$(command -v charsiu_run 2>/dev/null || true)
[ -n "$RUN" ] || for d in /opt/charsiu /usr/bin "$HOME/charsiu" ./build .; do
	[ -x "$d/charsiu_run" ] && { RUN="$d/charsiu_run"; break; }
done
[ -n "${RUN:-}" ] || { echo "board_w4_m8: charsiu_run not found" >&2; exit 1; }

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
	echo "board_w4_m8: no int4 gguf found in $DIRS" >&2
	echo "  pass one:  board_w4_m8.sh /path/to/model-Q4_0.gguf" >&2
	exit 1
}

OUTDIR=${CHARSIU_BOARD_DIR:-$HOME/charsiu-board}
mkdir -p "$OUTDIR"

# ⚠ 8, NOT 80. Every other width is settled and each one costs a full pass over
# 113 tensors twice -- once a row at a time for the reference. This round is
# about one width, so it caps there and the other seven do not get re-measured.
MMAX=${CHARSIU_PROBE_MMAX:-8}

# the int4 environment board_vendor.sh runs, plus the switch that lets the
# broken width reach the hardware at all
W4_ENV="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 CHARSIU_NPU_W4_M8=1"

echo "model    $MODEL"
echo "binary   $RUN"
echo "probe    --batch-probe $MMAX   (widths 2, 4, 8; 8 is the question)"
echo

for ARM in baseline onedev nmax4096; do
	case $ARM in
	baseline) EXTRA=""
	   label="nothing extra -- this MUST reproduce 871 of 904" ;;
	onedev)   EXTRA="CHARSIU_NPU_ONEDEV=1"
	   label="one core: the two K slices stop running concurrently" ;;
	nmax4096) EXTRA="CHARSIU_NPU_NMAX=4096"
	   label="n = 8192 in two slices of 4096, the vendor's own widest" ;;
	esac
	echo "===== $ARM -- $label ====="
	out="$OUTDIR/w4-m8-$ARM.txt"
	# shellcheck disable=SC2086
	env $W4_ENV $EXTRA "$RUN" "$MODEL" --batch-probe "$MMAX" \
	    >"$out" 2>&1
	rc=$?
	if [ $rc -ne 0 ]; then
		echo "  THE RUN FAILED (exit $rc), last lines:"
		tail -12 "$out" | sed 's/^/    /'
		echo
		continue
	fi
	if grep -q "misses row 0 of the n=8192" "$out"; then
		echo "  THE GATE REFUSED m = 8 -- CHARSIU_NPU_W4_M8 did not reach"
		echo "  the binary, so this arm never ran the width it is about."
		echo
		continue
	fi
	if ! grep -q "batching" "$out"; then
		echo "  THE PROBE NEVER RAN -- no batching table in the output."
		tail -12 "$out" | sed 's/^/    /'
		echo
		continue
	fi
	# ⚠ NO head CAP: the deciding lines are at the BOTTOM of this output,
	# and this tree has lost two rounds to a cap that cut them.
	sed -n '/batching .* layers/,$p' "$out" | sed 's/^/  /'
	echo
done

echo "======================================================================"
echo "what to read, in this order:"
echo
echo "  1. THE BASELINE MUST FAIL. 871 of 904 at m = 8, with MISS lines"
echo "     naming n = 8192 tensors and row 0. If it comes back exact then"
echo "     something else moved and neither arm below means anything."
echo
echo "  2. THE MISS LINES ARE UNCAPPED TO FORTY NOW. 33 rows was written"
echo "     down as every ffn_gate and ffn_up, and that is 32 -- there is a"
echo "     thirty third miss nobody has seen. If it is the output head the"
echo "     rule is 'slices 8192 wide'; if it is something with no 8192 in"
echo "     it, the rule is not the width at all."
echo
echo "  3. THE where-did-it-go LINE, printed for the first row that missed:"
echo "     'N of 8192 wanted values are somewhere in the batch, M of its"
echo "     slots came back exactly zero'. Near zero present means the block"
echo "     never wrote that row and the surface is short; near 8192 present"
echo "     means it was written and misplaced, and the read order is back"
echo "     in question after all."
echo
echo "  4. THEN THE TWO ARMS. Exactly one of them going exact at m = 8 is"
echo "     the answer; both staying wrong says it is neither the core pair"
echo "     nor the width, and the next question is the surface itself."
echo
echo "  ⚠ ONEDEV HALVES THE HARDWARE, so its tok/s and its us-a-row are not"
echo "  comparable with the others. This arm is about the ROWS column."
echo
echo "  full logs: $OUTDIR/w4-m8-{baseline,onedev,nmax4096}.txt"
echo "======================================================================"
