#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# The overlap fault at the element: which row, which output channels, which
# core's slot.
#
# board_intermittent.sh says whether the TEXT is wrong when both cores are in
# flight at a width, 16 runs a cell, and four days of it drew this map on
# phi3 at KMAX 2048:
#
#   wrong   8, 10, 22 (1 of 16), 24 (3 to 15 of 16), a TAIL of 24 (1 of 16)
#   clean   12, 14, 16, 18, 20, 26, and every width 28..80 that was asked
#   and at 24: KMAX 1024 wrong 13 of 16, KMAX 4096 clean 16 of 16
#
# Text cannot say which numbers were wrong or which core produced them. This
# runs --batch-probe at ONE width on the real weights, every staged tensor
# batched against the m = 1 path row by row, and under each MISS the probe
# now prints the wrong channel span, the n slice that covers it, the core that
# slice was dealt to, and the cores its K slices ran on.
#
# Two arms:
#   serial     the shipped default, both cores never at once: MUST be clean
#   parallel   CHARSIU_NPU_BATCH_PARALLEL=1, PASSES times, because the fault
#              is intermittent and one pass over 225 tensors is about one
#              prompt's worth of width 24 calls
#
#   CHARSIU_OVL_WIDTH=24     the width (one; the fault has one)
#   CHARSIU_OVL_KMAX=2048    the shipped K slice; 1024 is worse, 4096 is clean
#   CHARSIU_OVL_PASSES=4     parallel passes
#   CHARSIU_OVL_MAXT=        cap staged tensors a pass (CHARSIU_PROBE_MAXT)
#   CHARSIU_OVL_EXTRA=       more env for every arm; the three that separate
#                            fd, physical core and CBUF window:
#         CHARSIU_CBUF_SWAP=1           device 0 takes window 1, device 1 window 0
#         CHARSIU_NPU_SUBMIT_FIRST=1    device 1 is submitted first (lands on core 0)
#         CHARSIU_CBUF_W1_BANK=8        window 1 starts at bank 8 (0x2000) not 7;
#                                       6 overlaps window 0 and should fail worse
#
# `charsiu update dev` installs this at /opt/charsiu/board_overlap_slots.sh.
#
# Usage: board_overlap_slots.sh [MODEL.gguf]
set -u

MODEL=${1:-}
W=${CHARSIU_OVL_WIDTH:-24}
KMAX=${CHARSIU_OVL_KMAX:-2048}
PASSES=${CHARSIU_OVL_PASSES:-4}

RUN=${CHARSIU_RUN_BIN:-}
[ -n "$RUN" ] || RUN=$(command -v charsiu_run 2>/dev/null || true)
[ -n "$RUN" ] || for d in /opt/charsiu /usr/bin "$HOME/charsiu" ./build .; do
	[ -x "$d/charsiu_run" ] && { RUN="$d/charsiu_run"; break; }
done
[ -n "${RUN:-}" ] || { echo "board_overlap_slots: charsiu_run not found" >&2; exit 1; }

DIRS="$HOME/.charsiu/models $HOME/models /opt/charsiu/models"
if [ -z "$MODEL" ]; then
	for pat in '*Phi-3*Q4_0*.gguf' '*phi*Q4_0*.gguf' '*Q4_0*.gguf'; do
		for d in $DIRS; do
			for f in $d/$pat; do
				[ -r "$f" ] && { MODEL="$f"; break 3; }
			done
		done
	done
fi
[ -n "${MODEL:-}" ] && [ -r "$MODEL" ] || {
	echo "board_overlap_slots: no int4 gguf found in $DIRS" >&2
	echo "  pass one:  board_overlap_slots.sh /path/to/model-Q4_0.gguf" >&2
	exit 1
}

OUTDIR=${CHARSIU_BOARD_DIR:-$HOME/charsiu-board}
mkdir -p "$OUTDIR"

# the int4 environment board_intermittent.sh runs, at the K slice asked for
W4="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
CHARSIU_NPU_KMAX=$KMAX CHARSIU_NPU_W4_GROUP=1024 \
CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 \
CHARSIU_PROBE_WIDTHS=$W ${CHARSIU_OVL_MAXT:+CHARSIU_PROBE_MAXT=$CHARSIU_OVL_MAXT} ${CHARSIU_OVL_EXTRA:-}"

echo "model    $MODEL"
echo "binary   $RUN"
_ksha=$(sha256sum /boot/Image 2>/dev/null | cut -c1-8)
case "$_ksha" in
c0772d2a) _kname="August release (latest): rocket attaches the IOMMU per job" ;;
5418f487) _kname="v11-control: v11 as sent + second core, no rocket patches" ;;
2422dc11) _kname="attach-once-v11: v11 + attach-once + hardirq completion" ;;
"")       _kname="no /boot/Image readable" ;;
*)        _kname="not a release this script knows" ;;
esac
echo "kernel   $(uname -r) built $(uname -v | sed 's/^#[0-9]* *//; s/SMP PREEMPT *//'), Image ${_ksha:-?} = $_kname"
[ -f /boot/Image ] && [ /boot/Image -nt /proc/1 ] && \
	echo "⚠⚠ /boot/Image is NEWER THAN THIS BOOT: the kernel running is the one before it"
echo "config   width $W, KMAX $KMAX, serial once then parallel x$PASSES${CHARSIU_OVL_EXTRA:+, extra: $CHARSIU_OVL_EXTRA}"
echo

# one pass: the arm's env, the pass label; prints the width line and the misses
pass() {
	arm=$1; n=$2; extra=$3
	out="$OUTDIR/overlap-$W-$arm-$n.txt"
	# shellcheck disable=SC2086
	env $W4 $extra "$RUN" "$MODEL" --batch-probe "$W" >"$out" 2>&1
	rc=$?
	if [ $rc -ne 0 ]; then
		echo "  $arm $n: THE RUN FAILED (exit $rc), last lines:"
		tail -8 "$out" | sed 's/^/    /'
		return 1
	fi
	if ! grep -q "^ *$W  " "$out"; then
		echo "  $arm $n: THE PROBE NEVER RAN THE WIDTH -- no width $W line:"
		tail -8 "$out" | sed 's/^/    /'
		return 1
	fi
	# the width line: m, tensors, worst, rows ok of total, ...
	grep "^ *$W  " "$out" | sed "s/^/  $arm $n: /"
	grep -c "^ *MISS " "$out" | sed "s/^/       MISS lines: /"
	awk '/^ *MISS /{p=1; print; next} /^ *(wrong channels|n slice|\()/{if (p) print; next} {p=0}' "$out" | sed 's/^/    /'
	return 0
}

echo "===== serial (shipped): the control, and it must be clean ====="
pass serial 1 "" || exit 1
echo
echo "===== parallel: CHARSIU_NPU_BATCH_PARALLEL=1, $PASSES passes ====="
i=1
while [ "$i" -le "$PASSES" ]; do
	pass parallel "$i" "CHARSIU_NPU_BATCH_PARALLEL=1"
	echo
	i=$((i + 1))
done

echo "======================================================================"
echo "tally over the parallel passes (MISS lines, capped at 40 a pass):"
cat "$OUTDIR"/overlap-"$W"-parallel-*.txt 2>/dev/null | awk '
/^ *MISS /       { miss++; for (i = 1; i <= NF; i++) if ($i == "row") rows[$(i+1)]++ }
/^ *n slice /    { for (i = 1; i <= NF; i++) { if ($i == "slice") ns[$(i+1)]++; if ($i == "core") core[$(i+1)":"]++ } }
/^ *\([0-9]+,[0-9]+\) want / {
	# the K slice whose word is off: the one furthest from the others, by device
	big = 0; bigd = ""
	for (i = 1; i <= NF; i++) if ($i ~ /^k[0-9]+\/core[0-9]+$/) {
		v = $(i+1) + 0; if (v < 0) v = -v
		if (v > big) { big = v; bigd = $i; sub(/^k[0-9]+\//, "", bigd) }
		if ($(i+2) == "|") break
	}
	if (bigd != "") baddev[bigd]++
}
/STALE READ/      { stale++ }
/HARDWARE WROTE/  { hw++ }
/NEITHER, the fresh/ { neither++ }
END {
	printf "  misses %d\n", miss
	printf "  by row:"; for (r in rows) printf " row %s x%d", r, rows[r]; printf "\n"
	printf "  by n slice:"; for (n in ns) printf " slice %s x%d", n, ns[n]; printf "\n"
	printf "  by core the n slice ran on:"; for (c in core) printf " core %s x%d", c, core[c]; printf "\n"
	printf "  THE DEVICE HOLDING THE WRONG K SLICE WORD:"; for (c in baddev) printf " %s x%d", c, baddev[c]; printf "\n"
	printf "  the wrong word re-read after a cache invalidate: right (STALE READ) x%d, same wrong (HARDWARE WROTE IT) x%d, a third value x%d\n", stale, hw, neither
}'
echo
echo "what to read:"
echo "  1. the serial control's rows must be all of them. If not, the probe"
echo "     itself is wrong at this width and nothing below is about overlap."
echo "  2. a parallel pass with fewer rows is the fault at the element."
echo "     Under each MISS: the wrong channel span, the n slice covering it"
echo "     and its core, and the cores of that slice's K slices."
echo "  3. one core, one n slice, one row index every time is a slot story;"
echo "     whole rows across both cores is a surface or a fence story."
echo "  4. THE LAST TALLY LINE IS THE VERDICT. Under each wrong element the"
echo "     probe prints every K slice's word as the gather saw it and again"
echo "     after invalidating the CPU's cache of the buffer. Fresh sum RIGHT:"
echo "     the CPU read a stale line, a fence or cache story on our side."
echo "     Fresh sum the SAME wrong number: the hardware wrote it, and no"
echo "     amount of software reads it right."
echo "  full logs: $OUTDIR/overlap-$W-{serial,parallel}-*.txt"
echo "======================================================================"
