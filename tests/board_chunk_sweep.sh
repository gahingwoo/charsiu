#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# Which prefill chunk widths give the RIGHT TEXT, on the board?
#
# ⚠⚠ WHY THIS AND NOT ANOTHER PROBE ROUND.
#
# board_w4_axis.sh says phi3's batched matmul is exact: 225 tensors, every
# width it asks about, 18000 of 18000 rows at m = 80, worst relative 1.61e-04.
# And phi3's batched TEXT is wrong on this board. Both are true, so the fault
# is at a width the probe never asks about.
#
# phi3's prompt is 87 tokens. At the default chunk of 32 that is
#
#     32, 32, 23
#
# and the probe's list is 2 4 8 16 32 48 64 80. TWENTY THREE has never been
# asked. m = 8 was only ever found because 8 happened to be in the list.
#
# This asks with the only oracle that cannot be fooled -- the model's own text
# against its own token loop -- and it varies the one thing that changes which
# widths run: CHARSIU_PREFILL_CHUNK. A chunk that DIVIDES the prompt runs one
# width and no tail; a chunk that does not runs that width and a tail of
# whatever is left.
#
# ⚠ CHARSIU_BATCH_FORCE IS A PROBE SWITCH. These models are refused; a number
# measured here is a number about a model that is still refused.
#
# `charsiu update dev` installs this at /opt/charsiu/board_chunk_sweep.sh.
#
#   sh board_chunk_sweep.sh [MODEL-or-substring]
#
#   CHARSIU_CHUNKS="32 31 30 29 24 16 4 2"   chunk widths to try
#   CHARSIU_CHUNK_NGEN=8                     tokens generated
set -u

RUN=${CHARSIU_RUN_BIN:-}
[ -n "$RUN" ] || RUN=$(command -v charsiu_run 2>/dev/null || true)
[ -n "$RUN" ] || for d in /opt/charsiu /usr/bin "$PWD/build" ./build .; do
	[ -x "$d/charsiu_run" ] && { RUN="$d/charsiu_run"; break; }
done
[ -n "${RUN:-}" ] || { echo "charsiu_run not found" >&2; exit 1; }

if [ ! -e /dev/accel/accel0 ] && [ -z "${CHARSIU_ALLOW_NO_NPU:-}" ]; then
	echo "NO /dev/accel/accel0 -- without the NPU the batched matmul never" >&2
	echo "runs, every chunk agrees, and the sweep reads as a pass. Run it" >&2
	echo "on the board.  CHARSIU_ALLOW_NO_NPU=1 overrides." >&2
	exit 1
fi

DIRS="$HOME/.charsiu/models $HOME/models /opt/charsiu/models \
${CHARSIU_BOARD_DIR:-$HOME/charsiu-board}"
norm() { echo "$1" | tr 'A-Z' 'a-z' | tr -cd 'a-z0-9'; }

MODEL=${1:-}
if [ -n "$MODEL" ] && [ ! -r "$MODEL" ]; then
	w=$(norm "$MODEL"); hit=
	for d in $DIRS; do
		for f in "$d"/*.gguf; do
			[ -r "$f" ] || continue
			case $(norm "$(basename "$f")") in
			*"$w"*) hit=$f; break 2 ;;
			esac
		done
	done
	[ -n "$hit" ] || { echo "no model matching '$MODEL'" >&2; exit 1; }
	MODEL=$hit
fi
[ -n "${MODEL:-}" ] && [ -r "$MODEL" ] || {
	echo "usage: board_chunk_sweep.sh <model.gguf or substring>" >&2
	exit 1
}

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
PROMPT=$(seq 1 32 | tr '\n' ' ')
NGEN=${CHARSIU_CHUNK_NGEN:-8}
CHUNKS=${CHARSIU_CHUNKS:-"32 31 30 29 24 16 4 2"}
W4="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"

echo "model    $MODEL"
echo "binary   $RUN"
echo "prompt   \"1 2 ... 32\", gen $NGEN, int4 on the NPU"
echo "chunks   $CHUNKS"
echo

# --- the control, once -----------------------------------------------------
# shellcheck disable=SC2086
env $W4 CHARSIU_NO_BATCH_PREFILL=1 "$RUN" "$MODEL" -p "$PROMPT" -n "$NGEN" \
	--ignore-eos >"$T/c.out" 2>"$T/c.err"
# ⚠ READ THE TOKEN COUNT BEFORE STRIPPING THE LINE IT IS ON. charsiu_run puts
# "prompt N tok in ..." in the bracketed summary on STDOUT, and the next line
# deletes every bracketed line because the text is what gets compared. The
# first version parsed after the strip and printed "prompt is ? tokens" -- the
# one number this whole sweep is indexed by.
NTOK=$(sed -n 's/.*prompt \([0-9]*\) tok in.*/\1/p' "$T/c.out" | head -1)
sed -i 's/^\[.*//' "$T/c.out"
echo "prompt is ${NTOK:-?} tokens"
echo "control (token loop): ...$(tr -d '\n' < "$T/c.out" | tail -c 52)"
echo

printf '%-7s %-14s %-9s %s\n' chunk widths text path
printf '%-7s %-14s %-9s %s\n' ----- ------ ---- ----
ngood=0; nbad=0; taildiv=""; tailrem=""
for c in $CHUNKS; do
	# shellcheck disable=SC2086
	env $W4 CHARSIU_BATCH_FORCE=1 CHARSIU_PREFILL_CHUNK="$c" "$RUN" \
		"$MODEL" -p "$PROMPT" -n "$NGEN" --ignore-eos \
		>"$T/b.out" 2>"$T/b.err"
	sed -i 's/^\[.*//' "$T/b.out"
	# ⚠ WHAT WIDTHS THIS CHUNK ACTUALLY RAN, spelled out. "chunk 32" is
	# not a width -- on 87 tokens it is 32, 32 and 23, and the 23 is the
	# whole question.
	if [ -n "${NTOK:-}" ]; then
		rem=$((NTOK % c))
		[ "$rem" -lt 2 ] && rem=0
		if [ "$rem" -eq 0 ]; then w="$c only"; else w="$c + tail $rem"; fi
	else
		w="?"
	fi
	# ⚠ WIDE ENOUGH FOR "chunks of 32". cut -c8-40 rendered it "chunks of
	# 3" and "chunks of 2", so the column that says which chunk ran
	# disagreed with the column that asked for it.
	p=$(grep -oE "prompt batched.*" "$T/b.err" | head -1 | cut -c1-48)
	if cmp -s "$T/c.out" "$T/b.out"; then
		v=same; ngood=$((ngood + 1))
		[ "${rem:-1}" -eq 0 ] && taildiv="$taildiv $c"
	else
		v="DIFFERS"; nbad=$((nbad + 1))
		[ "${rem:-0}" -ne 0 ] && tailrem="$tailrem $c/$rem"
	fi
	printf '%-7s %-14s %-9s %s\n' "$c" "$w" "$v" "${p:-?}"
	[ "$v" = same ] || printf '        got: ...%s\n' \
		"$(tr -d '\n' < "$T/b.out" | tail -c 52)"
done

echo
echo "======================================================================"
if [ "$nbad" -eq 0 ]; then
	echo "EVERY CHUNK GAVE THE CONTROL'S TEXT. This model's batched prompt is"
	echo "right at every width tried, so whatever was seen wrong is not the"
	echo "chunk width -- go back to the round that saw it and copy its exact"
	echo "command, flags and all."
elif [ "$ngood" -eq 0 ]; then
	echo "EVERY CHUNK WAS WRONG, including ones that divide the prompt and"
	echo "run a single width with no tail. So it is not a width at all: the"
	echo "batched loop does something the token loop does not, at every"
	echo "width. Next: CHARSIU_DBG_LAYERS=1 on both paths -- it is the tool"
	echo "that found gemma4's three, and it names the layer."
else
	echo "SOME CHUNKS ARE RIGHT AND SOME ARE WRONG, which is the answer this"
	echo "sweep exists for. Read the widths column:"
	[ -n "$taildiv" ] && echo "  right with no tail:$taildiv"
	[ -n "$tailrem" ] && echo "  wrong with a tail (chunk/tail):$tailrem"
	echo
	echo "  If every wrong row has a tail and every right row has none, the"
	echo "  fault is the LAST PARTIAL CHUNK -- a width the probe never asks"
	echo "  about. Then run it directly:"
	echo "    CHARSIU_PROBE_WIDTHS=\"<the tails above>\" \\"
	echo "      CHARSIU_AXIS_ARMS=w sh board_w4_axis.sh <this model>"
	echo "  and the probe will name the tensor and the row at that width."
fi
echo "======================================================================"
