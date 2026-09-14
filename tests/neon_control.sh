#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
#
# The NEON kernels against the portable ones, on every quantisation a model
# file actually uses. Needs no reference implementation and no network: the
# two builds are the same source and have to agree.
#
# It is a real control -- it can fail, and the point of running it is that a
# vector kernel which is wrong still produces fluent text.
#
#   tests/neon_control.sh models/  [PROMPT]

set -e

# ⚠ SOURCED HERE AND NOT FURTHER DOWN: board_clk.sh is what makes
# CHARSIU_RUN and CHARSIU_RUN_BIN two names for one knob, and this
# script picks its binary below. Sourcing it after that point set the
# alias too late to be read -- which is how round 414 measured the
# INSTALLED binary for twenty minutes while believing otherwise.
. "$(dirname "$0")/board_clk.sh"
# ⚠ THE ENVIRONMENT IS THE SECOND WAY IN. The board's regress.sh calls
# this with no argument, having exported CHARSIU_BOARD_DIR, and a hard ${1:?}
# turned that into a usage line inside a pipeline -- so the section printed
# its heading and nothing else, and the regression carried on green. Same
# hole arch_sanity.sh had; censused by the property rather than the name.
DIR="${1:-${CHARSIU_BOARD_DIR:-}}"
if [ -z "$DIR" ]; then
	echo "neon_control.sh: no model directory." >&2
	echo "  give one as \$1, or set CHARSIU_BOARD_DIR." >&2
	exit 2
fi
if [ ! -d "$DIR" ]; then
	echo "neon_control.sh: $DIR is not a directory" >&2
	exit 2
fi
P="${2:-The capital of France is}"
N=32

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# ⚠ SAME REASON AS arch_sanity.sh: the models are on the board and the board has
# no compiler, so both binaries may be handed in instead of built.
NEON="${CHARSIU_RUN:-$ROOT/build/charsiu_run}"
SCAL="${CHARSIU_RUN_SCALAR:-$ROOT/build/charsiu_run_scalar}"

[ -n "${CHARSIU_RUN:-}${CHARSIU_RUN_SCALAR:-}" ] || \
	make -C "$ROOT" build/charsiu_run build/charsiu_run_scalar >/dev/null

bad=0
for m in "$DIR"/*.gguf; do
	[ -e "$m" ] || continue
	a=$("$NEON" "$m" -p "$P" -n "$N" --ignore-eos -q -c 512 | head -1)
	b=$("$SCAL" "$m" -p "$P" -n "$N" --ignore-eos -q -c 512 | head -1)
	if [ "$a" = "$b" ]; then
		echo "ok    $(basename "$m")"
	else
		echo "FAIL  $(basename "$m")"
		echo "  neon   $a"
		echo "  scalar $b"
		bad=$((bad + 1))
	fi
done
echo "$bad mismatches"
exit $bad
