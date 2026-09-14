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

# SOURCED HERE AND NOT FURTHER DOWN: board_clk.sh is what makes
# CHARSIU_RUN and CHARSIU_RUN_BIN two names for one knob, and this
# script picks its binary below. Sourcing it after that point set the
# alias too late to be read -- which is how round 414 measured the
# INSTALLED binary for twenty minutes while believing otherwise.
. "$(dirname "$0")/board_clk.sh"
# THE ENVIRONMENT IS THE SECOND WAY IN. The board's regress.sh calls
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
# SAME REASON AS arch_sanity.sh: the models are on the board and the board has
# no compiler, so both binaries may be handed in instead of built.
NEON="${CHARSIU_RUN:-$ROOT/build/charsiu_run}"
SCAL="${CHARSIU_RUN_SCALAR:-$ROOT/build/charsiu_run_scalar}"

[ -n "${CHARSIU_RUN:-}${CHARSIU_RUN_SCALAR:-}" ] || \
	make -C "$ROOT" build/charsiu_run build/charsiu_run_scalar >/dev/null

# THE MAKEFILE STATES THE INVARIANT AND THIS SCRIPT DID NOT SET IT.
#
# The rule that builds charsiu_run_scalar says, since round 372:
#
#     charsiu_run_scalar  ==  charsiu_run with CHARSIU_EXACT_ATTN and
#                             CHARSIU_EXACT_SILU set
#
# because vector paths that reorder arithmetic ON PURPOSE have no scalar twin.
# This script compared the DEFAULT neon arm against the scalar build, so those
# were guaranteed to differ and were reported as "mismatches" -- a comment
# stating an invariant, and the check for it not setting what the invariant
# requires.
#
# AND THE MAKEFILE'S LIST IS ITSELF SHORT BY ONE: it names two and there are
# THREE. Round 414 measured which, on the two models that were failing:
#
#   ATTN + SILU                      Llama-Q8_0 DIFFERS   Qwen3-Q4_0 DIFFERS
#   ATTN + SILU + GELU               DIFFERS              DIFFERS
#   ATTN + SILU + SOFTMAX            IDENTICAL            IDENTICAL
#
# So the third is the SOFTMAX, and GELU is not involved at all. The invariant
# was written when there were two reordering paths and did not grow when the
# third landed.
#
# CHARSIU_EXACT_GELU IS DELIBERATELY NOT IN THIS LIST. The gelu vector path
# is meant to be bit identical, section 4 of the regression checks exactly
# that, and setting it here would hide a real gelu regression behind the arm
# that papers over it.
EX="CHARSIU_EXACT_ATTN=1 CHARSIU_EXACT_SILU=1 CHARSIU_EXACT_SOFTMAX=1"

# AND THE CONTROL HAS TO BE THE SAME COMMIT. The board's
# /opt/charsiu/charsiu_run_scalar does not answer --version at all, which means
# it predates the build stamp -- so section 3 of the regression was comparing
# two different COMMITS and charging the difference to the vector kernels.
# Refuse rather than measure that.
va=$("$NEON" --version 2>/dev/null | head -1)
vb=$("$SCAL" --version 2>/dev/null | head -1)
case "$vb" in
*usage*|"")
	echo "" >&2
	echo "THE SCALAR CONTROL DOES NOT ANSWER --version." >&2
	echo "     $SCAL" >&2
	echo "   It predates the build stamp, so its commit is unknown and" >&2
	echo "   this would compare two commits, not two kernels." >&2
	echo "   Rebuild it: make build/charsiu_run_scalar" >&2
	echo "" >&2
	exit 2
	;;
esac
if [ "$va" != "$vb" ]; then
	echo "" >&2
	echo "THE TWO ARMS ARE DIFFERENT COMMITS." >&2
	echo "     neon   $va" >&2
	echo "     scalar $vb" >&2
	echo "   Every difference between the commits would be reported as a" >&2
	echo "   vector-kernel mismatch. Rebuild the control from this tree." >&2
	echo "" >&2
	exit 2
fi
echo "   both arms $va  (exact-arm knobs: $EX)"

bad=0
for m in "$DIR"/*.gguf; do
	[ -e "$m" ] || continue
	a=$(env $EX "$NEON" "$m" -p "$P" -n "$N" --ignore-eos -q -c 512 | head -1)
	b=$(env $EX "$SCAL" "$m" -p "$P" -n "$N" --ignore-eos -q -c 512 | head -1)
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
