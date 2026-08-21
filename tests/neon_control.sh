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
DIR="${1:?usage: neon_control.sh MODEL_DIR [PROMPT]}"
P="${2:-The capital of France is}"
N=32

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NEON="$ROOT/build/charsiu_run"
SCAL="$ROOT/build/charsiu_run_scalar"

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
