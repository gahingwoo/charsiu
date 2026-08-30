#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
#
# Every knob in the tower's attention, timed against its own control, in one
# process each.
#
# ⚠⚠ THIS EXISTS BECAUSE THE BOARD IS THE MACHINE THAT DECIDES AND IT IS NOT
# THIS ONE. Every default in src/vision.c was picked from a development host
# that is compute bound where the board is bandwidth bound, and holds a 12 MB
# working set in cache where the board has about a megabyte of L2 for a whole
# cluster. Two of the six answers below already came out differently at the two
# ends of this host's own range. Run it on the card and believe the card.
#
# ⚠ THE HOST'S ONLY WAY TO REACH THE BOARD'S REGIME IS n. Attention is O(n^2)
# work, so per layer time over (n/1024)^2 is flat while a machine is compute
# bound and rises once its cache stops holding the working set. On the board,
# n = 1024 is already that regime and the larger sizes are just slower.
#
#     tests/vattn_sweep.sh [build/vattn_bench] [reps] [n] [layers]

# ⚠ THE INSTALLED PATH, NOT THE SOURCE TREE'S. `charsiu update dev` puts the
# binary in /opt/charsiu and this defaulted to build/vattn_bench, which exists
# only where it was compiled -- so shipping the script alone would have failed
# a second time, one line further on.
B=${1:-}
[ -n "$B" ] || B=$(command -v vattn_bench 2>/dev/null || true)
[ -n "$B" ] || for d in /opt/charsiu /usr/bin "$PWD/build" ./build .; do
	[ -x "$d/vattn_bench" ] && { B="$d/vattn_bench"; break; }
done
[ -n "${B:-}" ] || { echo "vattn_bench not found; it ships on the dev channel" >&2; exit 1; }
R=${2:-5}
N=${3:-1024}
L=${4:-12}

run() {
	echo "-- $2"
	"$B" "$1" -n "$N" -l "$L" -r "$R" | sed -n '2,$p'
}

echo "== n=$N layers=$L reps=$R"
"$B" -n "$N" -l "$L" -r 1 | sed -n '1p'
run -B "the round as a whole: as it was, against as it is"
run -c "which thread gets which head"
run -Q "the query block"
run -K "the fused kernel's key tile"
run -F "the three pass kernel against the fused one"
run -P "one pair at a time, against the blocked kernels"
run -E "glibc's expf against the polynomial one"
