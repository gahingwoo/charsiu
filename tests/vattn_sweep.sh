#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
#
# The attention at four working set sizes, both schedules, interleaved.
#
# ⚠ THE POINT IS THE SHAPE OF THE CURVE, NOT THE SECONDS. Attention is O(n^2)
# work, so the per layer time divided by (n/1024)^2 is flat while the machine is
# compute bound and RISES once the working set stops fitting in cache. Where it
# rises is where this host starts asking the same question the board asks at
# n = 1024, because the board has about a megabyte of L2 for a whole cluster and
# a development host has an order of magnitude more.
#
# ⚠ ON THE BOARD, RUN THIS AT n = 1024 AND BELIEVE THAT ROW. The larger n are a
# host's only way to reach the board's regime; the board is already in it.
#
#     tests/vattn_sweep.sh [build/vattn_bench] [reps]

B=${1:-build/vattn_bench}
R=${2:-3}
for spec in "1024 12" "2048 4" "4096 2" "8192 1"; do
	set -- $spec
	echo "== n=$1 layers=$2"
	"$B" -c -n "$1" -l "$2" -r "$R" | sed -n '2,$p'
done
