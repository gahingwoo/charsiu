#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
#
# The fused kernel against the exact one at every awkward shape.
#
# ⚠⚠ THE RAGGED ENDS ARE WHERE A TILED KERNEL IS WRONG. vision_cross runs a toy
# tower whose sequence is 16, which is smaller than the query block and an exact
# multiple of the key tile -- so it exercises neither remainder. The tower's own
# sequence is 1024, or 1025 when the model has a class token, and 1025 is a
# prime-ish length that leaves a partial query block AND a partial key tile.
#
# A shape here that prints anything other than a difference in the 1e-7s is the
# fused kernel losing part of the answer at an edge, which is exactly the
# failure a stopwatch reports as a speedup.
#
#     tests/vattn_edges.sh [build/vattn_bench]

B=${1:-build/vattn_bench}
rc=0
# n, heads: 1025 is the tower with a class token; 1000 leaves a partial query
# block; 17 and 3 are shorter than one tile; 64 and 65 straddle the block size
for spec in "1025 12" "1024 12" "1000 12" "999 12" "65 12" "64 12" "17 12" \
            "3 12" "1 12" "1025 1" "257 4"; do
	set -- $spec
	printf 'n=%-6s heads=%-3s ' "$1" "$2"
	out=$("$B" -F -n "$1" -H "$2" -l 1 -r 1 | sed -n 's/.*worst diff //p')
	echo "worst |fused - exact| = $out"
	case $out in
	*e-0[4-9]|*e-1[0-9]|0.00e+00) ;;
	*) echo "  FAILED: that is not rounding"; rc=1 ;;
	esac
done
[ $rc = 0 ] && echo PASS || echo FAILED
exit $rc
