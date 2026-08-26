#!/usr/bin/env python3
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0-or-later
"""
Which registers carry M? Ask the vendor's own file.

charsiu dispatches every projection at M = 1, one row, because that is what
decode is. Prefill has a whole prompt available at once and wants M = 32, where
the same weight bytes serve thirty two rows instead of one -- the board
measured 6.9 us a row at M = 32 against 200 us a row at M = 1.

charsiu's own M > 1 has never been right, and sweeping activation layouts to
find out why is the wrong instrument: at M = 1 every stride in the input
surface block is degenerate, so a register that is wrong for M > 1 looks fine
and nine layout candidates all fail for the same invisible reason.

A .rkllm carries thousands of dispatches at M = 2 through 128. Two of them at
the SAME shape and dtype, differing only in M, name the M-carrying registers
outright -- no board, no sweep.

    rkllm_mdiff.py <model.rkllm>              the block, across every M at one shape
    rkllm_mdiff.py <model.rkllm> --all        every register that varies with M

⚠ NEEDS numpy < 2 on this VM: the 2.x aarch64 wheel dies with SIGILL.
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkllm_regcmd import streams, decode, geometry, TARGETS, CNA   # noqa: E402

# The input surface block. Everything here is degenerate at M = 1: a stride
# over one row, a count of one, a last index of zero.
BLOCK = [0x1028, 0x102c, 0x1034, 0x1078, 0x1090, 0x1094, 0x1098, 0x118c]


def collect(path):
    """One representative stream per (ic, oc, M)."""
    rep = {}
    for _, ws in streams(path):
        regs = decode(ws)
        geo = geometry(regs)
        if geo:
            rep.setdefault((geo["ic"], geo["oc"], geo["rows"]), (regs, geo))
    return rep


def widest_shape(rep):
    """The (ic, oc) the vendor dispatches at the most different M."""
    ms = {}
    for ic, oc, m in rep:
        ms.setdefault((ic, oc), set()).add(m)
    return max(ms.items(), key=lambda kv: len(kv[1]))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    every = "--all" in sys.argv[2:]

    rep = collect(path)
    if not rep:
        print("no register command streams in %s" % path)
        return 1

    (ic, oc), mset = widest_shape(rep)
    ms = sorted(mset)

    bits = Counter()
    for (i, o, m), (_, geo) in rep.items():
        bits[geo["weight_bits"]] += 0
    # M histogram per weight width: the strategic fact, not a register one
    hist = {}
    for _, ws in streams(path):
        geo = geometry(decode(ws))
        if geo:
            hist.setdefault(geo["weight_bits"], Counter())[geo["rows"]] += 1
    print("what the vendor batches, by weight width")
    for wb in sorted(hist):
        h = hist[wb]
        span = "M=1 only" if set(h) == {1} else "M = %d..%d" % (min(h), max(h))
        print("   %-5s bit weights   %-14s %d dispatches" % (wb, span, sum(h.values())))
    print()

    cols = [rep[(ic, oc, m)][0] for m in ms]
    if every:
        keys = sorted({k for r in cols for k in r})
    else:
        keys = [(0x0201, r) for r in BLOCK]

    print("ic=%d oc=%d, one shape, M = %s" % (ic, oc, ms))
    print("  %-6s %-6s %s %s" % ("tgt", "reg",
                                 "".join("M=%-9d" % m for m in ms), "name"))
    for t, r in keys:
        vals = [c.get((t, r)) for c in cols]
        if every and len(set(vals)) == 1:
            continue
        held = "  <- constant in M" if len(set(vals)) == 1 else ""
        print("  %-6s %04x   %s %s%s" % (
            TARGETS.get(t, "?"), r,
            "".join(("%08x   " % v) if v is not None else "--------   "
                    for v in vals),
            CNA.get(r, "") if t == 0x0201 else "", held))
    return 0


if __name__ == "__main__":
    sys.exit(main())
