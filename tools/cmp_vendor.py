#!/usr/bin/env python3
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0-or-later
"""
Diff the stream charsiu emits against the vendor's, for the same shape.

The vendor's own dispatches are the oracle: if what we build matches the geometry
of what the closed runtime submits, the board round is a confirmation rather than
a discovery. This runs entirely on a desktop.

⚠ IT DIFFS THE EMITTER THAT ACTUALLY RUNS. This called build/emit_dump, which
is regcmd.c -- a geometry only emitter nothing else uses. The stream that
reaches the hardware comes from charsiu_emit_job() in job.c, and the two do not
agree about the M axis, so every comparison this tool ran was of a file that no
board has ever executed. build/emit_job is the product one.

⚠ AND IT MERGES THE VENDOR'S TWO STREAMS. The vendor splits one matmul into a
CNA stream and the DPU stream right behind it; charsiu emits one. Comparing
against the CNA stream alone reports every DPU register as "charsiu only",
which is how the DPU's output width -- the register that decides whether M rows
are written at all -- stayed invisible.

Address registers are skipped: a static .rkllm carries them unpatched at 0.

Usage: cmp_vendor.py <model.rkllm> <M> <K> <N> [wdtype] [adtype]
       CHARSIU_M_AXIS=w cmp_vendor.py ...   compare the width axis form
"""
import os
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rkllm_regcmd import streams, decode, geometry     # noqa: E402

ROOT = __file__.rsplit("/", 2)[0]
TAG = {0x201: "CNA", 0x801: "CORE", 0x1001: "DPU", 0x2001: "RDMA"}
# input surface, weights, output, coefficients: unpatched in a static file
ADDR = {(0x201, 0x1088), (0x201, 0x1110), (0x1001, 0x4018),
        (0x2001, 0x5020), (0x2001, 0x5024)}


def ours(m, k, n, wd, ad):
    """The product emitter's stream, as {(target, register): value}."""
    out = subprocess.run([ROOT + "/build/emit_job", str(m), str(k), str(n),
                          wd, ad], capture_output=True, text=True).stdout
    regs = {}
    for line in out.split("\n"):
        # CS 00 t=0201 r=1004 v=0000000e
        f = line.split()
        if len(f) == 5 and f[0] == "CS":
            t = int(f[2][2:], 16)
            r = int(f[3][2:], 16)
            regs[(t, r)] = int(f[4][2:], 16)
    return regs


def vendor(path, m, k, n, bits):
    """The vendor's CNA stream for this shape, merged with the DPU stream
    that follows it."""
    runs = streams(path)
    for i, (off, ws) in enumerate(runs):
        regs = decode(ws)
        g = geometry(regs)
        if not g or g["ic"] != k or g["oc"] != n or g["m"] != m:
            continue
        if bits is not None and g["weight_bits"] != bits:
            continue
        merged = dict(regs)
        if i + 1 < len(runs):
            nxt = decode(runs[i + 1][1])
            if geometry(nxt) is None:          # a DPU only stream
                merged.update(nxt)
        return off, merged, g
    return None, None, None


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        return 1
    path, m, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    wd = sys.argv[5] if len(sys.argv) > 5 else "int4"
    ad = sys.argv[6] if len(sys.argv) > 6 else "fp16"
    bits = {"int4": 4.0, "int8": 8.0, "fp16": 16.0}.get(wd)

    off, vend, g = vendor(path, m, k, n, bits)
    if vend is None:
        print("no vendor %s stream at M=%d K=%d N=%d" % (wd, m, k, n))
        return 1
    mine = ours(m, k, n, wd, ad)
    if not mine:
        print("build/emit_job produced nothing -- run make first")
        return 1

    print("vendor stream at 0x%x (M=%d on the width, %d rows high), "
          "%d registers; charsiu emits %d"
          % (off, g["m"], g["rows"], len(vend), len(mine)))
    print("  axis: CHARSIU_M_AXIS=%s\n" % (os.environ.get("CHARSIU_M_AXIS") or "h (height, the default)"))
    same = miss = diff = extra = 0
    print("  %-6s %-10s %-10s" % ("reg", "charsiu", "vendor"))
    for key in sorted(set(mine) | set(vend)):
        if key in ADDR:
            continue
        a, b = mine.get(key), vend.get(key)
        if a == b:
            same += 1
            continue
        if a is None:
            miss += 1
            mark = "vendor only"
        elif b is None:
            extra += 1
            mark = "charsiu only"
        else:
            diff += 1
            mark = "DIFFERENT"
        print("  %-4s %04x  %-10s %-10s  %s"
              % (TAG.get(key[0], "?"), key[1], "--" if a is None else "%08x" % a,
                 "--" if b is None else "%08x" % b, mark))
    print("\n  identical %d, different %d, vendor only %d, charsiu only %d"
          % (same, diff, miss, extra))
    return 0


if __name__ == "__main__":
    sys.exit(main())
