#!/usr/bin/env python3
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0-or-later
"""
Diff the stream charsiu emits against the vendor's, for the same shape.

The vendor's own dispatches are the oracle: if what we build matches the geometry
of what the closed runtime submits, the board round is a confirmation rather than
a discovery. This runs entirely on a desktop.

Usage: cmp_vendor.py <model.rkllm> <M> <K> <N> [wdtype] [adtype]
"""
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rkllm_regcmd import streams, decode, geometry     # noqa: E402


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        return 1
    path, m, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    wd = sys.argv[5] if len(sys.argv) > 5 else "int4"
    ad = sys.argv[6] if len(sys.argv) > 6 else "fp16"

    ours = {}
    out = subprocess.run([__file__.rsplit("/",2)[0] + "/build/emit_dump", str(m), str(k), str(n), wd, ad],
                         capture_output=True, text=True).stdout
    for line in out.split("\n"):
        f = line.split()
        if len(f) == 3:
            ours[(int(f[0], 16), int(f[1], 16))] = int(f[2], 16)

    match = None
    for off, ws in streams(path):
        regs = decode(ws)
        g = geometry(regs)
        if g and g["ic"] == k and g["oc"] == n and g["rows"] == m:
            match = (off, regs)
            break
    if not match:
        print("no vendor stream at M=%d K=%d N=%d" % (m, k, n))
        return 1
    off, vend = match

    print("vendor stream at 0x%x, %d registers; charsiu emits %d\n"
          % (off, len(vend), len(ours)))
    same = miss = diff = extra = 0
    print("  %-6s %-10s %-10s" % ("reg", "charsiu", "vendor"))
    for key in sorted(set(ours) | set(vend), key=lambda x: (x[0], x[1])):
        a, b = ours.get(key), vend.get(key)
        tag = {0x201: "CNA", 0x801: "CORE", 0x1001: "DPU", 0x2001: "RDMA"}.get(key[0], "?")
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
              % (tag, key[1], "--" if a is None else "%08x" % a,
                 "--" if b is None else "%08x" % b, mark))
    print("\n  identical %d, different %d, vendor only %d, charsiu only %d"
          % (same, diff, miss, extra))
    return 0


if __name__ == "__main__":
    sys.exit(main())
