#!/usr/bin/env python3
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0-or-later
"""
Read the vendor LLM runtime's NPU dispatches straight out of a .rkllm file.

A .rkllm carries the register command streams the vendor runtime submits, the
same way a .rknn does, so what the closed stack asks the NPU to do for a large
language model can be read on a desktop with no board and no vendor runtime in
the loop.

Each stream entry is a little-endian u64:

    [63:48] target   [47:16] value   [15:0] register

with targets 0x0201 CNA, 0x0801 CORE, 0x1001 DPU, 0x2001 DPU_RDMA, 0x0041 sync
and 0x0081 broadcast. A stream is a maximal run of such words, and every op the
model runs is one of them.

Address registers in a static file are unpatched placeholders and read 0, which
is why this reports geometry and never addresses.

Usage:
    rkllm_regcmd.py <model.rkllm>                 summary: kinds and shapes
    rkllm_regcmd.py <model.rkllm> --shapes N      the N most common conv shapes
    rkllm_regcmd.py <model.rkllm> --dump OFFSET   one stream in full, by byte offset
    rkllm_regcmd.py <model.rkllm> --find ic=4096  dump the first stream matching
"""
import sys
from collections import Counter

import numpy as np

TARGETS = {0x0201: "CNA", 0x0801: "CORE", 0x1001: "DPU",
           0x2001: "RDMA", 0x0041: "SYNC", 0x0081: "BCAST"}

# The CNA geometry registers, named from the RK3576 conv work in
# github.com/gahingwoo/linux-rk3576-npu.
CNA = {
    0x1004: "mode",
    0x100c: "dw/flags",
    0x1014: "stride",
    0x1018: "cbuf pair",
    0x101c: "weight bytes",
    0x1020: "weight bytes per kernel",
    0x1024: "oc-1",
    0x1028: "surf*rows | ic-1",
    0x102c: "rows-1 | window rows-1",
    0x1030: "weight bytes per kernel x2 | ow-1",
    0x1034: "ow*oh-1",
    0x103c: "surf",
    0x1040: "cbuf pair 2",
    0x1044: "surf pair",
    0x107c: "ic-1",
    0x1080: "padding",
    0x1084: "pad value",
}


def streams(path, min_len=20):
    """Every maximal run of register command words, as (byte offset, words)."""
    # ⚠ MEMORY MAP IT. This read the whole file with np.fromfile and then
    # called .tobytes(), which is a SECOND full copy, so a 1.3 GB .rkllm asked
    # for 2.6 GB and the OOM killer took it twice. A memmap is zero copies and
    # the kernel pages what the scan touches.
    import os

    n = os.path.getsize(path) // 8
    words = np.memmap(path, dtype="<u8", mode="r", shape=(n,))
    ok = np.isin((words >> 48).astype(np.uint32), list(TARGETS))
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return []
    out = []
    for run in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
        if len(run) >= min_len:
            out.append((int(run[0]) * 8, words[run[0]:run[-1] + 1]))
    return out


def decode(ws):
    """One stream as {(target, register): value}, first write wins."""
    regs = {}
    for e in ws:
        e = int(e)
        key = ((e >> 48) & 0xffff, e & 0xffff)
        if key not in regs:
            regs[key] = (e >> 16) & 0xffffffff
    return regs


def geometry(regs):
    """The conv geometry a CNA stream describes, or None if it has no CNA."""
    if not any(t == 0x0201 for t, _ in regs):
        return None
    g = lambda r: regs.get((0x0201, r), 0)
    ic = (g(0x1028) & 0xffff) + 1
    oc = (g(0x1024) & 0xffff) + 1
    wt = g(0x101c)
    # bytes per weight element: 1 for int8, 1/2 for int4, 2 for the 16 bit types
    bits = (8 * wt) / (ic * oc) if ic and oc else 0
    return {
        "ic": ic,
        "oc": oc,
        "rows": (g(0x102c) & 0xffff) + 1,
        "window": ((g(0x102c) >> 16) & 0xffff) + 1,
        "surf": (g(0x103c) >> 16) & 0xffff,
        "pixels": g(0x1034) + 1,
        "weight_bytes": wt,
        "weight_bits": round(bits, 2),
        "split": (g(0x1018) & 0xffff) == 0x0505,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    args = sys.argv[2:]

    runs = streams(path)
    if not runs:
        print("no register command streams found in %s" % path)
        return 1

    if args and args[0] == "--dump":
        want = int(args[1], 0)
        for off, ws in runs:
            if off == want:
                for e in ws:
                    e = int(e)
                    t, reg, v = (e >> 48) & 0xffff, e & 0xffff, (e >> 16) & 0xffffffff
                    note = CNA.get(reg, "") if t == 0x0201 else ""
                    print("  %-5s %04x %08x  %s" % (TARGETS.get(t, "?"), reg, v, note))
                return 0
        print("no stream at 0x%x" % want)
        return 1

    if args and args[0] == "--find":
        key, _, val = args[1].partition("=")
        val = int(val, 0)
        for off, ws in runs:
            geo = geometry(decode(ws))
            if geo and geo.get(key) == val:
                print("stream at 0x%x, %d entries, %s" % (off, len(ws), geo))
                return 0
        print("nothing matched %s" % args[1])
        return 1

    kinds = Counter()
    shapes = Counter()
    bits = Counter()
    rows = Counter()
    for off, ws in runs:
        geo = geometry(decode(ws))
        kinds["conv" if geo else "dpu only"] += 1
        if geo:
            shapes[(geo["ic"], geo["oc"], geo["surf"], geo["rows"])] += 1
            bits[geo["weight_bits"]] += 1
            rows[geo["rows"]] += 1

    print("%s" % path)
    print("  streams            %d, of which %d are convolutions and %d DPU only"
          % (len(runs), kinds["conv"], kinds["dpu only"]))
    print("  distinct shapes    %d" % len(shapes))
    print("  weight bits        %s" % dict(bits.most_common(5)))
    print("  M (rows per op)    %s" % dict(rows.most_common(5)))
    print()
    n = int(args[1]) if len(args) > 1 and args[0] == "--shapes" else 12
    print("  %-8s %-8s %-8s %-6s %s" % ("ic", "oc", "surf", "M", "count"))
    for (ic, oc, surf, r), c in shapes.most_common(n):
        print("  %-8d %-8d %-8d %-6d %d" % (ic, oc, surf, r, c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
