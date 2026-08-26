#!/usr/bin/env python3
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0-or-later
"""
charsiu's register stream against MESA's, for the same matmul, at any M.

Round 380 asked the vendor's .rkllm which registers carry M and got a wrong
answer, because the vendor never batches a weight matmul: every one of its
M > 1 dispatches is fp16 against the KV cache. Mesa is the right oracle. Its
generic RK3576 encoder, fill_regcmd_rk3576_normal() in rkt_regcmd.c, is what
this board ran M = 1, 2, 3, 4 and 8 through EXACTLY at 512 to 1024 on
2026-08-14, and it is source, not a capture.

So model that encoder here for a 1x1 convolution and diff it against what
emit_job prints. Everything is symbolic in (M, K, N), so the same run answers
"where do they differ at M=2" and "do they agree at M=1".

⚠ M = 1 IS THE CONTROL AND IT RUNS FIRST. charsiu's M = 1 stream computes
correctly on the board, so if this model disagrees with it at M = 1 then the
MODEL is wrong -- a mis-transcribed formula -- and nothing it says about M = 2
can be trusted. That check is what makes the tool worth anything.

    mesa_mdiff.py [K] [N]              default 256 64
    mesa_mdiff.py [K] [N] --axis w     put M on the width axis instead
"""
import subprocess
import sys
import os

ATOM = 16          # FEATURE_ATOMIC_SIZE
CBUF_ENTRY = 128   # CBUF_ENTRY_SIZE


def align(v, a):
    return (v + a - 1) // a * a


def dru(a, b):
    return (a + b - 1) // b


def entries_per_slice(inw, ic):
    """rkt_task.c calc_entries_per_slice(), for one row."""
    ape = CBUF_ENTRY // ATOM
    total = dru(ic, ATOM)
    last = total % ape
    whole = (total // ape) * inw
    frac = inw if last == 3 else dru(last * inw, ape)
    return whole + frac


def mesa(m, k, n, axis):
    """fill_regcmd_rk3576_normal() for a 1x1 conv, geometry words only."""
    kw, s, dw = 1, 1, False
    ic, oc = k, n
    ic_pad = align(ic, ATOM)
    oc_pad = align(oc, 2)
    if axis == "w":
        inw, ow = m, m
        full_inh = win_irows = stg_irows = win_orows = full_oh = 1
    else:
        inw, ow = 1, 1
        full_inh = win_irows = stg_irows = win_orows = full_oh = m
    surf = entries_per_slice(inw, ic)
    wbpk = ic_pad * kw * kw * 2
    k_word = 0                      # k < 3
    dpu_lines = win_orows - 1
    return {
        0x1014: (s << 3) | s,
        0x101c: ic_pad * oc_pad * kw * kw,
        0x1020: ic_pad * kw * kw,
        0x1024: (k_word << 16) | (oc_pad - 1),
        0x1028: ((surf * stg_irows) << 16) | (ic_pad - 1),
        0x102c: ((inw - 1) << 16) | (win_irows - 1),
        0x1030: (wbpk << 16) | (ow - 1),
        0x1034: ow * win_orows - 1,
        0x103c: (surf << 16) | 0,
        0x1044: (inw << 16) | surf,
        0x1078: ((inw - 1) << 16) | (stg_irows - 1),
        0x107c: ic_pad - 1,
        0x1090: inw * 4,
        0x1094: inw * full_inh,
        0x1098: (inw * stg_irows + 3) & ~3,
        0x118c: ((inw - 1) << 16) | (full_inh - 1),
        0x301c: (dpu_lines << 16) | (ow - 1),
        0x3020: oc - 1,
        0x401c: ow * full_oh,
        0x4020: ow - 1,
        0x4024: dpu_lines,
        0x402c: oc - 1,
        0x4030: ((oc - 1) << 16) | 0x0710,
        0x4034: (dpu_lines << 16) | (ow - 1),
        0x40b8: ow * (2 * full_oh - win_orows),
    }


def charsiu(m, k, n, exe):
    """What emit_job prints, as {reg: value}. int8 both sides, like the test."""
    out = subprocess.run([exe, str(m), str(k), str(n)],
                         capture_output=True, text=True).stdout
    regs = {}
    for line in out.splitlines():
        f = line.split()
        # CS <i> t=0201 r=1094 v=00000020
        if len(f) >= 5 and f[3].startswith("r=") and f[4].startswith("v="):
            regs.setdefault(int(f[3][2:], 16), int(f[4][2:], 16))
    return regs


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 256
    n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 64
    axis = "w" if "--axis" in sys.argv and "w" in sys.argv[sys.argv.index("--axis") + 1:] \
        else "h"
    here = os.path.dirname(os.path.abspath(__file__))
    exe = os.path.join(here, "..", "build", "emit_job")
    if not os.path.exists(exe):
        print("build/emit_job is missing: make build/emit_job")
        return 1

    print("matmul %dx%d -> %dxN, N=%d, M on the %s axis\n"
          % (0, k, 0, n, "width" if axis == "w" else "height"))
    bad_control = False
    for m in (1, 2, 4, 32):
        want = mesa(m, k, n, axis)
        got = charsiu(m, k, n, exe)
        diff = [(r, want[r], got.get(r)) for r in sorted(want)
                if got.get(r) != want[r]]
        tag = "  (THE CONTROL)" if m == 1 else ""
        if not diff:
            print("  M=%-3d agrees on all %d geometry words%s" % (m, len(want), tag))
            continue
        print("  M=%-3d %d of %d differ%s" % (m, len(diff), len(want), tag))
        print("        %-8s %-12s %-12s" % ("reg", "mesa", "charsiu"))
        for r, w, g in diff:
            print("        %04x     %08x     %s" %
                  (r, w, "%08x" % g if g is not None else "(not emitted)"))
        if m == 1:
            bad_control = True

    if bad_control:
        print("\n  ⚠ M=1 DISAGREES, and charsiu's M=1 is correct on the board.\n"
              "  So this model is mis-transcribed and its M>1 rows mean nothing.\n"
              "  Fix the model against rkt_regcmd.c before reading further.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
