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

⚠ AND IT MERGES THE VENDOR'S TWO STREAMS. The vendor's matmul used to come back
as a CNA stream and a DPU stream behind it; charsiu emits one. Comparing
against the CNA stream alone reports every DPU register as "charsiu only",
which is how the DPU's output width -- the register that decides whether M rows
are written at all -- stayed invisible. rkllm_regcmd.py knowing target 0x0401
has since joined the two, so the merge only fires on a fragment now.

⚠⚠ AND IT NOW ASKS FOR acc_out, BECAUSE THE BOARD DOES. npudev.c's add_slice
sets job.acc_out = 1 on every staged projection slice, unconditionally, and
emit_job only sets it from CHARSIU_ACC_OUT. Without it the tool compared a
stream no projection has ever submitted, and reported 0x40b8 as a difference
when the board and the vendor both write 3 there. CHARSIU_ACC_OUT=0 asks for
the old behaviour.

Address registers are skipped: a static .rkllm carries them unpatched at 0.

Usage: cmp_vendor.py <model.rkllm> <M> <K> <N> [wdtype] [adtype]
       cmp_vendor.py <model.rkllm> --plan <K> <N>   how each side CUTS a tensor
       CHARSIU_M_AXIS=w cmp_vendor.py ...   compare the width axis form

⚠ THE SHAPE YOU PASS IS ONE DISPATCH, NOT ONE TENSOR. charsiu slices a tensor
into ceil(K/KMAX) * ceil(N/NMAX) dispatches before any of this, so a register
diff at a tensor's full shape compares something neither side submits. --plan
is what compares the CUTS, and the cut is where the two stacks actually differ.
"""
import os
import subprocess
import sys
from collections import Counter

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rkllm_regcmd import streams, decode, geometry     # noqa: E402

ROOT = __file__.rsplit("/", 2)[0]
TAG = {0x201: "CNA", 0x801: "CORE", 0x1001: "DPU", 0x2001: "RDMA"}
# input surface, weights, output, coefficients: unpatched in a static file
ADDR = {(0x201, 0x1088), (0x201, 0x1110), (0x1001, 0x4018),
        (0x2001, 0x5020), (0x2001, 0x5024)}


def child_env():
    """emit_job's environment, and the note to print about it."""
    env = dict(os.environ)
    v = env.get("CHARSIU_ACC_OUT")
    if v is None:
        env["CHARSIU_ACC_OUT"] = "1"
        return env, "acc_out ON (npudev sets it on every staged slice)"
    if v in ("", "0"):
        env.pop("CHARSIU_ACC_OUT")
        return env, "acc_out OFF by request -- the board does not run this"
    return env, "acc_out ON"


def ours(m, k, n, wd, ad):
    """The product emitter's stream, as {(target, register): value}."""
    env, _ = child_env()
    out = subprocess.run([ROOT + "/build/emit_job", str(m), str(k), str(n),
                          wd, ad], capture_output=True, text=True,
                         env=env).stdout
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
        #
        # ⚠ ONLY IF THIS RUN IS MISSING ITS OWN DPU BLOCK. rkllm_regcmd.py now
        # knows target 0x0401, so an op comes back as ONE run with its CNA,
        # its 0x2810 block and its DPU registers together, and merging the run
        # behind it would then pull in a DIFFERENT op's output stage. The merge
        # stays for files read by an older table, where the op is in pieces.
        #
        # 0x4018 is the output address: a run that has it carries a whole
        # output stage, and a run that does not is a fragment needing the
        # next one. Counting DPU registers instead would not work -- even the
        # fragment carries 0x4004, the DPU's own S_POINTER.
        if (0x1001, 0x4018) not in regs and i + 1 < len(runs):
            nxt = decode(runs[i + 1][1])
            if geometry(nxt) is None:          # a DPU only stream
                merged.update(nxt)
        return off, merged, g
    return None, None, None


def census(path, m, bits):
    """Every (ic, oc) the vendor dispatches at this M and weight width, with
    the number of streams the file holds for each.

    ⚠ THE STREAM COUNT IS NOT A DISPATCH COUNT. At one shape and one M the file
    carries a stream per CBUF window as well as per piece of the tensor -- two
    of the four at K=2048 N=1024 differ from the other two ONLY in 0x1018,
    0x1038, 0x103c and 0x1040 -- and only one window runs on a given core. Read
    it as "the file has this shape", not as "the hardware runs it this often".
    """
    c = Counter()
    for _, ws in streams(path):
        g = geometry(decode(ws))
        if g and g["m"] == m and (bits is None or g["weight_bits"] == bits):
            c[(g["ic"], g["oc"])] += 1
    return c


def cuts(total, cap):
    """How npudev.c cuts one axis: ceil(total / cap) pieces, the last one
    holding whatever is left. KFIT is off by default and is not modelled."""
    n = (total + cap - 1) // cap
    return [cap] * (n - 1) + [total - cap * (n - 1)]


def plan(path, k, n, kmax, nmax, wd):
    """What each side cuts a K by N tensor into, and what that costs."""
    bits = {"int4": 4.0, "int8": 8.0, "fp16": 16.0}.get(wd)
    per = {"int4": 0.5, "int8": 1.0, "fp16": 2.0}.get(wd, 1.0)

    ks, ns = cuts(k, kmax), cuts(n, nmax)
    print("a %d by %d tensor, %s weights\n" % (k, n, wd))
    print("charsiu: KMAX=%d NMAX=%d -> %d K pieces x %d N pieces = %d dispatches"
          % (kmax, nmax, len(ks), len(ns), len(ks) * len(ns)))
    for kk in sorted(set(ks)):
        for nn in sorted(set(ns)):
            cnt = ks.count(kk) * ns.count(nn)
            print("   %2d x  K=%-6d N=%-6d %8.2f MB of weights each"
                  % (cnt, kk, nn, kk * nn * per / 1e6))
    #
    # ⚠ SUM(dispatches * N) IS THE NUMBER THAT MOVES, not the weight bytes.
    # Both sides read every weight once whatever the cut, so the weight column
    # is equal by construction and says nothing. What a K cut multiplies is the
    # PER OUTPUT CHANNEL work: every K piece writes the full N wide accumulator
    # and fetches the full N of coefficient records, and the CPU then sums the
    # pieces. An N cut divides that same work instead of repeating it.
    #
    mine_ch = len(ks) * n
    print("   %d accumulator words a token, %d of them repeats of a K cut"
          % (mine_ch, mine_ch - n))

    print("\nvendor: shapes in this file at M=1 that tile %d by %d exactly," % (k, n))
    print("        widest tile first -- and the widest is the one to read.")
    #
    # ⚠ ARITHMETIC ALONE CANNOT PICK THE TILE. A 2048 by 2048 tensor is tiled
    # exactly by their K=2048 N=1024 shape AND by their K=2048 N=256 one, and
    # both are in the file, because 256 is the k_proj half of a 512 wide tensor
    # and has nothing to do with q_proj. What settles it is the file's own
    # order: walking the M=1 int4 streams by offset gives 32 a layer, in runs
    # of 4, 8, 4, 8, 8 at oc 1024, 256, 1024, 4096, 1024 -- q, k and v, o,
    # gate and up, down -- for 16 layers exactly. Each run is 2 pieces of the
    # tensor times the 2 CBUF windows, so the widest tile is the real one.
    #
    cen = census(path, 1, bits)
    hit = [(ic, oc, c) for (ic, oc), c in cen.items() if not (k % ic or n % oc)]
    for ic, oc, streams_held in sorted(hit, key=lambda t: -t[0] * t[1]):
        tiles = (k // ic) * (n // oc)
        print("   %2d x  K=%-6d N=%-6d %8.2f MB each   (%d streams in the file)"
              % (tiles, ic, oc, ic * oc * per / 1e6, streams_held))
        print("         %d accumulator words a token, %d of them repeats"
              % ((k // ic) * n, (k // ic) * n - n))
    if not hit:
        print("   none -- this file dispatches no %s shape that divides it" % wd)
    return 0


def main():
    if len(sys.argv) >= 3 and sys.argv[2] == "--plan":
        if len(sys.argv) < 5:
            print(__doc__)
            return 1
        a = sys.argv
        k, n = int(a[3]), int(a[4])
        # npudev.c's own defaults. ⚠ scripts/charsiu-runner writes its kslice,
        # 1024 by default, into CHARSIU_NPU_KMAX -- and into W4_GROUP with it,
        # because npudev.c gates the int4 path on kgroup == kmax. So the board
        # runs a K cut this default does not show, and raising KMAX alone
        # changes the QUANTISER, not just the schedule.
        kmax = int(a[5]) if len(a) > 5 else int(os.environ.get("CHARSIU_NPU_KMAX", 4096))
        nmax = int(a[6]) if len(a) > 6 else int(os.environ.get("CHARSIU_NPU_NMAX", 8192))
        return plan(a[1], k, n, kmax, nmax,
                    os.environ.get("CHARSIU_PLAN_WD", "int4"))

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
    print("  axis: CHARSIU_M_AXIS=%s" % (os.environ.get("CHARSIU_M_AXIS") or "h (height, the default)"))
    print("  %s\n" % child_env()[1])
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
