#!/usr/bin/env python3
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0-or-later
"""
Read the vendor's int4 SCALES out of a .rkllm, and score its quantiser.

THE IDEA. A per-row scale is proportional to that row's largest magnitude, and
we hold the same model in q8_0, whose per-row maxima are the same numbers to
within 0.4%. So a normalised sliding correlation of one vector against the
file's float region finds a tensor's scale array by VALUE, with no knowledge of
the layout, and 112 of them tiling the region with no gap and no overlap is a
test the region can fail.

WHAT IT FOUND on Llama-3.2-1B-Instruct-rk3576-w4a16.rkllm:

    per tensor:  `rows` fp32 scales, then `rows` fp32 zero points
    per layer:   the seven matrices in model order, then the two norms
    layer stride 51200 slots, blk.0.attn_q at slot 122728 of the region

and the quantiser is `w = scale * (q - zero)` with `q` in [-8, +7]:

    max/s + z  =  7.19 +- 0.34        min/s + z  =  -8.20 +- 0.33
    (max-min)/s = 14.9999 +- 0.0038   on blk.3/10/15 attn_q and attn_k

    -- four digits, and the +-0.004 is q8_0's own rounding.  Sixteen levels,
    asymmetric, ONE SCALE AND ONE INTEGER ZERO POINT PER OUTPUT ROW.

⚠ THE OTHER 106 TENSORS DO NOT SATISFY THAT, and the deviation is a per-tensor
factor with a few percent of per-row spread, which is what quantising
TRANSFORMED weights looks like. The vendor has a calibration step in front of
its quantiser. Scoring its scales against the untransformed reference produced
`blk.1.ffn_up 99.5%` -- a reconstruction uncorrelated with its own input, which
no shipped quantiser emits -- so `compare` scores ONLY the tensors that pass
`verify`, and says how many that is.

Usage:
    rkllm_scales.py map      <model.rkllm> <ref.gguf>   find every scale array
    rkllm_scales.py verify   <model.rkllm> <ref.gguf>   test the derived map
    rkllm_scales.py compare  <model.rkllm> <ref.gguf>   vendor vs charsiu vs q4_0
    rkllm_scales.py zero     x <ref.gguf>               price the zero point
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/Desktop/llama.cpp-ref/gguf-py"))
from gguf import GGUFReader     # noqa: E402

# The float region of a 1240 MB Llama-3.2-1B .rkllm. Both edges are slack: the
# scan zeroes everything outside a scale's plausible range anyway.
LO, HI = int(508.0 * 2**20), int(512.75 * 2**20)

# One layer of Llama-3.2-1B, in the order the file stores it, then the two
# norms. Each matrix takes 2 * rows slots: the scales, then the zero points.
ORDER = [("attn_q", 2048), ("attn_k", 512), ("attn_v", 512),
         ("attn_output", 2048), ("ffn_gate", 8192), ("ffn_up", 8192),
         ("ffn_down", 2048)]
NORMS, START, LAYERS = 2 * 2048, 122728, 16


def region(path):
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    s = np.frombuffer(mm[LO:HI].tobytes(), dtype=np.float32).astype(np.float64)
    # A scale is positive and small. The fp16 embedding tail in front of the
    # region reads as float32 values near 1e38, and squaring those inside the
    # window norm is what made the first version of this report r = 5e32.
    return np.where(np.isfinite(s) & (s > 0) & (s < 100.0), s, 0.0)


def dequant_q8(t):
    raw = np.array(t.data)
    blk = raw.reshape(raw.shape[0], -1, 34)
    d = blk[:, :, :2].reshape(-1, 2).copy().view(np.float16)
    d = d.reshape(blk.shape[0], -1).astype(np.float64)
    q = blk[:, :, 2:].reshape(blk.shape[0], -1).view(np.int8).astype(np.float64)
    nb = d.shape[1]
    return (q.reshape(q.shape[0], nb, 32) * d[:, :, None]).reshape(q.shape[0], -1)


def derived_map():
    m, at = {}, START
    for L in range(LAYERS):
        for suf, rows in ORDER:
            m[f"blk.{L}.{suf}.weight"] = (at, rows)
            at += 2 * rows
        at += NORMS
    return m


def matrices(r):
    for t in r.tensors:
        if int(t.tensor_type) == 8 and t.name != "token_embd.weight":
            yield t


def cmd_map(rk, ref):
    """Correlate every tensor's row maxima against the region, print the tiling."""
    sig = region(rk)
    n = sig.size
    nfft = 1 << int(np.ceil(np.log2(n + 16384)))
    Sf = np.fft.rfft(sig, nfft)
    cs = np.concatenate(([0.0], np.cumsum(sig * sig)))
    cn = np.concatenate(([0.0], np.cumsum((sig != 0).astype(np.float64))))
    out = []
    for t in matrices(GGUFReader(ref)):
        w = dequant_q8(t)
        pat = np.abs(w).max(axis=1)
        m = pat.size
        dot = np.fft.irfft(Sf * np.fft.rfft(pat[::-1], nfft), nfft)[m - 1:n]
        ss = cs[m:] - cs[:-m]
        # A window of zeros is not a match: without this the argmax lands in
        # the all-zero table after the scales and scores 1e140.
        ok = ((cn[m:] - cn[:-m]) >= 0.999 * m) & (ss > 0)
        c = np.zeros_like(ss)
        c[ok] = dot[ok] / (np.sqrt(ss[ok]) * np.linalg.norm(pat))
        i = int(np.argmax(c))
        out.append((i, m, float(c[i]), t.name))
    out.sort()
    print(f"{'slot':>9} {'rows':>6} {'gap':>7}   {'r':>7}   tensor")
    prev = None
    for i, m, c, name in out:
        gap = "" if prev is None else f"{i - prev:>7}"
        print(f"{i:>9} {m:>6} {gap}   {c:.4f}   {name}")
        prev = i + m
    return 0


def cmd_verify(rk, ref):
    """At each DERIVED slot, test the quantiser's own identity."""
    sig = region(rk)
    r = GGUFReader(ref)
    ok = []
    print(f"{'tensor':28s} {'slot':>8} {'(max-min)/scale':>22}  {'z integral':>11}")
    for name, (slot, rows) in derived_map().items():
        w = dequant_q8(next(x for x in r.tensors if x.name == name))
        s = sig[slot:slot + rows]
        z = sig[slot + rows:slot + 2 * rows]
        g = (w.max(axis=1) - w.min(axis=1)) / np.where(s > 0, s, 1.0)
        # ⚠ z is read through region()'s positive filter, so read it raw.
        good = abs(g.mean() - 15.0) < 0.05 and g.std() < 0.02
        if good:
            ok.append(name)
        print(f"{name:28s} {slot:>8}   {g.mean():8.3f} +-{g.std():8.4f}"
              f"        {'':>4}{'EXACT' if good else ''}")
    print(f"\n{len(ok)} of {len(derived_map())} tensors satisfy "
          f"scale = (max - min) / 15 exactly.")
    print("The rest are the vendor's calibration: it quantised transformed "
          "weights,\nso its scales cannot be scored against this reference.")
    return 0


def grouped(w, g):
    n, k = w.shape
    g = min(g, k)
    pad = (-k) % g
    return np.pad(w, ((0, 0), (0, pad))).reshape(n, -1, g), k


def sym(w, g):
    """q4_0 and charsiu's default: d = -vmax/8 a group, codes -8..7."""
    ww, k = grouped(w, g)
    i = np.abs(ww).argmax(axis=2)
    vmax = np.take_along_axis(ww, i[:, :, None], axis=2)
    d = np.where(vmax == 0, 1.0, vmax / -8.0)
    return (np.clip(np.rint(ww / d), -8, 7) * d).reshape(w.shape[0], -1)[:, :k]


def asym(w, g):
    """The vendor's: s = (max-min)/15, an integer zero, w = s * (q - z)."""
    ww, k = grouped(w, g)
    mx = ww.max(axis=2, keepdims=True)
    mn = ww.min(axis=2, keepdims=True)
    s = np.where(mx - mn == 0, 1.0, (mx - mn) / 15.0)
    z = np.rint(-mn / s) - 8.0
    return ((np.clip(np.rint(ww / s + z), -8, 7) - z) * s
            ).reshape(w.shape[0], -1)[:, :k]


def cmd_compare(rk, ref):
    """Score the vendor's OWN (s, z) beside charsiu's and q4_0's."""
    sig = region(rk)
    r = GGUFReader(ref)
    mm = np.memmap(rk, dtype=np.uint8, mode="r")
    arms = ("vendor", "charsiu 1024", "charsiu row", "q4_0 (32)")
    tot = {a: [0.0, 0.0] for a in arms}
    print(f"{'tensor':28s} " + " ".join(f"{a:>13}" for a in arms))
    scored = 0
    for name, (slot, rows) in derived_map().items():
        w = dequant_q8(next(x for x in r.tensors if x.name == name))
        s = sig[slot:slot + rows]
        g = (w.max(axis=1) - w.min(axis=1)) / np.where(s > 0, s, 1.0)
        if not (abs(g.mean() - 15.0) < 0.05 and g.std() < 0.02):
            continue
        scored += 1
        b = LO + (slot + rows) * 4
        z = np.frombuffer(mm[b:b + rows * 4].tobytes(),
                          dtype=np.float32).astype(np.float64)[:, None]
        ss = s[:, None]
        cases = {"vendor": (np.clip(np.rint(w / ss + z), -8, 7) - z) * ss,
                 "charsiu 1024": sym(w, 1024),
                 "charsiu row": sym(w, w.shape[1]),
                 "q4_0 (32)": sym(w, 32)}
        nw = float(np.linalg.norm(w)) ** 2
        line = []
        for a in arms:
            e = float(np.linalg.norm(w - cases[a])) ** 2
            tot[a][0] += e
            tot[a][1] += nw
            line.append(f"{np.sqrt(e / nw) * 100:12.3f}%")
        print(f"{name:28s} " + " ".join(line))
    print("-" * 84)
    print(f"{'THE ' + str(scored) + ' SCORABLE':28s} " + " ".join(
        f"{np.sqrt(tot[a][0] / tot[a][1]) * 100:12.3f}%" for a in arms))
    print("\n⚠ Weight error, not perplexity. CHARSIU_NPU_W4_CLIP minimises "
          "exactly this\n  number and made KL worse, 0.0989 to 0.2084.")
    return 0


def cmd_zero(rk, ref):
    """What an asymmetric zero point is worth to charsiu, at every group size.

    No vendor data in it, so `rk` is ignored.  The answer is 1.4% at the group
    charsiu ships and 7% at the group it cannot have -- the group IS the K
    slice -- so the vendor's asymmetry follows from its granularity rather than
    beating ours.

    Pass --fast to score every fourth tensor, which reproduces the full run to
    about a hundredth of a point and takes a quarter of the time.
    """
    step = 4 if "--fast" in sys.argv else 1
    r = GGUFReader(ref)
    # ⚠ The row arm's group is the tensor's own k, which is 2048 on most of
    # these and 8192 on ffn_down, so its scale bytes are counted per tensor
    # rather than assumed.  Quoting one group size for it would be a fiction.
    groups = [("row", None), ("1024", 1024), ("512", 512),
              ("128", 128), ("32", 32)]
    arms = [(f"{k:4s} {lab:>4}", f, g)
            for lab, g in groups for k, f in (("sym", sym), ("asym", asym))]
    tot = {a: [0.0, 0.0] for a, _, _ in arms}
    sbytes = {a: 0.0 for a, _, _ in arms}
    nw_all = 0.0
    n = 0
    for i, t in enumerate(matrices(r)):
        if i % step:
            continue
        w = dequant_q8(t).astype(np.float32)
        nw = float(np.linalg.norm(w)) ** 2
        rows, k = w.shape
        n += 1
        nw_all += rows * k
        for a, f, g in arms:
            gg = k if g is None else min(g, k)
            tot[a][0] += float(np.linalg.norm(w - f(w, gg))) ** 2
            tot[a][1] += nw
            per = 8.0 if a.startswith("asym") else 4.0
            sbytes[a] += per * rows * ((k + gg - 1) // gg)
    print(f"{n} int4 matrices of {ref.rsplit('/', 1)[-1]}"
          f"{'  (--fast: every 4th)' if step > 1 else ''}")
    print(f"  {'arm':11s} {'error':>9}  {'bytes a weight':>15}")
    for a, _, _ in arms:
        print(f"  {a:11s} {np.sqrt(tot[a][0] / tot[a][1]) * 100:8.3f}%"
              f"  {0.5 + sbytes[a] / nw_all:15.4f}")
    return 0


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    cmd, rk, ref = sys.argv[1], sys.argv[2], sys.argv[3]
    fn = {"map": cmd_map, "verify": cmd_verify, "compare": cmd_compare,
          "zero": cmd_zero}.get(cmd)
    if fn is None:
        print(__doc__)
        return 1
    return fn(rk, ref)


if __name__ == "__main__":
    sys.exit(main())
