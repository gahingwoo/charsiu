#!/usr/bin/env python3
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0-or-later
"""
Find where a known tensor sits inside a .rkllm, and in what order its nibbles
are stored, by matching SIGNS.

THE IDEA, and why it needs no scales. A symmetric 4 bit quantiser keeps the sign
of every weight it does not round to zero, so the file's nibble sign stream is
the weight sign stream in whatever order the vendor stores them. We hold the
same model in q8_0, which keeps the sign of every weight it does not round to
zero either. So a sign correlation locates a tensor and reads its order, and the
scale never enters.

Random alignments score 0 and a real one scores 0.8 or better, because only the
weights that quantise to zero disagree.

The whole file's nibble stream is correlated with one channel's sign pattern by
FFT, so every byte AND nibble alignment is tested at once rather than sampled.

Usage:
    rkllm_layout.py <model.rkllm> <ref.gguf> <tensor> [--row N] [--len L]
                    [--mode row|col] [--start OFF] [--end OFF]
    rkllm_layout.py <model.rkllm> x x --selftest OFF [--len L] [--start] [--end]

RESULT so far: row major is DEAD.  blk.0.ffn_gate row 0 against every byte and
nibble alignment of Llama-3.2-1B's 480 MB int4 region tops out at |r| = 0.133,
which is the noise floor for a 2048 long pattern.  The weights are in some
NPU-native order.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser("~/Desktop/llama.cpp-ref/gguf-py"))
from gguf import GGUFReader     # noqa: E402


NFFT = 1 << 21


def nibble_signs(mm, b0, b1):
    """+1 where the nibble is 0..7, -1 where it is 8..15, low nibble first.

    Which of those is 'negative' depends on whether the vendor stores a two's
    complement nibble or an unsigned one with a zero point of 8, and the answer
    only flips the SIGN of the correlation, so both are found by taking |r|.
    """
    b = np.asarray(mm[b0:b1])
    out = np.empty(b.size * 2, dtype=np.float32)
    out[0::2] = np.where((b & 0xF) < 8, 1.0, -1.0)
    out[1::2] = np.where((b >> 4) < 8, 1.0, -1.0)
    return out


def xcorr_max(path, pat, start, end, topk=8):
    """Every alignment of pat over the file, by overlap-save FFT.

    ⚠ ONE BLOCK AT A TIME.  The first version built the whole nibble stream
    first -- 1.5 G entries for a 1.3 GB .rkllm -- and the OOM killer took it
    before it printed anything.  Nothing here is larger than one FFT block, so
    the peak is tens of megabytes whatever the file size.
    """
    m = pat.size
    step = NFFT - m + 1
    pf = np.fft.rfft(pat[::-1].astype(np.float64), NFFT)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    best = []
    p = start * 2
    while p + m <= end * 2:
        take = min(NFFT, end * 2 - p)
        b0 = p // 2
        chunk = nibble_signs(mm, b0, (p + take + 1) // 2)[p - b0 * 2:][:take]
        if chunk.size < m:
            break
        c = np.fft.irfft(np.fft.rfft(chunk.astype(np.float64), NFFT) * pf, NFFT)
        valid = c[m - 1:m - 1 + chunk.size - m + 1]
        k = min(topk, valid.size)
        idx = np.argpartition(np.abs(valid), -k)[-k:]
        for i in idx:
            best.append((float(valid[i]) / m, int(p + i)))
        best.sort(key=lambda t: -abs(t[0]))
        del best[topk:]
        p += step
    return best


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    rk, ref, name = sys.argv[1], sys.argv[2], sys.argv[3]
    row = int(sys.argv[sys.argv.index("--row") + 1]) if "--row" in sys.argv else 0
    ln = int(sys.argv[sys.argv.index("--len") + 1]) if "--len" in sys.argv else 2048
    start = int(sys.argv[sys.argv.index("--start") + 1], 0) if "--start" in sys.argv \
        else 0x21000000

    if "--selftest" in sys.argv:
        #
        # Lift the pattern OUT OF THE FILE at a known offset and search for it.
        # A miss and a broken correlator print the same thing, and this is the
        # only arm that can tell them apart: it must score exactly +1.0000 at
        # the offset it was taken from, with the background around 0.13.
        #
        at = int(sys.argv[sys.argv.index("--selftest") + 1], 0)
        mm = np.memmap(rk, dtype=np.uint8, mode="r")
        pat = nibble_signs(mm, at, at + ln // 2 + 1)[:ln]
        end = int(sys.argv[sys.argv.index("--end") + 1], 0) if "--end" in sys.argv \
            else os.path.getsize(rk)
        print(f"selftest: {ln} nibbles lifted from {at:#x}, "
              f"expect r = +1.0000 at nibble {at * 2}")
        for c, i in xcorr_max(rk, pat, start, end):
            print(f"  {c:+9.4f}   {i:>12d}   {i // 2:#012x}")
        return 0

    r = GGUFReader(ref)
    t = next(x for x in r.tensors if x.name == name)
    raw = np.array(t.data)
    print(f"{name}: gguf shape {tuple(int(x) for x in t.shape)}, "
          f"raw array {raw.shape} {raw.dtype}")
    if raw.ndim != 2:
        print("not a matrix")
        return 1
    #
    # ⚠ gguf-py hands back the RAW BLOCKS for a quantised tensor, not floats,
    # and round one of this tool took np.sign of those bytes -- which is +1
    # nearly everywhere and correlated with the file's mean rather than with
    # anything. q8_0 is 34 bytes a block: an fp16 scale then 32 signed bytes,
    # and the scale is positive, so the sign of the weight IS the sign of the
    # stored byte.
    #
    blk = raw.reshape(raw.shape[0], -1, 34)
    qs = blk[:, :, 2:].reshape(raw.shape[0], -1).view(np.int8)
    print(f"q8_0: {blk.shape[1]} blocks a row, {qs.shape[1]} weights a row")
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "row"
    if mode == "row":
        pat = np.sign(qs[row][:ln]).astype(np.int8)
    elif mode == "col":
        #
        # The other simple order: consecutive nibbles are consecutive CHANNELS
        # at one k. Ruling out a whole family costs one run.
        #
        pat = np.sign(qs[:, row][:ln]).astype(np.int8)
    else:
        print("modes: row, col")
        return 1
    print(f"mode {mode}")
    nz = int(np.count_nonzero(pat))
    print(f"pattern: row {row}, {ln} weights, {nz} of them nonzero "
          f"({100.0 * nz / ln:.1f}%)")

    end = int(sys.argv[sys.argv.index("--end") + 1], 0) if "--end" in sys.argv \
        else os.path.getsize(rk)
    print(f"signal: {(end - start) * 2} nibbles from {start:#x} to {end:#x}")
    hits = xcorr_max(rk, pat, start, end)
    print("\n  correlation   nibble        byte offset   note")
    for c, i in hits:
        print(f"  {c:+9.4f}   {i:>10d}   {start + i // 2:#012x}   "
              f"{'HIGH nibble first' if i % 2 else 'low nibble first'}")
    print("\n  |r| ~ 0.9  the tensor is here and stored in this order")
    print("  |r| < 0.1  it is not stored this way anywhere in the file")
    #
    # ⚠ RUN --selftest FIRST ON A NEW FILE.  A miss and a broken correlator
    # look identical, and lifting the pattern out of the file at a known offset
    # scores exactly +1.0000 there when the machinery works.
    #
    return 0


if __name__ == "__main__":
    sys.exit(main())
