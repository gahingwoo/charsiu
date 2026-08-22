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
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser("~/Desktop/llama.cpp-ref/gguf-py"))
from gguf import GGUFReader     # noqa: E402


def nibble_signs(path, start, end):
    """+1 where the nibble is 0..7, -1 where it is 8..15, low nibble first.

    Which of those is 'negative' depends on whether the vendor stores a two's
    complement nibble or an unsigned one with a zero point of 8, and the answer
    only flips the SIGN of the correlation, so both are found by taking |r|.
    """
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    b = np.asarray(mm[start:end])
    out = np.empty(b.size * 2, dtype=np.int8)
    out[0::2] = np.where((b & 0xF) < 8, 1, -1)
    out[1::2] = np.where((b >> 4) < 8, 1, -1)
    return out


def xcorr_max(sig, pat, topk=8):
    """Every alignment of pat over sig, by overlap-save FFT. Returns the best."""
    m = pat.size
    nfft = 1 << 22
    step = nfft - m + 1
    pf = np.fft.rfft(pat[::-1].astype(np.float32), nfft)
    best = []
    for off in range(0, sig.size, step):
        chunk = sig[off:off + nfft]
        if chunk.size < m:
            break
        c = np.fft.irfft(np.fft.rfft(chunk.astype(np.float32), nfft) * pf, nfft)
        valid = c[m - 1:m - 1 + min(step, chunk.size - m + 1)]
        idx = np.argpartition(np.abs(valid), -topk)[-topk:]
        for i in idx:
            best.append((float(valid[i]) / m, int(off + i)))
    best.sort(key=lambda t: -abs(t[0]))
    return best[:topk]


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    rk, ref, name = sys.argv[1], sys.argv[2], sys.argv[3]
    row = int(sys.argv[sys.argv.index("--row") + 1]) if "--row" in sys.argv else 0
    ln = int(sys.argv[sys.argv.index("--len") + 1]) if "--len" in sys.argv else 2048
    start = int(sys.argv[sys.argv.index("--start") + 1], 0) if "--start" in sys.argv \
        else 0x21000000

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

    sig = nibble_signs(rk, start, os.path.getsize(rk))
    print(f"signal: {sig.size} nibbles from {start:#x}")
    hits = xcorr_max(sig, pat)
    print("\n  correlation   nibble        byte offset   note")
    for c, i in hits:
        print(f"  {c:+9.4f}   {i:>10d}   {start + i // 2:#012x}   "
              f"{'HIGH nibble first' if i % 2 else 'low nibble first'}")
    print("\n  |r| ~ 0.9  the tensor is here and stored in this order")
    print("  |r| < 0.1  it is not stored this way anywhere in the file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
