#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
"""
Quantise charsiu's NPU weights to four bits with GPTQ, offline.

WHY OFFLINE AND WHY GPTQ. The hardware gives one scale per output channel per
job. Round to nearest at that granularity is far worse than the group of 32 a
q4_0 file uses, and buying group 32 on this hardware costs sixty four jobs a
projection, which is more than the four bits save. Measured on blk.0.ffn_gate
with the real activation covariance, output weighted relative error:

    RTN  per channel   0.1067
    RTN  group 32      0.0666      (unaffordable)
    GPTQ per channel   0.0394      one scale, one job

So the granularity was never the problem, the quantiser was. GPTQ rounds the
weights one input at a time and pushes the error it makes onto the weights it
has not reached yet, through the inverse Hessian of the calibration activations.
Nothing is stored beyond the same per channel scale, and nothing changes at run
time.

⚠ It only works because THIS model's activations are correlated, which was
measured before the work rather than assumed: the input covariance of
blk.0.ffn_gate has its top eigenvalue carrying 10.4% and its top sixteen 33.7%.
With uncorrelated inputs H is the identity and GPTQ degenerates to RTN.

⚠ And "GPTQ scored worse than RTN" means the code is wrong, not the method:
it is RTN plus a correction to the same objective. The Cholesky is the place to
look. U must satisfy U^T U = H^-1, so U = cholesky(inv(H)).T, with no second
inversion.

Usage:
    charsiu_run ... CHARSIU_CALIB=cal.bin CHARSIU_CALIB_X=1     # dump cal.bin.x
    gptq.py cal.bin.x model.gguf w4-gptq.bin [--group K] [--damp 0.01]
    charsiu_run ... CHARSIU_NPU_W4=1 CHARSIU_W4_FILE=w4-gptq.bin
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/Desktop/llama.cpp-ref/gguf-py"))
from gguf import GGUFReader     # noqa: E402


def read_vectors(path):
    """{name: (n, k, X)} where X is (nxcal, k) float32."""
    out = {}
    with open(path, "rb") as f:
        while True:
            nm = f.read(80)
            if len(nm) < 80:
                break
            n, k, cnt = struct.unpack("<QQI", f.read(20))
            x = np.frombuffer(f.read(cnt * k * 4), dtype="<f4")
            out[nm.split(b"\0")[0].decode()] = (n, k, x.reshape(cnt, k))
    return out


def dequant_row_major(t):
    """A gguf tensor as float64 (n, k). q8_0 only, which is what charsiu runs."""
    raw = np.array(t.data)
    nb = raw.shape[1] // 34
    blk = raw.reshape(raw.shape[0], nb, 34)
    d = blk[:, :, :2].reshape(raw.shape[0], -1).view("<f2").astype(np.float64)
    q = blk[:, :, 2:].reshape(raw.shape[0], -1).view(np.int8).astype(np.float64)
    return (q.reshape(raw.shape[0], nb, 32) * d[:, :, None]).reshape(
        raw.shape[0], nb * 32)


def gptq(W, X, group, damp):
    """Returns the quantised nibbles (int8, -8..7) and the per row scale.

    Vectorised over the output channels, which is the only way this finishes:
    the scalar form is n*k Python iterations, 16.8 million for one feed forward
    tensor. Here it is k iterations of a rank one update over all rows at once.
    """
    n, k = W.shape
    Xf = X.astype(np.float64)
    H = (Xf.T @ Xf) / max(len(X), 1)
    H[np.diag_indices(k)] += damp * max(np.mean(np.diag(H)), 1e-12)
    # U upper triangular with U^T U = H^-1. One inversion, not two.
    U = np.linalg.cholesky(np.linalg.inv(H)).T

    Wf = W.astype(np.float32).copy()
    Uf = U.astype(np.float32)
    Q = np.empty((n, k), dtype=np.int8)
    idx = np.abs(W).argmax(axis=1)
    d = (W[np.arange(n), idx] / -8.0).astype(np.float32)
    dead = d == 0
    d[dead] = 1.0                       # quantises to zero, scale unused

    for lo in range(0, k, group):
        hi = min(lo + group, k)
        for i in range(lo, hi):
            q = np.clip(np.rint(Wf[:, i] / d), -8, 7)
            Q[:, i] = q.astype(np.int8)
            if i + 1 < k:
                e = (Wf[:, i] - q * d) / Uf[i, i]
                Wf[:, i + 1:] -= np.outer(e, Uf[i, i + 1:])
    d[dead] = 0.0
    Q[dead] = 0
    return Q, d


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    xs = read_vectors(sys.argv[1])
    r = GGUFReader(sys.argv[2])
    group = int(sys.argv[sys.argv.index("--group") + 1]) \
        if "--group" in sys.argv else 0
    damp = float(sys.argv[sys.argv.index("--damp") + 1]) \
        if "--damp" in sys.argv else 0.01
    only = sys.argv[sys.argv.index("--only") + 1] \
        if "--only" in sys.argv else None
    by_name = {t.name: t for t in r.tensors}

    with open(sys.argv[3], "wb") as out:
        for name, (n, k, X) in xs.items():
            if only and only not in name:
                continue
            t = by_name.get(name)
            if t is None:
                print(f"  {name}: not in the gguf, skipped")
                continue
            W = dequant_row_major(t)
            if W.shape != (n, k):
                print(f"  {name}: gguf is {W.shape}, calibration says "
                      f"{(n, k)}, skipped")
                continue
            g = group if group else k
            Q, S = gptq(W, X, g, damp)
            E = W - Q.astype(np.float64) * S[:, None]
            H = (X.T.astype(np.float64) @ X.astype(np.float64)) / max(len(X), 1)
            rel = np.sqrt(np.einsum("ij,jk,ik->", E, H, E)
                          / max(np.einsum("ij,jk,ik->", W, H, W), 1e-30))
            print(f"  {name:<28} n={n:<6} k={k:<6} vectors={len(X):<4} "
                  f"rel err {rel:.4f}")
            out.write(name.encode().ljust(80, b"\0"))
            out.write(struct.pack("<QQ", n, k))
            out.write(Q.tobytes())
            out.write(S.tobytes())
    print(f"wrote {sys.argv[3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
