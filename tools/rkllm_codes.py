#!/usr/bin/env python3
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Solve the vendor's int4 weight layout from the file, then score its codes.

The layout is DERIVED, not guessed: 336 guessed orderings sat at the noise
floor.  Three measurements pin it --

  1. blocks of 16 output channels (window-mean knee at 32768 codes; and WHICH
     16 from block means, which survive any permutation inside the block:
     r = 1.0000 for contiguous groups)
  2. a cycle of 512 codes (autocorrelation of window means inside blocks)
  3. the row of each of the 512 positions, fitted over all 128 blocks at once
     (each row used exactly 32 times, margin median 15.5)

and then k falls out the same way: at each slot the 128 blocks give 128
observed codes, and matching those against the 2048 candidate columns scores
48/48 for the right one against 16/48 for the runner-up.

Held out properly -- solved on blocks 0..47, scored on rows 768..2047 --
it reads 99.72% on codes away from a rounding boundary, and the residual
disagreements are +-1, which is the boundary signature.

⚠ SOLVE IT ON A TENSOR THE VENDOR DID NOT TRANSFORM. Caching the map by shape
and letting the first tensor of that shape fill the cache put blk.0.attn_q
(rho 3.507) in charge of the 2048x2048 map and the whole table came back at
279%: a mapping fitted to predictions that are wrong fits nothing.

ffn_down (k = 8192) IS this layout with k split: its block is 32 KB rather than
16 KB and holds 16 rows x 4096 k, with k as the OUTER loop -- blocks 0..127 are
row groups 0..127 over k 0..4095, blocks 128..255 the same groups over
k 4096..8191, and inside a block the 4096 k are two 2048-k blocks in a row.
Searching each block over the 128 row groups x 2 halves picks the identity with
a median 93.43% against a runner-up of 18.07%, all 256 blocks over 80%, and
exactly 256 distinct slots used.

RESULT, over the 41 tensors whose rho is within 5% of 1, so the reference IS
what the vendor quantised:

    vendor's own codes   17.577%
    charsiu group 1024   13.946%     <- what ships
    llama.cpp q4_0 / 32   8.856%

and on blk.3.attn_q alone the vendor reads 15.859% here against the 15.843%
that came out of applying the vendor's (scale, zero) to charsiu's own rounding.
Two routes, one number.
"""
import sys
import numpy as np
sys.path.insert(0, "/home/parallels/Desktop/llama.cpp-ref/gguf-py")
from gguf import GGUFReader     # noqa: E402

RK = "/home/parallels/Documents/kiln/model/Llama-3.2-1B-Instruct-rk3576-w4a16.rkllm"
REF = "/home/parallels/Desktop/charsiu/models/Llama-3.2-1B-Instruct-Q8_0.gguf"
LO = int(508.0 * 2**20)
INT4 = 0x20DDA9C4
mm = np.memmap(RK, dtype=np.uint8, mode="r")
S = [("attn_q", 2048, 2048), ("attn_k", 512, 2048), ("attn_v", 512, 2048),
     ("attn_output", 2048, 2048), ("ffn_gate", 8192, 2048),
     ("ffn_up", 8192, 2048), ("ffn_down", 2048, 8192)]


def f32(slot, n):
    o = LO + slot * 4
    return np.frombuffer(mm[o:o + n * 4].tobytes(), dtype=np.float32).astype(np.float64)


def deq(t):
    raw = np.array(t.data)
    bl = raw.reshape(raw.shape[0], -1, 34)
    d = bl[:, :, :2].reshape(-1, 2).copy().view(np.float16)
    d = d.reshape(bl.shape[0], -1).astype(np.float64)
    q = bl[:, :, 2:].reshape(bl.shape[0], -1).view(np.int8).astype(np.float64)
    return (q.reshape(q.shape[0], d.shape[1], 32) * d[:, :, None]).reshape(q.shape[0], -1)


def codes(off, nbytes):
    b = np.asarray(mm[off:off + nbytes])
    c = np.empty(b.size * 2, dtype=np.int8)
    c[0::2] = np.where((b & 0xF) > 7, (b & 0xF) - 16, (b & 0xF))
    c[1::2] = np.where((b >> 4) > 7, (b >> 4) - 16, (b >> 4))
    return c


def solve(P, C, fit):
    """P: predicted codes (rows, k).  C: file codes (nblk, slots)."""
    rows, k = P.shape
    nblk, slots = C.shape
    rowmean = P.astype(np.float64).mean(axis=1)
    M = C.astype(np.float64).reshape(nblk, slots // 512, 512).mean(axis=1)
    RM = rowmean.reshape(nblk, 16)
    rho = ((M[:, :, None] - RM[:, None, :]) ** 2).sum(axis=0).argmin(axis=1)
    kk = np.zeros(slots, dtype=np.int32)
    for rv in range(16):
        sl = np.where(rho == rv)[0]
        idx = (np.arange(slots // 512)[:, None] * 512 + sl[None, :]).reshape(-1)
        O = C[:fit][:, idx]
        Mk = P[np.arange(fit) * 16 + rv]
        cnt = np.zeros((idx.size, k), dtype=np.float32)
        for v in range(-8, 8):
            cnt += (O == v).T.astype(np.float32) @ (Mk == v).astype(np.float32)
        kk[idx] = cnt.argmax(axis=1)
    return rho, kk


def sym(w, g):
    n, k = w.shape
    g = min(g, k)
    ww = w.reshape(n, -1, g)
    i = np.abs(ww).argmax(axis=2)
    vm = np.take_along_axis(ww, i[:, :, None], axis=2)
    d = np.where(vm == 0, 1.0, vm / -8.0)
    return (np.clip(np.rint(ww / d), -8, 7) * d).reshape(n, k)


def main():
    r = GGUFReader(REF)
    G = {t.name: t for t in r.tensors}
    SLOT, a = {}, 122728
    for L in range(16):
        for nm, n, _ in S:
            SLOT[f"blk.{L}.{nm}.weight"] = (a, n)
            a += 2 * n
        a += 4096
    WOFF, a = {}, INT4
    for L in range(16):
        for nm, n, k in S:
            WOFF[f"blk.{L}.{nm}.weight"] = (a, n, k)
            a += n * k // 2
    #
    # ⚠ SOLVE THE MAP ON A TENSOR THE VENDOR DID NOT TRANSFORM. Caching it by
    # shape and letting the first tensor of that shape fill the cache put
    # blk.0.attn_q (rho 3.507) in charge of the 2048x2048 map, and the whole
    # table came back at 279% -- a mapping fitted to predictions that are
    # wrong. These three have rho within 2.5% of 1.
    #
    ANCHOR = {(2048, 2048): "blk.3.attn_q.weight",
              (512, 2048): "blk.3.attn_k.weight",
              (8192, 2048): "blk.6.ffn_gate.weight"}
    maps = {}
    for shape, anch in ANCHOR.items():
        off, rows, kd = WOFF[anch]
        slot, _ = SLOT[anch]
        W = deq(G[anch])
        s = f32(slot, rows)
        z = f32(slot + rows, rows)
        P = np.clip(np.rint(W / s[:, None] + z[:, None]), -8, 7).astype(np.int8)
        C = codes(off, rows * kd // 2).reshape(rows // 16, -1)
        maps[shape] = solve(P, C, min(48, rows // 16))
        print(f"  map for {shape} solved on {anch}")
    #
    # ffn_down: same block, k split in two and k on the OUTSIDE.  Built from
    # the 2048-k map rather than solved again, and the block search that says
    # so picks the identity at a median 93.43% against a runner-up of 18.07%.
    #
    rho2, kk2 = maps[(2048, 2048)]
    maps[(2048, 8192)] = (rho2, np.concatenate([kk2, kk2 + 2048]))
    T = {x: [0.0, 0.0] for x in ("vendor", "chr1024", "q40")}
    print(f"{'tensor':24s} {'vendor':>9} {'chr1024':>9} {'q4_0/32':>9}  note")
    for L in range(16):
        for nm, n, k in S:
            pass
            name = f"blk.{L}.{nm}.weight"
            off, rows, kd = WOFF[name]
            slot, _ = SLOT[name]
            W = deq(G[name])
            s = f32(slot, rows)
            z = f32(slot + rows, rows)
            P = np.clip(np.rint(W / s[:, None] + z[:, None]), -8, 7).astype(np.int8)
            C = codes(off, rows * kd // 2).reshape(rows // 16, -1)
            rho, kk = maps[(rows, kd)]
            if kd == 8192:
                #
                # 256 blocks of 16 rows x 4096 k.  The order is not
                # (row group, k half) and not (k half, row group): it is 64
                # row groups at a time, each superblock doing k-half 0 for all
                # 64 and then k-half 1 for all 64.  Assuming the simpler order
                # read 54.29% where this reads 98.99% on confident codes, and
                # the tell was that the FIRST twelve blocks agree with both.
                #
                C = codes(off, rows * kd // 2).reshape(-1, 65536)
                nb = C.shape[0]
                rowidx = rho[np.arange(65536) % 512]
                b = np.arange(nb)
                gid = 64 * (b // 128) + (b % 128) % 64
                hid = (b % 128) // 64
                R = (gid[:, None] * 16 + rowidx[None, :]).reshape(-1)
                K = (kk[None, :] + 4096 * hid[:, None]).reshape(-1)
            else:
                slots = C.shape[1]
                rowidx = rho[np.arange(slots) % 512]
                R = (np.arange(rows // 16)[:, None] * 16 + rowidx[None, :]).reshape(-1)
                K = np.broadcast_to(kk[None, :], (rows // 16, slots)).reshape(-1)
            Q = np.zeros((rows, kd))
            Q[R, K] = C.reshape(-1)
            V = (Q - z[:, None]) * s[:, None]
            #
            # Only tensors the vendor did NOT transform can be scored against
            # this reference: rho = 15 * scale / (max - min) is 1 exactly when
            # it quantised these weights.
            #
            rho_t = float((15.0 * s / (W.max(axis=1) - W.min(axis=1))).mean())
            if abs(rho_t - 1.0) > 0.05:
                continue
            nw = float(np.linalg.norm(W)) ** 2
            for key, arr in (("vendor", V), ("chr1024", sym(W, 1024)), ("q40", sym(W, 32))):
                T[key][0] += float(np.linalg.norm(W - arr)) ** 2
                T[key][1] += nw
            if True:
                print(f"{name:24s} "
                      f"{np.linalg.norm(W - V) / np.linalg.norm(W) * 100:8.3f}% "
                      f"{np.linalg.norm(W - sym(W, 1024)) / np.linalg.norm(W) * 100:8.3f}% "
                      f"{np.linalg.norm(W - sym(W, 32)) / np.linalg.norm(W) * 100:8.3f}%")
    print("-" * 60)
    print(f"{'THE SCORABLE ONES':24s} " + " ".join(
        f"{np.sqrt(T[x][0] / T[x][1]) * 100:8.3f}%" for x in ("vendor", "chr1024", "q40")))
    print("\n⚠ The vendor column is its ACTUAL stored codes read through the")
    print("  solved layout, against the same q8_0 reference. Tensors the vendor")
    print("  transformed before quantising will read HIGH here -- the transform")
    print("  is not undone -- so this is an upper bound on their error, tight")
    print("  only where rho == 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
