#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
"""
Diff CLIP's two towers against a numpy reference, with no model and no board.

vision_cross.py covers the towers that feed a language model. This covers the
two things a RETRIEVAL tower does differently, and each of them is invisible in
the output if it is wrong:

  - the CLASS TOKEN. CLIP prepends one, so the sequence is one longer than the
    patch grid, the position embedding has one more row, and the picture's
    embedding is that token's -- post normalised ALONE and projected.
  - the CAUSAL MASK on the text tower. With full attention every embedding
    still comes back finite and the cosine similarities still order themselves
    plausibly, and the end of text row is the one a missing mask changes least.

    tests/clip_cross.py build/charsiu_vision build/charsiu_clip
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmproj_synth as ms                                  # noqa: E402
from vision_cross import gelu, layernorm                   # noqa: E402

TOL = 2e-4


def vision_reference(T, eps=1e-6):
    """The CLIP vision tower: class token, full attention, pooled at row 0."""
    P, G, NP = ms.PATCH, ms.GRID, ms.PATCHES
    W, FF, H = ms.WIDTH, ms.FF, ms.HEADS
    hd = W // H

    n = 3 * ms.IMAGE * ms.IMAGE
    px = (np.arange(n) % 251).astype(np.float32) / np.float32(251.0)
    px = px * np.float32(2.0) - np.float32(1.0)
    px = (px.reshape(3, ms.IMAGE, ms.IMAGE) - np.float32(0.5)) / np.float32(0.5)

    patch = np.zeros((NP, 3 * P * P), dtype=np.float32)
    for p in range(NP):
        gy, gx = divmod(p, G)
        patch[p] = px[:, gy * P:(gy + 1) * P, gx * P:(gx + 1) * P].reshape(-1)

    def w2(name, nin, nout):
        return T[name].reshape(nout, nin)

    x = np.zeros((NP + 1, W), dtype=np.float32)
    x[0] = T["v.class_embd"]
    x[1:] = patch @ w2("v.patch_embd.weight", 3 * P * P, W).T \
        + T["v.patch_embd.bias"]
    x = x + T["v.position_embd.weight"].reshape(NP + 1, W)

    for i in range(ms.LAYERS):
        p = f"v.blk.{i}."
        xb = layernorm(x, T[p + "ln1.weight"], T[p + "ln1.bias"], eps)
        q = xb @ w2(p + "attn_q.weight", W, W).T + T[p + "attn_q.bias"]
        k = xb @ w2(p + "attn_k.weight", W, W).T + T[p + "attn_k.bias"]
        v = xb @ w2(p + "attn_v.weight", W, W).T + T[p + "attn_v.bias"]
        o = np.zeros_like(x)
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            sc = q[:, sl] @ k[:, sl].T / np.sqrt(hd)
            sc = np.exp(sc - sc.max(-1, keepdims=True))
            o[:, sl] = (sc / sc.sum(-1, keepdims=True)) @ v[:, sl]
        x = x + o @ w2(p + "attn_out.weight", W, W).T + T[p + "attn_out.bias"]
        xb = layernorm(x, T[p + "ln2.weight"], T[p + "ln2.bias"], eps)
        h1 = gelu(xb @ w2(p + "ffn_down.weight", W, FF).T + T[p + "ffn_down.bias"])
        x = x + h1 @ w2(p + "ffn_up.weight", FF, W).T + T[p + "ffn_up.bias"]

    pooled = layernorm(x[0], T["v.post_ln.weight"], T["v.post_ln.bias"], eps)
    return (pooled @ w2("visual_projection.weight", W, ms.PROJ).T
            ).astype(np.float32)


def text_reference(T, ids, eps=1e-6):
    """The CLIP text tower: causal, pooled at the last row."""
    W, FF, H, n = ms.TW, ms.TFF, ms.THEADS, len(ids)
    hd = W // H

    def w2(name, nin, nout):
        return T[name].reshape(nout, nin)

    x = T["t.token_embd.weight"].reshape(ms.TVOCAB, W)[list(ids)].copy()
    x = x + T["t.position_embd.weight"].reshape(ms.TCTX, W)[:n]

    mask = np.triu(np.full((n, n), -np.inf, dtype=np.float32), 1)
    for i in range(ms.TLAYERS):
        p = f"t.blk.{i}."
        xb = layernorm(x, T[p + "ln1.weight"], T[p + "ln1.bias"], eps)
        q = xb @ w2(p + "attn_q.weight", W, W).T + T[p + "attn_q.bias"]
        k = xb @ w2(p + "attn_k.weight", W, W).T + T[p + "attn_k.bias"]
        v = xb @ w2(p + "attn_v.weight", W, W).T + T[p + "attn_v.bias"]
        o = np.zeros_like(x)
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            sc = q[:, sl] @ k[:, sl].T / np.sqrt(hd) + mask
            sc = np.exp(sc - sc.max(-1, keepdims=True))
            o[:, sl] = (sc / sc.sum(-1, keepdims=True)) @ v[:, sl]
        x = x + o @ w2(p + "attn_out.weight", W, W).T + T[p + "attn_out.bias"]
        xb = layernorm(x, T[p + "ln2.weight"], T[p + "ln2.bias"], eps)
        h1 = gelu(xb @ w2(p + "ffn_down.weight", W, FF).T + T[p + "ffn_down.bias"])
        x = x + h1 @ w2(p + "ffn_up.weight", FF, W).T + T[p + "ffn_up.bias"]

    pooled = layernorm(x[-1], T["t.post_ln.weight"], T["t.post_ln.bias"], eps)
    return (pooled @ w2("text_projection.weight", W, ms.PROJ).T).astype(np.float32)


def compare(what, got, want):
    err = np.abs(got - want)
    worst = err.max()
    print(f"{what}: {got.size} values, worst {worst:.3e}")
    if worst > TOL:
        print(f"  FAILED: {int((err > TOL).sum())} above {TOL}")
        return 1
    return 0


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    vtool, ctool = sys.argv[1], sys.argv[2]
    rng = np.random.default_rng(20260828)

    data = {}
    for name, ne in ms.tensors("clip"):
        n = 1
        for d in ne:
            n *= d
        data[name] = rng.standard_normal(n).astype(np.float32) * np.float32(0.1)

    tmp = tempfile.mkdtemp()
    path = ms.write(os.path.join(tmp, "clip-cross.gguf"), data=data, kind="clip")
    fail = 0

    r = subprocess.run([vtool, path, "--encode"], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        return 1
    lines = r.stdout.strip().split("\n")
    got = np.array([float(v) for v in lines[1:]], dtype=np.float32)
    fail |= compare("vision (class token pooled)", got, vision_reference(data))

    ids = [ms.TVOCAB - 2, 3, 7, 3, 11, ms.TVOCAB - 1]
    r = subprocess.run([ctool, path, "--text-ids"] + [str(i) for i in ids],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        return 1
    lines = r.stdout.strip().split("\n")
    got = np.array([float(v) for v in lines[1:]], dtype=np.float32)
    fail |= compare("text (causal, pooled at the end)", got,
                    text_reference(data, ids))

    print("FAILED" if fail else "PASS")
    return fail


if __name__ == "__main__":
    sys.exit(main())
