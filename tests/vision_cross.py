#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
"""
Diff the vision tower against a numpy reference, with no model and no board.

The tower is written here -- random weights, a fixed seed, the same names and
shapes a real mmproj uses -- and the reference forward is fifty lines of numpy
directly above the comparison. So the C and the reference disagree only where
one of them is wrong, and both run on a laptop in a second.

⚠ WHAT THIS DOES NOT COVER, and both are the same kind of thing: the reference
was written by whoever wrote the C, so a convention they SHARE passes.

  - the tensor NAMES. Given tensors under those names the arithmetic is right;
    whether a real mmproj uses those names is a question only a real mmproj
    answers, and `charsiu_vision FILE` prints the misses when it does not.
  - the patch gather ORDER. Both sides lay a patch out x fastest, then y, then
    channel, because that is the order ggml stores a [kw][kh][in_c] kernel in.
    If that reading is wrong, both sides are wrong together and this still says
    PASS. It is the first thing to suspect if a real model produces fluent
    sentences about the wrong picture.

    tests/vision_cross.py build/charsiu_vision
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmproj_synth as ms                                  # noqa: E402

# f32 summation order alone. Anything above this is not rounding.
TOL = 2e-4


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 *
                                    (x + 0.044715 * x ** 3)))


def layernorm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def pixel_shuffle(x, grid, E, s):
    """transformers' Idefics3 pixel_shuffle, transcribed from the model file.

    ⚠ TWO RESHAPES WITH A TRANSPOSE BETWEEN THEM, not a block gather. The two
    differ, and both produce finite numbers."""
    x = x.reshape(grid, grid, E)
    x = x.reshape(grid, grid // s, E * s)
    x = x.transpose(1, 0, 2)
    x = x.reshape(grid // s, grid // s, E * s * s)
    x = x.transpose(1, 0, 2)
    return x.reshape((grid // s) * (grid // s), E * s * s)


def reference(T, eps=1e-6, kind="mlp"):
    """The same tower, in numpy. T is {name: array} in gguf shapes."""
    P, G, NP = ms.PATCH, ms.GRID, ms.PATCHES
    W, FF, H = ms.WIDTH, ms.FF, ms.HEADS
    hd = W // H

    # the image the C side generates, and the same normalisation
    n = 3 * ms.IMAGE * ms.IMAGE
    px = (np.arange(n) % 251).astype(np.float32) / np.float32(251.0)
    px = px * np.float32(2.0) - np.float32(1.0)
    px = (px.reshape(3, ms.IMAGE, ms.IMAGE) - np.float32(0.5)) / np.float32(0.5)

    # the patch gather: x fastest, then y, then channel
    patch = np.zeros((NP, 3 * P * P), dtype=np.float32)
    for p in range(NP):
        gy, gx = divmod(p, G)
        blk = px[:, gy * P:(gy + 1) * P, gx * P:(gx + 1) * P]
        patch[p] = blk.reshape(-1)

    def w2(name, nin, nout):
        """gguf ne is [in, out], so the flat data is [out][in]."""
        return T[name].reshape(nout, nin)

    x = patch @ w2("v.patch_embd.weight", 3 * P * P, W).T
    x = x + T["v.patch_embd.bias"]
    x = x + T["v.position_embd.weight"].reshape(NP, W)

    for i in range(ms.LAYERS):
        p = f"v.blk.{i}."
        xb = layernorm(x, T[p + "ln1.weight"], T[p + "ln1.bias"], eps)
        q = xb @ w2(p + "attn_q.weight", W, W).T + T[p + "attn_q.bias"]
        k = xb @ w2(p + "attn_k.weight", W, W).T + T[p + "attn_k.bias"]
        v = xb @ w2(p + "attn_v.weight", W, W).T + T[p + "attn_v.bias"]
        o = np.zeros_like(x)
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            s = q[:, sl] @ k[:, sl].T / np.sqrt(hd)
            s = s - s.max(-1, keepdims=True)
            s = np.exp(s)
            s = s / s.sum(-1, keepdims=True)
            o[:, sl] = s @ v[:, sl]
        x = x + o @ w2(p + "attn_out.weight", W, W).T + T[p + "attn_out.bias"]

        xb = layernorm(x, T[p + "ln2.weight"], T[p + "ln2.bias"], eps)
        h1 = xb @ w2(p + "ffn_down.weight", W, FF).T + T[p + "ffn_down.bias"]
        h1 = gelu(h1)
        x = x + h1 @ w2(p + "ffn_up.weight", FF, W).T + T[p + "ffn_up.bias"]

    x = layernorm(x, T["v.post_ln.weight"], T["v.post_ln.bias"], eps)
    if kind == "idefics3":
        s = ms.SCALE
        x = pixel_shuffle(x, G, W, s)
        x = x @ w2("mm.model.fc.weight", W * s * s, ms.PROJ).T
    else:
        x = x @ w2("mm.0.weight", W, ms.PROJ).T + T["mm.0.bias"]
    return x.astype(np.float32)


def one(tool, kind):
    rng = np.random.default_rng(20260828)

    data = {}
    for name, ne in ms.tensors(kind):
        n = 1
        for d in ne:
            n *= d
        # small values: a random ViT saturates its own softmax otherwise
        data[name] = rng.standard_normal(n).astype(np.float32) * np.float32(0.1)

    tmp = tempfile.mkdtemp()
    path = ms.write(os.path.join(tmp, f"mmproj-{kind}.gguf"), data=data,
                    kind=kind)

    r = subprocess.run([tool, path, "--encode"], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        print("FAILED: the tower would not run")
        return 1
    lines = r.stdout.strip().split("\n")
    tok, width = (int(v) for v in lines[0].split()[1:])
    got = np.array([float(v) for v in lines[1:]], dtype=np.float32)
    got = got.reshape(tok, width)

    want = reference(data, kind=kind)
    if want.shape != got.shape:
        print(f"FAILED: shape {got.shape} against {want.shape}")
        return 1

    err = np.abs(got - want)
    worst = err.max()
    at = np.unravel_index(err.argmax(), err.shape)
    print(f"{kind}: {tok} embeddings of {width}")
    print(f"worst |charsiu - reference| = {worst:.3e} at patch {at[0]} dim {at[1]}")
    print(f"  charsiu {got[at]:.6f}   reference {want[at]:.6f}")
    if worst > TOL:
        bad = int((err > TOL).sum())
        print(f"FAILED: {bad} of {err.size} above {TOL}")
        return 1
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    fail = 0
    for kind in ("mlp", "idefics3"):
        fail |= one(sys.argv[1], kind)
    print("FAILED" if fail else "PASS")
    return fail


if __name__ == "__main__":
    sys.exit(main())
