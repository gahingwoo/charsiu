#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
"""
Diff charsiu's whisper audio encoder against numpy, on the REAL weights.

The synthetic-tower trick the vision tests use does not help here: whisper's
container is not something a test should learn to write, and the model is small
enough that the real one is the easier reference. whisper_model.py parses the
file independently, so a tensor charsiu misreads does not pass.

    tests/whisper_encoder_cross.py build/charsiu_whisper MODEL.bin
"""

import subprocess
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from whisper_model import Model                            # noqa: E402
from mel_cross import reference as mel_reference           # noqa: E402

TOL = 2e-3


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 *
                                    (x + 0.044715 * x ** 3)))


def layernorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def conv1d(x, w, b, stride):
    """x [in][len], w [out][in][3], padding 1."""
    n_in, ln = x.shape
    n_out = w.shape[0]
    xp = np.pad(x, ((0, 0), (1, 1)))
    olen = (ln + stride - 1) // stride
    out = np.zeros((olen, n_out), dtype=np.float64)
    for tap in range(3):
        cols = xp[:, tap:tap + (olen - 1) * stride + 1:stride].T   # [olen][in]
        out += cols @ w[:, :, tap].T
    return out + b


def encoder(m, mel):
    W = m.hp["n_audio_state"]
    H = m.hp["n_audio_head"]
    T = m.hp["n_audio_ctx"]
    hd = W // H

    x = conv1d(mel.astype(np.float64), m["encoder.conv1.weight"],
               m["encoder.conv1.bias"].reshape(-1), 1)
    x = gelu(x)
    x = conv1d(x.T, m["encoder.conv2.weight"],
               m["encoder.conv2.bias"].reshape(-1), 2)
    x = gelu(x)
    assert x.shape == (T, W), x.shape
    x = x + m["encoder.positional_embedding"]

    for i in range(m.hp["n_audio_layer"]):
        p = f"encoder.blocks.{i}."
        xb = layernorm(x, m[p + "attn_ln.weight"], m[p + "attn_ln.bias"])
        q = xb @ m[p + "attn.query.weight"].T + m[p + "attn.query.bias"]
        k = xb @ m[p + "attn.key.weight"].T          # no bias, by design
        v = xb @ m[p + "attn.value.weight"].T + m[p + "attn.value.bias"]
        o = np.zeros_like(x)
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            s = q[:, sl] @ k[:, sl].T / np.sqrt(hd)
            s = np.exp(s - s.max(-1, keepdims=True))
            o[:, sl] = (s / s.sum(-1, keepdims=True)) @ v[:, sl]
        x = x + o @ m[p + "attn.out.weight"].T + m[p + "attn.out.bias"]

        xb = layernorm(x, m[p + "mlp_ln.weight"], m[p + "mlp_ln.bias"])
        h1 = gelu(xb @ m[p + "mlp.0.weight"].T + m[p + "mlp.0.bias"])
        x = x + h1 @ m[p + "mlp.2.weight"].T + m[p + "mlp.2.bias"]

    return layernorm(x, m["encoder.ln.weight"] if "encoder.ln.weight" in m.t
                     else m["encoder.ln_post.weight"],
                     m["encoder.ln_post.bias"]).astype(np.float32)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    tool, path = sys.argv[1], sys.argv[2]
    m = Model(path)

    secs = 1
    n = secs * 16000
    pcm = (np.arange(n) % 1009).astype(np.float32) / np.float32(1009.0)
    pcm = pcm * np.float32(2.0) - np.float32(1.0)
    mel = mel_reference(pcm, m.filters.astype(np.float64))

    r = subprocess.run([tool, path, "--encode", "--seconds", str(secs)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        return 1
    lines = r.stdout.strip().split("\n")
    T, W = (int(v) for v in lines[0].split()[1:])
    got = np.array([float(v) for v in lines[1:]], dtype=np.float32).reshape(T, W)

    want = encoder(m, mel)
    err = np.abs(got - want)
    at = np.unravel_index(err.argmax(), err.shape)
    print(f"encoder {T}x{W}: worst {err.max():.3e} at position {at[0]} dim {at[1]}")
    print(f"  charsiu {got[at]: .5f}   reference {want[at]: .5f}")
    if err.max() > TOL:
        print(f"  FAILED: {int((err > TOL).sum())} of {err.size} above {TOL}")
        print("FAILED")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
