#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
"""
Diff charsiu's whisper decoder against numpy, on the real weights.

⚠ CROSS ATTENTION IS WHAT THIS IS FOR. It is the one mechanism in this tree that
reads its keys and values from a different sequence than its queries, and every
way of getting it wrong -- the wrong tensor, the cache built from the wrong
side, the encoder's positions transposed -- leaves a decoder that still produces
fluent English. jfk.wav coming out right is evidence; a logit vector matching to
1e-3 is a measurement.

The prompt is the two markers an English-only model starts with, so what is
compared is the FIRST PREDICTION: everything in the decoder has run once.

    tests/whisper_decoder_cross.py build/charsiu_whisper MODEL.bin [AUDIO.wav]
"""

import subprocess
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from whisper_model import Model                                   # noqa: E402
from mel_cross import reference as mel_reference                  # noqa: E402
from whisper_encoder_cross import encoder, gelu, layernorm        # noqa: E402

TOL = 3e-3


def decoder_logits(m, enc, prompt):
    W = m.hp["n_text_state"]
    H = m.hp["n_text_head"]
    hd = W // H
    n = len(prompt)

    x = m["decoder.token_embedding.weight"][list(prompt)].astype(np.float64)
    x = x + m["decoder.positional_embedding"][:n]
    mask = np.triu(np.full((n, n), -np.inf), 1)

    for i in range(m.hp["n_text_layer"]):
        p = f"decoder.blocks.{i}."
        xb = layernorm(x, m[p + "attn_ln.weight"], m[p + "attn_ln.bias"])
        q = xb @ m[p + "attn.query.weight"].T + m[p + "attn.query.bias"]
        k = xb @ m[p + "attn.key.weight"].T
        v = xb @ m[p + "attn.value.weight"].T + m[p + "attn.value.bias"]
        o = np.zeros_like(x)
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            s = q[:, sl] @ k[:, sl].T / np.sqrt(hd) + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            o[:, sl] = (s / s.sum(-1, keepdims=True)) @ v[:, sl]
        x = x + o @ m[p + "attn.out.weight"].T + m[p + "attn.out.bias"]

        # ⚠ the keys and values come from the ENCODER, the queries from here
        xb = layernorm(x, m[p + "cross_attn_ln.weight"],
                       m[p + "cross_attn_ln.bias"])
        q = xb @ m[p + "cross_attn.query.weight"].T + m[p + "cross_attn.query.bias"]
        k = enc @ m[p + "cross_attn.key.weight"].T
        v = enc @ m[p + "cross_attn.value.weight"].T + m[p + "cross_attn.value.bias"]
        o = np.zeros_like(x)
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            s = q[:, sl] @ k[:, sl].T / np.sqrt(hd)
            s = np.exp(s - s.max(-1, keepdims=True))
            o[:, sl] = (s / s.sum(-1, keepdims=True)) @ v[:, sl]
        x = x + o @ m[p + "cross_attn.out.weight"].T + m[p + "cross_attn.out.bias"]

        xb = layernorm(x, m[p + "mlp_ln.weight"], m[p + "mlp_ln.bias"])
        h1 = gelu(xb @ m[p + "mlp.0.weight"].T + m[p + "mlp.0.bias"])
        x = x + h1 @ m[p + "mlp.2.weight"].T + m[p + "mlp.2.bias"]

    x = layernorm(x, m["decoder.ln.weight"], m["decoder.ln.bias"])
    # ⚠ the head is the embedding table, tied
    return (x[-1] @ m["decoder.token_embedding.weight"].T).astype(np.float32)


def load_wav(path):
    import wave

    with wave.open(path) as f:
        assert f.getsampwidth() == 2, "16 bit only"
        ch, rate = f.getnchannels(), f.getframerate()
        raw = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    x = raw.reshape(-1, ch).mean(1).astype(np.float32) / np.float32(32768.0)
    if rate != 16000:
        n = int(len(x) * 16000 / rate)
        x = np.interp(np.arange(n) * rate / 16000.0,
                      np.arange(len(x)), x).astype(np.float32)
    return x


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    tool, path = sys.argv[1], sys.argv[2]
    audio = sys.argv[3] if len(sys.argv) > 3 else None
    m = Model(path)

    cmd = [tool, path, "--logits"]
    if audio:
        pcm = load_wav(audio)
        cmd += ["--audio", audio]
    else:
        n = 16000
        pcm = (np.arange(n) % 1009).astype(np.float32) / np.float32(1009.0)
        pcm = pcm * np.float32(2.0) - np.float32(1.0)
        cmd += ["--seconds", "1"]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        return 1
    lines = r.stdout.strip().split("\n")
    got = np.array([float(v) for v in lines[1:]], dtype=np.float32)

    mel = mel_reference(pcm, m.filters.astype(np.float64))
    enc = encoder(m, mel).astype(np.float64)
    sot = 50257 + (m.hp["n_vocab"] == 51865)
    nots = 50362 + (m.hp["n_vocab"] == 51865)
    want = decoder_logits(m, enc, [sot, nots])

    err = np.abs(got - want)
    print(f"logits {got.size}: worst {err.max():.3e} at id {err.argmax()}")
    print(f"  argmax: charsiu {got.argmax()} ({m.vocab[got.argmax()]!r})"
          f"   reference {want.argmax()} ({m.vocab[want.argmax()]!r})")
    if got.argmax() != want.argmax():
        print("FAILED: the first token differs")
        return 1
    if err.max() > TOL:
        print(f"FAILED: {int((err > TOL).sum())} of {err.size} above {TOL}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
