#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
"""
Write a synthetic mmproj gguf, and check charsiu_vision reads it.

WHY SYNTHETIC. Every vision tensor name in this tree is llama.cpp's clip naming
as we understand it, checked against nothing -- and a real mmproj is a several
hundred megabyte download that a person at a desk without one cannot conjure.
This writes a tiny tower with the same NAMES and SHAPES and no download, so the
reader, the shapes and the reporting are all testable today. It does NOT make
the names right: only a real file can do that, and `charsiu_vision` printing
the misses by name is what turns that into a five minute fix.

    tests/mmproj_synth.py build/charsiu_vision

With no argument it just writes /tmp/mmproj-synth.gguf and stops.
"""

import struct
import subprocess
import sys
import tempfile
import os

U32, F32, STR, ARR = 4, 6, 8, 9
GGML_F32 = 0


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def kv_u32(k, v):
    return s(k) + struct.pack("<II", U32, v)


def kv_f32(k, v):
    return s(k) + struct.pack("<I", F32) + struct.pack("<f", v)


def kv_bool(k, v):
    return s(k) + struct.pack("<IB", 7, 1 if v else 0)


def kv_arr_f32(k, vals):
    out = s(k) + struct.pack("<I", ARR) + struct.pack("<IQ", F32, len(vals))
    for v in vals:
        out += struct.pack("<f", v)
    return out


def kv_arr_str(k, vals):
    out = s(k) + struct.pack("<I", ARR) + struct.pack("<IQ", STR, len(vals))
    for v in vals:
        out += s(v)
    return out


def kv_str(k, v):
    return s(k) + struct.pack("<I", STR) + s(v)


# a tower small enough to be instant and shaped like a real one
IMAGE, PATCH, WIDTH, FF, HEADS, LAYERS, PROJ = 64, 16, 32, 64, 4, 2, 48
GRID = IMAGE // PATCH
PATCHES = GRID * GRID
SCALE = 2                      # idefics3's pixel shuffle factor


TW, TFF, THEADS, TLAYERS, TCTX, TVOCAB = 24, 48, 4, 2, 16, 40


def text_tensors():
    t = [("t.token_embd.weight", [TW, TVOCAB]),
         ("t.position_embd.weight", [TW, TCTX]),
         ("t.post_ln.weight", [TW]), ("t.post_ln.bias", [TW]),
         ("text_projection.weight", [TW, PROJ])]
    for i in range(TLAYERS):
        p = f"t.blk.{i}."
        t += [
            (p + "ln1.weight", [TW]), (p + "ln1.bias", [TW]),
            (p + "attn_q.weight", [TW, TW]), (p + "attn_q.bias", [TW]),
            (p + "attn_k.weight", [TW, TW]), (p + "attn_k.bias", [TW]),
            (p + "attn_v.weight", [TW, TW]), (p + "attn_v.bias", [TW]),
            (p + "attn_out.weight", [TW, TW]), (p + "attn_out.bias", [TW]),
            (p + "ln2.weight", [TW]), (p + "ln2.bias", [TW]),
            (p + "ffn_down.weight", [TW, TFF]), (p + "ffn_down.bias", [TFF]),
            (p + "ffn_up.weight", [TFF, TW]), (p + "ffn_up.bias", [TW]),
        ]
    return t


def tensors(kind="mlp"):
    """(name, ne) with ne in gguf order: ne[0] is the fastest axis.

    ⚠ THE FEED FORWARD NAMES ARE THE REAL ONES, WHICH ARE BACKWARDS. A real
    mmproj calls the FIRST matmul ffn_down (n_embd -> n_ff) and the second
    ffn_up, the opposite way round from the language model. Writing them the
    intuitive way here would have made the synthetic file the only one the
    loader could read."""
    if kind == "idefics3":
        proj = [("mm.model.fc.weight", [WIDTH * SCALE * SCALE, PROJ])]
    elif kind == "clip":
        proj = ([("visual_projection.weight", [WIDTH, PROJ]),
                 ("v.class_embd", [WIDTH])] + text_tensors())
    else:
        proj = [("mm.0.weight", [WIDTH, PROJ]), ("mm.0.bias", [PROJ])]
    t = proj + [
        # a patch embedding is [out][in] with in = channels * patch * patch
        ("v.patch_embd.weight", [3 * PATCH * PATCH, WIDTH]),
        ("v.patch_embd.bias", [WIDTH]),
        ("v.position_embd.weight", [WIDTH, PATCHES + (1 if kind == "clip" else 0)]),
        ("v.post_ln.weight", [WIDTH]),
        ("v.post_ln.bias", [WIDTH]),
    ]
    for i in range(LAYERS):
        p = f"v.blk.{i}."
        t += [
            (p + "ln1.weight", [WIDTH]), (p + "ln1.bias", [WIDTH]),
            (p + "attn_q.weight", [WIDTH, WIDTH]), (p + "attn_q.bias", [WIDTH]),
            (p + "attn_k.weight", [WIDTH, WIDTH]), (p + "attn_k.bias", [WIDTH]),
            (p + "attn_v.weight", [WIDTH, WIDTH]), (p + "attn_v.bias", [WIDTH]),
            (p + "attn_out.weight", [WIDTH, WIDTH]),
            (p + "attn_out.bias", [WIDTH]),
            (p + "ln2.weight", [WIDTH]), (p + "ln2.bias", [WIDTH]),
            (p + "ffn_down.weight", [WIDTH, FF]), (p + "ffn_down.bias", [FF]),
            (p + "ffn_up.weight", [FF, WIDTH]), (p + "ffn_up.bias", [WIDTH]),
        ]
    return t


def write(path, drop=(), data=None, kind="mlp"):
    """drop: names to leave OUT, which is how the reporting gets tested.
    data: {name: flat float32 array}. Absent names are written as zeros, which
    is enough to test the reader and useless for testing the arithmetic."""
    ts = [t for t in tensors(kind) if t[0] not in drop]
    kvs = b"".join([
        kv_str("general.architecture", "clip"),
        kv_bool("clip.has_vision_encoder", True),
        kv_u32("clip.vision.image_size", IMAGE),
        kv_u32("clip.vision.patch_size", PATCH),
        kv_u32("clip.vision.embedding_length", WIDTH),
        kv_u32("clip.vision.feed_forward_length", FF),
        kv_u32("clip.vision.block_count", LAYERS),
        kv_u32("clip.vision.attention.head_count", HEADS),
        kv_u32("clip.vision.projection_dim", PROJ),
        kv_f32("clip.vision.attention.layer_norm_epsilon", 1e-6),
        kv_arr_f32("clip.vision.image_mean", [0.5, 0.5, 0.5]),
        kv_arr_f32("clip.vision.image_std", [0.5, 0.5, 0.5]),
        kv_u32("clip.use_gelu", 1),
    ] + ([kv_str("clip.projector_type", "idefics3"),
          kv_u32("clip.vision.projector.scale_factor", SCALE)]
         if kind == "idefics3" else []) + (
        [kv_bool("clip.has_text_encoder", True),
         kv_u32("clip.text.embedding_length", TW),
         kv_u32("clip.text.feed_forward_length", TFF),
         kv_u32("clip.text.block_count", TLAYERS),
         kv_u32("clip.text.attention.head_count", THEADS),
         kv_u32("clip.text.context_length", TCTX),
         kv_u32("clip.text.projection_dim", PROJ),
         kv_f32("clip.text.attention.layer_norm_epsilon", 1e-6),
         kv_arr_str("tokenizer.ggml.tokens",
                    [f"t{i}" for i in range(TVOCAB - 2)] +
                    ["<|startoftext|>", "<|endoftext|>"])]
        if kind == "clip" else []))
    n_kv = 13 + (2 if kind == "idefics3" else 0) + (9 if kind == "clip" else 0)

    infos, off, blobs = b"", 0, []
    for name, ne in ts:
        infos += s(name) + struct.pack("<I", len(ne))
        for d in ne:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<IQ", GGML_F32, off)
        n = 1
        for d in ne:
            n *= d
        if data is not None and name in data:
            a = data[name]
            assert a.size == n, f"{name}: {a.size} values for {n} slots"
            blobs.append(a.astype("<f4").tobytes())
        else:
            blobs.append(b"\x00" * (4 * n))
        off += 4 * n
        off = (off + 31) // 32 * 32

    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(ts), n_kv)
    head += kvs + infos
    pad = (32 - len(head) % 32) % 32
    body = b""
    for b in blobs:
        body += b
        body += b"\x00" * ((32 - len(body) % 32) % 32)
    open(path, "wb").write(head + b"\x00" * pad + body)
    return path


def main():
    tmp = tempfile.mkdtemp()
    good = write(os.path.join(tmp, "mmproj-synth.gguf"))
    print("wrote", good)
    if len(sys.argv) < 2:
        return 0
    tool = sys.argv[1]
    fail = 0

    r = subprocess.run([tool, good], capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        print("FAIL: a complete tower was rejected")
        fail = 1
    elif "complete" not in r.stdout:
        print("FAIL: a complete tower did not say so")
        fail = 1

    # and the whole point: a missing tensor has to be NAMED, not skipped
    bad = write(os.path.join(tmp, "mmproj-missing.gguf"),
                drop=("v.blk.1.attn_k.weight",))
    r = subprocess.run([tool, bad], capture_output=True, text=True)
    if r.returncode == 0:
        print("FAIL: a tower missing attn_k was accepted")
        fail = 1
    elif "v.blk.1.attn_k.weight" not in r.stdout:
        print("FAIL: the missing tensor was not named")
        print(r.stdout)
        fail = 1
    else:
        print("ok: a missing tensor is reported by name")

    print("FAILED" if fail else "PASS")
    return fail


if __name__ == "__main__":
    sys.exit(main())
