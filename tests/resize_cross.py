#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
"""
Diff charsiu's image resize against a numpy reference.

⚠ WHY THIS HAS ITS OWN TEST. The resize is the one step whose errors are
invisible in the output: half a pixel of shift produces a caption that is
fluent, confident and about a slightly different crop. Nothing downstream can
catch it, so it is checked here on its own, against the half pixel centre
mapping that torchvision and PIL both use.

The image is a BMP written below -- uncompressed, twenty lines of struct.pack --
so this needs no encoder, no PIL, and no download.

    tests/resize_cross.py build/charsiu_vision
"""

import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

TOL = 1e-6


def write_bmp(path, rgb):
    """rgb: [h][w][3] uint8. BMP rows are bottom up and padded to four bytes."""
    h, w, _ = rgb.shape
    row = (w * 3 + 3) // 4 * 4
    pad = row - w * 3
    body = b""
    for y in range(h - 1, -1, -1):
        for x in range(w):
            r, g, b = rgb[y, x]
            body += bytes((int(b), int(g), int(r)))
        body += b"\x00" * pad
    head = struct.pack("<2sIHHI", b"BM", 14 + 40 + len(body), 0, 0, 14 + 40)
    head += struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(body),
                        2835, 2835, 0, 0)
    open(path, "wb").write(head + body)
    return path


def reference(rgb, side):
    """Bilinear on the half pixel centres, which is the mapping
       src = (dst + 0.5) * scale - 0.5 and NOT (dst * (n-1)) / (m-1)."""
    h, w, _ = rgb.shape
    src = rgb.astype(np.float32)
    out = np.zeros((3, side, side), dtype=np.float32)
    fy = (np.arange(side, dtype=np.float32) + 0.5) * h / side - 0.5
    fx = (np.arange(side, dtype=np.float32) + 0.5) * w / side - 0.5
    y0 = np.clip(np.floor(fy).astype(int), 0, h - 1)
    y1 = np.clip(np.floor(fy).astype(int) + 1, 0, h - 1)
    x0 = np.clip(np.floor(fx).astype(int), 0, w - 1)
    x1 = np.clip(np.floor(fx).astype(int) + 1, 0, w - 1)
    wy = (fy - np.floor(fy))[:, None]
    wx = (fx - np.floor(fx))[None, :]
    for c in range(3):
        p = src[:, :, c]
        a = (1 - wx) * p[np.ix_(y0, x0)] + wx * p[np.ix_(y0, x1)]
        b = (1 - wx) * p[np.ix_(y1, x0)] + wx * p[np.ix_(y1, x1)]
        out[c] = ((1 - wy) * a + wy * b) / 255.0
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    tool = sys.argv[1]
    rng = np.random.default_rng(20260828)
    fail = 0
    tmp = tempfile.mkdtemp()

    # a square, a wide one and a tall one: the stretch is the interesting case
    for (w, h, side) in ((16, 16, 8), (37, 11, 24), (9, 40, 16), (5, 5, 32)):
        rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        path = write_bmp(os.path.join(tmp, f"t{w}x{h}.bmp"), rgb)
        r = subprocess.run([tool, "--resize", path, str(side)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAILED {w}x{h}->{side}: {r.stderr.strip()}")
            fail = 1
            continue
        lines = r.stdout.strip().split("\n")
        got = np.array([float(v) for v in lines[1:]], dtype=np.float32)
        got = got.reshape(3, side, side)
        want = reference(rgb, side)
        worst = np.abs(got - want).max()
        print(f"{w}x{h} -> {side}x{side}: worst {worst:.3e}")
        if worst > TOL:
            fail = 1
            print("  FAILED")
    print("FAILED" if fail else "PASS")
    return fail


if __name__ == "__main__":
    sys.exit(main())
