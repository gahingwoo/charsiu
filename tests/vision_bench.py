#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
"""
Write a synthetic mmproj at SmolVLM-256M's REAL shape and time the tower.

WHY THIS EXISTS. tests/vision_cross.py proves the arithmetic on a 64x64 toy
whose attention is 16 against 16; the board's problem is 1024 against 1024,
twelve heads, twelve layers, and a toy cannot see it. The weights here are
random and the answer is meaningless -- this measures TIME ONLY, and the
correctness oracle stays vision_cross.py.

⚠ THE HOST IS COMPUTE BOUND AND THE BOARD IS BANDWIDTH BOUND. A win here is
not automatically a win there. Quote the BYTES a change removes as well as the
seconds, because the bytes are what transfers.

⚠ THE FILE IS CACHED AT --keep. It is a third of a gigabyte of random weights
and building it costs more than the run does; delete it if SHAPE changes.

    tests/vision_bench.py build/charsiu_vision [--reps N] [--keep PATH]
"""

import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmproj_synth as ms                                  # noqa: E402

# SmolVLM-256M's vision tower: 512/16 = 32x32 = 1024 patches, width 768,
# 12 heads of 64, 12 layers, pixel shuffle 4 so 1024 patches leave as 64
# tokens. This is the shape the board's stage table was printed at.
SHAPE = dict(IMAGE=512, PATCH=16, WIDTH=768, FF=3072, HEADS=12, LAYERS=12,
             PROJ=576, SCALE=4)


def build(path):
    for k, v in SHAPE.items():
        setattr(ms, k, v)
    ms.GRID = ms.IMAGE // ms.PATCH
    ms.PATCHES = ms.GRID * ms.GRID

    if os.path.exists(path):
        return path
    rng = np.random.default_rng(20260829)
    data = {}
    for name, ne in ms.tensors("idefics3"):
        n = 1
        for d in ne:
            n *= d
        # small values: a random ViT saturates its own softmax otherwise, and a
        # saturated softmax is a different exp() workload from a real one
        data[name] = (rng.standard_normal(n) * 0.1).astype(np.float32)
    return ms.write(path, data=data, kind="idefics3")


def run(tool, path):
    env = dict(os.environ, CHARSIU_STAGES="1")
    t0 = time.time()
    r = subprocess.run([tool, path, "--encode"], capture_output=True,
                       text=True, env=env)
    wall = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("the tower would not run")
    return wall, r.stderr


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    tool = sys.argv[1]
    reps = 3
    keep = "/tmp/mmproj-bench-smolvlm.gguf"
    for i, a in enumerate(sys.argv):
        if a == "--reps":
            reps = int(sys.argv[i + 1])
        if a == "--keep":
            keep = sys.argv[i + 1]

    path = build(keep)
    best, table = None, ""
    for _ in range(reps):
        wall, err = run(tool, path)
        if best is None or wall < best:
            best, table = wall, err
    print(f"best of {reps}: {best:.2f} s wall")
    print(table.rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
