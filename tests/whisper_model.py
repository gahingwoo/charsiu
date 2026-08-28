#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
"""
Read a whisper.cpp model in numpy, so the references have real weights.

⚠ THIS IS A SECOND READER, ON PURPOSE. Checking charsiu's encoder against a
reference that shares charsiu's own file parsing would pass a misread tensor,
so this parses the container independently, from the format description alone.
"""

import struct

import numpy as np

HP = ["n_vocab", "n_audio_ctx", "n_audio_state", "n_audio_head",
      "n_audio_layer", "n_text_ctx", "n_text_state", "n_text_head",
      "n_text_layer", "n_mels", "ftype"]


class Model:
    def __init__(self, path):
        d = np.memmap(path, dtype=np.uint8, mode="r")
        buf = memoryview(d)
        o = 0
        magic, = struct.unpack_from("<I", buf, o)
        assert magic == 0x67676D6C, f"magic {magic:#x}"
        o += 4
        self.hp = {}
        for k in HP:
            self.hp[k], = struct.unpack_from("<i", buf, o)
            o += 4
        n_mel, n_fft = struct.unpack_from("<ii", buf, o)
        o += 8
        self.filters = np.frombuffer(d, dtype="<f4", count=n_mel * n_fft,
                                     offset=o).reshape(n_mel, n_fft)
        o += 4 * n_mel * n_fft
        nv, = struct.unpack_from("<i", buf, o)
        o += 4
        self.vocab = []
        for _ in range(nv):
            n, = struct.unpack_from("<I", buf, o)
            o += 4
            self.vocab.append(bytes(buf[o:o + n]))
            o += n
        self.t = {}
        end = len(d)
        while o < end:
            nd, nl, ft = struct.unpack_from("<iii", buf, o)
            o += 12
            ne = list(struct.unpack_from("<" + "i" * nd, buf, o))
            o += 4 * nd
            name = bytes(buf[o:o + nl]).decode()
            o += nl
            nel = int(np.prod(ne))
            dt = "<f4" if ft == 0 else "<f2"
            a = np.frombuffer(d, dtype=dt, count=nel, offset=o).astype(np.float32)
            o += nel * (4 if ft == 0 else 2)
            # ne is in ggml order, ne[0] fastest: reverse for numpy
            self.t[name] = a.reshape(list(reversed(ne)))

    def __getitem__(self, k):
        return self.t[k]
