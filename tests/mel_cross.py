#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
"""
Diff charsiu's log mel spectrogram against a numpy reference.

⚠ WHY THIS IS ITS OWN TEST. Everything about the front end fails quietly. A
symmetric window instead of a periodic one, centre padding that reflects the
wrong way, a clamp taken per frame instead of over the clip: none of them
crashes, none of them makes a number infinite, and every one of them produces a
transcript that is fluent and wrong. The encoder cannot tell you either, because
it will happily encode a wrong spectrogram.

The filterbank comes out of the model file on both sides, so this tests the
transform and not the filters.

    tests/mel_cross.py build/charsiu_whisper MODEL.bin
"""

import struct
import subprocess
import sys

import numpy as np

N_FFT, HOP, N_FRAMES = 400, 160, 3000
TOL = 2e-4


def read_filters(path):
    d = open(path, "rb").read(4 + 11 * 4 + 8 + 80 * 201 * 4)
    o = 4 + 11 * 4
    n_mel, n_fft = struct.unpack_from("<ii", d, o)
    o += 8
    filt = np.frombuffer(d, dtype="<f4", count=n_mel * n_fft, offset=o)
    return filt.reshape(n_mel, n_fft).astype(np.float64)


def reference(pcm, filt, dtype=np.float64):
    """dtype picks the arithmetic. charsiu computes the FFT in f32, so an f64
    reference and an f32 one bracket what rounding alone can produce."""
    n_mel, bins = filt.shape
    filt = filt.astype(dtype)
    # ⚠ PERIODIC: 1 - cos(2 pi i / N), not numpy.hanning's N - 1
    win = (0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT))
           ).astype(dtype)
    # ⚠ REFLECTED AT THE FRONT, SILENCE AT THE BACK. whisper.cpp mirrors 200
    # samples about sample 0 and then zero fills a whole 30 s window. Padding
    # both ends by reflection -- torch.stft(center=True), np.pad("reflect") --
    # fills the empty part of the clip with a repeat of the speech.
    pad = N_FFT // 2
    x = np.concatenate([pcm.astype(dtype)[pad:0:-1],
                        pcm.astype(dtype),
                        np.zeros(pad + N_FRAMES * HOP, dtype=dtype)])
    out = np.zeros((n_mel, N_FRAMES), dtype=dtype)
    for f in range(N_FRAMES):
        seg = x[f * HOP:f * HOP + N_FFT] * win
        spec = np.fft.rfft(seg, n=N_FFT)[:bins]
        power = (spec.real ** 2 + spec.imag ** 2)
        out[:, f] = np.log10(np.maximum(filt @ power, 1e-10))
    # ⚠ the clamp is over the WHOLE spectrogram
    out = np.maximum(out, out.max() - 8.0)
    return ((out + 4.0) / 4.0).astype(np.float32)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    tool, model = sys.argv[1], sys.argv[2]
    filt = read_filters(model)
    fail = 0

    for secs in (1, 3):
        n = secs * 16000
        pcm = (np.arange(n) % 1009).astype(np.float32) / np.float32(1009.0)
        pcm = pcm * np.float32(2.0) - np.float32(1.0)

        r = subprocess.run([tool, model, "--mel", "--seconds", str(secs)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr)
            return 1
        lines = r.stdout.strip().split("\n")
        n_mel, nf = (int(v) for v in lines[0].split()[1:])
        got = np.array([float(v) for v in lines[1:]], dtype=np.float32)
        got = got.reshape(n_mel, nf)

        want = reference(pcm, filt)
        want32 = reference(pcm, filt, np.float32).astype(np.float64)
        err = np.abs(got - want)
        # ⚠ THE NOISE FLOOR IS MEASURED, NOT ASSUMED. charsiu's FFT is f32 and
        # this reference is f64, so the same computation in f32 says what
        # rounding alone can produce. A tolerance picked to make a test pass is
        # not a tolerance.
        floor = np.abs(want32 - want).max()
        at = np.unravel_index(err.argmax(), err.shape)
        print(f"{secs}s: {n_mel}x{nf}, worst {err.max():.3e} "
              f"at band {at[0]} frame {at[1]}; f32 rounding alone gives "
              f"{floor:.3e}")
        if err.max() > max(TOL, 2.0 * floor):
            print(f"  FAILED: worst is more than twice what rounding explains")
            fail = 1

    print("FAILED" if fail else "PASS")
    return fail


if __name__ == "__main__":
    sys.exit(main())
