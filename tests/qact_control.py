#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
"""
The quantised activation against the exact f32 one, on the same weights.

Quantising the activation is an APPROXIMATION, so it needs its own control, and
this one needs no reference implementation: the two paths are the same code over
the same file and the only difference is whether the activation went through
int8. What it measures is how much that costs.

    tests/qact_control.py build/charsiu_run models/ [PROMPT]

Reads as a failure if a logit moves further than a threshold that is set from
what the weight quantisation itself already costs, or if the greedy text
diverges within the first few tokens.
"""

import os
import re
import subprocess
import sys

TOPK = 8
NGEN = 48
# One count of a q8_0 weight scale is already worth more than this on a logit
# near 16. A move larger than it means the activation path, not the weights.
TOL = 0.15
MIN_AGREE = 8


def run(cmd, noqact):
    env = dict(os.environ)
    if noqact:
        env["CHARSIU_NO_QACT"] = "1"
    else:
        env.pop("CHARSIU_NO_QACT", None)
    r = subprocess.run(cmd, capture_output=True, env=env)
    return r.stdout.decode("utf-8", "replace")


def logits(exe, model, prompt, noqact):
    out = run([exe, model, "-p", prompt, "--logits", str(TOPK), "-q"], noqact)
    rows = []
    for line in out.splitlines():
        m = re.match(r"\s*\d+\s+(\d+)\s+(-?\d+\.\d+)\s+'", line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2))))
    return rows


def text(exe, model, prompt, noqact):
    out = run([exe, model, "-p", prompt, "-n", str(NGEN), "--ignore-eos",
               "-q", "-c", "512"], noqact)
    return out.split("\n\n[load")[0]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    exe, mdir = sys.argv[1:3]
    prompt = sys.argv[3] if len(sys.argv) > 3 else \
        "The capital of France is Paris. The Eiffel Tower is located in Paris. The"

    bad = 0
    names = sorted(f for f in os.listdir(mdir) if f.endswith(".gguf"))
    for name in names:
        model = os.path.join(mdir, name)
        a = dict(logits(exe, model, prompt, noqact=True))
        b = dict(logits(exe, model, prompt, noqact=False))
        if not a or not b:
            print("FAIL  %-40s  no logits" % name)
            bad += 1
            continue
        dmax = max(abs(a[t] - b[t]) for t in a if t in b)

        ta = text(exe, model, prompt, noqact=True)
        tb = text(exe, model, prompt, noqact=False)
        agree = 0
        for x, y in zip(ta, tb):
            if x != y:
                break
            agree += 1

        # a float weight never takes the integer path, so nothing should move
        floaty = "F16" in name or "F32" in name or "f16" in name or "f32" in name
        want = 0.0 if floaty else TOL

        ok = dmax <= want and (floaty or agree >= MIN_AGREE)
        if not ok:
            bad += 1
        print("%-5s %-40s  dmax %.5f  first divergence at char %d of %d"
              % ("ok" if ok else "FAIL", name, dmax, agree, len(ta)))
        if not ok:
            print("      exact %r" % ta[:110])
            print("      qact  %r" % tb[:110])

    print("\n%d models, %d failures" % (len(names), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
