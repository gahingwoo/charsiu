#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
"""
Diff charsiu's forward pass against llama.cpp on the same gguf.

USE AN F32 MODEL. On a quantised one llama.cpp also quantises the ACTIVATION
before the dot product -- q8_0 weights are multiplied by a q8_0 copy of the
activation vector -- and charsiu does not, so the two disagree by up to 0.14 on
a logit of 16 for reasons that have nothing to do with whether either is right.
In f32 neither side quantises anything and the comparison means something.

    llama-quantize model-F16.gguf model-F32.gguf F32
    tests/forward_cross.py build/charsiu_run REF_LOGITS LLAMA_COMPLETION model-F32.gguf

REF_LOGITS is tests/ref_logits.c built against llama.cpp; see the comment at
the top of that file.
"""

import re
import subprocess
import sys

PROMPTS = [
    "The capital of France is",
    "The capital of France is Paris. The Eiffel Tower is located in Paris. The",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "1, 1, 2, 3, 5, 8, 13,",
    "Q: What is the largest planet in the solar system?\nA:",
    "Once upon a time, in a village at the edge of a forest, there lived",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
    "SELECT name, COUNT(*) FROM users GROUP BY",
    "北京是中国的首都。上海是",
    "Roses are red, violets are",
]

TOPK = 8
NGEN = 48
# f32 summation order alone. Anything above this is not rounding.
LOGIT_TOL = 0.02


def parse_rows(text):
    """rows of '<rank> <id> <logit> '<piece>' ' -> [(id, logit)]"""
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+\s+(\d+)\s+(-?\d+\.\d+)\s+'", line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def run(cmd):
    r = subprocess.run(cmd, capture_output=True)
    return r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        return 2
    charsiu, reflog, completion, model = sys.argv[1:]

    bad = 0
    worst = 0.0
    for p in PROMPTS:
        # ---- logits ----
        ref, _ = run([reflog, model, p, str(TOPK)])
        got, _ = run([charsiu, model, "-p", p, "--logits", str(TOPK), "-q"])
        rr, gg = parse_rows(ref), parse_rows(got)

        label = p.replace("\n", "\\n")[:44]
        if len(rr) != TOPK or len(gg) != TOPK:
            print("FAIL  %-46s  no logits (ref %d, charsiu %d)" % (label, len(rr), len(gg)))
            bad += 1
            continue

        gmap = dict(gg)
        dmax = 0.0
        missing = [t for t, _ in rr if t not in gmap]
        for t, v in rr:
            if t in gmap:
                dmax = max(dmax, abs(v - gmap[t]))
        worst = max(worst, dmax)
        # Order only has to agree where the reference itself separates two
        # tokens by more than the noise floor. Two logits 0.003 apart swapping
        # places is the f32 summation order, not a disagreement about which
        # token comes next, and the greedy text below is what settles that.
        gpos = {t: i for i, (t, _) in enumerate(gg)}
        order_same = True
        for i in range(len(rr) - 1):
            a, av = rr[i]
            b, bv = rr[i + 1]
            if av - bv <= LOGIT_TOL:
                continue
            if a not in gpos or b not in gpos or gpos[a] > gpos[b]:
                order_same = False

        # ---- greedy continuation ----
        rtxt, _ = run([completion, "-m", model, "-p", p, "-n", str(NGEN),
                       "-c", "512", "--temp", "0", "--top-k", "1", "--ignore-eos",
                       "--no-warmup", "-t", "6", "--no-conversation",
                       "--no-display-prompt"])
        gtxt, _ = run([charsiu, model, "-p", p, "-n", str(NGEN), "--ignore-eos", "-q"])
        gtxt = gtxt.split("\n\n[load")[0]
        same_text = rtxt.strip() == gtxt.strip()

        ok = not missing and dmax <= LOGIT_TOL and order_same and same_text
        if not ok:
            bad += 1
        print("%-5s %-46s  dmax %.5f  order %s  %d tok %s"
              % ("ok" if ok else "FAIL", label, dmax,
                 "same" if order_same else "DIFF",
                 NGEN, "same" if same_text else "DIFF"))
        if not same_text:
            print("      ref     %r" % rtxt.strip()[:150])
            print("      charsiu %r" % gtxt.strip()[:150])
        if missing:
            print("      not in charsiu's top %d: %s" % (TOPK, missing))

    print("\n%d prompts, %d failures, worst logit delta %.5f (tolerance %.2f)"
          % (len(PROMPTS), bad, worst, LOGIT_TOL))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
