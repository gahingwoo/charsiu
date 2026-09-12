#!/usr/bin/env python3
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
"""Compare two TTFT-against-length curves without fitting a line to two points.

⚠⚠ WHY THIS EXISTS. The prefill gap against the vendor was quoted as "1.23x
per prompt token" for weeks. That came from timing two prompt lengths a side
and reading the slope. r392 measured eight lengths and killed it: the curve is
convex, it is not even monotone in its marginal because prompt length decides
the chunking, and the two sides' secants were taken over different token
ranges. Two points always fit a line.

So this compares CURVES, and it answers the two different questions separately
rather than averaging them into one number:

  MATCHED TOKENS   how fast is each runtime per prompt token. Their tokeniser
                   wraps the prompt in a chat template, so the same text is
                   more tokens to them; comparing at equal token counts asks
                   what the runtime does with a token.
  MATCHED TEXT     what a user waits for. The same string, whatever each side
                   decides that string is worth in tokens.

Both are true and they do not agree. Reporting one without the other is how
the 1.76x-in-both-directions reading happened in the first place.

⚠ The interpolation is quadratic in n and local -- three nearest points -- so
one chunking step does not bend the whole curve. Extrapolation is refused: a
target outside a curve's measured range prints nothing rather than a number.

  python3 -P tools/ttft_compare.py A.txt B.txt
where each file has lines "<tokens> <ttft_ms>" and A is charsiu.
"""
import sys


def read(path):
    pts = []
    for line in open(path):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        f = line.split()
        if len(f) >= 2:
            try:
                pts.append((float(f[0]), float(f[1])))
            except ValueError:
                pass
    return sorted(pts)


def at(pts, x):
    """quadratic through the three nearest points; None outside the range"""
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return None
    near = sorted(pts, key=lambda p: abs(p[0] - x))[:3]
    if len(near) < 3:
        (x0, y0), (x1, y1) = near[0], near[1]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    (a, fa), (b, fb), (c, fc) = sorted(near)
    return (fa * (x - b) * (x - c) / ((a - b) * (a - c)) +
            fb * (x - a) * (x - c) / ((b - a) * (b - c)) +
            fc * (x - a) * (x - b) / ((c - a) * (c - b)))


def self_test():
    """⚠ THIS TOOL IS ABOUT TO PRODUCE A HEADLINE NUMBER, so the interpolation
    it rests on is checked against functions whose answer is known."""
    ok = True

    # exact on a quadratic, which is what a prefill curve mostly is
    quad = [(n, 100 + 5.0 * n + 0.004 * n * n) for n in (50, 100, 200, 400, 800)]
    for n in (75, 150, 300, 600):
        want = 100 + 5.0 * n + 0.004 * n * n
        got = at(quad, n)
        bad = abs(got - want) / want > 1e-9
        print("  quadratic at %-4d  %10.3f want %10.3f   %s"
              % (n, got, want, "WRONG" if bad else "exact"))
        ok &= not bad

    # refuses to extrapolate, in both directions
    for n in (10, 5000):
        got = at(quad, n)
        print("  outside the range at %-5d  %s" % (n, "refused" if got is None else "RETURNED %s" % got))
        ok &= got is None

    # a straight line is recovered too, and a single interval does not crash
    line = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)]
    got = at(line, 7)
    print("  line at 7          %10.3f want %10.3f   %s"
          % (got, 7.0, "WRONG" if abs(got - 7) > 1e-9 else "exact"))
    ok &= abs(got - 7) < 1e-9

    print("self-test:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        sys.exit(self_test())
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    A, B = read(sys.argv[1]), read(sys.argv[2])
    na, nb = sys.argv[1], sys.argv[2]
    if not A or not B:
        sys.exit("one of the curves is empty")

    print("== matched TOKEN COUNT -- what a runtime does with a token")
    print("   %8s %12s %12s   %s" % ("tokens", "charsiu ms", "vendor ms", "ratio"))
    lo = max(A[0][0], B[0][0])
    hi = min(A[-1][0], B[-1][0])
    n = lo
    step = 50
    shown = 0
    while n <= hi:
        ya, yb = at(A, n), at(B, n)
        if ya and yb:
            print("   %8d %12.1f %12.1f   %s %.3fx" %
                  (n, ya, yb, "vendor" if yb < ya else "charsiu",
                   max(ya, yb) / min(ya, yb)))
            shown += 1
        n += step
        if shown >= 18:
            break
    if not shown:
        print("   the two curves do not overlap in token count")
    print("   (overlap %d..%d tokens)" % (lo, hi))

    print()
    print("== matched INPUT TEXT -- what a user waits for")
    print("   the rows are the same prompts in order, so the token")
    print("   counts differ by their chat template")
    print("   %10s %10s %10s %10s   %s" %
          ("ch tok", "ch ms", "vn tok", "vn ms", "ratio"))
    for (xa, ya), (xb, yb) in zip(A, B):
        print("   %10d %10.1f %10d %10.1f   %s %.3fx" %
              (xa, ya, xb, yb, "vendor" if yb < ya else "charsiu",
               max(ya, yb) / min(ya, yb)))

    print()
    print("== local marginal ms/token, from the same interpolation")
    print("   %8s %12s %12s   %s" % ("tokens", "charsiu", "vendor", "ratio"))
    n = lo + 25
    while n <= hi - 25:
        da = at(A, n + 25), at(A, n - 25)
        db = at(B, n + 25), at(B, n - 25)
        if all(v is not None for v in da + db):
            ma, mb = (da[0] - da[1]) / 50, (db[0] - db[1]) / 50
            if ma > 0 and mb > 0:
                print("   %8d %12.3f %12.3f   %s %.3fx" %
                      (n, ma, mb, "vendor" if mb < ma else "charsiu",
                       max(ma, mb) / min(ma, mb)))
        n += 100


main()
