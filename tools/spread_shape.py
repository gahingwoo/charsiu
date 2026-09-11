#!/usr/bin/env python3
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Is a set of timings TWO CLUSTERS or ONE TAIL? Decided by a rule fixed first.

⚠⚠ THIS EXISTS BECAUSE THE ALTERNATIVE IS INVENTING A STORY. gemma4's TTFT
read 2133, 2182, 2185, 2325, 2408, 2707 and 3221 in one build, and "that looks
like two clusters rather than a tail" is a sentence written by a person looking
at seven numbers. This tree has already published a shape property for two
cells that were simply unstable. So the test is written BEFORE the data and its
decision rule is printed with its answer.

WHAT IT DECIDES, and none of it is a hypothesis test with a p-value, because
twenty readings cannot support one:

  1. BIMODALITY   the largest gap between consecutive sorted readings, against
                  the median gap. A tail's biggest gap is at its end and grows
                  smoothly; two clusters put an outsized gap in the MIDDLE.
                  Reported as (largest gap / median gap) and WHERE it falls.
  2. DRIFT        Spearman correlation of reading against its index. A board
                  that warms produces a monotone trend, which is neither a
                  cluster nor a tail -- it is the round being run wrong.
  3. TEMPERATURE  the same against the temperature stamped on each reading.
  4. RECURRENCE   if it is two clusters, membership should not be random in
                  time: runs of the same cluster. Counted against what
                  independent coin flips would give.

THE RULE, fixed here:

  two clusters   gap ratio >= 3 AND the gap sits between the 25th and 75th
                 percentile of the readings AND |rho_index| < 0.5
                 (a middle gap that is not just drift)
  drift          |rho_index| >= 0.5 or |rho_temp| >= 0.5
  one tail       anything else -- and then the row is NOT quotable, which is
                 a result and gets written down as one

Usage:
    python3 -P tools/spread_shape.py FILE   # lines of "value [temp]"
    ... | python3 -P tools/spread_shape.py -
"""
import sys


def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
            j += 1
        for k in range(i, j + 1):
            r[s[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return 0.0
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def pct(sv, q):
    if not sv:
        return 0.0
    i = q * (len(sv) - 1)
    lo = int(i)
    hi = min(lo + 1, len(sv) - 1)
    return sv[lo] + (i - lo) * (sv[hi] - sv[lo])


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "-"
    fh = sys.stdin if src == "-" else open(src)
    vals, temps = [], []
    for line in fh:
        p = line.replace(",", " ").split()
        if not p:
            continue
        try:
            v = float(p[0])
        except ValueError:
            continue
        vals.append(v)
        temps.append(float(p[1]) if len(p) > 1 and p[1].replace(".", "", 1)
                     .replace("-", "", 1).isdigit() else None)
    n = len(vals)
    if n < 5:
        print("fewer than 5 readings -- nothing to decide")
        return 2

    sv = sorted(vals)
    gaps = [sv[i + 1] - sv[i] for i in range(n - 1)]
    gs = sorted(gaps)
    medgap = gs[len(gs) // 2] if len(gs) % 2 else (gs[len(gs) // 2 - 1] + gs[len(gs) // 2]) / 2
    gmax = max(gaps)
    gi = gaps.index(gmax)
    ratio = gmax / medgap if medgap > 0 else float("inf")
    split = (sv[gi] + sv[gi + 1]) / 2.0
    q25, q75 = pct(sv, 0.25), pct(sv, 0.75)
    middle = q25 <= split <= q75

    rho_i = spearman(vals, list(range(n)))
    #
    # ⚠⚠ A PERFECTLY MONOTONE INPUT IS ALMOST CERTAINLY A SORTED QUOTE, NOT A
    # TIME SERIES, and the drift test then reads +1.000 by construction. The
    # seven gemma4 readings this tool was written for are quoted ascending
    # everywhere they appear -- feeding them in gives DRIFT, which is a
    # statement about the person who typed them in order.
    #
    presorted = vals == sorted(vals) or vals == sorted(vals, reverse=True)
    have_t = [(v, t) for v, t in zip(vals, temps) if t is not None]
    rho_t = spearman([v for v, _ in have_t], [t for _, t in have_t]) if len(have_t) >= 3 else None

    lo = [v for v in vals if v < split]
    hi = [v for v in vals if v >= split]
    side = [0 if v < split else 1 for v in vals]
    runs = 1 + sum(1 for i in range(1, n) if side[i] != side[i - 1])
    exp_runs = 1 + 2 * len(lo) * len(hi) / n if n else 0

    print("  n            %d" % n)
    print("  median       %.1f      range %.1f .. %.1f" % (pct(sv, 0.5), sv[0], sv[-1]))
    print("  spread       %.1f%% of the median" % ((sv[-1] - sv[0]) / pct(sv, 0.5) * 100))
    print("  largest gap  %.1f at %.1f   (median gap %.1f)  ratio %.2f" %
          (gmax, split, medgap, ratio))
    print("  gap sits     %s  (q25 %.1f, q75 %.1f)" %
          ("IN THE MIDDLE" if middle else "at an END", q25, q75))
    print("  rho index    %+.3f" % rho_i)
    print("  rho temp     %s" % ("%+.3f" % rho_t if rho_t is not None else "no temperatures"))
    print("  split gives  %d low / %d high, %d runs (independent would give %.1f)" %
          (len(lo), len(hi), runs, exp_runs))
    print()

    if presorted and n >= 6:
        print("  ⛔ THE INPUT IS PERFECTLY SORTED, so it is a quoted list and")
        print("     not a time series. rho index is +-1 by construction and")
        print("     means nothing here. Re-run with the readings IN THE ORDER")
        print("     THEY WERE TAKEN -- board_record.sh stamps each with its")
        print("     time for exactly this reason. Only the gap test below is")
        print("     readable from sorted data.")
        print()

    drift = (not presorted) and (
        abs(rho_i) >= 0.5 or (rho_t is not None and abs(rho_t) >= 0.5))
    two = ratio >= 3 and middle and (presorted or abs(rho_i) < 0.5)
    if two:
        print("  VERDICT: TWO CLUSTERS. Something is switching, and that is a")
        print("  finding rather than a caveat. Next: what differs between the")
        print("  %d low and the %d high readings that is not time or heat." % (len(lo), len(hi)))
    elif drift:
        print("  VERDICT: DRIFT. The readings track their own order or the")
        print("  temperature, so this is the round being run wrong, not a")
        print("  property of the model. Warm up first and alternate the arms.")
    else:
        print("  VERDICT: ONE TAIL, no structure. The row is NOT quotable and")
        print("  the paper says so. A median with this range beside it is the")
        print("  most that can honestly be reported.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
