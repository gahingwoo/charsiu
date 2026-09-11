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

  too tight      total spread < 5% of the median -- the shape question is
                 not asked at all (see below)
  two clusters   gap ratio >= 3 AND at least 2 readings on EACH side of the
                 gap AND |rho_index| < 0.5
  drift          |rho_index| >= 0.5 or |rho_temp| >= 0.5
  no structure   anything else

⚠⚠ AND QUOTABILITY IS A SEPARATE QUESTION FROM SHAPE. The first version said
"no structure -> the row is NOT quotable", which is written for a 46% spread
and is exactly backwards for a 1.6% one: a tight unimodal arm is the most
quotable thing there is. Shape and spread are reported separately now, and the
quotable line keys off the SPREAD.

⚠⚠ AND A GAP RATIO IS SCALE-FREE, so on a very tight sample the readings'
own last digit manufactures structure: the big-cluster arm spans 1.6% and its
"largest gap" is 0.1 tok/s, which the ratio test happily called two clusters.
Below a 5% spread the shape question is not asked -- at that width two
clusters cannot be told from rounding, and saying so is the honest answer.

⚠ The middle-gap test was also too strict. It required the split to sit
between the quartiles, so 7 low and 2 high -- a gap ratio of 49 -- came back
"at an END" because both quartiles were inside the low cluster. Two clusters
of very different sizes are still two clusters; what a single outlier must not
do is get called one, hence 2 readings a side.

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
    # ⚠ 2 A SIDE, NOT "between the quartiles": a lopsided split is still a
    # split, and gi is the index of the gap in the sorted list, so the sides
    # are gi+1 below and n-gi-1 above.
    middle = (gi + 1) >= 2 and (n - gi - 1) >= 2

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
    print("  gap splits   %d below / %d above  %s  (q25 %.1f, q75 %.1f)" %
          (gi + 1, n - gi - 1,
           "-- a real split" if middle else "-- ONE OUTLIER, not a split",
           q25, q75))
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

    spread = (sv[-1] - sv[0]) / pct(sv, 0.5) * 100
    #
    # ⚠⚠ THE WIDTH GATE. A ratio has no units, so a sample spanning 1.6% has
    # a "largest gap" made of its own last digit. Do not ask the shape
    # question of something too tight to hold an answer.
    #
    tight = spread < 5.0
    drift = (not presorted) and (not tight) and (
        abs(rho_i) >= 0.5 or (rho_t is not None and abs(rho_t) >= 0.5))
    two = (not tight) and ratio >= 3 and middle and (
        presorted or abs(rho_i) < 0.5)
    if two:
        print("  VERDICT: TWO CLUSTERS. Something is switching, and that is a")
        print("  finding rather than a caveat. Next: what differs between the")
        print("  %d low and the %d high readings that is not time or heat." % (len(lo), len(hi)))
    elif drift:
        print("  VERDICT: DRIFT. The readings track their own order or the")
        print("  temperature, so this is the round being run wrong, not a")
        print("  property of the model. Warm up first and alternate the arms.")
    elif tight:
        print("  VERDICT: TOO TIGHT TO HAVE A SHAPE -- %.1f%% spread. Two"
              % spread)
        print("  clusters cannot be told from rounding at this width, so the")
        print("  question is not asked. This is the good case.")
    else:
        print("  VERDICT: NO STRUCTURE -- one population, not two, no trend.")
    #
    # ⚠⚠ SHAPE IS NOT QUOTABILITY. A tight unimodal arm is the most quotable
    # thing there is; a 46% spread is not, whatever its shape.
    #
    print()
    if spread < 5:
        print("  QUOTABLE: yes -- %.1f%% spread. Quote the median." % spread)
    elif spread < 15:
        print("  QUOTABLE: with its range -- %.1f%% spread. Median AND range,"
              % spread)
        print("  never the median alone." % ())
    else:
        print("  QUOTABLE: NO -- %.1f%% spread. Report that it is unstable and"
              % spread)
        print("  what the modes are; a single figure from this is an artefact")
        print("  of whichever statistic was chosen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
