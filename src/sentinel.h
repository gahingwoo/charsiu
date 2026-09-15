/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
#ifndef CHARSIU_SENTINEL_H
#define CHARSIU_SENTINEL_H

#include <stdint.h>
#include <stddef.h>

/*
 * DID THE HARDWARE WRITE THIS OUTPUT?
 *
 * A job that never ran and a job that computed zero leave the same four bytes,
 * so the buffer is poisoned before the submit and read back after it. This is
 * the rule on its own, in a header, because the version that lived inline
 * poisoned and then counted EVERY CELL -- m * n words an op, thirty two ops a
 * group, on both sides of the submit -- and at 852 tokens that was 668 ms of
 * poisoning plus 581 of counting, more than the hardware's own 915.
 *
 * ONE SENTINEL A ROW IS NOT A WEAKENING. A job writes its whole output or
 * none of it, so a row whose first word survived is a row that was not
 * written. The every-cell form could only ever answer "all of it is poison"
 * or "not all of it" -- a half written output was accepted in silence. This
 * one separates the three cases and costs m words instead of m * n.
 *
 * It is a header so tests/sentinel.c runs the same code the runtime runs.
 * A check that has never been seen to fire is not a check: that test drives
 * all three verdicts, including the partial one nothing could report before.
 */
#define CHARSIU_POISON 0xdeadbeefu

static inline void charsiu_poison_rows(uint32_t *o, unsigned m, unsigned n)
{
	unsigned r;

	for (r = 0; r < m; r++)
		o[(size_t)r * n] = CHARSIU_POISON;
}

/*
 * 0 if the op wrote, -1 if it wrote nothing. *unwritten, when given, is how
 * many rows still carry their sentinel -- 0 on a healthy op, m on a dead one,
 * and anything between on a job that stopped in the middle.
 */
static inline int charsiu_poison_verdict(const uint32_t *o, unsigned m,
					 unsigned n, unsigned *unwritten)
{
	unsigned r, live = 0;

	for (r = 0; r < m; r++)
		live += o[(size_t)r * n] != CHARSIU_POISON;
	if (unwritten)
		*unwritten = m - live;
	return live ? 0 : -1;
}

/*
 * IS THE BUFFER STILL POISONED FOR EXACTLY THIS GROUP?
 *
 * The sentinels can be written early -- once the previous group's answers have
 * been read -- so that the release's cache flush carries them and the next
 * group skips a prep and a fini of the whole output buffer. That is only sound
 * while the shapes repeat, and this is the predicate that decides.
 *
 * GETTING IT WRONG IS SILENT. If it says yes when the sentinels are not
 * there, the readback reads whatever was in the buffer, finds it is not
 * poison, and reports a healthy job -- which is what it would report for a
 * healthy job. The check would simply stop being a check, and nothing in a
 * log would say so. That is why it is a testable function and not four
 * conditions inside a loop.
 *
 * Offsets alone are NOT enough: a sentinel sits at o[r * n] for r < m, so a
 * group with the same regions and a different m or n inspects words nobody
 * wrote.
 */
static inline int charsiu_poison_matches(unsigned nops,
					 const size_t *ooff,
					 const unsigned *m,
					 const unsigned *n,
					 unsigned pnops,
					 const size_t *pooff,
					 const unsigned *pm,
					 const unsigned *pn)
{
	unsigned i;

	if (!nops || nops != pnops)
		return 0;
	for (i = 0; i < nops; i++)
		if (ooff[i] != pooff[i] || m[i] != pm[i] || n[i] != pn[i])
			return 0;
	return 1;
}

#endif /* CHARSIU_SENTINEL_H */
