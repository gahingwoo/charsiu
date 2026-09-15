// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The output sentinel, driven through all three verdicts.
 *
 * A check that has never been seen to fire is not a check. The rule this
 * replaces could only ever answer "every cell is poison" or "not every cell
 * is poison", and the board has never produced the first -- so in practice
 * it was a full scalar pass over the answer, twice a submit, whose result was
 * constant. This drives it deliberately: nothing written, everything written,
 * and the case nothing could report before, a job that wrote some rows.
 *
 * AND THE POISON VALUE IS A LEGAL FLOAT. 0xdeadbeef read as one is about
 * -6.26e13, so an op that genuinely computes it in a row's first cell is read
 * as an op that did not write that row. That is not new -- the every-cell
 * form had the same exposure, needing every cell to collide instead of one --
 * but it is narrower now, so the last case here pins the behaviour rather
 * than leaving it to be rediscovered.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "sentinel.h"

static int fail;

static void expect(const char *what, int got, int want,
		   unsigned unwritten, unsigned uwant)
{
	if (got == want && unwritten == uwant) {
		printf("  ok    %-34s verdict %d, %u rows unwritten\n",
		       what, got, unwritten);
		return;
	}
	printf("  FAIL  %-34s verdict %d (want %d), %u unwritten (want %u)\n",
	       what, got, want, unwritten, uwant);
	fail = 1;
}

/*
 * The predicate that decides whether the buffer is still poisoned for this
 * group. It is the one piece of the early-sentinel path that fails SILENTLY:
 * a wrong yes leaves no sentinels, the readback then finds live data and
 * reports a healthy job, which is what it reports for a healthy job.
 */
static void matches(void)
{
	size_t off[3]  = { 0, 8192, 24576 };
	unsigned m[3]  = { 78, 78, 78 }, n[3] = { 864, 864, 864 };
	size_t poff[3] = { 0, 8192, 24576 };
	unsigned pm[3] = { 78, 78, 78 }, pn[3] = { 864, 864, 864 };
	unsigned ok = 0, cases = 0;

#define WANT(w, what) do {                                              \
	int got = charsiu_poison_matches(3, off, m, n, 3, poff, pm, pn);\
	cases++;                                                        \
	if (got == (w)) { ok++; printf("  ok    %-32s %d\n", what, got); }\
	else { printf("  FAIL  %-32s %d want %d\n", what, got, (w));    \
	       fail = 1; }                                              \
} while (0)

	printf("== the early-sentinel predicate\n");
	WANT(1, "everything the same");
	m[1] = 77;   WANT(0, "one op's m moved");       m[1] = 78;
	n[2] = 880;  WANT(0, "one op's n moved");       n[2] = 864;
	off[0] = 4096; WANT(0, "one op's offset moved"); off[0] = 0;
	WANT(1, "and back again");
	cases++;
	if (charsiu_poison_matches(2, off, m, n, 3, poff, pm, pn) == 0) {
		ok++; printf("  ok    %-32s 0\n", "fewer ops than poisoned");
	} else {
		printf("  FAIL  fewer ops than poisoned\n"); fail = 1;
	}
	cases++;
	if (charsiu_poison_matches(0, off, m, n, 0, poff, pm, pn) == 0) {
		ok++; printf("  ok    %-32s 0\n", "no ops at all");
	} else {
		printf("  FAIL  no ops at all\n"); fail = 1;
	}
#undef WANT
	printf("sentinel: %u of %u predicate cases as expected\n", ok, cases);
}

int main(void)
{
	const unsigned shape[][2] = { {1, 1}, {1, 4096}, {160, 864},
				      {37, 64}, {2, 2}, {255, 3} };
	unsigned s;

	printf("== the output sentinel, all three verdicts\n");
	for (s = 0; s < sizeof(shape) / sizeof(shape[0]); s++) {
		unsigned m = shape[s][0], n = shape[s][1], uw = 99, r, c;
		size_t cells = (size_t)m * n;
		uint32_t *o = malloc(cells * sizeof(*o));
		char what[64];

		if (!o) { printf("  out of memory\n"); return 1; }

		/* 1. the hardware wrote nothing */
		memset(o, 0, cells * sizeof(*o));
		charsiu_poison_rows(o, m, n);
		snprintf(what, sizeof(what), "%ux%u wrote nothing", m, n);
		expect(what, charsiu_poison_verdict(o, m, n, &uw), -1, uw, m);

		/* 2. it wrote all of it. Zero is the value that motivated the
		 * sentinel in the first place, so the healthy case uses it */
		memset(o, 0, cells * sizeof(*o));
		snprintf(what, sizeof(what), "%ux%u wrote zeros", m, n);
		expect(what, charsiu_poison_verdict(o, m, n, &uw), 0, uw, 0);

		/* 3. it wrote some rows and stopped */
		if (m > 1) {
			charsiu_poison_rows(o, m, n);
			for (r = 0; r < m / 2; r++)
				for (c = 0; c < n; c++)
					o[(size_t)r * n + c] = 0x3f800000u;
			snprintf(what, sizeof(what), "%ux%u wrote %u of %u",
				 m, n, m / 2, m);
			expect(what, charsiu_poison_verdict(o, m, n, &uw), 0,
			       uw, m - m / 2);
		}

		/* 4. it wrote the poison value itself, in every first cell */
		for (r = 0; r < m; r++)
			for (c = 0; c < n; c++)
				o[(size_t)r * n + c] = CHARSIU_POISON;
		snprintf(what, sizeof(what), "%ux%u computed the poison", m, n);
		expect(what, charsiu_poison_verdict(o, m, n, &uw), -1, uw, m);

		free(o);
	}
	matches();
	printf("sentinel: %s\n", fail ? "FAILED" : "ok");
	return fail;
}
