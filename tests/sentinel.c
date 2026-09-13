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
 * ⚠ AND THE POISON VALUE IS A LEGAL FLOAT. 0xdeadbeef read as one is about
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
	printf("sentinel: %s\n", fail ? "FAILED" : "ok");
	return fail;
}
