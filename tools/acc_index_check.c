/*
 * Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
 * SPDX-License-Identifier: GPL-2.0
 *
 * Is the accumulator read order a bijection at this batch width?
 *
 * ⚠⚠ THIS IS THE CHECK THAT SHOULD HAVE EXISTED BEFORE THE FIRST BOARD ROUND.
 *
 * Four rounds went into asking the hardware which batch widths are wrong, and
 * the answer was in charsiu_acc_index the whole time. In its roleswap2 branch
 * the map covers 64 * P slots per group of 32 channels where the group needs
 * 32 * m, and 64P == 32m only when P == m/2 -- which is to say only when m is
 * EVEN. The accumulator surface is organised in PAIRS OF ROWS.
 *
 * So an odd width collides with itself, at every n, and no board is needed to
 * find that out. Odd widths are not a bug to fix: no integer P works for them.
 *
 * This walks m = 2..96 against several n and checks four properties of the
 * map -- every index in range, no collision, no hole, and the four
 * consecutive slots the gather moves off one index -- then ASSERTS the law
 * rather than only printing it. A table nobody reads is how `m must be even`
 * would quietly stop being true the next time that function is touched, and
 * the two places that encode it (w4_batch_why_not in src/npudev.c, and
 * prefill_width in tools/charsiu_run.c) cannot see each other.
 *
 * It links against the real src/job.c, so it cannot drift from the function
 * that actually runs.
 */
#include <stdio.h>
#include <stdlib.h>
#include "charsiu.h"

static int check(unsigned m, unsigned n, const char **why)
{
	size_t tot = (size_t)m * n;
	unsigned char *hit = calloc(tot, 1);
	int ok = 1;

	if (!hit) {
		*why = "out of memory";
		return 0;
	}
	*why = "";
	for (unsigned mi = 0; mi < m && ok; mi++)
		for (unsigned ni = 0; ni < n; ni++) {
			size_t k = charsiu_acc_index(mi, ni, m, 1);

			if (k >= tot) { ok = 0; *why = "out of range"; break; }
			if (hit[k])   { ok = 0; *why = "collision";    break; }
			hit[k] = 1;
		}
	if (ok)
		for (size_t k = 0; k < tot; k++)
			if (!hit[k]) { ok = 0; *why = "hole"; break; }
	if (ok)
		for (unsigned mi = 0; mi < m && ok; mi++)
			for (unsigned ni = 0; ni + 3 < n; ni += 4) {
				size_t b = charsiu_acc_index(mi, ni, m, 1);

				for (unsigned q = 1; q < 4; q++)
					if (charsiu_acc_index(mi, ni + q, m, 1)
					    != b + q) {
						ok = 0;
						*why = "not 4 in a row";
						break;
					}
				if (!ok)
					break;
			}
	free(hit);
	return ok;
}

int main(void)
{
	unsigned ns[] = { 512, 2048, 8192 };
	int bad = 0;

	printf("m    ");
	for (unsigned i = 0; i < 3; i++)
		printf("n=%-11u", ns[i]);
	printf("verdict\n");
	for (unsigned m = 2; m <= 96; m++) {
		const char *why = "", *w;
		int all = 1;

		printf("%-5u", m);
		for (unsigned i = 0; i < 3; i++) {
			int ok = check(m, ns[i], &w);

			printf("%-13s", ok ? "ok" : w);
			if (!ok) {
				all = 0;
				if (!*why)
					why = w;
			}
		}
		printf("%s\n", all ? "SAFE" : "BROKEN");
		if (all != (int)(m % 2 == 0)) {
			printf("\n*** THE LAW IS BROKEN AT m = %u: predicted %s,"
			       " measured %s (%s)\n", m,
			       m % 2 == 0 ? "SAFE" : "BROKEN",
			       all ? "SAFE" : "BROKEN", why);
			bad++;
		}
	}
	if (bad) {
		printf("\n%d width(s) contradict `m %% 2 == 0`. The gate in"
		       " npudev.c and the chunker in charsiu_run.c are both"
		       " built on it.\n", bad);
		return 1;
	}
	printf("\nevery width 2..96 agrees with `m %% 2 == 0`, at every n.\n");
	return 0;
}
