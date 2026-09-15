/*
 * Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
 * SPDX-License-Identifier: GPL-2.0
 *
 * Is the accumulator read order a bijection at this batch width?
 *
 * THIS IS THE CHECK THAT SHOULD HAVE EXISTED BEFORE THE FIRST BOARD ROUND.
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
 * would quietly stop being true the next time that function is touched.
 *
 * AND IT ASSERTS IT AGAINST THE PREDICATE THAT SHIPS. The law used to be
 * written out four separate times -- the gate in src/npudev.c, the chunker in
 * tools/charsiu_run.c, this sweep, and the map itself -- and this sweep
 * compared the map only against its own copy, so the two that reach the
 * hardware were never checked by anything. They all call charsiu_acc_width_ok
 * now, which lives next to charsiu_acc_index because P = m / 2 is where the
 * rule comes from, and that is the one this sweeps.
 *
 * It links against the real src/job.c, so it cannot drift from the function
 * that actually runs.
 */
#include <stdio.h>
#include <string.h>
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

int main(int argc, char **argv)
{
	/* A BINARY THAT CANNOT SAY ITS COMMIT CANNOT BE AN ARM. Round 415 lost
	 * a comparison to an installed control that predated the build stamp,
	 * and round 413 added this to eleven of the eighteen probe tools and
	 * recorded that it had done all of them. This is one of the five it
	 * never reached. Before any argument parsing, because the usual
	 * failure is argv[1] going straight to atoi and an unrecognised
	 * --version becoming a dimension of ZERO submitted to the hardware. */
	if (argc > 1 && !strcmp(argv[1], "--version")) {
		printf("%s\n", CHARSIU_BUILD);
		return 0;
	}
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
		/*
		 * AGAINST THE PREDICATE THAT SHIPS, not against a copy of it
		 * written here. charsiu_acc_width_ok is what the gate in
		 * npudev.c and the chunker in charsiu_run.c both call, so a
		 * map that disagrees with it fails this sweep whichever of the
		 * two somebody edited.
		 */
		if (all != !!charsiu_acc_width_ok(m)) {
			printf("\n*** THE LAW IS BROKEN AT m = %u: charsiu_acc"
			       "_width_ok says %s, the map is %s (%s)\n", m,
			       charsiu_acc_width_ok(m) ? "SAFE" : "BROKEN",
			       all ? "SAFE" : "BROKEN", why);
			bad++;
		}
	}
	if (bad) {
		printf("\n%d width(s) contradict charsiu_acc_width_ok. The"
		       " gate in npudev.c and the chunker in charsiu_run.c"
		       " both call it, so both are now wrong too.\n", bad);
		return 1;
	}
	printf("\nevery width 2..96 agrees with charsiu_acc_width_ok, at"
	       " every n.\n");
	return 0;
}
