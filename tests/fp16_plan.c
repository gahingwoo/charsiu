// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The layout of a grouped fp16 submit, on a desk.
 *
 * charsiu_fp16_matmul_group runs N matmuls in one submit out of four shared
 * buffers. If two ops' regions overlap in any of them, nothing faults: op i
 * comes back holding part of op j. That is a wrong number, it looks like the
 * hardware disagreeing, and this project has read exactly that wrong before.
 *
 * A device cannot be opened here, and it does not need to be. Everything that
 * decides whether the group is addressable is arithmetic:
 *
 *   - every region is at least as big as the job that writes into it needs;
 *   - no two regions in the same buffer share a byte;
 *   - every region starts on a page, because the weight and coefficient bases
 *     have no established alignment and a page is a superset of every
 *     plausible one;
 *   - the totals cover the last region;
 *   - a shape the single call would refuse refuses the whole group, rather
 *     than running the other ops and returning a short answer.
 *
 * The mixed shape cases are the point. With every op the same size, an offset
 * that is wrong by a whole region still lands inside a legal region.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu.h"
#include "charsiu_llm.h"
#include "../src/fp16plan.h"

static int fail;
static unsigned cases;

static void bad(const char *what, unsigned i, unsigned j)
{
	printf("  %s: ops %u and %u\n", what, i, j);
	fail++;
}

/* [a, a+asz) and [b, b+bsz) must not share a byte */
static int overlap(size_t a, size_t asz, size_t b, size_t bsz)
{
	return a < b + bsz && b < a + asz;
}

static void check(struct charsiu_fp16_op *ops, unsigned n)
{
	struct charsiu_fp16_plan p;
	unsigned i, j;

	cases++;
	if (charsiu_fp16_make_plan(ops, n, &p)) {
		printf("  a legal group of %u was refused\n", n);
		fail++;
		return;
	}
	for (i = 0; i < n; i++) {
		struct charsiu_matmul mm = { ops[i].m, ops[i].k, ops[i].n,
					     CHARSIU_FP16, CHARSIU_FP16 };

		if (p.wsz[i] < charsiu_weight_bytes(&mm) ||
		    p.isz[i] < (size_t)ops[i].m * ops[i].k * 2 ||
		    p.osz[i] < (size_t)ops[i].m * ops[i].n * 4 ||
		    p.csz[i] < charsiu_coef_bytes(&mm))
			bad("a region is smaller than its job needs", i, i);
		if ((p.woff[i] | p.ioff[i] | p.ooff[i] | p.coff[i]) & 4095u)
			bad("a region does not start on a page", i, i);
		if (p.woff[i] + p.wsz[i] > p.wtot ||
		    p.ioff[i] + p.isz[i] > p.itot ||
		    p.ooff[i] + p.osz[i] > p.otot ||
		    p.coff[i] + p.csz[i] > p.ctot)
			bad("a region runs past its buffer", i, i);
		if (ops[i].n > p.nmax)
			bad("nmax does not cover an op", i, i);
		for (j = 0; j < i; j++) {
			if (overlap(p.woff[i], p.wsz[i], p.woff[j], p.wsz[j]))
				bad("weights overlap", i, j);
			if (overlap(p.ioff[i], p.isz[i], p.ioff[j], p.isz[j]))
				bad("inputs overlap", i, j);
			if (overlap(p.ooff[i], p.osz[i], p.ooff[j], p.osz[j]))
				bad("outputs overlap", i, j);
			if (overlap(p.coff[i], p.csz[i], p.coff[j], p.csz[j]))
				bad("coefficients overlap", i, j);
		}
	}
}

static void refuses(struct charsiu_fp16_op *ops, unsigned n, const char *why)
{
	struct charsiu_fp16_plan p;

	cases++;
	if (!charsiu_fp16_make_plan(ops, n, &p)) {
		printf("  %s was planned instead of refused\n", why);
		fail++;
	}
}

int main(void)
{
	static const unsigned ms[] = { 1, 2, 4, 8, 32, 80, 178 };
	static const unsigned ks[] = { 32, 64, 128, 256, 512, 1024, 2048 };
	static const unsigned ns[] = { 32, 64, 96, 128, 512, 1024 };
	struct charsiu_fp16_op ops[FP16_GROUP_MAX];
	unsigned a, b, c, i, n;

	printf("fp16 group plan\n");

	/* uniform groups: every shape this has ever run, at every group size */
	for (a = 0; a < sizeof(ms) / sizeof(*ms); a++)
		for (b = 0; b < sizeof(ks) / sizeof(*ks); b++)
			for (c = 0; c < sizeof(ns) / sizeof(*ns); c++)
				for (n = 1; n <= FP16_GROUP_MAX; n += 7) {
					for (i = 0; i < n; i++) {
						ops[i].m = ms[a];
						ops[i].k = ks[b];
						ops[i].n = ns[c];
						ops[i].X = (const float *)1;
						ops[i].W = (const void *)1;
						ops[i].Y = (float *)1;
					}
					check(ops, n);
				}

	/* mixed shapes, which is where an offset that is wrong by a whole
	 * region stops being invisible */
	for (n = 2; n <= FP16_GROUP_MAX; n++) {
		for (i = 0; i < n; i++) {
			ops[i].m = ms[i % (sizeof(ms) / sizeof(*ms))];
			ops[i].k = ks[i % (sizeof(ks) / sizeof(*ks))];
			ops[i].n = ns[i % (sizeof(ns) / sizeof(*ns))];
			ops[i].X = (const float *)1;
			ops[i].W = (const void *)1;
			ops[i].Y = (float *)1;
		}
		check(ops, n);
	}

	/* a growing KV cache: the shape attention actually presents, one head
	 * a token, n climbing while k stays the head dimension */
	for (n = 2; n <= FP16_GROUP_MAX; n++) {
		for (i = 0; i < n; i++) {
			ops[i].m = 8;
			ops[i].k = 64;
			ops[i].n = 32 + i * 32;
			ops[i].X = (const float *)1;
			ops[i].W = (const void *)1;
			ops[i].Y = (float *)1;
		}
		check(ops, n);
	}

	/* and what must be refused */
	for (i = 0; i < 4; i++) {
		ops[i].m = 8; ops[i].k = 64; ops[i].n = 64;
		ops[i].X = (const float *)1;
		ops[i].W = (const void *)1;
		ops[i].Y = (float *)1;
	}
	refuses(ops, 0, "an empty group");
	refuses(ops, FP16_GROUP_MAX + 1, "a group of 33");
	ops[2].k = 16;  refuses(ops, 4, "k = 16, which wedged both cores");
	ops[2].k = 64;
	ops[2].n = 8;   refuses(ops, 4, "n = 8, which wedged both cores");
	ops[2].n = 64;
	ops[2].m = 0;   refuses(ops, 4, "m = 0");
	ops[2].m = 8;
	check(ops, 4);  /* and the same group, put back, is legal again */

	printf("%s: %u cases\n", fail ? "FAIL" : "ok", cases);
	return fail ? 1 : 0;
}
