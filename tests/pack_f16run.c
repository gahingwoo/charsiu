// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * charsiu_f2h_run against charsiu_f2h, over EVERY float there is.
 *
 * The vector run exists because packing a group's activations is where an
 * fp16 attention layer spends its CPU time, and it must produce the same
 * bytes as the per element definition -- not "the same to within a rounding
 * mode". charsiu_f2h truncates the mantissa and flushes subnormals to zero;
 * the hardware's own vcvt_f16_f32 rounds to nearest even and keeps them, so
 * the two disagree on the low bit of most inputs and the native convert is
 * NOT a legal substitute here. Every fp16 buffer this project has checked
 * against the NPU was built with the truncating form.
 *
 * A sampled check would not settle it: the disagreements a wrong vector form
 * produces live at the boundaries -- exp exactly 0 or 0x1f, the subnormals,
 * the NaNs -- which is a vanishing fraction of a random draw. So this walks
 * all 2^32 bit patterns. It runs in a couple of seconds on the desk, because
 * the development host is itself aarch64.
 *
 * ⚠ The SEQUENCE the strided pack walks is checked here too, but against a
 * copy of the loop it replaced rather than against npufp16.c itself, which
 * needs a device. npu_fp16_test --group is what checks the real path, bit for
 * bit at mixed shapes, on the board.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "charsiu.h"

static int fail;

static void exhaustive(void)
{
	uint64_t bad = 0, first = 0;
	uint64_t sum = 0;
	float x[8];
	uint16_t a[8], b[8];

	for (uint64_t base = 0; base < 0x100000000ULL; base += 8) {
		for (int i = 0; i < 8; i++) {
			uint32_t u = (uint32_t)(base + i);

			memcpy(&x[i], &u, sizeof(u));
			a[i] = charsiu_f2h(x[i]);
		}
		charsiu_f2h_run(b, x, 8);
		if (memcmp(a, b, sizeof(a))) {
			if (!bad)
				first = base;
			bad++;
		}
		/* consumed below, so no part of this can be elided */
		for (int i = 0; i < 8; i++)
			sum += b[i];
	}
	printf("pack_f16run: 2^32 bit patterns, %llu mismatching blocks"
	       " (checksum %llu)\n", (unsigned long long)bad,
	       (unsigned long long)sum);
	if (bad) {
		printf("pack_f16run: FAIL first at 0x%08llx\n",
		       (unsigned long long)first);
		fail = 1;
	}
}

/* the loop the strided arm of the pack replaced, kept whole */
static void strided_old(uint16_t *d, const float *X, unsigned m, unsigned k,
			size_t xs, size_t nel)
{
	size_t e = 0;

	for (unsigned r = 0; r < m && e < nel; r++) {
		const float *xr = X + (size_t)r * xs;

		for (unsigned c = 0; c < k && e < nel; c++, e++)
			d[e] = charsiu_f2h(xr[c]);
	}
}

static void strided_new(uint16_t *d, const float *X, unsigned m, unsigned k,
			size_t xs, size_t nel)
{
	size_t e = 0;

	for (unsigned r = 0; r < m && e < nel; r++) {
		size_t cn = k;

		if (cn > nel - e)
			cn = nel - e;
		charsiu_f2h_run(d + e, X + (size_t)r * xs, cn);
		e += cn;
	}
}

static void strided(void)
{
	/* m, k, xstride, and a cap that TRUNCATES the last row -- the bound
	 * the old inner loop carried per element is the only thing that made
	 * a short cap legal, and hoisting it is where a rewrite goes wrong */
	static const unsigned shape[][4] = {
		{ 1,  1,  1,   1 }, { 1,  7,  9,   7 }, { 3, 64, 2048, 192 },
		{ 5, 13, 13,  65 }, { 5, 13, 40,  60 }, { 5, 13, 40,  7 },
		{ 80, 64, 2048, 5120 }, { 80, 64, 2048, 5119 },
		{ 17, 1024, 1024, 17408 }, { 17, 1024, 1024, 9000 },
		{ 4, 3, 5, 0 }, { 9, 8, 8, 71 },
	};
	size_t big = 80 * 2048 + 4096;
	float *X = malloc(big * sizeof(*X));
	uint16_t *a = malloc(20000 * sizeof(*a));
	uint16_t *b = malloc(20000 * sizeof(*b));
	unsigned s, ok = 0;

	if (!X || !a || !b) {
		printf("pack_f16run: out of memory\n");
		fail = 1;
		return;
	}
	for (size_t i = 0; i < big; i++) {
		/* a spread that reaches the flush, the saturation and the
		 * ordinary range, not one decade of well behaved floats */
		static const float sc[] = { 1.0f, 1e-8f, 1e8f, -3.5e-5f,
					    65504.0f, 1e30f };

		X[i] = ((float)((int)(i % 977) - 488) * 0.0137f)
		     * sc[i % (sizeof(sc) / sizeof(sc[0]))];
	}
	for (s = 0; s < sizeof(shape) / sizeof(shape[0]); s++) {
		unsigned m = shape[s][0], k = shape[s][1];
		size_t xs = shape[s][2], nel = shape[s][3];

		memset(a, 0xa5, 20000 * sizeof(*a));
		memset(b, 0xa5, 20000 * sizeof(*b));
		strided_old(a, X, m, k, xs, nel);
		strided_new(b, X, m, k, xs, nel);
		if (memcmp(a, b, 20000 * sizeof(*a))) {
			printf("pack_f16run: FAIL strided m=%u k=%u xs=%zu"
			       " nel=%zu\n", m, k, xs, nel);
			fail = 1;
		} else {
			ok++;
		}
	}
	/* ⚠ the count that AGREED, not the count that ran. The first draft
	 * printed s either way, so a run with eight failures above it still
	 * ended in a line saying twelve shapes agree. */
	printf("pack_f16run: %u of %u strided shapes agree with the per"
	       " element loop, cap truncation included\n", ok, s);
	free(X); free(a); free(b);
}

int main(void)
{
	strided();
	exhaustive();
	if (fail)
		printf("pack_f16run: FAILED\n");
	else
		printf("pack_f16run: ok\n");
	return fail;
}
