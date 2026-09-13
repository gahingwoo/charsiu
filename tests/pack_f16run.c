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

/*
 * The causal triangle: row r may be nonzero only in its first (tri0 + r)
 * entries, so the pack memsets the rest instead of converting it. The claim
 * is that the two produce the same bytes, and it holds ONLY because every
 * zero in that tail is a positive zero -- charsiu_f2h(-0.0f) is 0x8000, not
 * 0x0000, so a negative zero in the tail would make memset wrong. This puts
 * one in deliberately, in the part the promise does NOT cover, and a matching
 * positive zero where it does.
 */
static void triangle(void)
{
	const unsigned m = 37, k = 128;
	float *X = malloc((size_t)m * k * sizeof(*X));
	uint16_t *a = malloc((size_t)m * k * sizeof(*a));
	uint16_t *b = malloc((size_t)m * k * sizeof(*b));
	unsigned tri0 = 9, cases = 0, ok = 0;

	if (!X || !a || !b) { fail = 1; return; }
	for (unsigned pass = 0; pass < 2; pass++) {
		for (unsigned r = 0; r < m; r++) {
			size_t keep = (size_t)tri0 + r;

			if (keep > k)
				keep = k;
			for (unsigned c = 0; c < k; c++) {
				float *p = &X[(size_t)r * k + c];

				if (c < keep)
					/* pass 1 puts a NEGATIVE zero inside
					 * the promise, where it is converted */
					*p = (pass && c + 1 == keep) ? -0.0f
					   : (float)((int)c - 40) * 0.031f;
				else
					*p = 0.0f;
			}
		}
		for (unsigned r = 0; r < m; r++) {
			size_t keep = (size_t)tri0 + r;

			if (keep > k)
				keep = k;
			/* everything, the way the pack ran before */
			for (unsigned c = 0; c < k; c++)
				a[(size_t)r * k + c] =
					charsiu_f2h(X[(size_t)r * k + c]);
			/* the promise, the way it runs now */
			charsiu_f2h_run(b + (size_t)r * k, X + (size_t)r * k,
					keep);
			memset(b + (size_t)r * k + keep, 0, (k - keep) * 2);
		}
		cases++;
		if (memcmp(a, b, (size_t)m * k * sizeof(*a)) == 0)
			ok++;
		else
			printf("pack_f16run: FAIL triangle pass %u\n", pass);
	}
	/* and the control: a triangle whose tail is NOT zero must differ,
	 * or this test would pass on a promise nobody keeps */
	X[(size_t)5 * k + k - 1] = 1.5f;
	for (unsigned c = 0; c < k; c++)
		a[(size_t)5 * k + c] = charsiu_f2h(X[(size_t)5 * k + c]);
	charsiu_f2h_run(b + (size_t)5 * k, X + (size_t)5 * k, tri0 + 5);
	memset(b + (size_t)5 * k + tri0 + 5, 0, (k - tri0 - 5) * 2);
	cases++;
	if (memcmp(a + (size_t)5 * k, b + (size_t)5 * k, k * sizeof(*a)))
		ok++;
	else
		printf("pack_f16run: FAIL the broken-promise control agreed\n");
	if (ok != cases)
		fail = 1;
	printf("pack_f16run: %u of %u triangle cases as expected, negative"
	       " zero and broken promise included\n", ok, cases);
	free(X); free(a); free(b);
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
	triangle();
	exhaustive();
	if (fail)
		printf("pack_f16run: FAILED\n");
	else
		printf("pack_f16run: ok\n");
	return fail;
}
