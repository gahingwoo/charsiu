// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The K slicing arithmetic, on a desk with no NPU.
 *
 * What a slicing has to be, whichever rule made it: the slices tile [0, K)
 * with no gap and no overlap, none is empty, and none is wider than KMAX
 * (KFIT excepted, where the last one is deliberately wider). What EVEN adds:
 * every slice but the last is the same width and a multiple of the feature
 * atom, and the last is no wider than the others.
 *
 * ⚠ THE POINT OF THE EVEN RULE IS deal_pick's STABILITY, so the last check is
 * the one that matters: two slices of equal width cost the load balancer the
 * same, which is what makes the slice to device map repeat from tensor to
 * tensor. A slicing that is "nearly" even does not buy that.
 */
#include <stdio.h>
#include <stdlib.h>
#include "../src/kslice.h"

static int fail;

static void check(uint64_t k, unsigned kmax, int even)
{
	unsigned ks = (unsigned)((k + kmax - 1) / kmax);
	unsigned base = charsiu_even_k_base(k, ks, kmax, even, 0);
	uint64_t seen = 0;
	unsigned prev_end = 0, first_w = 0;

	for (unsigned ki = 0; ki < ks; ki++) {
		unsigned k0 = charsiu_slice_k0(k, ks, ki, kmax, even, 0);
		unsigned w = charsiu_slice_kw(k, ks, ki, kmax, even, 0);

		if (k0 != prev_end) {
			printf("  K=%llu kmax=%u even=%d: slice %u starts at %u, "
			       "previous ended at %u\n",
			       (unsigned long long)k, kmax, even, ki, k0, prev_end);
			fail++;
			return;
		}
		if (!w) {
			printf("  K=%llu kmax=%u even=%d: slice %u is empty\n",
			       (unsigned long long)k, kmax, even, ki);
			fail++;
			return;
		}
		if (w > kmax) {
			printf("  K=%llu kmax=%u even=%d: slice %u is %u, past "
			       "KMAX\n", (unsigned long long)k, kmax, even, ki, w);
			fail++;
			return;
		}
		if (ki == 0)
			first_w = w;
		if (base) {
			if (ki + 1 < ks && w != base) {
				printf("  K=%llu kmax=%u: slice %u is %u, not the "
				       "even base %u\n", (unsigned long long)k,
				       kmax, ki, w, base);
				fail++;
				return;
			}
			if (ki + 1 < ks && (w % 32)) {
				printf("  K=%llu kmax=%u: even slice %u is %u, "
				       "not a multiple of the atom\n",
				       (unsigned long long)k, kmax, ki, w);
				fail++;
				return;
			}
			if (ki + 1 == ks && w > first_w) {
				printf("  K=%llu kmax=%u: the last slice %u is "
				       "wider than the others %u\n",
				       (unsigned long long)k, kmax, w, first_w);
				fail++;
				return;
			}
		}
		seen += w;
		prev_end = k0 + w;
	}
	if (seen != k) {
		printf("  K=%llu kmax=%u even=%d: the slices add to %llu\n",
		       (unsigned long long)k, kmax, even, (unsigned long long)seen);
		fail++;
	}
}

int main(void)
{
	static const unsigned kmaxes[] = { 1024, 2048, 3072, 4096 };
	unsigned n = 0;

	/* every K a real model brings, and a dense walk for the edges */
	static const uint64_t models[] = {
		576, 1024, 1152, 1536, 2048, 3072, 4096, 5120, 8192, 8960,
		11008, 16384,
	};

	for (unsigned i = 0; i < sizeof(kmaxes) / sizeof(kmaxes[0]); i++)
		for (unsigned j = 0; j < sizeof(models) / sizeof(models[0]); j++)
			for (int e = 0; e < 2; e++, n++)
				check(models[j], kmaxes[i], e);
	for (uint64_t k = 32; k <= 12288; k += 32)
		for (unsigned i = 0; i < sizeof(kmaxes) / sizeof(kmaxes[0]); i++)
			for (int e = 0; e < 2; e++, n++)
				check(k, kmaxes[i], e);
	/*
	 * ⚠ AND THE ONE THAT MATTERS: the two shapes that lost every reuse.
	 * Phi-3.5's 3072 and gemma4's, at the shipped KMAX, must come out as
	 * equal halves -- if they do not, the change buys nothing on the only
	 * two models it was written for.
	 */
	if (charsiu_even_k_base(3072, 2, 2048, 1, 0) != 1536) {
		printf("  K=3072 at KMAX 2048 does not halve\n");
		fail++;
	}
	if (charsiu_even_k_base(8960, 5, 2048, 1, 0) != 1792) {
		printf("  K=8960 at KMAX 2048 is not 1792 x5\n");
		fail++;
	}
	/* KFIT keeps the old rule, whose last slice is wider than KMAX */
	if (charsiu_even_k_base(3072, 2, 2048, 1, 1) != 0) {
		printf("  KFIT was evened out\n");
		fail++;
	}
	printf("  K slicing: %d of %u cases wrong\n", fail, n);
	return fail ? 1 : 0;
}
