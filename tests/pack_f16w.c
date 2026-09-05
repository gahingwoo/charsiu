// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The three candidate fp16 weight layouts, on a desk.
 *
 * A layout does not have to be RIGHT to be testable: whichever one the
 * hardware wants, all three must be injective into the buffer the size
 * registers describe, or the probe that walks them will report a layout as
 * wrong when what actually happened is that two weights landed on the same
 * bytes. That failure looks exactly like "the hardware disagrees", which is
 * the reading this project has been wrong about before.
 *
 * So: every element in bounds, no two elements sharing a byte, and every
 * layout covering exactly n*k elements.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charsiu.h"

static int fail;

static void check(unsigned k, unsigned n, enum charsiu_w16_layout L)
{
	struct charsiu_matmul mm = { 1, k, n, CHARSIU_FP16, CHARSIU_FP16 };
	size_t bytes = charsiu_weight_bytes(&mm);
	unsigned char *seen = calloc(bytes, 1);
	unsigned nn, kk, hit = 0;

	if (!seen) { printf("  out of memory\n"); fail++; return; }
	for (nn = 0; nn < n; nn++)
		for (kk = 0; kk < k; kk++) {
			size_t off = charsiu_w16_offset(&mm, nn, kk, L);

			if (off == (size_t)-1 || off + 1 >= bytes) {
				printf("  L%d K=%u N=%u: (%u,%u) lands at %zu, "
				       "buffer is %zu\n", L, k, n, nn, kk,
				       off, bytes);
				fail++; free(seen); return;
			}
			if (seen[off] || seen[off + 1]) {
				printf("  L%d K=%u N=%u: (%u,%u) collides at "
				       "%zu\n", L, k, n, nn, kk, off);
				fail++; free(seen); return;
			}
			seen[off] = seen[off + 1] = 1;
			hit++;
		}
	if (hit != n * k) {
		printf("  L%d K=%u N=%u: covered %u of %u\n", L, k, n, hit, n * k);
		fail++;
	}
	free(seen);
}

/*
 * ⚠⚠ THE LAW A KV CACHE RESTS ON: while every group of 16 output channels is
 * full, a GROUP offset does not depend on n at all.
 *
 * charsiu_fp16_woffset is
 *     (n/16)*16*ke + (k/32)*32*ngsz + (n%16)*kgsz + (k%32)
 * with ke the padded k and ngsz = min(n_pad - (n/16)*16, 16). Only ngsz
 * carries n, and only for a partial last group. So a cache allocated at its
 * final width can be appended to along n and run at whatever multiple of 16 it
 * has reached, and the hardware finds every channel where the writer left it.
 *
 * That is a derivation, and this project has published derivations that the
 * board then refused. So it is checked here on every shape, and the board
 * checks the same thing end to end with npu_fp16_test --own.
 */
static void stable_in_n(unsigned k, unsigned n, unsigned nbig)
{
	struct charsiu_matmul a = { 1, k, n, CHARSIU_FP16, CHARSIU_FP16 };
	struct charsiu_matmul b = { 1, k, nbig, CHARSIU_FP16, CHARSIU_FP16 };
	unsigned c, kk;

	for (c = 0; c < n; c++)
		for (kk = 0; kk < k; kk++) {
			size_t oa = charsiu_w16_offset(&a, c, kk,
						       CHARSIU_W16_GROUP);
			size_t ob = charsiu_w16_offset(&b, c, kk,
						       CHARSIU_W16_GROUP);

			if (oa != ob) {
				printf("  k=%u channel %u element %u moves "
				       "when n goes %u -> %u: %zu vs %zu\n",
				       k, c, kk, n, nbig, oa, ob);
				fail++;
				return;
			}
		}
}

/*
 * ⚠ AND THE PRECONDITION IS REAL, which is the half that makes the check
 * above worth having. With a partial last group the offsets DO move, so a
 * cache read at a width that is not a multiple of 16 is a different
 * permutation of the same bytes. If this ever stops finding a difference,
 * stable_in_n is passing because it cannot fail.
 */
static void moves_in_n(unsigned k, unsigned n, unsigned nbig)
{
	struct charsiu_matmul a = { 1, k, n, CHARSIU_FP16, CHARSIU_FP16 };
	struct charsiu_matmul b = { 1, k, nbig, CHARSIU_FP16, CHARSIU_FP16 };
	/*
	 * ⚠ ngsz REACHES THE OFFSET ONLY THROUGH (k/32)*32*ngsz, so with one
	 * k group -- k up to the 32 element k group -- it cancels and a
	 * partial n group moves nothing. This test asked for a difference at
	 * k=8 and did not get one, which is the arithmetic saying so and not a
	 * defect. Above one k group the difference is real, and that is the
	 * case a cache is in.
	 */
	unsigned want = k > charsiu_weight_kgroup(CHARSIU_FP16);
	unsigned c, kk, moved = 0;

	for (c = 0; c < n && !moved; c++)
		for (kk = 0; kk < k; kk++)
			if (charsiu_w16_offset(&a, c, kk, CHARSIU_W16_GROUP) !=
			    charsiu_w16_offset(&b, c, kk, CHARSIU_W16_GROUP)) {
				moved = 1;
				break;
			}
	if (moved != want) {
		printf("  k=%u n=%u -> %u: a partial group %s\n", k, n, nbig,
		       moved ? "moved and should not have"
			     : "did not move and should have");
		fail++;
	}
}

int main(void)
{
	/* head_dim as N is the vendor's own attention shape; the rest are the
	 * projection widths this runtime already dispatches */
	static const unsigned ks[] = { 8, 64, 128, 512, 1024, 1536, 2048, 3072 };
	static const unsigned ns[] = { 2, 8, 32, 64, 96, 128, 256 };
	unsigned i, j, cases = 0;

	for (i = 0; i < sizeof(ks) / sizeof(ks[0]); i++)
		for (j = 0; j < sizeof(ns) / sizeof(ns[0]); j++)
			for (int L = 0; L < CHARSIU_W16_NLAYOUT; L++, cases++)
				check(ks[i], ns[j], (enum charsiu_w16_layout)L);
	/* and the appending law, at every full width pair */
	{
		static const unsigned ng = 16;
		unsigned a, b2;

		for (i = 0; i < sizeof(ks) / sizeof(ks[0]); i++)
			for (a = ng; a <= 512; a += ng)
				for (b2 = a; b2 <= 1024; b2 += ng, cases++)
					stable_in_n(ks[i], a, b2);
	}
	for (i = 0; i < sizeof(ks) / sizeof(ks[0]); i++) {
		moves_in_n(ks[i], 24, 1024); cases++;
		moves_in_n(ks[i], 40, 512);  cases++;
		moves_in_n(ks[i], 100, 256); cases++;
	}
	printf("  fp16 weight layouts: %d of %u cases wrong\n", fail, cases);
	return fail ? 1 : 0;
}
