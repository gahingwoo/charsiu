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
	printf("  fp16 weight layouts: %d of %u cases wrong\n", fail, cases);
	return fail ? 1 : 0;
}
