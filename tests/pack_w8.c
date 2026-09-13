// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * charsiu_w8_offset against the packer that has written the int8 weight tile
 * since round 139, cell by cell.
 *
 * ⚠⚠ WHY THIS TEST AND NOT A READING OF THE FORMULA. An int8 KV surface is
 * written a position at a time, in place, by an offset function -- exactly
 * what charsiu_fp16_woffset does for the fp16 one. The whole safety of that
 * arrangement is that the offset function and the bulk packer describe the
 * SAME permutation, and the two are different code. A disagreement between
 * them does not crash and does not print anything: it produces a matmul over
 * a permuted weight, which is a plausible wrong number.
 *
 * So this fills a buffer through charsiu_pack_weights, fills a second through
 * charsiu_w8_offset, and requires every byte to match -- with a marker per
 * cell, so a permutation cannot pass by being a permutation of equal values.
 *
 * And the ng = 32 property the KV surface depends on: an offset must not
 * depend on n once every n group is full, or a surface appended along n could
 * not be read at a width shorter than its allocation.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charsiu.h"

static int fail;

/* every cell injective, in bounds, and covering the buffer exactly once */
static void injective(unsigned k, unsigned n)
{
	struct charsiu_matmul mm = { 1, k, n, CHARSIU_INT8, CHARSIU_FP16 };
	size_t bytes = charsiu_weight_bytes(&mm);
	unsigned char *seen = calloc(bytes, 1);
	unsigned nn, kk;

	if (!seen) { printf("  out of memory\n"); fail++; return; }
	for (nn = 0; nn < n; nn++)
		for (kk = 0; kk < k; kk++) {
			size_t off = charsiu_w8_offset(&mm, nn, kk);

			if (off == (size_t)-1 || off >= bytes) {
				printf("  K=%u N=%u: (%u,%u) lands at %zu, "
				       "buffer is %zu\n", k, n, nn, kk, off,
				       bytes);
				fail++; free(seen); return;
			}
			if (seen[off]) {
				printf("  K=%u N=%u: (%u,%u) collides at %zu\n",
				       k, n, nn, kk, off);
				fail++; free(seen); return;
			}
			seen[off] = 1;
		}
	free(seen);
}

/*
 * The offset function against charsiu_pack_weights, which is what the int4 and
 * int8 model paths have always used. src[n][k] carries a marker that names its
 * own cell, so a permutation of equal bytes cannot pass.
 */
static void against_packer(unsigned k, unsigned n)
{
	struct charsiu_matmul mm = { 1, k, n, CHARSIU_INT8, CHARSIU_FP16 };
	size_t bytes = charsiu_weight_bytes(&mm);
	uint8_t *src = malloc((size_t)n * k);
	uint8_t *bulk = calloc(bytes, 1), *one = calloc(bytes, 1);
	unsigned nn, kk;

	if (!src || !bulk || !one) { printf("  out of memory\n"); fail++; goto out; }
	for (nn = 0; nn < n; nn++)
		for (kk = 0; kk < k; kk++)
			src[(size_t)nn * k + kk] =
				(uint8_t)(nn * 31u + kk * 7u + 1u);

	charsiu_pack_weights(&mm, src, bulk);
	/* the same bytes, one cell at a time, through the offset function --
	 * with the SAME bias the bulk packer applies, because the bias is part
	 * of what a caller writing in place has to reproduce */
	for (nn = 0; nn < n; nn++)
		for (kk = 0; kk < k; kk++) {
			size_t off = charsiu_w8_offset(&mm, nn, kk);

			if (off == (size_t)-1 || off >= bytes) {
				printf("  K=%u N=%u: (%u,%u) out of range\n",
				       k, n, nn, kk);
				fail++; goto out;
			}
			one[off] = (uint8_t)(src[(size_t)nn * k + kk] - 0x80);
		}
	if (memcmp(bulk, one, bytes)) {
		size_t i;

		for (i = 0; i < bytes && bulk[i] == one[i]; i++)
			;
		printf("  K=%u N=%u: byte %zu is %02x through the packer and "
		       "%02x through the offset\n", k, n, i, bulk[i], one[i]);
		fail++;
	}
out:
	free(src); free(bulk); free(one);
}

/*
 * ⚠ THE PROPERTY THE KV SURFACE RESTS ON. A surface is allocated at its final
 * width and read at whatever width the prompt has reached, so the offset of
 * (n, k) must be the same in a buffer of N1 channels and one of N2 -- for
 * every n and k -- as long as both widths are multiples of the n group. If
 * that fails, a cache appended at one width and run at another is a different
 * permutation of the same bytes.
 *
 * ng is 32 for int8 where it is 16 for fp16, so the multiple this holds at is
 * NOT the one the fp16 surface uses. The second half of this checks that the
 * property really does break at 16, because a test that only passes says
 * nothing about where the edge is.
 */
static void width_stable(unsigned k, unsigned n1, unsigned n2, int expect)
{
	struct charsiu_matmul a = { 1, k, n1, CHARSIU_INT8, CHARSIU_FP16 };
	struct charsiu_matmul b = { 1, k, n2, CHARSIU_INT8, CHARSIU_FP16 };
	unsigned nn, kk, same = 1;

	for (nn = 0; nn < n1 && same; nn++)
		for (kk = 0; kk < k; kk++)
			if (charsiu_w8_offset(&a, nn, kk) !=
			    charsiu_w8_offset(&b, nn, kk)) {
				same = 0;
				break;
			}
	if ((int)same != expect) {
		printf("  K=%u: widths %u and %u %s, expected %s\n", k, n1, n2,
		       same ? "agree" : "differ",
		       expect ? "agree" : "differ");
		fail++;
	}
}

int main(void)
{
	unsigned ks[] = { 64, 128, 256, 1024 };
	unsigned ns[] = { 32, 64, 96, 864, 1024 };
	unsigned i, j;

	for (i = 0; i < sizeof(ks) / sizeof(ks[0]); i++)
		for (j = 0; j < sizeof(ns) / sizeof(ns[0]); j++) {
			injective(ks[i], ns[j]);
			against_packer(ks[i], ns[j]);
		}

	/* multiples of the int8 n group: stable */
	width_stable(64, 64, 1024, 1);
	width_stable(64, 864, 1024, 1);
	width_stable(128, 32, 256, 1);
	/* ⚠ and 16, which is the fp16 surface's rung, is NOT */
	width_stable(64, 16, 1024, 0);
	width_stable(64, 848, 1024, 0);

	printf("%s: charsiu_w8_offset, %d failures\n",
	       fail ? "FAIL" : "ok", fail);
	return fail ? 1 : 0;
}
