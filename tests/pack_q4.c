// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * charsiu_pack_weights_rows_q4 against the path it replaces, on the host.
 *
 * The reference is what staging did before: every code of the slice
 * unpacked to a byte (npudev's pack_rows, v & 0xf) and the bytes packed by
 * charsiu_pack_weights_rows. The direct pack reads the packed codes where
 * they lie. Random codes, odd and even K, K not a multiple of 32, N not a
 * multiple of 16, slices at even offsets inside a wider tensor, and the
 * rows split into ranges as the pool splits them. Both destinations start
 * from the same non-zero fill, so a byte either side writes and the other
 * does not shows up as well as a byte they write differently.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charsiu.h"

static unsigned nib(const uint8_t *row, unsigned i)
{
	return (i & 1) ? (unsigned)(row[i >> 1] >> 4) : (unsigned)(row[i >> 1] & 0xf);
}

static uint32_t rng = 12345;
static uint32_t rnd(void)
{
	rng = rng * 1103515245u + 12345u;
	return rng >> 8;
}

static int one(unsigned kt, unsigned nt, unsigned k0, unsigned k, unsigned n0,
	       unsigned n, unsigned split)
{
	struct charsiu_matmul mm = { 1, k, n, CHARSIU_INT4, CHARSIU_FP16 };
	size_t qs = ((size_t)kt + 1) / 2, bytes = charsiu_weight_bytes(&mm);
	uint8_t *q = malloc(qs * nt), *scr = malloc((size_t)n * k);
	uint8_t *ref = malloc(bytes), *got = malloc(bytes);
	unsigned r, c, at;
	int bad = 0;

	for (size_t i = 0; i < qs * nt; i++)
		q[i] = (uint8_t)rnd();
	for (r = 0; r < n; r++)
		for (c = 0; c < k; c++)
			scr[(size_t)r * k + c] =
				(uint8_t)nib(q + (size_t)(n0 + r) * qs, k0 + c);
	memset(ref, 0xa5, bytes);
	memset(got, 0xa5, bytes);
	charsiu_pack_weights_rows(&mm, scr, ref, 0, n);
	for (at = 0; at < n; at += split) {
		unsigned nr = n - at < split ? n - at : split;

		if (charsiu_pack_weights_rows_q4(&mm, q + (size_t)n0 * qs + k0 / 2,
						 qs, got, at, nr)) {
			printf("pack_q4: refused K=%u N=%u\n", k, n);
			bad = 1;
			goto out;
		}
	}
	if (memcmp(ref, got, bytes)) {
		size_t i = 0;

		while (ref[i] == got[i])
			i++;
		printf("pack_q4: K=%u N=%u (tensor %ux%u, k0 %u, n0 %u, split %u): "
		       "byte %zu of %zu is 0x%02x, want 0x%02x\n", k, n, kt, nt, k0,
		       n0, split, i, bytes, got[i], ref[i]);
		bad = 1;
	}
out:
	free(q); free(scr); free(ref); free(got);
	return bad;
}

int main(void)
{
	static const unsigned ks[] = { 1, 2, 7, 31, 32, 33, 63, 64, 100, 511, 1024, 1025, 2048 };
	static const unsigned ns[] = { 1, 2, 3, 15, 16, 17, 31, 64, 130 };
	static const unsigned sp[] = { 1, 5, 16, 1000 };
	unsigned a, b, c, bad = 0, runs = 0;

	for (a = 0; a < sizeof(ks) / sizeof(ks[0]); a++)
		for (b = 0; b < sizeof(ns) / sizeof(ns[0]); b++)
			for (c = 0; c < sizeof(sp) / sizeof(sp[0]); c++) {
				unsigned k = ks[a], n = ns[b];

				/* the slice alone, then inside a wider tensor at
				 * an even K offset and a row offset */
				bad += one(k, n, 0, k, 0, n, sp[c]);
				bad += one(k + 2 * 37 + 1, n + 9, 2 * 37, k, 5, n, sp[c]);
				runs += 2;
			}
	/* the real shapes: K slices of 1024 out of 2048 and 8192 */
	bad += one(2048, 512, 1024, 1024, 0, 512, 37);
	bad += one(8192, 64, 7168, 1024, 0, 64, 16);
	runs += 2;
	if (bad) {
		printf("pack_q4: %u of %u FAILED\n", bad, runs);
		return 1;
	}
	printf("pack_q4: %u shapes byte for byte\n", runs);
	return 0;
}
