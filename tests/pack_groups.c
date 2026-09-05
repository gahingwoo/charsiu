// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The batched activation packer, split by groups, against the whole buffer it
 * replaces.
 *
 * charsiu_pack_input_f16_groups exists so the pool can have the packer: it is
 * 0.61 ms a row of a Llama prompt and 1.11 of a Qwen3 one, the largest piece
 * of the pack. A split that writes one byte differently would not fault; it
 * would feed the hardware a slightly wrong activation and come back as a
 * slightly wrong sentence, which is the failure this project has shipped
 * before and now tests for.
 *
 * So: the same shapes packed both ways, byte for byte, at several splits --
 * including splits that do not divide the group count evenly, and a source
 * stride wider than k, which is what a K slice of a wider row looks like.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charsiu.h"

static int fail;
static unsigned cases;

static void check(unsigned m, unsigned k, size_t stride, unsigned nsplit)
{
	struct charsiu_matmul mm = { m, k, 64, CHARSIU_FP16, CHARSIU_FP16 };
	size_t sz = (size_t)charsiu_entries_per_row(&mm) * 64 * m + 256;
	float *src = malloc((size_t)m * stride * sizeof(float));
	uint8_t *a = malloc(sz), *b = malloc(sz);
	unsigned ng, g, done;

	cases++;
	if (!src || !a || !b) { printf("  out of memory\n"); fail++; goto out; }
	for (size_t i = 0; i < (size_t)m * stride; i++)
		src[i] = (float)((int)(i % 251) - 125) * 0.0137f;
	memset(a, 0xa5, sz);
	memset(b, 0xa5, sz);

	charsiu_pack_input_f16_stride(&mm, src, stride, a, sz);

	ng = charsiu_pack_input_f16_ngroups(&mm);
	if (!ng) {
		/* the shape does not take the vector path; nothing to split */
		goto out;
	}
	charsiu_pack_input_f16_edges(&mm, src, stride, b, sz);
	done = 0;
	for (g = 0; g < nsplit && done < ng; g++) {
		unsigned per = (ng + nsplit - 1) / nsplit;
		unsigned n = ng - done < per ? ng - done : per;

		charsiu_pack_input_f16_groups(&mm, src, stride, b, done, n);
		done += n;
	}
	if (done != ng) {
		printf("  m=%u k=%u split=%u covered %u of %u groups\n",
		       m, k, nsplit, done, ng);
		fail++;
		goto out;
	}
	if (memcmp(a, b, sz)) {
		size_t i;

		for (i = 0; i < sz && a[i] == b[i]; i++)
			;
		printf("  m=%u k=%u stride=%zu split=%u differ at byte %zu\n",
		       m, k, stride, nsplit, i);
		fail++;
	}
out:
	free(src); free(a); free(b);
}

int main(void)
{
	static const unsigned ms[] = { 2, 4, 8, 32, 48, 64, 80 };
	static const unsigned ks[] = { 8, 64, 512, 1024, 1536, 2048, 3072 };
	static const unsigned splits[] = { 1, 2, 3, 4, 7, 8, 16 };
	unsigned a, b, c;

	for (a = 0; a < sizeof(ms) / sizeof(*ms); a++)
		for (b = 0; b < sizeof(ks) / sizeof(*ks); b++)
			for (c = 0; c < sizeof(splits) / sizeof(*splits); c++) {
				check(ms[a], ks[b], ks[b], splits[c]);
				/* a K slice of a wider row: the stride is the
				 * whole row and the slice is part of it */
				check(ms[a], ks[b], (size_t)ks[b] + 512,
				      splits[c]);
			}
	/* and a k that is not a multiple of the atom, where the edges carry a
	 * remainder the group loop never sees */
	for (a = 0; a < sizeof(ms) / sizeof(*ms); a++) {
		check(ms[a], 1000, 1000, 4);
		check(ms[a], 1023, 2048, 3);
		check(ms[a], 12, 12, 2);
	}
	printf("  packer group split: %d of %u cases wrong\n", fail, cases);
	return fail ? 1 : 0;
}
