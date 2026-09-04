// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The strided input packer against the gather it replaces.
 *
 * A K slice is columns [k0, k0 + sk) of every row. The batched path used to
 * copy them into a scratch so the packer could read contiguous rows, which is
 * fourteen bytes moved per six packed; it now hands the packer the caller's
 * matrix and a row stride. The bytes must not move: same shapes, same values,
 * same layout, and the scalar control (CHARSIU_NPU_PLAIN) must agree too.
 */
/* the strided packer must write byte for byte what the gather plus the
 * contiguous packer wrote, at every shape the batched path uses */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/charsiu.h"
int main(void)
{
	unsigned ms[] = {1,2,4,8,14,24,30,32,64,80}, ks[] = {1,7,8,15,64,768,1024,1536,2048,3072,4096};
	unsigned bad = 0, n = 0;
	for (unsigned a = 0; a < sizeof(ms)/sizeof(*ms); a++)
	for (unsigned b = 0; b < sizeof(ks)/sizeof(*ks); b++)
	for (unsigned k0 = 0; k0 <= 2048; k0 += 1024) {
		unsigned m = ms[a], sk = ks[b], K = sk + k0 + 512;   /* a wider row than the slice */
		size_t nx = (size_t)m * K, nd = (size_t)sk * m * 2 + 4096;
		float *X = malloc(nx * 4), *scr = malloc((size_t)m * sk * 4);
		uint8_t *d1 = malloc(nd), *d2 = malloc(nd);
		struct charsiu_matmul mm = { m, sk, 64, CHARSIU_INT4, CHARSIU_FP16 };
		for (size_t i = 0; i < nx; i++)
			X[i] = (float)(((i * 2654435761u) >> 7) & 0xffff) / 4096.0f - 8.0f;
		memset(d1, 0xa5, nd); memset(d2, 0x5a, nd);
		for (unsigned r = 0; r < m; r++)
			memcpy(scr + (size_t)r * sk, X + (size_t)r * K + k0, sk * 4);
		charsiu_pack_input_f16(&mm, scr, d1, nd);
		charsiu_pack_input_f16_stride(&mm, X + k0, K, d2, nd);
		n++;
		if (memcmp(d1, d2, nd)) {
			printf("  pack stride: m=%u k=%u k0=%u DIFFERS\n", m, sk, k0);
			bad++;
		}
		free(X); free(scr); free(d1); free(d2);
	}
	printf("  pack stride: %u of %u shapes wrong\n", bad, n);
	return bad != 0;
}
