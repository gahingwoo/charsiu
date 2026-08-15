// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The int4 packer against the layout the board measured, on the host.
 *
 * One live weight at a time, 4096 of them, each checked to land at the byte and
 * the nibble that rounds 167 and 168 established and NOWHERE else. An off by one
 * in a packer is invisible in an output vector and costs a whole board round; it
 * is visible here in a second.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charsiu.h"
int main(void)
{
	struct charsiu_matmul mm = { 1, 64, 64, CHARSIU_INT4, CHARSIU_INT8 };
	uint8_t *src = calloc(64 * 64, 1), *dst = calloc(charsiu_weight_bytes(&mm), 1);
	unsigned n, k, bad = 0;

	/* one live weight at a time, and check it lands where rounds 167 and 168
	 * say: byte = k%32 within the row, nibble = (k%64)/32 */
	for (n = 0; n < 64; n++) {
		for (k = 0; k < 64; k++) {
			size_t off, want;
			unsigned hi;

			memset(src, 0, 64 * 64);
			src[n * 64 + k] = 0x7;
			charsiu_pack_weights(&mm, src, dst);
			want = (size_t)(n / 32) * 32 * 64 / 2 + (size_t)(n % 32) * 32 + (k % 32);
			hi = (k % 64) >= 32;
			if (dst[want] != (hi ? 0x70 : 0x07)) {
				if (bad < 5)
					printf("  n=%2u k=%2u expected byte %zu %s nibble, got %02x\n",
					       n, k, want, hi ? "high" : "low", dst[want]);
				bad++;
			}
			for (off = 0; off < charsiu_weight_bytes(&mm); off++)
				if (off != want && dst[off]) { bad++; break; }
		}
	}
	printf("  int4 packer: %u of %u placements wrong\n", bad, 64 * 64);
	return bad != 0;
}
