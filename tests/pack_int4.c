// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The int4 packer against ITS OWN layout, on the host, which is not the same as
 * against the hardware's.
 *
 * One live weight at a time, 4096 of them, each checked to land at the byte and
 * the nibble the packer intends and nowhere else. That catches an off by one,
 * which is invisible in an output vector and costs a whole board round.
 *
 * IT DOES NOT CATCH THE LAYOUT BEING WRONG, and the layout IS wrong: rounds 167
 * and 168 were withdrawn in 171, and a sparse map since measures the hardware's
 * int4 row at 8 bytes against the 32 this packs. So a pass here means the packer
 * does what it says, not that what it says is right. It is kept because when the
 * real layout is known this is the check that it was implemented without a slip.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charsiu.h"
int main(void)
{
	struct charsiu_matmul mm = { 1, 64, 64, CHARSIU_INT4, CHARSIU_INT8 };
	uint8_t *src = calloc(64 * 64, 1), *dst = calloc(charsiu_weight_bytes(&mm), 1);
	unsigned n, k, bad = 0, checked = 0;

	/*
	 * The MEASURED row: channel n starts at (n / 32) * 512 + (n % 32) * 8 and
	 * runs eight bytes. Only k = 0 to 15 are placed, because only that much
	 * of the map is data.
	 */
	for (n = 0; n < 64; n++) {
		for (k = 0; k < 16; k++) {
			size_t row = (size_t)(n / 32) * 512 + (size_t)(n % 32) * 8;
			size_t want = row + k / 2, off;
			unsigned high = k & 1, live = 0;

			memset(src, 0, 64 * 64);
			src[n * 64 + k] = 0x7;
			charsiu_pack_weights(&mm, src, dst);
			checked++;
			if (dst[want] != (high ? 0x70 : 0x07)) {
				if (bad < 5)
					printf("  n=%2u k=%2u expected byte %zu %s nibble, got %02x\n",
					       n, k, want, high ? "high" : "low", dst[want]);
				bad++;
				continue;
			}
			for (off = 0; off < charsiu_weight_bytes(&mm); off++)
				if (off != want && dst[off]) { bad++; break; }
		}
	}
	printf("  int4 packer: %u of %u placements wrong (k < 16 only, which is\n"
	       "  all the map is data for)\n", bad, checked);
	return bad != 0;
}
