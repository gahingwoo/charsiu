// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The int4 packer against ITS OWN layout, on the host.
 *
 * One live weight at a time, 4096 of them at K = 64, N = 64, each checked to
 * land at the byte and the nibble the packer intends and nowhere else. That
 * catches an off by one, which is invisible in an output vector and costs a
 * whole board round.
 *
 * The layout is the one the board runs: for channel n and input k the nibble
 * sits at
 *
 *   (n / 16) * 512 * (K / 32)  +  (n % 16) * 32  +  (k / 32) * 512  +  k % 32
 *
 * blocks of 32 k for each of 16 channels, a 512-nibble block per (channel
 * group, k block), the low nibble of a byte for an even k. It replaced the
 * round 171 sparse map in round 280, and int4 has decoded bit-identical to
 * the CPU on the card since (14.70 tok/s on two cores).
 *
 * ⚠ THIS SAT RED FOR THIRTEEN DAYS after round 280: its map was the withdrawn
 * one and nobody ran make test, so a red here had come to mean nothing. It
 * means something again: a pass says the packer does what the board verified,
 * and a failure says the PACKER moved, since the hardware does not.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charsiu.h"

int main(void)
{
	struct charsiu_matmul mm = { 1, 64, 64, CHARSIU_INT4, CHARSIU_INT8 };
	size_t bytes = charsiu_weight_bytes(&mm);
	uint8_t *src = calloc(64 * 64, 1), *dst = calloc(bytes, 1);
	unsigned n, k, bad = 0, checked = 0;

	for (n = 0; n < 64; n++) {
		for (k = 0; k < 64; k++) {
			size_t nib = (size_t)(n / 16) * 512 * (64 / 32)
				   + (size_t)(n % 16) * 32
				   + (size_t)(k / 32) * 512 + (k % 32);
			size_t want = nib >> 1, off;
			unsigned high = nib & 1;

			memset(src, 0, 64 * 64);
			src[n * 64 + k] = 0x7;
			charsiu_pack_weights(&mm, src, dst);
			checked++;
			if (want >= bytes || dst[want] != (high ? 0x70 : 0x07)) {
				if (bad < 5)
					printf("  n=%2u k=%2u expected byte %zu %s nibble, got %02x\n",
					       n, k, want, high ? "high" : "low",
					       want < bytes ? dst[want] : 0);
				bad++;
				continue;
			}
			for (off = 0; off < bytes; off++)
				if (off != want && dst[off]) { bad++; break; }
		}
	}
	printf("  int4 packer: %u of %u placements wrong\n", bad, checked);
	return bad != 0;
}
