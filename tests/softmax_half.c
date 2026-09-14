// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * softmax_scaled followed by the fp16 conversion, against softmax_scaled_half,
 * bit for bit.
 *
 * ⚠⚠ WHY THIS EXISTS. Two copies of one piece of arithmetic is the hazard this
 * tree keeps meeting: the fp16 weight layout, the accumulator read order, the
 * activation packer. The fused softmax is a second copy by construction -- it
 * has to be, because the point of it is to write halves instead of floats --
 * so the thing to hold is not "one copy" but "they agree".
 *
 * They must agree to the LAST BIT, not to a tolerance. The value that reaches
 * the hardware is a half, and a half is either the same half or a different
 * number; a softmax that is right to 1e-7 and rounds the other way on one
 * element of a row changes a token.
 *
 * Both arms are exercised: fast_softmax's NEON path and the scalar one, at
 * lengths that straddle its n >= 8 and its 4 and 8 wide tails.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "charsiu.h"

int charsiu_softmax_half_selftest(void);

int main(void)
{
	int rc = charsiu_softmax_half_selftest();

	printf("%s: softmax_scaled_half against softmax_scaled, %d failures\n",
	       rc ? "FAIL" : "ok", rc);
	return rc ? 1 : 0;
}
