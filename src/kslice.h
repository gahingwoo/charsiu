/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * How one tensor's contraction axis is cut into slices, as arithmetic alone.
 *
 * ⚠⚠ THE OLD RULE AND WHY IT COST SOMETHING. Every slice took KMAX and the
 * last one took the remainder, so Phi-3.5's K = 3072 at KMAX 2048 is
 * 2048 + 1024 and Qwen2.5's K = 8960 is 2048 x4 + 768. Two costs, and phase 9
 * on 2026-09-05 measured the second:
 *
 *   - the two cores get unequal work on every K sliced tensor, and the fence
 *     waits for the wide one;
 *   - deal_pick() balances by ACCUMULATED load, so a wide slice and a narrow
 *     one leave the loads uneven and the next tensor gets the opposite
 *     assignment. The map flips per tensor, so a follower's slice always
 *     lands on the device that does not hold it. Phi-3.5 reused its packed
 *     input 0 times out of 2304 asks and gemma4 0 of 528, and every one of
 *     those misses was counted "the K slices are on the other device". They
 *     are the only two models here whose K is above KMAX.
 *
 * Equal slices cost deal_pick the same load twice, so the assignment is
 * stable by construction and the cores get equal work: one change for both,
 * without touching the dealer or the reuse key.
 *
 * In a header of its own so the tiling can be checked on a desk with no NPU
 * (tests/even_ks.c, under make test) -- that the slices tile K exactly, that
 * none is wider than KMAX, and that the even ones really are even. That the
 * hardware agrees with the shapes is the board's.
 */
#ifndef CHARSIU_KSLICE_H
#define CHARSIU_KSLICE_H
#include <stdint.h>

/*
 * The width every slice but the last takes when the slices are equal: K over
 * ks rounded up to the feature atom. 0 means "not equal slicing" and the
 * caller falls back to the KMAX rule -- which is KFIT (whose last slice is
 * deliberately WIDER than KMAX, so evening it out would drop the tail), a
 * single slice, or a rounding that would leave the last slice empty.
 */
static inline unsigned charsiu_even_k_base(uint64_t k, unsigned ks,
					   unsigned kmax, int even, int kfit)
{
	/* 32 is the int4 feature atom and a multiple of int8's 16 */
	const unsigned atom = 32;
	uint64_t base;

	if (!even || kfit || ks < 2 || !kmax)
		return 0;
	base = ((k + ks - 1) / ks + atom - 1) / atom * atom;
	if (base > kmax || (uint64_t)(ks - 1) * base >= k)
		return 0;
	return (unsigned)base;
}

/* the first column of slice ki */
static inline unsigned charsiu_slice_k0(uint64_t k, unsigned ks, unsigned ki,
					unsigned kmax, int even, int kfit)
{
	unsigned base = charsiu_even_k_base(k, ks, kmax, even, kfit);

	return (unsigned)((uint64_t)ki * (base ? base : kmax));
}

/* the width of slice ki */
static inline unsigned charsiu_slice_kw(uint64_t k, unsigned ks, unsigned ki,
					unsigned kmax, int even, int kfit)
{
	unsigned base = charsiu_even_k_base(k, ks, kmax, even, kfit);
	unsigned step = base ? base : kmax;

	/* ⚠ THE LAST SLICE TAKES WHATEVER IS LEFT. Under KFIT that is more
	 * than KMAX, and clamping it here would drop the tail of the tensor
	 * without a word. */
	return ki + 1 == ks ? (unsigned)(k - (uint64_t)ki * step) : step;
}

#endif /* CHARSIU_KSLICE_H */
