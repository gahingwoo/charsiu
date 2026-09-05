/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * How one tensor's contraction axis is cut into slices, as arithmetic alone.
 *
 * ⚠⚠ THE OLD RULE AND WHY IT COST SOMETHING. Every slice took KMAX and the
 * last one took the remainder, so at KMAX 2048 a K of 3072 is 2048 + 1024 and
 * Qwen2.5's 8960 is 2048 x4 + 768. Two costs, and phase 9 on 2026-09-05
 * measured the second:
 *
 *   - the two cores get unequal work on every K sliced tensor, and the fence
 *     waits for the wide one;
 *   - deal_pick() balances by ACCUMULATED load, so a wide slice and a narrow
 *     one leave the loads uneven and the next tensor gets the opposite
 *     assignment. The map flips per tensor, so a follower's slice always
 *     lands on the device that does not hold it. Phi-3.5 reused its packed
 *     input 0 times out of 2304 asks and gemma4 0 of 528, and every one of
 *     those misses was counted "the K slices are on the other device".
 *
 * Equal slices cost deal_pick the same load twice, so the assignment is
 * stable by construction and the cores get equal work: one change for both,
 * without touching the dealer or the reuse key.
 *
 * ⚠⚠ MEASURED, AND IT IS WORTH ALMOST NOTHING. Board, 2026-09-05, phase 2
 * clean on nine models and phase 9 with it on:
 *
 *   gemma4      528 misses -> 0,   prompt 30110 -> 29884 ms   (-0.8%)
 *   Phi-3.5    2304 misses -> 2304, unchanged
 *
 * -0.8% is inside the run to run spread, so eliminating a model's reuse
 * misses entirely bought nothing measurable. That is the answer to "is the
 * reuse worth chasing", and it is no. OFF by default, kept because the
 * arithmetic is right and the next person should not re-derive it.
 *
 * ⚠ AND WHY PHI-3.5 DID NOT MOVE, WHICH IS NOT WHAT THE FIRST COMMIT SAID.
 * llama.c picks KMAX itself: 1024 unless NO n_ff of the model is a multiple
 * of a candidate. Phi-3.5's n_ff is 8192, a multiple of both, so it runs at
 * KMAX 1024 -- where its K of 3072 is already 1024 x3 and its 8192 is
 * 1024 x8. Both were already even and this changes nothing for it. Its
 * misses are the case equal widths CANNOT fix: three slices across two
 * devices is two and one, so the accumulated load still alternates and the
 * map still flips. An even slice COUNT is a different fix from an even slice
 * WIDTH, and nothing here does the first.
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
