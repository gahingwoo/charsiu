/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * THE GREEDY VERIFY ONLY WANTS ONE NUMBER A ROW, AND IT WAS BUILDING 151936.
 *
 * A speculative pass scores each of its rows by llama_argmax over the output
 * head, and before it gets there the head's answer goes through three passes
 * over n x n_vocab floats: the read back writes them, the tail per channel
 * scale (an ungrouped or int8 head) reads and rewrites them, and the final
 * softcap (gemma) reads and rewrites them again. Then the argmax reads them a
 * fourth time. Only the index survives.
 *
 * This is the last three as one read: each element is scaled and capped by
 * the same operations, in the same order, as tail_scale_rows and softcap_rows
 * would have stored it, and compared the way llama_argmax compares -- first
 * index of the strict maximum, `v > best`, so a NaN at index 0 holds and a NaN
 * later never wins. The value compared is bit for bit the value the three
 * passes would have left in memory:
 *
 *   - tail_scale_rows is y * scale[j], one IEEE single multiply, vector or
 *     scalar alike; here it is the scalar one.
 *   - softcap_rows is tanhf(y / c) * c on the scaled value.
 *   - there is no add anywhere for a compiler to contract into an fma.
 *
 * NOT FUSED INTO THE READ BACK ITSELF, and that is deliberate: an output
 * channel's value only exists once every K slice of it has been accumulated,
 * on whichever device each slice was dealt to, so the gather still writes Y.
 * What this removes is every pass after it.
 *
 * In a header of its own so tests/argmax_fused.c can hold it against the
 * three passes it replaces on a host with no NPU.
 */
#ifndef CHARSIU_ARGMAXROW_H
#define CHARSIU_ARGMAXROW_H
#include <math.h>
#include <stdint.h>

/*
 * ONE ELEMENT, exactly as the passes it replaces would have stored it. The
 * branches are on loop invariants and the three loops below are the same
 * function specialised by hand: -O2 does not unswitch.
 */
static inline float charsiu_argmax_val(float v, const float *scale,
				       uint32_t i, float cap)
{
	if (scale)
		v *= scale[i];
	if (cap > 0.0f)
		v = tanhf(v / cap) * cap;
	return v;
}

/* scale may be NULL (a grouped head is already scaled); cap <= 0 is none */
static inline int32_t charsiu_argmax_scaled(const float *y, const float *scale,
					    uint32_t n, float cap)
{
	int32_t best = 0;
	float bv;
	uint32_t i;

	if (!n)
		return 0;
	bv = charsiu_argmax_val(y[0], scale, 0, cap);
	if (cap > 0.0f) {
		for (i = 1; i < n; i++) {
			float v = charsiu_argmax_val(y[i], scale, i, cap);

			if (v > bv) {
				bv = v;
				best = (int32_t)i;
			}
		}
	} else if (scale) {
		for (i = 1; i < n; i++) {
			float v = y[i] * scale[i];

			if (v > bv) {
				bv = v;
				best = (int32_t)i;
			}
		}
	} else {
		for (i = 1; i < n; i++)
			if (y[i] > bv) {
				bv = y[i];
				best = (int32_t)i;
			}
	}
	return best;
}

#endif
