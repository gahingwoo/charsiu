// SPDX-License-Identifier: GPL-2.0-or-later
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */

/*
 * The NPU's number format, computed on the CPU.
 *
 * Step 2 of the plan moves a projection onto the NPU. Before any of that is
 * worth writing, one question has to be answered and it can be answered here:
 * does the format the hardware imposes -- signed int8 weights with one scale
 * per output channel, against an activation with ONE scale for the whole
 * vector -- cost the model its output?
 *
 * The CPU decode loop is the oracle, so the way to ask is to run the whole
 * model in exactly that format and compare the tokens.
 *
 * It is deliberately NOT fast. It holds a second copy of every routed tensor in
 * int8 and does a plain widening dot. What it produces is an accuracy answer
 * and the exact buffers a converter will later write to a file: q, scale, and
 * the per channel weight sum the coefficient buffer needs.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu_llm.h"

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
#include <arm_neon.h>
#endif

int npu_tensor_build(struct npu_tensor *t, const struct gguf_tensor *w)
{
	uint64_t n = w->ne[1], k = w->ne[0];
	float *row;
	double se = 0.0, sw = 0.0;

	memset(t, 0, sizeof(*t));
	t->n = n;
	t->k = k;
	t->q = malloc((size_t)n * k);
	t->scale = malloc((size_t)n * sizeof(float));
	t->wsum = malloc((size_t)n * sizeof(int32_t));
	row = malloc((size_t)k * sizeof(float));
	if (!t->q || !t->scale || !t->wsum || !row) {
		free(row);
		npu_tensor_free(t);
		return -1;
	}

	for (uint64_t r = 0; r < n; r++) {
		int8_t *dst = t->q + r * k;
		float amax = 0.0f, d, id;
		int32_t sum = 0;

		gguf_row_f32(w, r, row);
		for (uint64_t i = 0; i < k; i++) {
			float a = fabsf(row[i]);

			if (a > amax)
				amax = a;
		}
		/*
		 * SYMMETRIC, and 127 rather than 128: the hardware's operand is
		 * a byte biased by 128, so -128 has no positive partner and a
		 * scale built on it would make the largest weight unreachable in
		 * one direction.
		 */
		d = amax / 127.0f;
		id = d != 0.0f ? 1.0f / d : 0.0f;
		t->scale[r] = d;

		for (uint64_t i = 0; i < k; i++) {
			int v = (int)lrintf(row[i] * id);

			if (v > 127) v = 127;
			if (v < -127) v = -127;
			dst[i] = (int8_t)v;
			sum += v;
			{
				double e = (double)row[i] - (double)v * d;

				se += e * e;
				sw += (double)row[i] * (double)row[i];
			}
		}
		t->wsum[r] = sum;
	}

	free(row);
	t->rms_rel = sw > 0.0 ? sqrt(se / sw) : 0.0;
	return 0;
}

void npu_tensor_free(struct npu_tensor *t)
{
	if (!t)
		return;
	free(t->q);
	free(t->scale);
	free(t->wsum);
	memset(t, 0, sizeof(*t));
}

static int32_t idot(const int8_t *w, const int8_t *x, uint64_t n)
{
#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
	int32x4_t acc = vdupq_n_s32(0);
	uint64_t i = 0;
	int32_t s;

	for (; i + 16 <= n; i += 16) {
		int8x16_t a = vld1q_s8(w + i), b = vld1q_s8(x + i);

		acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(a),  vget_low_s8(b)));
		acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(a), vget_high_s8(b)));
	}
	s = vaddvq_s32(acc);
	for (; i < n; i++)
		s += (int32_t)w[i] * (int32_t)x[i];
	return s;
#else
	int32_t s = 0;

	for (uint64_t i = 0; i < n; i++)
		s += (int32_t)w[i] * (int32_t)x[i];
	return s;
#endif
}

void npu_matvec(const struct npu_tensor *t, const struct charsiu_act *a,
		float *y, uint64_t row0, uint64_t nrows)
{
	for (uint64_t r = 0; r < nrows; r++) {
		uint64_t n = row0 + r;

		y[n] = (float)idot(t->q + n * t->k, a->q1, t->k) * a->d1 * t->scale[n];
	}
}
