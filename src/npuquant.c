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
#include <string.h>
#include <stdio.h>
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

	/*
	 * ⚠ THE ACCURACY QUESTION int4 HAS TO ANSWER BEFORE IT IS WORTH WIRING
	 * IN. The hardware's coefficient buffer carries ONE multiplier per
	 * output channel, so charsiu's NPU weights are quantised per channel --
	 * fine at eight bits, and q4_0 uses a scale every 32 weights precisely
	 * because four bits per channel is not.
	 *
	 * But K is sliced anyway (KMAX), and acc_out makes a K split exact, so
	 * a slice can carry its own scale for free. CHARSIU_NPU_W4_GROUP asks
	 * what that buys: the weights are quantised in groups of that many k,
	 * with a scale each, which is what the runtime would actually get.
	 * CHARSIU_NPU_W4 alone is one scale for the whole row, the worst case.
	 */
	/*
	 * CHARSIU_NPU_W4_ONLY narrows int4 to the tensors whose name contains a
	 * substring, so "ffn" puts the feed forward -- which is 65% of the
	 * bytes a token moves -- at four bits and leaves attention at eight.
	 * Most of the saving for a fraction of the error, if the error turns
	 * out to live in attention.
	 */
	const char *w4only = getenv("CHARSIU_NPU_W4_ONLY");
	unsigned bits = getenv("CHARSIU_NPU_W4")
		&& (!w4only || strstr(w->name, w4only)) ? 4 : 8;
	uint64_t grp = getenv("CHARSIU_NPU_W4_GROUP")
		? (uint64_t)atoi(getenv("CHARSIU_NPU_W4_GROUP")) : k;
	uint64_t ngrp;
	float qmax = bits == 4 ? 7.0f : 127.0f;

	if (grp == 0 || grp > k)
		grp = k;
	ngrp = (k + grp - 1) / grp;

	/*
	 * ⚠ THE k FACTOR, and why it is free.
	 *
	 * charsiu reads the RAW int32 accumulator (acc_out), so the hardware's
	 * per channel multiplier is not in the path at all: the dequantise
	 * happens on the CPU. The only real constraint is that one job's
	 * accumulator sums all of K under one scale a channel.
	 *
	 * But a factor that depends on k ALONE cancels inside the product. Put
	 * c_k into the weights as a divide and into the activation as a
	 * multiply and the accumulator is unchanged, while the weights lose
	 * whatever part of their spread across k is common to every channel --
	 * which is the part one scale a row cannot cover. That is the AWQ and
	 * SmoothQuant trick, and here it costs one multiply a k on a vector of
	 * 2048 and nothing on the hardware.
	 *
	 * CHARSIU_NPU_AWQ is the exponent, 0 for off and 0.5 for the usual
	 * square root balance.
	 */
	double alpha = getenv("CHARSIU_NPU_AWQ")
		? atof(getenv("CHARSIU_NPU_AWQ")) : 0.0;

	memset(t, 0, sizeof(*t));
	t->n = n;
	t->k = k;
	t->kgroup = grp;
	t->q = malloc((size_t)n * k);
	t->scale = malloc((size_t)n * ngrp * sizeof(float));
	t->wsum = malloc((size_t)n * sizeof(int32_t));
	row = malloc((size_t)k * sizeof(float));
	if (!t->q || !t->scale || !t->wsum || !row) {
		free(row);
		npu_tensor_free(t);
		return -1;
	}

	if (alpha != 0.0) {
		double *col = calloc(k, sizeof(*col));
		double gm = 0.0;

		if (!col) { free(row); npu_tensor_free(t); return -1; }
		for (uint64_t r = 0; r < n; r++) {
			gguf_row_f32(w, r, row);
			for (uint64_t i = 0; i < k; i++)
				col[i] += fabs((double)row[i]);
		}
		t->kscale = malloc((size_t)k * sizeof(float));
		if (!t->kscale) { free(col); free(row); npu_tensor_free(t); return -1; }
		for (uint64_t i = 0; i < k; i++) {
			double v = col[i] / (double)n;

			col[i] = v > 0.0 ? pow(v, alpha) : 1.0;
			gm += log(col[i]);
		}
		gm = exp(gm / (double)k);            /* keep the mean factor at 1 */
		for (uint64_t i = 0; i < k; i++)
			t->kscale[i] = (float)(col[i] / gm);
		free(col);
	}

	for (uint64_t r = 0; r < n; r++) {
		int8_t *dst = t->q + r * k;
		int32_t sum = 0;

		gguf_row_f32(w, r, row);
		if (t->kscale)
			for (uint64_t i = 0; i < k; i++)
				row[i] /= t->kscale[i];
		for (uint64_t g = 0; g < ngrp; g++) {
		uint64_t lo = g * grp, hi = lo + grp < k ? lo + grp : k;
		float amax = 0.0f, d, id;

		for (uint64_t i = lo; i < hi; i++) {
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
		/*
		 * ⚠ USE ALL SIXTEEN LEVELS. amax/7 is symmetric and throws away
		 * -8, which is one level of the sixteen a nibble has -- and
		 * q4_0, the yardstick this is measured against, takes the
		 * element of largest magnitude and divides by -8 so the range is
		 * [-8, 7]. Round 335's first int4 numbers were a fifteen level
		 * quantiser being compared against a sixteen level one.
		 * CHARSIU_NPU_W4_SYM restores the symmetric version.
		 */
		if (bits == 4 && !getenv("CHARSIU_NPU_W4_SYM")) {
			float vmax = 0.0f;

			for (uint64_t i = lo; i < hi; i++)
				if (fabsf(row[i]) > fabsf(vmax))
					vmax = row[i];
			d = vmax / -8.0f;
			/*
			 * ⚠ absmax IS THE WRONG SCALE FOR FOUR BITS, and it is
			 * the cheapest thing to fix. One outlier in two thousand
			 * weights sets the step for all of them, so every other
			 * weight rounds into a grid that is far too coarse.
			 *
			 * Search the clip instead: try the scale shrunk by a
			 * factor and keep whichever minimises the squared error
			 * of the row. Nothing is stored -- the chosen d is the
			 * per channel scale that would have been stored anyway
			 * -- and nothing changes on the hardware. This is the
			 * cheap half of what a calibrating quantiser does.
			 *
			 * ⚠ MEASURED AND IT IS WORSE: KL went 0.0989 to 0.2084
			 * and 0.3660 to 0.5535 on two prompts. Minimising the
			 * WEIGHT error clips exactly the large weights that
			 * carry the output, which is why AWQ and GPTQ optimise
			 * the OUTPUT error against calibration activations
			 * instead. Off by default, kept as the control that
			 * says a weight space objective is the wrong one.
			 */
			if (getenv("CHARSIU_NPU_W4_CLIP")) {
				double bestе = -1.0;
				float bestd = d;
				int ci;

				for (ci = 0; ci <= 20; ci++) {
					float dc = d * (1.0f - 0.025f * ci);
					double err = 0.0;

					if (dc == 0.0f)
						continue;
					for (uint64_t i = lo; i < hi; i++) {
						int v = (int)lrintf(row[i] / dc);
						double e;

						if (v > 7) v = 7;
						if (v < -8) v = -8;
						e = (double)row[i] - (double)v * dc;
						err += e * e;
					}
					if (bestе < 0.0 || err < bestе) {
						bestе = err;
						bestd = dc;
					}
				}
				d = bestd;
			}
		} else {
			d = amax / qmax;
		}
		id = d != 0.0f ? 1.0f / d : 0.0f;
		t->scale[r * ngrp + g] = d;

		for (uint64_t i = lo; i < hi; i++) {
			int v = (int)lrintf(row[i] * id);

			if (bits == 4 && !getenv("CHARSIU_NPU_W4_SYM")) {
				if (v > 7) v = 7;
				if (v < -8) v = -8;
			} else {
				if (v > (int)qmax) v = (int)qmax;
				if (v < -(int)qmax) v = -(int)qmax;
			}
			dst[i] = (int8_t)v;
			sum += v;
			{
				double e = (double)row[i] - (double)v * d;

				se += e * e;
				sw += (double)row[i] * (double)row[i];
			}
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
	free(t->kscale);
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

		uint64_t grp = t->kgroup ? t->kgroup : t->k;
		uint64_t ngrp = (t->k + grp - 1) / grp;
		double acc = 0.0;
		const int8_t *aq = a->q1;
		int8_t *free_after = NULL;
		float ad = a->d1;

		/*
		 * The k factor rides on the ACTIVATION, so a tensor that has one
		 * needs its own quantisation of the same input vector. On the
		 * board this is one multiply a k before the pack; here it is
		 * done straight so the measurement is honest.
		 */
		if (t->kscale) {
			/*
			 * ⚠ PER CALL, NOT static. npu_matvec runs on the worker
			 * threads, and a static scratch buffer here was a race
			 * and a double free: three of the five exponents in the
			 * first sweep produced no output at all.
			 */
			int8_t *tmp = malloc((size_t)t->k);
			float amax = 0.0f;

			if (!tmp)
				return;
			for (uint64_t i = 0; i < t->k; i++) {
				float v = a->f[i] * t->kscale[i];

				if (fabsf(v) > amax) amax = fabsf(v);
			}
			ad = amax / 127.0f;
			for (uint64_t i = 0; i < t->k; i++) {
				int v = (int)lrintf(a->f[i] * t->kscale[i]
						    / (ad != 0.0f ? ad : 1.0f));

				if (v > 127) v = 127;
				if (v < -127) v = -127;
				tmp[i] = (int8_t)v;
			}
			aq = tmp;
			free_after = tmp;
		}

		for (uint64_t g = 0; g < ngrp; g++) {
			uint64_t lo = g * grp;
			uint64_t len = lo + grp < t->k ? grp : t->k - lo;

			acc += (double)idot(t->q + n * t->k + lo, aq + lo,
					    len) * t->scale[n * ngrp + g];
		}
		y[n] = (float)(acc * ad);
		free(free_after);
	}
}

static uint64_t npu_cal_calls(void)
{
	static long c = -1;

	if (c < 0) {
		const char *e = getenv("CHARSIU_NPU_CAL");

		c = e ? atol(e) : 32;
		if (c < 1)
			c = 1;
	}
	return (uint64_t)c;
}

int npu_out8_mode(void)
{
	static int m = -1;

	if (m < 0) {
		const char *e = getenv("CHARSIU_NPU_OUT8");

		m = e ? atoi(e) : 0;
	}
	return m;
}

void npu_quantise_output(struct npu_tensor *t, float *y, uint64_t n, int mode)
{
	float amax = 0.0f, d, id;
	uint64_t clipped = 0;

	if (mode <= 0)
		return;

	for (uint64_t i = 0; i < n; i++) {
		float a = fabsf(y[i]);

		if (a > amax)
			amax = a;
	}

	if (t->amax_lo <= 0.0f || amax < t->amax_lo)
		t->amax_lo = amax;
	if (amax > t->amax_hi)
		t->amax_hi = amax;

	if (mode >= 2) {
		/*
		 * A coefficient buffer holds ONE set of numbers for the whole
		 * run, so the scale cannot look at the vector it is about to
		 * quantise.
		 *
		 * mode 2 freezes on the FIRST call, which is the worst possible
		 * calibration and not a fair test of the idea. mode 3 tracks the
		 * running maximum over the first CHARSIU_NPU_CAL calls (32 by
		 * default, so several tokens) and freezes after that, which is
		 * what a calibration pass would actually produce.
		 */
		uint64_t cal = mode >= 3 ? npu_cal_calls() : 1;

		if (t->out_calls < cal) {
			float want = amax / 127.0f;

			if (want > t->out_scale)
				t->out_scale = want;
		}
		d = t->out_scale;
	} else {
		d = amax / 127.0f;
	}

	id = d != 0.0f ? 1.0f / d : 0.0f;
	for (uint64_t i = 0; i < n; i++) {
		int v = (int)lrintf(y[i] * id);

		if (v > 127) { v = 127; clipped++; }
		if (v < -127) { v = -127; clipped++; }
		y[i] = (float)v * d;
	}
	t->out_clip += (double)clipped / (double)n;
	t->out_calls++;
}

/*
 * What a FIXED output scale is up against.
 *
 * The scale a coefficient buffer holds has to cover the largest vector the
 * model will ever produce, and every other vector then uses 127 divided by how
 * much larger that one was. This prints that ratio.
 */
void npu_report(const struct npu_tensor *t, unsigned count)
{
	fprintf(stderr, "\n%-28s  %10s %10s %8s  %8s\n",
		"tensor", "min |y|", "max |y|", "ratio", "clip%");
	for (unsigned i = 0; i < count; i++) {
		const struct npu_tensor *x = &t[i];

		if (!x->out_calls)
			continue;
		fprintf(stderr, "%-28s  %10.3f %10.3f %8.1f  %8.3f\n",
			x->name, (double)x->amax_lo, (double)x->amax_hi,
			x->amax_lo > 0.0f ? (double)(x->amax_hi / x->amax_lo) : 0.0,
			x->out_clip / (double)x->out_calls * 100.0);
	}
}
