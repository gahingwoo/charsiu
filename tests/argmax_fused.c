// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The fused verify argmax (src/argmaxrow.h) against the three passes it
 * replaces, on a host with no NPU: the tail per channel scale exactly as
 * npudev.c's tail_scale_rows stores it (the NEON multiply where the build has
 * it), the softcap exactly as llama.c's softcap_rows stores it, then
 * llama_argmax over the stored row. The index must be the same on every row,
 * and not approximately: these are the cases where "the same maximum" is
 * easiest to get wrong.
 *
 *   - ties, planted, so the first index has to win in both
 *   - a softcap large enough to turn distinct values into equal ones: the
 *     uncapped argmax differs there and the capped one must not
 *   - NaN at index 0 (it holds: nothing is > NaN) and NaN later (never wins)
 *   - infinities, zero and negative scales, rows of length 1
 *
 * The negative control is the uncapped argmax on the saturating rows: it has
 * to DIFFER from the reference at least once, or the softcap cases are not
 * testing anything.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/argmaxrow.h"
#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
#include <arm_neon.h>
#endif

/* npudev.c tail_scale_rows, one row */
static void ref_scale(float *y, const float *scale, unsigned n)
{
	unsigned j = 0;

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
	for (; j + 4 <= n; j += 4)
		vst1q_f32(y + j, vmulq_f32(vld1q_f32(y + j),
					   vld1q_f32(scale + j)));
#endif
	for (; j < n; j++)
		y[j] *= scale[j];
}

/* llama.c softcap_rows */
static void ref_cap(float *y, unsigned n, float c)
{
	if (c <= 0.0f)
		return;
	for (unsigned i = 0; i < n; i++)
		y[i] = tanhf(y[i] / c) * c;
}

/* llama.c llama_argmax */
static int32_t ref_argmax(const float *logits, uint32_t n)
{
	int32_t best = 0;

	for (uint32_t i = 1; i < n; i++)
		if (logits[i] > logits[best])
			best = (int32_t)i;
	return best;
}

static uint64_t rng = 0x9e3779b97f4a7c15ull;

static float urand(void)
{
	rng ^= rng << 13;
	rng ^= rng >> 7;
	rng ^= rng << 17;
	return (float)((rng >> 40) & 0xffffff) / (float)0x1000000;
}

static float nrand(void)
{
	float u = urand() + 1e-7f, v = urand();

	return sqrtf(-2.0f * logf(u)) * cosf(6.2831853f * v);
}

static unsigned fails, cases, ctrl_differs;

static void check(const float *y, const float *scale, unsigned n, float cap,
		  const char *what)
{
	float *t = malloc((n ? n : 1) * sizeof(*t));
	int32_t want, got;

	memcpy(t, y, n * sizeof(*t));
	if (scale)
		ref_scale(t, scale, n);
	ref_cap(t, n, cap);
	want = n ? ref_argmax(t, n) : 0;
	got = charsiu_argmax_scaled(y, scale, n, cap);
	cases++;
	if (got != want) {
		if (fails < 10)
			fprintf(stderr, "argmax_fused: %s n=%u cap=%g scale=%s:"
				" got %d want %d\n", what, n, (double)cap,
				scale ? "yes" : "no", got, want);
		fails++;
	}
	free(t);
}

int main(void)
{
	static const unsigned ns[] = { 1, 2, 3, 4, 5, 7, 8, 9, 31, 1000,
				       151936 };
	static const float caps[] = { 0.0f, 30.0f, 1.0f };
	unsigned nmax = 151936;
	float *y = malloc(nmax * sizeof(*y)), *sc = malloc(nmax * sizeof(*sc));

	if (!y || !sc)
		return 1;
	for (unsigned ni = 0; ni < sizeof(ns) / sizeof(ns[0]); ni++)
		for (unsigned ci = 0; ci < 3; ci++)
			for (int rep = 0; rep < 6; rep++) {
				unsigned n = ns[ni];
				float cap = caps[ci];

				for (unsigned i = 0; i < n; i++) {
					y[i] = nrand() * (rep >= 3 ? 400.0f : 4.0f);
					sc[i] = 0.5f + urand();
				}
				/* planted ties at the maximum */
				if (n > 4 && rep % 2 == 0) {
					unsigned a = n / 3, b = n / 2;

					y[a] = y[b] = 1e4f;
					sc[a] = sc[b] = 1.0f;
				}
				/* zero and negative scales */
				if (n > 8 && rep == 1) {
					sc[3] = 0.0f;
					sc[5] = -2.0f;
				}
				check(y, NULL, n, cap, "plain");
				check(y, sc, n, cap, "scaled");
			}
	/* NaN at index 0 holds; NaN later never wins; infinities */
	for (unsigned n = 1; n < 12; n++) {
		for (unsigned i = 0; i < n; i++) {
			y[i] = nrand();
			sc[i] = 1.0f + urand();
		}
		y[0] = NAN;
		check(y, NULL, n, 0.0f, "nan0");
		check(y, sc, n, 30.0f, "nan0");
		y[0] = 0.0f;
		y[n / 2] = NAN;
		check(y, NULL, n, 0.0f, "nanmid");
		check(y, sc, n, 30.0f, "nanmid");
		y[n - 1] = INFINITY;
		check(y, NULL, n, 0.0f, "inf");
		check(y, sc, n, 30.0f, "inf");
		y[0] = -INFINITY;
		check(y, sc, n, 0.0f, "-inf");
	}
	/*
	 * THE CONTROL: values a cap of 1 saturates. Uncapped they have one
	 * clear maximum late in the row; capped, many become tanhf(x) == 1.0f
	 * and the first of those wins. If the two agree everywhere, the cap
	 * cases above prove nothing.
	 */
	for (int rep = 0; rep < 20; rep++) {
		unsigned n = 1000;
		float *t = malloc(n * sizeof(*t));

		for (unsigned i = 0; i < n; i++)
			y[i] = 10.0f + 50.0f * urand();
		y[n - 7] = 1000.0f;
		check(y, NULL, n, 1.0f, "saturated");
		memcpy(t, y, n * sizeof(*t));
		ref_cap(t, n, 1.0f);
		if (ref_argmax(t, n) != charsiu_argmax_scaled(y, NULL, n, 0.0f))
			ctrl_differs++;
		free(t);
	}
	free(y);
	free(sc);
	if (!ctrl_differs) {
		fprintf(stderr, "argmax_fused: the saturating control never "
			"differed from the uncapped argmax -- it tests nothing\n");
		return 1;
	}
	printf("argmax_fused: %u cases, %u differ (control differed %u of 20)"
	       "\n", cases, fails, ctrl_differs);
	return fails ? 1 : 0;
}
