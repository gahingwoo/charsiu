// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * Time the vision tower's attention on its own, at the board's shape.
 *
 * ⚠⚠ THE HOST CANNOT SEE THIS STAGE THROUGH THE TOWER. On the board the
 * matmuls run on the NPU and the attention is 4010 ms of an 8156 ms encode --
 * half of it. On a development host the same matmuls run on the CPU, so the
 * stage table reads: feed forward 66%, attention 5.8%. Tuning a 5.8% row by
 * running the whole tower is measuring the feed forward's noise.
 *
 * ⚠⚠ AND THE HOST IS COMPUTE BOUND WHERE THE BOARD IS BANDWIDTH BOUND. This
 * host holds the whole q/k/v/o working set -- 12 MB at n = 1024 -- in cache,
 * and the board, with about a megabyte of L2 for the whole A72 cluster, does
 * not. So a change that only removes DRAM traffic reads as a NON RESULT here.
 * That is what -n is for: at n = 8192 the working set is 100 MB and no cache
 * on either machine holds it, so a locality change has to show up. A win that
 * appears at n = 8192 and not at n = 1024 is a BOARD win and a host non result,
 * and saying so is the whole point of running both.
 *
 * ⚠⚠ AND THE TWO VARIANTS ARE TIMED IN ONE PROCESS, INTERLEAVED. Two builds
 * run one after the other disagreed by 1.8x on this host with the SAME binary
 * on both sides, because six cores are shared with an editor and other agents.
 * -c runs both schedules rep by rep so they meet the same interference.
 *
 * The numbers are random and the answer is thrown away; tests/vision_cross.py
 * is the correctness oracle and this is not one.
 *
 *   vattn_bench [-n TOKENS] [-W WIDTH] [-H HEADS] [-l LAYERS] [-r REPS]
 *               [-c]     time EVERY schedule, interleaved, and diff them
 *               [-Q]     sweep the query block size the same way
 *               [-F]     the exact kernel against the fused one
 *               [-K]     sweep the fused kernel's key tile
 *               [-B]     the whole round: the code as it was, against as it is
 *               [-E]     glibc's expf against the polynomial one
 *               [-P]     one pair at a time, against the blocked kernels
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "charsiu_llm.h"
#include "charsiu_vision.h"

static double now_ms(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

/*
 * ⚠ NOT rand(). The same stream every run means two builds are compared on the
 * same numbers, and a softmax over random scores is sensitive to the spread:
 * scaled to about a unit normal, which is where a trained ViT's scores live.
 */
static void fill(float *p, size_t n, uint32_t seed)
{
	size_t i;

	for (i = 0; i < n; i++) {
		seed = seed * 1664525u + 1013904223u;
		p[i] = ((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.5f;
	}
}

/*
 * ⚠ ELEMENT WISE, NOT A CHECKSUM. Most of the variants here reorder only the
 * issue of the arithmetic and are bit identical, and a checksum would catch
 * those. The fused kernel is not bit identical, and a checksum cannot tell a
 * few ulp of rounding from an output that is wrong in two places and right
 * everywhere else -- which is the shape of the fast wrong answer this tree
 * shipped once already.
 */
static float worst(const float *a, const float *b, size_t n)
{
	float w = 0.0f;
	size_t i;

	for (i = 0; i < n; i++) {
		float d = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];

		if (d > w)
			w = d;
	}
	return w;
}

int main(int argc, char **argv)
{
	/* the block sizes worth asking about: 1 is the unblocked form */
	static const unsigned qbs[] = { 1, 4, 8, 16, 32, 64, 128, 256 };
	static const unsigned kts[] = { 16, 32, 64, 128, 256, 512 };
	unsigned n = 1024, W = 768, H = 12, L = 12, reps = 3, cmp = 0, qsw = 0;
	unsigned nv = CHARSIU_VATTN_SCHEDS, fsw = 0, ksw = 0, bef = 0, esw = 0, psw = 0;
	double best[16] = { 0.0 };
	float diff[16] = { 0.0f };
	size_t sz;
	float *q, *k, *v, *o, *ref;
	unsigned i, r, s, lo, hi;

	for (i = 1; i < (unsigned)argc; i++) {
		if (!strcmp(argv[i], "-n") && i + 1 < (unsigned)argc)
			n = (unsigned)atoi(argv[++i]);
		else if (!strcmp(argv[i], "-W") && i + 1 < (unsigned)argc)
			W = (unsigned)atoi(argv[++i]);
		else if (!strcmp(argv[i], "-H") && i + 1 < (unsigned)argc)
			H = (unsigned)atoi(argv[++i]);
		else if (!strcmp(argv[i], "-l") && i + 1 < (unsigned)argc)
			L = (unsigned)atoi(argv[++i]);
		else if (!strcmp(argv[i], "-r") && i + 1 < (unsigned)argc)
			reps = (unsigned)atoi(argv[++i]);
		else if (!strcmp(argv[i], "-c"))
			cmp = 1;
		else if (!strcmp(argv[i], "-Q"))
			qsw = 1;
		else if (!strcmp(argv[i], "-F"))
			fsw = 1;
		else if (!strcmp(argv[i], "-K"))
			ksw = 1;
		else if (!strcmp(argv[i], "-B"))
			bef = 1;
		else if (!strcmp(argv[i], "-E"))
			esw = 1;
		else if (!strcmp(argv[i], "-P"))
			psw = 1;
	}
	if (psw) {
		cmp = 1;
		nv = 2;
	}
	if (esw) {
		cmp = 1;
		nv = 2;
	}
	if (bef) {
		cmp = 1;
		nv = 2;
	}
	if (ksw) {
		cmp = 1;
		nv = (unsigned)(sizeof(kts) / sizeof(kts[0]));
	}
	if (qsw) {
		cmp = 1;
		nv = (unsigned)(sizeof(qbs) / sizeof(qbs[0]));
	}
	if (fsw) {
		cmp = 1;
		nv = 2;
	}
	if (!H || W % H) {
		fprintf(stderr, "vattn_bench: %u heads do not divide %u\n",
			H, W);
		return 2;
	}
	charsiu_threads_start(0);

	sz = (size_t)n * W * sizeof(float);
	q = malloc(sz); k = malloc(sz); v = malloc(sz); o = malloc(sz);
	ref = malloc(sz);
	if (!q || !k || !v || !o || !ref) {
		fprintf(stderr, "vattn_bench: out of memory\n");
		return 1;
	}
	fill(q, (size_t)n * W, 1u);
	fill(k, (size_t)n * W, 2u);
	fill(v, (size_t)n * W, 3u);

	printf("n %u W %u heads %u layers %u threads %d\n", n, W, H, L,
	       charsiu_threads());
	lo = cmp ? 0 : (unsigned)charsiu_vision_attn_sched_get();
	hi = cmp ? nv : lo + 1;
	/*
	 * ⚠ REP OUTSIDE, SCHEDULE INSIDE. The other way round is two benchmarks
	 * run at two different times, which on a host whose six cores are shared
	 * with an editor disagreed by 1.8x with the same code on both sides.
	 */
	for (r = 0; r < reps; r++)
		for (s = lo; s < hi; s++) {
			double t0, dt;
			unsigned l;

			if (bef) {
				/*
				 * ⚠ THE WHOLE ROUND IN ONE PROCESS. Every one
				 * of these four was measured on its own; this
				 * is the only number that says what a person
				 * actually gets, and it has to meet the same
				 * interference as the thing it is compared to.
				 */
				charsiu_vision_attn_sched_set(s ? 2 : 0);
				charsiu_vision_attn_fused_set((int)s);
				charsiu_vision_attn_qb_set(s ? 64 : 8);
				charsiu_vision_attn_kt_set(s ? 16 : 16);
				charsiu_softmax_exact_set(!s);
				charsiu_attn_plain_set(!s);
			} else if (esw)
				charsiu_softmax_exact_set(!s);
			else if (psw)
				charsiu_attn_plain_set(!s);
			else if (qsw)
				charsiu_vision_attn_qb_set(qbs[s]);
			else if (ksw)
				charsiu_vision_attn_kt_set(kts[s]);
			else if (fsw)
				charsiu_vision_attn_fused_set((int)s);
			else
				charsiu_vision_attn_sched_set((int)s);
			t0 = now_ms();
			for (l = 0; l < L; l++)
				charsiu_vision_attention(q, k, v, o, n, W, H,
					1.0f / sqrtf((float)(W / H)));
			dt = now_ms() - t0;
			if (!best[s] || dt < best[s])
				best[s] = dt;
			/*
			 * ⚠ AND THE RESULT HAS TO BE READ, AND COMPARED.
			 * Without it -O2 is entitled to notice the output is
			 * dead; and a variant that is faster because it stopped
			 * computing the answer is the failure mode a stopwatch
			 * reports as a speedup. The first variant is the
			 * reference and every other is diffed against it.
			 */
			if (s == lo && r == 0)
				memcpy(ref, o, sz);
			else
				diff[s] = worst(o, ref, (size_t)n * W);
		}

	for (s = lo; s < hi; s++) {
		char nm[32];

		if (bef)
			snprintf(nm, sizeof(nm), "%s", s ? "after" : "before");
		else if (esw)
			snprintf(nm, sizeof(nm), "%s", s ? "poly" : "glibc");
		else if (psw)
			snprintf(nm, sizeof(nm), "%s", s ? "blocked" : "1 pair");
		else if (qsw)
			snprintf(nm, sizeof(nm), "qb=%u", qbs[s]);
		else if (ksw)
			snprintf(nm, sizeof(nm), "kt=%u", kts[s]);
		else if (fsw)
			snprintf(nm, sizeof(nm), "%s", s ? "fused" : "exact");
		else
			snprintf(nm, sizeof(nm), "%s",
				 charsiu_vision_attn_sched_name((int)s));
		printf("%-9s %8.0f ms best of %u  (%8.2f ms a layer)",
		       nm, best[s], reps, best[s] / L);
		if (cmp && s > lo)
			printf("   %.2fx   worst diff %.2e",
			       best[lo] / best[s], (double)diff[s]);
		printf("\n");
	}
	free(q); free(k); free(v); free(o); free(ref);
	return 0;
}
