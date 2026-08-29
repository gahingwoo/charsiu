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

static double checksum(const float *o, size_t n)
{
	double s = 0.0;
	size_t i;

	for (i = 0; i < n; i++)
		s += o[i];
	return s;
}

int main(int argc, char **argv)
{
	unsigned n = 1024, W = 768, H = 12, L = 12, reps = 3, cmp = 0;
	double best[CHARSIU_VATTN_SCHEDS] = { 0.0 };
	double chk[CHARSIU_VATTN_SCHEDS] = { 0.0 };
	size_t sz;
	float *q, *k, *v, *o;
	unsigned i, r, s, lo, hi, bad = 0;

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
	}
	if (!H || W % H) {
		fprintf(stderr, "vattn_bench: %u heads do not divide %u\n",
			H, W);
		return 2;
	}
	charsiu_threads_start(0);

	sz = (size_t)n * W * sizeof(float);
	q = malloc(sz); k = malloc(sz); v = malloc(sz); o = malloc(sz);
	if (!q || !k || !v || !o) {
		fprintf(stderr, "vattn_bench: out of memory\n");
		return 1;
	}
	fill(q, (size_t)n * W, 1u);
	fill(k, (size_t)n * W, 2u);
	fill(v, (size_t)n * W, 3u);

	printf("n %u W %u heads %u layers %u threads %d\n", n, W, H, L,
	       charsiu_threads());
	lo = cmp ? 0 : (unsigned)charsiu_vision_attn_sched_get();
	hi = cmp ? CHARSIU_VATTN_SCHEDS : lo + 1;
	/*
	 * ⚠ REP OUTSIDE, SCHEDULE INSIDE. The other way round is two benchmarks
	 * run at two different times, which on a host whose six cores are shared
	 * with an editor disagreed by 1.8x with the same code on both sides.
	 */
	for (r = 0; r < reps; r++)
		for (s = lo; s < hi; s++) {
			double t0, dt;
			unsigned l;

			charsiu_vision_attn_sched_set((int)s);
			t0 = now_ms();
			for (l = 0; l < L; l++)
				charsiu_vision_attention(q, k, v, o, n, W, H,
					1.0f / sqrtf((float)(W / H)));
			dt = now_ms() - t0;
			if (!best[s] || dt < best[s])
				best[s] = dt;
			/*
			 * ⚠ AND THE RESULT HAS TO BE READ. Without a checksum
			 * -O2 is entitled to notice the output is dead; and the
			 * schedules are meant to be BIT identical, so a
			 * difference here is a bug that a stopwatch would have
			 * reported as a speedup.
			 */
			chk[s] = checksum(o, (size_t)n * W);
		}

	for (s = lo; s < hi; s++) {
		printf("%-9s %8.0f ms best of %u  (%8.2f ms a layer)",
		       charsiu_vision_attn_sched_name((int)s), best[s], reps,
		       best[s] / L);
		if (cmp && s)
			printf("   %.2fx", best[0] / best[s]);
		printf("\n");
		if (chk[s] != chk[lo])
			bad = 1;
	}
	printf("checksum %.9g%s\n", chk[lo],
	       bad ? "   ⚠ THE SCHEDULES DISAGREE, THIS IS A BUG" : "");
	free(q); free(k); free(v); free(o);
	return bad ? 1 : 0;
}
