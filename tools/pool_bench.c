// SPDX-License-Identifier: GPL-2.0-or-later
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */

/*
 * THE PRICE OF ONE POOL DISPATCH, with nothing else in it.
 *
 * charsiu_parallel_for over n items, each item busy for `unit_us`, with
 * `gap_us` of serial work on the caller between dispatches (the shape of a
 * prompt: a pack, a read, a scale, each a dispatch, with the NPU's wait or
 * the next stage's setup between them). Prints the wall time a dispatch and
 * the overhead above the ideal n * unit / threads.
 *
 *   pool_bench THREADS ITEMS UNIT_US GAP_US REPS [wide]
 *
 * The pool reads CHARSIU_CPUS, CHARSIU_POOL_FAST and CHARSIU_POOL_SPIN_US
 * exactly as charsiu_run does; `wide` widens it as a batched prompt chunk does.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "charsiu_llm.h"

static double now_us(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static void busy(double us)
{
	double t0 = now_us();

	while (now_us() - t0 < us)
		;
}

static double g_unit;
static volatile uint64_t g_sink[1024];

static void unit_fn(void *ctx, uint64_t r0, uint64_t n)
{
	(void)ctx;
	for (uint64_t i = 0; i < n; i++)
		if (g_unit > 0)
			busy(g_unit);
	g_sink[(r0 * 8) % 1024] += n;
}

static int cmp(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;

	return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
	int t, items, reps, wide;
	double gap, *d;

	if (argc < 6) {
		fprintf(stderr, "usage: %s THREADS ITEMS UNIT_US GAP_US REPS "
			"[wide]\n", argv[0]);
		return 2;
	}
	t = atoi(argv[1]);
	items = atoi(argv[2]);
	g_unit = atof(argv[3]);
	gap = atof(argv[4]);
	reps = atoi(argv[5]);
	wide = argc > 6 && !strcmp(argv[6], "wide");
	charsiu_threads_start(t);
	if (wide)
		llama_prefill_scope(80);
	d = calloc((size_t)reps, sizeof(*d));
	for (int w = 0; w < 50; w++)
		charsiu_parallel_for(unit_fn, NULL, (uint64_t)items);
	for (int r = 0; r < reps; r++) {
		double t0;

		if (gap > 0)
			busy(gap);
		t0 = now_us();
		charsiu_parallel_for(unit_fn, NULL, (uint64_t)items);
		d[r] = now_us() - t0;
	}
	qsort(d, (size_t)reps, sizeof(*d), cmp);
	{
		double s = 0, ideal = (double)items * g_unit / t;

		for (int r = 0; r < reps; r++)
			s += d[r];
		printf("threads %d items %d unit %.1f gap %.1f %s: mean %.1f "
		       "p50 %.1f p90 %.1f us; ideal %.1f, over %.1f us\n",
		       t, items, g_unit, gap, wide ? "wide" : "narrow",
		       s / reps, d[reps / 2], d[reps * 9 / 10], ideal,
		       d[reps / 2] - ideal);
	}
	return 0;
}
