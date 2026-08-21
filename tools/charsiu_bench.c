// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * What does a matmul on this NPU actually cost, and what is the cost OF?
 *
 * ROUND 165 ANSWERED THE FIRST HALF, on a ROCK 4D, and the answer was not
 * dispatch. Sweeping tasks per job from 1 to 128 at three shapes, the marginal
 * cost of one more task is
 *
 *   M=1  K=2048 N=1024   2.10 MB of weights   201.9 us   10.4 GB/s
 *   M=1  K=512  N=1024   0.52 MB              57.4 us     9.1 GB/s
 *   M=32 K=2048 N=1024   2.10 MB             217.6 us     9.6 GB/s
 *
 * M is nearly free. The same weights at M = 32 cost 1.08 times what they cost at
 * M = 1 while doing 32 times the arithmetic, so the cost is not MAC and not the
 * row count. It tracks the WEIGHT BYTES, at something close to 10 GB/s, which is
 * about half of what this board's LPDDR5 can do.
 *
 * There is also a fixed cost of 180 to 195 us per submit that chaining removes,
 * which is why a small projection gains 4.4 times from batching and a large one
 * only 1.9 times.
 *
 * If that reading is right it decides the whole runtime. Llama-3.2-1B reads
 * about 973 M projection weights per token, once each, so at 10 GB/s int8 is
 * 97 ms a token and int4 is 49 ms, against the vendor's shipped 13 tokens a
 * second. Decode would be DRAM bound and int4 would be the only 2x on the table.
 *
 * SO THIS ROUND TRIES TO BREAK IT. --shapes runs a table at a fixed task count
 * and prints the implied bandwidth for each, including TWO PAIRS THAT HAVE THE
 * SAME WEIGHT BYTES IN DIFFERENT SHAPES. If the cost is bytes, the pairs match
 * and the column is flat. If it is anything else, they do not.
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "charsiu.h"

#define MAX_TASKS 128

static double now_us(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

/*
 * The weight bytes a shape actually moves. int4 packs two weights to a byte, so
 * k*n is right for int8 and twice the truth for int4, and a GB/s that means
 * different things on the two paths is worse than none.
 */
static double wbytes_of(unsigned k, unsigned n)
{
	return (double)k * n / (getenv("CHARSIU_W4") ? 2.0 : 1.0);
}

/*
 * The same matmul on one core, as the denominator.
 *
 * ROUND 165's VERSION MEASURED 0.0 us, because the compiler deleted the whole
 * loop: its result went into a local that was only compared against a value it
 * could never take. A denominator that reads zero is worse than no denominator,
 * since the ratio it feeds is the entire RK3588 argument. The accumulator is
 * volatile now and is printed by the caller.
 */
static volatile int64_t cpu_sink;

static double cpu_us(const struct charsiu_job *job, const uint8_t *a,
		     const uint8_t *b, unsigned reps)
{
	const struct charsiu_matmul *mm = &job->mm;
	double t0 = now_us();
	unsigned r, n, k;

	for (r = 0; r < reps; r++) {
		int64_t sum = 0;

		for (n = 0; n < mm->n; n++) {
			int32_t acc = 0;

			for (k = 0; k < mm->k; k++)
				acc += ((int)a[k] - job->input_zero_point) *
				       ((int)b[n * mm->k + k] - job->weight_zero_point);
			sum += acc;
		}
		cpu_sink += sum;
	}
	return (now_us() - t0) / reps;
}

struct bench {
	struct charsiu_device *dev;
	struct charsiu_job job;
	struct charsiu_bo regcmd, in, wt, outbo, coef;
	struct charsiu_task tasks[MAX_TASKS];
	uint32_t in_handles[3], out_handles[1];
	uint8_t *a_raw, *b_raw;
	int32_t *bias, *wsums;
};

static void bench_free(struct bench *b)
{
	charsiu_bo_free(b->dev, &b->regcmd);
	charsiu_bo_free(b->dev, &b->in);
	charsiu_bo_free(b->dev, &b->wt);
	charsiu_bo_free(b->dev, &b->outbo);
	charsiu_bo_free(b->dev, &b->coef);
	free(b->a_raw); free(b->b_raw); free(b->bias); free(b->wsums);
	memset(b, 0, sizeof(*b));
}

/* Stage one shape: buffers, operands, coefficients and MAX_TASKS streams, each
 * with its own output slice so nothing here is cheaper than a runtime's. */
static int bench_setup(struct bench *b, struct charsiu_device *dev,
		       unsigned m, unsigned k, unsigned n)
{
	size_t stride = 4096, nreg;
	unsigned i, t;
	int ret;

	memset(b, 0, sizeof(*b));
	b->dev = dev;
	b->job.mm.m = m; b->job.mm.k = k; b->job.mm.n = n;
	/*
	 * ⚠ CHARSIU_W4 BENCHES THE int4 PATH, which this file has never had.
	 * The whole int4 line, rounds 265 to 303, exists to answer "what does it
	 * buy", and until now the only benchmark in the repo was int8 only, so
	 * the question could not be asked at all.
	 *
	 * The envelope it is valid in is measured: K a multiple of 32 up to 224,
	 * N a multiple of 8 up to 160, M = 1. Outside that the packer refuses
	 * and the numbers would be timing an empty weight buffer.
	 */
	b->job.mm.wdtype = getenv("CHARSIU_W4") ? CHARSIU_INT4 : CHARSIU_INT8;
	b->job.mm.adtype = CHARSIU_INT8;
	b->job.input_scale = 0.02f;
	b->job.weight_scale = 0.01f;
	b->job.output_scale = 0.25f;
	b->job.input_zero_point = 128;
	b->job.weight_zero_point =
		b->job.mm.wdtype == CHARSIU_INT4 ? 0 : 128;
	b->job.output_zero_point = 0;

	b->a_raw = malloc((size_t)m * k);
	b->b_raw = malloc((size_t)n * k);
	b->bias = calloc(n, sizeof(*b->bias));
	b->wsums = calloc(n, sizeof(*b->wsums));
	if (!b->a_raw || !b->b_raw || !b->bias || !b->wsums)
		return -1;
	for (i = 0; i < m * k; i++)
		b->a_raw[i] = (uint8_t)(128 + (int)(i * 7 % 61) - 30);
	for (i = 0; i < n * k; i++)
		b->b_raw[i] = b->job.mm.wdtype == CHARSIU_INT4
			? (uint8_t)(((int)(i * 13 % 15) - 7) & 0xf)
			: (uint8_t)(128 + (int)(i * 13 % 41) - 20);
	for (i = 0; i < n; i++) {
		unsigned j;

		for (j = 0; j < k; j++) {
			int wv = b->b_raw[(size_t)i * k + j];

			if (b->job.mm.wdtype == CHARSIU_INT4)
				wv = (wv & 0x8) ? (wv & 0xf) - 16 : (wv & 0xf);
			b->wsums[i] += wv - (int)b->job.weight_zero_point;
		}
	}

	ret = charsiu_bo_alloc(dev, stride * MAX_TASKS, &b->regcmd);
	ret |= charsiu_bo_alloc(dev, (size_t)charsiu_entries_per_row(&b->job.mm) * 64 * m + 4096, &b->in);
	ret |= charsiu_bo_alloc(dev, charsiu_weight_bytes(&b->job.mm) + 4096, &b->wt);
	/* four bytes an element on the int4 path, as everywhere else */
	ret |= charsiu_bo_alloc(dev, (size_t)m * n * MAX_TASKS
		* (b->job.mm.wdtype == CHARSIU_INT4 ? 4 : 1) + 4096, &b->outbo);
	ret |= charsiu_bo_alloc(dev, charsiu_coef_bytes(&b->job.mm) + 4096, &b->coef);
	if (ret)
		return ret;

	b->job.input_addr = (uint32_t)b->in.dma_address;
	b->job.weight_addr = (uint32_t)b->wt.dma_address;
	b->job.coef_addr = (uint32_t)b->coef.dma_address;

	charsiu_bo_prep(dev, &b->in, 1000000000);
	charsiu_pack_input(&b->job.mm, b->a_raw, b->in.map, b->in.size, 128);
	charsiu_bo_fini(dev, &b->in);
	charsiu_bo_prep(dev, &b->wt, 1000000000);
	charsiu_pack_weights(&b->job.mm, b->b_raw, b->wt.map);
	charsiu_bo_fini(dev, &b->wt);
	charsiu_bo_prep(dev, &b->coef, 1000000000);
	charsiu_build_coefs(&b->job, b->bias, b->wsums, b->coef.map);
	charsiu_bo_fini(dev, &b->coef);

	charsiu_bo_prep(dev, &b->regcmd, 1000000000);
	for (t = 0; t < MAX_TASKS; t++) {
		uint64_t *stream = (uint64_t *)((uint8_t *)b->regcmd.map + t * stride);

		b->job.output_addr = (uint32_t)b->outbo.dma_address + t * m * n;
		nreg = charsiu_emit_job(&b->job, stream, stride / 8);
		if (!nreg)
			return -1;
		b->tasks[t].regcmd = (uint32_t)(b->regcmd.dma_address + t * stride);
		b->tasks[t].regcmd_count = (uint32_t)nreg;
	}
	charsiu_bo_fini(dev, &b->regcmd);

	b->in_handles[0] = b->in.handle;
	b->in_handles[1] = b->wt.handle;
	b->in_handles[2] = b->coef.handle;
	b->out_handles[0] = b->outbo.handle;
	return 0;
}

/* Wall time for one submit of `tasks` chained tasks, including the fence,
 * because that is what a runtime pays and a number without it cannot become a
 * token rate. Returns microseconds per submit, or -1. */
static double bench_run(struct bench *b, unsigned tasks, unsigned reps)
{
	struct charsiu_joblist jl;
	double t0;
	unsigned r;

	jl.tasks = b->tasks;
	jl.task_count = tasks;
	jl.in_handles = b->in_handles;
	jl.in_count = 3;
	jl.out_handles = b->out_handles;
	jl.out_count = 1;

	/* one untimed pass: the first fault and the first power up are not the
	 * steady state a runtime lives in */
	if (charsiu_submit_jobs(b->dev, &jl, 1))
		return -1;
	if (charsiu_bo_prep(b->dev, &b->outbo, 2000000000))
		return -1;
	charsiu_bo_fini(b->dev, &b->outbo);

	t0 = now_us();
	for (r = 0; r < reps; r++) {
		if (charsiu_submit_jobs(b->dev, &jl, 1))
			return -1;
		if (charsiu_bo_prep(b->dev, &b->outbo, 2000000000))
			return -1;
		charsiu_bo_fini(b->dev, &b->outbo);
	}
	return (now_us() - t0) / reps;
}

/*
 * THE FALSIFIER. Same weight bytes, different shapes, at a fixed task count deep
 * enough that the fixed submit cost is amortised.
 *
 * K=2048 N=1024 and K=1024 N=2048 are both 2 MB. K=1024 N=1024 and K=2048 N=512
 * are both 1 MB. If the marginal cost is the weight fetch, each pair costs the
 * same and the GB/s column is flat across the whole table. If it is anything
 * with a shape term in it, the pairs come apart, and the number that has been
 * built on top of this reading has to come down.
 */
static void shape_table(struct charsiu_device *dev, unsigned tasks, unsigned reps)
{
	static const unsigned shapes[][2] = {
		{ 512, 1024 }, { 1024, 1024 }, { 2048, 512 },
		{ 2048, 1024 }, { 1024, 2048 }, { 4096, 1024 }, { 2048, 4096 },
	};
	unsigned i;

	printf("  %u chained tasks, %u reps, M = 1\n\n", tasks, reps);
	printf("  %-6s %-6s %-9s %-11s %-11s %-9s\n",
	       "K", "N", "weight MB", "us/matmul", "GB/s", "GOP/s");
	for (i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
		unsigned k = shapes[i][0], n = shapes[i][1];
		double mb = (double)k * n / 1e6, us;
		struct bench b;

		if (bench_setup(&b, dev, 1, k, n)) {
			printf("  %-6u %-6u setup FAILED\n", k, n);
			bench_free(&b);
			continue;
		}
		us = bench_run(&b, tasks, reps);
		if (us < 0)
			printf("  %-6u %-6u run FAILED\n", k, n);
		else
			/* MB over microseconds is already GB/s times 1e-3, and
			 * round 166 printed 0.01 down the whole column because
			 * this had an extra 1e-3 in it. The measurement was
			 * fine; the column was unreadable. */
			printf("  %-6u %-6u %-9.2f %-11.2f %-11.2f %-9.1f\n",
			       k, n, mb, us / tasks,
			       mb / (us / tasks) * 1e3,
			       2.0 * k * n * tasks / us / 1e3);
		bench_free(&b);
	}
}

int main(int argc, char **argv)
{
	struct charsiu_device *dev;
	struct bench b;
	unsigned m, k, n, reps, t;
	double us, one;

	dev = charsiu_open(NULL);
	if (!dev) { printf("open FAILED\n"); return 1; }

	if (argc > 1 && !strcmp(argv[1], "--shapes")) {
		shape_table(dev, argc > 2 ? (unsigned)atoi(argv[2]) : 32,
			    argc > 3 ? (unsigned)atoi(argv[3]) : 50);
		charsiu_close(dev);
		return 0;
	}

	m = argc > 1 ? (unsigned)atoi(argv[1]) : 1;
	k = argc > 2 ? (unsigned)atoi(argv[2]) : 2048;
	n = argc > 3 ? (unsigned)atoi(argv[3]) : 1024;
	reps = argc > 4 ? (unsigned)atoi(argv[4]) : 200;

	printf("bench M=%u K=%u N=%u %s, %u reps, %.2f MOP, %.2f MB of weights\n",
	       m, k, n, getenv("CHARSIU_W4") ? "int4" : "int8", reps,
	       2.0 * m * k * n / 1e6,
	       (double)k * n / (getenv("CHARSIU_W4") ? 2e6 : 1e6));
	if (bench_setup(&b, dev, m, k, n)) { printf("setup FAILED\n"); return 1; }

	printf("\n  %-8s %-12s %-12s %-12s %-12s\n",
	       "tasks", "us/submit", "us/matmul", "GB/s", "GOP/s");
	one = 0;
	for (t = 1; t <= MAX_TASKS; t *= 2) {
		us = bench_run(&b, t, reps);
		if (us < 0) { printf("  %-8u FAILED\n", t); continue; }
		if (t == 1)
			one = us;
		printf("  %-8u %-12.1f %-12.2f %-12.2f %-12.1f\n", t, us, us / t,
		       wbytes_of(k, n) / (us / t) / 1e3,
		       2.0 * m * k * n * t / us / 1e3);
		/* the weight bytes over us microseconds is GB/s directly. */
	}

	/*
	 * The marginal cost of one more task, which is the number that matters:
	 * the difference between 128 and 64 chained tasks has no fixed submit
	 * cost in it at all.
	 */
	{
		double a128 = bench_run(&b, 128, reps), a64 = bench_run(&b, 64, reps);

		if (a128 > 0 && a64 > 0)
			printf("\n  marginal cost of one task: %.1f us  (%.2f GB/s)\n"
			       "  fixed cost of a submit:    %.1f us\n",
			       (a128 - a64) / 64,
			       wbytes_of(k, n) / ((a128 - a64) / 64) / 1e3,
			       one - (a128 - a64) / 64);
	}

	printf("\n  cpu, one thread, same matmul: %.1f us   (sink %lld)\n",
	       cpu_us(&b.job, b.a_raw, b.b_raw, 3), (long long)cpu_sink);
	bench_free(&b);
	charsiu_close(dev);
	return 0;
}
