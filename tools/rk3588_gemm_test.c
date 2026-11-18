// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * Does the RK3588 emitter compute a matmul, and at which shapes?
 *
 *   rk3588_gemm_test [K] [N]
 *
 * THE DECISION RULE, STATED BEFORE ANY RUN.
 *
 * The NPU's output is int8 through the DPU requantiser, so the oracle is that
 * requantiser modelled on the CPU: q = clamp(round(acc * scale / 2^(shift-1))),
 * with the two fields rk3588_out_cvt computes. Its rounding mode is NOT known,
 * so a value counts as
 *   exact   if it equals the model,
 *   near    if it is one away (a rounding disagreement),
 *   wrong   otherwise.
 * A shape PASSES when it has no wrong values and at least 90% exact.
 *
 * THE CONTROL RUNS FIRST and is the square shape Mesa runs MobileNet through,
 * w = h = 4. If the control does not pass, the emitter is wrong and no other
 * row means anything; the program says so and stops.
 *
 * AND THE ORACLE MUST BE ABLE TO FAIL. Every passing row is scored a second
 * time against the reference computed with B's rows rotated by one. That has
 * to come out mostly wrong; if it does not, the outputs are too narrow to tell
 * a right answer from a wrong one and the PASS is void. The output must also
 * carry at least 32 distinct values, so a row of saturated or zero outputs
 * cannot pass.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "charsiu.h"
#include "charsiu_rk3588.h"

#define REG_WORDS 512

struct score {
	unsigned exact, near, wrong, distinct;
};

/*
 * xorshift, NOT an LCG. The first version of this used an LCG's bits 8..11,
 * which repeat every 4096 draws: at K = 1024 every row of A was the same few
 * rows, the outputs collapsed to two to four values, and a right answer could
 * not be told from a wrong one. The distinct-value floor caught it.
 */
static uint32_t xs = 0x9e3779b9u;

static int rnd(int lo, int hi)
{
	xs ^= xs << 13;
	xs ^= xs >> 17;
	xs ^= xs << 5;
	return lo + (int)(xs % (unsigned)(hi - lo + 1));
}

static uint8_t requant(const struct rk3588_mm *mm, int64_t acc)
{
	unsigned scale, shift;
	double v;
	long q;

	rk3588_out_cvt(mm, &scale, &shift);
	v = (double)acc * scale / (double)(1ull << (shift - 1));
	q = lround(v) + ((long)mm->output_zp - 0x80);
	if (q < -128) q = -128;
	if (q > 127) q = 127;
	return (uint8_t)(q + 0x80);
}

static void reference(const struct rk3588_mm *mm, const uint8_t *A,
		      const uint8_t *B, unsigned rot, uint8_t *C)
{
	unsigned m = mm->w * mm->h;

	for (unsigned r = 0; r < m; r++)
		for (unsigned o = 0; o < mm->n; o++) {
			const uint8_t *b = B + (size_t)((o + rot) % mm->n) * mm->k;
			int64_t acc = 0;

			for (unsigned i = 0; i < mm->k; i++)
				acc += ((int)A[(size_t)r * mm->k + i] - (int)mm->input_zp) *
				       ((int)b[i] - (int)mm->weight_zp);
			C[(size_t)r * mm->n + o] = requant(mm, acc);
		}
}

static struct score compare(const uint8_t *got, const uint8_t *want, size_t n)
{
	struct score s = { 0 };
	unsigned char seen[256] = { 0 };

	for (size_t i = 0; i < n; i++) {
		int d = (int)got[i] - (int)want[i];

		if (!d) s.exact++;
		else if (d == 1 || d == -1) s.near++;
		else s.wrong++;
		if (!seen[got[i]]++)
			s.distinct++;
	}
	return s;
}

static int passes(struct score s, size_t n)
{
	unsigned floor = n / 2 < 32 ? (unsigned)(n / 2) : 32;

	return !s.wrong && s.exact * 10 >= n * 9 && s.distinct >= floor;
}

/* returns 1 pass, 0 fail, -1 the run itself failed */
static int run(struct charsiu_device *dev, unsigned w, unsigned h,
	       unsigned k, unsigned n, const char *label)
{
	struct rk3588_mm mm = {
		.w = w, .h = h, .k = k, .n = n,
		.input_zp = 0x80, .weight_zp = 0x80, .output_zp = 0x80,
		.input_scale = 1.0f, .weight_scale = 1.0f,
	};
	struct charsiu_bo in = { 0 }, wt = { 0 }, bs = { 0 }, ob = { 0 }, reg = { 0 };
	unsigned m = w * h;
	uint8_t *A = malloc((size_t)m * k), *B = malloc((size_t)n * k);
	uint8_t *C = malloc((size_t)m * n), *R = malloc((size_t)m * n);
	uint8_t *X = malloc((size_t)m * n);
	size_t nreg;
	int rc = -1;

	/*
	 * A POWER OF TWO SCALE SIZED TO K. An accumulator over K products of
	 * values in [-8, 7] has a spread of about 21 * sqrt(K); dividing by a
	 * power of two near a quarter of that keeps most outputs inside int8
	 * without saturating them all. A fixed 1/8 saturated everything at
	 * K = 1024.
	 */
	mm.output_scale = exp2f(roundf(log2f(21.0f * sqrtf((float)k) / 40.0f)));

	printf("  %-22s w=%-3u h=%-3u M=%-4u ", label, w, h, m);
	if (!A || !B || !C || !R || !X) {
		printf("out of memory\n");
		goto out;
	}
	for (size_t i = 0; i < (size_t)m * k; i++) A[i] = (uint8_t)(0x80 + rnd(-8, 7));
	for (size_t i = 0; i < (size_t)n * k; i++) B[i] = (uint8_t)(0x80 + rnd(-8, 7));

	if (charsiu_bo_alloc(dev, rk3588_input_bytes(&mm) + 4096, &in) ||
	    charsiu_bo_alloc(dev, rk3588_weight_bytes(&mm) + 4096, &wt) ||
	    charsiu_bo_alloc(dev, rk3588_bias_bytes(&mm) + 4096, &bs) ||
	    charsiu_bo_alloc(dev, rk3588_output_bytes(&mm) + 4096, &ob) ||
	    charsiu_bo_alloc(dev, REG_WORDS * 8, &reg)) {
		printf("a buffer would not allocate\n");
		goto out;
	}
	mm.input_addr = (uint32_t)in.dma_address;
	mm.weight_addr = (uint32_t)wt.dma_address;
	mm.bias_addr = (uint32_t)bs.dma_address;
	mm.output_addr = (uint32_t)ob.dma_address;

	charsiu_bo_prep(dev, &in, 1000000000);
	memset(in.map, 0, in.size);
	rk3588_pack_input(&mm, A, in.map);
	charsiu_bo_fini(dev, &in);

	charsiu_bo_prep(dev, &wt, 1000000000);
	memset(wt.map, 0, wt.size);
	rk3588_pack_weights(&mm, B, wt.map);
	charsiu_bo_fini(dev, &wt);

	charsiu_bo_prep(dev, &bs, 1000000000);
	rk3588_fill_bias(&mm, B, NULL, bs.map);
	charsiu_bo_fini(dev, &bs);

	/* a pattern, not zero, so "the NPU wrote nothing" cannot read as 0x80 */
	charsiu_bo_prep(dev, &ob, 1000000000);
	memset(ob.map, 0x5a, ob.size);
	charsiu_bo_fini(dev, &ob);

	charsiu_bo_prep(dev, &reg, 1000000000);
	memset(reg.map, 0, reg.size);
	nreg = rk3588_emit_matmul(&mm, reg.map, REG_WORDS);
	charsiu_bo_fini(dev, &reg);
	if (!nreg) {
		printf("refused: needs more than one task\n");
		rc = 0;
		goto out;
	}

	{
		uint32_t ins[3] = { in.handle, wt.handle, bs.handle };
		uint32_t outs[1] = { ob.handle };

		if (charsiu_submit(dev, &reg, (unsigned)nreg, ins, 3, outs, 1)) {
			printf("the submit failed\n");
			goto out;
		}
	}
	if (charsiu_bo_prep(dev, &ob, 2000000000)) {
		printf("timed out waiting for the output\n");
		goto out;
	}
	rk3588_unpack_output(&mm, ob.map, C);
	charsiu_bo_fini(dev, &ob);

	reference(&mm, A, B, 0, R);
	{
		size_t cells = (size_t)m * n;
		struct score s = compare(C, R, cells);
		int ok = passes(s, cells);

		printf("%5u exact %4u near %5u wrong  %3u distinct  %s\n",
		       s.exact, s.near, s.wrong, s.distinct, ok ? "PASS" : "FAIL");
		if (ok) {
			struct score x;

			reference(&mm, A, B, 1, X);
			x = compare(C, X, cells);
			if (x.wrong * 2 < cells) {
				printf("      VOID: the rotated reference scores %u wrong of %zu, "
				       "so this oracle cannot tell\n", x.wrong, cells);
				ok = 0;
			}
		} else if (s.wrong) {
			printf("      got :");
			for (unsigned i = 0; i < 8 && i < cells; i++) printf(" %4d", (int)C[i] - 0x80);
			printf("\n      want:");
			for (unsigned i = 0; i < 8 && i < cells; i++) printf(" %4d", (int)R[i] - 0x80);
			printf("\n");
		}
		rc = ok;
	}
out:
	if (reg.handle) charsiu_bo_free(dev, &reg);
	if (ob.handle) charsiu_bo_free(dev, &ob);
	if (bs.handle) charsiu_bo_free(dev, &bs);
	if (wt.handle) charsiu_bo_free(dev, &wt);
	if (in.handle) charsiu_bo_free(dev, &in);
	free(A); free(B); free(C); free(R); free(X);
	return rc;
}

static double now_us(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_double(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;

	return x < y ? -1 : x > y;
}

#define MAX_TILES 256
#define TIMED_REPS 20

/*
 * THE TILED PROJECTION. Same oracle and same decision rule as run(), at a
 * shape that needs more than one task. as_tasks = 0 sends one job per tile,
 * which the scheduler may spread over the three cores; 1 sends one job
 * holding every tile as a task, which runs them in order on one core.
 *
 * Correctness is scored on the FIRST submit; the timed repeats after it
 * reuse the same buffers and are not re-scored, so a time is only reported
 * for a run that passed.
 */
static int run_tiled(struct charsiu_device *dev, unsigned w, unsigned h,
		     unsigned k, unsigned n, unsigned nt, int as_tasks)
{
	struct rk3588_mm mm = {
		.w = w, .h = h, .k = k, .n = n,
		.input_zp = 0x80, .weight_zp = 0x80, .output_zp = 0x80,
		.input_scale = 1.0f, .weight_scale = 1.0f,
	};
	struct rk3588_tile tiles[MAX_TILES];
	struct charsiu_task tasks[MAX_TILES];
	struct charsiu_joblist jobs[MAX_TILES];
	struct charsiu_bo in = { 0 }, wt = { 0 }, bs = { 0 }, ob = { 0 }, reg = { 0 };
	unsigned m = w * h, count;
	size_t wbytes = 0, stride = 0;
	uint8_t *A = malloc((size_t)m * k), *B = malloc((size_t)n * k);
	uint8_t *C = malloc((size_t)m * n), *R = malloc((size_t)m * n);
	uint8_t *X = malloc((size_t)m * n);
	double t[TIMED_REPS];
	int rc = -1;

	mm.output_scale = exp2f(roundf(log2f(21.0f * sqrtf((float)k) / 40.0f)));
	count = rk3588_plan_tiles(&mm, nt, tiles, MAX_TILES, &wbytes);
	printf("  K=%-5u N=%-5u M=%-3u (w=%u h=%u)  %3u tiles of %-3u %-6s ",
	       k, n, m, w, h, count, count ? tiles[0].n : 0, as_tasks ? "tasks" : "jobs");
	if (!count) {
		printf("no tiling fits\n");
		rc = 0;
		goto out;
	}
	if (!A || !B || !C || !R || !X) {
		printf("out of memory\n");
		goto out;
	}
	for (size_t i = 0; i < (size_t)m * k; i++) A[i] = (uint8_t)(0x80 + rnd(-8, 7));
	for (size_t i = 0; i < (size_t)n * k; i++) B[i] = (uint8_t)(0x80 + rnd(-8, 7));

	stride = (REG_WORDS * 8 + 63) & ~(size_t)63;    /* Mesa aligns each stream to 64 */
	if (charsiu_bo_alloc(dev, rk3588_input_bytes(&mm) + 4096, &in) ||
	    charsiu_bo_alloc(dev, wbytes + 4096, &wt) ||
	    charsiu_bo_alloc(dev, rk3588_bias_bytes(&mm) + 4096, &bs) ||
	    charsiu_bo_alloc(dev, rk3588_output_bytes(&mm) + 4096, &ob) ||
	    charsiu_bo_alloc(dev, stride * count, &reg)) {
		printf("a buffer would not allocate\n");
		goto out;
	}
	mm.input_addr = (uint32_t)in.dma_address;
	mm.weight_addr = (uint32_t)wt.dma_address;
	mm.bias_addr = (uint32_t)bs.dma_address;
	mm.output_addr = (uint32_t)ob.dma_address;

	charsiu_bo_prep(dev, &in, 1000000000);
	memset(in.map, 0, in.size);
	rk3588_pack_input(&mm, A, in.map);
	charsiu_bo_fini(dev, &in);

	charsiu_bo_prep(dev, &wt, 1000000000);
	memset(wt.map, 0, wt.size);
	rk3588_pack_weights_tiled(&mm, tiles, count, B, wt.map);
	charsiu_bo_fini(dev, &wt);

	charsiu_bo_prep(dev, &bs, 1000000000);
	rk3588_fill_bias(&mm, B, NULL, bs.map);
	charsiu_bo_fini(dev, &bs);

	charsiu_bo_prep(dev, &ob, 1000000000);
	memset(ob.map, 0x5a, ob.size);
	charsiu_bo_fini(dev, &ob);

	charsiu_bo_prep(dev, &reg, 1000000000);
	memset(reg.map, 0, reg.size);
	for (unsigned i = 0; i < count; i++) {
		struct rk3588_mm sub;
		size_t nreg;

		rk3588_tile_mm(&mm, &tiles[i], &sub);
		nreg = rk3588_emit_matmul(&sub, (uint64_t *)((uint8_t *)reg.map + i * stride),
					  REG_WORDS);
		if (!nreg) {
			charsiu_bo_fini(dev, &reg);
			printf("tile %u refused\n", i);
			goto out;
		}
		tasks[i].regcmd = (uint32_t)(reg.dma_address + i * stride);
		tasks[i].regcmd_count = (uint32_t)nreg;
	}
	charsiu_bo_fini(dev, &reg);

	{
		uint32_t ins[3] = { in.handle, wt.handle, bs.handle };
		uint32_t outs[1] = { ob.handle };
		unsigned njobs = as_tasks ? 1 : count;

		for (unsigned i = 0; i < njobs; i++) {
			jobs[i].tasks = as_tasks ? tasks : &tasks[i];
			jobs[i].task_count = as_tasks ? count : 1;
			jobs[i].in_handles = ins;
			jobs[i].in_count = 3;
			jobs[i].out_handles = outs;
			jobs[i].out_count = 1;
		}

		for (int rep = -1; rep < TIMED_REPS; rep++) {
			double t0 = now_us();

			if (charsiu_submit_jobs(dev, jobs, njobs)) {
				printf("the submit failed\n");
				goto out;
			}
			if (charsiu_bo_prep(dev, &ob, 2000000000)) {
				printf("timed out waiting for the output\n");
				goto out;
			}
			if (rep >= 0)
				t[rep] = now_us() - t0;
			if (rep == -1) {
				size_t cells = (size_t)m * n;
				struct score s;

				rk3588_unpack_output(&mm, ob.map, C);
				reference(&mm, A, B, 0, R);
				s = compare(C, R, cells);
				printf("%7u exact %5u wrong  ", s.exact, s.wrong);
				if (!passes(s, cells)) {
					charsiu_bo_fini(dev, &ob);
					printf("FAIL\n");
					rc = 0;
					goto out;
				}
				reference(&mm, A, B, 1, X);
				if (compare(C, X, cells).wrong * 2 < cells) {
					charsiu_bo_fini(dev, &ob);
					printf("VOID (rotated reference does not fail)\n");
					rc = 0;
					goto out;
				}
			}
			charsiu_bo_fini(dev, &ob);
		}
	}
	qsort(t, TIMED_REPS, sizeof(t[0]), cmp_double);
	printf("PASS  median %7.0f us  [%5.0f .. %5.0f]  %5.2f GB/s of weights\n",
	       t[TIMED_REPS / 2], t[0], t[TIMED_REPS - 1],
	       (double)k * n / (t[TIMED_REPS / 2] * 1e3));
	rc = 1;
out:
	if (reg.handle) charsiu_bo_free(dev, &reg);
	if (ob.handle) charsiu_bo_free(dev, &ob);
	if (bs.handle) charsiu_bo_free(dev, &bs);
	if (wt.handle) charsiu_bo_free(dev, &wt);
	if (in.handle) charsiu_bo_free(dev, &in);
	free(A); free(B); free(C); free(R); free(X);
	return rc;
}

/*
 *   rk3588_gemm_test --tiled [K] [N] [tile N]
 *
 * Llama-3.2-1B's projection shapes by default, at M = 1 padded to h = 4 (the
 * shape decode needs, and the smallest height measured exact) and at M = 16.
 */
static int tiled_main(struct charsiu_device *dev, int argc, char **argv)
{
	static const unsigned shapes[][2] = {
		{ 2048, 2048 }, { 2048, 512 }, { 2048, 8192 }, { 8192, 2048 },
	};
	unsigned nt = argc > 4 ? (unsigned)atoi(argv[4]) : 0;
	int pass = 0, total = 0;

	printf("RK3588 emitter, N tiled, int8 in and out\n\n");
	for (unsigned s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
		unsigned k = argc > 2 ? (unsigned)atoi(argv[2]) : shapes[s][0];
		unsigned n = argc > 3 ? (unsigned)atoi(argv[3]) : shapes[s][1];

		for (int as_tasks = 0; as_tasks < 2; as_tasks++) {
			pass += run_tiled(dev, 1, 4, k, n, nt, as_tasks) == 1; total++;
			pass += run_tiled(dev, 4, 4, k, n, nt, as_tasks) == 1; total++;
		}
		if (argc > 2)
			break;
	}
	printf("\n%d of %d pass\n", pass, total);
	return 0;
}

int main(int argc, char **argv)
{
	unsigned k = argc > 1 ? (unsigned)atoi(argv[1]) : 256;
	unsigned n = argc > 2 ? (unsigned)atoi(argv[2]) : 64;
	static const unsigned ms[] = { 1, 2, 3, 4, 8, 16, 32 };
	struct charsiu_device *dev;
	int pass = 0, total = 0;

	if (argc > 1 && !strcmp(argv[1], "--version")) {
		printf("rk3588_gemm_test %s\n", CHARSIU_BUILD);
		return 0;
	}
	dev = charsiu_open(NULL);
	if (!dev) {
		fprintf(stderr, "no NPU: could not open an accel node\n");
		return 1;
	}
	if (argc > 1 && !strcmp(argv[1], "--tiled")) {
		int rc = tiled_main(dev, argc, argv);

		charsiu_close(dev);
		return rc;
	}
	printf("RK3588 emitter, int8 in and out, K=%u N=%u\n\n", k, n);

	printf("the control, Mesa's square shape:\n");
	if (run(dev, 4, 4, k, n, "control") != 1) {
		printf("\nTHE CONTROL DID NOT PASS. The emitter is wrong; nothing below "
		       "would mean anything, so nothing below runs.\n");
		charsiu_close(dev);
		return 2;
	}

	printf("\nM on the height (w = 1):\n");
	for (unsigned i = 0; i < sizeof(ms) / sizeof(ms[0]); i++, total++)
		pass += run(dev, 1, ms[i], k, n, "height") == 1;
	printf("\nM on the width (h = 1):\n");
	for (unsigned i = 0; i < sizeof(ms) / sizeof(ms[0]); i++, total++)
		pass += run(dev, ms[i], 1, k, n, "width") == 1;
	printf("\nM on the width, h = 4:\n");
	for (unsigned i = 0; i < 4; i++, total++)
		pass += run(dev, ms[i], 4, k, n, "width, h=4") == 1;

	printf("\n%d of %d shapes pass\n", pass, total);
	charsiu_close(dev);
	return 0;
}
