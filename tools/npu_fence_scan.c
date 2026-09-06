// SPDX-License-Identifier: GPL-2.0
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * WHAT DOES ONE DISPATCH COST, AS A FUNCTION OF ITS OUTPUT WIDTH?
 *
 * The batched prefill's fence is 1.35 ms a row on Llama and it is not the
 * weight fetch (doubling the chunk halves the passes over the weights and it
 * does not move), not the two cores being serialised (worth 2.7x and the
 * default already takes it), and not CPU idle exit (the PM QoS hold forbids
 * it). Bucketing the fence by tensor width inside a real forward pass said the
 * time per dispatch is proportional to n, at about 1.3e-4 ms per output channel
 * on two models -- but the buckets there mix tensors with different numbers of
 * K slices, and a device is charged a whole fence even on the calls where the
 * deal gave it no slot. That is a fit over a composition this tree does not
 * control, and four fitted rules died in one day for exactly that.
 *
 * So: one dispatch, one device, k and m FIXED, and only n moving.
 *
 * ⚠⚠ THE BUFFERS ARE ALLOCATED ONCE, AT THE WIDEST n IN THE SWEEP, AND REUSED.
 * This tree has already measured an allocation and called it a matmul: the
 * batched output buffer used to be allocated per tensor and the counter said
 * 225 allocations and 652 ms at ONE width, 36% of an 1811 ms round. A sweep
 * that reallocates per point measures malloc and mmap and IOVA, and the shape
 * of that is also "bigger n costs more".
 *
 * ⚠ AND THE FENCE STILL CONTAINS AN INVALIDATE. rocket_ioctl_prep_bo is a
 * dma_resv wait followed by dma_sync_sgtable_for_cpu over the WHOLE buffer, so
 * with one buffer sized for the widest point that invalidate is a CONSTANT
 * across the sweep rather than something that grows with n. That is deliberate:
 * a constant lands in the intercept, and the slope is then the part that is
 * really about the width. Both are printed.
 *
 * ⚠⚠ AND IT REPEATS ONE SHAPE, WHICH THIS TREE HAS WARNED ABOUT IN WRITING.
 *
 * bench_batch's own header says: "the first version looped on one tensor 200
 * times, which left it in cache and measured arithmetic rather than memory".
 * This loops one shape `reps` times. So its slope is a WARM number, and the
 * board says so: round 165 measured the marginal cost of a task tracking the
 * WEIGHT BYTES at close to 10 GB/s, which at k bytes an output channel would
 * make the slope double every time k doubles. It does not --
 *
 *      k = 512    0.141 us a channel   2.75x what 10 GB/s would cost
 *      k = 1024   0.152                1.48x
 *      k = 2048   0.208                1.02x
 *
 * -- so only the widest k here is paying a cold weight fetch, and the narrow
 * ones are being served from somewhere closer. The INTERCEPT is not affected by
 * that (a fixed cost per submit is paid warm or cold, and round 165 put it at
 * 180 to 195 us at M = 1 against the 118 to 305 this measures across m), but
 * the SLOPE understates a first, cold dispatch.
 *
 * Walking a model's layers the way bench_batch does is the fix, and it is not
 * done here.
 *
 *   npu_fence_scan [k] [m] [reps]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "charsiu.h"

static double us(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static const unsigned NS[] = { 256, 512, 1024, 2048, 3072, 4096, 6144, 8192 };
#define NN (sizeof(NS) / sizeof(NS[0]))

int main(int argc, char **argv)
{
	unsigned k = argc > 1 ? (unsigned)atoi(argv[1]) : 1024;
	unsigned m = argc > 2 ? (unsigned)atoi(argv[2]) : 80;
	unsigned reps = argc > 3 ? (unsigned)atoi(argv[3]) : 20;
	unsigned nmax = NS[NN - 1], i, r;
	struct charsiu_device *dev;
	struct charsiu_bo wt = { 0 }, in = { 0 }, ob = { 0 }, coef = { 0 },
			  reg = { 0 };
	uint8_t *A = NULL, *B = NULL;
	int32_t *zero = NULL;
	double sub[NN], fen[NN];
	size_t insz;
	int rc = 1;

	dev = charsiu_open(NULL);
	if (!dev) { fprintf(stderr, "no accel device\n"); return 1; }
	printf("one dispatch, k=%u m=%u, %u repeats a point, buffers allocated"
	       " once at n=%u\n", k, m, reps, nmax);

	/* sized for the widest point, so nothing here moves during the sweep */
	{
		struct charsiu_job big = { 0 };

		big.mm.m = m; big.mm.k = k; big.mm.n = nmax;
		big.mm.wdtype = CHARSIU_INT8;
		big.mm.adtype = CHARSIU_INT8;
		insz = (size_t)charsiu_entries_per_row(&big.mm) * 64 * m + 4096;
		if (charsiu_bo_alloc(dev, charsiu_weight_bytes(&big.mm) + 4096, &wt) ||
		    charsiu_bo_alloc(dev, insz, &in) ||
		    charsiu_bo_alloc(dev, (size_t)m * nmax * 4 + 4096, &ob) ||
		    charsiu_bo_alloc(dev, charsiu_coef_bytes(&big.mm) + 4096, &coef) ||
		    charsiu_bo_alloc(dev, 4096, &reg)) {
			fprintf(stderr, "a buffer would not allocate\n");
			goto out;
		}
	}
	A = malloc((size_t)m * k);
	B = malloc((size_t)k * nmax);
	zero = calloc(nmax, sizeof(int32_t));
	if (!A || !B || !zero) { fprintf(stderr, "out of memory\n"); goto out; }
	for (size_t q = 0; q < (size_t)m * k; q++) A[q] = (uint8_t)(q * 7 + 1);
	for (size_t q = 0; q < (size_t)k * nmax; q++) B[q] = (uint8_t)(q * 5 + 3);

	printf("\n  %6s %11s %11s %14s\n", "n", "submit us", "fence us",
	       "fence us per n");
	for (i = 0; i < NN; i++) {
		struct charsiu_job job = { 0 };
		size_t nreg;

		job.cbuf_window = (unsigned)charsiu_cbuf_window();
		job.mm.m = m; job.mm.k = k; job.mm.n = NS[i];
		job.mm.wdtype = CHARSIU_INT8;
		job.mm.adtype = CHARSIU_INT8;
		job.input_zero_point = 128;
		job.weight_zero_point = 128;
		job.input_scale = job.weight_scale = job.output_scale = 1.0f;
		job.acc_out = 1;

		charsiu_bo_prep(dev, &wt, 1000000000);
		memset(wt.map, 0, charsiu_weight_bytes(&job.mm));
		charsiu_pack_weights(&job.mm, B, wt.map);
		charsiu_bo_fini(dev, &wt);
		charsiu_bo_prep(dev, &in, 1000000000);
		charsiu_pack_input(&job.mm, A, in.map, insz, 128);
		charsiu_bo_fini(dev, &in);
		charsiu_bo_prep(dev, &coef, 1000000000);
		charsiu_build_coefs(&job, zero, zero, coef.map);
		charsiu_bo_fini(dev, &coef);

		job.input_addr = (uint32_t)in.dma_address;
		job.output_addr = (uint32_t)ob.dma_address;
		job.weight_addr = (uint32_t)wt.dma_address;
		job.coef_addr = (uint32_t)coef.dma_address;
		charsiu_bo_prep(dev, &reg, 1000000000);
		nreg = charsiu_emit_job(&job, reg.map, 4096 / 8);
		charsiu_bo_fini(dev, &reg);
		if (!nreg) { printf("  %6u  the stream came back empty\n", NS[i]); continue; }

		sub[i] = fen[i] = 0.0;
		for (r = 0; r < reps; r++) {
			uint32_t ins[2] = { in.handle, wt.handle };
			uint32_t outs[1] = { ob.handle };
			double t0 = us(), t1;

			if (charsiu_submit(dev, &reg, (unsigned)nreg, ins, 2,
					   outs, 1)) {
				printf("  %6u  the submit failed\n", NS[i]);
				break;
			}
			t1 = us();
			charsiu_bo_prep(dev, &ob, 2000000000);
			/* ⚠ the first repeat warms whatever the first repeat
			 * warms; it is dropped rather than explained */
			if (r) { sub[i] += t1 - t0; fen[i] += us() - t1; }
			charsiu_bo_fini(dev, &ob);
		}
		if (reps > 1) {
			sub[i] /= (reps - 1);
			fen[i] /= (reps - 1);
			printf("  %6u %11.1f %11.1f %14.4f\n", NS[i], sub[i],
			       fen[i], fen[i] / NS[i]);
		}
	}
	printf("\n  a flat last column means the dispatch is paid by output"
	       " channel;\n  one that falls means there is a fixed cost the wide"
	       " points amortise.\n");
	rc = 0;
out:
	free(A); free(B); free(zero);
	charsiu_bo_free(dev, &reg); charsiu_bo_free(dev, &coef);
	charsiu_bo_free(dev, &ob); charsiu_bo_free(dev, &in);
	charsiu_bo_free(dev, &wt);
	charsiu_close(dev);
	return rc;
}
