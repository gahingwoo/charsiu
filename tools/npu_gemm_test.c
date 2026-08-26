// SPDX-License-Identifier: GPL-2.0
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * Does the NPU compute a matmul with MORE THAN ONE ROW of A?
 *
 * charsiu has only ever asked for m=1: npudev.c pins `s->job.mm.m = 1` and
 * every token is a fresh matrix-vector product, so a prompt reads every weight
 * once per token. That is why prefill throughput is flat with prompt length.
 *
 * The job format is a GEMM -- `struct charsiu_matmul` carries m, and
 * charsiu_pack_input has a general m>1 path with the m==1 fast path beside it
 * -- and the emitter differentiates: m=2 doubles the input surface width in
 * 0x1028 and sets the height fields to 1. None of that has ever been run.
 *
 * ⚠ m=1 IS THE CONTROL AND IT RUNS FIRST. If m=1 disagrees with the CPU then
 * this test is wrong and says so, rather than blaming a field it was built to
 * examine. Only m=1 passing makes an m=2 failure mean anything.
 *
 *   npu_gemm_test [K] [N]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu.h"

static int run(struct charsiu_device *dev, unsigned m, unsigned k, unsigned n,
	       const uint8_t *A, const uint8_t *B, int32_t *out)
{
	struct charsiu_job job = { 0 };
	struct charsiu_bo wt = { 0 }, in = { 0 }, ob = { 0 }, coef = { 0 }, reg = { 0 };
	size_t nreg, insz;
	int rc = -1;

	job.cbuf_window = (unsigned)charsiu_cbuf_window();
	job.mm.m = m; job.mm.k = k; job.mm.n = n;
	job.mm.wdtype = CHARSIU_INT8;
	job.mm.adtype = CHARSIU_INT8;
	job.input_zero_point = 128;
	job.weight_zero_point = 128;
	job.output_zero_point = 0;
	job.input_scale = 1.0f;
	job.weight_scale = 1.0f;
	job.output_scale = 1.0f;
	job.acc_out = 1;              /* the raw int32 accumulator */

	/*
	 * ⚠ entries_per_row IS PER ROW, and the packed input holds m of them
	 * interleaved: [K/atom][M][atom]. Sizing it without the m factor gave a
	 * 64 byte buffer for a 128 byte pack, which only survived on the slack
	 * and would have failed on the board as something mysterious. Verified
	 * on the host: with the m factor, 0 of 128 elements land outside.
	 */
	insz = (size_t)charsiu_entries_per_row(&job.mm) * 64 * m + 4096;
	if (charsiu_bo_alloc(dev, charsiu_weight_bytes(&job.mm) + 4096, &wt) ||
	    charsiu_bo_alloc(dev, insz, &in) ||
	    charsiu_bo_alloc(dev, (size_t)m * n * 4 + 4096, &ob) ||
	    charsiu_bo_alloc(dev, charsiu_coef_bytes(&job.mm) + 4096, &coef) ||
	    charsiu_bo_alloc(dev, 4096, &reg)) {
		fprintf(stderr, "  a buffer would not allocate\n");
		goto out;
	}

	charsiu_bo_prep(dev, &wt, 1000000000);
	memset(wt.map, 0, charsiu_weight_bytes(&job.mm));
	charsiu_pack_weights(&job.mm, B, wt.map);
	charsiu_bo_fini(dev, &wt);

	charsiu_bo_prep(dev, &in, 1000000000);
	charsiu_pack_input(&job.mm, A, in.map, insz, job.input_zero_point);
	charsiu_bo_fini(dev, &in);

	charsiu_bo_prep(dev, &coef, 1000000000);
	charsiu_build_coefs(&job, NULL, NULL, coef.map);
	charsiu_bo_fini(dev, &coef);

	job.input_addr = (uint32_t)in.dma_address;
	job.output_addr = (uint32_t)ob.dma_address;
	job.weight_addr = (uint32_t)wt.dma_address;
	job.coef_addr = (uint32_t)coef.dma_address;

	charsiu_bo_prep(dev, &reg, 1000000000);
	nreg = charsiu_emit_job(&job, reg.map, 4096 / 8);
	charsiu_bo_fini(dev, &reg);
	if (!nreg) {
		fprintf(stderr, "  the register stream came back empty\n");
		goto out;
	}

	charsiu_bo_prep(dev, &ob, 1000000000);
	memset(ob.map, 0, (size_t)m * n * 4);
	charsiu_bo_fini(dev, &ob);

	{
		uint32_t ins[2] = { in.handle, wt.handle };
		uint32_t outs[1] = { ob.handle };

		if (charsiu_submit(dev, &reg, (unsigned)nreg, ins, 2, outs, 1)) {
			fprintf(stderr, "  the submit failed\n");
			goto out;
		}
	}
	charsiu_bo_prep(dev, &ob, 1000000000);
	memcpy(out, ob.map, (size_t)m * n * 4);
	charsiu_bo_fini(dev, &ob);
	rc = 0;
out:
	charsiu_bo_free(dev, &reg); charsiu_bo_free(dev, &coef);
	charsiu_bo_free(dev, &ob); charsiu_bo_free(dev, &in);
	charsiu_bo_free(dev, &wt);
	return rc;
}

/* what the hardware is being asked for: (a - zp) . (b - zp) per output */
static void reference(unsigned m, unsigned k, unsigned n,
		      const uint8_t *A, const uint8_t *B, int32_t *out)
{
	for (unsigned r = 0; r < m; r++)
		for (unsigned c = 0; c < n; c++) {
			int32_t acc = 0;

			for (unsigned i = 0; i < k; i++)
				acc += ((int)A[(size_t)r * k + i] - 128)
				     * ((int)B[(size_t)c * k + i] - 128);
			out[(size_t)r * n + c] = acc;
		}
}

static int check(const char *what, unsigned m, unsigned n,
		 const int32_t *got, const int32_t *want)
{
	unsigned bad = 0;
	long worst = 0;

	for (size_t i = 0; i < (size_t)m * n; i++) {
		long d = (long)got[i] - (long)want[i];

		if (d) { bad++; if (labs(d) > labs(worst)) worst = d; }
	}
	printf("  %-22s %5u of %5u wrong, worst delta %ld\n",
	       what, bad, m * n, worst);
	if (bad) {
		printf("      first few  got:");
		for (unsigned i = 0; i < 6 && i < m * n; i++) printf(" %d", got[i]);
		printf("\n                 want:");
		for (unsigned i = 0; i < 6 && i < m * n; i++) printf(" %d", want[i]);
		printf("\n");
	}
	return bad != 0;
}

int main(int argc, char **argv)
{
	unsigned k = argc > 1 ? (unsigned)atoi(argv[1]) : 256;
	unsigned n = argc > 2 ? (unsigned)atoi(argv[2]) : 64;
	struct charsiu_device *dev;
	uint8_t *A, *B;
	int32_t *got, *want;
	int fail = 0;

	dev = charsiu_open(NULL);
	if (!dev) {
		fprintf(stderr, "no /dev/accel/accel0\n");
		return 77;
	}
	A = malloc((size_t)2 * k);
	B = malloc((size_t)n * k);
	got = malloc((size_t)2 * n * 4);
	want = malloc((size_t)2 * n * 4);
	/* ⚠ THE TWO ROWS OF A MUST DIFFER, or m=2 could be right by copying */
	for (unsigned i = 0; i < k; i++) {
		A[i]     = (uint8_t)(128 + (int)(i % 7) - 3);
		A[k + i] = (uint8_t)(128 - (int)(i % 5) + 2);
	}
	for (unsigned c = 0; c < n; c++)
		for (unsigned i = 0; i < k; i++)
			B[(size_t)c * k + i] = (uint8_t)(128 + (int)((c + i) % 9) - 4);

	printf("K=%u N=%u, int8 weights and activations, raw accumulator\n", k, n);

	reference(1, k, n, A, B, want);
	if (run(dev, 1, k, n, A, B, got))
		fail = 1;
	else
		fail |= check("m=1  (the control)", 1, n, got, want);

	if (fail) {
		printf("\n  m=1 already disagrees, so this test cannot say anything\n"
		       "  about m=2. Fix the control first.\n");
		charsiu_close(dev);
		return 1;
	}

	reference(2, k, n, A, B, want);
	if (run(dev, 2, k, n, A, B, got))
		fail = 1;
	else
		fail |= check("m=2  (the question)", 2, n, got, want);

	printf("\n  %s\n", fail
	       ? "m=2 is NOT usable as it stands: a batched prefill would be wrong."
	       : "m=2 computes correctly. A batched prefill has somewhere to go.");
	charsiu_close(dev);
	return fail;
}
