// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * One fp16-weight matmul on the hardware, to settle the one thing the vendor's
 * model file cannot say: what order the bytes of a 16 bit weight buffer go in.
 *
 * Everything else is already read off the file and written down in
 * docs/vendor-dispatch.md -- the register set is the same as int4's, five
 * values differ, and the buffer is ic*oc*2 bytes with ic*2 per output channel.
 * A tiling does not change a total, so the byte ORDER is invisible from
 * outside and this asks the silicon.
 *
 * Two modes:
 *
 *   (default)  pack the same random B under each candidate layout, run it, and
 *              compare against a CPU fp16 reference. If one is exact, that is
 *              the layout and no map is needed.
 *   --map      one weight at a time: B is zero except (n, k) = 1, A is a ramp,
 *              so a correct dispatch puts A[k] in output channel n and nothing
 *              anywhere else. What actually lights up IS the permutation, and
 *              it is printed whether or not it matches a candidate.
 *
 * ⚠ THE ACCUMULATOR IS PRINTED BOTH WAYS. acc_out gives the raw accumulator
 * and nothing here has established whether an fp16 job accumulates in int32 or
 * in fp32. Reading it as one and not saying so is how a probe reports "wrong"
 * when what it means is "I do not know how to read this".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "charsiu.h"

static const char *lname[CHARSIU_W16_NLAYOUT] = { "dense", "atom", "group" };

static int run(struct charsiu_device *dev, unsigned m, unsigned k, unsigned n,
	       const float *A, const float *B, enum charsiu_w16_layout L,
	       uint32_t *out)
{
	struct charsiu_job job = { 0 };
	struct charsiu_bo wt = { 0 }, in = { 0 }, ob = { 0 }, coef = { 0 }, reg = { 0 };
	size_t nreg, insz, wsz;
	int rc = -1;

	job.cbuf_window = (unsigned)charsiu_cbuf_window();
	job.mm.m = m; job.mm.k = k; job.mm.n = n;
	job.mm.wdtype = CHARSIU_FP16;
	job.mm.adtype = CHARSIU_FP16;
	job.input_zero_point = 0;      /* a float carries its own sign */
	job.weight_zero_point = 0;
	job.output_zero_point = 0;
	job.input_scale = 1.0f;
	job.weight_scale = 1.0f;
	job.output_scale = 1.0f;
	job.acc_out = 1;

	wsz = charsiu_weight_bytes(&job.mm);
	insz = (size_t)charsiu_entries_per_row(&job.mm) * 64 * m + 4096;
	if (charsiu_bo_alloc(dev, wsz + 4096, &wt) ||
	    charsiu_bo_alloc(dev, insz, &in) ||
	    charsiu_bo_alloc(dev, (size_t)m * n * 4 + 4096, &ob) ||
	    charsiu_bo_alloc(dev, charsiu_coef_bytes(&job.mm) + 4096, &coef) ||
	    charsiu_bo_alloc(dev, 4096, &reg)) {
		fprintf(stderr, "  a buffer would not allocate\n");
		goto out;
	}
	if (!wt.map || !in.map || !ob.map || !coef.map || !reg.map) {
		fprintf(stderr, "  a buffer allocated but did not map\n");
		goto out;
	}

	charsiu_bo_prep(dev, &wt, 1000000000);
	charsiu_pack_weights_f16(&job.mm, B, wt.map, wsz, L);
	charsiu_bo_fini(dev, &wt);

	charsiu_bo_prep(dev, &in, 1000000000);
	charsiu_pack_input_f16(&job.mm, A, in.map, insz);
	charsiu_bo_fini(dev, &in);

	{
		int32_t *zero = calloc(n, sizeof(int32_t));

		if (!zero) { fprintf(stderr, "  out of memory\n"); goto out; }
		charsiu_bo_prep(dev, &coef, 1000000000);
		charsiu_build_coefs(&job, zero, zero, coef.map);
		charsiu_bo_fini(dev, &coef);
		free(zero);
	}

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

/* the same arithmetic on the CPU, rounded through fp16 the way the pack is */
static void reference(unsigned m, unsigned k, unsigned n,
		      const float *A, const float *B, float *out)
{
	for (unsigned r = 0; r < m; r++)
		for (unsigned c = 0; c < n; c++) {
			float acc = 0;

			for (unsigned i = 0; i < k; i++)
				acc += charsiu_half_to_float(charsiu_float_to_half(A[(size_t)r * k + i]))
				     * charsiu_half_to_float(charsiu_float_to_half(B[(size_t)c * k + i]));
			out[(size_t)r * n + c] = acc;
		}
}

static float asf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char **argv)
{
	unsigned k = argc > 1 ? (unsigned)atoi(argv[1]) : 64;
	unsigned n = argc > 2 ? (unsigned)atoi(argv[2]) : 8;
	unsigned m = 1;
	int domap = argc > 3 && !strcmp(argv[3], "--map");
	struct charsiu_device *dev = charsiu_open(NULL);
	float *A, *B, *ref;
	uint32_t *got;
	int rc = 1;

	if (!dev) { fprintf(stderr, "no NPU device\n"); return 1; }
	A = calloc((size_t)m * k, sizeof(*A));
	B = calloc((size_t)n * k, sizeof(*B));
	ref = calloc((size_t)m * n, sizeof(*ref));
	got = calloc((size_t)m * n, sizeof(*got));
	if (!A || !B || !ref || !got) return 1;

	printf("fp16 weights: K=%u N=%u M=%u\n", k, n, m);
	if (domap) {
		/*
		 * ⚠ ONE NON ZERO WEIGHT AT A TIME. A is a ramp with no repeats,
		 * so the VALUE that comes back names the k it was fetched from
		 * and the channel it lands in names the n. That is the whole
		 * permutation, read directly, with no candidate assumed.
		 */
		for (unsigned i = 0; i < k; i++)
			A[i] = (float)(i + 1);
		printf("one weight at a time, A[k] = k+1; layout %s\n",
		       lname[CHARSIU_W16_DENSE]);
		printf("  put at      lit up (channel: value)\n");
		for (unsigned nn = 0; nn < n; nn++)
			for (unsigned kk = 0; kk < k; kk += (k > 16 ? k / 8 : 1)) {
				memset(B, 0, (size_t)n * k * sizeof(*B));
				B[(size_t)nn * k + kk] = 1.0f;
				if (run(dev, m, k, n, A, B, CHARSIU_W16_DENSE, got))
					goto done;
				printf("  n=%-3u k=%-5u", nn, kk);
				for (unsigned c = 0; c < n; c++)
					if (got[c])
						printf("  %u:%g/%g", c,
						       (double)(int32_t)got[c],
						       (double)asf(got[c]));
				printf("\n");
			}
		rc = 0;
		goto done;
	}

	for (unsigned i = 0; i < m * k; i++)
		A[i] = (float)((i % 13) - 6) * 0.25f;
	for (unsigned i = 0; i < n * k; i++)
		B[i] = (float)((i % 7) - 3) * 0.5f;
	reference(m, k, n, A, B, ref);

	for (int L = 0; L < CHARSIU_W16_NLAYOUT; L++) {
		double worst = 0;
		int any = 0;

		if (run(dev, m, k, n, A, B, (enum charsiu_w16_layout)L, got)) {
			printf("  %-6s could not run\n", lname[L]);
			continue;
		}
		for (unsigned i = 0; i < m * n; i++) {
			double d = fabs(asf(got[i]) - ref[i]);

			if (got[i]) any = 1;
			if (d > worst) worst = d;
		}
		printf("  %-6s worst |fp32 read - reference| %.4g%s\n",
		       lname[L], worst, any ? "" : "   (every output zero)");
		printf("         first four: raw %08x %08x %08x %08x\n",
		       got[0], m * n > 1 ? got[1] : 0, m * n > 2 ? got[2] : 0,
		       m * n > 3 ? got[3] : 0);
		printf("         as float   %g %g %g %g   want %g %g %g %g\n",
		       (double)asf(got[0]), (double)asf(m * n > 1 ? got[1] : 0),
		       (double)asf(m * n > 2 ? got[2] : 0),
		       (double)asf(m * n > 3 ? got[3] : 0),
		       (double)ref[0], (double)(m * n > 1 ? ref[1] : 0),
		       (double)(m * n > 2 ? ref[2] : 0),
		       (double)(m * n > 3 ? ref[3] : 0));
	}
	rc = 0;
done:
	charsiu_close(dev);
	free(A); free(B); free(ref); free(got);
	return rc;
}
