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

static int run_core(struct charsiu_device *dev, unsigned m, unsigned k,
		    unsigned n, const float *A, const float *B,
		    enum charsiu_w16_layout L, const uint8_t *W, size_t wlen,
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
	if (W) {                        /* the slot sweep supplies its own */
		memset(wt.map, 0, wsz);
		memcpy(wt.map, W, wlen < wsz ? wlen : wsz);
	} else {
		charsiu_pack_weights_f16(&job.mm, B, wt.map, wsz, L);
	}
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

static int run(struct charsiu_device *dev, unsigned m, unsigned k, unsigned n,
	       const float *A, const float *B, enum charsiu_w16_layout L,
	       uint32_t *out)
{
	return run_core(dev, m, k, n, A, B, L, NULL, 0, out);
}

/* the weight buffer supplied verbatim, for the slot sweep */
static int run_raw(struct charsiu_device *dev, unsigned m, unsigned k,
		   unsigned n, const float *A, const uint8_t *W, size_t wlen,
		   uint32_t *out)
{
	return run_core(dev, m, k, n, A, NULL, CHARSIU_W16_DENSE, W, wlen, out);
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
	int doslots = argc > 3 && !strcmp(argv[3], "--slots");
	int dobits = argc > 3 && !strcmp(argv[3], "--bits");
	int doholes = argc > 3 && !strcmp(argv[3], "--holes");
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
	if (doholes) {
		/*
		 * ⚠⚠ THE ONE HOT PROBE WAS NOT VALID AND --bits IS WHY.
		 *
		 * --slots puts ONE non zero half in the weight buffer and
		 * leaves the other 99.99% zero. Its firing set moved from run
		 * to run and covered 12 slots of 128, which read as a coverage
		 * defect. --bits then showed every channel summing all sixteen
		 * k, three runs running, at three shapes: coverage is COMPLETE.
		 * So the sparse buffer was the problem, not the hardware -- a
		 * weight fetch that skips zero blocks would behave exactly like
		 * that, and this silicon is documented to have sparsity.
		 *
		 * --bits cannot answer the layout on its own either: with every
		 * weight 1.0 the sum is the same under ANY permutation.
		 *
		 * So: every weight 1.0 except ONE HOLE, and A[k] = 2^k. The
		 * buffer stays dense, and the channel that comes back missing a
		 * bit names both halves of the hole -- the channel is n and the
		 * missing bit is k. One run per slot, and the answer does not
		 * depend on any candidate layout.
		 */
		unsigned kk = k > 16 ? 16 : k;
		unsigned full = (1u << kk) - 1u;
		struct charsiu_matmul mm = { m, k, n, CHARSIU_FP16, CHARSIU_FP16 };
		size_t wsz = charsiu_weight_bytes(&mm);
		unsigned slots = (unsigned)(wsz / 2);
		uint8_t *raw = calloc(wsz, 1);
		uint16_t one = charsiu_float_to_half(1.0f);

		if (!raw) goto done;
		for (unsigned i = 0; i < k; i++)
			A[i] = i < kk ? (float)(1u << i) : 0.0f;
		printf("every weight 1.0 except one hole; A[k] = 2^k; full = 0x%x\n",
		       full);
		printf("  hole slot  -> channel : missing bit (= k)\n");
		for (unsigned sl = 0; sl < slots; sl++) {
			int said = 0;

			for (size_t i = 0; i < wsz; i += 2) {
				raw[i] = (uint8_t)(one & 0xff);
				raw[i + 1] = (uint8_t)(one >> 8);
			}
			raw[sl * 2] = raw[sl * 2 + 1] = 0;
			if (run_raw(dev, m, k, n, A, raw, wsz, got))
				break;
			for (unsigned c = 0; c < n; c++) {
				float v = asf(got[c]);
				unsigned long b = (v >= 0 && v < 1e9 &&
						   v == (float)(unsigned long)v)
						  ? (unsigned long)v : ~0ul;

				if (b == full || b == ~0ul)
					continue;
				printf("  %-10u -> %u : 0x%lx", sl, c, full ^ b);
				if (__builtin_popcountl(full ^ b) == 1)
					printf(" so k=%d\n",
					       __builtin_ctzl(full ^ b));
				else
					printf("   (not one bit)\n");
				said = 1;
			}
			if (!said)
				printf("  %-10u -> no channel changed\n", sl);
		}
		free(raw);
		rc = 0;
		goto done;
	}
	if (dobits) {
		/*
		 * ⚠⚠ ONE RUN FOR THE WHOLE COVERAGE MAP.
		 *
		 * --slots needs a run per slot and each run draws a different
		 * subset, so 128 runs give 128 unrelated samples rather than one
		 * picture. Weights all 1.0 and A[k] = 2^k makes each channel's
		 * output a BITMASK of exactly which k reached it, and every
		 * channel is read from the same run. fp16 holds 2^k exactly to
		 * k = 15 (32768 < 65504) and the fp32 sum of distinct powers of
		 * two is exact, so the number that comes back is the answer and
		 * not a rounding of it.
		 */
		unsigned kk = k > 16 ? 16 : k;

		for (unsigned i = 0; i < k; i++)
			A[i] = i < kk ? (float)(1u << i) : 0.0f;
		for (unsigned i = 0; i < n * k; i++)
			B[i] = 1.0f;
		printf("weights all 1.0, A[k] = 2^k for k < %u; each channel's"
		       " output names the k it summed\n", kk);
		printf("  want every channel = 0x%x (%u terms)\n",
		       (1u << kk) - 1u, kk);
		for (int L = 0; L < CHARSIU_W16_NLAYOUT; L++) {
			if (run(dev, m, k, n, A, B, (enum charsiu_w16_layout)L, got))
				continue;
			printf("  %-6s", lname[L]);
			for (unsigned c = 0; c < n; c++) {
				float v = asf(got[c]);
				unsigned long b = (v >= 0 && v < 1e9 &&
						   v == (float)(unsigned long)v)
						  ? (unsigned long)v : 0;

				printf("  ch%u=%s%lx", c, b ? "0x" : "?", b);
			}
			printf("\n");
		}
		rc = 0;
		goto done;
	}
	if (doslots) {
		/*
		 * ⚠⚠ SWEEP THE BYTE OFFSET, NOT THE LOGICAL INDEX.
		 *
		 * --map places a weight at the (n, k) THIS layout chooses, so a
		 * cell only lights up where our layout already agrees with the
		 * hardware's. It cannot see what the hardware read the other
		 * cells as, which is the whole permutation. Four rounds of that
		 * gave twelve firing cells and no rule.
		 *
		 * This writes 1.0 into slot s of the buffer directly, so the
		 * channel it comes back in is the hardware's n and the value is
		 * A[k] for the hardware's k. That is H inverse, read off, with
		 * no candidate layout in the way.
		 */
		unsigned slots = (unsigned)(charsiu_weight_bytes(&(struct charsiu_matmul){
			m, k, n, CHARSIU_FP16, CHARSIU_FP16 }) / 2);
		struct charsiu_matmul mm = { m, k, n, CHARSIU_FP16, CHARSIU_FP16 };
		size_t wsz = charsiu_weight_bytes(&mm);
		uint8_t *raw = calloc(wsz, 1);

		for (unsigned i = 0; i < k; i++)
			A[i] = (float)(i + 1);
		if (!raw) goto done;
		printf("one 1.0 per SLOT, %u slots of %zu bytes; A[k] = k+1\n",
		       slots, wsz);
		printf("  slot  byte   -> channel : A[k], so k\n");
		for (unsigned sl = 0; sl < slots; sl++) {
			uint16_t h = charsiu_float_to_half(1.0f);
			int lit = 0;

			memset(raw, 0, wsz);
			raw[sl * 2] = (uint8_t)(h & 0xff);
			raw[sl * 2 + 1] = (uint8_t)(h >> 8);
			if (run_raw(dev, m, k, n, A, raw, wsz, got))
				break;
			for (unsigned c = 0; c < n; c++)
				if (got[c]) {
					float v = asf(got[c]);

					printf("  %-5u %-6u -> %u : %g%s\n", sl,
					       sl * 2, c, (double)v,
					       (v >= 1 && v <= (float)k &&
						v == (float)(int)v)
					       ? "" : "   (not a ramp value)");
					lit = 1;
				}
			if (!lit)
				printf("  %-5u %-6u -> nothing\n", sl, sl * 2);
		}
		free(raw);
		rc = 0;
		goto done;
	}
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

	/*
	 * ⚠⚠ (i % 13) - 6 ON AN UNSIGNED i IS 4 BILLION, NOT -6, and the first
	 * run of this probe was entirely that. The data came out ~1e9, fp16
	 * turned it into inf, inf + -inf made the reference NaN, and the
	 * hardware wrote 0x7f7f7f7f. Cast before subtracting.
	 */
	for (unsigned i = 0; i < m * k; i++)
		A[i] = (float)((int)(i % 13) - 6) * 0.25f;
	for (unsigned i = 0; i < n * k; i++)
		B[i] = (float)((int)(i % 7) - 3) * 0.5f;
	reference(m, k, n, A, B, ref);
	for (unsigned i = 0; i < m * n; i++)
		if (!isfinite(ref[i])) {
			printf("  the CPU reference is not finite at %u"
			       " -- the INPUTS are wrong, not the hardware."
			       " Refusing to compare.\n", i);
			goto done;
		}

	for (int L = 0; L < CHARSIU_W16_NLAYOUT; L++) {
		double worst = 0;
		int any = 0, bad = 0;

		if (run(dev, m, k, n, A, B, (enum charsiu_w16_layout)L, got)) {
			printf("  %-6s could not run\n", lname[L]);
			continue;
		}
		/*
		 * ⚠⚠ A NaN MUST NOT SCORE ZERO. `d > worst` is false when d is
		 * NaN, so the first version of this loop reported a worst error
		 * of 0 -- a perfect match -- for a run whose reference was
		 * entirely NaN, on all three layouts at once. A probe that
		 * cannot measure has to say so, not agree with you.
		 */
		for (unsigned i = 0; i < m * n; i++) {
			double d = fabs(asf(got[i]) - ref[i]);

			if (got[i]) any = 1;
			if (!isfinite(d)) { bad = 1; continue; }
			if (d > worst) worst = d;
		}
		if (bad)
			printf("  %-6s the hardware returned something that is "
			       "not a finite float -- NOT a match\n", lname[L]);
		else
			printf("  %-6s worst |fp32 read - reference| %.4g%s%s\n",
			       lname[L], worst,
			       any ? "" : "   (every output zero)",
			       worst == 0.0 && any ? "   <== EXACT" : "");
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
