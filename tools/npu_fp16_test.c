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
 * ⚠⚠⚠ USE A REAL SHAPE. K=16 N=8 AND K=64 N=8 WEDGE THE NPU.
 *
 * Measured 2026-09-05, --loop 32, the same job every time:
 *
 *     K=16  N=8     2 of 32 wrote
 *     K=64  N=8    24 of 512 wrote (the --holes sweep)
 *     K=64  N=64   32 of 32 wrote, dmesg clean
 *     K=256 N=64   32 of 32 wrote, dmesg clean
 *
 * and on the small shapes dmesg carries "rocket: NPU job timed out" for BOTH
 * cores followed by "rk_iommu: Error during raw reset. MMU_DTE_ADDR is not
 * functioning", which is the reset defect this project already has on record.
 * Two jobs get through -- one a core -- and then nothing until the timeout,
 * which is what "two writes then eighteen silent, repeating" was all along.
 *
 * The board is fine: npu_gemm_test at 256x64 runs six times over with zero
 * timeouts, and a model decodes normally.
 *
 * ⚠ THIS COST SIX WRONG EXPLANATIONS. A coverage defect, a zero-skipping
 * weight fetch, a single zero killing the job, an fp16 register wedging the
 * core, the buffer churn, a read that beat the fence -- every one of them a
 * story about the thing under test, while the parameter that was actually
 * responsible was the one I never moved. Two things would have found it in one
 * round: reading dmesg, and varying the shape. npu_gemm_test has always used
 * 256x64 and has never shown this, which was sitting there the whole time.
 *
 * ⚠ THE ACCUMULATOR IS PRINTED BOTH WAYS. acc_out gives the raw accumulator
 * and nothing here has established whether an fp16 job accumulates in int32 or
 * in fp32. Reading it as one and not saying so is how a probe reports "wrong"
 * when what it means is "I do not know how to read this".
 */
/* clock_gettime: the Makefile builds at -std=c11, which hides it */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#include "charsiu.h"

/*
 * ⚠ WHERE THE 2 ms GOES, because without this the loop times the PROBE and
 * calls it the hardware. K=512 N=64, K=1024 N=64 and K=64 N=512 all came back
 * 2.09 to 2.28 ms, which is nearly independent of both dimensions -- the shape
 * of a fixed cost, not of a matmul. The pack alone walks k*n elements through
 * charsiu_w16_offset one at a time.
 */
static struct { double pack, cf, emit, run, rd; } t_split;

static double now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static const char *lname[CHARSIU_W16_NLAYOUT] = { "dense", "atom", "group" };

/*
 * ⚠⚠ THE BUFFERS ARE ALLOCATED ONCE, AND THE BOARD IS WHY.
 *
 * This allocated five buffer objects, submitted, and freed them, every call.
 * Run the SAME job 128 times that way and only about twelve of them write
 * anything: the pattern is two writes, then eighteen to twenty-two silent,
 * repeating, and it is IDENTICAL for an int8 job (CHARSIU_TEST_INT8), so it is
 * the churn and not the precision.
 *
 * That cost four wrong explanations before the control existed -- a coverage
 * defect, a zero-skipping weight fetch, a single zero killing the job, and an
 * fp16 register wedging the core. Every one of them was a story about a
 * variable that was never moving.
 *
 * The buffers now live across calls, sized for the largest shape asked for.
 */
static struct {
	struct charsiu_device *dev;
	struct charsiu_bo wt, in, ob, coef, reg;
	size_t wsz, insz, obsz, coefsz;
} pool;

static void pool_free(void)
{
	if (!pool.dev)
		return;
	charsiu_bo_free(pool.dev, &pool.reg);  charsiu_bo_free(pool.dev, &pool.coef);
	charsiu_bo_free(pool.dev, &pool.ob);   charsiu_bo_free(pool.dev, &pool.in);
	charsiu_bo_free(pool.dev, &pool.wt);
	memset(&pool, 0, sizeof(pool));
}

static int pool_want(struct charsiu_device *dev, size_t wsz, size_t insz,
		     size_t obsz, size_t coefsz)
{
	if (pool.dev == dev && pool.wsz >= wsz && pool.insz >= insz &&
	    pool.obsz >= obsz && pool.coefsz >= coefsz)
		return 0;
	pool_free();
	pool.dev = dev;
	pool.wsz = wsz; pool.insz = insz; pool.obsz = obsz; pool.coefsz = coefsz;
	if (charsiu_bo_alloc(dev, wsz, &pool.wt) ||
	    charsiu_bo_alloc(dev, insz, &pool.in) ||
	    charsiu_bo_alloc(dev, obsz, &pool.ob) ||
	    charsiu_bo_alloc(dev, coefsz, &pool.coef) ||
	    charsiu_bo_alloc(dev, 4096, &pool.reg)) {
		fprintf(stderr, "  a buffer would not allocate\n");
		pool_free();
		return -1;
	}
	if (!pool.wt.map || !pool.in.map || !pool.ob.map || !pool.coef.map ||
	    !pool.reg.map) {
		fprintf(stderr, "  a buffer allocated but did not map\n");
		pool_free();
		return -1;
	}
	return 0;
}

static int run_core(struct charsiu_device *dev, unsigned m, unsigned k,
		    unsigned n, const float *A, const float *B,
		    enum charsiu_w16_layout L, const uint8_t *W, size_t wlen,
		    uint32_t *out)
{
	struct charsiu_job job = { 0 };
	size_t nreg, insz, wsz;
	double tp;
	int rc = -1;

	job.cbuf_window = (unsigned)charsiu_cbuf_window();
	job.mm.m = m; job.mm.k = k; job.mm.n = n;
	/*
	 * ⚠ THE CONTROL ARM. CHARSIU_TEST_INT8 runs this identical loop as an
	 * int8 job, which is the only way to say whether "an fp16 job writes
	 * nothing 89% of the time" is about fp16 or about submitting 128 jobs
	 * in one process with five BOs allocated and freed each time.
	 */
	int i8 = getenv("CHARSIU_TEST_INT8") != NULL;

	job.mm.wdtype = i8 ? CHARSIU_INT8 : CHARSIU_FP16;
	job.mm.adtype = i8 ? CHARSIU_INT8 : CHARSIU_FP16;
	job.input_zero_point = i8 ? 128 : 0;   /* a float carries its own sign */
	job.weight_zero_point = i8 ? 128 : 0;
	job.output_zero_point = 0;
	job.input_scale = 1.0f;
	job.weight_scale = 1.0f;
	job.output_scale = 1.0f;
	job.acc_out = 1;

	wsz = charsiu_weight_bytes(&job.mm);
	insz = (size_t)charsiu_entries_per_row(&job.mm) * 64 * m + 4096;
	if (pool_want(dev, wsz + 4096, insz, (size_t)m * n * 4 + 4096,
		      charsiu_coef_bytes(&job.mm) + 4096))
		return -1;
#define wt   (pool.wt)
#define in   (pool.in)
#define ob   (pool.ob)
#define coef (pool.coef)
#define reg  (pool.reg)

	tp = now_ms();
	charsiu_bo_prep(dev, &wt, 1000000000);
	if (i8) {
		memset(wt.map, 0x80, wsz);   /* every weight the zero point */
	} else if (W) {                 /* the slot sweep supplies its own */
		memset(wt.map, 0, wsz);
		memcpy(wt.map, W, wlen < wsz ? wlen : wsz);
	} else {
		charsiu_pack_weights_f16(&job.mm, B, wt.map, wsz, L);
	}
	charsiu_bo_fini(dev, &wt);

	charsiu_bo_prep(dev, &in, 1000000000);
	if (i8) {
		uint8_t *a8 = malloc((size_t)m * k);

		if (!a8) goto out;
		for (size_t z = 0; z < (size_t)m * k; z++)
			a8[z] = 128;
		charsiu_pack_input(&job.mm, a8, in.map, insz,
				   job.input_zero_point);
		free(a8);
	} else {
		charsiu_pack_input_f16(&job.mm, A, in.map, insz);
	}
	charsiu_bo_fini(dev, &in);
	t_split.pack += now_ms() - tp;
	tp = now_ms();

	{
		int32_t *zero = calloc(n, sizeof(int32_t));

		if (!zero) { fprintf(stderr, "  out of memory\n"); goto out; }
		charsiu_bo_prep(dev, &coef, 1000000000);
		charsiu_build_coefs(&job, zero, zero, coef.map);
		charsiu_bo_fini(dev, &coef);
		free(zero);
	}

	t_split.cf += now_ms() - tp;
	tp = now_ms();
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

	/*
	 * ⚠⚠ A SENTINEL, NOT ZEROS. --holes reported the entire output zero for
	 * 116 of 128 single weight holes, and "the hardware computed zero" and
	 * "the job never wrote" are the same four bytes when the buffer starts
	 * at zero. This project has read the second as the first three times in
	 * a week. 0xdeadbeef survives only if nothing was written.
	 */
	t_split.emit += now_ms() - tp;
	tp = now_ms();
	charsiu_bo_prep(dev, &ob, 1000000000);
	for (unsigned i = 0; i < m * n; i++)
		((uint32_t *)ob.map)[i] = 0xdeadbeefu;
	charsiu_bo_fini(dev, &ob);
	{
		uint32_t ins[2] = { in.handle, wt.handle };
		uint32_t outs[1] = { ob.handle };

		if (charsiu_submit(dev, &reg, (unsigned)nreg, ins, 2, outs, 1)) {
			fprintf(stderr, "  the submit failed\n");
			goto out;
		}
	}
	t_split.run += now_ms() - tp;
	tp = now_ms();
	charsiu_bo_prep(dev, &ob, 1000000000);   /* this is the fence wait */
	memcpy(out, ob.map, (size_t)m * n * 4);
	charsiu_bo_fini(dev, &ob);
	t_split.rd += now_ms() - tp;
	rc = 0;
out:
	return rc;
#undef wt
#undef in
#undef ob
#undef coef
#undef reg
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
	int doloop = argc > 3 && !strcmp(argv[3], "--loop");
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
	if (doloop) {
		/*
		 * ⚠⚠ THE SAME JOB, N TIMES, COUNTING HOW MANY WRITE.
		 *
		 * --holes with a hole of 1.0 is no hole at all, and 116 of its
		 * 128 submits still wrote nothing. So the weight contents were
		 * never the variable and every "which slots fire" reading in
		 * this file was sampling a submit that succeeds about a tenth
		 * of the time. This measures that rate on its own, and
		 * CHARSIU_TEST_INT8 runs the identical loop as an int8 job so
		 * the rate can be attributed to fp16 or to the loop.
		 */
		unsigned reps = argc > 4 ? (unsigned)atoi(argv[4]) : 128;
		unsigned wrote = 0, first_fail = 0;
		double t0;

		for (unsigned i = 0; i < m * k; i++)
			A[i] = 1.0f;
		for (unsigned i = 0; i < n * k; i++)
			B[i] = 1.0f;
		printf("%s, the same job %u times\n",
		       getenv("CHARSIU_TEST_INT8") ? "int8" : "fp16", reps);
		t0 = now_ms();
		for (unsigned r = 0; r < reps; r++) {
			unsigned untouched = 0;

			if (run(dev, m, k, n, A, B, CHARSIU_W16_DENSE, got))
				break;
			for (unsigned c = 0; c < n; c++)
				untouched += got[c] == 0xdeadbeefu;
			if (untouched == n) {
				if (!first_fail)
					first_fail = r + 1;
			} else {
				wrote++;
			}
			if (r < 40)
				printf("%c", untouched == n ? '.' : '#');
		}
		printf("\n  %u of %u wrote (# wrote, . nothing); first"
		       " silent submit was number %u\n", wrote, reps,
		       first_fail);
		/*
		 * ⚠ AND WHAT IT COSTS, because the whole reason for an fp16
		 * matmul here is attention, and attention is only worth moving
		 * if the hardware beats the 6 to 13 ms a row the CPU takes.
		 * Timed over the submits that actually wrote; a wedged shape
		 * would otherwise report the timeout as throughput.
		 */
		if (wrote) {
			double el = now_ms() - t0;

			printf("  %.3f ms a matmul over %u that wrote"
			       " (%.1f ms wall)\n", el / wrote, wrote, el);
			printf("    pack %.3f  coefs %.3f  emit %.3f"
			       "  submit %.3f  fence+read %.3f  ms each\n",
			       t_split.pack / reps, t_split.cf / reps,
			       t_split.emit / reps, t_split.run / reps,
			       t_split.rd / reps);
		}
		rc = 0;
		goto done;
	}
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
		/*
		 * ⚠⚠ THE HOLE DOES NOT HAVE TO BE ZERO, AND THAT IS THE
		 * EXPERIMENT. With a hole of 0.0, 116 of 128 slots made the job
		 * write NOTHING -- the sentinel survived in every channel -- and
		 * only 12 gave a clean answer. A weight fetch that skips zero
		 * blocks explains that; a fetch that does not, does not. So the
		 * hole value is an argument. Anything other than 1.0 still names
		 * k, because the channel's sum moves by (hole - 1) * 2^k, and a
		 * non zero hole never makes the buffer look sparse.
		 *
		 * 3.0 is the default: the sum moves UP by 2 * 2^k, which cannot
		 * be confused with a bit that failed to arrive.
		 */
		float hv = argc > 4 ? (float)atof(argv[4]) : 3.0f;
		uint16_t hb = charsiu_float_to_half(hv);

		if (!raw) goto done;
		printf("hole value %g\n", (double)hv);
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
			raw[sl * 2] = (uint8_t)(hb & 0xff);
			raw[sl * 2 + 1] = (uint8_t)(hb >> 8);
			if (run_raw(dev, m, k, n, A, raw, wsz, got))
				break;
			{
				unsigned untouched = 0;

				for (unsigned c = 0; c < n; c++)
					untouched += got[c] == 0xdeadbeefu;
				if (untouched == n) {
					printf("  %-10u -> THE JOB WROTE NOTHING\n",
					       sl);
					continue;
				}
				if (untouched) {
					printf("  %-10u -> %u of %u channels"
					       " not written\n", sl, untouched, n);
					said = 1;
				}
			}
			for (unsigned c = 0; c < n; c++) {
				float v = asf(got[c]);
				unsigned long b = (got[c] == 0xdeadbeefu) ? ~0ul
					: (v >= 0 && v < 1e9 &&
					   v == (float)(unsigned long)v)
					  ? (unsigned long)v : ~1ul;

				long d;

				if (b == full || b == ~0ul)
					continue;
				if (b == ~1ul) {
					printf("  %-10u -> %u : %g, not an"
					       " integer\n", sl, c, (double)v);
					said = 1;
					continue;
				}
				/*
				 * the sum moved by (hole - 1) * 2^k, so the
				 * delta over (hole - 1) is the bit itself
				 */
				d = (long)b - (long)full;
				if (hv != 1.0f && d % (long)(hv - 1.0f) == 0) {
					long bit = d / (long)(hv - 1.0f);

					if (bit > 0 && (bit & (bit - 1)) == 0) {
						printf("  %-10u -> %u : +%ld"
						       " so k=%d\n", sl, c, d,
						       __builtin_ctzl(bit));
						said = 1;
						continue;
					}
				}
				printf("  %-10u -> %u : 0x%lx delta %ld",
				       sl, c, b, d);
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
	pool_free();
	charsiu_close(dev);
	free(A); free(B); free(ref); free(got);
	return rc;
}
