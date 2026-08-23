// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * What value does this hardware give each 4 bit code, measured under the
 * VENDOR'S OWN register stream and with NO fitted reference.
 *
 * WHY THIS TOOL EXISTS. Every int4 round in this tree has scored the board
 * against an arithmetic model, and the model that fits is a bit pattern
 * product whose representable weights are 0 and the band 1.00 to 1.18. A
 * sixteen level grid crammed into eighteen percent is not a grid a matmul can
 * use, so either the hardware cannot do this or charsiu is asking it wrongly.
 * The capture from the vendor's LLM runtime settles which: 123 registers for
 * one int4 projection at K = 2048, N = 1024. Fifteen of them differ from what
 * this tree emits, and three whole groups exist on one side only -- five
 * registers at 0x2810 that nothing here has ever written, and, on our side,
 * the DPU_RDMA block and the 0x1d enable trailer that the vendor does not
 * write at all.
 *
 * THE MEASUREMENT. Submit once and keep the output. Then change ONE nibble of
 * the weight buffer, submit again, and subtract. The difference is that
 * nibble's contribution and nothing else, so there is no reference to fit and
 * no model to pick. Sweeping the nibble through 0..15 gives the value the
 * hardware assigns each code, up to one common activation factor.
 *
 * If the answer is a straight line in the code, the block is a real int4
 * multiplier and the whole fp16bits reading was a configuration fault. If it
 * is the 1.00 to 1.18 band again under the vendor's own stream, then it is the
 * silicon and the int4 path is closed for good.
 *
 * TWO CONTROLS, both of which can fail:
 *   - submitting twice with nothing changed must give byte identical output,
 *     or the board is not repeatable and nothing below it can be read;
 *   - putting the nibble back must return the output to the baseline, which
 *     catches a stale buffer or a job that quietly stopped running.
 * And a poison gate: an all zero baseline is VOID, not a score of zero. Round
 * 339 printed "64 of 64" on a job that timed out and returned zeros.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "charsiu.h"

#define WT_BYTES   (8u << 20)   /* N=2048 K=2048 int8 is 4 MiB */
#define IN_BYTES   (1u << 20)
/*
 * ⚠ THE OUTPUT BO IS SIZED FROM THE ENVIRONMENT because round 348's timing
 * measured this tool and not the hardware: run() memsets the whole output
 * buffer and cache-maintains it twice a submit, and at 4 MiB that is most of
 * the 1418 us it reported. int8 came out at 1.31 GB/s against a standing 9.38,
 * which is the giveaway.
 */
static size_t out_bytes(void)
{
	const char *e = getenv("CHARSIU_OUT_KB");

	return (size_t)(e ? strtoul(e, NULL, 0) : 4096) << 10;
}

static size_t load(const char *path, void *dst, size_t max)
{
	FILE *f = fopen(path, "rb");
	size_t n;

	if (!f) {
		printf("cannot open %s\n", path);
		exit(2);
	}
	n = fread(dst, 1, max, f);
	fclose(f);
	if (!n) {
		printf("%s is empty\n", path);
		exit(2);
	}
	return n;
}

static int reread;

struct ctx {
	struct charsiu_device *dev;
	struct charsiu_bo regcmd, in, wt, out, coef;
	uint32_t in_h[3], out_h[1];
	size_t nreg;
};

/* one submit; returns the number of nonzero 32 bit words in the output */
static unsigned run(struct ctx *c, int32_t *dst, unsigned words)
{
	unsigned i, live = 0;
	int ret;

	memset(c->out.map, 0, c->out.size);
	charsiu_bo_fini(c->dev, &c->out);
	ret = charsiu_submit(c->dev, &c->regcmd, (unsigned)c->nreg,
			     c->in_h, 3, c->out_h, 1);
	if (ret) {
		printf("submit FAILED %d\n", ret);
		exit(1);
	}
	ret = charsiu_bo_prep(c->dev, &c->out, 5000000000LL);
	if (ret) {
		printf("wait FAILED %d -- the job did not complete\n", ret);
		exit(1);
	}
	/*
	 * ⚠ DID THE DATA ARRIVE LATE, OR WAS IT NEVER WRITTEN.
	 *
	 * Round 359: a solo process is clean three times out of three, but with
	 * two running concurrently some output words come back ZERO that should
	 * be live -- 1020 and 1022 of 1024 -- and the repeat control differs by
	 * tens to hundreds of words. Missing writes, not swapped data, and only
	 * ever in the bigger of the two jobs.
	 *
	 * CHARSIU_REREAD takes the buffer through another cache round trip and
	 * reads it again. If the missing words are there the second time, the
	 * fence signalled before the NPU's writes had landed and the fix is
	 * ordering in the driver. If they are still missing, they were never
	 * written and the fix is somewhere else entirely.
	 *
	 * A plain second memcpy would prove nothing: the lines are in cache by
	 * then and would read back exactly what the first one saw.
	 */
	memcpy(dst, c->out.map, (size_t)words * 4);
	if (reread) {
		static int32_t *again;
		unsigned d = 0, w;

		if (!again)
			again = malloc((size_t)words * 4);
		charsiu_bo_fini(c->dev, &c->out);
		charsiu_bo_prep(c->dev, &c->out, 5000000000LL);
		memcpy(again, c->out.map, (size_t)words * 4);
		for (w = 0; w < words; w++)
			if (again[w] != dst[w])
				d++;
		if (d)
			printf("  REREAD: %u words CHANGED after a second cache "
			       "round trip -- the data was still landing\n", d);
		memcpy(dst, again, (size_t)words * 4);
	}
	for (i = 0; i < words; i++)
		if (dst[i])
			live++;
	return live;
}

/*
 * The submit alone: no memset of the output, no copy back. Everything run()
 * does around the submit is this tool's own bookkeeping, and round 348 timed
 * all of it. The overhead is REMOVED rather than subtracted -- round 327 was
 * lost to a subtraction.
 */
static void run_bare(struct ctx *c)
{
	if (charsiu_bo_fini(c->dev, &c->out) ||
	    charsiu_submit(c->dev, &c->regcmd, (unsigned)c->nreg,
			   c->in_h, 3, c->out_h, 1) ||
	    charsiu_bo_prep(c->dev, &c->out, 5000000000LL)) {
		printf("timed submit FAILED\n");
		exit(1);
	}
}

/* set one nibble of the weight buffer and push it back to the device */
static void poke(struct ctx *c, size_t byte, int high, unsigned v)
{
	uint8_t *w;

	charsiu_bo_prep(c->dev, &c->wt, 1000000000);
	w = c->wt.map;
	if (high)
		w[byte] = (uint8_t)((w[byte] & 0x0f) | (v << 4));
	else
		w[byte] = (uint8_t)((w[byte] & 0xf0) | v);
	charsiu_bo_fini(c->dev, &c->wt);
}

static unsigned diff(const int32_t *a, const int32_t *b, unsigned words,
		     unsigned *first, int64_t *delta)
{
	unsigned i, n = 0;

	*first = 0;
	*delta = 0;
	for (i = 0; i < words; i++) {
		if (a[i] == b[i])
			continue;
		if (!n) {
			*first = i;
			*delta = (int64_t)b[i] - (int64_t)a[i];
		}
		n++;
	}
	return n;
}

int main(int argc, char **argv)
{
	const char *stream = argc > 1 ? argv[1] : NULL;
	const char *wtf    = argc > 2 ? argv[2] : NULL;
	const char *inf    = argc > 3 ? argv[3] : NULL;
	size_t byte = argc > 4 ? strtoul(argv[4], NULL, 0) : 0;
	int high    = argc > 5 ? atoi(argv[5]) : 0;
	unsigned words = argc > 6 ? (unsigned)atoi(argv[6]) : 4096;
	struct charsiu_job job = { 0 };
	struct ctx c = { 0 };
	int32_t *base, *cur, *v0;
	unsigned live, live0, changed, first, keep;
	int64_t delta;
	uint8_t orig;
	unsigned v;
	uint8_t *gw = NULL;
	float *ga = NULL;

	if (!stream || !wtf || !inf) {
		printf("usage: charsiu_vendor <regcmd|-> <weights> <input> "
		       "[byte] [high] [words]\n");
		return 1;
	}
	/*
	 * THE CONTROL ARM. A regcmd of "-" runs the identical measurement under
	 * the stream THIS TREE emits for the same shape, so the two nibble
	 * curves are read off the same buffers, the same board and the same
	 * code, and the only thing that differs is the 123 registers. Without
	 * it a straight line under the vendor's stream would still leave open
	 * that something else in this tool changed the answer.
	 */
	if (!strcmp(stream, "-")) {
		unsetenv("CHARSIU_VENDOR_STREAM");
		printf("STREAM: charsiu's own, M=32 K=2048 N=1024 int4/fp16 "
		       "(this is the CONTROL arm)\n");
	} else if (*stream == '+') {
		/*
		 * "+<file>" is the MERGE arm: charsiu's stream carrying the
		 * vendor's values. The verbatim replay timed out on all four
		 * arms in round 343 because nothing enables the units, so this
		 * comes at the same 20 registers from the side that runs.
		 */
		unsetenv("CHARSIU_VENDOR_STREAM");
		setenv("CHARSIU_VENDOR_MERGE", stream + 1, 1);
		printf("STREAM: charsiu's own, MERGED with the vendor's values "
		       "from %s\n", stream + 1);
	} else {
		setenv("CHARSIU_VENDOR_STREAM", stream, 1);
	}
	/*
	 * CHARSIU_M. The vendor's capture is M = 32 and charsiu's decode loop is
	 * M = 1, and every surface stride above depends on M: at M = 1 the input
	 * slot collapses to k and the output word to n. So the mode has to be
	 * asked the question at M = 1 rather than assumed to survive it.
	 */
	job.mm.m = getenv("CHARSIU_M") ? (unsigned)atoi(getenv("CHARSIU_M")) : 32;
	job.mm.k = getenv("CHARSIU_K") ? (unsigned)atoi(getenv("CHARSIU_K")) : 2048;
	job.mm.n = getenv("CHARSIU_N") ? (unsigned)atoi(getenv("CHARSIU_N")) : 1024;
	/* ⚠ the two lines that used to sit here reassigned k and n to 2048 and
	 * 1024, so round 350's whole N sweep ran four times at the same shape
	 * and printed four nearly equal times as if they were a slope. */
	printf("M = %u K = %u N = %u\n", job.mm.m, job.mm.k, job.mm.n);
	/*
	 * CHARSIU_W8 times an int8 job at the SAME shape through the SAME code,
	 * so the int4 speed claim is measured against something rather than
	 * against a number from a different tool and a different boot. Its
	 * output is not checked -- the layout differs -- only its time.
	 */
	job.mm.wdtype = getenv("CHARSIU_W8") ? CHARSIU_INT8 : CHARSIU_INT4;
	job.mm.adtype = getenv("CHARSIU_W8") ? CHARSIU_INT8 : CHARSIU_FP16;
	job.input_scale = 0.02f;
	job.weight_scale = 0.01f;
	job.output_scale = 0.25f;
	job.input_zero_point = 0;
	job.weight_zero_point = 0;
	job.output_zero_point = 0;
	job.acc_out = 1;

	reread = getenv("CHARSIU_REREAD") != NULL;
	c.dev = charsiu_open(NULL);
	if (!c.dev) { printf("open FAILED\n"); return 1; }
	if (charsiu_bo_alloc(c.dev, 8192, &c.regcmd) ||
	    charsiu_bo_alloc(c.dev, IN_BYTES, &c.in) ||
	    charsiu_bo_alloc(c.dev, WT_BYTES, &c.wt) ||
	    charsiu_bo_alloc(c.dev, out_bytes(), &c.out) ||
	    charsiu_bo_alloc(c.dev, 1 << 20, &c.coef)) {
		printf("bo alloc FAILED\n");
		return 1;
	}
	printf("iova: regcmd 0x%llx in 0x%llx wt 0x%llx out 0x%llx\n",
	       (unsigned long long)c.regcmd.dma_address,
	       (unsigned long long)c.in.dma_address,
	       (unsigned long long)c.wt.dma_address,
	       (unsigned long long)c.out.dma_address);

	job.input_addr  = (uint32_t)c.in.dma_address;
	job.weight_addr = (uint32_t)c.wt.dma_address;
	job.output_addr = (uint32_t)c.out.dma_address;
	job.coef_addr   = (uint32_t)c.coef.dma_address;
	charsiu_bo_prep(c.dev, &c.coef, 1000000000);
	memset(c.coef.map, 0, c.coef.size);
	charsiu_bo_fini(c.dev, &c.coef);

	/*
	 * CHARSIU_GEN builds the operands with the PRODUCTION packers --
	 * charsiu_pack_weights and charsiu_pack_input_f16, the same two the LLM
	 * path calls -- instead of loading the vendor's captured buffers. The
	 * captured buffers proved the hardware; this proves the code that will
	 * feed it, which is a different claim and the one that matters next.
	 *
	 * The reference is then computed from the SOURCE arrays, not from the
	 * packed ones, so a packer that is wrong in the same way as the
	 * reference cannot pass.
	 */
	if (getenv("CHARSIU_GEN")) {
		unsigned nn, kk;

		gw = malloc((size_t)job.mm.n * job.mm.k);
		ga = malloc((size_t)job.mm.m * job.mm.k * sizeof(*ga));
		for (nn = 0; nn < job.mm.n; nn++)
			for (kk = 0; kk < job.mm.k; kk++)
				gw[(size_t)nn * job.mm.k + kk] =
					(uint8_t)((nn * 2654435761u + kk * 40503u
						   + (nn ^ kk)) & 0xf);
		/*
		 * ⚠ EXACTLY REPRESENTABLE IN fp16, in sixteenths. Round 350
		 * generated multiples of 0.05, which fp16 cannot hold, so the
		 * hardware multiplied by the rounded value while the reference
		 * multiplied by the float -- 256 of 1024 "failed" at 2.4e-04
		 * and the board was right every time. The reference now also
		 * rounds through fp16, and the values are chosen so that it
		 * does not have to.
		 */
		for (nn = 0; nn < job.mm.m; nn++)
			for (kk = 0; kk < job.mm.k; kk++)
				ga[(size_t)nn * job.mm.k + kk] =
					(float)((int)((nn * 97u + kk * 31u) % 41)
						- 20) / 16.0f;
		charsiu_bo_prep(c.dev, &c.wt, 1000000000);
		memset(c.wt.map, 0, c.wt.size);
		charsiu_pack_weights(&job.mm, gw, c.wt.map);
		orig = ((uint8_t *)c.wt.map)[byte];
		charsiu_bo_fini(c.dev, &c.wt);

		charsiu_bo_prep(c.dev, &c.in, 1000000000);
		memset(c.in.map, 0, c.in.size);
		charsiu_pack_input_f16(&job.mm, ga, c.in.map, c.in.size);
		charsiu_bo_fini(c.dev, &c.in);
		printf("operands GENERATED through charsiu_pack_weights and "
		       "charsiu_pack_input_f16\n");
	} else {
	charsiu_bo_prep(c.dev, &c.wt, 1000000000);
	memset(c.wt.map, 0, c.wt.size);
	printf("weights: %zu bytes from %s\n", load(wtf, c.wt.map, WT_BYTES), wtf);
	orig = ((uint8_t *)c.wt.map)[byte];
	charsiu_bo_fini(c.dev, &c.wt);

	charsiu_bo_prep(c.dev, &c.in, 1000000000);
	memset(c.in.map, 0, c.in.size);
	printf("input:   %zu bytes from %s\n", load(inf, c.in.map, IN_BYTES), inf);
	charsiu_bo_fini(c.dev, &c.in);
	}

	charsiu_bo_prep(c.dev, &c.regcmd, 1000000000);
	c.nreg = charsiu_emit_job(&job, c.regcmd.map, 8192 / 8);
	charsiu_bo_fini(c.dev, &c.regcmd);
	if (!c.nreg) { printf("emit FAILED\n"); return 1; }

	c.in_h[0] = c.in.handle;
	c.in_h[1] = c.wt.handle;
	c.in_h[2] = c.coef.handle;
	c.out_h[0] = c.out.handle;

	base = calloc(words, 4);
	cur  = calloc(words, 4);
	v0   = calloc(words, 4);

	live = run(&c, base, words);
	printf("\nbaseline: %u live words of %u\n", live, words);
	if (!live) {
		printf("VOID -- the output is all zero, so nothing below this "
		       "line means anything\n");
		return 1;
	}
	printf("  first eight: %d %d %d %d %d %d %d %d\n",
	       base[0], base[1], base[2], base[3],
	       base[4], base[5], base[6], base[7]);

	/*
	 * THE TIME, which is the question int4 has never been able to answer
	 * honestly: every previous measurement was taken on a configuration
	 * that did not compute a weighted sum. Weight bytes are the thing int4
	 * halves, so the rate is reported over them.
	 */
	if (getenv("CHARSIU_TIME")) {
		unsigned reps = (unsigned)strtoul(getenv("CHARSIU_TIME"),
						  NULL, 0);
		size_t wb = (size_t)job.mm.k * job.mm.n
			  / (job.mm.wdtype == CHARSIU_INT4 ? 2 : 1);
		struct timespec t0, t1;
		double us;
		unsigned r;

		run_bare(&c);                           /* warm */
		clock_gettime(CLOCK_MONOTONIC, &t0);
		for (r = 0; r < reps; r++)
			run_bare(&c);
		clock_gettime(CLOCK_MONOTONIC, &t1);
		us = ((double)(t1.tv_sec - t0.tv_sec) * 1e6
		      + (double)(t1.tv_nsec - t0.tv_nsec) / 1e3) / reps;
		printf("TIME: %s K=%u N=%u out_bo=%zuKB  %.1f us/submit over "
		       "%u reps, %zu weight bytes = %.2f GB/s\n",
		       job.mm.wdtype == CHARSIU_INT4 ? "int4" : "int8",
		       job.mm.k, job.mm.n, out_bytes() >> 10, us, reps, wb,
		       (double)wb / us / 1000.0);
	}

	/* CONTROL 1: the same job twice */
	run(&c, cur, words);
	changed = diff(base, cur, words, &first, &delta);
	printf("CONTROL repeat, nothing changed: %u words differ%s\n", changed,
	       changed ? "  <-- NOT REPEATABLE, stop reading here" : "  (ok)");
	if (changed)
		return 1;

	/*
	 * The sweep. v = 0 is the reference the rest are measured against, so
	 * the printed delta is value(v) - value(0) times whatever activation
	 * this nibble meets, common to every row.
	 */
	printf("\nnibble at byte %zu, %s half (was 0x%x)\n",
	       byte, high ? "high" : "low", high ? orig >> 4 : orig & 0xf);
	printf("  %-4s %-8s %-10s %-14s %-8s\n",
	       "v", "live", "words", "delta[first]", "first");
	{
		unsigned ref_live = 0;

		for (v = 0; v < 16; v++) {
			unsigned tries;

			poke(&c, byte, high, v);
			/*
			 * ⚠ A ROW WHOSE LIVE COUNT IS NOT THE REFERENCE'S DID
			 * NOT COMPUTE. Round 342 printed three impossible
			 * values -- v = 3, 9 and 10, two of them sharing a
			 * delta to the digit -- because the job timed out and
			 * the tool still subtracted whatever was in the buffer.
			 * Retry once, then say so rather than printing a
			 * number that is not a measurement.
			 */
			for (tries = 0; tries < 2; tries++) {
				live0 = run(&c, cur, words);
				if (!v || live0 == ref_live)
					break;
			}
			if (!v) {
				ref_live = live0;
				memcpy(v0, cur, (size_t)words * 4);
				printf("  %-4u %-8u %-10s %-14s %-8s  (reference)\n",
				       v, live0, "-", "-", "-");
				continue;
			}
			changed = diff(v0, cur, words, &first, &delta);
			if (live0 != ref_live) {
				printf("  %-4u %-8u %-10u %-14s %-8s  DEAD ROW, "
				       "not a measurement\n",
				       v, live0, changed, "-", "-");
				continue;
			}
			printf("  %-4u %-8u %-10u %-14lld %-8u\n", v, live0,
			       changed, (long long)delta, first);
		}
	}

	/*
	 * THE SLOPE MAP. Round 344 showed the contribution of one nibble is
	 * EXACTLY s times a constant, and that constant is the activation the
	 * nibble meets -- 8744 at byte 0's low half, 10312 at its high half,
	 * 3532 at byte 1024. So sweeping the BYTE instead of the value maps the
	 * layout directly: which activation each nibble of the weight buffer is
	 * multiplied by, and which output word it lands in.
	 *
	 * Two submits a byte, v = 0 then v = 1, and the difference is the slope.
	 */
	if (getenv("CHARSIU_SLOPE_MAP")) {
		unsigned nb = (unsigned)strtoul(getenv("CHARSIU_SLOPE_MAP"),
						NULL, 0);
		size_t b;

		printf("\nslope map, %u bytes, both halves\n", nb);
		printf("  %-8s %-5s %-12s %-8s %-6s\n",
		       "byte", "half", "slope", "first", "words");
		for (b = 0; b < nb; b++) {
			int h;

			for (h = 0; h < 2; h++) {
				unsigned l0, l1;
				uint8_t keepb;

				charsiu_bo_prep(c.dev, &c.wt, 1000000000);
				keepb = ((uint8_t *)c.wt.map)[b];
				charsiu_bo_fini(c.dev, &c.wt);

				poke(&c, b, h, 0);
				l0 = run(&c, v0, words);
				poke(&c, b, h, 1);
				l1 = run(&c, cur, words);
				changed = diff(v0, cur, words, &first, &delta);

				/* put the byte back before moving on */
				charsiu_bo_prep(c.dev, &c.wt, 1000000000);
				((uint8_t *)c.wt.map)[b] = keepb;
				charsiu_bo_fini(c.dev, &c.wt);

				if (l0 != l1 || !l0) {
					printf("  %-8zu %-5s %-12s %-8s %-6s  "
					       "DEAD\n", b, h ? "high" : "low",
					       "-", "-", "-");
					continue;
				}
				printf("  %-8zu %-5s %-12lld %-8u %-6u\n",
				       b, h ? "high" : "low",
				       (long long)delta, first, changed);
			}
		}
	}

	/*
	 * THE CORRECTNESS CHECK, against a CPU reference rather than against a
	 * fitted model. Round 345's slope map decoded the whole layout, and the
	 * eight words the log printed check out on the host to one fp32 ulp.
	 * This does the same for EVERY live word, on the board, in one arm.
	 *
	 *   out[n][m] = sum_k  s(n,k) * a(k,m)      in float
	 *   weight nibble(n,k) = 32*16*(K/32)*(n/16) + 512*(k/32)
	 *                        + 32*(n%16) + (k%32)
	 *   activation (k,m)   = fp16 slot  8*M*(k/8) + 8*m + (k%8)
	 *   output word (n,m)  = 4*(M*(n/4) + m) + (n%4)
	 *
	 * ⚠ The comparison is RELATIVE and the tolerance is one part in 1e-5,
	 * because the board returns float32 and this sums in double. An exact
	 * equality test here would fail on rounding and read as a wrong layout.
	 */
	if (getenv("CHARSIU_VERIFY")) {
		unsigned kk, nn, mm, w, ok = 0, seen = 0, dead = 0;
		unsigned Kd = 2048, Nd = 1024, Md = 32;
		const uint8_t *wp;
		const uint16_t *ap;
		double worst = 0.0;
		unsigned loose = 0;

		charsiu_vendor_stream_shape(&Kd, &Nd, NULL);
		Md = job.mm.m;
		Kd = job.mm.k;
		Nd = job.mm.n;
		printf("\nVERIFY: K=%u N=%u M=%u against a CPU reference\n",
		       Kd, Nd, Md);
		charsiu_bo_prep(c.dev, &c.wt, 1000000000);
		charsiu_bo_prep(c.dev, &c.in, 1000000000);
		wp = c.wt.map;
		ap = c.in.map;
		for (w = 0; w < words; w++) {
			double acc = 0.0;
			float got;

			mm = (w / 4) % Md;
			nn = 4 * (w / (4 * Md)) + w % 4;
			if (nn >= Nd)
				continue;
			for (kk = 0; kk < Kd; kk++) {
				int v;
				double av;

				if (gw) {
					v = gw[(size_t)nn * Kd + kk] & 0xf;
					/* through fp16, because that is what the
					 * packer gave the hardware */
					av = (double)charsiu_half_to_float(
						charsiu_float_to_half(
						  ga[(size_t)mm * Kd + kk]));
				} else {
					size_t ni = (size_t)32 * 16 * (Kd / 32)
						  * (nn / 16)
						  + (size_t)512 * (kk / 32)
						  + 32 * (nn % 16) + (kk % 32);
					uint8_t byt = wp[ni >> 1];
					size_t ai = (size_t)8 * Md * (kk / 8)
						  + 8 * mm + (kk % 8);

					v = (ni & 1) ? (byt >> 4) : (byt & 0xf);
					av = (double)charsiu_half_to_float(ap[ai]);
				}
				acc += (double)(v < 8 ? v : v - 16) * av;
			}
			memcpy(&got, &base[w], 4);
			if (!base[w] && acc == 0.0) { dead++; continue; }
			seen++;
			{
				double d = fabs((double)got - acc)
					 / (fabs(acc) > 1e-9 ? fabs(acc) : 1.0);

				if (d > worst) worst = d;
				if (d < 1e-5) ok++;
				if (d < 1e-4) loose++;
				else if (seen - loose <= 4)
					printf("  w=%-6u n=%-5u m=%-3u board %+.6f "
					       "ref %+.6f  rel %.2e\n",
					       w, nn, mm, got, acc, d);
			}
		}
		/*
		 * ⚠ TWO TOLERANCES, because 1e-5 cried wolf in round 346: five
		 * words failed it whose printed digits agreed with the
		 * reference. The board returns float32 and the reference sums
		 * 2048 terms in double, so a cancelling sum reaches 1e-5
		 * relative honestly. 1e-4 is the one to read; 1e-5 is there to
		 * show the margin.
		 */
		printf("VERIFY: %u of %u within 1e-4, %u within 1e-5 "
		       "(%u skipped, both zero), worst %.2e\n",
		       loose, seen, ok, dead, worst);
		charsiu_bo_fini(c.dev, &c.in);
		charsiu_bo_fini(c.dev, &c.wt);
	}

	/* CONTROL 2: put it back */
	poke(&c, byte, high, high ? orig >> 4 : orig & 0xf);
	keep = run(&c, cur, words);
	if (!keep)
		keep = run(&c, cur, words);   /* once, in case the block reset */
	changed = diff(base, cur, words, &first, &delta);
	printf("\nCONTROL restore, %u live: %u words differ%s\n", keep, changed,
	       changed ? "  <-- the buffer or the job drifted" : "  (ok)");

	charsiu_close(c.dev);
	return 0;
}
