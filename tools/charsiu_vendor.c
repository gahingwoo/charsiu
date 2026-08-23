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

#include "charsiu.h"

#define WT_BYTES   (2u << 20)
#define IN_BYTES   (1u << 20)
#define OUT_BYTES  (4u << 20)

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
	memcpy(dst, c->out.map, (size_t)words * 4);
	for (i = 0; i < words; i++)
		if (dst[i])
			live++;
	return live;
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
	job.mm.m = 32;
	job.mm.k = 2048;
	job.mm.n = 1024;
	job.mm.wdtype = CHARSIU_INT4;
	job.mm.adtype = CHARSIU_FP16;
	job.input_scale = 0.02f;
	job.weight_scale = 0.01f;
	job.output_scale = 0.25f;
	job.input_zero_point = 0;
	job.weight_zero_point = 0;
	job.output_zero_point = 0;
	job.acc_out = 1;

	c.dev = charsiu_open(NULL);
	if (!c.dev) { printf("open FAILED\n"); return 1; }
	if (charsiu_bo_alloc(c.dev, 8192, &c.regcmd) ||
	    charsiu_bo_alloc(c.dev, IN_BYTES, &c.in) ||
	    charsiu_bo_alloc(c.dev, WT_BYTES, &c.wt) ||
	    charsiu_bo_alloc(c.dev, OUT_BYTES, &c.out) ||
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

	charsiu_bo_prep(c.dev, &c.wt, 1000000000);
	memset(c.wt.map, 0, c.wt.size);
	printf("weights: %zu bytes from %s\n", load(wtf, c.wt.map, WT_BYTES), wtf);
	orig = ((uint8_t *)c.wt.map)[byte];
	charsiu_bo_fini(c.dev, &c.wt);

	charsiu_bo_prep(c.dev, &c.in, 1000000000);
	memset(c.in.map, 0, c.in.size);
	printf("input:   %zu bytes from %s\n", load(inf, c.in.map, IN_BYTES), inf);
	charsiu_bo_fini(c.dev, &c.in);

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
