// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * An fp16 matmul whose WEIGHT belongs to the caller.
 *
 * ⚠⚠ WHY THIS IS NOT IN npudev.c. Everything there is built around a weight
 * tensor that is STAGED once -- charsiu_npu_add, then matmul by id -- because
 * a model's weights do not change. Attention's second operand is the KV cache:
 * a different buffer for every layer and every head, and one that grows by a
 * row per token. Threading that through the staging machinery would put a
 * growing, per-head buffer into the path that decode and prefill already run
 * on, for no benefit to either. This is its own unit and the int4 path cannot
 * see it.
 *
 * What the board established on 2026-09-05, and what this encodes:
 *
 *   - the weight layout is GROUP: ngroup 16, kgroup 32, two byte elements.
 *     charsiu_fp16_woffset() is that layout, so the caller can write the cache
 *     STRAIGHT INTO IT as tokens are appended and never pay a pack.
 *   - fp16 needs the w4a16 output stage, CORE 0x3018's 0x200 form, and
 *     DPU 0x40b8 = (oc/4 + 3) - (M*oc)/4. All of that lives in job.c and is
 *     selected by wdtype == CHARSIU_FP16; nothing here repeats it.
 *   - the job cost is nearly flat in m: 0.366 ms at m=80 and 0.480 at m=178,
 *     governor pinned, five alternating rounds, spreads 0.339-0.377 and
 *     0.459-0.512. So the caller should batch as wide as it can.
 *
 * ⚠ THE PACK IS THE WHOLE GAME. Packing a K=1024 N=64 weight costs 1.62 ms,
 * four times the job. If a caller packs per dispatch there is nothing here
 * worth having; charsiu_fp16_woffset() exists so it does not have to.
 */
/* clock_gettime: the Makefile builds at -std=c11, which hides it */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "charsiu.h"
#include "charsiu_llm.h"
#include "fp16plan.h"

struct charsiu_fp16 {
	struct charsiu_device *dev;
	struct charsiu_bo wt, in, ob, coef, reg;
	size_t wsz, insz, obsz, coefsz, regsz;
	unsigned long calls, refused, submits;
	struct charsiu_fp16_times t;
};

static double now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}


/* the weight buffer a (k, n) fp16 matmul needs, in bytes */
size_t charsiu_fp16_wbytes(unsigned k, unsigned n)
{
	struct charsiu_matmul mm = { 1, k, n, CHARSIU_FP16, CHARSIU_FP16 };

	return charsiu_weight_bytes(&mm);
}

/* where element (n, k) of the weight goes, in bytes. SIZE_MAX if out of range */
size_t charsiu_fp16_woffset(unsigned k, unsigned n, unsigned ni, unsigned ki)
{
	struct charsiu_matmul mm = { 1, k, n, CHARSIU_FP16, CHARSIU_FP16 };

	return charsiu_w16_offset(&mm, ni, ki, CHARSIU_W16_GROUP);
}

static int want(struct charsiu_fp16 *f, size_t wsz, size_t insz, size_t obsz,
		size_t coefsz, size_t regsz)
{
	if (f->wsz >= wsz && f->insz >= insz && f->obsz >= obsz &&
	    f->coefsz >= coefsz && f->regsz >= regsz)
		return 0;
	/* grow, never shrink. A single call and a group share these buffers,
	 * and alternating between them must not free and reallocate five
	 * buffer objects every time. */
	if (wsz < f->wsz) wsz = f->wsz;
	if (insz < f->insz) insz = f->insz;
	if (obsz < f->obsz) obsz = f->obsz;
	if (coefsz < f->coefsz) coefsz = f->coefsz;
	if (regsz < f->regsz) regsz = f->regsz;
	charsiu_bo_free(f->dev, &f->reg);  charsiu_bo_free(f->dev, &f->coef);
	charsiu_bo_free(f->dev, &f->ob);   charsiu_bo_free(f->dev, &f->in);
	charsiu_bo_free(f->dev, &f->wt);
	memset(&f->wt, 0, sizeof(f->wt));   memset(&f->in, 0, sizeof(f->in));
	memset(&f->ob, 0, sizeof(f->ob));   memset(&f->coef, 0, sizeof(f->coef));
	memset(&f->reg, 0, sizeof(f->reg));
	f->wsz = wsz; f->insz = insz; f->obsz = obsz; f->coefsz = coefsz;
	f->regsz = regsz;
	if (charsiu_bo_alloc(f->dev, wsz, &f->wt) ||
	    charsiu_bo_alloc(f->dev, insz, &f->in) ||
	    charsiu_bo_alloc(f->dev, obsz, &f->ob) ||
	    charsiu_bo_alloc(f->dev, coefsz, &f->coef) ||
	    charsiu_bo_alloc(f->dev, regsz, &f->reg))
		return -1;
	return (f->wt.map && f->in.map && f->ob.map && f->coef.map &&
		f->reg.map) ? 0 : -1;
}

struct charsiu_fp16 *charsiu_fp16_open(void)
{
	struct charsiu_fp16 *f = calloc(1, sizeof(*f));

	if (!f)
		return NULL;
	f->dev = charsiu_open(NULL);
	if (!f->dev) {
		free(f);
		return NULL;
	}
	return f;
}

void charsiu_fp16_close(struct charsiu_fp16 *f)
{
	if (!f)
		return;
	charsiu_bo_free(f->dev, &f->reg);  charsiu_bo_free(f->dev, &f->coef);
	charsiu_bo_free(f->dev, &f->ob);   charsiu_bo_free(f->dev, &f->in);
	charsiu_bo_free(f->dev, &f->wt);
	charsiu_close(f->dev);
	free(f);
}

/*
 * X is m by k, row major, float. W holds charsiu_fp16_wbytes(k, n) bytes in
 * the layout charsiu_fp16_woffset describes. Y is m by n, row major, float.
 *
 * ⚠ IT REFUSES RATHER THAN COMPUTES WHAT IT HAS NOT BEEN SHOWN. K=16 N=8
 * wedged the NPU for two jobs and then timed out both cores, and the shapes
 * that survive a loop cleanly are the ones with a real n. Anything under the
 * two byte feature atom on either axis is not a shape this has ever run.
 */
int charsiu_fp16_matmul(struct charsiu_fp16 *f, const float *X, unsigned m,
			unsigned k, unsigned n, const void *W, float *Y)
{
	struct charsiu_job job = { 0 };
	size_t nreg, insz, wsz;

	if (!f || !X || !W || !Y || !m || k < 32 || n < 32) {
		if (f)
			f->refused++;
		return -1;
	}
	job.cbuf_window = (unsigned)charsiu_cbuf_window();
	job.mm.m = m; job.mm.k = k; job.mm.n = n;
	job.mm.wdtype = CHARSIU_FP16;
	job.mm.adtype = CHARSIU_FP16;
	job.input_scale = 1.0f; job.weight_scale = 1.0f; job.output_scale = 1.0f;
	job.acc_out = 1;                /* the accumulator, which for fp16 is fp32 */

	wsz = charsiu_weight_bytes(&job.mm);
	insz = (size_t)charsiu_entries_per_row(&job.mm) * 64 * m + 4096;
	if (want(f, wsz + 4096, insz, (size_t)m * n * 4 + 4096,
		 charsiu_coef_bytes(&job.mm) + 4096, FP16_REG_STRIDE))
		return -1;

	charsiu_bo_prep(f->dev, &f->wt, 1000000000);
	memcpy(f->wt.map, W, wsz);
	charsiu_bo_fini(f->dev, &f->wt);

	/*
	 * ⚠⚠ ROW MAJOR, NOT charsiu_pack_input_f16's INTERLEAVE, AND THE BOARD
	 * SAID SO SLOT BY SLOT.
	 *
	 * charsiu_pack_input_f16 writes [k/8][m][8]: row r's k values are
	 * spread through the buffer at a stride of m*8. npu_fp16_test
	 * --inslots put one 1.0 into each packed input slot in turn and read
	 * which output row answered and with which k. At m=2, K=256:
	 *
	 *     slots   0..255   -> row 0, k = slot
	 *     slots 256..511   -> row 1
	 *
	 * so this path wants row r's k CONTIGUOUS at r*k. Feeding it the
	 * interleaved layout is why every output row came back holding
	 * k in [r*K/m, (r+1)*K/m) -- it reads contiguous runs and gets a
	 * mixture -- and why m=1 was exact: at m=1 the interleave is the
	 * identity.
	 *
	 * Six register-level fixes were tried against that symptom and none
	 * moved it, because the fault was never in the stream.
	 */
	charsiu_bo_prep(f->dev, &f->in, 1000000000);
	{
		uint8_t *d = f->in.map;

		memset(d, 0, insz);
		for (size_t i = 0; i < (size_t)m * k; i++) {
			uint16_t h = charsiu_float_to_half(X[i]);

			if ((i + 1) * 2 > insz)
				break;
			d[i * 2] = (uint8_t)(h & 0xff);
			d[i * 2 + 1] = (uint8_t)(h >> 8);
		}
	}
	charsiu_bo_fini(f->dev, &f->in);

	{
		int32_t *zero = calloc(n, sizeof(int32_t));

		if (!zero)
			return -1;
		charsiu_bo_prep(f->dev, &f->coef, 1000000000);
		charsiu_build_coefs(&job, zero, zero, f->coef.map);
		charsiu_bo_fini(f->dev, &f->coef);
		free(zero);
	}

	job.input_addr = (uint32_t)f->in.dma_address;
	job.output_addr = (uint32_t)f->ob.dma_address;
	job.weight_addr = (uint32_t)f->wt.dma_address;
	job.coef_addr = (uint32_t)f->coef.dma_address;

	charsiu_bo_prep(f->dev, &f->reg, 1000000000);
	nreg = charsiu_emit_job(&job, f->reg.map, 4096 / 8);
	charsiu_bo_fini(f->dev, &f->reg);
	if (!nreg)
		return -1;

	/*
	 * ⚠ A SENTINEL, NOT ZEROS. A job that never wrote and a job that
	 * computed zero are the same four bytes otherwise, and this project
	 * has read the first as the second four times in a week.
	 */
	charsiu_bo_prep(f->dev, &f->ob, 1000000000);
	for (unsigned i = 0; i < m * n; i++)
		((uint32_t *)f->ob.map)[i] = 0xdeadbeefu;
	charsiu_bo_fini(f->dev, &f->ob);
	{
		uint32_t ins[2] = { f->in.handle, f->wt.handle };
		uint32_t outs[1] = { f->ob.handle };

		if (charsiu_submit(f->dev, &f->reg, (unsigned)nreg, ins, 2,
				   outs, 1))
			return -1;
		f->submits++;
	}
	charsiu_bo_prep(f->dev, &f->ob, 1000000000);   /* the fence wait */
	{
		const uint32_t *o = f->ob.map;
		unsigned untouched = 0;

		for (unsigned i = 0; i < m * n; i++)
			untouched += o[i] == 0xdeadbeefu;
		if (untouched == m * n) {
			charsiu_bo_fini(f->dev, &f->ob);
			f->refused++;
			return -1;              /* the job wrote nothing */
		}
		/* flat, and measured: --outmap reads m by n row major */
		memcpy(Y, o, (size_t)m * n * 4);
	}
	charsiu_bo_fini(f->dev, &f->ob);
	f->calls++;
	return 0;
}


/*
 * ⚠⚠ SEVERAL MATMULS, ONE SUBMIT AND ONE FENCE, BECAUSE THE FENCE IS THE JOB.
 *
 * npu_fp16_test --loop split a single call three ways on 2026-09-05 and put
 * 98% of it in the wait: fence+sync 0.345 to 0.454 ms against 6 to 107 us for
 * the copy and the cache maintenance together. charsiu_npu_matvec_group has
 * said the same since round 321 -- "the fence at 94% of the hardware path" --
 * and answers it the same way. Tasks inside one job are chained by the program
 * counter on one core, so N matmuls cost one submit and one wait rather than
 * N of each.
 *
 * A layer's attention is exactly this shape. Every head's scores matmul reads
 * a different K cache and writes its own output and none of them reads
 * another's result, so the whole layer is one submit; softmax runs on the CPU;
 * the values matmuls are a second. Per layer that is 2 fences where a loop
 * over charsiu_fp16_matmul pays 2H.
 *
 * ⚠ THE GROUPING CHANGES NO ARITHMETIC, so the results must be IDENTICAL to
 * the same ops run one at a time. npu_fp16_test --group checks that bit for
 * bit, at MIXED SHAPES, before it reports a single millisecond: uniform shapes
 * cannot catch an offset that is wrong by a whole sub buffer.
 *
 * ⚠ AND THE WEIGHT COPY IS STILL HERE. Every op's W is memcpy'd into the
 * device buffer, which for a K=1024 N=64 cache is 128 kB an op. That is what
 * charsiu_fp16_woffset exists to remove -- a caller that appends its KV cache
 * straight into the buffer pays none of it -- and the times below name it
 * separately so the next round can see what is left after it goes.
 */
int charsiu_fp16_matmul_group(struct charsiu_fp16 *f,
			      const struct charsiu_fp16_op *ops, unsigned nops)
{
	struct charsiu_job job[FP16_GROUP_MAX];
	struct charsiu_task task[FP16_GROUP_MAX];
	struct charsiu_fp16_plan pl;
	unsigned i, bad = 0;
	struct charsiu_joblist jl;
	uint32_t ins[3], outs[1];
	int32_t *zero;
	double t0;

	if (!f)
		return -1;
	/* the shapes, and where each op sits in the four shared buffers.
	 * tests/fp16_plan.c walks this on a desk; nothing below recomputes it */
	if (charsiu_fp16_make_plan(ops, nops, &pl)) {
		f->refused++;
		return -1;
	}
	for (i = 0; i < nops; i++)
		if (!ops[i].X || !ops[i].W || !ops[i].Y) {
			f->refused++;
			return -1;
		}
	if (want(f, pl.wtot + 4096, pl.itot + 4096, pl.otot + 4096,
		 pl.ctot + 4096, (size_t)nops * FP16_REG_STRIDE + 4096))
		return -1;

	/* ONE prep and ONE fini a buffer for the whole group. Cache
	 * maintenance per op would put back a per dispatch cost of exactly
	 * the kind this function exists to remove. */
	t0 = now_ms();
	charsiu_bo_prep(f->dev, &f->wt, 1000000000);
	for (i = 0; i < nops; i++)
		memcpy((uint8_t *)f->wt.map + pl.woff[i], ops[i].W, pl.wsz[i]);
	charsiu_bo_fini(f->dev, &f->wt);
	f->t.wcopy += now_ms() - t0;

	/* row major, [m][k], which is what --inslots measured slot by slot */
	t0 = now_ms();
	charsiu_bo_prep(f->dev, &f->in, 1000000000);
	for (i = 0; i < nops; i++) {
		uint8_t *d = (uint8_t *)f->in.map + pl.ioff[i];
		size_t nel = (size_t)ops[i].m * ops[i].k;

		memset(d, 0, charsiu_fp16_up4k(pl.isz[i]));
		for (size_t e = 0; e < nel; e++) {
			uint16_t h = charsiu_float_to_half(ops[i].X[e]);

			if ((e + 1) * 2 > pl.isz[i])
				break;
			d[e * 2] = (uint8_t)(h & 0xff);
			d[e * 2 + 1] = (uint8_t)(h >> 8);
		}
	}
	charsiu_bo_fini(f->dev, &f->in);
	f->t.pack += now_ms() - t0;

	t0 = now_ms();
	zero = calloc(pl.nmax, sizeof(*zero));
	if (!zero)
		return -1;
	charsiu_bo_prep(f->dev, &f->coef, 1000000000);
	for (i = 0; i < nops; i++) {
		memset(&job[i], 0, sizeof(job[i]));
		job[i].cbuf_window = (unsigned)charsiu_cbuf_window();
		job[i].mm.m = ops[i].m;
		job[i].mm.k = ops[i].k;
		job[i].mm.n = ops[i].n;
		job[i].mm.wdtype = CHARSIU_FP16;
		job[i].mm.adtype = CHARSIU_FP16;
		job[i].input_scale = 1.0f;
		job[i].weight_scale = 1.0f;
		job[i].output_scale = 1.0f;
		job[i].acc_out = 1;
		job[i].input_addr = (uint32_t)f->in.dma_address + pl.ioff[i];
		job[i].output_addr = (uint32_t)f->ob.dma_address + pl.ooff[i];
		job[i].weight_addr = (uint32_t)f->wt.dma_address + pl.woff[i];
		job[i].coef_addr = (uint32_t)f->coef.dma_address + pl.coff[i];
		charsiu_build_coefs(&job[i], zero, zero,
				    (uint8_t *)f->coef.map + pl.coff[i]);
	}
	charsiu_bo_fini(f->dev, &f->coef);
	free(zero);
	f->t.coefs += now_ms() - t0;

	t0 = now_ms();
	charsiu_bo_prep(f->dev, &f->reg, 1000000000);
	for (i = 0; i < nops; i++) {
		size_t nreg = charsiu_emit_job(&job[i],
			(uint64_t *)((uint8_t *)f->reg.map
				     + (size_t)i * FP16_REG_STRIDE),
			FP16_REG_STRIDE / 8);

		if (!nreg) {
			charsiu_bo_fini(f->dev, &f->reg);
			return -1;
		}
		task[i].regcmd = (uint32_t)f->reg.dma_address
			       + (uint32_t)(i * FP16_REG_STRIDE);
		task[i].regcmd_count = (uint32_t)nreg;
	}
	charsiu_bo_fini(f->dev, &f->reg);
	f->t.emit += now_ms() - t0;

	/* the sentinel, per op: a job that never wrote and a job that computed
	 * zero are the same four bytes otherwise */
	charsiu_bo_prep(f->dev, &f->ob, 1000000000);
	for (i = 0; i < nops; i++) {
		uint32_t *o = (uint32_t *)((uint8_t *)f->ob.map + pl.ooff[i]);

		for (unsigned e = 0; e < ops[i].m * ops[i].n; e++)
			o[e] = 0xdeadbeefu;
	}
	charsiu_bo_fini(f->dev, &f->ob);

	t0 = now_ms();
	ins[0] = f->in.handle;
	ins[1] = f->wt.handle;
	ins[2] = f->coef.handle;
	outs[0] = f->ob.handle;
	/*
	 * ⚠ THE PROBE HATCH, because "the group is wrong" has two causes and
	 * they are not the same repair.
	 *
	 * The default is ONE job of N tasks: the program counter walks them on
	 * a single core, which is what npudev has always done and what keeps
	 * the two cores from being in flight together -- they corrupt single
	 * words when they are, and the rail is why. CHARSIU_FP16_JOBS=split
	 * sends N jobs of one task in the same submit instead, so the
	 * scheduler may place them on both cores. If the split arm is correct
	 * and the chained arm is not, the fault is task chaining and not this
	 * function's addressing.
	 */
	if (getenv("CHARSIU_FP16_JOBS") &&
	    !strcmp(getenv("CHARSIU_FP16_JOBS"), "split")) {
		struct charsiu_joblist split[FP16_GROUP_MAX];

		for (i = 0; i < nops; i++) {
			split[i].tasks = &task[i];
			split[i].task_count = 1;
			split[i].in_handles = ins;
			split[i].in_count = 3;
			split[i].out_handles = outs;
			split[i].out_count = 1;
		}
		if (charsiu_submit_jobs(f->dev, split, nops))
			return -1;
	} else {
		jl.tasks = task;
		jl.task_count = nops;
		jl.in_handles = ins;
		jl.in_count = 3;
		jl.out_handles = outs;
		jl.out_count = 1;
		if (charsiu_submit_jobs(f->dev, &jl, 1))
			return -1;
	}
	f->submits++;
	f->t.submit += now_ms() - t0;

	t0 = now_ms();
	charsiu_bo_prep(f->dev, &f->ob, 1000000000);   /* the one fence wait */
	f->t.fence += now_ms() - t0;

	t0 = now_ms();
	for (i = 0; i < nops; i++) {
		const uint32_t *o = (const uint32_t *)
			((const uint8_t *)f->ob.map + pl.ooff[i]);
		unsigned untouched = 0, cells = ops[i].m * ops[i].n;

		for (unsigned e = 0; e < cells; e++)
			untouched += o[e] == 0xdeadbeefu;
		if (untouched == cells) {
			bad++;
			continue;              /* this op wrote nothing */
		}
		memcpy(ops[i].Y, o, (size_t)cells * 4);
	}
	charsiu_bo_fini(f->dev, &f->ob);
	f->t.read += now_ms() - t0;
	if (bad) {
		f->refused += bad;
		return -1;
	}
	f->calls += nops;
	return 0;
}

void charsiu_fp16_stats(const struct charsiu_fp16 *f, unsigned long *calls,
			unsigned long *refused)
{
	*calls = f ? f->calls : 0;
	*refused = f ? f->refused : 0;
}

/*
 * How many times the hardware was waited on. calls / submits is the whole
 * point of the group: 1 for a loop over charsiu_fp16_matmul, the group size
 * for charsiu_fp16_matmul_group, and it is the number the fence is paid per.
 */
unsigned long charsiu_fp16_submits(const struct charsiu_fp16 *f)
{
	return f ? f->submits : 0;
}

/* ⚠ the group fills these and the single call does not: the split for one
 * call is npu_fp16_test --loop, which measures the same stages around
 * job.c directly and does not need the unit to carry a clock. */
void charsiu_fp16_get_times(const struct charsiu_fp16 *f,
			    struct charsiu_fp16_times *t)
{
	if (!t)
		return;
	if (!f)
		memset(t, 0, sizeof(*t));
	else
		*t = f->t;
}
