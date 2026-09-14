// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * An fp16 matmul whose WEIGHT belongs to the caller.
 *
 * WHY THIS IS NOT IN npudev.c. Everything there is built around a weight
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
 * THE PACK IS THE WHOLE GAME. Packing a K=1024 N=64 weight costs 1.62 ms,
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
#include "sentinel.h"

struct charsiu_fp16 {
	struct charsiu_device *dev;
	int borrowed;               /* the device belongs to the caller */
	struct charsiu_bo wt, in, ob, coef, reg;
	size_t wsz, insz, obsz, coefsz, regsz;
	unsigned long calls, refused, submits;
	/* elements converted by the pack, so `pack` can be read per element
	 * rather than per call -- the two differ by a factor of the CONTEXT
	 * LENGTH, because the values matmul contracts over the whole of it */
	unsigned long long packel, trisk, partial, preskip, prewrote;
	/* THE ARITHMETIC, so "the hardware is the wall" is a reading
	 * and not a division somebody did in a report. sum of m*k*n. */
	unsigned long long macs;
	/* how often the register streams had to be rebuilt, against how often
	 * the same bytes were already there */
	unsigned long long emitbuilt, emitskip;
	/* which handle this is, when a caller runs more than one -- the
	 * attention mirror runs two, one a matmul, so that two stage tables
	 * in a log can be told apart */
	const char *name;
	struct charsiu_fp16_times t;
	/* what is currently sitting in the coefficient buffer, so a group
	 * that asks for the same shapes twice does not build it twice */
	unsigned gen, coef_gen, ncoef;
	unsigned coefn[FP16_GROUP_MAX];
	size_t coefoff[FP16_GROUP_MAX], coefsz_[FP16_GROUP_MAX];
	/*
	 * THE REGISTER STREAMS, WHICH ARE THE SAME BYTES EVERY LAYER.
	 *
	 * charsiu_emit_job depends on the shape and on four addresses, and a
	 * layer of attention asks for the same shapes at the same offsets in
	 * buffers that have not moved -- sixteen layers and eleven chunks of
	 * one prompt, 32 ops a group, all identical. At 852 tokens that was
	 * 56 ms of rebuilding streams byte for byte, plus a prep and a fini of
	 * the register buffer around it.
	 *
	 * Same shape as the coefficient cache above and keyed the same way:
	 * f->gen covers the buffers moving, and the per op signature covers
	 * everything else emit_job reads.
	 *
	 * 171 of 176 groups on the values handle and 165 of 176 on the
	 * scores one, emit 28 -> 10 ms and 29 -> 11. The misses are the chunk
	 * boundaries, where the prompt's extent changes the shape.
	 *
	 * CHARSIU_FP16_REGCACHE=0 rebuilds every stream, which is the control.
	 */
	struct {
		unsigned m, k, n;
		uint32_t in, out, wt, coef;
		size_t nreg;
	} regsig[FP16_GROUP_MAX];
	unsigned reg_gen, nregsig;
	/* the last group's layout, and whether its output buffer is still
	 * held open for a caller reading the answers where they lie */
	struct charsiu_fp16_plan last;
	int held;
	/*
	 * THE SENTINELS FOR THE NEXT CALL, IF A CALLER WROTE THEM EARLY.
	 *
	 * `last` records every op's offset but osz is m * n * 4, which does
	 * not say which m and which n -- and the row sentinel sits at
	 * o[r * n] for r < m, so both are needed to know what was poisoned
	 * and whether the next call's shapes still match it.
	 */
	unsigned pm[FP16_GROUP_MAX], pn[FP16_GROUP_MAX];
	int prepoisoned;
	/*
	 * ONE GROUP IN FLIGHT, which is what lets a caller do CPU work
	 * between the submit and the fence.
	 *
	 * The fence is a sleeping ioctl, so every millisecond of it is four
	 * idle cores; the softmax between the two attention groups is the
	 * mirror image, NPU idle. Splitting this function in two lets a caller
	 * hold two units and interleave them. The ops are COPIED because the
	 * caller's array is usually a local that goes out of scope -- the
	 * buffers they point AT must still be alive at the wait.
	 */
	struct charsiu_fp16_plan inpl;
	struct charsiu_fp16_op inops[FP16_GROUP_MAX];
	unsigned innops;
	int invec, inflight;
	unsigned long abandoned;
	double intcall, intprev;
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
	/* the buffers moved, so whatever the coefficient cache remembers about
	 * their contents is about a buffer that no longer exists */
	f->gen++;
	f->ncoef = 0;
	if (charsiu_bo_alloc(f->dev, wsz, &f->wt) ||
	    charsiu_bo_alloc(f->dev, insz, &f->in) ||
	    charsiu_bo_alloc(f->dev, obsz, &f->ob) ||
	    charsiu_bo_alloc(f->dev, coefsz, &f->coef) ||
	    charsiu_bo_alloc(f->dev, regsz, &f->reg))
		return -1;
	return (f->wt.map && f->in.map && f->ob.map && f->coef.map &&
		f->reg.map) ? 0 : -1;
}

/*
 * A WEIGHT THE CALLER OWNS, WHICH IS THE POINT OF THE WHOLE FILE.
 *
 * The group's memcpy of the caller's weights was 0.39 to 2.21 ms of a round
 * on the first board round -- the largest cost left once the fence was
 * amortised, and pure waste for a KV cache, which is appended to a row at a
 * time and never changes afterwards. This is one device buffer the caller
 * writes rows into at charsiu_fp16_woffset and the hardware reads where it
 * lies.
 *
 * IT IS ZEROED ON THE WAY OUT AND THAT IS NOT TIDINESS. A group runs with
 * whatever n it is given, and the hardware reads the whole weight surface for
 * that n -- including the channels of a cache that has no token in them yet.
 * Zero there contributes zero to a score, which the softmax mask then throws
 * away; uninitialised memory contributes a NaN that spreads through the row.
 */
struct charsiu_fp16_w {
	struct charsiu_bo bo;
	size_t bytes;      /* what the CURRENT k and n occupy */
	size_t room;       /* what was allocated, which may be more */
	unsigned k, n;
	unsigned kroom;    /* the largest k this allocation can be laid out at */
};

struct charsiu_fp16_w *charsiu_fp16_w_alloc_room(struct charsiu_fp16 *f,
						 unsigned k, unsigned n,
						 unsigned kroom)
{
	struct charsiu_matmul mm = { 1, k, n, CHARSIU_FP16, CHARSIU_FP16 };
	struct charsiu_matmul mr = { 1, kroom, n, CHARSIU_FP16, CHARSIU_FP16 };
	struct charsiu_fp16_w *w;

	if (!f || k < 32 || n < 32)
		return NULL;
	if (kroom < k)
		kroom = k;
	mr.k = kroom;
	w = calloc(1, sizeof(*w));
	if (!w)
		return NULL;
	w->bytes = charsiu_weight_bytes(&mm);
	w->room = charsiu_weight_bytes(&mr);
	w->k = k;
	w->n = n;
	w->kroom = kroom;
	if (charsiu_bo_alloc(f->dev, w->room + 4096, &w->bo) || !w->bo.map) {
		free(w);
		return NULL;
	}
	charsiu_bo_prep(f->dev, &w->bo, 1000000000);
	/* THE WHOLE ROOM, not the current bytes. A later set_k exposes the
	 * rest of it to the hardware without anything else zeroing it, and
	 * uninitialised memory in a KV surface is a NaN that spreads. */
	memset(w->bo.map, 0, w->room + 4096);
	charsiu_bo_fini(f->dev, &w->bo);
	return w;
}

struct charsiu_fp16_w *charsiu_fp16_w_alloc(struct charsiu_fp16 *f,
					    unsigned k, unsigned n)
{
	return charsiu_fp16_w_alloc_room(f, k, n, k);
}

/*
 * The caller has re-laid the bytes out at `k` and is saying so. Refuses a k
 * the allocation cannot hold; everything else is the caller's promise.
 */
int charsiu_fp16_w_set_k(struct charsiu_fp16_w *w, unsigned k)
{
	struct charsiu_matmul mm;

	if (!w || k < 32 || k > w->kroom)
		return -1;
	mm.m = 1; mm.k = k; mm.n = w->n;
	mm.wdtype = CHARSIU_FP16; mm.adtype = CHARSIU_FP16;
	if (charsiu_weight_bytes(&mm) > w->room)
		return -1;
	w->bytes = charsiu_weight_bytes(&mm);
	w->k = k;
	return 0;
}

unsigned charsiu_fp16_w_room(const struct charsiu_fp16_w *w)
{
	return w ? w->kroom : 0;
}

void charsiu_fp16_w_free(struct charsiu_fp16 *f, struct charsiu_fp16_w *w)
{
	if (!f || !w)
		return;
	charsiu_bo_free(f->dev, &w->bo);
	free(w);
}

void *charsiu_fp16_w_map(struct charsiu_fp16_w *w)
{
	return w ? w->bo.map : NULL;
}

size_t charsiu_fp16_w_bytes(const struct charsiu_fp16_w *w)
{
	return w ? w->bytes : 0;
}

void charsiu_fp16_w_begin(struct charsiu_fp16 *f, struct charsiu_fp16_w *w)
{
	if (f && w)
		charsiu_bo_prep(f->dev, &w->bo, 1000000000);
}

void charsiu_fp16_w_end(struct charsiu_fp16 *f, struct charsiu_fp16_w *w)
{
	if (f && w)
		charsiu_bo_fini(f->dev, &w->bo);
}

/*
 * A SECOND OPEN OF THE SAME DEVICE IS NOT FREE, AND THE BOARD SAID SO.
 *
 * This used to call charsiu_open() unconditionally, so a runtime that turned
 * the fp16 attention mirror on held TWO file descriptors on one accel device.
 * rocket keeps its scheduler entities and, since attach-once, its IOMMU domain
 * per FILE, so the second handle is a second client of the same hardware even
 * when it submits nothing.
 *
 * Decode paid for it, and the board has now said so with the arms paired. The
 * decode column of board_vendor.sh is bimodal and best-of-two lands in the low
 * mode often enough to be useless; at SIX repeats both arms sit on the high one
 * and become comparable:
 *
 *     Qwen3      base 24.76, 24.73    mirror 24.79, 24.77
 *     TinyLLAMA  base 20.68, 20.63    mirror 20.68, 20.66
 *
 * Identical, within 0.2%. Before this change and the one that stopped decode
 * writing the mirror, the same configuration cost 12 to 14% of decode. The two
 * of them together take it to nothing.
 *
 * What is left is TTFT, and it is not this: 685/673 -> 939/940 on Qwen3 and
 * 912/910 -> 1108/1110 on TinyLLAMA, which is the mirror being BUILT for a 110
 * token prompt that is far too short to pay it back. That is the envelope, not
 * a defect.
 *
 * charsiu_fp16_open_on() borrows a device the caller already has and does not
 * close it. charsiu_fp16_open() keeps its old meaning for the standalone
 * probes, which have no other handle to lend.
 */
struct charsiu_fp16 *charsiu_fp16_open_on(struct charsiu_device *dev)
{
	struct charsiu_fp16 *f;

	if (!dev)
		return NULL;
	f = calloc(1, sizeof(*f));
	if (!f)
		return NULL;
	f->dev = dev;
	f->borrowed = 1;
	return f;
}

/* the device this handle runs on, so a caller can open a SECOND handle on it
 * rather than a second file. See the two-handle note in struct attn_npu. */
void charsiu_fp16_name(struct charsiu_fp16 *f, const char *name)
{
	if (f)
		f->name = name;
}

struct charsiu_device *charsiu_fp16_device(struct charsiu_fp16 *f)
{
	return f ? f->dev : NULL;
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

/*
 * THE ARM IS NAMED IN BOTH DIRECTIONS. CHARSIU_FP16_PACK=1 is the vector
 * run, =0 the per element loop this file shipped with; the scalar arm stays
 * compiled so a board round can price the change against itself in one boot
 * rather than against a number from another one.
 */
static int pack_vector(void)
{
	static int vec = -1;

	if (vec < 0)
		vec = charsiu_env_flag("CHARSIU_FP16_PACK", 1);
	return vec;
}

static inline void pack_run(int vec, uint16_t *d, const float *x, size_t n)
{
	if (vec) {
		charsiu_f2h_run(d, x, n);
	} else {
		for (size_t e = 0; e < n; e++)
			d[e] = charsiu_f2h(x[e]);
	}
}

static int fullscan(void);

/*
 * THREE OF THIS FUNCTION'S LOOPS ARE PER OP AND WERE ALL ON ONE CORE.
 *
 * pack, the poison and the readback each walk nops independent regions, and
 * at 852 tokens they were 489, 668 and 404 ms against the hardware's own 915.
 * The pool is right there -- the softmax two frames up this stack uses it --
 * and this path never asked for it, which is the same shape as `silu * up`
 * and as the softmax itself.
 *
 * The pool degrades to one core when nothing started it (a whisper or a
 * vision graph has no llama_state), so this is safe in every caller; it is
 * simply not a speed-up there.
 *
 * AND THE SHARED COUNTERS BECOME PER OP ARRAYS. `f->packel += nel` from
 * four workers is a lost update, and a counter that is quietly low is worse
 * than no counter: it is the number somebody divides by.
 */
struct fp16_ops_ctx {
	struct charsiu_fp16 *f;
	const struct charsiu_fp16_op *ops;
	const struct charsiu_fp16_plan *pl;
	int vec;
	unsigned long long el[FP16_GROUP_MAX], sk[FP16_GROUP_MAX];
	unsigned unwritten[FP16_GROUP_MAX];
	unsigned char bad[FP16_GROUP_MAX], borrowed[FP16_GROUP_MAX];
};

static int fp16_pool(void)
{
	static int v = -1;

	if (v < 0)
		v = charsiu_env_flag("CHARSIU_FP16_POOL", 1);
	return v;
}

static void fp16_run_ops(void (*fn)(void *, uint64_t, uint64_t),
			 struct fp16_ops_ctx *c, unsigned nops)
{
	if (fp16_pool())
		charsiu_parallel_for(fn, c, (uint64_t)nops);
	else
		fn(c, 0, (uint64_t)nops);
}

static void fp16_pack_ops(void *ctx, uint64_t i0, uint64_t n)
{
	struct fp16_ops_ctx *c = ctx;
	uint64_t i;

	for (i = i0; i < i0 + n; i++) {
		/*
		 * THE BOUND, THE STORE AND THE ZEROING WERE ALL PER ELEMENT,
		 * and this loop runs 262 thousand times at 32 ops.
		 *
		 * It was a call into another translation unit, a comparison
		 * against the region size, and two byte stores an element, on
		 * top of zeroing the whole region first. The board put it at
		 * 1.69 ms a round. The bound is loop arithmetic, the store is
		 * one aligned 16 bit write -- the region starts on a page --
		 * the conversion is charsiu_f2h inline -- a native 16 bit
		 * store, and this SoC is little endian, which is the same
		 * bytes the two byte stores wrote -- and only the TAIL past
		 * the activation needs zeroing, because that is the only part
		 * the hardware reads that this does not write.
		 */
		const struct charsiu_fp16_op *op = &c->ops[i];
		uint16_t *d = (uint16_t *)((uint8_t *)c->f->in.map
					   + c->pl->ioff[i]);
		size_t nel = (size_t)op->m * op->k;
		size_t cap = c->pl->isz[i] / 2;
		size_t xs = op->xstride ? op->xstride : op->k;
		const float *X = op->X;
		unsigned long long sk = 0;
		int tri = op->xtri0 != 0;

		if (nel > cap)
			nel = cap;
		if (op->fill) {
			/* the caller writes the halves; this still owns the
			 * offsets, the bound and the causal tail, so xtri0
			 * means exactly what it means on the converting path */
			size_t e = 0;

			for (unsigned r = 0; r < op->m && e < nel; r++) {
				size_t cn = op->k, keep;

				if (cn > nel - e)
					cn = nel - e;
				keep = cn;
				if (tri) {
					keep = (size_t)op->xtri0 + r;
					if (keep > cn)
						keep = cn;
				}
				if (keep < cn) {
					memset(d + e + keep, 0,
					       (cn - keep) * 2);
					sk += cn - keep;
				}
				op->fill(op->fill_ctx, d + e, (unsigned)i, r,
					 (unsigned)keep);
				e += cn;
			}
			c->el[i] = nel;
			c->sk[i] = sk;
			/* the same slack the converting path zeroes below:
			 * the CBUF reads past what a matmul writes */
			memset(d + nel, 0,
			       charsiu_fp16_up4k(c->pl->isz[i]) - nel * 2);
			continue;
		}
		if (xs == op->k && !tri) {
			pack_run(c->vec, d, X, nel);
		} else {
			size_t e = 0;

			/* the bound was per element and the row length is
			 * known: a row contributes min(k, what is left) */
			for (unsigned r = 0; r < op->m && e < nel; r++) {
				const float *xr = X + (size_t)r * xs;
				size_t cn = op->k, keep;

				if (cn > nel - e)
					cn = nel - e;
				keep = cn;
				if (tri) {
					keep = (size_t)op->xtri0 + r;
					if (keep > cn)
						keep = cn;
				}
				pack_run(c->vec, d + e, xr, keep);
				if (keep < cn) {
					memset(d + e + keep, 0,
					       (cn - keep) * 2);
					sk += cn - keep;
				}
				e += cn;
			}
		}
		c->el[i] = nel;
		c->sk[i] = sk;
		memset(d + nel, 0,
		       charsiu_fp16_up4k(c->pl->isz[i]) - nel * 2);
	}
}

static void fp16_poison_ops(void *ctx, uint64_t i0, uint64_t n)
{
	struct fp16_ops_ctx *c = ctx;
	uint64_t i;

	for (i = i0; i < i0 + n; i++) {
		uint32_t *o = (uint32_t *)((uint8_t *)c->f->ob.map
					   + c->pl->ooff[i]);

		if (fullscan()) {
			for (unsigned e = 0; e < c->ops[i].m * c->ops[i].n;
			     e++)
				o[e] = CHARSIU_POISON;
		} else {
			charsiu_poison_rows(o, c->ops[i].m, c->ops[i].n);
		}
	}
}

/*
 * THE CAUSAL TRIANGLE HAS NO KNOB, AND THAT IS DELIBERATE. It began as one
 * -- CHARSIU_FP16_TRI=0 converted the tail instead of zeroing it, and r400
 * priced the difference at 8% of the pack. Then the caller stopped zeroing
 * the source, because it only ever zeroed it so this could convert a zero
 * into a zero. Past xtri0 + r the source now holds the RAW scores, so the
 * memset is not an optimisation any more: it is the answer. An arm that
 * turned it off would be an arm that is wrong.
 */

/*
 * THE ARM IS NAMED IN BOTH DIRECTIONS. CHARSIU_FP16_FULLSCAN=1 counts every
 * poisoned word the way this file shipped, =0 stops at the first one that is
 * not. Same outcome either way; the knob exists so a board round can price the
 * difference inside one boot.
 */
static void fp16_read_ops(void *ctx, uint64_t i0, uint64_t n)
{
	struct fp16_ops_ctx *c = ctx;
	uint64_t i;

	for (i = i0; i < i0 + n; i++) {
		const struct charsiu_fp16_op *op = &c->ops[i];
		const uint32_t *o = (const uint32_t *)
			((const uint8_t *)c->f->ob.map + c->pl->ooff[i]);
		unsigned cells = op->m * op->n;

		/*
		 * THE OTHER SIDE OF THE SENTINEL. With one a row the question
		 * "did this op write nothing" is m comparisons, and "did it
		 * write only some of its rows" becomes askable for the first
		 * time -- the every-cell form could only ever say all or not
		 * all, so a half written output was accepted in silence.
		 */
		if (fullscan()) {
			unsigned untouched = 0, e;

			for (e = 0; e < cells; e++)
				untouched += o[e] == CHARSIU_POISON;
			if (untouched == cells) {
				c->bad[i] = 1;
				continue;      /* this op wrote nothing */
			}
		} else {
			unsigned unwritten;

			if (charsiu_poison_verdict(o, op->m, op->n,
						   &unwritten)) {
				c->bad[i] = 1;
				continue;      /* this op wrote nothing */
			}
			c->unwritten[i] = unwritten;
		}
		/*
		 * A NULL Y MEANS LEAVE IT WHERE IT IS. The board put the
		 * copy out at 1.17 ms a round on the 80 row scores shape, and
		 * a caller that is about to run a softmax over these numbers
		 * reads them once either way -- the copy is a write and a
		 * second read on top. charsiu_fp16_out hands back the address
		 * and charsiu_fp16_release closes the buffer.
		 */
		if (op->Y && (!op->ystride || op->ystride == op->n)) {
			memcpy(op->Y, o, (size_t)cells * 4);
		} else if (op->Y) {
			for (unsigned r = 0; r < op->m; r++)
				memcpy(op->Y + (size_t)r * op->ystride,
				       (const float *)o + (size_t)r * op->n,
				       (size_t)op->n * 4);
		} else {
			c->borrowed[i] = 1;
		}
	}
}

static int fullscan(void)
{
	static int v = -1;

	if (v < 0)
		v = charsiu_env_flag("CHARSIU_FP16_FULLSCAN", 0);
	return v;
}


/*
 * EVERY ONE OF THESE COUNTERS ALREADY EXISTED AND NOTHING READ THEM.
 *
 * `calls`, `submits`, `refused` and the whole `t` struct -- wcopy, pack,
 * coefs, emit, submit, fence, read -- have been accumulated since this file
 * was written, and no caller has ever printed one. So "attention on the NPU is
 * 61% dearer than on the CPU at head_dim 64" was a wall-clock difference with
 * nothing inside it, and the obvious next question -- IS IT THE FENCE, or the
 * weight copy this file's own comment says charsiu_fp16_woffset exists to
 * remove -- could not be asked.
 *
 * Same shape as the attention refusal counters two rounds ago: the number that
 * was printed was the number that was easy to count.
 */
static void fp16_report(const struct charsiu_fp16 *f)
{
	double tot = f->t.plan + f->t.wcopy + f->t.pack + f->t.psync
		   + f->t.coefs + f->t.emit + f->t.poison + f->t.submit
		   + f->t.fence + f->t.read + f->t.other;

	if (!f->calls)
		return;
	fprintf(stderr, "charsiu fp16 %s: %lu calls, %lu submits, %lu"
		" refused; %.0f ms accounted\n", f->name ? f->name : "-",
		f->calls, f->submits, f->refused, tot);
	if (tot <= 0.0)
		return;
	fprintf(stderr, "charsiu fp16:  plan %.0f  wcopy %.0f  pack %.0f"
		"  psync %.0f  coefs %.0f  emit %.0f  poison %.0f  submit %.0f"
		"  fence %.0f  read %.0f  other %.0f ms\n",
		f->t.plan, f->t.wcopy, f->t.pack, f->t.psync, f->t.coefs,
		f->t.emit, f->t.poison, f->t.submit, f->t.fence, f->t.read,
		f->t.other);
	if (f->abandoned)
		fprintf(stderr, "charsiu fp16:  %lu groups were submitted"
			" and never waited on\n", f->abandoned);
	if (f->partial)
		fprintf(stderr, "charsiu fp16:  %llu ROWS WERE NEVER"
			" WRITTEN by a job that wrote others\n", f->partial);
	fprintf(stderr, "charsiu fp16:  %.0f%% fence, %.0f%% weight copy,"
		" %.3f ms a call\n", 100.0 * f->t.fence / tot,
		100.0 * f->t.wcopy / tot, tot / (double)f->calls);
	/*
	 * THE PER ELEMENT COST IS THE NUMBER, NOT THE PER CALL ONE. Reading
	 * `pack` against `calls` put a conversion at 235 ns, which is fifty
	 * times what fourteen instructions can cost and sent one round looking
	 * for a wall that was not there. The count is right here now.
	 */
	if (f->packel)
		fprintf(stderr, "charsiu fp16:  pack %llu elements, %.2f ns"
			" each (%s arm)\n", f->packel,
			f->t.pack * 1e6 / (double)f->packel,
			pack_vector() ? "vector" : "scalar");
	if (f->macs && f->t.fence > 0.0)
		fprintf(stderr, "charsiu fp16:  %.1f GMAC in %.0f ms of fence"
			" = %.3f TMAC/s, %.0f us a submit\n",
			f->macs / 1e9, f->t.fence,
			f->macs / f->t.fence / 1e9,
			1000.0 * f->t.fence / (double)f->submits);
	if (f->preskip || f->prewrote)
		fprintf(stderr, "charsiu fp16:  the sentinels were already in"
			" the buffer for %llu of %llu groups\n", f->preskip,
			f->preskip + f->prewrote);
	if (f->emitbuilt || f->emitskip)
		fprintf(stderr, "charsiu fp16:  the register streams were"
			" already there for %llu of %llu groups\n",
			f->emitskip, f->emitskip + f->emitbuilt);
	if (f->trisk)
		fprintf(stderr, "charsiu fp16:  %llu of them memset as the"
			" causal tail (%.0f%%)\n", f->trisk,
			100.0 * (double)f->trisk / (double)f->packel);
}

void charsiu_fp16_close(struct charsiu_fp16 *f)
{
	if (!f)
		return;
	if (charsiu_env_flag("CHARSIU_STAGES", 0))
		fp16_report(f);
	charsiu_fp16_release(f);
	charsiu_bo_free(f->dev, &f->reg);  charsiu_bo_free(f->dev, &f->coef);
	charsiu_bo_free(f->dev, &f->ob);   charsiu_bo_free(f->dev, &f->in);
	charsiu_bo_free(f->dev, &f->wt);
	if (!f->borrowed)
		charsiu_close(f->dev);
	free(f);
}

/*
 * X is m by k, row major, float. W holds charsiu_fp16_wbytes(k, n) bytes in
 * the layout charsiu_fp16_woffset describes. Y is m by n, row major, float.
 *
 * IT REFUSES RATHER THAN COMPUTES WHAT IT HAS NOT BEEN SHOWN. K=16 N=8
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
	/* BEFORE want(), which may free the buffer a borrowed answer is
	 * still sitting in: releasing it afterwards would be a fini on a
	 * handle that no longer exists */
	charsiu_fp16_release(f);
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
	 * ROW MAJOR, NOT charsiu_pack_input_f16's INTERLEAVE, AND THE BOARD
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
		/* AND THE GROUP'S CACHE NO LONGER DESCRIBES THIS BUFFER.
		 * This writes its own coefficients at offset 0, which is where
		 * a group's first region sits. Without this line a group that
		 * ran before a single call and again after it would find its
		 * bookkeeping intact, skip the rebuild, and multiply by
		 * whatever the single call left there -- a wrong number with
		 * nothing anywhere reporting an error. */
		f->ncoef = 0;
		/* AND THE REGISTER STREAMS FOR THE SAME REASON. This call
		 * emits its own stream over the group's first slot. */
		f->nregsig = 0;
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
	 * A SENTINEL, NOT ZEROS. A job that never wrote and a job that
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
 * SEVERAL MATMULS, ONE SUBMIT AND ONE FENCE, BECAUSE THE FENCE IS THE JOB.
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
 * THE GROUPING CHANGES NO ARITHMETIC, so the results must be IDENTICAL to
 * the same ops run one at a time. npu_fp16_test --group checks that bit for
 * bit, at MIXED SHAPES, before it reports a single millisecond: uniform shapes
 * cannot catch an offset that is wrong by a whole sub buffer.
 *
 * AND THE WEIGHT COPY IS STILL HERE. Every op's W is memcpy'd into the
 * device buffer, which for a K=1024 N=64 cache is 128 kB an op. That is what
 * charsiu_fp16_woffset exists to remove -- a caller that appends its KV cache
 * straight into the buffer pays none of it -- and the times below name it
 * separately so the next round can see what is left after it goes.
 */
/*
 * ALLOCATE FOR THE WIDEST SHAPE ONCE, INSTEAD OF GROWING INTO IT.
 *
 * want() only grows, so a caller whose shapes get bigger reallocates all five
 * buffers every time -- five buffer objects, one of them 21 MB for the
 * attention scores, mmap and first touch included. At 852 tokens the scores
 * unit spent 73 ms in `plan`, which is where that lands, for about five
 * growths in the first layer of each prompt chunk; every later layer runs the
 * same shapes and grows nothing.
 *
 * IT ONLY EVER MAKES THE BUFFERS BIGGER, and it does not poison, submit or
 * touch `last`: it is the allocation half of a call and nothing else. A caller
 * that reserves a shape it never runs has wasted memory and changed no answer.
 */
int charsiu_fp16_reserve(struct charsiu_fp16 *f,
			 const struct charsiu_fp16_op *ops, unsigned nops)
{
	struct charsiu_fp16_plan pl;

	if (!f || charsiu_fp16_make_plan(ops, nops, &pl))
		return -1;
	return want(f, pl.wtot + 4096, pl.itot + 4096, pl.otot + 4096,
		    pl.ctot + 4096,
		    (size_t)nops * FP16_REG_STRIDE + 4096);
}

int charsiu_fp16_matmul_group_submit(struct charsiu_fp16 *f,
				     const struct charsiu_fp16_op *ops,
				     unsigned nops)
{
	struct charsiu_job job[FP16_GROUP_MAX];
	struct charsiu_task task[FP16_GROUP_MAX];
	struct charsiu_fp16_plan pl;
	unsigned i;
	int prepoisoned = 0;
	struct charsiu_joblist jl;
	uint32_t ins[3 + FP16_GROUP_MAX], outs[1];
	unsigned nin;
	int32_t *zero;
	int vec = pack_vector();
	double t0, tcall = 0.0, tprev = 0.0;

	if (!f)
		return -1;
	tcall = now_ms();
	tprev = f->t.plan + f->t.wcopy + f->t.pack + f->t.psync + f->t.coefs
	      + f->t.emit + f->t.poison + f->t.submit + f->t.fence + f->t.read;
	t0 = tcall;
	/*
	 * A JOB STILL IN FLIGHT OWNS THIS BUFFER. Since the group split,
	 * a caller that gives up between submit() and wait() -- every
	 * `fallbacks++; return -1` in the attention layer does -- leaves the
	 * hardware writing into the output buffer this call is about to
	 * re-plan. Wait it out and throw the answers away: the caller that
	 * abandoned them has already fallen back to the CPU for that layer.
	 */
	if (f->inflight) {
		charsiu_bo_prep(f->dev, &f->ob, 1000000000);
		charsiu_bo_fini(f->dev, &f->ob);
		f->inflight = 0;
		f->held = 0;
		f->prepoisoned = 0;
		f->abandoned++;
	}
	/* a previous group's answers are still being read out of the buffer
	 * this is about to overwrite */
	charsiu_fp16_release(f);
	/* the shapes, and where each op sits in the four shared buffers.
	 * tests/fp16_plan.c walks this on a desk; nothing below recomputes it */
	if (charsiu_fp16_make_plan(ops, nops, &pl)) {
		f->refused++;
		return -1;
	}
	for (i = 0; i < nops; i++) {
		const struct charsiu_fp16_op *o = &ops[i];

		/* a NULL Y is not a missing argument, it is the borrow: the
		 * answer stays in the device buffer and charsiu_fp16_out
		 * points at it. Refusing it here made the whole feature
		 * unreachable and the probe's own arm is what said so. */
		/* X may be NULL when the caller fills the surface itself,
		 * and only then: a NULL X with no fill is a missing argument */
		if ((!o->X && !o->fill) || (!o->W && !o->Wbuf)) {
			f->refused++;
			return -1;
		}
		/*
		 * WHAT A CALLER OWNED WEIGHT MAY BE RUN AT, and it is not
		 * "anything smaller".
		 *
		 * charsiu_fp16_woffset is
		 *     (n/16)*16*ke + (k/32)*32*ngsz + (n%16)*kgsz + k%32
		 * where ke is the PADDED k and ngsz is 16 for every group but
		 * a partial last one. So an offset does not depend on n while
		 * every group is full, which is what lets a cache be appended
		 * to along n and read at whatever n it has reached -- and it
		 * DOES depend on k through ke, always. A buffer written at one
		 * k and run at another is not a smaller matmul of the same
		 * weights, it is a different permutation of them, and it comes
		 * back as a plausible wrong number.
		 */
		if (o->Wbuf) {
			unsigned ng = charsiu_weight_ngroup(CHARSIU_FP16);

			if (o->Wbuf->k != o->k || o->n > o->Wbuf->n ||
			    (o->n != o->Wbuf->n && (o->n % ng))) {
				f->refused++;
				return -1;
			}
		}
	}
	if (want(f, pl.wtot + 4096, pl.itot + 4096, pl.otot + 4096,
		 pl.ctot + 4096, (size_t)nops * FP16_REG_STRIDE + 4096))
		return -1;
	/*
	 * IS THE BUFFER ALREADY POISONED FOR EXACTLY THIS GROUP? Offsets
	 * alone are not enough: the row sentinel sits at o[r * n] for r < m,
	 * so a group with the same regions but a different m or n would be
	 * checking words nobody wrote.
	 */
	if (f->prepoisoned) {
		unsigned om[FP16_GROUP_MAX], on[FP16_GROUP_MAX];

		for (i = 0; i < nops; i++) {
			om[i] = ops[i].m;
			on[i] = ops[i].n;
		}
		prepoisoned = charsiu_poison_matches(pl.nops, pl.ooff, om, on,
						     f->last.nops,
						     f->last.ooff,
						     f->pm, f->pn);
	}
	f->prepoisoned = 0;
	f->preskip += prepoisoned;
	f->prewrote += !prepoisoned;
	f->t.plan += now_ms() - t0;

	/* ONE prep and ONE fini a buffer for the whole group. Cache
	 * maintenance per op would put back a per dispatch cost of exactly
	 * the kind this function exists to remove. */
	t0 = now_ms();
	{
		unsigned copies = 0;

		for (i = 0; i < nops; i++)
			copies += pl.wsz[i] != 0;
		if (copies) {
			charsiu_bo_prep(f->dev, &f->wt, 1000000000);
			for (i = 0; i < nops; i++)
				if (pl.wsz[i])
					memcpy((uint8_t *)f->wt.map
					       + pl.woff[i], ops[i].W,
					       pl.wsz[i]);
			charsiu_bo_fini(f->dev, &f->wt);
		}
	}
	f->t.wcopy += now_ms() - t0;

	/* row major, [m][k], which is what --inslots measured slot by slot */
	/*
	 * THE PREP ON THE INPUT IS AN INVALIDATE FOR A BUFFER THE HARDWARE
	 * ONLY EVER READS.
	 *
	 * attn_npu_layer already makes this argument for the KV surfaces --
	 * "No prep: the hardware only ever reads these, so the CPU's copy is
	 * never stale and there is nothing to invalidate" -- and the input
	 * surface is the same kind of buffer. What the prep also does is wait
	 * on the buffer's fence, and that wait is already over: the previous
	 * group's own fence ran before this call started.
	 *
	 * It is not free. The sync is charged on the WHOLE buffer object and
	 * `want` grows it and never shrinks it, so at 852 tokens the values
	 * group's two syncs were 112 ms of 877.
	 *
	 * OFF BY DEFAULT, AND THE BOARD SAID SO. 852 tokens, the two fp16
	 * handles, arms alternating: psync 111 -> 62 on the values group and
	 * 18 -> 11 on the scores one, 56 ms. The pack pays 13 of that back,
	 * because it now takes the cache misses the invalidate used to absorb,
	 * so the net is about 43 ms. All nine models on the card are identical
	 * to their token loops with fp16 attention forced on, which is the
	 * check that matters for a cache maintenance change.
	 *
	 * CHARSIU_FP16_IN_PREP=1 puts it back.
	 */
	t0 = now_ms();
	if (charsiu_env_flag("CHARSIU_FP16_IN_PREP", 0))
		charsiu_bo_prep(f->dev, &f->in, 1000000000);
	f->t.psync += now_ms() - t0;
	t0 = now_ms();
	{
		struct fp16_ops_ctx pc = { f, ops, &pl, vec, {0}, {0},
					   {0}, {0}, {0} };

		fp16_run_ops(fp16_pack_ops, &pc, nops);
		for (i = 0; i < nops; i++) {
			f->packel += pc.el[i];
			f->trisk += pc.sk[i];
		}
	}
	f->t.pack += now_ms() - t0;
	t0 = now_ms();
	charsiu_bo_fini(f->dev, &f->in);
	f->t.psync += now_ms() - t0;

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
		job[i].weight_addr = ops[i].Wbuf
			? (uint32_t)ops[i].Wbuf->bo.dma_address
			: (uint32_t)f->wt.dma_address + pl.woff[i];
		job[i].coef_addr = (uint32_t)f->coef.dma_address + pl.coff[i];
	}

	/*
	 * THE COEFFICIENTS ARE THE SAME BYTES EVERY TIME, AND BUILDING
	 * THEM WAS UP TO 23% OF A GROUP.
	 *
	 * charsiu_build_coefs starts by zeroing the whole buffer, which at the
	 * default 65536 element bound is 262 kB, and this unit always feeds it
	 * a zero bias, zero weight sums and unit scales -- so the result
	 * depends on n and nothing else. The first board round measured 0.48
	 * to 1.29 ms a round making up to sixteen identical copies of it.
	 *
	 * The plan already gives one region per distinct n. This builds each
	 * region once, and then not at all: a second call with the same shapes
	 * finds the same regions at the same offsets in a buffer that has not
	 * moved, and the bytes it would write are the bytes already there. A
	 * layer of attention calls this with one shape, twice a layer, every
	 * layer, so the steady state is zero.
	 */
	t0 = now_ms();
	{
		int stale = f->coef_gen != f->gen || f->ncoef != pl.ncoef;
		unsigned char built[FP16_GROUP_MAX] = { 0 };

		for (unsigned c = 0; !stale && c < pl.ncoef; c++)
			stale = f->coefn[c] != pl.coefn[c] ||
				f->coefoff[c] != pl.coefoff[c] ||
				f->coefsz_[c] != pl.coefsz[c];
		if (stale) {
			zero = calloc(pl.nmax, sizeof(*zero));
			if (!zero)
				return -1;
			charsiu_bo_prep(f->dev, &f->coef, 1000000000);
			for (i = 0; i < nops; i++) {
				unsigned c;

				for (c = 0; c < pl.ncoef; c++)
					if (pl.coefoff[c] == pl.coff[i])
						break;
				if (c == pl.ncoef || built[c])
					continue;
				built[c] = 1;
				charsiu_build_coefs(&job[i], zero, zero,
					(uint8_t *)f->coef.map + pl.coff[i]);
			}
			charsiu_bo_fini(f->dev, &f->coef);
			free(zero);
			for (unsigned c = 0; c < pl.ncoef; c++) {
				f->coefn[c] = pl.coefn[c];
				f->coefoff[c] = pl.coefoff[c];
				f->coefsz_[c] = pl.coefsz[c];
			}
			f->ncoef = pl.ncoef;
			f->coef_gen = f->gen;
		}
	}
	f->t.coefs += now_ms() - t0;

	t0 = now_ms();
	{
		/*
		 * THE WEIGHT ADDRESS IS NOT IN THE SIGNATURE, and that is
		 * the whole reason this cache pays.
		 *
		 * Every other field repeats across a chunk's sixteen layers;
		 * the KV surface does not, because each layer has its own. A
		 * cache that required it too hit 0 of 176 groups, measured.
		 * charsiu_patch_weight_addr rewrites the one word that carries
		 * it, and tests/patch_waddr.c requires the result to be byte
		 * for byte what emit_job would have produced -- at every shape
		 * attention runs and all three weight dtypes.
		 */
		int stale = !charsiu_env_flag("CHARSIU_FP16_REGCACHE", 1) ||
			    f->reg_gen != f->gen || f->nregsig != nops;
		int patched = 0;

		for (i = 0; !stale && i < nops; i++)
			stale = f->regsig[i].m != job[i].mm.m ||
				f->regsig[i].k != job[i].mm.k ||
				f->regsig[i].n != job[i].mm.n ||
				f->regsig[i].in != job[i].input_addr ||
				f->regsig[i].out != job[i].output_addr ||
				f->regsig[i].coef != job[i].coef_addr;
		if (!stale) {
			for (i = 0; i < nops; i++)
				patched += f->regsig[i].wt
					!= job[i].weight_addr;
			if (patched) {
				charsiu_bo_prep(f->dev, &f->reg, 1000000000);
				for (i = 0; i < nops; i++) {
					uint64_t *st;

					if (f->regsig[i].wt ==
					    job[i].weight_addr)
						continue;
					st = (uint64_t *)((uint8_t *)f->reg.map
						+ (size_t)i * FP16_REG_STRIDE);
					/* exactly one word, or give up and
					 * emit: the test says one, and a
					 * stream that disagrees is not one
					 * this code has ever seen */
					if (charsiu_patch_weight_addr(st,
						f->regsig[i].nreg,
						job[i].weight_addr) != 1) {
						stale = 1;
						break;
					}
					f->regsig[i].wt = job[i].weight_addr;
				}
				charsiu_bo_fini(f->dev, &f->reg);
			}
		}
		if (stale) {
			charsiu_bo_prep(f->dev, &f->reg, 1000000000);
			for (i = 0; i < nops; i++) {
				size_t nreg = charsiu_emit_job(&job[i],
					(uint64_t *)((uint8_t *)f->reg.map
						     + (size_t)i
						     * FP16_REG_STRIDE),
					FP16_REG_STRIDE / 8);

				if (!nreg) {
					charsiu_bo_fini(f->dev, &f->reg);
					f->nregsig = 0;
					return -1;
				}
				f->regsig[i].m = job[i].mm.m;
				f->regsig[i].k = job[i].mm.k;
				f->regsig[i].n = job[i].mm.n;
				f->regsig[i].in = job[i].input_addr;
				f->regsig[i].out = job[i].output_addr;
				f->regsig[i].wt = job[i].weight_addr;
				f->regsig[i].coef = job[i].coef_addr;
				f->regsig[i].nreg = nreg;
			}
			charsiu_bo_fini(f->dev, &f->reg);
			f->nregsig = nops;
			f->reg_gen = f->gen;
			f->emitbuilt++;
		} else {
			f->emitskip++;
		}
		for (i = 0; i < nops; i++) {
			task[i].regcmd = (uint32_t)f->reg.dma_address
				       + (uint32_t)(i * FP16_REG_STRIDE);
			task[i].regcmd_count = (uint32_t)f->regsig[i].nreg;
		}
	}
	f->t.emit += now_ms() - t0;

	/*
	 * THE SENTINEL, AND IT WAS WRITTEN INTO EVERY CELL OF EVERY OUTPUT.
	 *
	 * A job that never wrote and a job that computed zero are the same
	 * four bytes otherwise, so the buffer is poisoned before the submit.
	 * Poisoning ALL of it is a full scalar pass over the answer before the
	 * hardware is even asked -- for the scores shape that is m * npad
	 * words an op and thirty two ops a group -- and at 852 tokens it was
	 * 668 ms, 23% of the two group calls, sitting in the one region of
	 * this function that had no clock on it. Its twin on the readback side
	 * was found a round earlier; this is the expensive half.
	 *
	 * ONE SENTINEL A ROW IS STRICTLY MORE INFORMATIVE, not a weakening.
	 * A job writes its whole output or none of it, so if any row's first
	 * word survived, that row was not written -- and the old check could
	 * only ever say "ALL of it is poison", never "some of it". This one
	 * says both, and it costs m words instead of m * n.
	 *
	 * CHARSIU_FP16_FULLSCAN=1 puts back the every-cell form on BOTH sides,
	 * so the arms are coherent and a board round can price it in one boot.
	 */
	t0 = now_ms();
	if (!prepoisoned) {
		charsiu_bo_prep(f->dev, &f->ob, 1000000000);
		{
			struct fp16_ops_ctx qc = { f, ops, &pl, vec, {0}, {0},
						   {0}, {0}, {0} };

			fp16_run_ops(fp16_poison_ops, &qc, nops);
		}
		charsiu_bo_fini(f->dev, &f->ob);
	}
	f->t.poison += now_ms() - t0;

	t0 = now_ms();
	ins[0] = f->in.handle;
	ins[1] = f->wt.handle;
	ins[2] = f->coef.handle;
	nin = 3;
	/* every caller owned weight is its own buffer object, and a job names
	 * the buffers it reads. Twice is not an error but it is not useful. */
	for (i = 0; i < nops; i++)
		if (ops[i].Wbuf) {
			unsigned j, seen = 0;

			for (j = 0; j < nin; j++)
				if (ins[j] == ops[i].Wbuf->bo.handle)
					seen = 1;
			if (!seen)
				ins[nin++] = ops[i].Wbuf->bo.handle;
		}
	outs[0] = f->ob.handle;
	/*
	 * THE PROBE HATCH, because "the group is wrong" has two causes and
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
			split[i].in_count = nin;
			split[i].out_handles = outs;
			split[i].out_count = 1;
		}
		if (charsiu_submit_jobs(f->dev, split, nops))
			return -1;
	} else {
		jl.tasks = task;
		jl.task_count = nops;
		jl.in_handles = ins;
		jl.in_count = nin;
		jl.out_handles = outs;
		jl.out_count = 1;
		if (charsiu_submit_jobs(f->dev, &jl, 1))
			return -1;
	}
	f->submits++;
	for (i = 0; i < nops; i++)
		f->macs += (unsigned long long)ops[i].m * ops[i].k * ops[i].n;
	f->t.submit += now_ms() - t0;

	/* everything the wait needs that this half computed */
	f->inpl = pl;
	memcpy(f->inops, ops, nops * sizeof(*ops));
	f->innops = nops;
	f->invec = vec;
	f->intcall = tcall;
	f->intprev = tprev;
	f->inflight = 1;
	return 0;
}

/*
 * The other half: the fence, the readback and the bookkeeping. A unit with
 * nothing in flight returns -1 rather than waiting on whatever the buffer
 * last held.
 */
int charsiu_fp16_matmul_group_wait(struct charsiu_fp16 *f)
{
	const struct charsiu_fp16_op *ops;
	struct charsiu_fp16_plan pl;
	unsigned i, nops, bad = 0, borrowed = 0;
	double t0, tcall, tprev;
	int vec;

	if (!f || !f->inflight)
		return -1;
	f->inflight = 0;
	ops = f->inops;
	pl = f->inpl;
	nops = f->innops;
	vec = f->invec;
	tcall = f->intcall;
	tprev = f->intprev;

	t0 = now_ms();
	charsiu_bo_prep(f->dev, &f->ob, 1000000000);   /* the one fence wait */
	f->t.fence += now_ms() - t0;

	t0 = now_ms();
	{
		struct fp16_ops_ctx rc = { f, ops, &pl, vec, {0}, {0},
					   {0}, {0}, {0} };

		fp16_run_ops(fp16_read_ops, &rc, nops);
		for (i = 0; i < nops; i++) {
			bad += rc.bad[i];
			borrowed |= rc.borrowed[i];
			f->partial += rc.unwritten[i];
		}
	}
	f->last = pl;
	for (i = 0; i < nops; i++) {
		f->pm[i] = ops[i].m;
		f->pn[i] = ops[i].n;
	}
	if (borrowed) {
		f->held = 1;
	} else {
		charsiu_bo_fini(f->dev, &f->ob);
		f->held = 0;
	}
	f->t.read += now_ms() - t0;
	f->t.other += now_ms() - tcall
		    - (f->t.plan + f->t.wcopy + f->t.pack + f->t.psync
		       + f->t.coefs + f->t.emit + f->t.poison + f->t.submit
		       + f->t.fence + f->t.read - tprev);
	if (bad) {
		f->refused += bad;
		return -1;
	}
	f->calls += nops;
	return 0;
}

int charsiu_fp16_matmul_group(struct charsiu_fp16 *f,
			      const struct charsiu_fp16_op *ops, unsigned nops)
{
	if (charsiu_fp16_matmul_group_submit(f, ops, nops))
		return -1;
	return charsiu_fp16_matmul_group_wait(f);
}

/*
 * The answer of op i, where the hardware left it: m by n floats, row major,
 * valid until the next call or charsiu_fp16_release. NULL if op i was given a
 * Y of its own, or if the last call did not run.
 */
const float *charsiu_fp16_out(const struct charsiu_fp16 *f, unsigned i)
{
	return charsiu_fp16_out_w((struct charsiu_fp16 *)f, i);
}

/*
 * The same address, writable, for a caller that REDUCES OVER THE ANSWER IN
 * PLACE. Attention's softmax does: it reads every score of a row, scales,
 * exponentiates and divides, and the alternative is to copy the whole m by n
 * answer into the caller's own array first and then read it again. At 852
 * tokens that copy is 1.5 GB and 245 ms of `read`.
 *
 * THE BUFFER IS STILL THE DEVICE'S. It is held until the next group on THIS
 * handle, so a caller doing this must not run its next group through the same
 * handle before it has finished -- which is why the attention mirror runs the
 * two matmuls on two handles.
 */
float *charsiu_fp16_out_w(struct charsiu_fp16 *f, unsigned i)
{
	if (!f || !f->held || i >= f->last.nops)
		return NULL;
	return (float *)((uint8_t *)f->ob.map + f->last.ooff[i]);
}

/* the hardware is done with the output buffer and nothing was read out of it */
void charsiu_fp16_drain(struct charsiu_fp16 *f)
{
	if (f && f->inflight) {
		charsiu_bo_prep(f->dev, &f->ob, 1000000000);
		charsiu_bo_fini(f->dev, &f->ob);
		f->inflight = 0;
		f->held = 0;
		f->prepoisoned = 0;
		f->abandoned++;
	}
}

void charsiu_fp16_release(struct charsiu_fp16 *f)
{
	if (f && f->held) {
		charsiu_bo_fini(f->dev, &f->ob);
		f->held = 0;
		f->prepoisoned = 0;
	}
}

/*
 * RELEASE, BUT LEAVE THE NEXT CALL'S SENTINELS BEHIND. This exists to
 * remove two whole-buffer dma_syncs a call, and it has to be a caller-side
 * entry point rather than something charsiu_fp16_matmul_group does at the end
 * of itself.
 *
 * The output buffer is poisoned before every submit so that "the job wrote
 * nothing" can be told from "the job computed zero", and doing it in the usual
 * place costs a prep and a fini of the WHOLE buffer -- 8.6 MB an attention
 * scores call. Those two go away if the sentinels are already in the buffer
 * when the release's fini pushes it to the device.
 *
 * AND THEY CANNOT BE WRITTEN AT THE END OF THE PREVIOUS CALL, which was the
 * obvious way and is wrong: a row's sentinel is its FIRST WORD, and a caller
 * that reduces over the answer in place -- which is the only kind of caller
 * that holds the buffer at all -- writes every word of every row afterwards.
 * The sentinel has to go in after the caller has finished reading, which only
 * the caller knows.
 *
 * Shapes must repeat for it to pay: matmul_group compares the next plan's
 * offsets and each op's m and n against what was poisoned, and poisons
 * normally if anything moved. In attention that is 15 of every 16 calls --
 * one per layer at a fixed shape, changing only when the chunk does.
 */
void charsiu_fp16_poison_and_release(struct charsiu_fp16 *f)
{
	unsigned i;

	if (!f || !f->held)
		return;
	for (i = 0; i < f->last.nops; i++) {
		uint32_t *o = (uint32_t *)((uint8_t *)f->ob.map
					   + f->last.ooff[i]);

		if (fullscan()) {
			for (unsigned e = 0; e < f->pm[i] * f->pn[i]; e++)
				o[e] = CHARSIU_POISON;
		} else {
			charsiu_poison_rows(o, f->pm[i], f->pn[i]);
		}
	}
	charsiu_bo_fini(f->dev, &f->ob);
	f->held = 0;
	f->prepoisoned = 1;
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

/* the group fills these and the single call does not: the split for one
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
