/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * WHERE A GROUP'S MATMULS SIT INSIDE FOUR SHARED BUFFERS, on a desk.
 *
 * charsiu_fp16_matmul_group puts N matmuls in ONE submit, so there is one
 * weight buffer, one input buffer, one output buffer and one coefficient
 * buffer with N regions in each. Every region is named by arithmetic, and
 * arithmetic that overlaps two regions does not fault: it makes op i answer
 * with part of op j, which is a wrong number and not an error. A wrong answer
 * already shipped in this tree for four commits because its guard lived in the
 * probe and not in the product, so the plan is its own unit and
 * tests/fp16_plan.c walks 3000 of them for overlap, bounds and alignment
 * without a device anywhere.
 *
 * Page alignment is deliberate and not free-ish arithmetic: offsets INSIDE a
 * buffer object are old practice for input and output (npudev has packed K
 * slices at in_stride since round 200), but nothing has established what the
 * weight and coefficient bases want, and a page is a superset of every
 * plausible answer.
 */
#ifndef CHARSIU_FP16PLAN_H
#define CHARSIU_FP16PLAN_H

#include <stddef.h>

#include "charsiu.h"
#include "charsiu_llm.h"

/* one matmul's regcmd stream. 4096/8 entries is what the single call has
 * always handed charsiu_emit_job; a group gives each task the same, so task
 * i's stream begins at i * FP16_REG_STRIDE. charsiu_emit_job returns 0 rather
 * than overrunning, and the group treats that as a refusal. */
#define FP16_REG_STRIDE 4096
#define FP16_GROUP_MAX  32

struct charsiu_fp16_plan {
	size_t woff[FP16_GROUP_MAX], wsz[FP16_GROUP_MAX];
	size_t ioff[FP16_GROUP_MAX], isz[FP16_GROUP_MAX];
	size_t ooff[FP16_GROUP_MAX], osz[FP16_GROUP_MAX];
	/* ⚠ coff MAY BE SHARED: two ops with the same n and the same
	 * coefficient size have byte identical coefficients here, and the
	 * board says building them twice is not free. See below. */
	size_t coff[FP16_GROUP_MAX], csz[FP16_GROUP_MAX];
	unsigned ncoef;                       /* distinct coefficient regions */
	unsigned coefn[FP16_GROUP_MAX];       /* the n of each, in order made */
	size_t coefoff[FP16_GROUP_MAX], coefsz[FP16_GROUP_MAX];
	size_t wtot, itot, otot, ctot;
	unsigned nops, nmax;
};

static inline size_t charsiu_fp16_up4k(size_t v)
{
	return (v + 4095u) & ~(size_t)4095u;
}

/*
 * 0, or -1 if the group is one this cannot run. The shape refusal is the
 * SINGLE call's refusal, repeated here on purpose: K=16 N=8 wedged both cores
 * and timed them out, so anything under the two byte feature atom on either
 * axis is refused rather than attempted, and one bad op refuses the group
 * rather than quietly running the rest.
 */
static inline int charsiu_fp16_make_plan(const struct charsiu_fp16_op *ops,
					 unsigned nops,
					 struct charsiu_fp16_plan *p)
{
	unsigned i;

	if (!ops || !p || !nops || nops > FP16_GROUP_MAX)
		return -1;
	p->wtot = p->itot = p->otot = p->ctot = 0;
	p->ncoef = 0;
	p->nops = nops;
	p->nmax = 0;
	for (i = 0; i < nops; i++) {
		const struct charsiu_fp16_op *o = &ops[i];
		struct charsiu_matmul mm = { o->m, o->k, o->n, CHARSIU_FP16,
					     CHARSIU_FP16 };

		if (!o->m || o->k < 32 || o->n < 32)
			return -1;
		/* ⚠ an op whose weight is already on the device takes no room
		 * in the shared buffer, and must take none: giving a zero
		 * sized region a page would make the plan disagree with what
		 * the group copies, and the ops either side of it would still
		 * have to not overlap */
		p->wsz[i] = o->Wbuf ? 0 : charsiu_weight_bytes(&mm);
		p->isz[i] = (size_t)charsiu_entries_per_row(&mm) * 64 * o->m;
		p->osz[i] = (size_t)o->m * o->n * 4;
		p->csz[i] = charsiu_coef_bytes(&mm);
		p->woff[i] = p->wtot;
		p->wtot += p->wsz[i] ? charsiu_fp16_up4k(p->wsz[i]) : 0;
		p->ioff[i] = p->itot; p->itot += charsiu_fp16_up4k(p->isz[i]);
		p->ooff[i] = p->otot; p->otot += charsiu_fp16_up4k(p->osz[i]);
		/*
		 * ⚠⚠ ONE COEFFICIENT REGION PER DISTINCT n, NOT PER OP.
		 *
		 * This unit builds coefficients from a zero bias, zero weight
		 * sums and unit scales, so their content depends on n and on
		 * nothing else in the op -- and charsiu_build_coefs begins by
		 * zeroing the whole buffer, which at the default 65536 element
		 * bound is 262 kB. The first board round of the group measured
		 * that at 0.48 to 1.29 ms of a round, up to 23% of it, for
		 * sixteen byte identical copies. A layer of attention is H ops
		 * of ONE shape, so this is H regions becoming 1.
		 *
		 * Keyed on the size as well as n because CHARSIU_COEF_ELEMS=0
		 * asks for a k*n bound, and then two ops with the same n and
		 * different k do not have the same buffer.
		 */
		{
			unsigned c, found = 0;

			for (c = 0; c < p->ncoef; c++)
				if (p->coefn[c] == o->n &&
				    p->coefsz[c] == p->csz[i]) {
					p->coff[i] = p->coefoff[c];
					found = 1;
					break;
				}
			if (!found) {
				p->coefn[p->ncoef] = o->n;
				p->coefsz[p->ncoef] = p->csz[i];
				p->coefoff[p->ncoef] = p->ctot;
				p->coff[i] = p->ctot;
				p->ctot += charsiu_fp16_up4k(p->csz[i]);
				p->ncoef++;
			}
		}
		if (o->n > p->nmax)
			p->nmax = o->n;
	}
	return 0;
}

#endif /* CHARSIU_FP16PLAN_H */
