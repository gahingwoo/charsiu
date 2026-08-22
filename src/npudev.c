// SPDX-License-Identifier: GPL-2.0-or-later
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */

/*
 * A projection, on the NPU, inside the decode loop.
 *
 * This is the substitution the whole project has been building towards, and it
 * is a substitution rather than a design because the CPU already computes the
 * same arithmetic:
 *
 *     acc[n] = sum_k a_q[k] * w_q[n][k]        <- npu_matvec, and now the DPU
 *     y[n]   = acc[n] * a_scale * w_scale[n]   <- always the CPU
 *
 * So the acceptance test is not "coherent text". It is BIT IDENTICAL TOKENS
 * against CHARSIU_NPU_QUANT=1 on the CPU, because both sides compute the same
 * integer sum and only the machine differs. Anything else is a defect with a
 * known reference to bisect against.
 *
 * The output comes back as the raw signed 32 bit accumulator (job.acc_out),
 * which board round 312 measured at a projection's shape: M=1 K=2048 N=1024,
 * 1024 of 1024 elements byte exact.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu.h"
#include "charsiu_llm.h"

struct npu_slot {
	struct charsiu_bo wt, coef, regcmd;
	struct charsiu_job job;
	unsigned nreg;
	const struct npu_tensor *t;
};

struct charsiu_npu {
	struct charsiu_device *dev;
	struct charsiu_bo in, out;
	uint8_t *scratch;          /* unsigned bytes for the packers */
	unsigned max_k, max_n;
	struct npu_slot *slot;
	unsigned n_slot, cap;
	unsigned long submits;
};

struct charsiu_npu *charsiu_npu_open(unsigned max_k, unsigned max_n,
				     unsigned max_tensors)
{
	struct charsiu_npu *g = calloc(1, sizeof(*g));

	if (!g)
		return NULL;
	g->dev = charsiu_open("/dev/accel/accel0");
	if (!g->dev) {
		free(g);
		return NULL;
	}
	g->max_k = max_k;
	g->max_n = max_n;
	g->cap = max_tensors;
	g->slot = calloc(max_tensors, sizeof(*g->slot));
	g->scratch = malloc((size_t)max_n * max_k);
	if (!g->slot || !g->scratch)
		goto fail;

	/*
	 * One input and one output buffer for every tensor. The stream carries
	 * their addresses, so they have to be allocated before any job is
	 * built, and they are sized for the widest the model will ever ask for.
	 */
	{
		struct charsiu_matmul widest = { 1, max_k, max_n,
						 CHARSIU_INT8, CHARSIU_INT8 };

		if (charsiu_bo_alloc(g->dev,
				     (size_t)charsiu_entries_per_row(&widest) * 64 + 4096,
				     &g->in) ||
		    charsiu_bo_alloc(g->dev, (size_t)max_n * 4 + 4096, &g->out))
			goto fail;
	}
	return g;

fail:
	charsiu_npu_close(g);
	return NULL;
}

void charsiu_npu_close(struct charsiu_npu *g)
{
	if (!g)
		return;
	if (g->dev) {
		for (unsigned i = 0; i < g->n_slot; i++) {
			charsiu_bo_free(g->dev, &g->slot[i].wt);
			charsiu_bo_free(g->dev, &g->slot[i].coef);
			charsiu_bo_free(g->dev, &g->slot[i].regcmd);
		}
		charsiu_bo_free(g->dev, &g->in);
		charsiu_bo_free(g->dev, &g->out);
		charsiu_close(g->dev);
	}
	free(g->slot);
	free(g->scratch);
	free(g);
}

unsigned long charsiu_npu_submits(const struct charsiu_npu *g)
{
	return g ? g->submits : 0;
}

int charsiu_npu_add(struct charsiu_npu *g, const struct npu_tensor *t)
{
	struct npu_slot *s;
	int32_t *bias;
	size_t i;

	if (g->n_slot == g->cap || t->k > g->max_k || t->n > g->max_n)
		return -1;
	s = &g->slot[g->n_slot];
	s->t = t;

	s->job.mm.m = 1;
	s->job.mm.k = (unsigned)t->k;
	s->job.mm.n = (unsigned)t->n;
	s->job.mm.wdtype = CHARSIU_INT8;
	s->job.mm.adtype = CHARSIU_INT8;
	s->job.input_zero_point = 128;
	s->job.weight_zero_point = 128;
	s->job.output_zero_point = 0;
	/*
	 * The scales are not applied by the hardware on this path: acc_out
	 * bypasses the requant entirely and the CPU multiplies afterwards. They
	 * are still set because charsiu_build_coefs computes with them.
	 */
	s->job.input_scale = 1.0f;
	s->job.weight_scale = 1.0f;
	s->job.output_scale = 1.0f;
	s->job.acc_out = 1;

	if (charsiu_bo_alloc(g->dev, charsiu_weight_bytes(&s->job.mm) + 4096, &s->wt) ||
	    charsiu_bo_alloc(g->dev, charsiu_coef_bytes(&s->job.mm) + 4096, &s->coef) ||
	    charsiu_bo_alloc(g->dev, 4096, &s->regcmd))
		return -1;

	s->job.input_addr = (uint32_t)g->in.dma_address;
	s->job.output_addr = (uint32_t)g->out.dma_address;
	s->job.weight_addr = (uint32_t)s->wt.dma_address;
	s->job.coef_addr = (uint32_t)s->coef.dma_address;

	/* the packers take unsigned bytes around a zero point of 128 */
	for (i = 0; i < (size_t)t->n * t->k; i++)
		g->scratch[i] = (uint8_t)((int)t->q[i] + 128);

	charsiu_bo_prep(g->dev, &s->wt, 1000000000);
	charsiu_pack_weights(&s->job.mm, g->scratch, s->wt.map);
	charsiu_bo_fini(g->dev, &s->wt);

	/*
	 * Zero bias and no lift, so the accumulator arrives unmodified. The
	 * lift exists to clear a fused ReLU in the REQUANT domain, and acc_out
	 * bypasses that domain, so adding it here would only corrupt the sum.
	 */
	bias = calloc(t->n, sizeof(*bias));
	if (!bias)
		return -1;
	setenv("CHARSIU_NO_LIFT", "1", 1);
	charsiu_bo_prep(g->dev, &s->coef, 1000000000);
	charsiu_build_coefs(&s->job, bias, t->wsum, s->coef.map);
	charsiu_bo_fini(g->dev, &s->coef);
	unsetenv("CHARSIU_NO_LIFT");
	free(bias);

	charsiu_bo_prep(g->dev, &s->regcmd, 1000000000);
	s->nreg = (unsigned)charsiu_emit_job(&s->job, s->regcmd.map, 4096 / 8);
	charsiu_bo_fini(g->dev, &s->regcmd);
	if (!s->nreg)
		return -1;

	return (int)g->n_slot++;
}

int charsiu_npu_matvec(struct charsiu_npu *g, int id,
		       const struct charsiu_act *a, float *y)
{
	struct npu_slot *s;
	const int32_t *acc;
	uint32_t inh[3], outh;
	unsigned i;

	if (id < 0 || (unsigned)id >= g->n_slot)
		return -1;
	s = &g->slot[id];
	if ((unsigned)a->n != s->job.mm.k)
		return -1;

	charsiu_bo_prep(g->dev, &g->in, 1000000000);
	for (i = 0; i < s->job.mm.k; i++)
		g->scratch[i] = (uint8_t)((int)a->q1[i] + 128);
	charsiu_pack_input(&s->job.mm, g->scratch, g->in.map, g->in.size,
			   s->job.input_zero_point);
	charsiu_bo_fini(g->dev, &g->in);

	inh[0] = g->in.handle;
	inh[1] = s->wt.handle;
	inh[2] = s->coef.handle;
	outh = g->out.handle;
	if (charsiu_submit(g->dev, &s->regcmd, s->nreg, inh, 3, &outh, 1))
		return -1;
	g->submits++;

	charsiu_bo_prep(g->dev, &g->out, 2000000000);
	acc = g->out.map;
	/* the surface is [n/atom][m][n%atom] and at M = 1 that collapses to n */
	for (i = 0; i < s->job.mm.n; i++)
		y[i] = (float)acc[i] * a->d1 * s->t->scale[i];
	charsiu_bo_fini(g->dev, &g->out);
	return 0;
}
