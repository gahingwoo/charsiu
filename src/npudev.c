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
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu.h"
#include "charsiu_llm.h"

/*
 * One SLICE of a projection.
 *
 * Round 313 put every attention projection on the hardware and got tokens
 * identical to the CPU, then wedged the board on the feed forward: ffn_gate is
 * N = 8192 and ffn_down is K = 8192, and the largest shape anything had been
 * measured at was N = 1024, K = 2048. A 16.8 MB job is eight times the biggest
 * one that has ever worked here.
 *
 * So a projection is cut into slices of a shape that HAS been measured. The
 * output makes that exact rather than approximate: acc_out hands back a raw
 * int32 sum, so partial sums over a split K add up with no rounding at all.
 */
struct npu_slot {
	struct charsiu_bo wt, coef, regcmd;
	struct charsiu_job job;
	unsigned nreg;
	unsigned n0, k0;           /* where this slice starts in the tensor */
};

struct npu_entry {
	const struct npu_tensor *t;
	unsigned first, count;     /* slots, n fastest */
	unsigned n_slices, k_slices;
};

struct charsiu_npu {
	struct charsiu_device *dev;
	struct charsiu_bo in, out;
	uint8_t *scratch;          /* unsigned bytes for the packers */
	int32_t *acc;              /* partial sums over the K slices */
	unsigned nmax, kmax, max_n;
	struct npu_slot *slot;
	unsigned n_slot, slot_cap;
	struct npu_entry *ent;
	unsigned n_ent, ent_cap;
	unsigned long submits;

	/*
	 * A wedged block answers every submit with a driver side timeout and
	 * the ioctl still returns success, so the only reliable detector is the
	 * clock. Three slow submits and this path retires itself: the caller
	 * falls back to the CPU and the run finishes and reports, instead of
	 * repeating a 1.9 second timeout until someone pulls the power.
	 */
	double slow_us;
	int strikes, dead, whined;
	unsigned long slices;
};

static double now_us(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static unsigned env_u(const char *name, unsigned dflt)
{
	const char *e = getenv(name);

	return e ? (unsigned)strtoul(e, NULL, 0) : dflt;
}

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
	/*
	 * The defaults are the shapes round 312 measured exact, not the shapes
	 * the model happens to want.
	 */
	g->nmax = env_u("CHARSIU_NPU_NMAX", 1024);
	g->kmax = env_u("CHARSIU_NPU_KMAX", 2048);
	g->slow_us = (double)env_u("CHARSIU_NPU_SLOW_US", 100000);
	g->max_n = max_n;
	g->ent_cap = max_tensors;
	/* worst case slices per tensor, times the tensors */
	g->slot_cap = max_tensors *
		((max_n + g->nmax - 1) / g->nmax) *
		((max_k + g->kmax - 1) / g->kmax);
	g->ent = calloc(g->ent_cap, sizeof(*g->ent));
	g->slot = calloc(g->slot_cap, sizeof(*g->slot));
	g->scratch = malloc((size_t)g->nmax * g->kmax + max_k);
	g->acc = calloc(max_n, sizeof(*g->acc));
	if (!g->ent || !g->slot || !g->scratch || !g->acc)
		goto fail;

	{
		struct charsiu_matmul widest = { 1, g->kmax, g->nmax,
						 CHARSIU_INT8, CHARSIU_INT8 };

		if (charsiu_bo_alloc(g->dev,
				     (size_t)charsiu_entries_per_row(&widest) * 64 + 4096,
				     &g->in) ||
		    charsiu_bo_alloc(g->dev, (size_t)g->nmax * 4 + 4096, &g->out))
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
	free(g->ent);
	free(g->scratch);
	free(g->acc);
	free(g);
}

unsigned long charsiu_npu_submits(const struct charsiu_npu *g)
{
	return g ? g->submits : 0;
}

/*
 * What the hardware actually did, printed whether it went well or not. A run
 * that cannot say how many jobs it submitted cannot be read as evidence about
 * the hardware, and round 315 was exactly that run.
 */
void charsiu_npu_report(const struct charsiu_npu *g)
{
	if (!g)
		return;
	fprintf(stderr,
		"charsiu NPU: %u tensors, %lu slices, %lu submits%s\n",
		g->n_ent, g->slices, g->submits,
		g->dead ? "  (RETIRED: it stopped answering)" : "");
	if (!g->submits)
		fprintf(stderr,
			"charsiu NPU: NOTHING RAN ON THE HARDWARE. Every number "
			"in this run came from the CPU.\n");
}

/* One slice: rows [n0, n0+n) and columns [k0, k0+k) of t. */
/*
 * Say why, out loud, the first time.
 *
 * Round 315 ran a whole ladder in which the hardware never engaged, and the
 * text was right every time because the CPU quietly did the work. The rung that
 * caught it was the CONTROL THAT WAS SUPPOSED TO FAIL and did not. A silent
 * fallback is worse than a loud failure: it produces a result that looks like
 * evidence.
 */
static void whine(struct charsiu_npu *g, const char *what, unsigned k, unsigned n)
{
	if (g->whined)
		return;
	g->whined = 1;
	fprintf(stderr, "charsiu: NOT on the NPU -- %s (K=%u N=%u)\n", what, k, n);
}

static int add_slice(struct charsiu_npu *g, const struct npu_tensor *t,
		     unsigned n0, unsigned n, unsigned k0, unsigned k)
{
	struct npu_slot *s = &g->slot[g->n_slot];
	int32_t *bias = NULL, *wsum = NULL;
	int rc = -1;

	memset(s, 0, sizeof(*s));
	s->n0 = n0;
	s->k0 = k0;
	s->job.mm.m = 1;
	s->job.mm.k = k;
	s->job.mm.n = n;
	s->job.mm.wdtype = CHARSIU_INT8;
	s->job.mm.adtype = CHARSIU_INT8;
	s->job.input_zero_point = 128;
	s->job.weight_zero_point = 128;
	s->job.output_zero_point = 0;
	s->job.input_scale = 1.0f;
	s->job.weight_scale = 1.0f;
	s->job.output_scale = 1.0f;
	s->job.acc_out = 1;

	if (charsiu_bo_alloc(g->dev, charsiu_weight_bytes(&s->job.mm) + 4096, &s->wt) ||
	    charsiu_bo_alloc(g->dev, charsiu_coef_bytes(&s->job.mm) + 4096, &s->coef) ||
	    charsiu_bo_alloc(g->dev, 4096, &s->regcmd)) {
		whine(g, "a buffer would not allocate", k, n);
		goto out;
	}

	s->job.input_addr = (uint32_t)g->in.dma_address;
	s->job.output_addr = (uint32_t)g->out.dma_address;
	s->job.weight_addr = (uint32_t)s->wt.dma_address;
	s->job.coef_addr = (uint32_t)s->coef.dma_address;

	/* gather the slice as unsigned bytes around a zero point of 128 */
	for (unsigned r = 0; r < n; r++) {
		const int8_t *src = t->q + (size_t)(n0 + r) * t->k + k0;
		uint8_t *dst = g->scratch + (size_t)r * k;

		for (unsigned c = 0; c < k; c++)
			dst[c] = (uint8_t)((int)src[c] + 128);
	}
	charsiu_bo_prep(g->dev, &s->wt, 1000000000);
	charsiu_pack_weights(&s->job.mm, g->scratch, s->wt.map);
	charsiu_bo_fini(g->dev, &s->wt);

	/* the weight sums this slice's K range accounts for, not the tensor's */
	bias = calloc(n, sizeof(*bias));
	wsum = calloc(n, sizeof(*wsum));
	if (!bias || !wsum)
		goto out;
	for (unsigned r = 0; r < n; r++) {
		const int8_t *src = t->q + (size_t)(n0 + r) * t->k + k0;
		int32_t acc = 0;

		for (unsigned c = 0; c < k; c++)
			acc += src[c];
		wsum[r] = acc;
	}

	/*
	 * Zero bias and no lift, so the accumulator arrives unmodified. The
	 * lift clears a fused ReLU in the REQUANT domain and acc_out bypasses
	 * that domain, so adding it here would only corrupt the sum.
	 */
	setenv("CHARSIU_NO_LIFT", "1", 1);
	charsiu_bo_prep(g->dev, &s->coef, 1000000000);
	charsiu_build_coefs(&s->job, bias, wsum, s->coef.map);
	charsiu_bo_fini(g->dev, &s->coef);
	unsetenv("CHARSIU_NO_LIFT");

	charsiu_bo_prep(g->dev, &s->regcmd, 1000000000);
	s->nreg = (unsigned)charsiu_emit_job(&s->job, s->regcmd.map, 4096 / 8);
	charsiu_bo_fini(g->dev, &s->regcmd);
	if (!s->nreg) {
		whine(g, "the register stream came back empty", k, n);
		goto out;
	}

	g->n_slot++;
	rc = 0;
out:
	free(bias);
	free(wsum);
	return rc;
}

int charsiu_npu_add(struct charsiu_npu *g, const struct npu_tensor *t)
{
	struct npu_entry *e;
	unsigned ns, ks, first = g->n_slot;

	if (g->dead) {
		whine(g, "the hardware path is already retired", (unsigned)t->k,
		      (unsigned)t->n);
		return -1;
	}
	if (g->n_ent == g->ent_cap) {
		whine(g, "no tensor slots left", (unsigned)t->k, (unsigned)t->n);
		return -1;
	}
	if (t->n > g->max_n) {
		whine(g, "wider than the device was opened for", (unsigned)t->k,
		      (unsigned)t->n);
		return -1;
	}

	ns = (unsigned)((t->n + g->nmax - 1) / g->nmax);
	ks = (unsigned)((t->k + g->kmax - 1) / g->kmax);
	if (first + ns * ks > g->slot_cap) {
		whine(g, "no slice slots left", (unsigned)t->k, (unsigned)t->n);
		return -1;
	}

	for (unsigned ki = 0; ki < ks; ki++) {
		unsigned k0 = ki * g->kmax;
		unsigned k = (unsigned)(t->k - k0) < g->kmax
			   ? (unsigned)(t->k - k0) : g->kmax;

		for (unsigned ni = 0; ni < ns; ni++) {
			unsigned n0 = ni * g->nmax;
			unsigned n = (unsigned)(t->n - n0) < g->nmax
				   ? (unsigned)(t->n - n0) : g->nmax;

			if (add_slice(g, t, n0, n, k0, k) < 0) {
				/* unwind: whatever was allocated is freed on close */
				g->n_slot = first;
				return -1;
			}
			g->slices++;
		}
	}

	e = &g->ent[g->n_ent];
	e->t = t;
	e->first = first;
	e->count = ns * ks;
	e->n_slices = ns;
	e->k_slices = ks;
	return (int)g->n_ent++;
}

int charsiu_npu_matvec(struct charsiu_npu *g, int id,
		       const struct charsiu_act *a, float *y)
{
	struct npu_entry *e;
	unsigned i;

	if (g->dead || id < 0 || (unsigned)id >= g->n_ent)
		return -1;
	e = &g->ent[id];
	if ((unsigned)a->n != e->t->k)
		return -1;

	memset(g->acc, 0, (size_t)e->t->n * sizeof(*g->acc));

	for (unsigned si = 0; si < e->count; si++) {
		struct npu_slot *s = &g->slot[e->first + si];
		uint32_t inh[3], outh;
		const int32_t *out;
		double t0;

		/*
		 * The activation slice. Repacked per K slice, and the N slices
		 * inside one K slice reuse it, which is why the loop in
		 * charsiu_npu_add puts K on the outside.
		 */
		if (si % e->n_slices == 0) {
			charsiu_bo_prep(g->dev, &g->in, 1000000000);
			for (i = 0; i < s->job.mm.k; i++)
				g->scratch[i] =
					(uint8_t)((int)a->q1[s->k0 + i] + 128);
			charsiu_pack_input(&s->job.mm, g->scratch, g->in.map,
					   g->in.size, s->job.input_zero_point);
			charsiu_bo_fini(g->dev, &g->in);
		}

		inh[0] = g->in.handle;
		inh[1] = s->wt.handle;
		inh[2] = s->coef.handle;
		outh = g->out.handle;

		t0 = now_us();
		if (charsiu_submit(g->dev, &s->regcmd, s->nreg, inh, 3, &outh, 1)) {
			g->strikes = 3;
		} else {
			charsiu_bo_prep(g->dev, &g->out, 2000000000);
			out = g->out.map;
			for (i = 0; i < s->job.mm.n; i++)
				g->acc[s->n0 + i] += out[i];
			charsiu_bo_fini(g->dev, &g->out);
			g->submits++;

			if (now_us() - t0 > g->slow_us)
				g->strikes++;
			else
				g->strikes = 0;
		}

		if (g->strikes >= 3) {
			g->dead = 1;
			fprintf(stderr,
				"charsiu: the NPU stopped answering at K=%u N=%u "
				"(slice %u of %u); everything from here runs on "
				"the CPU\n",
				s->job.mm.k, s->job.mm.n, si + 1, e->count);
			return -1;
		}
	}

	for (i = 0; i < (unsigned)e->t->n; i++)
		y[i] = (float)g->acc[i] * a->d1 * e->t->scale[i];
	return 0;
}
