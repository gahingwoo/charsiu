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
 * measured at was N = 1024, K = 2048. So a projection is cut into slices of a
 * shape that HAS been measured. acc_out is what makes that exact: a raw int32
 * sum means partial sums over a split K add with no rounding at all.
 *
 * ⚠ AND THE SLICES OF ONE PROJECTION GO IN ONE SUBMIT. Round 316 ran all 112
 * projections correctly at 480 UNCHAINED submits a token, which is about 91 ms
 * of fixed cost and left the NPU slower than the CPU. Chaining is what the
 * measurement has been pointing at since round 165: break even is 2.2 MB of
 * weights a submit, and a whole ffn_gate is 16 MB.
 *
 * That needs every slice to write somewhere different, so each one gets its own
 * region of the output buffer and its own region of the input buffer, baked
 * into its register stream when it is built.
 */
struct npu_slot {
	struct charsiu_bo wt, coef, regcmd;
	struct charsiu_job job;
	unsigned nreg;
	unsigned n0, k0;           /* where this slice starts in the tensor */
	unsigned out_slot;         /* which output region it writes */
};

struct npu_entry {
	const struct npu_tensor *t;
	unsigned first, count;     /* slots, n fastest */
	unsigned n_slices, k_slices;
	double weight_mb;          /* what one submit fetches */
};

struct charsiu_npu {
	struct charsiu_device *dev;
	struct charsiu_bo in, out;
	unsigned in_stride, out_stride, max_slices;
	uint8_t *scratch;
	int32_t *acc;
	struct charsiu_task *tasks;
	uint32_t *handles;
	unsigned nmax, kmax, max_n;
	struct npu_slot *slot;
	unsigned n_slot, slot_cap;
	struct npu_entry *ent;
	unsigned n_ent, ent_cap;
	unsigned long submits;
	double weight_mb;          /* summed over submits, for the report */

	/*
	 * A wedged block answers every submit with a driver side timeout and
	 * the ioctl still returns success, so the only reliable detector is the
	 * clock. Three slow submits and this path retires itself.
	 */
	double slow_us;
	int strikes, dead, whined, nochain;
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

/*
 * Say why, out loud, the first time.
 *
 * Round 315 ran a whole ladder in which the hardware never engaged, and the
 * text was right every time because the CPU quietly did the work. A silent
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

struct charsiu_npu *charsiu_npu_open(unsigned max_k, unsigned max_n,
				     unsigned max_tensors)
{
	struct charsiu_npu *g = calloc(1, sizeof(*g));
	unsigned ns, ks;

	if (!g)
		return NULL;
	g->dev = charsiu_open("/dev/accel/accel0");
	if (!g->dev) {
		free(g);
		return NULL;
	}
	/* the shapes round 312 measured exact, not the shapes a model wants */
	g->nmax = env_u("CHARSIU_NPU_NMAX", 1024);
	g->kmax = env_u("CHARSIU_NPU_KMAX", 2048);
	g->slow_us = (double)env_u("CHARSIU_NPU_SLOW_US", 100000);
	g->nochain = getenv("CHARSIU_NPU_NOCHAIN") != NULL;
	g->max_n = max_n;
	g->ent_cap = max_tensors;

	ns = (max_n + g->nmax - 1) / g->nmax;
	ks = (max_k + g->kmax - 1) / g->kmax;
	g->max_slices = ns * ks;
	g->slot_cap = max_tensors * g->max_slices;

	{
		struct charsiu_matmul widest = { 1, g->kmax, g->nmax,
						 CHARSIU_INT8, CHARSIU_INT8 };

		g->in_stride = charsiu_entries_per_row(&widest) * 64;
		g->out_stride = g->nmax * 4;
	}

	g->ent = calloc(g->ent_cap, sizeof(*g->ent));
	g->slot = calloc(g->slot_cap, sizeof(*g->slot));
	g->scratch = malloc((size_t)g->nmax * g->kmax + max_k);
	g->acc = calloc(max_n, sizeof(*g->acc));
	g->tasks = calloc(g->max_slices, sizeof(*g->tasks));
	/* input + one weight and one coefficient buffer per chained task */
	g->handles = calloc(1 + 2 * g->max_slices, sizeof(*g->handles));
	if (!g->ent || !g->slot || !g->scratch || !g->acc || !g->tasks ||
	    !g->handles)
		goto fail;

	if (charsiu_bo_alloc(g->dev, (size_t)g->in_stride * ks + 4096, &g->in) ||
	    charsiu_bo_alloc(g->dev, (size_t)g->out_stride * g->max_slices + 4096,
			     &g->out))
		goto fail;
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
	free(g->tasks);
	free(g->handles);
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
 *
 * The megabytes per submit are here because that is the number the whole
 * chaining question turns on: break even against the fixed submit cost is 2.2.
 */
void charsiu_npu_report(const struct charsiu_npu *g)
{
	if (!g)
		return;
	fprintf(stderr,
		"charsiu NPU: %u tensors, %lu slices, %lu submits, %.2f MB per "
		"submit%s\n",
		g->n_ent, g->slices, g->submits,
		g->submits ? g->weight_mb / (double)g->submits : 0.0,
		g->dead ? "  (RETIRED: it stopped answering)" : "");
	if (!g->submits)
		fprintf(stderr,
			"charsiu NPU: NOTHING RAN ON THE HARDWARE. Every number "
			"in this run came from the CPU.\n");
}

/* One slice: rows [n0, n0+n) and columns [k0, k0+k) of t, writing region si. */
static int add_slice(struct charsiu_npu *g, const struct npu_tensor *t,
		     unsigned n0, unsigned n, unsigned k0, unsigned k,
		     unsigned ki, unsigned si)
{
	struct npu_slot *s = &g->slot[g->n_slot];
	int32_t *bias = NULL, *wsum = NULL;
	int rc = -1;

	memset(s, 0, sizeof(*s));
	s->n0 = n0;
	s->k0 = k0;
	s->out_slot = si;
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

	/* its own slot in each shared buffer, baked into the stream */
	s->job.input_addr = (uint32_t)g->in.dma_address + ki * g->in_stride;
	s->job.output_addr = (uint32_t)g->out.dma_address + si * g->out_stride;
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

	bias = calloc(n, sizeof(*bias));
	wsum = calloc(n, sizeof(*wsum));
	if (!bias || !wsum)
		goto out;
	/* the weight sums this slice's K range accounts for, not the tensor's */
	for (unsigned r = 0; r < n; r++) {
		const int8_t *src = t->q + (size_t)(n0 + r) * t->k + k0;
		int32_t a = 0;

		for (unsigned c = 0; c < k; c++)
			a += src[c];
		wsum[r] = a;
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
	unsigned ns, ks, first = g->n_slot, si = 0;

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
	if (ns * ks > g->max_slices || first + ns * ks > g->slot_cap) {
		whine(g, "no slice slots left", (unsigned)t->k, (unsigned)t->n);
		return -1;
	}

	for (unsigned ki = 0; ki < ks; ki++) {
		unsigned k0 = ki * g->kmax;
		unsigned k = (unsigned)(t->k - k0) < g->kmax
			   ? (unsigned)(t->k - k0) : g->kmax;

		for (unsigned ni = 0; ni < ns; ni++, si++) {
			unsigned n0 = ni * g->nmax;
			unsigned n = (unsigned)(t->n - n0) < g->nmax
				   ? (unsigned)(t->n - n0) : g->nmax;

			if (add_slice(g, t, n0, n, k0, k, ki, si) < 0) {
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
	e->weight_mb = (double)t->n * (double)t->k / 1e6;
	return (int)g->n_ent++;
}

int charsiu_npu_matvec(struct charsiu_npu *g, int id,
		       const struct charsiu_act *a, float *y)
{
	struct npu_entry *e;
	struct charsiu_joblist jl;
	const int32_t *out;
	unsigned nh = 0, i;
	double t0;

	if (g->dead || id < 0 || (unsigned)id >= g->n_ent)
		return -1;
	e = &g->ent[id];
	if ((unsigned)a->n != e->t->k) {
		whine(g, "the activation is not this tensor's K", (unsigned)a->n,
		      (unsigned)e->t->n);
		return -1;
	}

	/* every K slice's activation, each in its own region */
	charsiu_bo_prep(g->dev, &g->in, 1000000000);
	for (unsigned ki = 0; ki < e->k_slices; ki++) {
		const struct npu_slot *s = &g->slot[e->first + ki * e->n_slices];

		for (i = 0; i < s->job.mm.k; i++)
			g->scratch[i] = (uint8_t)((int)a->q1[s->k0 + i] + 128);
		charsiu_pack_input(&s->job.mm, g->scratch,
				   (uint8_t *)g->in.map + ki * g->in_stride,
				   g->in_stride, s->job.input_zero_point);
	}
	charsiu_bo_fini(g->dev, &g->in);

	/*
	 * ONE submit for the whole projection.
	 *
	 * CHARSIU_NPU_NOCHAIN puts it back to a submit per slice, so a round can
	 * carry its own before and after in one boot rather than comparing
	 * against a different one.
	 */
	g->handles[nh++] = g->in.handle;
	for (i = 0; i < e->count; i++) {
		const struct npu_slot *s = &g->slot[e->first + i];

		g->tasks[i].regcmd = (uint32_t)s->regcmd.dma_address;
		g->tasks[i].regcmd_count = s->nreg;
		g->handles[nh++] = s->wt.handle;
		g->handles[nh++] = s->coef.handle;
	}
	jl.tasks = g->tasks;
	jl.task_count = e->count;
	jl.in_handles = g->handles;
	jl.in_count = nh;
	jl.out_handles = &g->out.handle;
	jl.out_count = 1;

	t0 = now_us();
	if (g->nochain) {
		for (i = 0; i < e->count; i++) {
			struct charsiu_joblist one;
			uint32_t h[3];

			h[0] = g->in.handle;
			h[1] = g->slot[e->first + i].wt.handle;
			h[2] = g->slot[e->first + i].coef.handle;
			one.tasks = &g->tasks[i];
			one.task_count = 1;
			one.in_handles = h;
			one.in_count = 3;
			one.out_handles = &g->out.handle;
			one.out_count = 1;
			if (charsiu_submit_jobs(g->dev, &one, 1)) {
				g->strikes = 3;
				break;
			}
			g->submits++;
		}
	} else if (charsiu_submit_jobs(g->dev, &jl, 1)) {
		g->strikes = 3;
	} else {
		g->submits++;
	}

	if (g->strikes < 3) {
		charsiu_bo_prep(g->dev, &g->out, 2000000000);
		memset(g->acc, 0, (size_t)e->t->n * sizeof(*g->acc));
		for (i = 0; i < e->count; i++) {
			const struct npu_slot *s = &g->slot[e->first + i];

			out = (const int32_t *)((const uint8_t *)g->out.map +
						s->out_slot * g->out_stride);
			for (unsigned j = 0; j < s->job.mm.n; j++)
				g->acc[s->n0 + j] += out[j];
		}
		charsiu_bo_fini(g->dev, &g->out);
		g->weight_mb += e->weight_mb;

		if (now_us() - t0 > g->slow_us * (g->nochain ? e->count : 1))
			g->strikes++;
		else
			g->strikes = 0;
	}

	if (g->strikes >= 3) {
		g->dead = 1;
		fprintf(stderr,
			"charsiu: the NPU stopped answering on a %u task submit "
			"(K=%u N=%u); everything from here runs on the CPU\n",
			e->count, (unsigned)e->t->k, (unsigned)e->t->n);
		return -1;
	}

	for (i = 0; i < (unsigned)e->t->n; i++)
		y[i] = (float)g->acc[i] * a->d1 * e->t->scale[i];
	return 0;
}
