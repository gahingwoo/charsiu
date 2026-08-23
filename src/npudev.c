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
	unsigned di;               /* which device its buffers live on */
};

struct npu_entry {
	const struct npu_tensor *t;
	struct charsiu_bo out;     /* ITS OWN, sized for ITS slices */
	unsigned first, count;     /* slots, n fastest */
	unsigned n_slices, k_slices;
	unsigned di;               /* which device, and so which CBUF window */
	double weight_mb;
};

struct charsiu_npu {
	/*
	 * TWO DEVICES, BECAUSE TWO OPEN FILES ARE WHAT REACH TWO CORES.
	 *
	 * rocket gives every open DRM file its own drm_sched_entity, built over
	 * the list of every core, and an entity runs one job at a time. One file
	 * therefore never uses more than one core no matter how deep the queue.
	 *
	 * The two cores share the CBUF, so the two devices also carry different
	 * CBUF windows -- round 363: six concurrent processes on split windows,
	 * every one 1024 of 1024 exact, against three of four corrupt when they
	 * share a window.
	 *
	 * CHARSIU_NPU_ONEDEV puts it back to a single device, which is the
	 * control for every number this buys.
	 */
	struct charsiu_device *dev[2];
	unsigned ndev;
	struct charsiu_bo in[2];      /* one per device: they cannot be shared */
	unsigned in_stride, out_stride, max_slices, maxtask;
	uint8_t *scratch;
	int32_t *acc;
	/*
	 * CHARSIU_NPU_W4V, the int4 decode path. Rounds 344 to 351 settled that
	 * this hardware computes a real int4 by fp16 dot product into float32
	 * once CORE 0x3018, 0x301c and 0x3020 carry the vendor's values, and
	 * that it is 1.42 to 1.78 times faster than int8 at every shape swept.
	 *
	 * It is OPT IN. The int8 path here produces tokens identical to the CPU
	 * at 6.55 tok/s and is not going to be replaced by a path that has never
	 * decoded a sentence.
	 *
	 * The activation goes in as the REAL float, not the int8 quantised one,
	 * so this mode has no input zero point, no wsum correction and no d1 in
	 * the dequantisation -- and it skips charsiu_act's quantisation
	 * entirely, which is 10.1 ms of a 153 ms token on the CPU side.
	 */
	int w4;
	/*
	 * CHARSIU_NPU_W4_MIDRISE: the vendor's grid, w = (s + 0.5) * d, with no
	 * code for zero. The hardware still computes sum(s * a), so the half
	 * step is 0.5 * d * sum(a) -- one number per K slice per token, shared
	 * by every output channel, which is why it is nearly free here.
	 */
	int midrise;
	double *asum;      /* per K slice, the sum of the activation */
	float *fscr;
	float *accf;
	uint8_t *wpack;
	double add_us, t_first;
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
	 * Wall clock actually spent in the hardware path, submit and read back
	 * together. Three tokens per second predictions in a row were wrong
	 * because a cost was assumed rather than measured, so the split between
	 * the NPU and the CPU stops being an inference here.
	 */
	double busy_us;
	/*
	 * And what that time is MADE of. bo_prep is not a read: it WAITS for the
	 * job, so the fence and the copy have to be told apart or the 23 ms this
	 * leaves over stays a residual rather than a measurement.
	 */
	double submit_us, fence_us, copy_us;

	/*
	 * A wedged block answers every submit with a driver side timeout and
	 * the ioctl still returns success, so the only reliable detector is the
	 * clock. Three slow submits and this path retires itself.
	 */
	double slow_us, min_gbs;
	int strikes, dead, whined, nochain, slowed;
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
/*
 * GROUPED SCALES FOR FREE, when the K slice IS the quantisation group.
 *
 * charsiu already cuts K into slices and sums their accumulators, so if each
 * slice covers exactly one group of the quantiser, the group's scale can be
 * applied to that slice's contribution on the way in and nothing extra has to
 * run on the hardware. Round 352's int4 sentence was English, on topic and
 * repetitive, which is what ONE absmax scale for a whole 2048 long row does to
 * four bits: measured offline, per channel RTN is 0.1067 relative error against
 * group 32's 0.0666.
 *
 * Set CHARSIU_NPU_KMAX and CHARSIU_NPU_W4_GROUP to the same value. The
 * condition is deliberately strict -- the slice must BE the group -- because a
 * slice covering part of a group would need a scale per part and there is
 * nowhere to put one.
 */
static int tensor_grouped(const struct charsiu_npu *g, const struct npu_tensor *t)
{
	return g->w4 && t->kgroup && t->kgroup < t->k &&
	       (t->k % t->kgroup) == 0 && t->kgroup == (uint64_t)g->kmax;
}

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
	g->dev[0] = charsiu_open("/dev/accel/accel0");
	g->ndev = 1;
	if (g->dev[0] && !getenv("CHARSIU_NPU_ONEDEV")) {
		g->dev[1] = charsiu_open("/dev/accel/accel0");
		if (g->dev[1])
			g->ndev = 2;
		else
			fprintf(stderr, "charsiu: only one NPU file could be "
				"opened; the second core stays idle\n");
	}
	if (!g->dev[0]) {
		free(g);
		return NULL;
	}
	/*
	 * The defaults are the widest slice MEASURED to give identical tokens,
	 * not the narrowest that was ever verified.
	 *
	 * Round 321 swept them and the cost turned out to be per TASK rather
	 * than per submit -- round 319 had varied tasks per submit at a fixed
	 * 606 slices and got a flat line, so the two sweeps together say the
	 * submit is nearly free and the task is about 35 us:
	 *
	 *   slices  606    542    319    287
	 *   tok/s   5.55   5.62   5.83   5.90
	 *
	 * Round 322 pushed it further and found the limit is on K and not on N:
	 *
	 *   slices  287    192    144    128
	 *   tok/s   5.90   6.08   6.09   0.55 <- K=8192, 131 job timeouts
	 *
	 * ⚠ N = 8192 DOES NOT WEDGE. That is the shape round 313 needed a power
	 * cycle to escape, so 313's hang was the requantised byte output or the
	 * 67 MB coefficient buffer, NOT the width. K = 8192 is the one that
	 * collapses, to 0.65 GB/s, and it collapses rather than hanging.
	 */
	g->nmax = env_u("CHARSIU_NPU_NMAX", 8192);
	g->kmax = env_u("CHARSIU_NPU_KMAX", 4096);
	g->slow_us = (double)env_u("CHARSIU_NPU_SLOW_US", 100000);
	g->nochain = getenv("CHARSIU_NPU_NOCHAIN") != NULL;
	/*
	 * 0 is unlimited. A cap exists because the output head is 126 chained
	 * tasks and 253 buffer handles in one submit, and it reached only
	 * 4.2 GB/s where an eight task submit reaches 10.
	 */
	g->maxtask = env_u("CHARSIU_NPU_MAXTASK", 0);
	/*
	 * ⚠ THE RETIREMENT GUARD WAS BLIND TO A THIRTEEN FOLD SLOWDOWN.
	 *
	 * Round 322's K = 8192 rung ran at 0.65 GB/s with 131 driver timeouts
	 * and 3718 IOMMU errors, and nothing printed: the budget is 100 ms plus
	 * a millisecond a megabyte, which is 1 GB/s, and 0.65 sat just under it
	 * while every submit stayed inside the flat allowance.
	 *
	 * Retiring the path on that would be wrong -- the answers were still
	 * correct, it was only slow -- so this is a separate, non fatal notice.
	 * "It stopped answering" and "it is thirteen times slower than it has
	 * ever been" are different things and a run should say which.
	 */
	g->min_gbs = (double)env_u("CHARSIU_NPU_MIN_MBPS", 2000) / 1000.0;
	g->max_n = max_n;
	g->ent_cap = max_tensors;

	g->w4 = getenv("CHARSIU_NPU_W4V") != NULL;
	g->midrise = g->w4 && getenv("CHARSIU_NPU_W4_MIDRISE") != NULL;
	ns = (max_n + g->nmax - 1) / g->nmax;
	ks = (max_k + g->kmax - 1) / g->kmax;
	g->max_slices = ns * ks;
	g->slot_cap = max_tensors * g->max_slices;

	{
		struct charsiu_matmul widest = { 1, g->kmax, g->nmax,
						 CHARSIU_INT8, CHARSIU_INT8 };

		g->in_stride = charsiu_entries_per_row(&widest) * 64;
		g->out_stride = g->nmax * 4;
		/* an fp16 activation is two bytes where an int8 one is one */
		if (g->w4)
			g->in_stride *= 2;
	}

	g->ent = calloc(g->ent_cap, sizeof(*g->ent));
	g->slot = calloc(g->slot_cap, sizeof(*g->slot));
	g->scratch = malloc((size_t)g->nmax * g->kmax + max_k);
	g->acc = calloc(max_n, sizeof(*g->acc));
	g->accf = calloc(max_n, sizeof(*g->accf));
	g->fscr = calloc(max_k ? max_k : 1, sizeof(*g->fscr));
	g->wpack = malloc((size_t)g->nmax * g->kmax + 4096);
	g->asum = calloc(ks ? ks : 1, sizeof(*g->asum));
	/* a GROUP can carry several tensors' slices, so four times over */
	g->tasks = calloc(4 * g->max_slices, sizeof(*g->tasks));
	g->handles = calloc(1 + 8 * g->max_slices, sizeof(*g->handles));
	if (!g->ent || !g->slot || !g->scratch || !g->acc || !g->accf ||
	    !g->fscr || !g->wpack || !g->asum || !g->tasks || !g->handles)
		goto fail;

	/* one activation buffer per device: a buffer object belongs to the file
	 * that made it, so the two cannot share one */
	for (unsigned d = 0; d < g->ndev; d++)
		if (charsiu_bo_alloc(g->dev[d],
				     (size_t)g->in_stride * ks + 4096, &g->in[d]))
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
	if (g->dev[0]) {
		for (unsigned i = 0; i < g->n_slot; i++) {
			unsigned d = g->slot[i].di;

			charsiu_bo_free(g->dev[d], &g->slot[i].wt);
			charsiu_bo_free(g->dev[d], &g->slot[i].coef);
			charsiu_bo_free(g->dev[d], &g->slot[i].regcmd);
		}
		for (unsigned i = 0; i < g->n_ent; i++)
			charsiu_bo_free(g->dev[g->ent[i].di], &g->ent[i].out);
		for (unsigned d = 0; d < g->ndev; d++) {
			charsiu_bo_free(g->dev[d], &g->in[d]);
			charsiu_close(g->dev[d]);
		}
	}
	free(g->slot);
	free(g->ent);
	free(g->scratch);
	free(g->acc);
	free(g->accf);
	free(g->fscr);
	free(g->wpack);
	free(g->asum);
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
	if (g->submits)
		fprintf(stderr,
			"charsiu NPU: %.0f ms in the hardware path, %.2f GB/s "
			"of weights, %.0f us a submit\n"
			"charsiu NPU: of that, %.0f ms submitting, %.0f ms "
			"waiting for the fence, %.0f ms reading back\n",
			g->busy_us / 1e3, g->weight_mb / g->busy_us * 1e3,
			g->busy_us / (double)g->submits,
			g->submit_us / 1e3, g->fence_us / 1e3, g->copy_us / 1e3);
	if (!g->submits)
		fprintf(stderr,
			"charsiu NPU: NOTHING RAN ON THE HARDWARE. Every number "
			"in this run came from the CPU.\n");
}

/* One slice: rows [n0, n0+n) and columns [k0, k0+k) of t, writing region si. */
static int add_slice(struct charsiu_npu *g, unsigned di,
		     const struct npu_tensor *t,
		     unsigned n0, unsigned n, unsigned k0, unsigned k,
		     unsigned ki, unsigned si, uint32_t out_base)
{
	struct npu_slot *s = &g->slot[g->n_slot];
	int32_t *bias = NULL, *wsum = NULL;
	int rc = -1;

	memset(s, 0, sizeof(*s));
	s->n0 = n0;
	s->k0 = k0;
	s->out_slot = si;
	s->di = di;
	/* the two cores share the CBUF, so the two devices take different
	 * windows -- see charsiu_job.cbuf_window */
	s->job.cbuf_window = di;
	s->job.mm.m = 1;
	s->job.mm.k = k;
	s->job.mm.n = n;
	s->job.mm.wdtype = g->w4 ? CHARSIU_INT4 : CHARSIU_INT8;
	s->job.mm.adtype = g->w4 ? CHARSIU_FP16 : CHARSIU_INT8;
	s->job.input_zero_point = 128;
	s->job.weight_zero_point = 128;
	s->job.output_zero_point = 0;
	s->job.input_scale = 1.0f;
	s->job.weight_scale = 1.0f;
	s->job.output_scale = 1.0f;
	s->job.acc_out = 1;

	if (charsiu_bo_alloc(g->dev[di], charsiu_weight_bytes(&s->job.mm) + 4096, &s->wt) ||
	    charsiu_bo_alloc(g->dev[di], charsiu_coef_bytes(&s->job.mm) + 4096, &s->coef) ||
	    charsiu_bo_alloc(g->dev[di], 4096, &s->regcmd)) {
		whine(g, "a buffer would not allocate", k, n);
		goto out;
	}

	/* its own slot in each shared buffer, baked into the stream */
	s->job.input_addr = (uint32_t)g->in[di].dma_address + ki * g->in_stride;
	s->job.output_addr = out_base + si * g->out_stride;
	s->job.weight_addr = (uint32_t)s->wt.dma_address;
	s->job.coef_addr = (uint32_t)s->coef.dma_address;

	/*
	 * int8 wants unsigned bytes around a zero point of 128; int4 wants the
	 * signed code in the low nibble, which is what two's complement already
	 * puts there for a value in [-8, 7].
	 */
	for (unsigned r = 0; r < n; r++) {
		const int8_t *src = t->q + (size_t)(n0 + r) * t->k + k0;
		uint8_t *dst = g->scratch + (size_t)r * k;

		for (unsigned c = 0; c < k; c++)
			dst[c] = g->w4 ? (uint8_t)(src[c] & 0xf)
				       : (uint8_t)((int)src[c] + 128);
	}
	/*
	 * ⚠ PACK INTO ORDINARY MEMORY AND THEN COPY, because the int4 layout
	 * writes STRIDED into the buffer -- sixteen consecutive bytes, then a
	 * jump of 256 -- and a buffer object's mapping does not absorb that the
	 * way a sequential write is absorbed. Round 352 spent 101 SECONDS
	 * staging 113 tensors this way against int8's 303 ms, at a steady
	 * second a tensor, and int8 only escapes it because its layout is
	 * nearly sequential. The copy afterwards is one sequential pass.
	 */
	memset(g->wpack, 0, charsiu_weight_bytes(&s->job.mm));
	charsiu_pack_weights(&s->job.mm, g->scratch, g->wpack);
	charsiu_bo_prep(g->dev[di], &s->wt, 1000000000);
	memcpy(s->wt.map, g->wpack, charsiu_weight_bytes(&s->job.mm));
	charsiu_bo_fini(g->dev[di], &s->wt);

	bias = calloc(n, sizeof(*bias));
	wsum = calloc(n, sizeof(*wsum));
	if (!bias || !wsum)
		goto out;
	/* the weight sums this slice's K range accounts for, not the tensor's.
	 * int4 has no input zero point, so there is nothing for them to
	 * correct and they stay at zero. */
	for (unsigned r = 0; r < n && !g->w4; r++) {
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
	charsiu_bo_prep(g->dev[di], &s->coef, 1000000000);
	charsiu_build_coefs(&s->job, bias, wsum, s->coef.map);
	charsiu_bo_fini(g->dev[di], &s->coef);
	unsetenv("CHARSIU_NO_LIFT");

	charsiu_bo_prep(g->dev[di], &s->regcmd, 1000000000);
	s->nreg = (unsigned)charsiu_emit_job(&s->job, s->regcmd.map, 4096 / 8);
	charsiu_bo_fini(g->dev[di], &s->regcmd);
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
	double t_add = now_us();
	struct npu_entry *e;

	/* ⚠ start the clock on the FIRST tensor, not the first heartbeat, or
	 * the first sixteen are free and round 353's log said "0 ms". */
	if (g->t_first == 0.0)
		g->t_first = t_add;
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

	e = &g->ent[g->n_ent];
	memset(e, 0, sizeof(*e));
	/*
	 * Alternate. A group is q, k, v or gate, up -- consecutive adds -- so
	 * alternating puts a group's members on different devices, which is
	 * exactly where the overlap has to come from.
	 */
	e->di = g->ndev > 1 ? (g->n_ent & 1) : 0;
	/*
	 * ⚠ ITS OWN OUTPUT BUFFER, AND THIS IS NOT TIDINESS.
	 *
	 * One shared buffer had to be sized for the WIDEST tensor, and round
	 * 318 added the 128256 wide output head, which took it from 128 KB to
	 * 2.48 MB. charsiu_bo_prep and _fini are cache maintenance over a WHOLE
	 * buffer object, and every one of the 113 matvecs a token paid it: 280
	 * MB of cache operations a token where there had been 14. That is most
	 * of why routing the head made the model 18% SLOWER.
	 */
	if (charsiu_bo_alloc(g->dev[e->di], (size_t)ns * ks * g->out_stride + 4096,
			     &e->out)) {
		whine(g, "an output buffer would not allocate", (unsigned)t->k,
		      (unsigned)t->n);
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

			if (add_slice(g, e->di, t, n0, n, k0, k, ki, si,
				      (uint32_t)e->out.dma_address) < 0) {
				g->n_slot = first;
				return -1;
			}
			g->slices++;
		}
	}

	e->t = t;
	e->first = first;
	e->count = ns * ks;
	e->n_slices = ns;
	e->k_slices = ks;
	/*
	 * ⚠ BYTES, NOT ELEMENTS. This counted n*k for both precisions, so every
	 * "GB/s of weights" this project has printed for int4 was DOUBLE the
	 * real figure -- 13.4 GB/s in round 356's log is 6.7. int8 was right by
	 * accident, one byte an element. The honest comparison is int4 at 6.7
	 * GB/s against int8's 9.46, which is what it looked like from the
	 * outside and what the shape sweep said.
	 */
	e->weight_mb = (double)t->n * (double)t->k
		     / (g->w4 ? 2.0 : 1.0) / 1e6;
	/*
	 * A HEARTBEAT WHILE THE WEIGHTS ARE STAGED. Round 352's int4 arm printed
	 * nothing for minutes and there was no way to tell a slow load from a
	 * wedge: charsiu_run's own output does not appear until the generation
	 * is done. Four lines for a 113 tensor model is not noise.
	 */
	/*
	 * ⚠ THE HEARTBEAT SPLITS THE TIME NOW. Round 353 showed int4 staging at
	 * 102 s against int8's 16 s -- SIX times, not the three hundred I first
	 * read, because int8's own staging is 16 s and its "load 345 ms" line is
	 * only the gguf mmap. Packing into ordinary memory and copying did NOT
	 * move it, so the strided write to the buffer object was not the cause
	 * and I have no second guess. This measures instead: g->add_us is time
	 * inside charsiu_npu_add, and whatever is left of the wall clock between
	 * heartbeats belongs to npu_tensor_build, which is the quantiser.
	 */
	g->add_us += now_us() - t_add;
	if ((g->n_ent % 16) == 15) {
		fprintf(stderr,
			"charsiu NPU: %u tensors staged, %.0f ms of which "
			"%.0f ms adding and %.0f ms quantising\n",
			g->n_ent + 1, (now_us() - g->t_first) / 1000.0,
			g->add_us / 1000.0,
			((now_us() - g->t_first) - g->add_us) / 1000.0);
	}
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
	charsiu_bo_prep(g->dev[e->di], &g->in[e->di], 1000000000);
	for (unsigned ki = 0; ki < e->k_slices; ki++) {
		const struct npu_slot *s = &g->slot[e->first + ki * e->n_slices];

		if (g->w4) {
			double as = 0.0;

			for (i = 0; i < s->job.mm.k; i++) {
				g->fscr[i] = a->f[s->k0 + i];
				as += (double)g->fscr[i];
			}
			/* the half step of the midrise grid rides on this */
			g->asum[ki] = as;
			charsiu_pack_input_f16(&s->job.mm, g->fscr,
					       (uint8_t *)g->in[e->di].map
					       + ki * g->in_stride,
					       g->in_stride);
		} else {
			for (i = 0; i < s->job.mm.k; i++)
				g->scratch[i] = (uint8_t)((int)a->q1[s->k0 + i]
							  + 128);
			charsiu_pack_input(&s->job.mm, g->scratch,
					   (uint8_t *)g->in[e->di].map
					   + ki * g->in_stride,
					   g->in_stride,
					   s->job.input_zero_point);
		}
	}
	charsiu_bo_fini(g->dev[e->di], &g->in[e->di]);

	/*
	 * ONE submit for the whole projection, unless a cap says otherwise.
	 *
	 * CHARSIU_NPU_NOCHAIN puts it back to a submit per slice, so a round can
	 * carry its own before and after in one boot. CHARSIU_NPU_MAXTASK caps
	 * the tasks per submit, which exists to test a specific suspicion: the
	 * output head is 126 chained tasks and 253 buffer handles in one submit
	 * and reached 4.2 GB/s, where an eight task submit reaches 10, so the
	 * driver's per handle work is a candidate for the difference.
	 */
	{
		unsigned per = g->nochain ? 1
			     : (g->maxtask && g->maxtask < e->count
				? g->maxtask : e->count);

		t0 = now_us();
		for (unsigned base = 0; base < e->count; base += per) {
			unsigned nt = e->count - base < per ? e->count - base : per;

			nh = 0;
			g->handles[nh++] = g->in[e->di].handle;
			for (i = 0; i < nt; i++) {
				const struct npu_slot *s = &g->slot[e->first + base + i];

				g->tasks[base + i].regcmd =
					(uint32_t)s->regcmd.dma_address;
				g->tasks[base + i].regcmd_count = s->nreg;
				g->handles[nh++] = s->wt.handle;
				g->handles[nh++] = s->coef.handle;
			}
			jl.tasks = &g->tasks[base];
			jl.task_count = nt;
			jl.in_handles = g->handles;
			jl.in_count = nh;
			jl.out_handles = &e->out.handle;
			jl.out_count = 1;

			if (charsiu_submit_jobs(g->dev[e->di], &jl, 1)) {
				g->strikes = 3;
				break;
			}
			g->submits++;
		}
		g->submit_us += now_us() - t0;
	}

	if (g->strikes < 3) {
		double t1 = now_us();

		charsiu_bo_prep(g->dev[e->di], &e->out, 2000000000);
		g->fence_us += now_us() - t1;
		t1 = now_us();
		int grp = tensor_grouped(g, e->t);

		memset(g->acc, 0, (size_t)e->t->n * sizeof(*g->acc));
		memset(g->accf, 0, (size_t)e->t->n * sizeof(*g->accf));
		for (i = 0; i < e->count; i++) {
			const struct npu_slot *s = &g->slot[e->first + i];
			const uint8_t *base = (const uint8_t *)e->out.map +
					      s->out_slot * g->out_stride;

			/* int4 writes float32, int8 the raw int32 accumulator */
			if (g->w4) {
				const float *fo = (const float *)base;

				if (grp) {
					uint64_t ng = e->t->k / e->t->kgroup;
					uint64_t gi = s->k0 / e->t->kgroup;
					double hs = g->midrise
						  ? 0.5 * g->asum[s->k0 / g->kmax]
						  : 0.0;

					for (unsigned j = 0; j < s->job.mm.n; j++)
						g->accf[s->n0 + j] +=
						  (float)((fo[j] + hs) *
						   e->t->scale[(s->n0 + j) * ng
							       + gi]);
				} else {
					for (unsigned j = 0; j < s->job.mm.n; j++)
						g->accf[s->n0 + j] += fo[j];
				}
				continue;
			}
			out = (const int32_t *)base;
			for (unsigned j = 0; j < s->job.mm.n; j++)
				g->acc[s->n0 + j] += out[j];
		}
		charsiu_bo_fini(g->dev[e->di], &e->out);
		g->copy_us += now_us() - t1;
		g->weight_mb += e->weight_mb;
		g->busy_us += now_us() - t0;

		/*
		 * The limit scales with what the submit fetches, or a legitimate
		 * big one is mistaken for a wedge. A millisecond a megabyte is
		 * ten times slower than this hardware has ever been.
		 */
		{
			double took = now_us() - t0;
			double gbs = e->weight_mb / took * 1e3;

			if (took > g->slow_us * (g->nochain ? e->count : 1)
				  + e->weight_mb * 1000.0)
				g->strikes++;
			else
				g->strikes = 0;

			/*
			 * ⚠ NOT ON A WARM UP. Round 323's first call to a
			 * tensor came in at 1.25 GB/s while the run averaged
			 * 9.35, so the notice fired on a cold buffer and said
			 * nothing true about the run. A warning that cries wolf
			 * on every boot is worse than none.
			 */
			if (!g->slowed && gbs < g->min_gbs &&
			    g->submits > g->n_ent * 2) {
				g->slowed = 1;
				fprintf(stderr,
					"charsiu: the NPU is SLOW, %.2f GB/s at "
					"K=%u N=%u -- correct but degraded, and "
					"nothing is being retired\n",
					gbs, (unsigned)e->t->k,
					(unsigned)e->t->n);
			}
		}
	}

	if (g->strikes >= 3) {
		g->dead = 1;
		fprintf(stderr,
			"charsiu: the NPU stopped answering on a %u task submit "
			"(K=%u N=%u); everything from here runs on the CPU\n",
			e->count, (unsigned)e->t->k, (unsigned)e->t->n);
		return -1;
	}

	/*
	 * int4 took the REAL activation, so there is no d1 to undo: the block
	 * returns sum_k code(n,k) * a(k) in float and only the weight scale is
	 * left. int8 took a->q1 and needs both.
	 */
	{
		int grp = tensor_grouped(g, e->t);
		double hs = 0.0;

		if (g->midrise && !grp)
			for (unsigned ki = 0; ki < e->k_slices; ki++)
				hs += 0.5 * g->asum[ki];
		for (i = 0; i < (unsigned)e->t->n; i++)
			y[i] = g->w4
			     ? (grp ? g->accf[i]
				    : (float)(((double)g->accf[i] + hs)
					      * e->t->scale[i]))
			     : (float)g->acc[i] * a->d1 * e->t->scale[i];
	}
	return 0;
}

/*
 * SEVERAL PROJECTIONS, ONE SUBMIT AND ONE FENCE.
 *
 * q, k and v all multiply the SAME RMSNorm output, and so do gate and up. They
 * are independent of each other, so there is no reason to wait for one before
 * starting the next -- and round 321 measured the fence at 94% of the hardware
 * path, so a fence removed is worth more than a submit removed.
 *
 * 113 fences a token becomes 65. Whether that is worth anything is what the
 * round measures; the arithmetic is unchanged either way, so the tokens must
 * stay identical.
 */
int charsiu_npu_matvec_group(struct charsiu_npu *g, const int *ids, unsigned n,
			     const struct charsiu_act *a, float **ys)
{
	struct npu_entry *e0;
	struct charsiu_joblist jl;
	uint32_t outh[8];
	unsigned nh = 0, ntask = 0, i, j;
	double t0, t1;

	if (g->dead || !n || n > 8)
		return -1;
	for (i = 0; i < n; i++)
		if (ids[i] < 0 || (unsigned)ids[i] >= g->n_ent)
			return -1;
	e0 = &g->ent[ids[0]];
	for (i = 1; i < n; i++)
		if (g->ent[ids[i]].t->k != e0->t->k)
			return -1;       /* a group shares one activation */
	if ((unsigned)a->n != e0->t->k)
		return -1;

	/*
	 * The activation, once for every K slice, INTO EVERY DEVICE THE GROUP
	 * USES. A group shares one input vector but its entries may sit on
	 * different devices, and a buffer object belongs to the file that
	 * created it. Packing it twice is a few kilobytes against the megabytes
	 * of weights each submit fetches.
	 */
	{
	unsigned nd = 0, dmask = 0;

	for (i = 0; i < n; i++)
		dmask |= 1u << g->ent[ids[i]].di;
	for (unsigned d = 0; d < g->ndev; d++) {
		if (!(dmask & (1u << d)))
			continue;
		nd++;
	charsiu_bo_prep(g->dev[d], &g->in[d], 1000000000);
	for (unsigned ki = 0; ki < e0->k_slices; ki++) {
		const struct npu_slot *s = &g->slot[e0->first + ki * e0->n_slices];

		if (g->w4) {
			double as = 0.0;

			for (i = 0; i < s->job.mm.k; i++) {
				g->fscr[i] = a->f[s->k0 + i];
				as += (double)g->fscr[i];
			}
			/* the half step of the midrise grid rides on this */
			g->asum[ki] = as;
			charsiu_pack_input_f16(&s->job.mm, g->fscr,
					       (uint8_t *)g->in[d].map
					       + ki * g->in_stride,
					       g->in_stride);
		} else {
			for (i = 0; i < s->job.mm.k; i++)
				g->scratch[i] = (uint8_t)((int)a->q1[s->k0 + i]
							  + 128);
			charsiu_pack_input(&s->job.mm, g->scratch,
					   (uint8_t *)g->in[d].map
					   + ki * g->in_stride,
					   g->in_stride,
					   s->job.input_zero_point);
		}
	}
	charsiu_bo_fini(g->dev[d], &g->in[d]);
	}
	(void)nd;
	}

	/*
	 * ONE JOBLIST PER DEVICE, BOTH SUBMITTED BEFORE EITHER IS WAITED ON.
	 *
	 * This is the whole point of the second file. Submitting is a queueing
	 * ioctl and the fence is waited separately, so issuing device 0's work
	 * and then device 1's leaves both cores running at once without a
	 * thread anywhere.
	 *
	 * Round 356 put 89 ms of a 117 ms token inside the fence, so this is
	 * the only place in the decode where a second core can be worth
	 * anything.
	 */
	t0 = now_us();
	for (unsigned d = 0; d < g->ndev; d++) {
		unsigned no = 0;

		nh = 0;
		ntask = 0;
		g->handles[nh++] = g->in[d].handle;
		for (i = 0; i < n; i++) {
			struct npu_entry *e = &g->ent[ids[i]];

			if (e->di != d)
				continue;
			outh[no++] = e->out.handle;
			for (j = 0; j < e->count; j++) {
				const struct npu_slot *s =
					&g->slot[e->first + j];

				if (ntask >= 4 * g->max_slices)
					return -1;
				g->tasks[ntask].regcmd =
					(uint32_t)s->regcmd.dma_address;
				g->tasks[ntask].regcmd_count = s->nreg;
				ntask++;
				g->handles[nh++] = s->wt.handle;
				g->handles[nh++] = s->coef.handle;
			}
		}
		if (!no)
			continue;
		jl.tasks = g->tasks;
		jl.task_count = ntask;
		jl.in_handles = g->handles;
		jl.in_count = nh;
		jl.out_handles = outh;
		jl.out_count = no;
		if (charsiu_submit_jobs(g->dev[d], &jl, 1)) {
			g->strikes = 3;
			g->dead = 1;
			fprintf(stderr, "charsiu: a %u task group submit "
				"failed on device %u\n", ntask, d);
			return -1;
		}
		g->submits++;
	}
	g->submit_us += now_us() - t0;

	t1 = now_us();
	for (i = 0; i < n; i++)
		charsiu_bo_prep(g->dev[g->ent[ids[i]].di],
				&g->ent[ids[i]].out, 2000000000);
	g->fence_us += now_us() - t1;

	t1 = now_us();
	for (i = 0; i < n; i++) {
		struct npu_entry *e = &g->ent[ids[i]];
		const int32_t *out;

		int grp = tensor_grouped(g, e->t);

		memset(g->acc, 0, (size_t)e->t->n * sizeof(*g->acc));
		memset(g->accf, 0, (size_t)e->t->n * sizeof(*g->accf));
		for (j = 0; j < e->count; j++) {
			const struct npu_slot *s = &g->slot[e->first + j];
			const uint8_t *base = (const uint8_t *)e->out.map +
					      s->out_slot * g->out_stride;

			if (g->w4) {
				const float *fo = (const float *)base;

				if (grp) {
					uint64_t ng = e->t->k / e->t->kgroup;
					uint64_t gi = s->k0 / e->t->kgroup;
					double hs = g->midrise
						  ? 0.5 * g->asum[s->k0 / g->kmax]
						  : 0.0;

					for (unsigned q = 0; q < s->job.mm.n; q++)
						g->accf[s->n0 + q] +=
						  (float)((fo[q] + hs) *
						   e->t->scale[(s->n0 + q) * ng
							       + gi]);
				} else {
					for (unsigned q = 0; q < s->job.mm.n; q++)
						g->accf[s->n0 + q] += fo[q];
				}
				continue;
			}
			out = (const int32_t *)base;
			for (unsigned q = 0; q < s->job.mm.n; q++)
				g->acc[s->n0 + q] += out[q];
		}
		charsiu_bo_fini(g->dev[e->di], &e->out);
		{
			double hsu = 0.0;

			if (g->midrise && !grp)
				for (unsigned ki = 0; ki < e->k_slices; ki++)
					hsu += 0.5 * g->asum[ki];
			for (unsigned q = 0; q < (unsigned)e->t->n; q++)
				ys[i][q] = g->w4
					 ? (grp ? g->accf[q]
					        : (float)(((double)g->accf[q]
						   + hsu) * e->t->scale[q]))
					 : (float)g->acc[q] * a->d1
					   * e->t->scale[q];
		}
		g->weight_mb += e->weight_mb;
	}
	g->copy_us += now_us() - t1;
	g->busy_us += now_us() - t0;
	return 0;
}
