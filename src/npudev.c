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
	/*
	 * THIS SLICE'S GROUP SCALES, ONE PER CHANNEL, LAID OUT AS THE SUM READS
	 * THEM.
	 *
	 * A grouped tensor keeps its scales as scale[channel * ngroups + group],
	 * so the read back walked them with a stride of ngroups: eight cache
	 * lines apart on the down projection, and unvectorisable everywhere.
	 * The values are the same values, gathered once when the tensor is
	 * staged rather than once a token for the life of the process.
	 *
	 * It costs what the scale array itself costs, 4.8 MB for this model,
	 * against 620 MB of weights.
	 */
	float *sc;
};

struct npu_entry {
	const struct npu_tensor *t;
	struct charsiu_bo out[2];  /* ITS OWN, one per device */
	unsigned first, count;     /* slots, n fastest */
	unsigned n_slices, k_slices;
	double weight_mb;
	/*
	 * THE ROWS THE CPU KEEPS, IF ANY.
	 *
	 * Round 370 measured what the memory controller has spare: the NPU
	 * alone pulls 10.46 GB/s, and with four threads reading DRAM beside it
	 * the pair reach 15.46, so about 5 GB/s is going unused. Decode is a
	 * dependency chain, so the only way to spend that is to give a second
	 * engine part of the SAME tensor -- and the cheapest second engine is
	 * the CPU, which is already here and is already BLOCKED in prep_bo for
	 * most of the fence.
	 *
	 * cq holds those rows' weights packed two to a byte, row major. It has
	 * to be packed: t->q keeps one BYTE per int4 weight, so reading rows
	 * out of it costs twice the bytes the NPU pays for the same weights,
	 * and the whole idea is bandwidth.
	 */
	unsigned n_npu;            /* rows [0, n_npu) go to the hardware */
	uint8_t *cq;               /* rows [n_npu, n), nibble packed */
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
	 *
	 * ⚠ AND THE FENCE NUMBER IS NOT PURE WAITING. rocket_ioctl_prep_bo is a
	 * dma_resv_wait_timeout FOLLOWED BY dma_sync_sgtable_for_cpu, so the
	 * invalidate over the whole output buffer is charged to the fence, not
	 * to the read back. fini_bo is the other half of that pair, a
	 * dma_sync_sgtable_for_device, and it IS separable -- so it is separate
	 * here, because "13 ms reading back" and "13 ms cleaning a cache the CPU
	 * only read" ask for different fixes.
	 */
	double submit_us, fence_us, copy_us, fini_us;
	/*
	 * ⚠ ON TOP OF busy_us, NOT INSIDE IT. Packing the activation happens
	 * before the timer that covers a submit, so round 368 left 10.6 ms a
	 * token between what the stage table charges to a projection and what
	 * this file measures inside one. This is the missing piece, measured
	 * rather than derived.
	 */
	double pack_us;
	/*
	 * And the CPU's share of the projections, which runs INSIDE the fence
	 * window rather than beside it: it is time the calling thread used to
	 * spend blocked. It is charged here so a round can see what it cost as
	 * well as what it bought.
	 */
	double cpu_us;
	/*
	 * ⚠ THE WHOLE CALL, entry to return, so the residue stops being a
	 * SUBTRACTION.
	 *
	 * The stage table's five NPU rows minus the hardware path minus the
	 * packing left 1.95 ms a token at 64 tokens and 7.22 at 384 -- and the
	 * hardware path itself was 3.16 ms a token FASTER at the longer
	 * context, which is time moving from inside these timers to outside
	 * them rather than any work being done. Two derived quantities have
	 * already been read wrong today. This one is measured.
	 *
	 * What is left over after THIS is only what llama.c does around the
	 * call: finding the tensor and quantising the output.
	 */
	double call_us;

	/*
	 * A wedged block answers every submit with a driver side timeout and
	 * the ioctl still returns success, so the only reliable detector is the
	 * clock. Three slow submits and this path retires itself.
	 */
	double slow_us, min_gbs;
	/*
	 * ⚠ AND HOW OFTEN, because the one-shot message on its own is a
	 * MISLEADING INSTRUMENT and it cost two rounds of wrong hypothesis.
	 *
	 * Rounds 373, 374 and 374's repeat each printed one notice, always at
	 * K=2048 N=2048, and only ever in a long context run. That looked like
	 * the NPU idling across attention's 16 to 38 ms gaps and paying to wake
	 * up. It is not, and arithmetic settles it without a board round:
	 *
	 *   - the floor is a RATE, so the stall needed to trip it scales with
	 *     the tensor. o_proj is 2.1 MB and trips on 1.0 ms; down is 8.4 MB
	 *     and needs 4.2; the output head is 131 MB and needs 65.7. o_proj
	 *     is simply the most sensitive thing being watched.
	 *   - the GROUPED path has no check at all, so q, k, v, gate and up can
	 *     never trip it. Only three tensors are watched and o_proj is the
	 *     smallest of them.
	 *   - and it cannot be per token: o_proj at 0.84 GB/s would take 98 ms
	 *     rather than 8, and the token would be 180 ms instead of 90.
	 *
	 * So it is a rare stall, and a long run trips it because it has six
	 * times as many submits, not because of the gaps. Counting them says
	 * that outright instead of leaving it to be re-guessed.
	 */
	unsigned long slow_n;
	double slow_worst;
	unsigned slow_worst_k, slow_worst_n;
	int strikes, dead, whined, nochain, slowed, nofini, inprep, plain;
	int serialpack;
	/*
	 * What fraction of every projection's OUTPUT CHANNELS the CPU keeps.
	 * 0 is the hardware doing all of it, which is every round before 371.
	 */
	double cpu_frac;
	float *afscr;              /* the activation, rounded through fp16 */
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
/*
 * acc[j] += fo[j] * sc[j], AND BIT FOR BIT WHAT THE SCALAR LOOP DID.
 *
 * Two things make that claim checkable rather than hopeful. The elements are
 * independent -- each j lands in its own accumulator, so there is no reduction
 * whose order could change. And the product of two floats has at most 48
 * significant bits, which is exact in a double, so the scalar loop's
 * (float)((double)a * (double)b) is the correctly rounded float product and
 * nothing else.
 *
 * ⚠ THE BARRIER IS LOAD BEARING. Without it the compiler fuses the multiply
 * and the add into fmla, which rounds ONCE where the source rounds twice, and
 * 460190 of 20.5 million accumulations came out different in the host check.
 * The cast in the scalar tail is the same barrier written in C.
 */
#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
#include <arm_neon.h>
static void scaled_add(float *acc, const float *fo, const float *sc, unsigned n)
{
	unsigned j = 0;

	for (; j + 4 <= n; j += 4) {
		float32x4_t p = vmulq_f32(vld1q_f32(fo + j), vld1q_f32(sc + j));

		__asm__("" : "+w"(p));
		vst1q_f32(acc + j, vaddq_f32(vld1q_f32(acc + j), p));
	}
	for (; j < n; j++)
		acc[j] += (float)((double)fo[j] * (double)sc[j]);
}
#else
static void scaled_add(float *acc, const float *fo, const float *sc, unsigned n)
{
	for (unsigned j = 0; j < n; j++)
		acc[j] += (float)((double)fo[j] * (double)sc[j]);
}
#endif

/*
 * THE CPU'S SHARE OF A PROJECTION, run in the window the calling thread
 * currently spends BLOCKED in prep_bo.
 *
 * y[r] = sum over groups of scale[r][g] * sum over the group of code * a,
 * which is the same arithmetic the hardware does: the same int4 codes, the
 * same per group scale, and an activation rounded through fp16 first so both
 * halves of the split see the same numbers.
 *
 * ⚠ ONE THREAD, DELIBERATELY. Round 370 measured the CPU reading memory at
 * 7.13 GB/s on one thread, 6.65 on two and 6.32 on four: a single thread
 * already saturates the controller, so a fan out here would cost a
 * synchronisation and buy nothing. It is also why this is worth trying at all
 * -- the same round found the NPU and a CPU reader reaching 15.46 GB/s
 * together against 10.46 for the NPU alone.
 */
static void cpu_rows(const struct npu_entry *e, const float *af, float *y)
{
	const struct npu_tensor *t = e->t;
	uint64_t k = t->k;
	uint64_t grp = t->kgroup ? t->kgroup : k;
	uint64_t ngrp = (k + grp - 1) / grp;
	size_t per = ((size_t)k + 1) / 2;
	unsigned nc = (unsigned)t->n - e->n_npu;

	for (unsigned r = 0; r < nc; r++) {
		const uint8_t *row = e->cq + (size_t)r * per;
		const float *sc = t->scale + (size_t)(e->n_npu + r) * ngrp;
		double acc = 0.0;

		for (uint64_t gi = 0; gi < ngrp; gi++) {
			uint64_t lo = gi * grp;
			uint64_t hi = lo + grp < k ? lo + grp : k;
			uint64_t i = lo;
			float part = 0.0f;

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
			{
			float32x4_t a0 = vdupq_n_f32(0.0f);
			float32x4_t a1 = vdupq_n_f32(0.0f);
			float32x4_t a2 = vdupq_n_f32(0.0f);
			float32x4_t a3 = vdupq_n_f32(0.0f);

			/*
			 * ⚠ THE VECTOR PATH READS BYTE i/2 AND TAKES ITS LOW
			 * NIBBLE AS WEIGHT i, so it only means that when i is
			 * EVEN. Every group in this runtime starts on a
			 * multiple of kmax and is even, but a group that
			 * started odd would silently read every weight off by
			 * one: at k = 34 with groups of 17 the test measured
			 * 2.26 relative against 1e-6 elsewhere. One scalar
			 * step fixes the alignment.
			 */
			if ((i & 1) && i < hi) {
				part += (float)(((row[i >> 1] >> 4) >= 8)
					? (row[i >> 1] >> 4) - 16
					: (row[i >> 1] >> 4)) * af[i];
				i++;
			}
			for (; i + 16 <= hi; i += 16) {
				uint8x8_t b = vld1_u8(row + (i >> 1));
				int8x8_t l = vshr_n_s8(vshl_n_s8(
					vreinterpret_s8_u8(vand_u8(b,
						vdup_n_u8(0x0f))), 4), 4);
				int8x8_t h = vshr_n_s8(vreinterpret_s8_u8(b), 4);
				int8x8x2_t z = vzip_s8(l, h);
				int16x8_t w;

				w = vmovl_s8(z.val[0]);
				a0 = vfmaq_f32(a0,
					vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))),
					vld1q_f32(af + i));
				a1 = vfmaq_f32(a1,
					vcvtq_f32_s32(vmovl_s16(vget_high_s16(w))),
					vld1q_f32(af + i + 4));
				w = vmovl_s8(z.val[1]);
				a2 = vfmaq_f32(a2,
					vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))),
					vld1q_f32(af + i + 8));
				a3 = vfmaq_f32(a3,
					vcvtq_f32_s32(vmovl_s16(vget_high_s16(w))),
					vld1q_f32(af + i + 12));
			}
			/* four accumulators, not two: it halves the
			 * dependency chain and the summation error with it */
			part += vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1),
						     vaddq_f32(a2, a3)));
			}
#endif
			for (; i < hi; i++) {
				uint8_t byte = row[i >> 1];
				int v = (i & 1) ? (byte >> 4) : (byte & 0xf);

				part += (float)(v >= 8 ? v - 16 : v) * af[i];
			}
			acc += (double)part * (double)sc[gi];
		}
		y[e->n_npu + r] = (float)acc;
	}
}

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
	/*
	 * SKIP THE FLUSH ON A BUFFER THE CPU ONLY READ.
	 *
	 * fini_bo is dma_sync_sgtable_for_device, which on arm64 CLEANS every
	 * line of the buffer to the point of coherency. That is what a buffer
	 * the CPU WROTE needs -- the activation buffer does need it -- but an
	 * output buffer is only ever read here, so its lines are clean already
	 * and the walk writes nothing back. What makes skipping it safe is that
	 * the next read is preceded by prep_bo, whose sync_for_cpu invalidates
	 * those stale clean lines before the CPU can see them.
	 *
	 * ROUND 367 RAN IT: 3.4 ms a token of flush went to 0.02, the sentence
	 * came back word for word identical to the arm that kept the flush and
	 * to the one device control, and the run went 9.86 to 10.26 tok/s. So it
	 * is the default now, and CHARSIU_NPU_FINI puts the clean back.
	 */
	g->nofini = getenv("CHARSIU_NPU_FINI") == NULL;
	/*
	 * AND THE SAME ARGUMENT ON THE OTHER SIDE, which round 367 did NOT run.
	 *
	 * prep_bo on the ACTIVATION buffer waits for a write fence and then
	 * invalidates. Neither is needed: the device only ever READS that
	 * buffer, so there is no write fence to wait for, and the CPU is about
	 * to overwrite every byte the next job will read, so there is nothing
	 * stale worth dropping first. Its fini stays -- the CPU wrote it, and
	 * that clean is what makes the bytes visible to the hardware.
	 *
	 * What makes it safe to write at all is the ORDER: the previous job on
	 * this buffer finished before this call, because its output prep waited
	 * on the fence that covers the whole job, reads included.
	 *
	 * It is 65 entries a token times two devices: 130 ioctls and 130 cache
	 * walks. CHARSIU_NPU_INPREP puts them back.
	 */
	g->inprep = getenv("CHARSIU_NPU_INPREP") != NULL;
	/*
	 * ONE SWITCH FOR THE THREE THINGS ROUND 369 CHANGED that have no
	 * behaviour of their own to show: the vectorised half conversion, the
	 * activation reaching the packer without a copy, and the sum landing in
	 * the caller's buffer instead of a staging one. All three are provably
	 * neutral -- the packer was compared byte for byte against the old loop
	 * at thirteen shapes -- so the control is not about whether they are
	 * right. It is so a round that comes out SLOWER can say which of them
	 * did it, which is the lesson round 368's attention arm taught.
	 */
	g->plain = getenv("CHARSIU_NPU_PLAIN") != NULL;
	/*
	 * ⚠ THE LEGACY BIT PATTERN LAYOUT CANNOT BE SPLIT. It accumulates with
	 * |=, so two channels can share a byte and two threads would race for
	 * it. charsiu_pack_weights_rows does not implement that layout at all,
	 * which would silently produce the CURRENT one instead, so the check
	 * belongs here rather than in a comment.
	 */
	g->serialpack = g->plain || getenv("CHARSIU_W4_BITPAT") != NULL;
	{
		const char *e = getenv("CHARSIU_NPU_CPU_FRAC");

		g->cpu_frac = e ? atof(e) : 0.0;
		if (g->cpu_frac < 0.0)
			g->cpu_frac = 0.0;
		if (g->cpu_frac > 0.9)
			g->cpu_frac = 0.9;
	}
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
	g->afscr = calloc(max_k ? max_k : 1, sizeof(*g->afscr));
	g->wpack = malloc((size_t)g->nmax * g->kmax + 4096);
	g->asum = calloc(ks ? ks : 1, sizeof(*g->asum));
	/* a GROUP can carry several tensors' slices, so four times over */
	g->tasks = calloc(4 * g->max_slices, sizeof(*g->tasks));
	g->handles = calloc(1 + 8 * g->max_slices, sizeof(*g->handles));
	if (!g->ent || !g->slot || !g->scratch || !g->acc || !g->accf ||
	    !g->fscr || !g->afscr || !g->wpack || !g->asum || !g->tasks || !g->handles)
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
		for (unsigned i = 0; i < g->n_slot; i++)
			free(g->slot[i].sc);
		for (unsigned i = 0; i < g->n_ent; i++)
			free(g->ent[i].cq);
		for (unsigned i = 0; i < g->n_ent; i++)
			for (unsigned d = 0; d < g->ndev; d++)
			charsiu_bo_free(g->dev[d], &g->ent[i].out[d]);
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
	free(g->afscr);
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
/*
 * Does the hardware path need the int8 activation? int4 takes the float one
 * and never looks at q1, so llama.c can skip realising it -- but only npudev
 * knows which mode it opened in, and duplicating the getenv in the caller is
 * how two switches drift apart.
 */
int charsiu_npu_needs_q1(const struct charsiu_npu *g)
{
	return !g || !g->w4;
}

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
			"waiting for the fence (the invalidate is in there), "
			"%.0f ms summing the slices, %.0f ms in the flush\n"
			"charsiu NPU: and %.0f ms packing the activation, "
			"which is ON TOP of the hardware path above; "
			"%.0f ms was the CPU's own share of the projections, "
			"inside the fence window\n"
			"charsiu NPU: %.0f ms in these calls end to end, so "
			"%.0f ms of them is neither hardware nor packing\n",
			g->busy_us / 1e3, g->weight_mb / g->busy_us * 1e3,
			g->busy_us / (double)g->submits,
			g->submit_us / 1e3, g->fence_us / 1e3,
			g->copy_us / 1e3, g->fini_us / 1e3, g->pack_us / 1e3,
			g->cpu_us / 1e3, g->call_us / 1e3,
			(g->call_us - g->busy_us - g->pack_us) / 1e3);
	if (g->slow_n)
		fprintf(stderr,
			"charsiu NPU: %lu of %lu submits came in under %.1f "
			"GB/s, worst %.2f at K=%u N=%u\n"
			"charsiu NPU: ⚠ the floor is a RATE, so the stall that "
			"trips it scales with the tensor -- and only the three "
			"UNGROUPED ones are watched at all, of which K=2048 "
			"N=2048 is the smallest and trips on a 1 ms hiccup\n",
			g->slow_n, g->submits, g->min_gbs, g->slow_worst,
			g->slow_worst_k, g->slow_worst_n);
	if (!g->submits)
		fprintf(stderr,
			"charsiu NPU: NOTHING RAN ON THE HARDWARE. Every number "
			"in this run came from the CPU.\n");
}

/*
 * One range of a slice's output channels: gather the quantised bytes into the
 * shape the packer wants, then pack them. Both are indexed by channel and both
 * write disjoint bytes, so the ranges do not meet.
 */
struct wrows {
	struct charsiu_npu *g;
	const struct npu_tensor *t;
	const struct charsiu_matmul *mm;
	unsigned n0, k0, k;
};

static void pack_rows(void *vw, uint64_t r0, uint64_t nr)
{
	const struct wrows *w = vw;
	struct charsiu_npu *g = w->g;

	for (uint64_t r = r0; r < r0 + nr; r++) {
		const int8_t *src = w->t->q
				  + (size_t)(w->n0 + r) * w->t->k + w->k0;
		uint8_t *dst = g->scratch + (size_t)r * w->k;

		for (unsigned c = 0; c < w->k; c++)
			dst[c] = g->w4 ? (uint8_t)(src[c] & 0xf)
				       : (uint8_t)((int)src[c] + 128);
	}
	charsiu_pack_weights_rows(w->mm, g->scratch, g->wpack,
				  (unsigned)r0, (unsigned)nr);
}

/* One slice: rows [n0, n0+n) and columns [k0, k0+k) of t, writing region si. */
static int add_slice(struct charsiu_npu *g, unsigned di,
		     const struct npu_tensor *t,
		     unsigned n0, unsigned n, unsigned k0, unsigned k,
		     unsigned ki, unsigned si, uint32_t out_base)
{
	struct npu_slot *s = &g->slot[g->n_slot];

	charsiu_note("staging a slice", (unsigned long)n, (unsigned long)k);
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

	/*
	 * Gather this slice's scales into the order the sum wants. Only a
	 * grouped tensor has a scale per (channel, group); an ungrouped one is
	 * scaled once at the end, where the stride does not arise.
	 */
	if (tensor_grouped(g, t)) {
		uint64_t ng = t->k / t->kgroup;
		uint64_t gi = k0 / t->kgroup;

		s->sc = malloc((size_t)n * sizeof(float));
		if (!s->sc) {
			whine(g, "the scale gather would not allocate", k, n);
			goto out;
		}
		for (unsigned j = 0; j < n; j++)
			s->sc[j] = t->scale[(uint64_t)(n0 + j) * ng + gi];
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
	 *
	 * ⚠ AND THIS IS THE OTHER HALF OF A COLD START. Round 369's board log
	 * reads 4.4 seconds "adding" against 7.9 "quantising", and where the
	 * quantiser now splits over the pool this did not: 620 MB of nibbles
	 * gathered and packed on one core. Both steps index by output channel
	 * and write disjoint bytes, so a range of channels is a whole unit of
	 * work -- checked byte for byte against the whole matrix call at nine
	 * shapes and five split counts.
	 */
	{
		struct wrows wr = { g, t, &s->job.mm, n0, k0, k };

		memset(g->wpack, 0, charsiu_weight_bytes(&s->job.mm));
		if (g->serialpack)
			pack_rows(&wr, 0, n);
		else
			charsiu_parallel_for(pack_rows, &wr, n);
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
	unsigned e_n_npu;

	/* ⚠ start the clock on the FIRST tensor, not the first heartbeat, or
	 * the first sixteen are free and round 353's log said "0 ms". */
	if (g->t_first == 0.0)
		g->t_first = t_add;
	/*
	 * ⚠ t->name IS A FIXED ARRAY INSIDE npu_tensor, not a stack buffer, so
	 * it is still readable from a signal handler after this frame is gone.
	 */
	charsiu_note(t->name, (unsigned long)t->n, (unsigned long)t->k);
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
	/*
	 * ⚠ THE TWO SIDES MUST AGREE ABOUT GROUPING, and when they did not the
	 * answer was wrong rather than absent. The quantiser rounded a partial
	 * last group up and wrote scales as scale[row * ngrp + group];
	 * tensor_grouped() below refuses a remainder, so the consumer read the
	 * same array as scale[row] and every row took some other row's scale.
	 * Qwen2.5-1.5B (k 1536 and 8960 against a 1024 slice) decoded fluent
	 * nonsense on the board while the same file was correct on the CPU.
	 *
	 * The quantiser no longer emits that state. This is here so that if it
	 * ever does again, the tensor falls back to the CPU and says why,
	 * instead of returning numbers nobody can tell are wrong.
	 */
	if (t->kgroup && t->kgroup < t->k && (t->k % t->kgroup)) {
		whine(g, "a partial weight group would be read as one scale a row",
		      (unsigned)t->k, (unsigned)t->n);
		return -1;
	}

	/*
	 * DECIDE THE SPLIT FIRST, because everything below is geometry over the
	 * hardware's share.
	 *
	 * Rounded DOWN to sixteen: sixteen output channels is the feature atom
	 * the int4 weight layout blocks by, and giving the hardware a ragged
	 * count to save the CPU a handful of rows is a bad trade. A tensor too
	 * narrow to split keeps all of it.
	 */
	e_n_npu = (unsigned)t->n;
	/*
	 * ⚠ ONLY WHERE THE SUM ALREADY LANDS IN THE CALLER'S BUFFER. The split
	 * writes the CPU's rows into y before the fence, and the read back then
	 * fills the rest; that only works on the path where the hardware's rows
	 * go straight into y and the conversion at the end is skipped, which is
	 * grouped int4 with the vector paths on.
	 */
	if (g->cpu_frac > 0.0 && g->w4 && !g->plain && tensor_grouped(g, t) &&
	    t->n >= 64) {
		unsigned keep = (unsigned)((double)t->n * (1.0 - g->cpu_frac));

		keep &= ~15u;
		if (keep < 16)
			keep = 16;
		if (keep < (unsigned)t->n)
			e_n_npu = keep;
	}

	ns = (unsigned)((e_n_npu + g->nmax - 1) / g->nmax);
	ks = (unsigned)((t->k + g->kmax - 1) / g->kmax);
	if (ns * ks > g->max_slices || first + ns * ks > g->slot_cap) {
		whine(g, "no slice slots left", (unsigned)t->k, (unsigned)t->n);
		return -1;
	}

	e = &g->ent[g->n_ent];
	memset(e, 0, sizeof(*e));
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
	/*
	 * ⚠ SIZED FOR THE SLICES THIS DEVICE ACTUALLY GETS, not for all of them.
	 *
	 * The slices alternate, so each device holds half of ns*ks, and
	 * allocating both buffers at the full size doubled the cache
	 * maintenance a matvec pays: charsiu_bo_prep and _fini work over a WHOLE
	 * buffer object, and round 366 measured 13.4 ms a token in the readback
	 * against 11.6 on the one device build. The output head alone is 512 KB
	 * a buffer, so this is half a megabyte of cache operations a token
	 * bought back for nothing.
	 */
	for (unsigned d = 0; d < g->ndev; d++) {
		size_t slots = ((size_t)ns * ks + 1) / (g->ndev > 1 ? 2 : 1) + 1;

		if (charsiu_bo_alloc(g->dev[d],
				     slots * g->out_stride + 4096,
				     &e->out[d])) {
			whine(g, "an output buffer would not allocate",
			      (unsigned)t->k, (unsigned)t->n);
			return -1;
		}
	}

	/*
	 * ⚠ THE SLICES OF ONE TENSOR SPLIT TOO, not just the members of a
	 * group. Round 364 put the second core in and got only 7%, because the
	 * o_proj, the down_proj and the 128256 wide output head all go through
	 * the single projection path -- more than 40% of the weight traffic in
	 * tensors that were never grouped with anything. Alternating the SLICES
	 * gives those two cores as well.
	 */
	{
	unsigned sid[2] = { 0, 0 };

	for (unsigned ki = 0; ki < ks; ki++) {
		unsigned k0 = ki * g->kmax;
		unsigned k = (unsigned)(t->k - k0) < g->kmax
			   ? (unsigned)(t->k - k0) : g->kmax;

		for (unsigned ni = 0; ni < ns; ni++, si++) {
			unsigned n0 = ni * g->nmax;
			unsigned n = (e_n_npu - n0) < g->nmax
				   ? (e_n_npu - n0) : g->nmax;

			unsigned d = g->ndev > 1 ? ((ki * ns + ni) & 1) : 0;

			if (add_slice(g, d, t, n0, n, k0, k, ki, sid[d]++,
				      (uint32_t)e->out[d].dma_address) < 0) {
				g->n_slot = first;
				return -1;
			}
			g->slices++;
		}
	}
	}

	e->t = t;
	e->first = first;
	e->count = ns * ks;
	e->n_slices = ns;
	e->k_slices = ks;
	e->n_npu = e_n_npu;
	/*
	 * THE CPU'S ROWS, PACKED TWO WEIGHTS TO A BYTE.
	 *
	 * t->q keeps one whole byte per int4 weight, which is what the
	 * quantiser hands everything downstream. Reading the CPU's share out of
	 * it would cost twice the bytes the hardware pays for the same weights,
	 * and bandwidth is the entire point of the split, so those rows get
	 * their own packed copy: low nibble first, row major, nothing
	 * scrambled. It is f * n * k / 2 bytes, 136 MB of this model at a
	 * quarter of the rows, against 620 MB of weights already resident.
	 */
	if (e_n_npu < (unsigned)t->n) {
		unsigned nc = (unsigned)t->n - e_n_npu;
		size_t per = ((size_t)t->k + 1) / 2;

		e->cq = malloc((size_t)nc * per);
		if (!e->cq) {
			whine(g, "the CPU's share would not allocate",
			      (unsigned)t->k, (unsigned)t->n);
			e->n_npu = (unsigned)t->n;   /* fall back to all NPU */
		} else {
			for (unsigned r = 0; r < nc; r++) {
				const int8_t *src = t->q
					+ (size_t)(e_n_npu + r) * t->k;
				uint8_t *dst = e->cq + (size_t)r * per;
				uint64_t i;

				for (i = 0; i + 1 < t->k; i += 2)
					dst[i >> 1] = (uint8_t)((src[i] & 0xf) |
						((src[i + 1] & 0xf) << 4));
				if (i < t->k)
					dst[i >> 1] = (uint8_t)(src[i] & 0xf);
			}
		}
	}
	/*
	 * ⚠ BYTES, NOT ELEMENTS. This counted n*k for both precisions, so every
	 * "GB/s of weights" this project has printed for int4 was DOUBLE the
	 * real figure -- 13.4 GB/s in round 356's log is 6.7. int8 was right by
	 * accident, one byte an element. The honest comparison is int4 at 6.7
	 * GB/s against int8's 9.46, which is what it looked like from the
	 * outside and what the shape sweep said.
	 */
	e->weight_mb = (double)e_n_npu * (double)t->k
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
	double t0, tpack, tcall = now_us();
	float *af;

	if (g->dead || id < 0 || (unsigned)id >= g->n_ent)
		return -1;
	charsiu_note("a matvec on one tensor", (unsigned long)id,
		     (unsigned long)a->n);
	e = &g->ent[id];
	if ((unsigned)a->n != e->t->k) {
		whine(g, "the activation is not this tensor's K", (unsigned)a->n,
		      (unsigned)e->t->n);
		return -1;
	}

	/* every K slice's activation, each in its own region */
	/*
	 * ⚠ THE ACTIVATION GOES INTO EVERY DEVICE, and round 365 shipped without
	 * it. The slices of one tensor are spread across both devices now, so
	 * the ones on the other device read an input buffer nobody wrote --
	 * and the decode came back as word salad on BOTH int8 and int4 while
	 * the one device control was perfect. The grouped path had already been
	 * fixed for exactly this and the single path was not.
	 *
	 * A buffer object belongs to the file that created it, so there is no
	 * sharing one: it has to be packed twice. That is a few kilobytes
	 * against the megabytes of weights a submit fetches.
	 */
	tpack = now_us();
	for (unsigned d = 0; d < g->ndev; d++) {
	if (g->inprep)
		charsiu_bo_prep(g->dev[d], &g->in[d], 1000000000);
	for (unsigned ki = 0; ki < e->k_slices; ki++) {
		const struct npu_slot *s = &g->slot[e->first + ki * e->n_slices];

		if (g->w4) {
			const float *src = a->f + s->k0;

			/*
			 * STRAIGHT FROM THE ACTIVATION, NO COPY. The packer
			 * reads src[kk] at m = 1, so the scratch buffer was
			 * 463 thousand float copies a token to hand it bytes
			 * it could already see. The running total is the
			 * midrise grid's half step, and midrise is off by
			 * default and measured worse when it was on, so it no
			 * longer costs a double add per element either.
			 */
			if (g->midrise || g->plain) {
				double as = 0.0;

				for (i = 0; i < s->job.mm.k; i++) {
					g->fscr[i] = src[i];
					as += (double)g->fscr[i];
				}
				g->asum[ki] = as;
				src = g->fscr;
			}
			charsiu_pack_input_f16(&s->job.mm, src,
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
	g->pack_us += now_us() - tpack;

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
	/*
	 * ONE JOBLIST PER DEVICE, BOTH ISSUED BEFORE EITHER IS WAITED ON. The
	 * slices of this tensor alternate between the devices, so a projection
	 * that is not part of a group still uses both cores.
	 */
	t0 = now_us();
	for (unsigned d = 0; d < g->ndev; d++) {
		unsigned nt = 0;

		nh = 0;
		g->handles[nh++] = g->in[d].handle;
		for (i = 0; i < e->count; i++) {
			const struct npu_slot *s = &g->slot[e->first + i];

			if (s->di != d)
				continue;
			g->tasks[nt].regcmd = (uint32_t)s->regcmd.dma_address;
			g->tasks[nt].regcmd_count = s->nreg;
			nt++;
			g->handles[nh++] = s->wt.handle;
			g->handles[nh++] = s->coef.handle;
		}
		if (!nt)
			continue;
		jl.tasks = g->tasks;
		jl.task_count = nt;
		jl.in_handles = g->handles;
		jl.in_count = nh;
		jl.out_handles = &e->out[d].handle;
		jl.out_count = 1;
		if (charsiu_submit_jobs(g->dev[d], &jl, 1)) {
			g->strikes = 3;
			break;
		}
		g->submits++;
	}
	g->submit_us += now_us() - t0;

	/*
	 * THE CPU'S ROWS, WHILE THE HARDWARE HAS THE REST. The submit above is
	 * asynchronous, so from here until prep_bo the calling thread has
	 * nothing to do but block. Nothing is scheduled and nothing is waited
	 * on: the work simply fills a window that was already being spent.
	 */
	if (e->cq && e->n_npu < (unsigned)e->t->n) {
		double tc = now_us();

		for (uint64_t i = 0; i < e->t->k; i++)
			g->afscr[i] = charsiu_half_to_float(
					charsiu_float_to_half(a->f[i]));
		cpu_rows(e, g->afscr, y);
		g->cpu_us += now_us() - tc;
	}

	if (g->strikes < 3) {
		double t1 = now_us();

		for (unsigned d = 0; d < g->ndev; d++)
			charsiu_bo_prep(g->dev[d], &e->out[d], 2000000000);
		g->fence_us += now_us() - t1;
		t1 = now_us();
		int grp = tensor_grouped(g, e->t);

		/*
		 * ONE OF THESE, NOT BOTH. int4 sums into accf and int8 into
		 * acc, and clearing the other one is 2 MB a token of writes
		 * for an array nothing will read: the 128256 wide head alone
		 * is half a megabyte of it.
		 *
		 * AND WHEN THE LAST STEP WOULD BE A COPY, SUM WHERE THE ANSWER
		 * GOES. A grouped int4 tensor applies its scale per slice on
		 * the way in, so the conversion at the end of this function is
		 * y[i] = accf[i] and nothing else: a staging buffer read and
		 * written for no reason.
		 */
		af = (g->w4 && grp && !g->plain) ? y : g->accf;
		/* ⚠ the hardware's rows only: the CPU's are already written */
		if (g->w4)
			memset(af, 0, (size_t)e->n_npu * sizeof(*af));
		else
			memset(g->acc, 0, (size_t)e->t->n * sizeof(*g->acc));
		for (i = 0; i < e->count; i++) {
			const struct npu_slot *s = &g->slot[e->first + i];
			const uint8_t *base = (const uint8_t *)e->out[s->di].map +
					      s->out_slot * g->out_stride;

			/* int4 writes float32, int8 the raw int32 accumulator */
			if (g->w4) {
				const float *fo = (const float *)base;

				if (grp) {
					double hs = g->midrise
						  ? 0.5 * g->asum[s->k0 / g->kmax]
						  : 0.0;
					const float *sc = s->sc;

					/* non null whenever grp is: the two
					 * ask tensor_grouped the same
					 * question, and a gather that will not
					 * allocate fails the staging */
					if (hs == 0.0 && !g->plain) {
						scaled_add(af + s->n0, fo, sc,
							   s->job.mm.n);
						continue;
					}
					for (unsigned j = 0; j < s->job.mm.n; j++)
						af[s->n0 + j] +=
						  (float)((fo[j] + hs) * sc[j]);
				} else {
					for (unsigned j = 0; j < s->job.mm.n; j++)
						af[s->n0 + j] += fo[j];
				}
				continue;
			}
			out = (const int32_t *)base;
			for (unsigned j = 0; j < s->job.mm.n; j++)
				g->acc[s->n0 + j] += out[j];
		}
		g->copy_us += now_us() - t1;
		t1 = now_us();
		if (!g->nofini)
			for (unsigned d = 0; d < g->ndev; d++)
				charsiu_bo_fini(g->dev[d], &e->out[d]);
		g->fini_us += now_us() - t1;
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
			if (gbs < g->min_gbs && g->submits > g->n_ent * 2) {
				g->slow_n++;
				if (g->slow_worst == 0.0 || gbs < g->slow_worst) {
					g->slow_worst = gbs;
					g->slow_worst_k = (unsigned)e->t->k;
					g->slow_worst_n = (unsigned)e->t->n;
				}
			}
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

		if (g->w4 && grp && !g->plain) {
			g->call_us += now_us() - tcall;
			return 0;              /* it was summed into y */
		}
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
	g->call_us += now_us() - tcall;
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
	double t0, t1, tpack, fspent = 0.0, tcall = now_us();

	if (g->dead || !n || n > 8)
		return -1;
	for (i = 0; i < n; i++)
		if (ids[i] < 0 || (unsigned)ids[i] >= g->n_ent)
			return -1;
	charsiu_note("a group: checking the entries", (unsigned long)n,
		     (unsigned long)a->n);
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
	unsigned nd = 0;

	tpack = now_us();
	for (unsigned d = 0; d < g->ndev; d++) {
		nd++;
	charsiu_note("a group: packing the activation", (unsigned long)d,
		     (unsigned long)e0->k_slices);
	if (g->inprep)
		charsiu_bo_prep(g->dev[d], &g->in[d], 1000000000);
	for (unsigned ki = 0; ki < e0->k_slices; ki++) {
		const struct npu_slot *s = &g->slot[e0->first + ki * e0->n_slices];

		if (g->w4) {
			const float *src = a->f + s->k0;

			/*
			 * STRAIGHT FROM THE ACTIVATION, NO COPY. The packer
			 * reads src[kk] at m = 1, so the scratch buffer was
			 * 463 thousand float copies a token to hand it bytes
			 * it could already see. The running total is the
			 * midrise grid's half step, and midrise is off by
			 * default and measured worse when it was on, so it no
			 * longer costs a double add per element either.
			 */
			if (g->midrise || g->plain) {
				double as = 0.0;

				for (i = 0; i < s->job.mm.k; i++) {
					g->fscr[i] = src[i];
					as += (double)g->fscr[i];
				}
				g->asum[ki] = as;
				src = g->fscr;
			}
			charsiu_pack_input_f16(&s->job.mm, src,
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
	g->pack_us += now_us() - tpack;
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

		charsiu_note("a group: building the joblist", (unsigned long)d,
			     (unsigned long)n);
		nh = 0;
		ntask = 0;
		g->handles[nh++] = g->in[d].handle;
		for (i = 0; i < n; i++) {
			struct npu_entry *e = &g->ent[ids[i]];
			unsigned any = 0;

			for (j = 0; j < e->count; j++) {
				const struct npu_slot *s =
					&g->slot[e->first + j];

				/* ⚠ filter by the SLICE's device, not the
				 * entry's: since round 365 a tensor's slices
				 * are spread across both. */
				if (s->di != d)
					continue;
				if (ntask >= 4 * g->max_slices)
					return -1;
				g->tasks[ntask].regcmd =
					(uint32_t)s->regcmd.dma_address;
				g->tasks[ntask].regcmd_count = s->nreg;
				ntask++;
				g->handles[nh++] = s->wt.handle;
				g->handles[nh++] = s->coef.handle;
				any = 1;
			}
			if (any)
				outh[no++] = e->out[d].handle;
		}
		if (!no)
			continue;
		jl.tasks = g->tasks;
		jl.task_count = ntask;
		jl.in_handles = g->handles;
		jl.in_count = nh;
		jl.out_handles = outh;
		jl.out_count = no;
		charsiu_note("a group: submitting", (unsigned long)ntask,
			     (unsigned long)nh);
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

	/*
	 * The group's CPU rows, in the same window. A group shares one
	 * activation, so it is rounded through fp16 once for all of them.
	 */
	{
		unsigned any = 0;

		for (i = 0; i < n; i++)
			if (g->ent[ids[i]].cq)
				any = 1;
		if (any) {
			double tc = now_us();

			charsiu_note("a group: the CPU's own rows",
				     (unsigned long)n, (unsigned long)e0->t->k);
			for (uint64_t q = 0; q < e0->t->k; q++)
				g->afscr[q] = charsiu_half_to_float(
					charsiu_float_to_half(a->f[q]));
			for (i = 0; i < n; i++) {
				struct npu_entry *e = &g->ent[ids[i]];

				if (e->cq && e->n_npu < (unsigned)e->t->n)
					cpu_rows(e, g->afscr, ys[i]);
			}
			g->cpu_us += now_us() - tc;
		}
	}

	t1 = now_us();
	for (i = 0; i < n; i++)
		for (unsigned d = 0; d < g->ndev; d++) {
			charsiu_note("a group: waiting on the fence",
				     (unsigned long)i, (unsigned long)d);
			charsiu_bo_prep(g->dev[d], &g->ent[ids[i]].out[d],
					2000000000);
		}
	g->fence_us += now_us() - t1;

	t1 = now_us();
	for (i = 0; i < n; i++) {
		struct npu_entry *e = &g->ent[ids[i]];
		const int32_t *out;

		int grp = tensor_grouped(g, e->t);
		float *af = (g->w4 && grp && !g->plain) ? ys[i] : g->accf;

		/*
		 * ONE OF THESE, NOT BOTH. int4 sums into accf and int8 into
		 * acc, and clearing the other one is 2 MB a token of writes
		 * for an array nothing will read: the 128256 wide head alone
		 * is half a megabyte of it.
		 *
		 * AND WHEN THE LAST STEP WOULD BE A COPY, SUM WHERE THE ANSWER
		 * GOES. A grouped int4 tensor applies its scale per slice on
		 * the way in, so the conversion below is ys[i][q] = accf[q]
		 * and nothing else.
		 */
		/* ⚠ the hardware's rows only: the CPU's are already written */
		charsiu_note("a group: clearing the accumulator",
			     (unsigned long)e->t->n, (unsigned long)e->n_npu);
		if (g->w4)
			memset(af, 0, (size_t)e->n_npu * sizeof(*af));
		else
			memset(g->acc, 0, (size_t)e->t->n * sizeof(*g->acc));
		for (j = 0; j < e->count; j++) {
			charsiu_note("a group: reading a slice back",
				     (unsigned long)i, (unsigned long)e->count);
			const struct npu_slot *s = &g->slot[e->first + j];
			const uint8_t *base;

			/*
			 * ⚠ THE TWO NUMBERS THAT CAN MAKE THE NEXT LINE A NULL
			 * DEREFERENCE. e->out is an array of ndev buffers, so
			 * a slot whose di is not a device index reads past it
			 * and takes whatever .map happens to be there. Naming
			 * them here costs one store and turns a segfault into
			 * a sentence.
			 */
			charsiu_note("a group: a slice's device and out slot",
				     (unsigned long)s->di,
				     (unsigned long)s->out_slot);
			if (s->di >= g->ndev || !e->out[s->di].map) {
				fprintf(stderr, "charsiu: slice %u of tensor "
					"%u has device %u of %u and map %p\n",
					j, i, s->di, g->ndev,
					s->di < g->ndev ? e->out[s->di].map
							: NULL);
				g->dead = 1;
				return -1;
			}
			base = (const uint8_t *)e->out[s->di].map +
			       s->out_slot * g->out_stride;

			/*
			 * ⚠ THE SLICE'S OWN WIDTH AND OFFSET, which decide how
			 * far the two loops below walk. The guard above proved
			 * the base pointer is a real mapping; a garbage n or n0
			 * walks off the end of it, or off the accumulator, and
			 * looks exactly the same from outside.
			 */
			charsiu_note("a group: a slice's width and offset",
				     (unsigned long)s->job.mm.n,
				     (unsigned long)s->n0);
			if (s->n0 + s->job.mm.n > g->max_n ||
			    s->job.mm.n > g->nmax ||
			    (s->out_slot + 1) * (size_t)g->out_stride
				    > e->out[s->di].size) {
				fprintf(stderr, "charsiu: slice %u of tensor "
					"%u wants n0 %u + n %u of %u, slot %u "
					"of a %zu byte buffer\n",
					j, i, s->n0, s->job.mm.n, g->max_n,
					s->out_slot, (size_t)e->out[s->di].size);
				g->dead = 1;
				return -1;
			}

			if (g->w4) {
				const float *fo = (const float *)base;

				if (grp) {
					double hs = g->midrise
						  ? 0.5 * g->asum[s->k0 / g->kmax]
						  : 0.0;
					const float *sc = s->sc;

					/* non null whenever grp is: the two
					 * ask tensor_grouped the same
					 * question, and a gather that will not
					 * allocate fails the staging */
					if (hs == 0.0 && !g->plain) {
						scaled_add(af + s->n0, fo, sc,
							   s->job.mm.n);
						continue;
					}
					for (unsigned q = 0; q < s->job.mm.n; q++)
						af[s->n0 + q] +=
						  (float)((fo[q] + hs) * sc[q]);
				} else {
					for (unsigned q = 0; q < s->job.mm.n; q++)
						af[s->n0 + q] += fo[q];
				}
				continue;
			}
			out = (const int32_t *)base;
			for (unsigned q = 0; q < s->job.mm.n; q++)
				g->acc[s->n0 + q] += out[q];
		}
		{
			double tf = now_us();

			if (!g->nofini)
				for (unsigned d = 0; d < g->ndev; d++)
					charsiu_bo_fini(g->dev[d], &e->out[d]);
			fspent += now_us() - tf;
		}
		if (af != ys[i]) {
			double hsu = 0.0;

			if (g->midrise && !grp)
				for (unsigned ki = 0; ki < e->k_slices; ki++)
					hsu += 0.5 * g->asum[ki];
			/*
			 * ⚠ grp STILL HAS TO BE ASKED HERE. This branch is
			 * reached with CHARSIU_NPU_PLAIN, and a grouped tensor
			 * has already had a scale applied per slice on the way
			 * in: multiplying by the row scale as well would give
			 * the control a wrong answer, which is worse than
			 * having no control.
			 */
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
	g->copy_us += now_us() - t1 - fspent;
	g->fini_us += fspent;
	g->busy_us += now_us() - t0;
	g->call_us += now_us() - tcall;
	return 0;
}
