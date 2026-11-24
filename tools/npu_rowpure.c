// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
 *
 * npu_rowpure: IS ONE ROW OF THE int4 MATMUL A PURE FUNCTION OF THAT ROW?
 *
 * Speculative decoding changed the text of one prompt on the board while every
 * committed token up to the divergence was identical. The only thing that
 * differed was WHICH computation wrote the KV of the same positions: the m = 1
 * decode matvec, or row r of a batched verify pass of width w = 2, 4 or 6.
 * --batch-probe compares those two to a 1e-3 bar and reports 2.65e-05. This
 * tool asks the exact question instead, with memcmp: for real weights of a
 * real gguf, is the output of one input row bit identical across
 *
 *   (a) the batch width w in {2, 4, 6}
 *   (b) the row index r inside the batch
 *   (c) the other rows' contents (other targets, random, copies, zeros)
 *   (d) which of the two cores ran it
 *   (e) grouped (charsiu_npu_matmul_group) vs per tensor calls, plain and
 *       deferred (the verify pass uses matmul_defer / matmul_same_defer)
 *
 * and how does each of those relate to the m = 1 decode path, bit for bit?
 *
 * ------------------------------------------------------------------------
 * WHAT THE SOURCE SAYS CAN MOVE THE BITS (read 2026-11-24, branch performance
 * be5d60b; line numbers are approximate)
 * ------------------------------------------------------------------------
 *
 * One output word is y[r][c] = sum over K slices ki of P(r, c, ki) * s, where
 * P is the fp32 partial the NPU writes for one K slice (KMAX = 1024 and
 * W4_GROUP = 1024 on Qwen3-0.6B, pinned by llama_auto_kmax) and s is the
 * per channel scale. Three stages, three questions.
 *
 * 1. THE INPUT PACK (CPU). m = 1: charsiu_pack_input_f16 -> the m == 1 branch
 *    of charsiu_pack_input_f16_stride, charsiu_vhalf four at a time and
 *    charsiu_float_to_half for the tail. m > 1: batch_pack_slice ->
 *    pack_f16_pooled -> the m > 1 branch of the same function or the
 *    edges + groups split (charsiu_vhalf eight at a time, charsiu_f2h tail).
 *    Every element is converted on its own by one function that truncates
 *    (verified over all 2^32 patterns, regcmd.c), and nothing is scaled per
 *    batch: the int4 path has no per row or per batch activation scale (that
 *    is int8's bd1, not reached with W4V=1). AWQ's kscale multiplies per
 *    element in both paths and is NULL for a plain Q4_0 gguf.
 *    => the pack CANNOT depend on w, r, neighbours or grouping. Checked here
 *       on the desk and the board ("pack" lines), by unpacking row r of a
 *       width w pack and comparing it with the m = 1 pack.
 *
 * 2. THE NPU (hardware). The register stream differs only in job.mm.m and the
 *    addresses (charsiu_emit_job); the weights and coefficients are the staged
 *    slot's and are read unchanged. What the CNA/CMAC/CACC do with an input
 *    surface of height 1 against height w -- tiling, the order in which the
 *    1024 products of one slice are accumulated in fp32, any internal split
 *    of K -- is not visible in the source. THIS IS THE UNKNOWN, and the tool
 *    measures it directly: charsiu_npu_slot_word hands back the raw word each
 *    slot wrote for (r, c), so the "partials" columns compare the hardware's
 *    own output, before any CPU arithmetic, across w, r and neighbours.
 *    Which core ran a slot is fixed at staging (deal_pick over the running
 *    load); the "core copy" lines stage a second copy of the same weights,
 *    which the load balancer deals differently, and compare.
 *
 * 3. THE GATHER (CPU). m = 1 (charsiu_npu_matvec, charsiu_npu_matvec_group):
 *    memset the accumulator to 0, then for slots in index order i (K slice
 *    ascending within each n range) acc += P * s (scaled_add: a separately
 *    rounded multiply and add, the fmla is barred), or acc += P and a final
 *    (float)((double)acc * scale) for an ungrouped tensor.
 *    m > 1 (batch_read_device via read_rows / read_rows2 / read_fused_rows,
 *    then tail_scale_apply): for DEVICE 0 THEN DEVICE 1, and slots in index
 *    order within a device, the first contribution to an n range ASSIGNS and
 *    the rest add; an ungrouped tensor is then multiplied by scale[c] in
 *    float. The bmap read order, the pool split by rows and the row pair /
 *    quad forms only change which address an element comes from and which
 *    thread writes it -- no element is summed in a width dependent order.
 *    No fused multiply-add in any of these functions (objdump of an -O2
 *    -std=c11 build: zero fmadd/fmla in read_rows, read_rows2, read_rows4,
 *    read_fused_rows, scaled_add, tail_scale_rows, charsiu_npu_matvec).
 *    So, per tensor:
 *      - one K slice (Qwen3-0.6B: attn_q/k/v, ffn_gate/up, output, all K =
 *        1024): m = 1 and batched are the same arithmetic on the same P,
 *        except that 0.0f + (-0.0f) is +0 where an assign keeps -0. Any
 *        value difference here is the NPU's.
 *      - two K slices (attn_output, K = 2048): order is p0+p1 or p1+p0,
 *        which IEEE addition gives the same bits for (commutative).
 *      - three or more (ffn_down, K = 3072): if the deal puts the slices of
 *        one n range on the devices as {0,2 | 1} or {1,2 | 0}, the batched
 *        gather computes (p0+p2)+p1 or (p1+p2)+p0 where m = 1 computes
 *        (p0+p1)+p2 -- a real CPU-side difference, FIXED PER TENSOR (the
 *        deal does not move with w or r), so it cannot break row purity
 *        among batched widths, only batched against m = 1. The tool prints
 *        each tensor's deal and this prediction.
 *    The gather is the same for plain, deferred (npu_flush_pending calls the
 *    same batch_read_device) and grouped (npu_matmul_group_inner calls it per
 *    member, devices outer) calls.
 *
 * SO FROM THE SOURCE: (a) (b) (c) (e) can only move the bits through the NPU
 * itself; (d) moves which device's partial arrives first, which matters only
 * for >= 3 K slices and only against m = 1; and m = 1 vs batched can differ
 * in the NPU (height 1 vs height w) and, on >= 3 slice tensors with a split
 * deal, in the CPU gather order.
 *
 * ------------------------------------------------------------------------
 * WHAT A RESULT MEANS
 * ------------------------------------------------------------------------
 *
 *   batched vs batched-ref all SAME, vs m1 DIFFER
 *       Row pure across width, row, neighbours, grouping and deferral; only
 *       the m = 1 path is different arithmetic. Running plain decode steps
 *       through the batched kernel (width 2 with a pad row) whenever spec is
 *       on makes spec bit exact. If additionally "regather in m1 order"
 *       equals m1 on a tensor, that tensor's m = 1 difference is the CPU
 *       gather order alone (fixable on the CPU, see 3 above); if it does
 *       not, the NPU's partials themselves differ between height 1 and
 *       height w.
 *   batched vs batched-ref DIFFER, partials DIFFER the same way
 *       The NPU's own output depends on w / r / neighbours. The breakdown
 *       lines say which; no CPU change can fix that, and the exact design
 *       has to run every row of a position at one fixed width and row.
 *   batched vs batched-ref DIFFER, partials SAME
 *       The CPU gather is width or row dependent after all, i.e. the
 *       analysis above is wrong somewhere: read the breakdown.
 *   everything SAME including vs m1
 *       The matmul is not the cause; look at the batched attention / KV
 *       write in llama_verify_batch instead.
 *
 * On a machine with no NPU (CHARSIU_NPU=0, e.g. the aarch64 desk) every arm
 * runs the CPU path llama.c falls back to -- the quantised matvec a row at a
 * time -- so everything must be SAME; that is the harness control, and the
 * "pack" lines are a real test of the CPU half of the NPU path. A positive
 * control (two different targets compared) must say DIFFER on both.
 *
 *   npu_rowpure MODEL.gguf [--layer L] [--targets R] [--no-copies]
 *
 * One line per comparison and a VERDICT block at the end. Exit status 0 when
 * the probe ran, whatever it found; 1 when it could not run.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "charsiu.h"
#include "charsiu_llm.h"

#define MAXR   8      /* targets per input */
#define MAXW   6
#define MAXM   3      /* members of a unit: q k v */
#define MAXS   64     /* slots of one tensor we track partials for */

static const unsigned WIDTHS[] = { 2, 4, 6 };
#define NW (sizeof(WIDTHS) / sizeof(WIDTHS[0]))

enum { ARM_PLAIN, ARM_DEFER, ARM_GROUP, NARM };
static const char *ARMN[NARM] = { "plain", "defer", "group" };
enum { MODE_ROT, MODE_RAND, MODE_SELF, MODE_ZERO, NMODE };
static const char *MODEN[NMODE] = { "rot", "rand", "self", "zero" };

static double now_ms(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

/* ---- deterministic inputs --------------------------------------------- */

static uint64_t rng_s = 0x9e3779b97f4a7c15ull;

static void rng_seed(uint64_t s)
{
	rng_s = s ? s : 1;
}

static uint64_t rng_u64(void)
{
	rng_s ^= rng_s >> 12;
	rng_s ^= rng_s << 25;
	rng_s ^= rng_s >> 27;
	return rng_s * 2685821657736338717ull;
}

static float rng_unit(void)            /* [-1, 1) */
{
	return (float)((double)(rng_u64() >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
}

/* roughly normal, with the occasional outlier channel real activations have */
static void rng_row(float *x, uint64_t k)
{
	for (uint64_t i = 0; i < k; i++) {
		float v = (rng_unit() + rng_unit() + rng_unit()) * 0.6f;

		if ((rng_u64() & 255) == 0)
			v *= 12.0f;
		x[i] = v;
	}
}

static void rmsnorm_gain(float *x, uint64_t k, const struct gguf_tensor *g)
{
	double ss = 0.0;
	float *gw = NULL;

	for (uint64_t i = 0; i < k; i++)
		ss += (double)x[i] * x[i];
	float sc = (float)(1.0 / sqrt(ss / (double)k + 1e-6));

	if (g && g->ne[0] == k) {
		gw = malloc(k * sizeof(*gw));
		if (gw)
			gguf_row_f32(g, 0, gw);
	}
	for (uint64_t i = 0; i < k; i++)
		x[i] = x[i] * sc * (gw ? gw[i] : 1.0f);
	free(gw);
}

static uint64_t fnv(uint64_t h, const void *p, size_t n)
{
	const uint8_t *b = p;

	for (size_t i = 0; i < n; i++) {
		h ^= b[i];
		h *= 0x100000001b3ull;
	}
	return h;
}

/* ---- comparison ------------------------------------------------------- */

struct cmp {
	unsigned long rows, rows_eq, fl_bits, fl_val;
	double maxabs, maxrel;
};

static void cmp_add(struct cmp *c, const float *got, const float *want,
		    uint64_t n)
{
	int same = 1;

	c->rows++;
	if (!memcmp(got, want, n * sizeof(*got))) {
		c->rows_eq++;
		return;
	}
	for (uint64_t j = 0; j < n; j++) {
		uint32_t a, b;

		memcpy(&a, &got[j], 4);
		memcpy(&b, &want[j], 4);
		if (a == b)
			continue;
		c->fl_bits++;
		if (got[j] != want[j]) {
			double d = fabs((double)got[j] - (double)want[j]);
			double s = fabs((double)want[j]);
			double rel = d / (s > 1e-6 ? s : 1e-6);

			c->fl_val++;
			same = 0;
			if (d > c->maxabs)
				c->maxabs = d;
			if (rel > c->maxrel)
				c->maxrel = rel;
		}
	}
	(void)same;
}

static int cmp_same(const struct cmp *c)
{
	return c->rows && c->rows == c->rows_eq;
}

static void cmp_print(const char *what, const struct cmp *c)
{
	if (!c->rows) {
		printf("%s: not run", what);
		return;
	}
	printf("%s: rows bit-equal %lu/%lu, floats differ %lu (by value %lu), "
	       "max abs %.3g, max rel %.3g  %s", what, c->rows_eq, c->rows,
	       c->fl_bits, c->fl_val, c->maxabs, c->maxrel,
	       cmp_same(c) ? "SAME" : c->fl_val ? "DIFFER"
			   : "DIFFER(sign of zero only)");
}

/* ---- the units: tensors that share one input -------------------------- */

struct member {
	const struct gguf_tensor *w;
	const struct npu_tensor *nt;
	int id;                         /* NPU entry, or -1 */
	uint64_t n;
	float *ref;                     /* [R][n]  m = 1, single tensor path */
	float *bref;                    /* [R][n]  first batched result */
	int have_bref[MAXR];
	/* the hardware's partials: a hash per (target, slot), first seen */
	unsigned nslot;
	unsigned sdi[MAXS], ski[MAXS], sn0[MAXS], sn1[MAXS];
	uint64_t phash[MAXR][MAXS];
	int have_ph[MAXR][MAXS];
	struct cmp vs_m1[NARM][NW], vs_bref[NARM][NW];
	struct cmp rec_m1[NW], y_rec[NW];
	unsigned long ph_rows[NW], ph_eq[NW];
	unsigned long by_r_bad[MAXW], by_r_all[MAXW];
	unsigned long by_mode_bad[NMODE], by_mode_all[NMODE];
	int refused[NARM][NW];
	struct cmp rep, grp, pos;       /* m1 again, m1 group, control */
};

struct unit {
	const char *label;
	unsigned nm;
	struct member mb[MAXM];
	uint64_t k;
	float *X;                       /* [R][k] targets */
	unsigned R;
	int head;
};

static struct llama_state *S;
static struct charsiu_npu *G;           /* NULL on the CPU path */

static int pool_id_of(const struct npu_tensor *nt)
{
	for (unsigned i = 0; i < S->pool.n; i++)
		if (&S->pool.t[i] == nt)
			return S->pool.id[i];
	return -1;
}

/* one row through one tensor the way decode does it; 0 or -1 */
static int m1(struct member *mb, const float *x, float *y)
{
	struct charsiu_act *a = &S->act;

	charsiu_act_set(a, x, (int)mb->nt->k);
	if (G && mb->id >= 0) {
		if (charsiu_npu_needs_q1(G))
			charsiu_act_q1(a);
		if (charsiu_npu_matvec(G, mb->id, a, y))
			return -1;
		npu_quantise_output((struct npu_tensor *)mb->nt, y, mb->nt->n,
				    npu_out8_mode(),
				    a->q1_valid ? a->d1 : 0.0f);
		return 0;
	}
	/* llama.c's CPU fallback for a quantised copy: matvec_again */
	charsiu_act_q1(a);
	npu_matvec(mb->nt, a, y, 0, mb->nt->n);
	npu_quantise_output((struct npu_tensor *)mb->nt, y, mb->nt->n,
			    npu_out8_mode(), a->q1_valid ? a->d1 : 0.0f);
	return 0;
}

/* the decode path for q/k/v and gate/up: one group submit */
static int m1_group(struct unit *u, const float *x, float **ys)
{
	int ids[MAXM];

	if (!G)
		return -1;
	for (unsigned i = 0; i < u->nm; i++) {
		ids[i] = u->mb[i].id;
		if (ids[i] < 0)
			return -1;
	}
	charsiu_act_set(&S->act, x, (int)u->k);
	if (charsiu_npu_needs_q1(G))
		charsiu_act_q1(&S->act);
	return charsiu_npu_matvec_group(G, ids, u->nm, &S->act, ys);
}

/* w rows through every member of the unit along one arm; 0 or -1 */
static int batched(struct unit *u, int arm, const float *X, unsigned w,
		   float **Y)
{
	if (!G) {
		/* the CPU loop matmul_rows_x falls back to */
		for (unsigned i = 0; i < u->nm; i++)
			for (unsigned r = 0; r < w; r++)
				m1(&u->mb[i], X + (size_t)r * u->k,
				   Y[i] + (size_t)r * u->mb[i].n);
		return 0;
	}
	for (unsigned i = 0; i < u->nm; i++)
		if (u->mb[i].id < 0)
			return -1;
	switch (arm) {
	case ARM_PLAIN:
		for (unsigned i = 0; i < u->nm; i++)
			if (charsiu_npu_matmul(G, u->mb[i].id, X, w, Y[i]))
				return -1;
		return 0;
	case ARM_DEFER:
		/* what llama_verify_batch's matmul_rows_split does */
		for (unsigned i = 0; i < u->nm; i++) {
			int rc = i == 0
			       ? charsiu_npu_matmul_defer(G, u->mb[i].id, X, w, Y[i])
			       : charsiu_npu_matmul_same_defer(G, u->mb[i].id, X,
							       w, Y[i]);
			if (rc) {
				charsiu_npu_flush(G);
				return -1;
			}
		}
		return charsiu_npu_flush(G) ? -1 : 0;
	case ARM_GROUP: {
		int ids[MAXM];

		if (u->nm < 2)
			return -1;
		for (unsigned i = 0; i < u->nm; i++)
			ids[i] = u->mb[i].id;
		return charsiu_npu_matmul_group(G, ids, u->nm, X, w, Y) ? -1 : 0;
	}
	}
	return -1;
}

static void member_slots(struct member *mb)
{
	unsigned di, ki, n0, n1;

	mb->nslot = 0;
	if (!G || mb->id < 0)
		return;
	while (mb->nslot < MAXS &&
	       !charsiu_npu_slot_deal(G, mb->id, mb->nslot, &di, &ki, &n0, &n1)) {
		mb->sdi[mb->nslot] = di;
		mb->ski[mb->nslot] = ki;
		mb->sn0[mb->nslot] = n0;
		mb->sn1[mb->nslot] = n1;
		mb->nslot++;
	}
}

/*
 * What the source predicts for the CPU gather order of this tensor: m = 1 sums
 * an n range's slices in slot order, batched in device order then slot order.
 */
static const char *order_prediction(const struct member *mb, char *buf,
				    size_t max)
{
	unsigned worst = 0;         /* 0 same, 1 swap only, 2 reordered */
	unsigned maxks = 0;

	for (unsigned i = 0; i < mb->nslot; i++) {
		unsigned m1o[MAXS], bo[MAXS], c = 0, cb = 0;

		if (mb->ski[i] != 0)
			continue;    /* one pass per n range, from its ki = 0 slot */
		for (unsigned j = 0; j < mb->nslot; j++)
			if (mb->sn0[j] == mb->sn0[i])
				m1o[c++] = mb->ski[j];
		for (unsigned d = 0; d < 2; d++)
			for (unsigned j = 0; j < mb->nslot; j++)
				if (mb->sn0[j] == mb->sn0[i] && mb->sdi[j] == d)
					bo[cb++] = mb->ski[j];
		if (c > maxks)
			maxks = c;
		if (c != cb)
			continue;
		if (memcmp(m1o, bo, c * sizeof(*m1o))) {
			unsigned lvl = c <= 2 ? 1 : 2;

			/* the first two swapped and the rest in place is
			 * still (a+b)+c == (b+a)+c */
			if (c > 2 && m1o[0] == bo[1] && m1o[1] == bo[0] &&
			    !memcmp(m1o + 2, bo + 2, (c - 2) * sizeof(*m1o)))
				lvl = 1;
			if (lvl > worst)
				worst = lvl;
		}
	}
	snprintf(buf, max, "%u K slice%s a range; %s", maxks,
		 maxks == 1 ? "" : "s",
		 maxks <= 1 ? "m1 and batched gather are the same arithmetic "
			      "(modulo the sign of an exact zero)"
		 : worst == 0 ? "same summation order in m1 and batched"
		 : worst == 1 ? "order swapped but commutative: same bits "
				"(modulo the sign of an exact zero)"
		 : "batched sums the slices in a DIFFERENT order than m1: "
		   "a CPU-side difference against m1 is expected here");
	return buf;
}

static void print_deal(const struct member *mb)
{
	printf("    deal:");
	for (unsigned i = 0; i < mb->nslot && i < 24; i++)
		printf(" k%u n[%u,%u)@d%u", mb->ski[i], mb->sn0[i], mb->sn1[i],
		       mb->sdi[i]);
	if (mb->nslot > 24)
		printf(" ... (%u slots)", mb->nslot);
	printf("\n");
}

/*
 * The hardware's own words for row r, read back through slot_word, and the
 * same words summed the way the m = 1 path sums them. Only valid right after
 * a batched call of THIS tensor.
 */
static int partials(struct member *mb, unsigned t, unsigned r, unsigned wi,
		    float *rec, const float *Yrow, const float *ref_t)
{
	int ok = 1;
	float fin = 1.0f;

	if (!G || !mb->nslot)
		return -1;
	memset(rec, 0, mb->n * sizeof(*rec));
	for (unsigned i = 0; i < mb->nslot; i++) {
		uint64_t h = 0xcbf29ce484222325ull;

		for (unsigned c = mb->sn0[i]; c < mb->sn1[i]; c++) {
			uint32_t raw;
			float contrib, final;

			if (charsiu_npu_slot_word(G, mb->id, i, r, c, 0, &raw,
						  &contrib, &final))
				return -1;
			h = fnv(h, &raw, 4);
			rec[c] += contrib;          /* 0 + p0 + p1 ..., slot order */
			fin = final;
		}
		if (t < MAXR && i < MAXS) {
			if (!mb->have_ph[t][i]) {
				mb->phash[t][i] = h;
				mb->have_ph[t][i] = 1;
			} else if (mb->phash[t][i] != h) {
				ok = 0;
			}
		}
	}
	/* an ungrouped tensor: m1 does (float)((double)acc * scale[c]);
	 * the same test as npudev's tensor_grouped */
	(void)fin;
	if (!(mb->nt->kgroup && mb->nt->kgroup < mb->nt->k &&
	      mb->nt->k % mb->nt->kgroup == 0 &&
	      mb->nt->kgroup <= (uint64_t)charsiu_npu_kmax(G)))
		for (uint64_t c = 0; c < mb->n; c++)
			rec[c] = (float)((double)rec[c] *
					 (double)mb->nt->scale[c]);
	mb->ph_rows[wi]++;
	if (ok)
		mb->ph_eq[wi]++;
	cmp_add(&mb->rec_m1[wi], rec, ref_t, mb->n);
	cmp_add(&mb->y_rec[wi], Yrow, rec, mb->n);
	return 0;
}

/* ---- the input pack, CPU only, desk and board ------------------------- */

static int pack_check(uint64_t k)
{
	unsigned atom = charsiu_feature_atom(CHARSIU_FP16);
	size_t sz1 = (size_t)k * 2 + 4096;
	uint8_t *p1 = malloc(sz1);
	float *X = malloc((size_t)MAXW * k * sizeof(*X));
	int all = 1;

	if (!p1 || !X) {
		free(p1);
		free(X);
		return -1;
	}
	for (unsigned wi = 0; wi < NW; wi++) {
		unsigned w = WIDTHS[wi];
		size_t szw = (size_t)w * k * 2 + 4096;
		uint8_t *pw = malloc(szw), *pg = malloc(szw);
		unsigned long rows = 0, eq = 0, geq = 0;

		if (!pw || !pg) {
			free(pw);
			free(pg);
			all = 0;
			continue;
		}
		for (unsigned trial = 0; trial < 4; trial++) {
			struct charsiu_matmul mm = { w, (unsigned)k, 16,
						     CHARSIU_INT4, CHARSIU_FP16 };
			struct charsiu_matmul m1m = { 1, (unsigned)k, 16,
						      CHARSIU_INT4, CHARSIU_FP16 };
			unsigned ng;

			rng_seed(0x5eed0000u + trial * 97 + w);
			for (unsigned r = 0; r < w; r++)
				rng_row(X + (size_t)r * k, k);
			charsiu_pack_input_f16_stride(&mm, X, k, pw, szw);
			/* the pooled form pack_f16_pooled takes */
			ng = charsiu_pack_input_f16_ngroups(&mm);
			if (ng) {
				charsiu_pack_input_f16_edges(&mm, X, k, pg, szw);
				charsiu_pack_input_f16_groups(&mm, X, k, pg, 0, ng);
			} else {
				memcpy(pg, pw, szw);
			}
			for (unsigned r = 0; r < w; r++) {
				int same = 1, gsame;

				charsiu_pack_input_f16(&m1m, X + (size_t)r * k,
						       p1, sz1);
				for (uint64_t kk = 0; kk < k; kk++) {
					size_t e = (size_t)(kk / atom) * w * atom
						 + (size_t)r * atom + kk % atom;

					if (memcmp(pw + e * 2, p1 + kk * 2, 2)) {
						same = 0;
						break;
					}
				}
				gsame = !memcmp(pw, pg, (size_t)w * k * 2);
				rows++;
				eq += same;
				geq += gsame;
			}
		}
		printf("  pack k=%-5llu w=%u: row r of the width-w pack == the "
		       "m=1 pack: %lu/%lu rows, pooled form == serial form: "
		       "%lu/%lu  %s\n", (unsigned long long)k, w, eq, rows, geq,
		       rows, eq == rows && geq == rows ? "SAME" : "DIFFER");
		if (eq != rows || geq != rows)
			all = 0;
		free(pw);
		free(pg);
	}
	free(p1);
	free(X);
	return all ? 0 : 1;
}

/* ---- running one unit --------------------------------------------------- */

static void fill_targets(struct unit *u, const struct llama_model *m,
			 const struct gguf_tensor *gain, const float *real,
			 unsigned nreal)
{
	static const int32_t TOK[] = { 9707, 1000, 279, 13, 30000, 220, 42, 151 };

	for (unsigned t = 0; t < u->R; t++) {
		float *x = u->X + (size_t)t * u->k;

		if (real && t < nreal) {
			memcpy(x, real + (size_t)t * u->k, u->k * sizeof(*x));
		} else if (!real && t < u->R / 2 && m->tok_embd &&
			   m->tok_embd->ne[0] == u->k) {
			gguf_row_f32(m->tok_embd,
				     (uint64_t)TOK[t] % m->tok_embd->ne[1], x);
			rmsnorm_gain(x, u->k, gain);
		} else {
			rng_seed(0xabcdef12345ull + t * 7919 + u->k);
			rng_row(x, u->k);
		}
	}
}

static unsigned long g_neigh;

static void neighbour(float *x, uint64_t k, int zero)
{
	if (zero) {
		memset(x, 0, k * sizeof(*x));
		return;
	}
	rng_seed(0x77777777ull + (g_neigh++) * 104729);
	rng_row(x, k);
}

/* the target sitting in each row of a call; -1 for a neighbour */
static void run_unit(struct unit *u, int do_partials_all)
{
	unsigned R = u->R;
	float *Xb = malloc((size_t)MAXW * u->k * sizeof(*Xb));
	float *Yb[MAXM], *rec = NULL, *tmp[MAXM];
	uint64_t nmax = 0;

	for (unsigned i = 0; i < u->nm; i++)
		if (u->mb[i].n > nmax)
			nmax = u->mb[i].n;
	rec = malloc(nmax * sizeof(*rec));
	for (unsigned i = 0; i < u->nm; i++) {
		Yb[i] = malloc((size_t)MAXW * u->mb[i].n * sizeof(float));
		tmp[i] = malloc(u->mb[i].n * sizeof(float));
	}
	if (!Xb || !rec)
		goto out;

	/* m = 1, twice, and the group form */
	{
		struct cmp rep[MAXM], grp[MAXM], pos[MAXM];
		int grp_ran = 0, refused = 0;

		memset(rep, 0, sizeof(rep));
		memset(grp, 0, sizeof(grp));
		memset(pos, 0, sizeof(pos));
		for (unsigned t = 0; t < R; t++)
			for (unsigned i = 0; i < u->nm; i++)
				if (m1(&u->mb[i], u->X + (size_t)t * u->k,
				       u->mb[i].ref + (size_t)t * u->mb[i].n))
					refused = 1;
		if (refused)
			printf("    m1: the single tensor matvec REFUSED a call; "
			       "nothing below is meaningful for this unit\n");
		for (unsigned t = 0; t < R; t++) {
			for (unsigned i = 0; i < u->nm; i++) {
				m1(&u->mb[i], u->X + (size_t)t * u->k, tmp[i]);
				cmp_add(&rep[i], tmp[i],
					u->mb[i].ref + (size_t)t * u->mb[i].n,
					u->mb[i].n);
			}
			if (u->nm >= 2 && !m1_group(u, u->X + (size_t)t * u->k,
						     tmp)) {
				grp_ran = 1;
				for (unsigned i = 0; i < u->nm; i++)
					cmp_add(&grp[i], tmp[i],
						u->mb[i].ref + (size_t)t * u->mb[i].n,
						u->mb[i].n);
			}
		}
		/* positive control: target t against target t+1 */
		for (unsigned t = 0; t + 1 < R; t++)
			for (unsigned i = 0; i < u->nm; i++)
				cmp_add(&pos[i],
					u->mb[i].ref + (size_t)(t + 1) * u->mb[i].n,
					u->mb[i].ref + (size_t)t * u->mb[i].n,
					u->mb[i].n);
		for (unsigned i = 0; i < u->nm; i++) {
			u->mb[i].rep = rep[i];
			u->mb[i].grp = grp[i];
			u->mb[i].pos = pos[i];
			printf("    %-22s ", u->mb[i].w->name);
			cmp_print("m1 again vs m1", &rep[i]);
			printf("\n");
			if (u->nm >= 2) {
				printf("    %-22s ", u->mb[i].w->name);
				if (grp_ran)
					cmp_print("m1 group (decode) vs m1",
						  &grp[i]);
				else
					printf("m1 group (decode): not run%s",
					       G ? " (refused)" : " (no NPU)");
				printf("\n");
			}
			printf("    %-22s ", u->mb[i].w->name);
			cmp_print("control: target t+1 vs t (must DIFFER)",
				  &pos[i]);
			printf("\n");
		}
	}

	/* the batched arms */
	for (int arm = 0; arm < NARM; arm++) {
		if (arm == ARM_GROUP && u->nm < 2)
			continue;
		for (unsigned wi = 0; wi < NW; wi++) {
			unsigned w = WIDTHS[wi];

			for (int mode = 0; mode < NMODE; mode++) {
				unsigned ncall = R;

				for (unsigned call = 0; call < ncall; call++) {
					int who[MAXW];

					for (unsigned r = 0; r < w; r++) {
						float *x = Xb + (size_t)r * u->k;
						unsigned t;

						who[r] = -1;
						switch (mode) {
						case MODE_ROT:
							t = (call + r) % R;
							who[r] = (int)t;
							break;
						case MODE_SELF:
							who[r] = (int)call;
							break;
						case MODE_RAND:
						case MODE_ZERO:
							if (r == (call + (mode == MODE_ZERO)) % w)
								who[r] = (int)call;
							break;
						}
						if (who[r] >= 0)
							memcpy(x, u->X + (size_t)who[r] * u->k,
							       u->k * sizeof(*x));
						else
							neighbour(x, u->k,
								  mode == MODE_ZERO);
					}
					/* plain arm: one member at a time, so the
					 * partials of each can be read before the
					 * next call replaces them */
					if (G && arm == ARM_PLAIN) {
						int bad = 0;

						for (unsigned i = 0; i < u->nm && !bad; i++) {
							struct member *mb = &u->mb[i];

							if (mb->id < 0 ||
							    charsiu_npu_matmul(G, mb->id, Xb, w, Yb[i])) {
								bad = 1;
								break;
							}
							if (!do_partials_all && mode != MODE_ROT)
								continue;
							for (unsigned r = 0; r < w; r++)
								if (who[r] >= 0)
									partials(mb, (unsigned)who[r], r, wi, rec,
										 Yb[i] + (size_t)r * mb->n,
										 mb->ref + (size_t)who[r] * mb->n);
						}
						if (bad) {
							for (unsigned i = 0; i < u->nm; i++)
								u->mb[i].refused[arm][wi] = 1;
							continue;
						}
					} else if (batched(u, arm, Xb, w, Yb)) {
						for (unsigned i = 0; i < u->nm; i++)
							u->mb[i].refused[arm][wi] = 1;
						continue;
					}
					for (unsigned i = 0; i < u->nm; i++) {
						struct member *mb = &u->mb[i];

						for (unsigned r = 0; r < w; r++) {
							int t = who[r];
							const float *y = Yb[i] + (size_t)r * mb->n;
							unsigned long before;

							if (t < 0)
								continue;
							cmp_add(&mb->vs_m1[arm][wi], y,
								mb->ref + (size_t)t * mb->n, mb->n);
							if (!mb->have_bref[t]) {
								memcpy(mb->bref + (size_t)t * mb->n,
								       y, mb->n * sizeof(*y));
								mb->have_bref[t] = 1;
							}
							before = mb->vs_bref[arm][wi].rows_eq;
							cmp_add(&mb->vs_bref[arm][wi], y,
								mb->bref + (size_t)t * mb->n, mb->n);
							mb->by_r_all[r]++;
							mb->by_mode_all[mode]++;
							if (mb->vs_bref[arm][wi].rows_eq == before) {
								mb->by_r_bad[r]++;
								mb->by_mode_bad[mode]++;
							}
						}
					}
				}
			}
		}
	}

	for (unsigned i = 0; i < u->nm; i++) {
		struct member *mb = &u->mb[i];

		for (int arm = 0; arm < NARM; arm++) {
			if (arm == ARM_GROUP && u->nm < 2)
				continue;
			for (unsigned wi = 0; wi < NW; wi++) {
				printf("    %-22s %-5s w=%u  ", mb->w->name,
				       ARMN[arm], WIDTHS[wi]);
				if (mb->refused[arm][wi] &&
				    !mb->vs_m1[arm][wi].rows) {
					printf("REFUSED by the batched path at this "
					       "width -- not a result\n");
					continue;
				}
				if (mb->refused[arm][wi])
					printf("(some calls REFUSED) ");
				cmp_print("vs m1", &mb->vs_m1[arm][wi]);
				printf(" | ");
				cmp_print("vs batched-ref", &mb->vs_bref[arm][wi]);
				printf("\n");
				if (arm == ARM_PLAIN && G && mb->ph_rows[wi]) {
					printf("    %-22s plain w=%u  NPU partials "
					       "(raw words a slot wrote) equal "
					       "to the first seen: %lu/%lu rows  %s"
					       " | ", mb->w->name, WIDTHS[wi],
					       mb->ph_eq[wi], mb->ph_rows[wi],
					       mb->ph_eq[wi] == mb->ph_rows[wi]
					       ? "SAME" : "DIFFER");
					cmp_print("regather in m1 order vs m1",
						  &mb->rec_m1[wi]);
					printf(" | ");
					cmp_print("gathered Y vs regather",
						  &mb->y_rec[wi]);
					printf("\n");
				}
			}
		}
		{
			int any = 0;

			for (unsigned r = 0; r < MAXW; r++)
				if (mb->by_r_bad[r])
					any = 1;
			if (any) {
				printf("    %-22s rows unlike batched-ref by row "
				       "index:", mb->w->name);
				for (unsigned r = 0; r < MAXW; r++)
					printf(" r%u %lu/%lu", r, mb->by_r_bad[r],
					       mb->by_r_all[r]);
				printf("; by neighbours:");
				for (int md = 0; md < NMODE; md++)
					printf(" %s %lu/%lu", MODEN[md],
					       mb->by_mode_bad[md],
					       mb->by_mode_all[md]);
				printf("\n");
			}
		}
		{
			uint64_t h1 = 0xcbf29ce484222325ull, h2 = h1;

			h1 = fnv(h1, mb->ref, (size_t)R * mb->n * sizeof(float));
			for (unsigned t = 0; t < R; t++)
				if (mb->have_bref[t])
					h2 = fnv(h2, mb->bref + (size_t)t * mb->n,
						 mb->n * sizeof(float));
			printf("    %-22s hash m1 %016llx batched-ref %016llx "
			       "(compare across runs: CHARSIU_NPU_ONEDEV=1, "
			       "CHARSIU_NPU_BATCH_PARALLEL=1)\n", mb->w->name,
			       (unsigned long long)h1, (unsigned long long)h2);
		}
	}
out:
	for (unsigned i = 0; i < u->nm; i++) {
		free(Yb[i]);
		free(tmp[i]);
	}
	free(rec);
	free(Xb);
}

/*
 * (d), WITHIN ONE RUN: the same weights staged again under a different
 * address, which the load balancer deals to the cores differently. Compares
 * the copy's m1 against the original's and the copy's plain w=2 against the
 * original's batched-ref.
 */
static int core_copy(struct unit *u, unsigned mi, int *tested, int *same)
{
	struct member *mb = &u->mb[mi];
	struct gguf_tensor *cp;
	struct member c;
	float *Y, *Xb;
	struct cmp a, b;

	*tested = 0;
	*same = 1;
	if (!G || mb->id < 0 || !mb->nslot)
		return 0;
	for (int attempt = 0; attempt < 3; attempt++) {
		int differs = 0;

		cp = malloc(sizeof(*cp));    /* kept: the pool keys on it */
		if (!cp)
			return -1;
		*cp = *mb->w;
		memset(&c, 0, sizeof(c));
		c.w = cp;
		c.nt = charsiu_pool_get(&S->pool, cp);
		if (!c.nt)
			return 0;
		c.id = pool_id_of(c.nt);
		c.n = mb->n;
		if (c.id < 0)
			return 0;
		member_slots(&c);
		for (unsigned i = 0; i < c.nslot && i < mb->nslot; i++)
			if (c.sdi[i] != mb->sdi[i])
				differs = 1;
		if (!differs)
			continue;
		*tested = 1;
		Y = malloc((size_t)2 * mb->n * sizeof(*Y));
		Xb = malloc((size_t)2 * u->k * sizeof(*Xb));
		if (!Y || !Xb) {
			free(Y);
			free(Xb);
			return -1;
		}
		memset(&a, 0, sizeof(a));
		memset(&b, 0, sizeof(b));
		for (unsigned t = 0; t < u->R; t++) {
			if (!m1(&c, u->X + (size_t)t * u->k, Y))
				cmp_add(&a, Y, mb->ref + (size_t)t * mb->n, mb->n);
			if (!mb->have_bref[t])
				continue;
			memcpy(Xb, u->X + (size_t)t * u->k, u->k * sizeof(*Xb));
			memcpy(Xb + u->k, u->X + (size_t)((t + 1) % u->R) * u->k,
			       u->k * sizeof(*Xb));
			if (!charsiu_npu_matmul(G, c.id, Xb, 2, Y))
				cmp_add(&b, Y, mb->bref + (size_t)t * mb->n,
					mb->n);
		}
		printf("    %-22s core copy, ", mb->w->name);
		print_deal(&c);
		printf("    %-22s ", mb->w->name);
		cmp_print("core copy m1 vs m1", &a);
		printf(" | ");
		cmp_print("core copy plain w=2 r0 vs batched-ref", &b);
		printf("\n");
		*same = cmp_same(&a) && cmp_same(&b);
		free(Y);
		free(Xb);
		return 0;
	}
	printf("    %-22s core copy: three copies all dealt like the original; "
	       "(d) not reached for this tensor\n", mb->w->name);
	return 0;
}

static int unit_init(struct unit *u, const char *label,
		     const struct gguf_tensor *const *ws, unsigned nw,
		     unsigned R)
{
	memset(u, 0, sizeof(*u));
	u->label = label;
	u->R = R;
	for (unsigned i = 0; i < nw; i++) {
		struct member *mb = &u->mb[u->nm];

		if (!ws[i])
			continue;
		mb->w = ws[i];
		mb->nt = charsiu_pool_get(&S->pool, ws[i]);
		if (!mb->nt) {
			printf("  %s: %s did not stage (type %s); skipped\n",
			       label, ws[i]->name, ggml_type_name(ws[i]->type));
			continue;
		}
		mb->id = pool_id_of(mb->nt);
		mb->n = mb->nt->n;
		mb->ref = calloc((size_t)R * mb->n, sizeof(float));
		mb->bref = calloc((size_t)R * mb->n, sizeof(float));
		if (!mb->ref || !mb->bref)
			return -1;
		if (u->nm && mb->nt->k != u->k) {
			printf("  %s: %s has a different K; skipped\n", label,
			       ws[i]->name);
			continue;
		}
		u->k = mb->nt->k;
		member_slots(mb);
		u->nm++;
	}
	if (!u->nm)
		return 1;
	u->X = malloc((size_t)R * u->k * sizeof(float));
	return u->X ? 0 : -1;
}

static void unit_header(const struct unit *u)
{
	char buf[256];

	printf("\n== %s  (K=%llu, %u target rows: the first half real-ish "
	       "activations, the rest pseudo-random)\n", u->label,
	       (unsigned long long)u->k, u->R);
	for (unsigned i = 0; i < u->nm; i++) {
		const struct member *mb = &u->mb[i];
		int grouped = mb->nt->kgroup && mb->nt->kgroup < mb->nt->k;

		printf("  %-22s k=%llu n=%llu %s, kgroup %llu (%s), %s\n",
		       mb->w->name, (unsigned long long)mb->nt->k,
		       (unsigned long long)mb->n, ggml_type_name(mb->w->type),
		       (unsigned long long)mb->nt->kgroup,
		       grouped ? "scale per slice in the gather"
			       : "one scale a row, applied after the sum",
		       mb->id >= 0 ? "on the NPU" : "on the CPU");
		if (G && mb->id >= 0) {
			print_deal(mb);
			printf("    CPU gather order: %s\n",
			       order_prediction(mb, buf, sizeof(buf)));
		}
	}
}

/* ---- verdict -------------------------------------------------------- */

static void verdict(struct unit *us, unsigned nu, int pack_bad,
		    int core_tested, int core_bad)
{
	int m1_det = 1, grp_eq = 1, b_pure = 1, b_eq_m1 = 1, ph_ok = 1,
	    any_b = 0, pos_ok = 1;
	char diff_m1[1024] = "", cpu_only[1024] = "", hw_m1[1024] = "";
	unsigned long by_w_bad[NW] = { 0 };
	unsigned long by_arm_bad[NARM] = { 0 };

	for (unsigned ui = 0; ui < nu; ui++)
		for (unsigned i = 0; i < us[ui].nm; i++) {
			struct member *mb = &us[ui].mb[i];
			int this_m1 = 1, rec_all = 1, rec_any = 0;

			if (!cmp_same(&mb->rep))
				m1_det = 0;
			if (mb->grp.rows && !cmp_same(&mb->grp))
				grp_eq = 0;
			/* every pair of distinct targets must differ */
			if (mb->pos.rows_eq)
				pos_ok = 0;

			for (int arm = 0; arm < NARM; arm++)
				for (unsigned wi = 0; wi < NW; wi++) {
					struct cmp *a = &mb->vs_m1[arm][wi];
					struct cmp *b = &mb->vs_bref[arm][wi];

					if (!a->rows)
						continue;
					any_b = 1;
					if (!cmp_same(a))
						this_m1 = 0;
					if (!cmp_same(b)) {
						b_pure = 0;
						by_w_bad[wi] += b->rows - b->rows_eq;
						by_arm_bad[arm] += b->rows - b->rows_eq;
					}
				}
			for (unsigned wi = 0; wi < NW; wi++) {
				if (mb->ph_rows[wi] && mb->ph_eq[wi] != mb->ph_rows[wi])
					ph_ok = 0;
				if (mb->rec_m1[wi].rows) {
					rec_any = 1;
					if (!cmp_same(&mb->rec_m1[wi]))
						rec_all = 0;
				}
			}
			if (!this_m1) {
				b_eq_m1 = 0;
				snprintf(diff_m1 + strlen(diff_m1),
					 sizeof(diff_m1) - strlen(diff_m1), " %s",
					 mb->w->name);
				if (rec_any && rec_all)
					snprintf(cpu_only + strlen(cpu_only),
						 sizeof(cpu_only) - strlen(cpu_only),
						 " %s", mb->w->name);
				else if (rec_any)
					snprintf(hw_m1 + strlen(hw_m1),
						 sizeof(hw_m1) - strlen(hw_m1),
						 " %s", mb->w->name);
			}
		}
	printf("\nVERDICT (%s)\n", G ? "NPU path" : "CPU path, the harness "
	       "control: everything must be SAME");
	printf("  positive control (different targets compare DIFFER): %s\n",
	       pos_ok ? "yes" : "NO -- the comparator cannot be trusted");
	printf("  m1 deterministic (same row twice, same bits): %s\n",
	       m1_det ? "yes" : "NO");
	if (G)
		printf("  m1 group (the decode path for q/k/v, gate/up) == m1 "
		       "single: %s\n", grp_eq ? "yes" : "NO");
	printf("  input pack row-pure (CPU half of the NPU path): %s\n",
	       pack_bad ? "NO" : "yes");
	printf("  batched rows a pure function of the row, across width, row "
	       "index, neighbours, plain/defer/group: %s\n",
	       !any_b ? "NOT RUN" : b_pure ? "YES" : "NO");
	if (!b_pure) {
		printf("    rows unlike batched-ref by width:");
		for (unsigned wi = 0; wi < NW; wi++)
			printf(" w%u %lu", WIDTHS[wi], by_w_bad[wi]);
		printf("; by arm:");
		for (int a = 0; a < NARM; a++)
			printf(" %s %lu", ARMN[a], by_arm_bad[a]);
		printf(" (row index and neighbours: the per tensor lines)\n");
	}
	if (G)
		printf("  NPU partials (raw words) stable across width, row, "
		       "neighbours: %s\n", ph_ok ? "YES" : "NO");
	printf("  core (d) within the run: %s\n",
	       !G ? "n/a" : !core_tested ? "not reached (no copy dealt "
	       "differently)" : core_bad ? "DIFFERS" : "SAME on every copy tested");
	printf("  batched == m1 bitwise: %s%s%s\n", b_eq_m1 ? "YES" : "NO,",
	       b_eq_m1 ? "" : diff_m1, "");
	if (!b_eq_m1 && G) {
		if (*cpu_only)
			printf("    m1 difference that is the CPU gather order "
			       "ONLY (hardware partials re-summed in m1 order "
			       "equal m1):%s\n", cpu_only);
		if (*hw_m1)
			printf("    m1 difference in the NPU's own partials "
			       "(height 1 vs height w):%s\n", hw_m1);
	}
	printf("  ==> ");
	if (!any_b)
		printf("nothing batched ran; no conclusion\n");
	else if (!G)
		printf("%s\n", b_pure && b_eq_m1 && !pack_bad && pos_ok &&
		       m1_det ? "harness control PASSES (CPU path is row pure, as it "
			 "must be)"
		       : "harness control FAILS: the probe itself is wrong");
	else if (b_pure && b_eq_m1)
		printf("the matmul is row pure AND equal to m1: the spec "
		       "difference is not the projections -- look at the "
		       "batched attention / KV write\n");
	else if (b_pure && !core_bad)
		printf("ROW PURE; only m1 differs. Exact fix: when spec is on, "
		       "run plain decode steps through the batched kernel "
		       "(width 2 with a pad row)%s\n",
		       *hw_m1 ? "" : *cpu_only ? " -- or, since the m1 "
		       "difference is gather order only, sum the batched "
		       "gather in K slice order" : "");
	else if (b_pure)
		printf("row pure across width and row, but the CORE changes the "
		       "bits: the exact design must also pin the deal\n");
	else
		printf("NOT row pure (%s): one position must always be "
		       "computed at one fixed width and row; see the "
		       "breakdown\n", ph_ok ? "partials stable, so the CPU "
		       "gather" : "the NPU's own partials move");
}

int main(int argc, char **argv)
{
	struct llama_model m;
	const char *path = NULL;
	int layer = 13, copies = 1;
	unsigned R = 6;
	struct unit us[5];
	unsigned nu = 0;
	int pack_bad = 0, core_tested = 0, core_bad = 0;
	double t0;
	char lname[96];

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--layer") && i + 1 < argc)
			layer = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--targets") && i + 1 < argc)
			R = (unsigned)atoi(argv[++i]);
		else if (!strcmp(argv[i], "--no-copies"))
			copies = 0;
		else if (argv[i][0] == '-') {
			fprintf(stderr, "npu_rowpure: unknown option %s\n",
				argv[i]);
			return 1;
		} else
			path = argv[i];
	}
	if (!path) {
		fprintf(stderr, "usage: npu_rowpure MODEL.gguf [--layer L] "
			"[--targets R] [--no-copies]\n"
			"  board: CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 "
			"CHARSIU_NPU_W4V=1\n"
			"  desk:  CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1 "
			"CHARSIU_NPU_W4_GROUP=1024\n");
		return 1;
	}
	if (R < 2)
		R = 2;
	if (R > MAXR)
		R = MAXR;
	if (!getenv("CHARSIU_NPU_QUANT"))
		fprintf(stderr, "npu_rowpure: CHARSIU_NPU_QUANT is unset; the "
			"quantised copies this compares need it\n");

	t0 = now_ms();
	if (llama_load(&m, path) < 0)
		return 1;
	S = llama_state_new(&m, 64);
	if (!S)
		return 1;
	G = S->pool.dev;
	if (layer < 0 || (unsigned)layer >= m.n_layer)
		layer = (int)m.n_layer / 2;

	printf("npu_rowpure %s, model %s\n", CHARSIU_BUILD, path);
	printf("  path: %s\n", G ? "NPU (the device opened; batched arms are "
	       "charsiu_npu_matmul*, m1 is charsiu_npu_matvec)"
	       : "CPU (no NPU device; every arm is the per row quantised "
		 "matvec llama.c falls back to)");
#define ENV(x) (getenv(x) ? getenv(x) : "-")
	printf("  env: CHARSIU_NPU=%s QUANT=%s W4V=%s KMAX=%s W4_GROUP=%s "
	       "ONEDEV=%s BATCH_PARALLEL=%s READ_FUSE=%s READ4=%s "
	       "DEFER_READ=%s CPU_FRAC=%s\n",
	       ENV("CHARSIU_NPU"), ENV("CHARSIU_NPU_QUANT"),
	       ENV("CHARSIU_NPU_W4V"), ENV("CHARSIU_NPU_KMAX"),
	       ENV("CHARSIU_NPU_W4_GROUP"), ENV("CHARSIU_NPU_ONEDEV"),
	       ENV("CHARSIU_NPU_BATCH_PARALLEL"), ENV("CHARSIU_NPU_READ_FUSE"),
	       ENV("CHARSIU_NPU_READ4"), ENV("CHARSIU_NPU_DEFER_READ"),
	       ENV("CHARSIU_NPU_CPU_FRAC"));
	if (G) {
		printf("  NPU: %s, batches %s, kmax %u\n",
		       charsiu_npu_needs_q1(G) ? "int8 (W4V off: NOT the path "
		       "this question is about)" : "int4 w4a16",
		       charsiu_npu_batches(G) ? "yes" : "no",
		       charsiu_npu_kmax(G));
		printf("  overlap: %s\n", charsiu_npu_overlap_note());
	}

	/* the order the first forward pass stages in, so the deal is the
	 * deal charsiu_run gets */
	if (G) {
		unsigned on = 0;

		for (unsigned l = 0; l < m.n_layer; l++) {
			const struct llama_layer *L = &m.layers[l];
			const struct gguf_tensor *w[7] = { L->wq, L->wk, L->wv,
							   L->wo, L->gate, L->up,
							   L->down };

			for (unsigned j = 0; j < 7; j++)
				if (w[j])
					charsiu_pool_get(&S->pool, w[j]);
		}
		if (m.output)
			charsiu_pool_get(&S->pool, m.output);
		for (unsigned i = 0; i < S->pool.n; i++)
			on += S->pool.id[i] >= 0;
		printf("  staged %u tensors in forward order, %u on the NPU, "
		       "%.0f ms since start\n", S->pool.n, on, now_ms() - t0);
	}
	printf("  widths 2 4 6; neighbours: rot (the other targets), rand, "
	       "self (every row the target), zero; arms plain, defer (what "
	       "the verify pass calls), group\n");

	/* 1. the pack */
	printf("\n== input pack (CPU code, runs identically on desk and "
	       "board)\n");
	{
		unsigned sk = G ? charsiu_npu_kmax(G) : 1024;

		if (pack_check(sk) || pack_check(m.n_embd) || pack_check(96))
			pack_bad = 1;
	}

	/* 2. the units */
	{
		const struct llama_layer *L0 = &m.layers[0];
		const struct llama_layer *LL = &m.layers[layer];
		const struct gguf_tensor *qkv[3] = { L0->wq, L0->wk, L0->wv };
		const struct gguf_tensor *gu[2] = { L0->gate, L0->up };
		const struct gguf_tensor *o[1] = { L0->wo };
		const struct gguf_tensor *dn[1] = { LL->down };
		const struct gguf_tensor *hd[1] = { m.output };
		struct unit *uq = NULL, *ug = NULL;

		if (!unit_init(&us[nu], "blk.0 q/k/v (one RMSNorm output)", qkv,
			       3, R)) {
			uq = &us[nu];
			fill_targets(uq, &m, L0->attn_norm, NULL, 0);
			nu++;
		}
		if (!unit_init(&us[nu], "blk.0 gate/up", gu, 2, R)) {
			ug = &us[nu];
			fill_targets(ug, &m, L0->ffn_norm, NULL, 0);
			nu++;
		}
		if (!unit_init(&us[nu], "blk.0 attn_output", o, 1, R)) {
			struct unit *u = &us[nu];
			float *real = NULL;
			unsigned nreal = 0;

			/* real-ish: the q projection of the real-ish rows */
			if (uq && uq->mb[0].n == u->k) {
				real = malloc((size_t)R * u->k * sizeof(float));
				if (real) {
					nreal = R / 2;
					for (unsigned t = 0; t < nreal; t++)
						m1(&uq->mb[0], uq->X + (size_t)t * uq->k,
						   real + (size_t)t * u->k);
				}
			}
			fill_targets(u, &m, NULL, real, nreal);
			free(real);
			nu++;
		}
		snprintf(lname, sizeof(lname), "blk.%d ffn_down", layer);
		if (!unit_init(&us[nu], lname, dn, 1, R)) {
			struct unit *u = &us[nu];
			struct member g, up;
			float *real = malloc((size_t)R * u->k * sizeof(float));
			float *x = malloc((size_t)m.n_embd * sizeof(float));
			float *yu = malloc((size_t)u->k * sizeof(float));
			unsigned nreal = 0;

			memset(&g, 0, sizeof(g));
			memset(&up, 0, sizeof(up));
			g.w = LL->gate;
			up.w = LL->up;
			g.nt = LL->gate ? charsiu_pool_get(&S->pool, LL->gate) : NULL;
			up.nt = LL->up ? charsiu_pool_get(&S->pool, LL->up) : NULL;
			if (real && x && yu && g.nt && up.nt && g.nt->n == u->k &&
			    m.tok_embd && m.tok_embd->ne[0] == m.n_embd) {
				g.id = pool_id_of(g.nt);
				up.id = pool_id_of(up.nt);
				g.n = g.nt->n;
				up.n = up.nt->n;
				nreal = R / 2;
				for (unsigned t = 0; t < nreal; t++) {
					float *h = real + (size_t)t * u->k;

					gguf_row_f32(m.tok_embd, (uint64_t)(4000 + 37 * t)
						     % m.tok_embd->ne[1], x);
					rmsnorm_gain(x, m.n_embd, LL->ffn_norm);
					m1(&g, x, h);
					m1(&up, x, yu);
					for (uint64_t j = 0; j < u->k; j++)
						h[j] = h[j] / (1.0f + expf(-h[j])) * yu[j];
				}
			}
			fill_targets(u, &m, NULL, nreal ? real : NULL, nreal);
			free(real);
			free(x);
			free(yu);
			nu++;
		}
		if (m.output && !unit_init(&us[nu], "output head", hd, 1, R)) {
			us[nu].head = 1;
			fill_targets(&us[nu], &m, m.out_norm, NULL, 0);
			nu++;
		}
		(void)ug;
	}

	for (unsigned ui = 0; ui < nu; ui++) {
		double tu = now_ms();

		unit_header(&us[ui]);
		run_unit(&us[ui], !us[ui].head);
		if (copies && G && !us[ui].head)
			for (unsigned i = 0; i < us[ui].nm; i += 2) {
				int tested, same;

				core_copy(&us[ui], i, &tested, &same);
				if (tested) {
					core_tested = 1;
					if (!same)
						core_bad = 1;
				}
			}
		printf("  (%s: %.0f ms)\n", us[ui].label, now_ms() - tu);
	}

	verdict(us, nu, pack_bad, core_tested, core_bad);
	printf("  total %.1f s\n", (now_ms() - t0) / 1e3);
	fflush(stdout);
	/* no teardown: the process exits, and the copies the pool keys on
	 * would have to outlive it anyway */
	return 0;
}
