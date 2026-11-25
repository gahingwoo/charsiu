// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The grouped decode attention against the kernels it replaces, BIT FOR BIT.
 *
 * attn_gqa_group claims to change only the issue order of the arithmetic of
 * attn_heads (the token loop) and of attn_block_heads (the verify rows), and
 * under CHARSIU_KV16 to give exactly what the float kernel gives on a float
 * cache whose values went through fp16 once. Those are equalities, so this
 * checks them with memcmp and not with a tolerance: a tolerance is how an
 * index that is off by one position at a block edge ships, because the answer
 * it produces is close.
 *
 * IT INCLUDES THE TRANSLATION UNIT, as tests/attn_two_thresholds.c does, so
 * what it calls is the shipped code and not a copy of it (tests/axpy8.c says
 * what a copy can and cannot prove). It needs no model: the kernels read only
 * the cache, the scores scratch and the query, and a llama_state with those
 * four filled in is all they touch.
 *
 * What it walks: head sizes 64, 72, 100, 128 and 256 in a 512 stride (a ragged
 * tail of hd % 8, and gemma4's two head sizes in one cache), groups 1, 2, 3, 4
 * and 8, one and three kv heads, positions 0 to 257 with the block edges
 * (3, 4, 5, 7, 8, 9 ...) in them, a sliding window starting at 0, a third of
 * the way in and two before the end; and for the verify rows 1 to 8 rows at
 * those positions, windowed or not, with the rows sliced over 1 and 3 units.
 * And the half converter, scalar against the vector instruction, at every
 * half and at sixteen million floats.
 */
#include "../src/llama.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t seed = 12345;

static float rnd(void)
{
	seed = seed * 1664525u + 1013904223u;
	return ((float)(seed >> 8) / 8388608.0f - 1.0f) * 1.5f;
}

struct rig {
	struct llama_state s;
	uint32_t nl, nkv, hdmax;
	size_t cells;
	float *k32, *v32;        /* the original float cache */
	float *kr, *vr;          /* the same, rounded through fp16 */
	uint16_t *kh, *vh;
};

static void rig_init(struct rig *r, uint32_t nl, uint32_t nkv, uint32_t nh,
		     uint32_t hdmax, int n_ctx)
{
	size_t i;

	memset(r, 0, sizeof(*r));
	r->nl = nl; r->nkv = nkv; r->hdmax = hdmax;
	r->cells = (size_t)nl * nkv * n_ctx * hdmax;
	r->k32 = malloc(r->cells * sizeof(float));
	r->v32 = malloc(r->cells * sizeof(float));
	r->kr = malloc(r->cells * sizeof(float));
	r->vr = malloc(r->cells * sizeof(float));
	r->kh = malloc(r->cells * sizeof(uint16_t));
	r->vh = malloc(r->cells * sizeof(uint16_t));
	for (i = 0; i < r->cells; i++) {
		r->k32[i] = rnd();
		r->v32[i] = rnd();
	}
	kv16_put(r->kh, r->k32, (uint32_t)r->cells);
	kv16_put(r->vh, r->v32, (uint32_t)r->cells);
	kv16_get(r->kr, r->kh, (uint32_t)r->cells);
	kv16_get(r->vr, r->vh, (uint32_t)r->cells);
	r->s.n_ctx = n_ctx;
	r->s.att = calloc((size_t)nh * n_ctx, sizeof(float));
	r->s.xb = calloc((size_t)nh * hdmax, sizeof(float));
	r->s.batt = calloc((size_t)nh * GQA_VERIFY_ROWS * n_ctx, sizeof(float));
	r->s.batt_rows = GQA_VERIFY_ROWS;
	r->s.batt_units = nh;
}

static void rig_free(struct rig *r)
{
	free(r->k32); free(r->v32); free(r->kr); free(r->vr);
	free(r->kh); free(r->vh);
	free(r->s.att); free(r->s.xb); free(r->s.batt);
}

/* which cache the old kernel reads and which the new one does */
static void use_float(struct rig *r, int rounded)
{
	r->s.kcache = rounded ? r->kr : r->k32;
	r->s.vcache = rounded ? r->vr : r->v32;
	r->s.kc16 = r->s.vc16 = NULL;
}

static void use_half(struct rig *r)
{
	r->s.kcache = r->s.vcache = NULL;   /* must not be read */
	r->s.kc16 = r->kh;
	r->s.vc16 = r->vh;
}

static unsigned cases, bad;

static void decode_case(struct rig *r, uint32_t hd, uint32_t gqa, int pos,
			int tlo, const float *q)
{
	uint32_t nh = r->nkv * gqa;
	size_t ob = (size_t)nh * hd * sizeof(float);
	float *ref = malloc(ob), *mine = malloc(ob);
	int half;

	for (half = 0; half < 2; half++) {
		struct attn_job aj = { &r->s, r->nl - 1, pos, tlo, hd, r->hdmax,
				       r->nkv * r->hdmax, gqa, r->nkv, q,
				       1.0f / sqrtf((float)hd) };

		use_float(r, half);
		memset(r->s.xb, 0x55, ob);
		attn_heads(&aj, 0, nh);
		memcpy(ref, r->s.xb, ob);
		if (half)
			use_half(r);
		memset(r->s.xb, 0x33, ob);
		attn_gqa_heads(&aj, 0, nh);
		/* and a split range, which is what the pool hands out */
		if (nh > 1) {
			memcpy(mine, r->s.xb, ob);
			memset(r->s.xb, 0x33, ob);
			attn_gqa_heads(&aj, 0, nh / 2 + 1);
			attn_gqa_heads(&aj, nh / 2 + 1, nh - (nh / 2 + 1));
			if (memcmp(mine, r->s.xb, ob)) {
				if (bad < 10)
					printf("  split differs: hd %u gqa %u "
					       "pos %d tlo %d %s\n", hd, gqa,
					       pos, tlo, half ? "half" : "float");
				bad++;
			}
		}
		memcpy(mine, r->s.xb, ob);
		cases++;
		if (memcmp(ref, mine, ob)) {
			if (bad < 10)
				printf("  decode differs: hd %u gqa %u nkv %u "
				       "pos %d tlo %d %s\n", hd, gqa, r->nkv,
				       pos, tlo, half ? "half" : "float");
			bad++;
		}
	}
	free(ref);
	free(mine);
}

static void verify_case(struct rig *r, uint32_t hd, uint32_t gqa, int pos0,
			int n, int swa, int n_swa, const float *q)
{
	uint32_t nh = r->nkv * gqa;
	size_t ob = (size_t)n * nh * hd * sizeof(float);
	float *ref = malloc(ob), *mine = malloc(ob);
	int half;
	unsigned split;

	for (half = 0; half < 2; half++)
		for (split = 1; split <= 3; split += 2) {
			struct attn_block_job bj = {
				&r->s, r->nl - 1, pos0, n, swa, n_swa, hd,
				r->hdmax, r->nkv * r->hdmax, gqa, r->nkv, nh,
				1.0f / sqrtf((float)hd), q, ref, n, 0, 0, 0 };
			struct gqa_rows_job g = { &bj, split > (unsigned)n
						  ? (unsigned)n : split };

			use_float(r, half);
			memset(ref, 0x55, ob);
			attn_block_heads(&bj, 0, nh);
			if (half)
				use_half(r);
			bj.out = mine;
			memset(mine, 0x33, ob);
			attn_gqa_rows(&g, 0, (uint64_t)r->nkv * g.split);
			cases++;
			if (memcmp(ref, mine, ob)) {
				if (bad < 10)
					printf("  verify differs: hd %u gqa %u "
					       "nkv %u pos0 %d n %d swa %d/%d "
					       "split %u %s\n", hd, gqa,
					       r->nkv, pos0, n, swa, n_swa,
					       g.split, half ? "half" : "float");
				bad++;
			}
		}
	free(ref);
	free(mine);
}

static int converter(void)
{
	unsigned diff = 0;
	uint32_t h;
	uint64_t u;

	/* every half back to float: scalar against the vector widen */
	for (h = 0; h < 65536; h++) {
		uint16_t x = (uint16_t)h;
		float a = kv16_h2f(x), b;

		kv16_get(&b, &x, 1);   /* the scalar tail: n < 8 */
		{
			uint16_t xs[8] = { x, x, x, x, x, x, x, x };
			float bs[8];

			kv16_get(bs, xs, 8);
			b = bs[3];
		}
		if (((h >> 10) & 0x1f) == 0x1f && (h & 0x3ff))
			continue;              /* nan: payloads are not ours */
		if (memcmp(&a, &b, 4))
			diff++;
	}
	/* sixteen million floats to half, a stride through every exponent */
	for (u = 0; u < 0x100000000ull; u += 251) {
		uint32_t b32 = (uint32_t)u;
		float f, fs[4];
		uint16_t a, bs[4];

		memcpy(&f, &b32, 4);
		if ((b32 & 0x7f800000u) == 0x7f800000u && (b32 & 0x7fffff))
			continue;
		a = kv16_f2h(f);
		fs[0] = fs[1] = fs[2] = fs[3] = f;
		kv16_put(bs, fs, 4);
		if (a != bs[2])
			diff++;
	}
	printf("dattn_exact: the half converter, scalar against vector: "
	       "%u differ\n", diff);
	return diff != 0;
}

int main(void)
{
	static const uint32_t hds[] = { 64, 72, 100, 128, 256 };
	static const uint32_t gqas[] = { 1, 2, 3, 4, 8 };
	static const int poss[] = { 0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17,
				    31, 63, 100, 257 };
	const int n_ctx = 272;
	unsigned hi, gi, pi, nkv;
	int rc;

	/* the arm must exist; gqa_bases reads the half cache only with it on */
	setenv("CHARSIU_KV16", "1", 1);
	rc = converter();

	for (nkv = 1; nkv <= 3; nkv += 2)
		for (hi = 0; hi < sizeof(hds) / sizeof(hds[0]); hi++) {
			uint32_t hd = hds[hi];
			uint32_t hdmax = hd == 256 ? 512 : hd;
			struct rig r;

			rig_init(&r, 2, nkv, nkv * 8, hdmax, n_ctx);
			for (gi = 0; gi < sizeof(gqas) / sizeof(gqas[0]); gi++) {
				uint32_t gqa = gqas[gi], nh = nkv * gqa;
				float *q = malloc((size_t)GQA_VERIFY_ROWS * nh
						  * hd * sizeof(float));
				size_t e;

				for (e = 0; e < (size_t)GQA_VERIFY_ROWS * nh * hd; e++)
					q[e] = rnd();
				for (pi = 0; pi < sizeof(poss) / sizeof(poss[0]); pi++) {
					int pos = poss[pi];
					int tlos[3] = { 0, pos / 3, pos >= 2 ? pos - 2 : 0 };
					int ti, n;

					for (ti = 0; ti < 3; ti++)
						decode_case(&r, hd, gqa, pos,
							    tlos[ti], q);
					for (n = 1; n <= GQA_VERIFY_ROWS; n++) {
						if (pos + n > n_ctx)
							continue;
						verify_case(&r, hd, gqa, pos, n,
							    0, 0, q);
						verify_case(&r, hd, gqa, pos, n,
							    1, 6, q);
					}
				}
				free(q);
			}
			rig_free(&r);
		}
	printf("dattn_exact: grouped decode and verify attention against "
	       "attn_heads and attn_block_heads, float and half: %u of %u "
	       "cases differ in a single bit\n", bad, cases);
	return rc || bad != 0;
}
