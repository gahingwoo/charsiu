// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The batched prompt's tiled attention against the kernel it replaced, BIT FOR
 * BIT.
 *
 * attn_block_heads now puts three rows of a block through one pass over a
 * quad of keys and four through one pass over an eight of values, and claims
 * to have moved only the issue order of independent operations. That is an
 * equality, so it is checked with memcmp: every output float of every row.
 * The kernel it replaced is copied below VERBATIM as ref_block_heads, the
 * one function here that is not the shipped code; everything it calls
 * (attn_dot4, attn_dot, attn_axpy8, attn_axpy4, attn_axpy, softmax) is the
 * shipped code, through the same include tests/dattn_exact.c uses.
 *
 * What it walks: head sizes 64, 128 and 256, and 72 and 100 (72 is tiled and
 * not a multiple of 16; 100 has a ragged hd % 8 and takes the untiled
 * kernels), groups 1, 2 and 4 over one and two kv heads, chunks of 2 to 160
 * rows at positions 0 and past it, with no window and with windows of 6 and
 * 33, blocks of 8 rows and of 1, 3 and 13, and every unit range split in two
 * the way the pool hands them out. Each arm of the switches the kernels read
 * -- the tile itself, the exact dot, the plain CPU, the four wide values loop,
 * the head-only split, the position major cache -- runs in its own process,
 * because each is read once into a function static, and the arms run at once.
 *
 * `attn_tile --bench` times the two at Qwen3-0.6B's shape on one thread, in
 * blocks of CHARSIU_ATTN_BLOCK rows (8) as the runtime would.
 */
#include "../src/llama.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* ---- the kernel before the tiles, unchanged but for its name ---- */
static void ref_block_heads(void *ctx, uint64_t u0, uint64_t nu)
{
	struct attn_block_job *j = ctx;
	struct llama_state *s = j->s;
	uint32_t hd = j->hd, hdmax = j->hdmax, gqa = j->gqa, nkv = j->nkv;
	int R = j->R, n = j->n;
	size_t qstride = (size_t)j->n_head * hd;
	int timed = stage_on > 0 && !attn_block_pool();
	int nblk = attn_hr() ? (n + R - 1) / R : 1;
	uint64_t u;

	for (u = u0; u < u0 + nu; u++) {
		uint32_t h = (uint32_t)(attn_hr() ? u / (unsigned)nblk : u);
		int only = attn_hr() ? (int)(u % (unsigned)nblk) : -1;
		uint32_t kvh = h / gqa;
		const float *kbase, *vbase;
		size_t kstride;
		float *sc = s->batt + (size_t)u * R * s->n_ctx;

		if (kv_posmajor()) {
			size_t b = (size_t)j->l * s->n_ctx * j->kvdim
				 + (size_t)kvh * hd;

			kbase = s->kcache + b;
			vbase = s->vcache + b;
			kstride = j->kvdim;
		} else {
			size_t b = (size_t)(j->l * nkv + kvh) * s->n_ctx * hdmax;

			kbase = s->kcache + b;
			vbase = s->vcache + b;
			kstride = hdmax;
		}
		/* one block when the unit names it, the whole range when the
		 * split is head-only */
		int rb0 = only >= 0 ? only * R : 0;
		int rbN = only >= 0 ? (rb0 + R < n ? rb0 + R : n) : n;

		for (int rb = rb0; rb < rbN; rb += R) {
			int re = rb + R < n ? rb + R : n;
			int tlo_first = j->swa && j->pos0 + rb + 1 > j->n_swa
				      ? j->pos0 + rb + 1 - j->n_swa : 0;
			int tmax = j->pos0 + re - 1;
			double t0 = timed ? now_ms() : 0.0, t1;

			for (int t = tlo_first; t <= tmax; t += 4) {
				const float *k0 = kbase + (size_t)t * kstride;
				int quad = t + 3 <= tmax;

				for (int r = rb; r < re; r++) {
					int pos = j->pos0 + r;
					int tlo = j->swa && pos + 1 > j->n_swa
						? pos + 1 - j->n_swa : 0;
					const float *qh = j->q + (size_t)r * qstride + h * hd;
					float *sr = sc + (size_t)(r - rb) * s->n_ctx;

					if (quad && t >= tlo && t + 3 <= pos) {
						float o[4];

						attn_dot4(qh, k0, k0 + kstride,
							  k0 + 2 * kstride, k0 + 3 * kstride,
							  hd, o);
						sr[t] = o[0] * j->scale;
						sr[t + 1] = o[1] * j->scale;
						sr[t + 2] = o[2] * j->scale;
						sr[t + 3] = o[3] * j->scale;
						continue;
					}
					for (int u = t; u < t + 4 && u <= tmax; u++) {
						if (u < tlo || u > pos)
							continue;
						sr[u] = attn_dot(qh, kbase + (size_t)u * kstride,
								 hd) * j->scale;
					}
				}
			}
			if (timed) { t1 = now_ms(); battn_ms[0] += t1 - t0; t0 = t1; }
			for (int r = rb; r < re; r++) {
				int pos = j->pos0 + r;
				int tlo = j->swa && pos + 1 > j->n_swa
					? pos + 1 - j->n_swa : 0;

				softmax(sc + (size_t)(r - rb) * s->n_ctx + tlo,
					pos + 1 - tlo);
				memset(j->out + (size_t)r * qstride + h * hd, 0,
				       hd * sizeof(float));
			}
			if (timed) { t1 = now_ms(); battn_ms[1] += t1 - t0; t0 = t1; }
			{
			int step = attn_axpy8_on() ? 8 : 4;

			for (int t = tlo_first; t <= tmax; t += step) {
				const float *v0 = vbase + (size_t)t * kstride;
				int wide = t + step - 1 <= tmax;

				for (int r = rb; r < re; r++) {
					int pos = j->pos0 + r;
					int tlo = j->swa && pos + 1 > j->n_swa
						? pos + 1 - j->n_swa : 0;
					float *out = j->out + (size_t)r * qstride + h * hd;
					const float *sr = sc + (size_t)(r - rb) * s->n_ctx;

					/*
					 * THE PARTIAL WINDOW GOES ONE
					 * POSITION AT A TIME AND THAT IS NOT A
					 * DIFFERENT ANSWER: every form here
					 * adds a0, a1, a2 ... in that order,
					 * and the only thing the wide forms
					 * remove is a store and a reload of a
					 * float32 in between.
					 */
					if (wide && t >= tlo && t + step - 1 <= pos) {
						if (step == 8)
							attn_axpy8(out, sr + t, v0,
								   kstride, hd);
						else
							attn_axpy4(out, sr + t, v0,
								   v0 + kstride,
								   v0 + 2 * kstride,
								   v0 + 3 * kstride, hd);
						continue;
					}
					for (int u = t; u < t + step && u <= tmax; u++) {
						if (u < tlo || u > pos)
							continue;
						attn_axpy(out, sr[u],
							  vbase + (size_t)u * kstride, hd);
					}
				}
			}
			}
			if (timed) { t1 = now_ms(); battn_ms[2] += t1 - t0; }
		}
	}
}

static uint32_t seed = 2026;

static float rnd(void)
{
	seed = seed * 1664525u + 1013904223u;
	return ((float)(seed >> 8) / 8388608.0f - 1.0f) * 1.5f;
}

#define NL 2
#define N_CTX 272

struct rig {
	struct llama_state s;
	float *k, *v, *q, *ref, *mine;
};

static void rig_init(struct rig *r, uint32_t nkv, uint32_t hdmax,
		     uint32_t nh, int nmax)
{
	size_t cells = (size_t)NL * nkv * N_CTX * hdmax, i;
	size_t qc = (size_t)nmax * nh * hdmax;

	memset(r, 0, sizeof(*r));
	r->k = malloc(cells * sizeof(float));
	r->v = malloc(cells * sizeof(float));
	r->q = malloc(qc * sizeof(float));
	r->ref = malloc(qc * sizeof(float));
	r->mine = malloc(qc * sizeof(float));
	for (i = 0; i < cells; i++) {
		r->k[i] = rnd();
		r->v[i] = rnd();
	}
	for (i = 0; i < qc; i++)
		r->q[i] = rnd();
	r->s.n_ctx = N_CTX;
	r->s.kcache = r->k;
	r->s.vcache = r->v;
	/* the largest scratch any case asks for: R = 1 at the longest chunk */
	r->s.batt = malloc((size_t)nh * nmax * 13 * N_CTX * sizeof(float));
}

static void rig_free(struct rig *r)
{
	free(r->k); free(r->v); free(r->q); free(r->ref); free(r->mine);
	free(r->s.batt);
}

static unsigned cases, bad;

static void one(struct rig *r, uint32_t hd, uint32_t hdmax, uint32_t gqa,
		uint32_t nkv, int pos0, int n, int swa, int n_swa, int R)
{
	uint32_t nh = nkv * gqa;
	int nblk = attn_hr() ? (n + R - 1) / R : 1;
	uint64_t nu = (uint64_t)nh * nblk, cut = nu / 2 + 1;
	size_t ob = (size_t)n * nh * hd * sizeof(float);
	struct attn_block_job bj = {
		&r->s, NL - 1, pos0, n, swa, n_swa, hd, hdmax, nkv * hdmax,
		gqa, nkv, nh, 1.0f / sqrtf((float)hd), r->q, r->ref, R };

	if (cut > nu)
		cut = nu;
	memset(r->ref, 0x55, ob);
	ref_block_heads(&bj, 0, nu);
	bj.out = r->mine;
	memset(r->mine, 0x33, ob);
	attn_block_heads(&bj, 0, cut);
	attn_block_heads(&bj, cut, nu - cut);
	cases++;
	if (memcmp(r->ref, r->mine, ob)) {
		if (bad < 10)
			printf("  differs: hd %u gqa %u nkv %u pos0 %d n %d "
			       "swa %d/%d R %d\n", hd, gqa, nkv, pos0, n, swa,
			       n_swa, R);
		bad++;
	}
}

static int sweep(const char *arm)
{
	static const uint32_t hds[] = { 64, 72, 100, 128, 256 };
	static const uint32_t gqas[] = { 1, 2, 4 };
	static const int ns[] = { 2, 3, 5, 7, 8, 9, 13, 16, 17, 31, 64, 100,
				  124, 160 };
	static const int pos0s[] = { 0, 1, 5, 37, 100 };
	static const int Rs[] = { 8, 1, 3, 13 };
	static const int swas[][2] = { { 0, 0 }, { 1, 6 }, { 1, 33 } };
	unsigned hi, gi, ni, pi, ri, wi, nkv;

	for (nkv = 1; nkv <= 2; nkv++)
		for (hi = 0; hi < sizeof(hds) / sizeof(hds[0]); hi++) {
			uint32_t hd = hds[hi];
			uint32_t hdmax = hd == 256 ? 512 : hd;

			for (gi = 0; gi < sizeof(gqas) / sizeof(gqas[0]); gi++) {
				uint32_t gqa = gqas[gi];
				struct rig r;

				rig_init(&r, nkv, hdmax, nkv * gqa, 160);
				for (ni = 0; ni < sizeof(ns) / sizeof(ns[0]); ni++)
				for (pi = 0; pi < sizeof(pos0s) / sizeof(pos0s[0]); pi++)
				for (wi = 0; wi < 3; wi++)
				for (ri = 0; ri < sizeof(Rs) / sizeof(Rs[0]); ri++) {
					int n = ns[ni], pos0 = pos0s[pi], R = Rs[ri];

					if (pos0 + n > N_CTX)
						continue;
					/* attn_block clamps the block to the chunk */
					if (R > n)
						R = n;
					/* the default block everywhere, the odd
					 * ones on the smaller heads */
					if (ri && hd > 128)
						continue;
					one(&r, hd, hdmax, gqa, nkv, pos0, n,
					    swas[wi][0], swas[wi][1], R);
				}
				rig_free(&r);
			}
		}
	printf("attn_tile: %-28s %u of %u cases differ in a single bit\n",
	       arm, bad, cases);
	return bad != 0;
}

static double now(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/*
 * Qwen3-0.6B: 16 query heads over 8 kv heads of 128, a chunk of n rows at
 * pos0, one thread, 28 layers a pass. Old and new alternate so a drift in the
 * desk's clock lands on both.
 */
static void bench(int n, int pos0)
{
	const uint32_t nkv = 8, gqa = 2, hd = 128, nh = 16;
	const int R = attn_block_rows() > 0 ? attn_block_rows() : 8, reps = 15;
	int nblk = (n + R - 1) / R, i, k;
	struct rig r;
	double best[2] = { 1e30, 1e30 }, split[2][3];

	rig_init(&r, nkv, hd, nh, n);
	stage_on = 1;
	for (i = 0; i < reps; i++)
		for (k = 0; k < 2; k++) {
			struct attn_block_job bj = {
				&r.s, NL - 1, pos0, n, 0, 0, hd, hd, nkv * hd,
				gqa, nkv, nh, 1.0f / sqrtf((float)hd), r.q,
				k ? r.mine : r.ref, R };
			double t0, t;
			int l;

			memset(battn_ms, 0, sizeof(battn_ms));
			t0 = now();
			for (l = 0; l < 28; l++)
				(k ? attn_block_heads : ref_block_heads)(
					&bj, 0, (uint64_t)nh * nblk);
			t = now() - t0;
			if (t < best[k]) {
				best[k] = t;
				memcpy(split[k], battn_ms, sizeof(battn_ms));
			}
		}
	printf("attn_tile: Qwen3-0.6B shape, %d rows at %d, one thread, 28 "
	       "layers: old %.2f ms, new %.2f ms, %.2fx\n", n, pos0, best[0],
	       best[1], best[0] / best[1]);
	for (k = 0; k < 2; k++)
		printf("  %s  scores %.2f  softmax %.2f  values %.2f ms\n",
		       k ? "new" : "old", split[k][0], split[k][1], split[k][2]);
	printf("  outputs %s\n", memcmp(r.ref, r.mine, (size_t)n * nh * hd
					* sizeof(float)) ? "DIFFER" : "identical");
	rig_free(&r);
}

int main(int argc, char **argv)
{
	static const char *const arms[][2] = {
		{ "the default", NULL },
		{ "CHARSIU_ATTN_TILE=0", "CHARSIU_ATTN_TILE" },
		{ "CHARSIU_EXACT_ATTN=1", "CHARSIU_EXACT_ATTN" },
		{ "CHARSIU_CPU_PLAIN=1", "CHARSIU_CPU_PLAIN" },
		{ "CHARSIU_ATTN_AXPY8=0", "CHARSIU_ATTN_AXPY8" },
		{ "CHARSIU_ATTN_HR=0", "CHARSIU_ATTN_HR" },
		{ "CHARSIU_KV_POSMAJOR=1", "CHARSIU_KV_POSMAJOR" },
	};
#define NARM (sizeof(arms) / sizeof(arms[0]))
	pid_t pid[NARM];
	unsigned a;
	int rc = 0;

	if (argc > 1 && !strcmp(argv[1], "--bench")) {
		/* the serial arm, so the kernel's own split is kept */
		setenv("CHARSIU_ATTN_BLOCK_POOL", "0", 1);
		bench(124, 0);
		bench(80, 80);
		bench(160, 0);
		return 0;
	}
	/* every arm at once, a process each, so the test costs one sweep */
	fflush(stdout);
	for (a = 0; a < NARM; a++) {
		pid[a] = fork();
		if (pid[a] < 0)
			return 1;
		if (pid[a] == 0) {
			int st;

			if (arms[a][1]) {
				const char *val = !strcmp(arms[a][1],
					"CHARSIU_ATTN_AXPY8")
					|| !strcmp(arms[a][1], "CHARSIU_ATTN_HR")
					|| !strcmp(arms[a][1], "CHARSIU_ATTN_TILE")
					? "0" : "1";

				setenv(arms[a][1], val, 1);
			}
			st = sweep(arms[a][0]);
			fflush(stdout);
			_exit(st);
		}
	}
	for (a = 0; a < NARM; a++) {
		int st;

		if (waitpid(pid[a], &st, 0) < 0 || !WIFEXITED(st)
		    || WEXITSTATUS(st))
			rc = 1;
	}
	return rc;
}
