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
/* the cache's length: the scores rows of attn_block_heads are this apart */
static int N_CTX = 272;

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
	/* the largest scratch any case asks for: ceil(n / R) * R < n + 13 */
	r->s.batt = malloc((size_t)nh * (nmax + 13) * N_CTX * sizeof(float));
}

static void rig_free(struct rig *r)
{
	free(r->k); free(r->v); free(r->q); free(r->ref); free(r->mine);
	free(r->s.batt);
}

static unsigned cases, kvcases, bad;

static void one(struct rig *r, uint32_t hd, uint32_t hdmax, uint32_t gqa,
		uint32_t nkv, int pos0, int n, int swa, int n_swa, int R)
{
	uint32_t nh = nkv * gqa;
	int nblk = attn_hr() ? (n + R - 1) / R : 1;
	uint64_t nu = (uint64_t)nh * nblk, cut = nu / 2 + 1;
	size_t ob = (size_t)n * nh * hd * sizeof(float);
	struct attn_block_job bj = {
		&r->s, NL - 1, pos0, n, swa, n_swa, hd, hdmax, nkv * hdmax,
		gqa, nkv, nh, 1.0f / sqrtf((float)hd), r->q, r->ref, R, 0, 0, 0 };

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
			       "swa %d/%d R %d n_ctx %d\n", hd, gqa, nkv, pos0,
			       n, swa, n_swa, R, N_CTX);
		bad++;
	}
#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
	/*
	 * THE KV UNIT, where it applies, over the same job: its units in
	 * three ranges, the way the pool hands them out one at a time.
	 */
	if (attn_kv_plan(&bj)) {
		int RB = bj.kv_rows;
		uint64_t ku = (uint64_t)nkv * ((n + RB - 1) / RB);
		uint64_t c1 = ku / 3, c2 = ku - ku / 3;

		memset(r->mine, 0x77, ob);
		attn_kv_units(&bj, 0, c1);
		attn_kv_units(&bj, c1, c2 - c1);
		attn_kv_units(&bj, c2, ku - c2);
		kvcases++;
		if (bj.kv_failed || memcmp(r->ref, r->mine, ob)) {
			if (bad < 10)
				printf("  kv unit differs: hd %u gqa %u nkv %u "
				       "pos0 %d n %d swa %d/%d n_ctx %d\n", hd,
				       gqa, nkv, pos0, n, swa, n_swa, N_CTX);
			bad++;
		}
	}
#endif
}

static void sweep_ctx(void);

static int sweep(const char *arm)
{
	static const int ctxs[] = { 272, 1024, 2048, 4096 };
	unsigned ci;

	for (ci = 0; ci < sizeof(ctxs) / sizeof(ctxs[0]); ci++) {
		N_CTX = ctxs[ci];
		sweep_ctx();
	}
	printf("attn_tile: %-28s %u of %u cases (%u kv unit) differ in a "
	       "single bit\n", arm, bad, cases + kvcases, kvcases);
	return bad != 0;
}

/* the shapes at one cache length; the long caches take the default block */
static void sweep_ctx(void)
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
					/* a chunk that ends on the cache's last
					 * position, once per cache length */
					if (N_CTX > 272 && ni == 0 && pi == 0
					    && wi == 0 && ri == 0)
						one(&r, hd, hdmax, gqa, nkv,
						    N_CTX - 160, 160, 0, 0, 8);
					/* attn_block clamps the block to the chunk */
					if (R > n)
						R = n;
					/* the default block everywhere, the odd
					 * ones on the smaller heads */
					if (ri && (hd > 128 || N_CTX > 272))
						continue;
					/* past 272 the cache length is the
					 * point, not the shape: fewer of them */
					if (N_CTX > 272 && (pi % 2 || wi == 1))
						continue;
					one(&r, hd, hdmax, gqa, nkv, pos0, n,
					    swas[wi][0], swas[wi][1], R);
				}
				rig_free(&r);
			}
		}
}

static double now(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/*
 * Qwen3-0.6B: 16 query heads over 8 kv heads of 128, a chunk of n rows at
 * pos0 in a 1024 position cache (the board's -c), 28 layers a pass. Three
 * arms alternate so a drift in the clock lands on all of them: the untiled
 * kernel, attn_block_heads (the tiles), attn_kv_units. `threads` above one
 * runs them on the pool the way attn_block_cpu does.
 */
/* busy time inside the kv unit, summed over threads, for the parallel
 * efficiency: busy / (threads * wall) */
static long kv_busy_ns;

static void kv_units_timed(void *ctx, uint64_t u0, uint64_t nu)
{
	double t0 = now_ms();

	attn_kv_units(ctx, u0, nu);
	t0 = now_ms() - t0;
	__atomic_fetch_add((long *)&kv_busy_ns, (long)(t0 * 1e6),
			   __ATOMIC_RELAXED);
}

static void bench(int n, int pos0, int threads)
{
	const uint32_t nkv = 8, gqa = 2, hd = 128, nh = 16;
	const int R = attn_block_rows() > 0 ? attn_block_rows() : 8, reps = 9;
	int nblk = (n + R - 1) / R, i, k;
	struct rig r;
	double best[3] = { 1e30, 1e30, 1e30 }, split[3][3];
	float *outs[3];
	static const char *const name[3] = { "untiled", "tiles", "kv unit" };

	N_CTX = 1024;
	rig_init(&r, nkv, hd, nh, n);
	outs[0] = r.ref;
	outs[1] = r.mine;
	outs[2] = malloc((size_t)n * nh * hd * sizeof(float));
	stage_on = threads > 1 ? 0 : 1;
	for (i = 0; i < reps; i++)
		for (k = 0; k < 3; k++) {
			struct attn_block_job bj = {
				&r.s, NL - 1, pos0, n, 0, 0, hd, hd, nkv * hd,
				gqa, nkv, nh, 1.0f / sqrtf((float)hd), r.q,
				outs[k], R, 0, 0, 0 };
			uint64_t ku = 0;
			double t0, t;
			int l;

			/* ATTN_TILE_BENCH_KV=1 times the kv unit alone */
			if (k < 2 && getenv("ATTN_TILE_BENCH_KV"))
				continue;
			if (k == 2) {
				if (!attn_kv_plan(&bj))
					continue;
				ku = (uint64_t)nkv * ((n + bj.kv_rows - 1)
						      / bj.kv_rows);
			}
			memset(battn_ms, 0, sizeof(battn_ms));
			kv_busy_ns = 0;
			t0 = now();
			for (l = 0; l < 28; l++) {
				if (k == 2) {
					if (threads > 1)
						pool_run_chunk(getenv("ATTN_TILE_BUSY")
							       ? kv_units_timed
							       : attn_kv_units, &bj,
							       ku, 1, 1);
					else
						attn_kv_units(&bj, 0, ku);
				} else if (threads > 1) {
					charsiu_parallel_for(k ? attn_block_heads
							       : ref_block_heads,
							     &bj, (uint64_t)nh * nblk);
				} else {
					(k ? attn_block_heads : ref_block_heads)(
						&bj, 0, (uint64_t)nh * nblk);
				}
			}
			t = now() - t0;
			if (k == 2 && getenv("ATTN_TILE_BUSY"))
				printf("  kv unit busy %.1f%% of %d x %.1f ms\n",
				       100.0 * kv_busy_ns / 1e6 / (threads * t),
				       threads, t);
			if (t < best[k]) {
				best[k] = t;
				memcpy(split[k], battn_ms, sizeof(battn_ms));
			}
		}
	printf("attn_tile: Qwen3-0.6B shape, %d rows at %d, %d thread(s), 28 "
	       "layers, ms a row:", n, pos0, threads);
	for (k = 0; k < 3; k++)
		if (best[k] < 1e30)
			printf("  %s %.3f", name[k], best[k] / n);
	if (best[1] < 1e30)
		printf("  (kv unit %.2fx the tiles)", best[1] / best[2]);
	printf("\n");
	if (threads <= 1)
		for (k = 0; k < 3; k++)
			if (best[k] < 1e30)
				printf("  %-8s scores %.3f  softmax %.3f  "
				       "values %.3f ms a row\n", name[k],
				       split[k][0] / n, split[k][1] / n,
				       split[k][2] / n);
	if (getenv("ATTN_TILE_BENCH_KV")) {
		free(outs[2]);
		rig_free(&r);
		return;
	}
	printf("  outputs %s\n",
	       memcmp(r.ref, r.mine, (size_t)n * nh * hd * sizeof(float))
	       || memcmp(r.ref, outs[2], (size_t)n * nh * hd * sizeof(float))
	       ? "DIFFER" : "identical");
	free(outs[2]);
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
		{ "CHARSIU_ATTN_KV=0", "CHARSIU_ATTN_KV" },
		{ "CHARSIU_ATTN_KV_ROWS=5", "CHARSIU_ATTN_KV_ROWS" },
		{ "CHARSIU_ATTN_KV_ROWS=1", "CHARSIU_ATTN_KV_ROWS=1" },
		{ "CHARSIU_ATTN_KV_TB=7", "CHARSIU_ATTN_KV_TB" },
		{ "CHARSIU_ATTN_KV_TB=4", "CHARSIU_ATTN_KV_TB=4" },
		{ "CHARSIU_ATTN_KV_PACK=0", "CHARSIU_ATTN_KV_PACK=0" },
	};
#define NARM (sizeof(arms) / sizeof(arms[0]))
	pid_t pid[NARM];
	unsigned a;
	int rc = 0;

	if (argc > 1 && !strcmp(argv[1], "--bench")) {
		/* `--bench [threads [rows [pos0]]]`; one thread is the
		 * serial arm, so the kernel's own split is kept */
		int th = argc > 2 ? atoi(argv[2]) : 1;

		if (th > 1)
			pool_start(th);
		else
			setenv("CHARSIU_ATTN_BLOCK_POOL", "0", 1);
		if (argc > 3) {
			bench(atoi(argv[3]), argc > 4 ? atoi(argv[4]) : 0, th);
			return 0;
		}
		bench(124, 0, th);
		bench(80, 80, th);
		bench(244, 0, th);
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

			if (arms[a][1] && strchr(arms[a][1], '=')) {
				char nm[64];
				const char *eq = strchr(arms[a][1], '=');

				snprintf(nm, sizeof(nm), "%.*s",
					 (int)(eq - arms[a][1]), arms[a][1]);
				setenv(nm, eq + 1, 1);
			} else if (arms[a][1]) {
				const char *val = !strcmp(arms[a][1],
					"CHARSIU_ATTN_AXPY8")
					|| !strcmp(arms[a][1], "CHARSIU_ATTN_HR")
					|| !strcmp(arms[a][1], "CHARSIU_ATTN_TILE")
					|| !strcmp(arms[a][1], "CHARSIU_ATTN_KV")
					? "0"
					: !strcmp(arms[a][1], "CHARSIU_ATTN_KV_ROWS")
					? "5"
					: !strcmp(arms[a][1], "CHARSIU_ATTN_KV_TB")
					? "7" : "1";

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
