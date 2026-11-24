// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * Time the decode attention on its own, at the Qwen3-0.6B shape: 28 layers,
 * 8 kv heads, 16 query heads, head_dim 128.
 *
 *   old    attn_heads for one row (the token loop), attn_block_heads for a
 *          verify pass of several rows -- what runs today
 *   gqa    attn_gqa_group over the float cache (CHARSIU_ATTN_GQA=1)
 *   kv16   attn_gqa_group over the half cache (CHARSIU_KV16=1)
 *
 * each serial and on the pool, a "token" being every layer once, at a list of
 * live positions. GB/s is the KV bytes one token has to read -- K and V of
 * every kv head of every layer, once, in the cache's own element size --
 * over the time; it is the same count for every arm of one cache type, so it
 * is a rate and not a claim about what the kernel actually fetched.
 *
 * IT INCLUDES src/llama.c, as tests/dattn_exact.c does, so what is timed is
 * the shipped kernels and not copies. The numbers are random and the answers
 * are thrown away; tests/dattn_exact.c is the correctness oracle.
 *
 * THE ARMS ARE INTERLEAVED REP BY REP in one process, for vattn_bench's
 * reason: this desk is shared, and two arms run one after the other meet
 * different interference. The median of the reps is printed.
 *
 * A DESK NUMBER IS NOT A BOARD NUMBER. This machine's cores and caches are
 * not the CM5's four A72s and their 1 MB of L2; the ratio between arms is
 * the transferable part, and only where the working set is far outside every
 * cache on both (the long positions).
 *
 *   dattn_bench [-p 128,512,1024,2048] [-m 1,6] [-T threads] [-r reps]
 */
#include "../src/llama.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NL 28
#define NKV 8
#define GQA 2
#define HD 128
#define NH (NKV * GQA)

static uint32_t seed = 777;

static void fill(float *p, size_t n)
{
	size_t i;

	for (i = 0; i < n; i++) {
		seed = seed * 1664525u + 1013904223u;
		p[i] = ((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.7f;
	}
}

static int cmpd(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;

	return x < y ? -1 : x > y;
}

enum { A_OLD, A_GQA, A_KV16, A_N };

static struct llama_state S;
static float *kc32, *vc32;
static uint16_t *kc16b, *vc16b;
static float *Q, *OUT;

static void arm_cache(int arm)
{
	S.kcache = kc32;
	S.vcache = vc32;
	S.kc16 = arm == A_KV16 ? kc16b : NULL;
	S.vc16 = arm == A_KV16 ? vc16b : NULL;
}

/* one token: every layer once, `rows` rows ending at position P - 1 */
static double token(int arm, int rows, int P, int pooled)
{
	double t0 = now_ms();
	float scale = 1.0f / sqrtf((float)HD);
	uint32_t l;

	arm_cache(arm);
	for (l = 0; l < NL; l++) {
		if (rows == 1) {
			struct attn_job aj = { &S, l, P - 1, 0, HD, HD,
					       NKV * HD, GQA, NKV, Q, scale };
			void (*fn)(void *, uint64_t, uint64_t) =
				arm == A_OLD ? attn_heads : attn_gqa_heads;

			if (pooled)
				pool_run(fn, &aj, NH, arm == A_OLD ? 1 : GQA);
			else
				fn(&aj, 0, NH);
		} else {
			struct attn_block_job bj = {
				&S, l, P - rows, rows, 0, 0, HD, HD, NKV * HD,
				GQA, NKV, NH, scale, Q, OUT, rows };

			if (arm == A_OLD) {
				if (pooled)
					pool_run(attn_block_heads, &bj, NH, 1);
				else
					attn_block_heads(&bj, 0, NH);
			} else {
				unsigned th = (unsigned)charsiu_threads();
				struct gqa_rows_job g = { &bj, 1 };

				if (pooled && th > NKV)
					g.split = (th + NKV - 1) / NKV;
				if (pooled)
					pool_run(attn_gqa_rows, &g,
						 (uint64_t)NKV * g.split, 1);
				else
					attn_gqa_rows(&g, 0, NKV);
			}
		}
	}
	return now_ms() - t0;
}

static int parse_list(const char *s, int *out, int max)
{
	int n = 0;

	while (*s && n < max) {
		out[n++] = atoi(s);
		while (*s && *s != ',')
			s++;
		if (*s == ',')
			s++;
	}
	return n;
}

int main(int argc, char **argv)
{
	int pl[16] = { 128, 512, 1024, 2048 }, np = 4;
	int ml[8] = { 1, 6 }, nm = 2;
	int threads = 4, reps = 7, n_ctx, i, a, pi, mi, pooled;
	size_t cells;

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-p") && i + 1 < argc)
			np = parse_list(argv[++i], pl, 16);
		else if (!strcmp(argv[i], "-m") && i + 1 < argc)
			nm = parse_list(argv[++i], ml, 8);
		else if (!strcmp(argv[i], "-T") && i + 1 < argc)
			threads = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-r") && i + 1 < argc)
			reps = atoi(argv[++i]);
		else {
			fprintf(stderr, "usage: dattn_bench [-p list] [-m list] "
				"[-T threads] [-r reps]\n");
			return 2;
		}
	}
	if (reps < 1)
		reps = 1;
	n_ctx = 0;
	for (i = 0; i < np; i++)
		if (pl[i] > n_ctx)
			n_ctx = pl[i];
	for (i = 0; i < nm; i++)
		if (ml[i] < 1 || ml[i] > GQA_VERIFY_ROWS) {
			fprintf(stderr, "dattn_bench: rows 1..%d\n",
				GQA_VERIFY_ROWS);
			return 2;
		}

	setenv("CHARSIU_KV16", "1", 1);   /* gqa_bases reads halves only with it */
	cells = (size_t)NL * NKV * n_ctx * HD;
	kc32 = malloc(cells * sizeof(float));
	vc32 = malloc(cells * sizeof(float));
	kc16b = malloc(cells * sizeof(uint16_t));
	vc16b = malloc(cells * sizeof(uint16_t));
	Q = malloc((size_t)GQA_VERIFY_ROWS * NH * HD * sizeof(float));
	OUT = malloc((size_t)GQA_VERIFY_ROWS * NH * HD * sizeof(float));
	S.n_ctx = n_ctx;
	S.att = calloc((size_t)NH * n_ctx, sizeof(float));
	S.xb = calloc((size_t)NH * HD, sizeof(float));
	S.batt = calloc((size_t)NH * GQA_VERIFY_ROWS * n_ctx, sizeof(float));
	if (!kc32 || !vc32 || !kc16b || !vc16b || !Q || !OUT || !S.att
	    || !S.xb || !S.batt) {
		fprintf(stderr, "dattn_bench: out of memory\n");
		return 1;
	}
	fill(kc32, cells);
	fill(vc32, cells);
	fill(Q, (size_t)GQA_VERIFY_ROWS * NH * HD);
	kv16_put(kc16b, kc32, (uint32_t)cells);
	kv16_put(vc16b, vc32, (uint32_t)cells);
	pool_start(threads);

	printf("dattn_bench: %d layers, %d kv heads, %d q heads, head_dim %d, "
	       "n_ctx %d, %d threads on the pool, median of %d\n",
	       NL, NKV, NH, HD, n_ctx, charsiu_threads(), reps);
	printf("%5s %5s %6s | %-24s | %-24s | %-24s\n", "rows", "pos",
	       "pool", "old fp32  ms   GB/s", "gqa fp32  ms   GB/s",
	       "kv16 fp16 ms   GB/s");
	for (mi = 0; mi < nm; mi++)
		for (pi = 0; pi < np; pi++)
			for (pooled = 0; pooled < 2; pooled++) {
				double t[A_N][64], med[A_N];
				int r, R = reps > 64 ? 64 : reps;

				for (a = 0; a < A_N; a++)
					token(a, ml[mi], pl[pi], pooled);
				for (r = 0; r < R; r++)
					for (a = 0; a < A_N; a++)
						t[a][r] = token(a, ml[mi],
								pl[pi], pooled);
				printf("%5d %5d %6s |", ml[mi], pl[pi],
				       pooled ? "yes" : "no");
				for (a = 0; a < A_N; a++) {
					double bytes = (double)NL * NKV * pl[pi]
						* HD * 2
						* (a == A_KV16 ? 2 : 4);

					qsort(t[a], R, sizeof(double), cmpd);
					med[a] = t[a][R / 2];
					printf(" %-9s %6.2f %6.2f  |", "",
					       med[a], bytes / (med[a] * 1e6));
				}
				printf("  gqa %.2fx  kv16 %.2fx\n",
				       med[A_OLD] / med[A_GQA],
				       med[A_OLD] / med[A_KV16]);
			}
	return 0;
}
