// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The batched fp16 activation pack, old against CHARSIU_NPU_PACK2, on the CPU
 * alone and in plain memory -- no device, so it runs on the desk and on a
 * board whose NPU is busy with something else (it still competes for the CPU
 * and the DRAM, so not beside a timing run).
 *
 * One "prompt" is what batch_pack_slice does for one chunk of m rows through
 * a whole model: per layer the four grouped calls (qkv, o, gate/up, down),
 * each K cut at kmax, each slice packed into BOTH devices' objects, and every
 * object cleaned to the point of coherency afterwards the way FINI_BO's
 * dma_sync does (dc cvac, so the next call meets what the board meets; it is
 * timed apart and not counted in the pack). Before each call the activation
 * is rewritten, which is what the rmsnorm or the activation ahead of it does.
 *
 * Arms, interleaved within every repetition so drift hits them alike:
 *   old     gather a cut slice into scratch, pool split by groups, pack twice
 *   rows    the pool split by rows, gather kept, pack twice
 *   direct  rows, and a cut slice read straight out of X
 *   pack2   direct, and the second device copies the first one's object
 *
 *   pack2_bench [m] [threads] [reps] [layers] [kmax] [K list...]
 * defaults: 124 4 5 28 1024 1024 2048 1024 3072 (Qwen3-0.6B)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "charsiu.h"
#include "charsiu_llm.h"

static double now_us(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void clean_poc(const void *p, size_t n)
{
#if defined(__aarch64__)
	uintptr_t a = (uintptr_t)p & ~(uintptr_t)63, e = (uintptr_t)p + n;

	for (; a < e; a += 64)
		__asm__ volatile("dc cvac, %0" :: "r"(a) : "memory");
	__asm__ volatile("dsb sy" ::: "memory");
#else
	(void)p; (void)n;
#endif
}

struct pj {
	const struct charsiu_matmul *mm;
	const float *src;
	size_t stride;
	uint8_t *dst;
};

static void w_groups(void *c, uint64_t a, uint64_t n)
{
	struct pj *j = c;

	charsiu_pack_input_f16_groups(j->mm, j->src, j->stride, j->dst,
				      (unsigned)a, (unsigned)n);
}

static void w_rows(void *c, uint64_t a, uint64_t n)
{
	struct pj *j = c;

	charsiu_pack_input_f16_rows(j->mm, j->src, j->stride, j->dst,
				    (unsigned)a, (unsigned)n);
}

#define CHUNK 16384
struct cj { const uint8_t *s; uint8_t *d; size_t n; };

static void w_copy(void *c, uint64_t a, uint64_t n)
{
	struct cj *j = c;
	size_t lo = (size_t)a * CHUNK, hi = (size_t)(a + n) * CHUNK;

	if (hi > j->n)
		hi = j->n;
	memcpy(j->d + lo, j->s + lo, hi - lo);
}

enum { OLD, ROWS, DIRECT, PACK2, NARM };
static const char *arm_name[NARM] = { "old", "rows", "direct", "pack2" };

int main(int argc, char **argv)
{
	unsigned m = argc > 1 ? (unsigned)atoi(argv[1]) : 124;
	int nth = argc > 2 ? atoi(argv[2]) : 4;
	int reps = argc > 3 ? atoi(argv[3]) : 5;
	unsigned layers = argc > 4 ? (unsigned)atoi(argv[4]) : 28;
	unsigned kmax = argc > 5 ? (unsigned)atoi(argv[5]) : 1024;
	unsigned K[16] = { 1024, 2048, 1024, 3072 }, nk = 4, kbig = 0;
	size_t bs;
	uint8_t *bin[2][16];
	float *X, *scr;
	double best[NARM], sum[NARM], fini = 0;
	uint64_t hash[NARM];
	unsigned elems = 0;

	if (argc > 6) {
		nk = 0;
		for (int i = 6; i < argc && nk < 16; i++)
			K[nk++] = (unsigned)atoi(argv[i]);
	}
	for (unsigned i = 0; i < nk; i++) {
		if (K[i] > kbig)
			kbig = K[i];
		elems += K[i];
	}
	bs = (size_t)kmax * m * 2;
	X = malloc((size_t)m * kbig * sizeof(float));
	scr = malloc((size_t)m * kmax * sizeof(float));
	for (int d = 0; d < 2; d++)
		for (unsigned i = 0; i < (kbig + kmax - 1) / kmax; i++) {
			bin[d][i] = aligned_alloc(4096, (bs + 4096 + 4095) & ~(size_t)4095);
			memset(bin[d][i], 0, bs + 4096);
		}
	charsiu_threads_start(nth);
	for (int a = 0; a < NARM; a++) {
		best[a] = 1e30;
		sum[a] = 0;
		hash[a] = 1469598103934665603ull;
	}
	for (int rep = 0; rep < reps; rep++)
		for (int a0 = 0; a0 < NARM; a0++) {
			int a = (a0 + rep) % NARM;
			double t = 0, tf = 0;

			for (unsigned l = 0; l < layers; l++)
				for (unsigned ti = 0; ti < nk; ti++) {
					unsigned k = K[ti], nks = (k + kmax - 1) / kmax;
					double t0;

					for (size_t i = 0; i < (size_t)m * k; i++)
						X[i] = (float)((int)(((i + l * 131 + ti) * 2654435761u) >> 18 & 4095) - 2048) * 0.0013f;
					t0 = now_us();
					for (int d = 0; d < 2; d++) {
						double f0;

						for (unsigned ki = 0; ki < nks; ki++) {
							unsigned k0 = ki * kmax, sk = k - k0 < kmax ? k - k0 : kmax;
							struct charsiu_matmul mm = { m, sk, 64, CHARSIU_FP16, CHARSIU_FP16 };
							struct pj j = { &mm, X + k0, k, bin[d][ki] };

							if (a == PACK2 && d == 1) {
								size_t used = (size_t)(sk / 8) * m * 16 + (sk % 8 ? (size_t)m * 16 : 0);
								struct cj c = { bin[0][ki], bin[1][ki], used };

								if (used >= 4 * CHUNK)
									charsiu_parallel_for(w_copy, &c, (used + CHUNK - 1) / CHUNK);
								else
									memcpy(c.d, c.s, used);
								if (bs > used)
									memset(c.d + used, 0, bs - used);
								continue;
							}
							if (sk != k && (a == OLD || a == ROWS)) {
								for (unsigned r = 0; r < m; r++)
									memcpy(scr + (size_t)r * sk, X + (size_t)r * k + k0, sk * sizeof(float));
								j.src = scr;
								j.stride = sk;
							}
							charsiu_pack_input_f16_edges(&mm, j.src, j.stride, j.dst, bs);
							if (a == OLD)
								charsiu_parallel_for(w_groups, &j, sk / 8);
							else
								charsiu_parallel_for_grain(w_rows, &j, m, 4);
						}
						f0 = now_us();
						for (unsigned ki = 0; ki < nks; ki++)
							clean_poc(bin[d][ki], bs + 4096);
						tf += now_us() - f0;
					}
					t += now_us() - t0;
					if (l == layers - 1 && rep == reps - 1)
						for (int d = 0; d < 2; d++)
							for (unsigned ki = 0; ki < nks; ki++)
								for (size_t i = 0; i < bs; i++)
									hash[a] = (hash[a] ^ bin[d][ki][i]) * 1099511628211ull;
				}
			t -= tf;
			fini += tf;
			sum[a] += t;
			if (t < best[a])
				best[a] = t;
		}
	printf("pack2_bench: m=%u threads=%d reps=%d layers=%u kmax=%u K=", m, nth, reps, layers, kmax);
	for (unsigned i = 0; i < nk; i++)
		printf("%u%s", K[i], i + 1 < nk ? "," : "\n");
	for (int a = 0; a < NARM; a++)
		printf("  %-7s best %8.2f ms  mean %8.2f ms  %.3f ms/row  %.2f GB/s written  %.2fx  bytes %016llx%s\n",
		       arm_name[a], best[a] / 1e3, sum[a] / reps / 1e3,
		       best[a] / 1e3 / m,
		       (double)m * elems * 2 * 2 * layers / best[a] / 1e3,
		       best[OLD] / best[a], (unsigned long long)hash[a],
		       hash[a] == hash[OLD] ? "" : "  DIFFERENT");
	printf("  (clean to PoC, not counted: %.1f ms a prompt)\n", fini / reps / NARM / 1e3);
	for (int a = 0; a < NARM; a++)
		if (hash[a] != hash[OLD])
			return 1;
	return 0;
}
