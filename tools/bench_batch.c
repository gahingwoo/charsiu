/*
 * Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
 * SPDX-License-Identifier: GPL-2.0
 *
 * Does batching the projections pay on THIS machine?
 *
 * ⚠ IT HAS TO WALK EVERY LAYER. The first version of this looped on one
 * tensor 200 times, which left it in cache and measured arithmetic rather
 * than memory: it reported 1.09x where the question is entirely about how
 * often the weights are read from DRAM. Walking the model streams them the
 * way a forward pass does.
 *
 * On the development host (a big cache, wide memory) this says 1.1x to 1.4x.
 * The board is 10 GB/s of LPDDR5 behind four A72s, where the same kernel is
 * far closer to the memory roof, so the number that decides whether a batched
 * forward is worth building is the one THIS prints on the board.
 *
 *   bench_batch MODEL.gguf M
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "charsiu_llm.h"
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
/* Walk EVERY layer, the way a forward pass does, so the weights are streamed
 * from memory instead of sitting in cache like a single-tensor loop. */
int main(int argc, char **argv)
{
	(void)argc;
	struct llama_model m;
	struct charsiu_act a[CHARSIU_BATCH_MAX];
	static const unsigned Ms[] = { 1, 2, 4, 8, 16 };
	uint64_t kmax = 0, nmax = 0;
	unsigned j;

	if (llama_load(&m, argv[1]) < 0)
		return 1;
	for (uint32_t l = 0; l < m.n_layer; l++) {
		const struct gguf_tensor *t[4] = { m.layers[l].wq, m.layers[l].wo,
						   m.layers[l].gate, m.layers[l].down };
		for (int i = 0; i < 4; i++) {
			if (t[i]->ne[0] > kmax) kmax = t[i]->ne[0];
			if (t[i]->ne[1] > nmax) nmax = t[i]->ne[1];
		}
	}
	float *y = calloc(nmax * CHARSIU_BATCH_MAX, sizeof(float));
	float *x = malloc(kmax * sizeof(float));

	for (unsigned i = 0; i < kmax; i++)
		x[i] = (float)(i % 17) / 8.0f - 1.0f;
	for (j = 0; j < CHARSIU_BATCH_MAX; j++) {
		memset(&a[j], 0, sizeof(a[j]));
		charsiu_act_alloc(&a[j], (int)kmax);
	}

	/*
	 * ⚠ ONE M IS NOT AN ANSWER. The board reported M=8 SLOWER than one at a
	 * time (1855 ms against 1711), which rules out M=8 and says nothing
	 * about M=2. Four A72s have 32 NEON registers and 32 KB of L1: eight
	 * accumulators and eight activation streams do not fit, and narrower
	 * ones might. Sweep, and let the machine say where the crossover is.
	 */
	/*
	 * ⚠ WARM UP FIRST. Without this the first M measured pays for every
	 * cold page of the model and reported 2.74x for M=1 -- where the two
	 * sides are LITERALLY the same code, since gguf_matmul(m=1) calls
	 * gguf_matvec. A ratio that cannot be anything but 1.00 coming out at
	 * 2.74 is the benchmark measuring itself.
	 */
	for (uint32_t l = 0; l < m.n_layer; l++) {
		const struct gguf_tensor *t[4] = { m.layers[l].wq, m.layers[l].wo,
						   m.layers[l].gate, m.layers[l].down };
		for (int i = 0; i < 4; i++) {
			charsiu_act_set(&a[0], x, (int)t[i]->ne[0]);
			charsiu_act_blocks(&a[0]);
			gguf_matvec(t[i], &a[0], y, 0, t[i]->ne[1]);
		}
	}

	printf("  %-6s %12s %12s %8s\n", "M", "per token", "batched", "ratio");
	for (unsigned mi = 0; mi < sizeof(Ms) / sizeof(*Ms); mi++) {
		unsigned M = Ms[mi];
		double t0, t1, per, bat;
		int R = 3;

		t0 = ms();
		for (int r = 0; r < R; r++)
			for (uint32_t l = 0; l < m.n_layer; l++) {
				const struct gguf_tensor *t[4] = { m.layers[l].wq,
					m.layers[l].wo, m.layers[l].gate, m.layers[l].down };
				for (int i = 0; i < 4; i++)
					for (j = 0; j < M; j++) {
						charsiu_act_set(&a[j], x, (int)t[i]->ne[0]);
						charsiu_act_blocks(&a[j]);
						gguf_matvec(t[i], &a[j], y + j * nmax, 0, t[i]->ne[1]);
					}
			}
		t1 = ms(); per = (t1 - t0) / R;

		t0 = ms();
		for (int r = 0; r < R; r++)
			for (uint32_t l = 0; l < m.n_layer; l++) {
				const struct gguf_tensor *t[4] = { m.layers[l].wq,
					m.layers[l].wo, m.layers[l].gate, m.layers[l].down };
				for (int i = 0; i < 4; i++) {
					for (j = 0; j < M; j++) {
						charsiu_act_set(&a[j], x, (int)t[i]->ne[0]);
						charsiu_act_blocks(&a[j]);
					}
					gguf_matmul(t[i], a, M, y, nmax, 0, t[i]->ne[1]);
				}
			}
		t1 = ms(); bat = (t1 - t0) / R;

		printf("  %-6u %9.1f ms %9.1f ms %7.2fx%s\n", M, per, bat,
		       bat > 0 ? per / bat : 0.0,
		       (per / bat) > 1.25 ? "  <-- worth it" : "");
	}
	return 0;
}
