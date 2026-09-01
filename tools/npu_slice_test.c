// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * ONE TENSOR, SLICED, BATCHED, AGAINST THE CPU. The rung that was missing.
 *
 * The wide K slice fault has been chased through three layers of inference and
 * three wrong hypotheses. What is established:
 *
 *   charsiu_matmul   ONE dispatch, int4, K=4096 N=1536 m=80  -> EXACT
 *   board phase 13   a whole model at KMAX 3072 or 4096      -> WRONG
 *
 * so it is not the register stream for a call, and it is not the quantiser,
 * and it is not the core pair (Qwen2.5 fails identically on one core). What is
 * between those two rungs is: several K slices accumulated into one Y, through
 * npudev's staging, with the batched buffers shared across them. Nothing in
 * this tree exercises exactly that and nothing else.
 *
 * This does. It builds one synthetic tensor, stages it through the same
 * charsiu_npu_add the runtime uses, runs charsiu_npu_matmul at m rows, and
 * compares against npu_matvec -- the CPU path, reading the SAME quantised
 * weights, which is the same comparison phase 2 makes at model scale.
 *
 * ⚠ m=1 IS THE CONTROL AND RUNS FIRST. charsiu_npu_matvec on the same staged
 * tensor is the decode path; if it disagrees then the harness is wrong and
 * says so, rather than blaming the slicing it was built to examine.
 *
 *   npu_slice_test K N M [KMAX]
 *
 * KMAX defaults to CHARSIU_NPU_KMAX or 1024. The number of slices is
 * ceil(K/KMAX) and is printed, because a run at one slice proves nothing about
 * a fault that needs several.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu.h"
#include "charsiu_llm.h"

static float frand(unsigned *s)
{
	*s = *s * 1103515245u + 12345u;
	return (float)((int)((*s >> 9) & 0xffff) - 32768) / 32768.0f;
}

int main(int argc, char **argv)
{
	unsigned k = argc > 1 ? (unsigned)atoi(argv[1]) : 8960;
	unsigned n = argc > 2 ? (unsigned)atoi(argv[2]) : 1536;
	unsigned m = argc > 3 ? (unsigned)atoi(argv[3]) : 80;
	const char *kmaxe = argc > 4 ? argv[4] : getenv("CHARSIU_NPU_KMAX");
	unsigned kmax = kmaxe ? (unsigned)atoi(kmaxe) : 1024;
	unsigned seed = 12345, r, j, bad = 0, ctrl_bad = 0;
	struct gguf_tensor w = { 0 };
	struct npu_tensor t = { 0 };
	struct charsiu_npu *g;
	struct charsiu_act act;
	float *raw, *X, *Ynpu, *Ycpu, worst = 0.0f;
	int id;

	if (argc > 4)
		setenv("CHARSIU_NPU_KMAX", argv[4], 1);

	printf("K=%u N=%u m=%u KMAX=%u -> %u slice(s), widest %u,"
	       " surface %u entries\n", k, n, m, kmax,
	       (k + kmax - 1) / kmax, k < kmax ? k : kmax,
	       ((k < kmax ? k : kmax) / 32) * m);
	if ((k + kmax - 1) / kmax < 2)
		printf("  ⚠ ONE SLICE: this cannot show a fault that needs"
		       " several. Raise K or lower KMAX.\n");

	raw = malloc((size_t)n * k * sizeof(float));
	X = malloc((size_t)m * k * sizeof(float));
	Ynpu = malloc((size_t)m * n * sizeof(float));
	Ycpu = malloc((size_t)m * n * sizeof(float));
	if (!raw || !X || !Ynpu || !Ycpu) {
		fprintf(stderr, "out of memory\n");
		return 1;
	}
	for (size_t i = 0; i < (size_t)n * k; i++)
		raw[i] = frand(&seed) * 0.05f;
	for (size_t i = 0; i < (size_t)m * k; i++)
		X[i] = frand(&seed);

	snprintf(w.name, sizeof(w.name), "slice_test");
	w.n_dims = 2; w.ne[0] = k; w.ne[1] = n;
	w.type = GGML_F32; w.data = raw;
	w.nbytes = (size_t)n * k * 4;
	if (npu_tensor_build(&t, &w)) {
		fprintf(stderr, "npu_tensor_build failed\n");
		return 1;
	}
	if (charsiu_act_alloc(&act, (int)k)) {
		fprintf(stderr, "act alloc failed\n");
		return 1;
	}

	g = charsiu_npu_open(k, n, 1);
	if (!g) {
		fprintf(stderr, "no /dev/accel/accel0\n");
		return 1;
	}
	id = charsiu_npu_add(g, &t);
	if (id < 0) {
		fprintf(stderr, "the tensor was not staged\n");
		return 1;
	}

	/* the CPU answer, row by row, from the same quantised weights */
	for (r = 0; r < m; r++) {
		charsiu_act_set(&act, X + (size_t)r * k, (int)k);
		npu_matvec(&t, &act, Ycpu + (size_t)r * n, 0, n);
	}

	/* ⚠ THE CONTROL FIRST: m=1 through the decode path on the same tensor */
	charsiu_act_set(&act, X, (int)k);
	memset(Ynpu, 0, (size_t)n * sizeof(float));
	if (charsiu_npu_matvec(g, id, &act, Ynpu)) {
		printf("  ⚠ the m=1 control did not run at all\n");
		return 1;
	}
	for (j = 0; j < n; j++) {
		float d = fabsf(Ynpu[j] - Ycpu[j]);
		float s = fabsf(Ycpu[j]) + 1e-6f;

		if (d / s > 1e-3f)
			ctrl_bad++;
	}
	printf("  m=1 control: %u of %u channels off by more than 0.1%%%s\n",
	       ctrl_bad, n, ctrl_bad ? "  ⚠ THE HARNESS IS WRONG, STOP HERE" : "");
	if (ctrl_bad)
		return 1;

	memset(Ynpu, 0, (size_t)m * n * sizeof(float));
	if (charsiu_npu_matmul(g, id, X, m, Ynpu)) {
		printf("  ⚠ the batched call was REFUSED -- see the whine above."
		       " Nothing was compared.\n");
		return 1;
	}
	for (r = 0; r < m; r++)
		for (j = 0; j < n; j++) {
			size_t i = (size_t)r * n + j;
			float d = fabsf(Ynpu[i] - Ycpu[i]);
			float s = fabsf(Ycpu[i]) + 1e-6f;

			if (d / s > worst)
				worst = d / s;
			if (d / s > 1e-3f)
				bad++;
		}
	printf("  m=%u batched: %u of %zu values off by more than 0.1%%,"
	       " worst %.3e\n", m, bad, (size_t)m * n, worst);
	printf("  %s\n", bad ? "⚠⚠ THE SLICED BATCH DISAGREES WITH THE CPU"
			     : "exact");
	charsiu_npu_close(g);
	return bad ? 1 : 0;
}
