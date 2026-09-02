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
	/*
	 * ⚠ THE SEED IS A KNOB BECAUSE THE ROW INDEX IS THE EVIDENCE. A single
	 * wrong row at a fixed index is either structure or one row of random
	 * data sitting on a quantisation boundary, and those two look identical
	 * in one run. Move the data: structure keeps the row, a boundary does
	 * not.
	 */
	const char *es = getenv("CHARSIU_SLICE_SEED");
	unsigned seed = es ? (unsigned)strtoul(es, NULL, 0) : 12345;
	unsigned r, j, bad = 0;
	const char *e4 = getenv("CHARSIU_NPU_W4V");
	int w4v = e4 && *e4 && *e4 != '0';
	struct gguf_tensor w = { 0 };
	struct npu_tensor t = { 0 };
	struct charsiu_npu *g;
	struct charsiu_act act;
	float *raw, *X, *Yb, *Yref, *Ycpu, worst = 0.0f;
	int id;

	if (argc > 4)
		setenv("CHARSIU_NPU_KMAX", argv[4], 1);

	/*
	 * ⚠ THE SURFACE IS NOT ONE FORMULA, and printing it as if it were is
	 * how the guard in npudev.c came to reach a path it was never measured
	 * on. charsiu_emit_job sets inw = m only on the WIDTH axis, so int4's
	 * surface is (slice / 32) * m and int8's is (slice / 32) * 1 -- the m
	 * rows go on the height there and cost the input nothing. Name the axis
	 * and print the number that actually applies to it.
	 */
	{
		int wideax = charsiu_m_axis_wide_for(w4v);
		unsigned slice = k < kmax ? k : kmax;

		printf("K=%u N=%u m=%u KMAX=%u -> %u slice(s), widest %u,"
		       " %s axis, surface %u entries\n", k, n, m, kmax,
		       (k + kmax - 1) / kmax, slice,
		       wideax ? "width" : "height",
		       (slice / 32) * (wideax ? m : 1));
	}
	if ((k + kmax - 1) / kmax < 2)
		printf("  ⚠ ONE SLICE: this cannot show a fault that needs"
		       " several. Raise K or lower KMAX.\n");

	raw = malloc((size_t)n * k * sizeof(float));
	X = malloc((size_t)m * k * sizeof(float));
	Yb = malloc((size_t)m * n * sizeof(float));
	Yref = malloc((size_t)m * n * sizeof(float));
	Ycpu = malloc((size_t)n * sizeof(float));
	if (!raw || !X || !Yb || !Yref || !Ycpu) {
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

	/*
	 * ⚠⚠ THE HALF A DESK CAN VERIFY, AND IT RUNS BEFORE THE HARDWARE IS
	 * OPENED. The reference below is the NPU's own row loop and needs a
	 * board; the tensor and the activation do not -- and the bug this file
	 * shipped with was exactly there. charsiu_act_set quantises NOTHING,
	 * npu_matvec reads a->q1, and charsiu_act_q1 fills it only under
	 * CHARSIU_NPU_QUANT=1. A machine with no /dev/accel can see all of
	 * that, and could not, because the open used to come first.
	 */
	charsiu_act_set(&act, X, (int)k);
	charsiu_act_q1(&act);
	npu_matvec(&t, &act, Ycpu, 0, n);
	{
		unsigned nz = 0;

		for (j = 0; j < n; j++)
			if (Ycpu[j] != 0.0f)
				nz++;
		printf("  CPU on row 0: %u of %u channels non-zero\n", nz, n);
		if (!nz) {
			printf("  ⚠⚠ THE HARNESS CANNOT COMPUTE ANYTHING.\n"
			       "     npu_matvec reads a->q1 and charsiu_act_q1\n"
			       "     fills it only when CHARSIU_NPU_QUANT=1%s.\n",
			       getenv("CHARSIU_NPU_QUANT")
			       ? " -- and it IS set, so this is something else"
			       : ", which it is NOT");
			return 1;
		}
	}

	/*
	 * ⚠⚠ AND THE TWO QUANTISERS, SIDE BY SIDE, ON A DESK. charsiu_act_q1
	 * multiplies by 1/d and the batched packer in npudev.c used to divide
	 * by d. Those are not the same float -- the reciprocal rounds once and
	 * the product rounds again -- so a value halfway between two codes came
	 * out one code apart, and one code out of 3072 moves every one of the n
	 * outputs by a hair. On the board that read as "1 row of 64 wrong, 127
	 * of 49152 channels, mean got/want 1.0007": small enough to look like
	 * noise, stable enough to look like structure, and it cost a round to
	 * tell those apart.
	 *
	 * It never needed one. This loop reproduces the board's row index from
	 * the same seed with no /dev/accel in the machine, and the seed sweep
	 * that would have been a board round is five runs of this line.
	 */
	{
		unsigned qdiff = 0, rows = 0, firstr = m;

		for (r = 0; r < m; r++) {
			float amax = 0.0f, d, id;
			unsigned any = 0;

			for (j = 0; j < k; j++) {
				float a = fabsf(X[(size_t)r * k + j]);

				if (a > amax) amax = a;
			}
			d = amax / 127.0f;
			id = d != 0.0f ? 1.0f / d : 0.0f;
			for (j = 0; j < k; j++) {
				float x = X[(size_t)r * k + j];
				int a = (int)lrintf(x * id);
				int b = d != 0.0f ? (int)lrintf(x / d) : 0;

				if (a > 127) a = 127;
				if (a < -127) a = -127;
				if (b > 127) b = 127;
				if (b < -127) b = -127;
				if (a != b) { any = 1; qdiff++; }
			}
			if (any) {
				rows++;
				if (r < firstr) firstr = r;
			}
		}
		printf("  quantiser check (no hardware): x*(1/d) vs x/d differ on"
		       " %u code(s) in %u of %u rows%s\n", qdiff, rows, m,
		       rows ? "" : " -- none");
		if (rows)
			printf("    first such row is %u. If the batch below is"
			       " wrong on exactly these rows,\n    that is this"
			       " and not the batching.\n", firstr);
	}

	g = charsiu_npu_open(k, n, 1);
	if (!g) {
		fprintf(stderr, "no /dev/accel/accel0 -- the CPU check above"
			" still ran, the comparison did not\n");
		return 1;
	}
	id = charsiu_npu_add(g, &t);
	if (id < 0) {
		fprintf(stderr, "the tensor was not staged\n");
		return 1;
	}

	/*
	 * ⚠⚠ THE REFERENCE IS THE NPU'S OWN ROW LOOP, NOT THE CPU'S.
	 *
	 * Two earlier versions compared the batch against npu_matvec and their
	 * controls failed on 1536 channels and then 1327. Neither was the
	 * runtime: npu_matvec reads an int8 quantised activation and w4a16
	 * hands the hardware the FLOAT one, which charsiu_npu_needs_q1 says in
	 * one line above its own declaration. Two arithmetics that differ
	 * cannot agree to 0.1% and it is not a bug that they do not.
	 *
	 * Phase 2 compares the batched prefill against the TOKEN LOOP, and the
	 * token loop is charsiu_npu_matvec. Same call, same weights, same
	 * activation, one row at a time. Whatever is left is the batching.
	 */
	for (r = 0; r < m; r++) {
		charsiu_act_set(&act, X + (size_t)r * k, (int)k);
		if (charsiu_npu_needs_q1(g))
			charsiu_act_q1(&act);
		if (charsiu_npu_matvec(g, id, &act, Yref + (size_t)r * n)) {
			printf("  ⚠ the reference did not run at row %u\n", r);
			return 1;
		}
	}
	{
		unsigned nz = 0, nd = 0, far = 0;
		float first = Yref[0];

		for (size_t i = 0; i < (size_t)m * n; i++) {
			if (Yref[i] != 0.0f)
				nz++;
			if (Yref[i] != first)
				nd++;
		}
		printf("  reference (NPU, row by row): %u of %zu non-zero,"
		       " %u distinct\n", nz, (size_t)m * n, nd);
		if (!nz || !nd) {
			printf("  ⚠⚠ THE REFERENCE IS DEGENERATE -- the harness,"
			       " not the hardware.\n");
			return 1;
		}
		/*
		 * ⚠ AND LOOSELY AGAINST THE CPU, at 10% and not 0.1%. The
		 * activation quantisation makes a tight comparison meaningless
		 * here; this only has to catch the hardware returning noise.
		 */
		for (j = 0; j < n; j++)
			if (fabsf(Yref[j] - Ycpu[j])
			    > 0.10f * (fabsf(Ycpu[j]) + 1e-6f))
				far++;
		printf("  sanity vs the CPU on row 0: %u of %u channels more"
		       " than 10%% apart%s\n", far, n,
		       far > n / 2 ? "  ⚠ the reference is not this tensor" : "");
		if (far > n / 2)
			return 1;
	}

	memset(Yb, 0, (size_t)m * n * sizeof(float));
	if (charsiu_npu_matmul(g, id, X, m, Yb)) {
		printf("  ⚠ the batched call was REFUSED -- see the whine"
		       " above. Nothing was compared.\n");
		return 1;
	}
	for (r = 0; r < m; r++)
		for (j = 0; j < n; j++) {
			size_t i = (size_t)r * n + j;
			float d = fabsf(Yb[i] - Yref[i]);
			float sc = fabsf(Yref[i]) + 1e-6f;

			if (d / sc > worst)
				worst = d / sc;
			if (d / sc > 1e-3f)
				bad++;
		}
	printf("  m=%u batched vs the same call row by row: %u of %zu off by"
	       " more than 0.1%%, worst %.3e\n", m, bad, (size_t)m * n, worst);
	printf("  %s\n", bad ? "⚠⚠ THE SLICED BATCH DISAGREES WITH ITS OWN"
			       " ROW LOOP" : "exact");

	/*
	 * ⚠⚠ WHERE IT IS WRONG IS THE DIAGNOSIS; THAT IT IS WRONG IS NOT.
	 *
	 * This fault has been chased for a day on "a model answers differently"
	 * and every hypothesis fitted to that died. The shape of the error
	 * separates the remaining candidates on its own:
	 *
	 *   every row wrong, all channels   -> the whole accumulation
	 *   some rows only                  -> the m axis, the read order
	 *   a channel range only            -> an output buffer offset
	 *   ratio near (ks-1)/ks            -> a slice's contribution missing
	 *
	 * so print all four rather than make the next round guess again.
	 */
	if (bad) {
		unsigned rows_bad = 0, r0 = m, rlast = 0;
		unsigned c0 = n, clast = 0;
		double rsum = 0.0;
		unsigned rn = 0;

		for (r = 0; r < m; r++) {
			unsigned any = 0;

			for (j = 0; j < n; j++) {
				size_t i = (size_t)r * n + j;
				float sc = fabsf(Yref[i]) + 1e-6f;

				if (fabsf(Yb[i] - Yref[i]) / sc <= 1e-3f)
					continue;
				any = 1;
				if (j < c0) c0 = j;
				if (j > clast) clast = j;
				if (fabsf(Yref[i]) > 1e-4f) {
					rsum += Yb[i] / Yref[i];
					rn++;
				}
			}
			if (any) {
				rows_bad++;
				if (r < r0) r0 = r;
				if (r > rlast) rlast = r;
			}
		}
		printf("    rows wrong     %u of %u, first %u last %u%s\n",
		       rows_bad, m, r0, rlast,
		       rows_bad == 1 ? "   (ONE row: rerun with"
				       " CHARSIU_SLICE_SEED to see if it moves)"
				     : "");
		printf("    channels       first %u last %u of %u\n",
		       c0, clast, n);
		if (rn)
			printf("    mean got/want  %.4f over %u values%s\n",
			       rsum / rn, rn,
			       fabs(rsum / rn - 1.0) < 0.5
			       ? "" : "   (far from 1: not a rounding difference)");
		/*
		 * ⚠⚠ AND HOW BIG THE DISAGREEING VALUES ARE, because a 0.1%
		 * RELATIVE threshold on a channel whose true value is a
		 * thousandth of the row is not a correctness test -- it is a
		 * test of whether two int8 activation codes rounded the same
		 * way. Print the wrong cells' |want| against their own row's
		 * largest, so the next reader does not have to assume.
		 */
		{
			double frac_sum = 0.0, frac_max = 0.0;
			unsigned fn = 0;

			for (r = 0; r < m; r++) {
				float rmax = 0.0f;

				for (j = 0; j < n; j++) {
					float a = fabsf(Yref[(size_t)r * n + j]);

					if (a > rmax) rmax = a;
				}
				if (rmax <= 0.0f)
					continue;
				for (j = 0; j < n; j++) {
					size_t i = (size_t)r * n + j;
					float sc = fabsf(Yref[i]) + 1e-6f;
					double f;

					if (fabsf(Yb[i] - Yref[i]) / sc <= 1e-3f)
						continue;
					f = fabsf(Yref[i]) / rmax;
					frac_sum += f;
					if (f > frac_max) frac_max = f;
					fn++;
				}
			}
			if (fn)
				printf("    magnitude      the wrong cells are"
				       " %.4f of their row's largest on average,"
				       " %.4f at most%s\n",
				       frac_sum / fn, frac_max,
				       frac_max < 0.05
				       ? "\n                   -> all of them are"
					 " near-zero channels, where a 0.1%%"
					 " RELATIVE\n                      threshold"
					 " measures rounding and not correctness"
				       : "");
		}
		{
			unsigned ks = (k + kmax - 1) / kmax;

			printf("    for reference, (ks-1)/ks with ks=%u is %.4f"
			       " -- a missing slice would land there\n",
			       ks, (double)(ks - 1) / ks);
		}
	}
	charsiu_npu_close(g);
	return bad ? 1 : 0;
}
