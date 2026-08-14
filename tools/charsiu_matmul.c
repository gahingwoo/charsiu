// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * One matmul on the NPU through charsiu alone, checked against the CPU.
 *
 * This is the whole project in miniature: pack the operands into the tile
 * layouts, build the coefficient buffer, emit the register stream, submit it
 * through rocket, and compare what comes back with the same arithmetic done on
 * the cores. Nothing from Mesa or the vendor is in the path.
 *
 * It prints what it did before it prints whether it worked, because a run that
 * fails is only useful if the shape and the buffers it used are on the record
 * beside the failure.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "charsiu.h"

static int cpu_reference(const struct charsiu_job *job, const uint8_t *a,
			 const uint8_t *b, const int32_t *bias, uint8_t *out)
{
	const struct charsiu_matmul *mm = &job->mm;
	float mult = job->input_scale * job->weight_scale / job->output_scale;
	unsigned m, n, k;

	for (m = 0; m < mm->m; m++) {
		for (n = 0; n < mm->n; n++) {
			int64_t acc = bias[n];
			float v;

			for (k = 0; k < mm->k; k++)
				acc += ((int)a[m * mm->k + k] - job->input_zero_point) *
				       ((int)b[n * mm->k + k] - job->weight_zero_point);

			/* The hardware rounds one shift half up; the reference
			 * has to do the same or every value is off by one. */
			v = (float)acc * mult + (float)job->output_zero_point;
			v = (float)(long)(v + 0.5f);
			if (v < 0) v = 0;
			if (v > 255) v = 255;
			out[m * mm->n + n] = (uint8_t)v;
		}
	}
	return 0;
}

int main(int argc, char **argv)
{
	struct charsiu_job job = { 0 };
	struct charsiu_device *dev;
	struct charsiu_bo regcmd = { 0 }, in = { 0 }, wt = { 0 }, outbo = { 0 },
			   coef = { 0 };
	uint8_t *a_raw, *b_raw, *ref;
	int32_t *bias, *wsums;
	uint32_t in_handles[3], out_handles[1];
	unsigned m, n, k, i, atom, bad = 0, nonzero = 0;
	size_t nreg;
	int ret;

	job.mm.m = argc > 1 ? (unsigned)atoi(argv[1]) : 4;
	job.mm.k = argc > 2 ? (unsigned)atoi(argv[2]) : 64;
	job.mm.n = argc > 3 ? (unsigned)atoi(argv[3]) : 32;
	job.mm.wdtype = CHARSIU_INT8;
	job.mm.adtype = CHARSIU_INT8;
	job.input_scale = 0.02f;
	job.weight_scale = 0.01f;
	job.output_scale = 0.25f;
	job.input_zero_point = 128;
	job.weight_zero_point = 128;
	job.output_zero_point = 128;
	m = job.mm.m; k = job.mm.k; n = job.mm.n;
	atom = charsiu_feature_atom(job.mm.adtype);

	printf("matmul M=%u K=%u N=%u int8, feature atom %u, %u entries per row\n",
	       m, k, n, atom, charsiu_entries_per_row(&job.mm));

	dev = charsiu_open(NULL);
	if (!dev) { printf("open FAILED\n"); return 1; }

	a_raw = malloc((size_t)m * k);
	b_raw = malloc((size_t)n * k);
	ref = malloc((size_t)m * n);
	bias = calloc(n, sizeof(*bias));
	wsums = calloc(n, sizeof(*wsums));
	for (i = 0; i < m * k; i++) a_raw[i] = (uint8_t)(128 + (int)(i * 7 % 61) - 30);
	for (i = 0; i < n * k; i++) b_raw[i] = (uint8_t)(128 + (int)(i * 13 % 41) - 20);
	for (i = 0; i < n; i++) bias[i] = (int)i * 3 - 40;

	/*
	 * CHARSIU_IMPULSE: one live weight per output channel and a zero bias,
	 * so output channel c is input channel c mod K and nothing else.
	 *
	 * A dense matmul that comes out wrong cannot say WHICH of the two
	 * packings is wrong, or whether it is the arithmetic. An impulse can:
	 * the output IS the layout, read directly. This is the probe that
	 * decoded the weight layout for the Mesa driver, and the reason it works
	 * is that a single live tap has nothing to sum and nothing to cancel.
	 */
	/*
	 * CHARSIU_CONST: every input byte at the zero point and a bias ramp.
	 *
	 * The MAC computes sum((in - 0x80) * w), so an input held at 0x80 makes
	 * it exactly zero by construction whatever the weights are, and the
	 * output can only be requant(A). Walking the bias then draws the whole
	 * output stage as a line: its slope is the scale and its intercept is
	 * the offset, both solved rather than guessed.
	 *
	 * This is how the driver work calibrated the same stage. It is worth
	 * more than another reading of the formula, because it does not depend
	 * on any layout being right.
	 */
	if (getenv("CHARSIU_CONST")) {
		unsigned c;

		for (i = 0; i < m * k; i++)
			a_raw[i] = (uint8_t)job.input_zero_point;
		for (c = 0; c < n; c++)
			bias[c] = (int)c * 2000 - 60000;
		printf("const: input at the zero point, bias ramp %d..%d\n",
		       bias[0], bias[n - 1]);
	}

	if (getenv("CHARSIU_IMPULSE")) {
		unsigned c, j;

		for (c = 0; c < n; c++)
			for (j = 0; j < k; j++)
				b_raw[c * k + j] =
					(uint8_t)(j == c % k ? 128 + 100 : 128);
		memset(bias, 0, n * sizeof(*bias));
		printf("impulse: weight[c][c mod K] live, bias zero\n");
	}
	for (i = 0; i < n; i++) {
		unsigned j;
		for (j = 0; j < k; j++)
			wsums[i] += (int)b_raw[i * k + j] - job.weight_zero_point;
	}

	ret = charsiu_bo_alloc(dev, 4096, &regcmd);
	ret |= charsiu_bo_alloc(dev, (size_t)charsiu_entries_per_row(&job.mm) * 64 * m + 4096, &in);
	ret |= charsiu_bo_alloc(dev, charsiu_weight_bytes(&job.mm) + 4096, &wt);
	ret |= charsiu_bo_alloc(dev, (size_t)m * n + 4096, &outbo);
	ret |= charsiu_bo_alloc(dev, charsiu_coef_bytes(&job.mm) + 4096, &coef);
	if (ret) { printf("bo alloc FAILED %d\n", ret); return 1; }
	printf("bo iova: regcmd 0x%llx  in 0x%llx  wt 0x%llx  out 0x%llx  coef 0x%llx\n",
	       (unsigned long long)regcmd.dma_address, (unsigned long long)in.dma_address,
	       (unsigned long long)wt.dma_address, (unsigned long long)outbo.dma_address,
	       (unsigned long long)coef.dma_address);

	job.input_addr = (uint32_t)in.dma_address;
	job.weight_addr = (uint32_t)wt.dma_address;
	job.output_addr = (uint32_t)outbo.dma_address;
	job.coef_addr = (uint32_t)coef.dma_address;

	charsiu_bo_prep(dev, &in, 1000000000);
	memset(in.map, job.input_zero_point, in.size);
	/* A packs as [K/atom][M][atom]: the atom is contiguous, then the rows. */
	for (i = 0; i < m; i++) {
		unsigned kk;
		for (kk = 0; kk < k; kk++)
			((uint8_t *)in.map)[(kk / atom) * m * atom + i * atom + kk % atom] =
				a_raw[i * k + kk];
	}
	charsiu_bo_fini(dev, &in);

	charsiu_bo_prep(dev, &wt, 1000000000);
	charsiu_pack_weights(&job.mm, b_raw, wt.map);
	charsiu_bo_fini(dev, &wt);

	charsiu_bo_prep(dev, &coef, 1000000000);
	charsiu_build_coefs(&job, bias, wsums, coef.map);
	charsiu_bo_fini(dev, &coef);

	charsiu_bo_prep(dev, &outbo, 1000000000);
	memset(outbo.map, 0xa5, outbo.size);        /* a sentinel, not zero */
	charsiu_bo_fini(dev, &outbo);

	charsiu_bo_prep(dev, &regcmd, 1000000000);
	nreg = charsiu_emit_job(&job, regcmd.map, 4096 / 8);
	charsiu_bo_fini(dev, &regcmd);
	if (!nreg) { printf("emit FAILED\n"); return 1; }
	printf("register stream: %zu entries\n", nreg);

	/*
	 * With CHARSIU_DUMP set, print the stream in the same format the Mesa
	 * driver's ROCKET_DUMP_REGCMD uses, so the two can be diffed line for
	 * line on a shape both can run. Three rounds of adding what looked
	 * missing did not find why the job hangs; a value diff against a stream
	 * that is known to compute will.
	 */
	if (getenv("CHARSIU_DUMP")) {
		charsiu_bo_prep(dev, &regcmd, 1000000000);
		for (i = 0; i < nreg; i++) {
			uint64_t ent = ((uint64_t *)regcmd.map)[i];

			printf("CS %02u t=%04x r=%04x v=%08x\n", i,
			       (unsigned)(ent >> 48), (unsigned)(ent & 0xffff),
			       (unsigned)((ent >> 16) & 0xffffffff));
		}
		charsiu_bo_fini(dev, &regcmd);
	}

	in_handles[0] = in.handle;
	in_handles[1] = wt.handle;
	in_handles[2] = coef.handle;
	out_handles[0] = outbo.handle;
	if (!regcmd.dma_address)
		printf("WARNING: the register stream is at IOVA 0\n");
	ret = charsiu_submit(dev, &regcmd, (unsigned)nreg, in_handles, 3,
			     out_handles, 1);
	if (ret) { printf("submit FAILED %d\n", ret); return 1; }
	printf("submit ok\n");

	ret = charsiu_bo_prep(dev, &outbo, 5000000000LL);
	if (ret) { printf("wait FAILED %d\n", ret); return 1; }

	cpu_reference(&job, a_raw, b_raw, bias, ref);
	for (i = 0; i < m * n; i++) {
		uint8_t got = ((uint8_t *)outbo.map)[i];
		int d = (int)got - (int)ref[i];

		if (got != 0xa5) nonzero++;
		if (d < -1 || d > 1) bad++;
	}
	printf("output: %u of %u bytes written, %u differ from the CPU by more than 1\n",
	       nonzero, m * n, bad);
	printf("  npu[0..15] ");
	for (i = 0; i < 16 && i < m * n; i++) printf("%4u", ((uint8_t *)outbo.map)[i]);
	printf("\n  cpu[0..15] ");
	for (i = 0; i < 16 && i < m * n; i++) printf("%4u", ref[i]);
	printf("\n");
	/* Where the npu's own values sit, which says whether it computed
	 * something wrong or nothing at all. */
	{
		unsigned lo = 255, hi = 0, seen[256] = { 0 }, distinct = 0;

		for (i = 0; i < m * n; i++) {
			uint8_t v = ((uint8_t *)outbo.map)[i];

			if (v < lo) lo = v;
			if (v > hi) hi = v;
			if (!seen[v]++) distinct++;
		}
		printf("  npu range %u..%u, %u distinct\n", lo, hi, distinct);
	}
	/*
	 * The WHOLE output when it is small enough to read. A ramp that comes
	 * back with its first channels flat and its variation somewhere else is
	 * a layout that moved, not a scale that changed, and only the whole
	 * vector tells those apart.
	 */
	if (m * n <= 128) {
		printf("  npu all:");
		for (i = 0; i < m * n; i++)
			printf("%s%4u", i % 16 ? "" : "\n         ",
			       ((uint8_t *)outbo.map)[i]);
		printf("\n  cpu all:");
		for (i = 0; i < m * n; i++)
			printf("%s%4u", i % 16 ? "" : "\n         ", ref[i]);
		printf("\n");
	}
	{
		/* A least squares fit of the npu against the reference, which
		 * turns "wrong" into a slope and an intercept. A slope of one
		 * with a nonzero intercept is an offset; a slope that is a
		 * power of two is a shift. */
		double sx = 0, sy = 0, sxx = 0, sxy = 0, nn = m * n, den;

		for (i = 0; i < m * n; i++) {
			double x = ((uint8_t *)outbo.map)[i], y = ref[i];

			sx += x; sy += y; sxx += x * x; sxy += x * y;
		}
		den = nn * sxx - sx * sx;
		if (den != 0)
			printf("  fit cpu = %.4f * npu + %.2f\n",
			       (nn * sxy - sx * sy) / den,
			       (sy * sxx - sx * sxy) / den);
	}
	charsiu_bo_fini(dev, &outbo);
	charsiu_close(dev);
	return bad ? 1 : 0;
}
