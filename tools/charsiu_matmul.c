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
#include <stdlib.h>
#include <string.h>

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

	in_handles[0] = in.handle;
	in_handles[1] = wt.handle;
	in_handles[2] = coef.handle;
	out_handles[0] = outbo.handle;
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
	printf("  npu[0..7] ");
	for (i = 0; i < 8 && i < m * n; i++) printf("%4u", ((uint8_t *)outbo.map)[i]);
	printf("\n  cpu[0..7] ");
	for (i = 0; i < 8 && i < m * n; i++) printf("%4u", ref[i]);
	printf("\n");
	charsiu_bo_fini(dev, &outbo);
	charsiu_close(dev);
	return bad ? 1 : 0;
}
