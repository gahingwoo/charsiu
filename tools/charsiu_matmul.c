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
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "charsiu.h"

/*
 * THE OUTPUT IS A SIGNED BYTE WITH A FLOOR UNDER IT.
 *
 *     out = clamp(max(requant, 0) + offset, -128, 127), stored as int8
 *
 * Both halves of that were paid for. The signed byte came from rounds 160 to
 * 162; the floor came from round 163, whose first entry was written to be the
 * milestone and instead falsified the model it was based on. Every earlier entry
 * had an offset that made a floor at zero and a rail at -128 predict the same
 * bytes, so none of them could tell the two apart. With the offset at 0 they
 * differ, and the negative half came back at 0 rather than negative.
 *
 * So there is a fused ReLU after all. It stops mattering once the accumulator is
 * lifted by 128 in the requant domain before the floor and the offset takes the
 * same 128 back, which is what charsiu_build_coefs does now.
 *
 * The reference works in the signed domain, which is what a projection wants.
 * CHARSIU_UINT8_OUT restores the old unsigned reading as the control.
 */
static int cpu_reference(const struct charsiu_job *job, const uint8_t *a,
			 const uint8_t *b, const int32_t *bias, uint8_t *out)
{
	const struct charsiu_matmul *mm = &job->mm;
	float mult = job->input_scale * job->weight_scale / job->output_scale;
	int unsigned_out = getenv("CHARSIU_UINT8_OUT") != NULL;
	unsigned m, n, k;

	for (m = 0; m < mm->m; m++) {
		for (n = 0; n < mm->n; n++) {
			int64_t acc = bias[n];
			float v;

			for (k = 0; k < mm->k; k++) {
				int w = (int)b[n * mm->k + k];

				/* a nibble is two's complement in four bits */
				if (mm->wdtype == CHARSIU_INT4)
					w = (w & 0x8) ? (w & 0xf) - 16 : (w & 0xf);
				acc += ((int)a[m * mm->k + k] - job->input_zero_point) *
				       (w - job->weight_zero_point);
			}

			/* The hardware rounds one shift half up; the reference
			 * has to do the same or every value is off by one, and
			 * half up is floor(v + 0.5) rather than a cast, which
			 * truncates towards zero and is wrong below it. */
			v = (float)acc * mult + (float)job->output_zero_point;
			v = floorf(v + 0.5f);
			if (unsigned_out) {
				if (v < 0) v = 0;
				if (v > 255) v = 255;
			} else {
				if (v < -128) v = -128;
				if (v > 127) v = 127;
			}
			out[m * mm->n + n] = (uint8_t)(int)v;
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
	unsigned m, n, k, i, atom, bad = 0, nonzero = 0, exact = 0;
	size_t nreg;
	int ret;

	job.mm.m = argc > 1 ? (unsigned)atoi(argv[1]) : 4;
	job.mm.k = argc > 2 ? (unsigned)atoi(argv[2]) : 64;
	job.mm.n = argc > 3 ? (unsigned)atoi(argv[3]) : 32;
	/*
	 * CHARSIU_W4 runs the same test with int4 weights. The layout came off
	 * the board in rounds 167 and 168 and the packer is unit checked on the
	 * host; this is the first thing to ask it for a NUMBER rather than a
	 * pattern, so the weights are confined to a nibble and the reference
	 * uses the same values.
	 */
	job.mm.wdtype = getenv("CHARSIU_W4") ? CHARSIU_INT4 : CHARSIU_INT8;
	job.mm.adtype = CHARSIU_INT8;
	job.input_scale = 0.02f;
	job.weight_scale = 0.01f;
	job.output_scale = 0.25f;
	job.input_zero_point = 128;
	job.weight_zero_point = 128;
	/*
	 * ZERO, because the output is a signed byte and a projection is signed.
	 * 128 with an offset of -128 is what every round up to 162 used, and it
	 * put the negative half of every result on the -128 rail.
	 * CHARSIU_UINT8_OUT is the control that puts it back.
	 */
	job.output_zero_point = getenv("CHARSIU_UINT8_OUT") ? 128 : 0;
	m = job.mm.m; k = job.mm.k; n = job.mm.n;
	atom = charsiu_feature_atom(job.mm.adtype);

	printf("matmul M=%u K=%u N=%u w%s a8, feature atom %u, %u entries per row\n",
	       m, k, n, job.mm.wdtype == CHARSIU_INT4 ? "4" : "8", atom,
	       charsiu_entries_per_row(&job.mm));

	dev = charsiu_open(NULL);
	if (!dev) { printf("open FAILED\n"); return 1; }

	a_raw = malloc((size_t)m * k);
	b_raw = malloc((size_t)n * k);
	ref = malloc((size_t)m * n);
	bias = calloc(n, sizeof(*bias));
	wsums = calloc(n, sizeof(*wsums));
	for (i = 0; i < m * k; i++) a_raw[i] = (uint8_t)(128 + (int)(i * 7 % 61) - 30);
	for (i = 0; i < n * k; i++) b_raw[i] = (uint8_t)(128 + (int)(i * 13 % 41) - 20);
	if (job.mm.wdtype == CHARSIU_INT4) {
		/*
		 * A nibble is signed, -8 to 7, and it is NOT biased the way a
		 * byte weight is: the packer stores it as it lies. So the zero
		 * point is 0 here and b_raw holds the two's complement nibble
		 * in its low four bits, which is what both the packer and the
		 * reference then read.
		 */
		job.weight_zero_point = 0;
		for (i = 0; i < n * k; i++)
			b_raw[i] = (uint8_t)(((int)(i * 13 % 15) - 7) & 0xf);
	}
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
	 * The MAC computes sum(in_stored * w_stored) over signed bytes, and the
	 * packer stores the input biased by 0x80, so an input held at the zero
	 * point of 0x80 is stored as 0 and the MAC is exactly zero by
	 * construction whatever the weights are. The output can then only be
	 * requant(A), and walking the bias draws the whole output stage as a
	 * line whose slope is the scale and whose intercept is the offset.
	 *
	 * Until round 162 the input went in unbiased, so this probe's MAC was
	 * NOT zero, it was -128 * weight_sum, and the residual that produced was
	 * written off as three counts of rounding for six rounds.
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

	/*
	 * CHARSIU_NEG: a MAC that is genuinely negative, with a zero bias.
	 *
	 * The floor is at requant = 0 and it is applied before 0x40ac's offset,
	 * which is measured. What is not known is WHERE it sits, and the two
	 * probes split that cleanly. CONST holds the MAC at zero and walks the
	 * bias, so it can only see a floor on the BS stage's add operand A.
	 * This one is the mirror: the bias is zero and the MAC itself walks
	 * through zero, so it can only see a floor after the multiply.
	 *
	 * Every input byte sits 40 above the zero point and every weight in a
	 * channel sits (c - N/2) away from the weight zero point, so
	 *
	 *     acc[c] = 40 * K * (c - N/2)
	 *
	 * which is a straight line through zero, negative for the first half of
	 * the channels and positive for the second, with no layout question in
	 * it because every tap of a channel carries the same value.
	 *
	 * ROUND 161 ANSWERED IT, and found something bigger on the way. The
	 * channels that came back pinned were exactly those whose requant was
	 * negative, with a zero bias, so the floor is after the multiply and the
	 * A record is exonerated. But the line came back with the WRONG SIGN and
	 * a gain of -2.2, which is what an input of zp+40 does if the hardware
	 * reads the byte as a signed value: 168 is -88, not +40. That is the
	 * input bias, fixed in round 162, and it fits all 64 bytes exactly.
	 */
	if (getenv("CHARSIU_NEG")) {
		unsigned c, j;
		int d;

		for (i = 0; i < m * k; i++)
			a_raw[i] = (uint8_t)(job.input_zero_point + 40);
		for (c = 0; c < n; c++) {
			d = (int)c - (int)(n / 2);
			if (d < -120) d = -120;
			if (d > 120) d = 120;
			for (j = 0; j < k; j++)
				b_raw[c * k + j] =
					(uint8_t)(job.weight_zero_point + d);
		}
		memset(bias, 0, n * sizeof(*bias));
		printf("neg: input at zp+40, weight walks zp%+d..zp%+d, bias zero\n",
		       -(int)(n / 2), (int)n - 1 - (int)(n / 2));
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

		for (j = 0; j < k; j++) {
			int w = b_raw[i * k + j];

			if (job.mm.wdtype == CHARSIU_INT4)
				w = (w & 0x8) ? (w & 0xf) - 16 : (w & 0xf);
			wsums[i] += w - job.weight_zero_point;
		}
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
	/*
	 * CHARSIU_INPUT_RAW is the control for round 162's fix. It writes the
	 * input the way every round up to 161 did, unbiased, which must
	 * reproduce the old wrong numbers exactly. A fix whose control does not
	 * bring the fault back has not been shown to be the fix.
	 */
	if (getenv("CHARSIU_INPUT_RAW")) {
		unsigned kk;

		memset(in.map, job.input_zero_point, in.size);
		for (i = 0; i < m; i++)
			for (kk = 0; kk < k; kk++)
				((uint8_t *)in.map)[(kk / atom) * m * atom
						    + i * atom + kk % atom] =
					a_raw[i * k + kk];
		printf("input written RAW, the pre-162 form\n");
	} else {
		charsiu_pack_input(&job.mm, a_raw, in.map, in.size,
				   (uint8_t)job.input_zero_point);
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
		if (d == 0) exact++;
	}
	/* BYTE EXACT is the claim worth making, and it is not the same as
	 * "within one count": a shape too large to print elementwise can only be
	 * reported honestly if the count is kept here. */
	printf("output: %u of %u bytes written, %u BYTE EXACT, %u differ by more than 1\n",
	       nonzero, m * n, exact, bad);
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
		/*
		 * THE SAME BYTES READ AS SIGNED, beside what the requant alone
		 * should be.
		 *
		 * 0x40ac holds the output zero point in the hardware's SIGNED
		 * domain: the vendor's own compiler writes -27 there for a model
		 * whose output zero point is 101. charsiu's is 128, so it writes
		 * -128, and the byte that comes out is requant - 128 in two's
		 * complement, whose unsigned reading is the correct output byte
		 * because the wrap does the +128. That makes an output which is
		 * RIGHT indistinguishable from one clamped at 128 unless the
		 * signed reading is printed too, and rounds 155 to 158 read that
		 * clamp as a fused ReLU. This row and the one under it are what
		 * tells those two apart.
		 */
		printf("\n  npu signed:");
		for (i = 0; i < m * n; i++) {
			int v = ((uint8_t *)outbo.map)[i];

			printf("%s%4d", i % 16 ? "" : "\n         ",
			       v > 127 ? v - 256 : v);
		}
		printf("\n  cpu signed:");
		for (i = 0; i < m * n; i++) {
			int v = ref[i];

			printf("%s%4d", i % 16 ? "" : "\n         ",
			       v > 127 ? v - 256 : v);
		}
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
