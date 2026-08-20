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
		/*
		 * AND THE SCALE HAS TO MOVE WITH IT, which round 169 did not do
		 * and which made that round unreadable. A nibble spans -7 to 7
		 * where a byte spans -128 to 127, so at int8's weight scale the
		 * whole reference vector fitted in -1 to +1: 54 of 64 channels
		 * "matched" because both sides were zero, and every channel
		 * that carried any signal was wrong by half of full scale. A
		 * matching channel is not a computed one.
		 *
		 * 18 times the weight scale puts int4's accumulator range where
		 * int8's is. CHARSIU_W4_SMALLSCALE restores round 169's setting
		 * as the control.
		 */
		if (!getenv("CHARSIU_W4_SMALLSCALE"))
			job.weight_scale = 0.18f;
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

		/*
		 * For int4 the live tap goes at k = c mod 16, not c mod K.
		 * Only k = 0 to 15 has a measured placement, so an impulse
		 * anywhere else would be testing a guess. Every channel still
		 * gets exactly one live weight, which is what the probe needs.
		 */
		unsigned span = job.mm.wdtype == CHARSIU_INT4 ? 16 : k;

		/*
		 * AND THE IMPULSE NEEDS ITS OWN SCALE, which round 176 did not
		 * give it. The dense int4 test was rescaled in round 170 and the
		 * impulse was left behind: one live nibble of 7 against an input
		 * spanning 30 either side of the zero point puts the whole
		 * reference in SEVEN distinct values, and both intra row orders
		 * came back TOO FLAT TO JUDGE A MATCH. The same mistake as round
		 * 169, one probe over.
		 *
		 * A multiplier of 1/7 and an input spanning 120 makes the
		 * reference a nibble of 7 times the input, so it fills the byte
		 * and one count means one count.
		 */
		if (job.mm.wdtype == CHARSIU_INT4) {
			job.weight_scale = 1.7857f;
			for (i = 0; i < m * k; i++)
				a_raw[i] = (uint8_t)(job.input_zero_point
						     + (int)(i * 37 % 241) - 120);
		} else if (!getenv("CHARSIU_I8_IMPULSE_SMALLSCALE")) {
			/*
			 * AND THE INT8 IMPULSE WAS LEFT BEHIND THE SAME WAY.
			 *
			 * Its live weight is 100 above the zero point and its
			 * input spanned 30 either side, so the accumulator only
			 * reached 3000 and the multiplier of 0.0008 put the
			 * whole reference in FIVE distinct values across four
			 * counts, -2 to +2. The tool's own check calls anything
			 * under eight TOO FLAT TO JUDGE A MATCH, so this probe
			 * has been failing its own readability test while its
			 * 64 of 64 was being read as evidence. Third time for
			 * the same mistake: round 169 on the int4 dense probe,
			 * round 176 on the int4 impulse, this one on int8.
			 *
			 * A weight scale of 0.12 with the wide input fills the
			 * byte without touching either rail. 0.1323 reaches
			 * exactly -127 to 127 but leaves a value sitting on the
			 * rail, and a saturated channel cannot be told from a
			 * wrong one. Computed with the tool's own
			 * cpu_reference() at M=1 K=64 N=64, M=1 K=32 N=64 and
			 * M=8 K=64 N=64 before this was ever flashed.
			 *
			 *   scale    distinct   range        at a rail
			 *   0.01          5     -2 to 2          0     old
			 *   0.10         54    -96 to 96         0
			 *   0.12         62   -115 to 115        0     this
			 *   0.1323       64   -127 to 127        1
			 *   0.14         61   -128 to 127        5
			 *
			 * CHARSIU_I8_IMPULSE_SMALLSCALE restores the old
			 * setting as the control, the way CHARSIU_W4_SMALLSCALE
			 * does for int4.
			 */
			job.weight_scale = 0.12f;
			for (i = 0; i < m * k; i++)
				a_raw[i] = (uint8_t)(job.input_zero_point
						     + (int)(i * 37 % 241) - 120);
		}

		for (c = 0; c < n; c++)
			for (j = 0; j < k; j++)
				b_raw[c * k + j] = job.mm.wdtype == CHARSIU_INT4
					? (uint8_t)(j == c % span ? 7 : 0)
					: (uint8_t)(j == c % span ? 128 + 100 : 128);
		memset(bias, 0, n * sizeof(*bias));
		/*
		 * ⚠ IT SAYS WHICH BRANCH IT TOOK AND WHAT IT SET.
		 *
		 * Five knobs in this family have turned out to miss the code
		 * they were meant to reach and returned a clean negative that
		 * was then read as evidence. A knob that cannot be seen working
		 * is not a knob, so print the scale, the input span and the
		 * branch rather than trusting that the string is in the binary.
		 */
		printf("impulse: weight[c][c mod %u] live, bias zero, "
		       "%s branch, weight_scale %g, input span %s\n",
		       span,
		       job.mm.wdtype == CHARSIU_INT4 ? "int4"
		       : getenv("CHARSIU_I8_IMPULSE_SMALLSCALE") ? "int8 SMALLSCALE"
								 : "int8",
		       job.weight_scale,
		       job.mm.wdtype == CHARSIU_INT4 ||
		       !getenv("CHARSIU_I8_IMPULSE_SMALLSCALE")
			       ? "128 +- 120" : "128 +- 30");
	}
	/*
	 * CHARSIU_W4_HALFK: zero every weight the int4 fetch does not read.
	 *
	 * Rounds 265 to 277 mapped which k each channel actually gets. With
	 * CORE 0x3020 = 111 at N = 64 every one of the 64 channels is written
	 * and reachable, and each is fed by 16 weight bytes in two runs, one at
	 * its own offset and one 256 bytes later, which is k 0..15 and k 32..47.
	 * Half the reduction, but a KNOWN half:
	 *
	 *   fetched  <=>  (j mod 32) < 16
	 *
	 * The hardware not reading a weight is only wrong if the weight matters.
	 * Zero the ones it never reads and the partial sum it computes IS the
	 * full sum, so cpu_reference() and the hardware are answering the same
	 * question with no change to the packer at all. A real 32 deep int4
	 * matmul on all 64 channels, at the cost of half the buffer.
	 *
	 * This is a data shape, not a workaround for a defect in the packing.
	 * The unfetched half stays open.
	 */
	if (getenv("CHARSIU_W4_HALFK")) {
		unsigned j, zeroed = 0;

		for (i = 0; i < n; i++)
			for (j = 0; j < k; j++)
				if (j % 32 >= 16) {
					b_raw[i * k + j] =
						(uint8_t)job.weight_zero_point;
					zeroed++;
				}
		printf("HALFK: zeroed %u of %u weights, the ones with "
		       "(k mod 32) >= 16, which the fetch never reads\n",
		       zeroed, n * k);
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

	/* the scales go in the log too. Round 174 lost a whole comparison to a
	 * probe setting that existed only in the source, and rounds 169 and 176
	 * lost two to a scale nobody could see from the console. */
	printf("  scales in %.4f wt %.4f out %.4f  zp in %u wt %u out %u\n",
	       job.input_scale, job.weight_scale, job.output_scale,
	       job.input_zero_point, job.weight_zero_point, job.output_zero_point);

	ret = charsiu_bo_alloc(dev, 4096, &regcmd);
	ret |= charsiu_bo_alloc(dev, (size_t)charsiu_entries_per_row(&job.mm) * 64 * m + 4096, &in);
	ret |= charsiu_bo_alloc(dev, charsiu_weight_bytes(&job.mm) + 4096, &wt);
	/* the surface is ALIGN(N, atom) wide, not N, so an N off the atom needs
	 * more than m*n bytes. Round 199's N = 40 came back "0 of 40 bytes
	 * written", which is what a short output buffer looks like. */
	ret |= charsiu_bo_alloc(dev,
		(size_t)((n + charsiu_feature_atom(CHARSIU_INT8) - 1)
			 / charsiu_feature_atom(CHARSIU_INT8))
		* charsiu_feature_atom(CHARSIU_INT8) * m + 4096, &outbo);
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
	/*
	 * WHERE OUTPUT ROW m CHANNEL n LANDS.
	 *
	 * Round 199 measured that charsiu is wrong at EVERY M above 1, at a flat
	 * rate: 32.0% byte exact at K=64 from M=8 all the way to M=3136, and
	 * 38.7% at K=33. It does not degrade with size, which is what a wrong
	 * LAYOUT looks like and not what a wrong computation looks like. Every
	 * correctness run this project ever did was at M = 1.
	 *
	 * The input surface is [k/atom][m][k%atom]. If the output is the mirror
	 * of that, it is [n/atom][m][n%atom], and at M = 1 that collapses to
	 * exactly n, which is why a row major reading has been right all along
	 * and only at M = 1.
	 *
	 * This changes only how the result is READ. The register stream, the
	 * packing and the buffers are untouched, so if the surface reading is
	 * byte exact then the hardware was computing correctly the whole time.
	 * CHARSIU_OUT_ROWMAJOR forces the old reading, and at M = 1 the two are
	 * the same expression and must agree.
	 */
	{
		unsigned atom = charsiu_feature_atom(CHARSIU_INT8);
		int surf = !getenv("CHARSIU_OUT_ROWMAJOR");
		unsigned mi, ni;

		printf("  reading the output as %s\n",
		       surf ? "[n/atom][m][n%atom], the mirror of the input surface"
			    : "[m][n] row major");
		for (mi = 0; mi < m; mi++)
			for (ni = 0; ni < n; ni++) {
				size_t at = surf
					? (size_t)(ni / atom) * m * atom
					  + (size_t)mi * atom + ni % atom
					: (size_t)mi * n + ni;
				uint8_t got = ((uint8_t *)outbo.map)[at];
				int d = (int)got - (int)ref[(size_t)mi * n + ni];

				if (got != 0xa5) nonzero++;
				if (d < -1 || d > 1) bad++;
				if (d == 0) exact++;
			}
	}
	/* BYTE EXACT is the claim worth making, and it is not the same as
	 * "within one count": a shape too large to print elementwise can only be
	 * reported honestly if the count is kept here. */
	/*
	 * COMPUTED against TRIVIAL. A channel where the reference is zero and
	 * the hardware is zero has matched nothing: round 169 read 54 of 64 on
	 * an int4 impulse whose reference was zero on 54 channels and called the
	 * layout confirmed. The reference's own spread goes beside the match
	 * count so that cannot happen quietly again.
	 */
	printf("output: %u of %u bytes written, %u BYTE EXACT, %u differ by more than 1\n",
	       nonzero, m * n, exact, bad);
	{
		unsigned rmin = 255, rmax = 0, rdist = 0, rseen[256] = { 0 };

		for (i = 0; i < m * n; i++) {
			if (ref[i] < rmin) rmin = ref[i];
			if (ref[i] > rmax) rmax = ref[i];
			if (!rseen[ref[i]]++) rdist++;
		}
		printf("  reference spread: %u distinct, %u..%u%s\n",
		       rdist, rmin, rmax,
		       rdist < 8 ? "   TOO FLAT TO JUDGE A MATCH" : "");
	}
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
