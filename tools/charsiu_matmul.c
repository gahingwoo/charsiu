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

/*
 * THE PHANTOM PACKING, derived from the map rather than guessed.
 *
 * Round 325 measured the hardware's int4 weight address map at every live byte
 * of three geometries, 1376 points, one form:
 *
 *     w = 32 * floor(s / K) + (s mod 32),      s = byte / 8
 *
 * Count what that delivers. A declared channel is fed by the slots with
 * s mod 32 == w mod 32 inside its own block, which is K/32 slots of 16 nibbles,
 * so **every declared channel gets exactly K/2 nibbles** -- and a buffer of
 * N*K/2 bytes is N*K nibbles, so the OTHER half of it feeds words N..2N-1.
 *
 * That is the whole half width, and it is not a defect: those extra words are
 * real computed outputs. **So put the second half of each channel's k there and
 * add the two.** out[c] + out[c+N] is then the full dot product, the buffer
 * stays exactly N*K/2 bytes, and every nibble in it is used.
 *
 * ⚠ Round 329 set 0x40b8 = 3 and 0x3020 = 2N-1, which is what makes the
 * hardware compute and write those 2N words, but left the PACKING alone -- so
 * the phantom words held nothing and nothing changed. The registers were half
 * the change.
 *
 * The intra-slot order is round 265's measured one: byte j of a slot holds
 * k = 2j in its low nibble and k = 2j+1 in its high.
 */
static void pack_phantom(const struct charsiu_matmul *mm, const uint8_t *b_raw,
			 uint8_t *dst)
{
	unsigned k = mm->k, n = mm->n, w, j, byt;
	size_t bytes = (size_t)n * k / 2;
	unsigned slots = k / 32;                 /* slots a word owns */

	/*
	 * THE PAIRED PACKING, and every term in it was MEASURED, not assumed.
	 *
	 * Round 332 pointed --kpair at the phantom settings and read which k
	 * each byte multiplies with. At K=64 N=64:
	 *
	 *   w0   byte 0 -> k 0,1    byte 4 -> k 8,9    byte 256 -> k 32,33
	 *   w1   byte 8 -> k 16,17                     byte 264 -> k 48,49
	 *   w64  byte 1024 -> k 0,1                    byte 1280 -> k 32,33
	 *
	 * So **w0 and w1 together cover all 64 k**, and w64 is a duplicate of
	 * w0's k in a different part of the buffer. ⚠ That kills round 330's
	 * pairing -- c with c+N is two copies of the same half -- and hands
	 * over the real one: **a channel is the pair (2r, 2r+1)**. With
	 * 0x3020 = 2N-1 there are 2N words, so N real channels each with the
	 * whole of K, and N*K nibbles is exactly the N*K/2 byte buffer.
	 *
	 *   k = 16*(w mod 2) + 32*j + 2*byte_in_slot + nibble
	 *
	 * where j indexes the slots the map gives word w:
	 *   slot = K*(w/32) + 32*j + (w mod 32),  byte = 8*slot + byte_in_slot
	 */
	memset(dst, 0, bytes);
	for (w = 0; w < 2 * n && w / 2 < n; w++) {
		unsigned c = w / 2;                    /* the real channel */

		for (j = 0; j < slots; j++) {
			size_t slot = (size_t)k * (w / 32) + 32 * j + (w % 32);

			for (byt = 0; byt < 8; byt++) {
				size_t at = slot * 8 + byt;
				unsigned kk = 16 * (w % 2) + 32 * j + 2 * byt;

				if (at >= bytes || kk + 1 >= k)
					continue;
				dst[at] = (uint8_t)
					((b_raw[(size_t)c * k + kk] & 0xf) |
					 ((b_raw[(size_t)c * k + kk + 1] & 0xf) << 4));
			}
		}
	}
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
	unsigned m, n, k, i, ci, atom, bad = 0, nonzero = 0, exact = 0;
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
		/*
		 * ⚠ ROUND 324 CAUGHT THIS WITH ITS OWN CONTROL. The generator
		 * was `i * 13 % 15`, which has FIFTEEN residues, so channel c
		 * and channel c + 15 hold an identical weight vector, and the
		 * halfk parity doubles that to thirty. Every reference value
		 * therefore repeats every 30 channels: round 324's search for
		 * "where did this value go" found channel 30's value sitting at
		 * channel 0's word and reported it found. Round 312's "the
		 * survivors are g0 and g30" is the same artifact, because
		 * ref[240] IS ref[0], not a wrapping stride.
		 *
		 * A per channel hash instead, so that no two channels agree.
		 * The old ramp is CHARSIU_W4_RAMPW, because every int4 round up
		 * to 324 was measured with it and a regression has to stay
		 * reproducible.
		 */
		for (ci = 0; ci < n; ci++)
			for (i = 0; i < k; i++) {
				unsigned h = getenv("CHARSIU_W4_RAMPW")
					? (unsigned)(((size_t)ci * k + i) * 13 % 15)
					: ((ci * 2654435761u) ^ (i * 40503u)) % 15u;

				b_raw[(size_t)ci * k + i] =
					(uint8_t)(((int)h - 7) & 0xf);
			}
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

		/*
		 * ⚠ THE MASK IS PER CHANNEL AND 278 AND 279 BOTH GOT IT WRONG.
		 * They zeroed (k mod 32) >= 16 for every channel, which is what
		 * an EVEN channel is fed. An odd channel is fed the other half,
		 * k 16..31 and 48..63, so the mask was exactly backwards on half
		 * the output. byte 0 pairs with k 0 on channel 0 and byte 8 with
		 * k 16 on channel 1, measured at K = 64 and again at K = 32.
		 */
		for (i = 0; i < n; i++)
			for (j = 0; j < k; j++)
				if (((j >> 4) & 1) != (i & 1)) {
					b_raw[i * k + j] =
						(uint8_t)job.weight_zero_point;
					zeroed++;
				}
		printf("HALFK: zeroed %u of %u weights. A channel is fed "
		       "k with bit4 equal to its own parity, so even channels "
		       "keep 0..15 and 32..47 and odd keep 16..31 and 48..63\n",
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
	/*
	 * ⚠ FOUR BYTES AN ELEMENT ON THE int4 PATH. w4a16 does not requantise:
	 * 0x40ac, 0x40b0 and 0x40b4 are 0, 1, 0, so what lands is the raw
	 * accumulator, a signed 32 bit integer per channel. Rounds 265, 267 and
	 * 278 each found a probe in this repo reading it as bytes; this is the
	 * fourth place and the last one that had not been fixed.
	 */
	ret |= charsiu_bo_alloc(dev,
		(size_t)((n + charsiu_feature_atom(CHARSIU_INT8) - 1)
			 / charsiu_feature_atom(CHARSIU_INT8))
		* charsiu_feature_atom(CHARSIU_INT8) * m
		* (job.mm.wdtype == CHARSIU_INT4 ? 4 : 1)
		* ((getenv("CHARSIU_W4_PHANTOM") || getenv("CHARSIU_W4_PAIRED")) ? 2 : 1) + 4096, &outbo);
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
	/*
	 * ⚠ CHARSIU_W4_AF16: A REAL HALF IN THE SLOT, not an int8 in its high
	 * byte.
	 *
	 * The int4 arithmetic this tree measured is
	 *
	 *     out = ((int16)fp16bits(w) * (int16)abits) >> 16
	 *
	 * and fp16bits is NOT proportional to the nibble. The effective weights
	 * it can represent are 0 and the band 1.000, 1.067, 1.100, 1.133,
	 * 1.150, 1.167, 1.183, which is sixteen levels crammed into eighteen
	 * percent. **That is not a grid a matmul can use**, so every int4 result
	 * in this tree is exact against a NONLINEAR reference rather than
	 * against a weighted sum.
	 *
	 * The shape of that formula is what a HALF weight multiplied by an
	 * INTEGER activation looks like, and charsiu puts an int8 in the high
	 * byte of a sixteen bit slot rather than a half. The vendor is W4A16
	 * with real halves. So fill the slot properly and print both references
	 * side by side: whichever the hardware matches is the answer.
	 */
	if (getenv("CHARSIU_W4_AF16")) {
		float *af = malloc((size_t)m * k * sizeof(*af));

		for (i = 0; i < m * k; i++)
			af[i] = (float)((int)a_raw[i] - 128);
		charsiu_pack_input_f16(&job.mm, af, in.map, in.size);
		free(af);
		printf("input packed as REAL halves, values %d..%d\n",
		       (int)a_raw[0] - 128, (int)a_raw[m * k - 1] - 128);
	} else if (getenv("CHARSIU_INPUT_RAW")) {
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
	if ((getenv("CHARSIU_W4_PHANTOM") || getenv("CHARSIU_W4_PAIRED")))
		pack_phantom(&job.mm, b_raw, wt.map);
	else
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
	if (job.mm.wdtype == CHARSIU_INT4) {
		/*
		 * ⚠ THE int4 COMPARISON IS NOT THE int8 ONE AND ROUND 278 RAN
		 * IT ANYWAY. Two things were wrong at once. The output was read
		 * as bytes, and the giveaway is in 278's own log: every group of
		 * four reads "X Y 255 255" or "X Y 0 0", which is a little
		 * endian int32 pulled apart. And cpu_reference() returns a
		 * REQUANTISED int8 while w4a16 returns a raw accumulator, so the
		 * two were never in the same domain and the round could not have
		 * passed whatever the hardware did.
		 *
		 * The arithmetic is measured, seven nibbles for seven, once the
		 * activation is read as the raw 16 bit slot rather than an fp16
		 * of the value:
		 *
		 *   out = ((int16)fp16bits(w) * (int16)abits) >> 16
		 *
		 * charsiu packs an int8 into the HIGH byte of a 2 byte slot, so
		 * abits is (a - 0x80) << 8.
		 *
		 * ⚠ WHERE THE SHIFT GOES IS NOT MEASURED. Every point behind
		 * that formula had ONE live nibble, so a shift per element and a
		 * shift on the sum are indistinguishable in all of them. Both
		 * are computed here and both counts are printed, because picking
		 * one and reporting its score is how a fit gets mistaken for a
		 * reading.
		 */
		/*
		 * ⚠ THE SURFACE READING, not row major. Every int4 matmul in
		 * this project has been M = 1, where [n/atom][m][n%atom]
		 * collapses to n and a row major read is accidentally right.
		 * The int8 path has read the surface since round 199 and this
		 * one never had to. It does now.
		 */
		/*
		 * ⚠ THE ATOM IS THE int4 PATH'S OWN, NOT 16. w4a16 packs its
		 * activation as a 2 byte element whose feature atom is 8, and
		 * this reader hardcoded charsiu_feature_atom(CHARSIU_INT8),
		 * which is 16. At M = 1 both collapse to n and the difference
		 * cannot show; round 289 ran M of 2, 4 and 8 for the first time
		 * and the odd numbered groups came back 0 of 8 at M = 4, which
		 * is exactly the set the two readings disagree on.
		 *
		 * ⚠ IT IS NOT THE WHOLE STORY. The even groups came back 4 of 8
		 * where both readings agree, and M = 2 gave every group 4 of 8
		 * with only g7 at zero, which is a different shape again. So the
		 * atom is corrected here and the surface is DUMPED below rather
		 * than guessed at a second time.
		 */
		/*
		 * ⚠ THE SURFACE GROUPS BY SIXTEEN BYTES, NOT SIXTEEN ELEMENTS.
		 * Round 290 dumped the raw surface at M = 2, N = 16 and read it
		 * off directly: row 0's channels 0 to 3 are at words 0 to 3 and
		 * its channels 4 to 7 are at words 8 to 11, so an atom group is
		 * FOUR elements and the stride between groups is 4*m.
		 *
		 * int8 writes one byte an element and its atom is 16 elements;
		 * w4a16 writes four and its atom is 4. Sixteen bytes either way,
		 * which is one statement covering both paths, where "atom 16"
		 * and "the feature atom" were two guesses that each covered one.
		 *
		 * At M = 1 every atom collapses to n, which is why eleven
		 * geometries passed without this mattering.
		 */
		unsigned atom = 16 / (job.mm.wdtype == CHARSIU_INT4 ? 4 : 1);
		const int phantom = getenv("CHARSIU_W4_PHANTOM") || getenv("CHARSIU_W4_PAIRED");
		unsigned ni, mi, ex_pe = 0, ex_sum = 0, written = 0;
		const int32_t *o = (const int32_t *)outbo.map;

		printf("  int4: reading the output as %u signed 32 bit words, "
		       "surface [n/%u][m][n%%%u]\n", m * n, atom, atom);
		for (mi = 0; mi < m; mi++)
			for (ni = 0; ni < n; ni++) {
				int64_t pe = 0, sm = 0;
				int32_t got;
				unsigned j;

				for (j = 0; j < k; j++) {
					int wv = b_raw[ni * k + j];
					int16_t wb, ab;

					wv = (wv & 0x8) ? (wv & 0xf) - 16
							: (wv & 0xf);
					wb = (int16_t)charsiu_float_to_half(
						(float)wv);
					ab = (int16_t)((((int)a_raw[mi * k + j]
							 - 0x80) & 0xff) << 8);
					pe += ((int32_t)wb * (int32_t)ab) >> 16;
					sm += (int32_t)wb * (int32_t)ab;
				}
				sm >>= 16;
				got = o[(size_t)(ni / atom) * m * atom
					+ (size_t)mi * atom + ni % atom];
				/*
				 * ⚠ THE PHANTOM WORD IS THE OTHER HALF OF THE
				 * SUM, not a spare. Its index follows the same
				 * surface formula at channel ni + n.
				 */
				if (phantom) {
					unsigned a2 = 2 * ni, b3 = 2 * ni + 1;

					got = o[(size_t)(a2 / atom) * m * atom
						+ (size_t)mi * atom + a2 % atom]
					    + o[(size_t)(b3 / atom) * m * atom
						+ (size_t)mi * atom + b3 % atom];
				}
				if ((uint32_t)got != 0xa5a5a5a5u) written++;
				if (got == (int32_t)pe) ex_pe++;
				if (got == (int32_t)sm) ex_sum++;
				if (ni < 8 && !mi)
					printf("    ch %2u  npu %8d   shift/elem %8d   shift/sum %8d\n",
					       ni, got, (int)pe, (int)sm);
			}
		/*
		 * At M > 1 print the raw surface and both rows' references, so
		 * the mapping can be READ instead of inferred from a score. Two
		 * closed forms for the weight layout were guessed and refuted
		 * before --map was pointed at it; this is the same mistake
		 * waiting to be made about the output.
		 */
		if (m > 1) {
			unsigned q;

			printf("  raw surface, first %u words:\n   ", 4 * n);
			for (q = 0; q < 4 * n && q < 128; q++) {
				printf(" %d", o[q]);
				if ((q & 7) == 7) printf("\n   ");
			}
			printf("\n  reference by row, first 8 channels:\n");
			for (mi = 0; mi < m && mi < 3; mi++) {
				printf("    row %u:", mi);
				for (ni = 0; ni < 8; ni++) {
					int64_t pe = 0;
					unsigned j;

					for (j = 0; j < k; j++) {
						int wv = b_raw[ni * k + j];
						int16_t wb, ab;

						wv = (wv & 0x8) ? (wv & 0xf) - 16
								: (wv & 0xf);
						wb = (int16_t)charsiu_float_to_half((float)wv);
						ab = (int16_t)((((int)a_raw[mi * k + j]
								 - 0x80) & 0xff) << 8);
						pe += ((int32_t)wb * (int32_t)ab) >> 16;
					}
					printf(" %d", (int)pe);
				}
				printf("\n");
			}
		}

		/*
		 * THE SECOND REFERENCE: a plain weighted sum, which is what an
		 * LLM needs and what the bit pattern formula is not. Fitted to
		 * one global scale, because the hardware's units are its own.
		 */
		{
			double num = 0.0, den = 0.0;
			unsigned ci, hit = 0, nz = 0;

			for (ci = 0; ci < n; ci++) {
				double lin = 0.0;
				unsigned j;
				int32_t got2;

				for (j = 0; j < k; j++) {
					int wv = b_raw[(size_t)ci * k + j];

					wv = (wv & 0x8) ? (wv & 0xf) - 16
							: (wv & 0xf);
					lin += (double)wv
					       * ((double)a_raw[j] - 128.0);
				}
				got2 = o[(size_t)(ci / atom) * m * atom
					 + ci % atom];
				num += lin * (double)got2;
				den += lin * lin;
			}
			if (den > 0.0) {
				double sc = num / den;

				for (ci = 0; ci < n; ci++) {
					double lin = 0.0;
					unsigned j;
					int32_t got2;

					for (j = 0; j < k; j++) {
						int wv = b_raw[(size_t)ci * k + j];

						wv = (wv & 0x8) ? (wv & 0xf) - 16
								: (wv & 0xf);
						lin += (double)wv
						       * ((double)a_raw[j] - 128.0);
					}
					got2 = o[(size_t)(ci / atom) * m * atom
						 + ci % atom];
					if (lin != 0.0)
						nz++;
					if (lin != 0.0 && fabs((double)got2
							       - sc * lin)
					    <= 0.01 * fabs(sc * lin))
						hit++;
				}
				/*
				 * ⚠ A DEAD OUTPUT SCORED n OF n. Round 339's
				 * 3d timed out, every word came back zero, the
				 * least squares scale fitted to 0.000000 and
				 * every channel was "within 1% of zero". That
				 * is a check that cannot fail, which is the
				 * one kind this file is not allowed to print.
				 */
				unsigned live = 0;

				for (ci = 0; ci < n; ci++)
					if (o[(size_t)(ci / atom) * m * atom
					      + ci % atom])
						live++;
				if (!live || sc == 0.0)
					printf("  PLAIN WEIGHTED SUM: VOID, "
					       "the output is dead (%u live "
					       "words, scale %.6f)\n", live, sc);
				else
					printf("  PLAIN WEIGHTED SUM: %u of %u "
					       "channels within 1%% of one "
					       "global scale %.6f  (%u live)\n",
					       hit, nz, sc, live);
			}
		}
		printf("int4 output: %u of %u words written, "
		       "%u EXACT with the shift per element, "
		       "%u EXACT with the shift on the sum\n",
		       written, m * n, ex_pe, ex_sum);
		/*
		 * ⚠ WHICH channels, in groups of eight, because the layout is
		 * indexed by n/8 and a whole group being wrong is a different
		 * fault from a scatter. Round 281 printed only channels 0 to 7
		 * and its two partial results, 24 of 32 and 8 of 16, could not
		 * be read past the count.
		 */
		{
			unsigned gi;

			printf("  per group of 8 (row 0):");
			for (gi = 0; gi * 8 < n; gi++) {
				unsigned ok = 0, ci;

				for (ci = gi * 8; ci < (gi + 1) * 8 && ci < n;
				     ci++) {
					int64_t pe = 0;
					unsigned j;

					for (j = 0; j < k; j++) {
						int wv = b_raw[ci * k + j];
						int16_t wb, ab;

						wv = (wv & 0x8)
							? (wv & 0xf) - 16
							: (wv & 0xf);
						wb = (int16_t)
						  charsiu_float_to_half((float)wv);
						ab = (int16_t)((((int)
						  a_raw[j] - 0x80) & 0xff) << 8);
						pe += ((int32_t)wb
						       * (int32_t)ab) >> 16;
					}
					if (((const int32_t *)outbo.map)
					    [(size_t)(ci / atom) * m * atom
					     + ci % atom] == (int32_t)pe)
						ok++;
				}
				printf("  g%u %u/8", gi, ok);
			}
			printf("\n");

			/*
			 * ⚠ AND WHICH ONES. "g1 4/8" is as consistent with the
			 * even channels failing as with the odd, and round 289's
			 * N = 20 and round 290's M > 1 both come down to that
			 * distinction. Name them.
			 */
			printf("  row 0 mismatches:");
			{
				unsigned ci, shown = 0;

				for (ci = 0; ci < n; ci++) {
					int64_t pe = 0;
					unsigned j;

					for (j = 0; j < k; j++) {
						int wv = b_raw[ci * k + j];
						int16_t wb, ab;

						wv = (wv & 0x8) ? (wv & 0xf) - 16
								: (wv & 0xf);
						wb = (int16_t)charsiu_float_to_half((float)wv);
						ab = (int16_t)((((int)a_raw[j]
								 - 0x80) & 0xff) << 8);
						pe += ((int32_t)wb * (int32_t)ab) >> 16;
					}
					if (((const int32_t *)outbo.map)
					    [(size_t)(ci / atom) * m * atom
					     + ci % atom] != (int32_t)pe) {
						if (shown < 24)
							printf(" %u", ci);
						shown++;
					}
				}
				if (!shown)
					printf(" none");
				else if (shown > 24)
					printf(" ... %u total", shown);
				printf("\n");
			}
		}
		/*
		 * ⚠ WHERE THE VALUES WENT, which a score cannot say. Round 312
		 * set 0x40b8 = 3 on int4: the words WRITTEN went from 136 to 256
		 * and the exact count FELL from 128 to 16, with the survivors at
		 * group 0 and group 30. Those two numbers together are equally
		 * consistent with "the hardware now computes something else" and
		 * with "it computes the same thing and puts it somewhere else",
		 * and only the second one leaves int4's 2x intact.
		 *
		 * So look, do not fit. Every expected value is searched for across
		 * the WHOLE output buffer rather than read at the one index the
		 * surface formula picks, and what gets printed is where each
		 * channel actually landed. A reference of zero is skipped: it
		 * matches any untouched-but-zeroed word and would inflate the
		 * count for nothing, which is the round 169 mistake.
		 */
		{
			const int32_t *ow = (const int32_t *)outbo.map;
			size_t owords = outbo.size / 4, q;
			unsigned found = 0, dup = 0, nz = 0, ident = 0, distinct = 0;
			unsigned wr_all = 0, hi_all = 0, shown = 0;
			int64_t *refv = malloc((size_t)n * sizeof(*refv));

			for (q = 0; q < owords; q++)
				if ((uint32_t)ow[q] != 0xa5a5a5a5u) {
					wr_all++;
					hi_all = (unsigned)q;
				}
			printf("  placement: %u words written in the whole %zu word buffer,"
			       " highest index %u\n", wr_all, owords, hi_all);

			for (ci = 0; ci < n && refv; ci++) {
				int64_t pe = 0;
				unsigned j;

				for (j = 0; j < k; j++) {
					int wv = b_raw[(size_t)ci * k + j];
					int16_t wb, ab;

					wv = (wv & 0x8) ? (wv & 0xf) - 16 : (wv & 0xf);
					wb = (int16_t)charsiu_float_to_half((float)wv);
					ab = (int16_t)((((int)a_raw[j] - 0x80) & 0xff) << 8);
					pe += ((int32_t)wb * (int32_t)ab) >> 16;
				}
				refv[ci] = pe;
			}
			/*
			 * ⚠ HOW MANY OF THE REFERENCES ARE EVEN DIFFERENT FROM EACH
			 * OTHER. This is the line round 324 needed and did not have:
			 * its generator gave only 30 distinct channel patterns, so a
			 * search for a value found a DUPLICATE and every count below
			 * was inflated for free. If this is less than the channel
			 * count, nothing under it can be read as placement.
			 */
			for (ci = 0; ci < n && refv; ci++) {
				unsigned q2, seen = 0;

				for (q2 = 0; q2 < ci; q2++)
					if (refv[q2] == refv[ci]) { seen = 1; break; }
				if (!seen)
					distinct++;
			}
			printf("  placement: %u of %u references are distinct%s\n",
			       distinct, n,
			       distinct == n ? "" :
			       "   !! DUPLICATES: the placement below is NOT readable");

			printf("  placement: channel -> the word that holds its expected"
			       " value (= its own, -- = nowhere)\n   ");
			for (ci = 0; ci < n && refv; ci++) {
				unsigned hits = 0, own = 0;
				size_t at = 0, mine = (size_t)(ci / atom) * m * atom
						+ ci % atom;

				if (!refv[ci])
					continue;
				nz++;
				/*
				 * ⚠ THE OWN INDEX IS CHECKED AGAINST ALL THE HITS, not
				 * against the first one. Round 324 printed ident = 30 on
				 * a run whose score was 64 of 64, because a duplicate at
				 * a lower index won the search. A metric that disagrees
				 * with a known answer is the metric that is wrong.
				 */
				for (q = 0; q < owords; q++)
					if ((uint32_t)ow[q] != 0xa5a5a5a5u
					    && ow[q] == (int32_t)refv[ci]) {
						if (!hits)
							at = q;
						if (q == mine)
							own = 1;
						hits++;
					}
				if (hits) {
					found++;
					if (own) {
						ident++;
						at = mine;
					}
				}
				if (hits > 1)
					dup++;
				if (shown < 64) {
					if (!hits)
						printf(" %u->--", ci);
					else if (own)
						printf(" %u->=", ci);
					else
						printf(" %u->w%zu", ci, at);
					if ((shown & 7) == 7)
						printf("\n   ");
					shown++;
				}
			}
			/*
			 * ⚠ READ THE PAIRING, DO NOT GUESS IT. Round 330 packed
			 * the second half of each channel's k into word c+N and
			 * added out[c] + out[c+N], and got zero of 64 -- but the
			 * hardware DID write 2N words (128 at N=64, exactly 2N,
			 * the first time charsiu has made it do that). So the
			 * words are there and the pairing is wrong.
			 *
			 * So look for it. For every channel, every written word
			 * i, ask whether ref - o[i] is also a written word. That
			 * is O(2N) a channel with a lookup and it prints the
			 * rule instead of testing one.
			 */
			if (phantom) {
				unsigned ci2, shown2 = 0, hit = 0;

				printf("  pairing: channel -> the two words that"
				       " sum to its reference\n   ");
				for (ci2 = 0; ci2 < n; ci2++) {
					unsigned a, b2, found2 = 0;

					for (a = 0; a < 2 * n && !found2; a++) {
						if ((uint32_t)ow[a] == 0xa5a5a5a5u)
							continue;
						for (b2 = a; b2 < 2 * n; b2++) {
							if ((uint32_t)ow[b2] == 0xa5a5a5a5u)
								continue;
							if ((int64_t)ow[a] + ow[b2]
							    != refv[ci2])
								continue;
							if (shown2 < 24)
								printf(" %u=w%u+w%u",
								       ci2, a, b2);
							found2 = 1;
							hit++;
							break;
						}
					}
					if (!found2 && shown2 < 24)
						printf(" %u=--", ci2);
					if (shown2 < 24 && (shown2 & 3) == 3)
						printf("\n   ");
					shown2++;
				}
				printf("\n  pairing: %u of %u channels are the sum"
				       " of two written words\n", hit, n);
			}
			printf("\n  placement: %u of %u channels with a nonzero reference"
			       " (of %u) are present somewhere, %u of them at the index the"
			       " surface formula reads, %u at more than one index\n",
			       found, nz, n, ident, dup);
			printf("    found == nz and ident small   the values are all there and"
			       " only the PLACEMENT moved\n"
			       "    found near nz/2               still half width, permuted\n"
			       "    found tiny                    the hardware computed"
			       " something else\n");
			free(refv);
		}
		charsiu_close(dev);
		return 0;
	}

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
