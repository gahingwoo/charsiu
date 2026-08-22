// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * Can an int8 WEIGHT job hand back the raw accumulator?
 *
 * WHY THIS IS THE GATING QUESTION FOR PUTTING A PROJECTION ON THE NPU.
 * charsiu's int8 path requantises through the coefficient buffer and writes a
 * BYTE, and a coefficient buffer's scale is fixed at build time. Measured on
 * the CPU model of that format (CHARSIU_NPU_OUT8 in charsiu_run), the output
 * magnitude of ffn_down varies by up to 2971x between tokens, so a scale sized
 * for the largest vector quantises a typical one to nothing: the model goes
 * from the right sentence to "a country". A projection therefore cannot leave
 * the NPU as a byte.
 *
 * The w4a16 path already writes FOUR bytes an element under an identity
 * requant. This asks whether the same output stage works with int8 weights,
 * one register at a time, because "the whole bundle changed something" does not
 * say which register the width lives in.
 *
 *   CHARSIU_WIDE8=<mask>   bit 0 0x4010, 1 0x4030, 2 0x4038,
 *                          3 0x4044, 4 0x4050, 5 0x40ac/b0/b4 identity
 *
 * The reference is the plain accumulator, with no bias and no lift:
 *
 *   acc[n] = sum_k (a[k] - 128) * (w[n][k] - 128)
 *
 * and BOTH readings of the output are scored every run -- as bytes and as 32
 * bit words -- because a round that scores only the reading it hopes for cannot
 * tell a wide output from a broken one.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu.h"

#define SENTINEL 0xa5

int main(int argc, char **argv)
{
	unsigned m = argc > 1 ? (unsigned)atoi(argv[1]) : 1;
	unsigned k = argc > 2 ? (unsigned)atoi(argv[2]) : 64;
	unsigned n = argc > 3 ? (unsigned)atoi(argv[3]) : 64;
	struct charsiu_device *dev;
	struct charsiu_bo regcmd, in, wt, out, coef;
	struct charsiu_job job;
	uint8_t *a_raw, *b_raw;
	int32_t *bias, *wsums, *ref;
	size_t outbytes;
	unsigned i, j, nreg;
	const char *mask = getenv("CHARSIU_WIDE8");

	memset(&job, 0, sizeof(job));
	job.mm.m = m;
	job.mm.k = k;
	job.mm.n = n;
	job.mm.wdtype = CHARSIU_INT8;
	job.mm.adtype = CHARSIU_INT8;
	job.input_scale = 0.02f;
	job.weight_scale = 0.18f;
	job.output_scale = 0.25f;
	job.input_zero_point = 128;
	job.weight_zero_point = 128;
	job.output_zero_point = 0;
	/*
	 * CHARSIU_ACC uses the first class mode instead of the register mask:
	 * job.acc_out turns on the whole w4a16 output stage plus 0x40b8 = 3,
	 * which round 312 measured at 1024 of 1024 elements exact.
	 */
	job.acc_out = getenv("CHARSIU_ACC") != NULL;

	printf("wide8 probe  M=%u K=%u N=%u int8   acc_out=%d  CHARSIU_WIDE8=%s"
	       "  coef %zu bytes\n",
	       m, k, n, job.acc_out, mask ? mask : "(unset)",
	       charsiu_coef_bytes(&job.mm));

	/*
	 * No bias and no lift, so the expected accumulator is exactly the sum of
	 * products and nothing has to be unpicked from it.
	 */
	setenv("CHARSIU_NO_LIFT", "1", 1);

	a_raw = malloc((size_t)m * k);
	b_raw = malloc((size_t)n * k);
	bias = calloc(n, sizeof(*bias));
	wsums = calloc(n, sizeof(*wsums));
	ref = calloc((size_t)m * n, sizeof(*ref));
	if (!a_raw || !b_raw || !bias || !wsums || !ref)
		return 1;

	/* operands with a wide spread, so a flat reference cannot flatter a match */
	for (i = 0; i < m * k; i++)
		a_raw[i] = (uint8_t)(128 + ((int)(i * 37 % 241) - 120));
	for (i = 0; i < n * k; i++)
		b_raw[i] = (uint8_t)(128 + ((int)(i * 53 % 197) - 98));

	for (i = 0; i < n; i++) {
		int32_t s = 0;

		for (j = 0; j < k; j++)
			s += (int)b_raw[i * k + j] - job.weight_zero_point;
		wsums[i] = s;
	}
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++) {
			int64_t acc = 0;
			unsigned q;

			for (q = 0; q < k; q++)
				acc += ((int)a_raw[i * k + q] - job.input_zero_point) *
				       ((int)b_raw[j * k + q] - job.weight_zero_point);
			ref[i * n + j] = (int32_t)acc;
		}

	dev = charsiu_open("/dev/accel/accel0");
	if (!dev) {
		printf("open failed\n");
		return 1;
	}

	/* four bytes an element ALWAYS, so a one byte output is still readable */
	outbytes = (size_t)m * n * 4 + 4096;
	if (charsiu_bo_alloc(dev, 4096, &regcmd) ||
	    charsiu_bo_alloc(dev, (size_t)charsiu_entries_per_row(&job.mm) * 64 * m + 4096, &in) ||
	    charsiu_bo_alloc(dev, charsiu_weight_bytes(&job.mm) + 4096, &wt) ||
	    charsiu_bo_alloc(dev, outbytes, &out) ||
	    charsiu_bo_alloc(dev, charsiu_coef_bytes(&job.mm) + 4096, &coef)) {
		printf("alloc failed\n");
		return 1;
	}

	job.input_addr = (uint32_t)in.dma_address;
	job.weight_addr = (uint32_t)wt.dma_address;
	job.output_addr = (uint32_t)out.dma_address;
	job.coef_addr = (uint32_t)coef.dma_address;

	charsiu_bo_prep(dev, &in, 1000000000);
	charsiu_pack_input(&job.mm, a_raw, in.map, in.size, job.input_zero_point);
	charsiu_bo_fini(dev, &in);
	charsiu_bo_prep(dev, &wt, 1000000000);
	charsiu_pack_weights(&job.mm, b_raw, wt.map);
	charsiu_bo_fini(dev, &wt);
	charsiu_bo_prep(dev, &coef, 1000000000);
	charsiu_build_coefs(&job, bias, wsums, coef.map);
	charsiu_bo_fini(dev, &coef);
	charsiu_bo_prep(dev, &out, 1000000000);
	memset(out.map, SENTINEL, out.size);
	charsiu_bo_fini(dev, &out);

	charsiu_bo_prep(dev, &regcmd, 1000000000);
	nreg = (unsigned)charsiu_emit_job(&job, regcmd.map, 4096 / 8);
	charsiu_bo_fini(dev, &regcmd);
	if (!nreg) {
		printf("emit failed\n");
		return 1;
	}
	printf("register stream: %u entries\n", nreg);

	{
		uint32_t inh[3] = { in.handle, wt.handle, coef.handle };
		uint32_t outh = out.handle;

		if (charsiu_submit(dev, &regcmd, nreg, inh, 3, &outh, 1)) {
			printf("submit FAILED\n");
			return 1;
		}
	}
	printf("submit ok\n");

	charsiu_bo_prep(dev, &out, 2000000000);
	{
		const uint8_t *ob = out.map;
		const int32_t *ow = out.map;
		unsigned touched = 0, byte_exact = 0, word_exact = 0;
		unsigned nonsent_hi = 0;

		for (i = 0; i < m * n * 4; i++)
			if (ob[i] != SENTINEL)
				touched++;

		/*
		 * READING A: bytes, the surface int8 writes today, atom 16.
		 * READING B: 32 bit words, the surface w4a16 writes, atom 4.
		 * Both are scored. The byte reading is compared against the
		 * accumulator clamped to a byte, which is what a requant to
		 * int8 with this scale would give; the word reading against the
		 * accumulator itself.
		 */
		for (i = 0; i < m; i++)
			for (j = 0; j < n; j++) {
				unsigned atom8 = 16, g8 = j / atom8;
				size_t bidx = (size_t)(g8 * m + i) * atom8 + (j % atom8);
				unsigned atom32 = 4, g32 = j / atom32;
				size_t widx = (size_t)(g32 * m + i) * atom32 + (j % atom32);
				int32_t want = ref[i * n + j];
				float v = (float)want * job.input_scale *
					  job.weight_scale / job.output_scale;
				int wantb = (int)(v < 0 ? v - 0.5f : v + 0.5f);

				if (wantb > 127) wantb = 127;
				if (wantb < -128) wantb = -128;
				if ((int8_t)ob[bidx] == (int8_t)wantb)
					byte_exact++;
				if (ow[widx] == want)
					word_exact++;
			}

		/* a byte output leaves three sentinel bytes in every word */
		for (i = 0; i < m * n; i++) {
			const uint8_t *w4 = ob + (size_t)i * 4;

			if (w4[1] != SENTINEL || w4[2] != SENTINEL || w4[3] != SENTINEL)
				nonsent_hi++;
		}

		printf("\n  %u of %zu output bytes moved off the sentinel\n",
		       touched, (size_t)m * n * 4);
		printf("  %u of %u elements have a non sentinel byte ABOVE the low one"
		       "   (a one byte output leaves these alone)\n",
		       nonsent_hi, m * n);
		printf("\n  reading as BYTES  (requantised, atom 16): %u of %u exact\n",
		       byte_exact, m * n);
		printf("  reading as WORDS  (accumulator, atom 4) : %u of %u exact\n",
		       word_exact, m * n);

		printf("\n  first 8 elements, raw:\n");
		for (i = 0; i < (m * n < 8 ? m * n : 8); i++)
			printf("    %2u  bytes %02x %02x %02x %02x   word %11d"
			       "   accumulator %11d\n", i,
			       ob[i * 4], ob[i * 4 + 1], ob[i * 4 + 2], ob[i * 4 + 3],
			       ow[i], ref[i]);
	}
	charsiu_bo_fini(dev, &out);

	charsiu_bo_free(dev, &regcmd);
	charsiu_bo_free(dev, &in);
	charsiu_bo_free(dev, &wt);
	charsiu_bo_free(dev, &out);
	charsiu_bo_free(dev, &coef);
	charsiu_close(dev);
	return 0;
}
