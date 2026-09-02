/*
 * Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
 * SPDX-License-Identifier: GPL-2.0
 *
 * The batched read back, on the host: gather off the table, or walk the
 * surface and scatter?
 *
 * read_rows in src/npudev.c reads the accumulator surface FOUR FLOATS AT A
 * TIME OFF ONE TABLE INDEX, row by row of Y. Its own comment did the sum: a
 * run is 16 bytes and a cache line is 64, the runs are scattered, so the DRAM
 * sees four times the surface -- 7 GB/s of the board's 9.4 at m = 80 -- and a
 * NEON form of the same loop moved nothing, because it was never instruction
 * bound. "The only lever left is fewer BYTES: walking the source sequentially
 * and scattering into Y would use whole lines instead of a quarter of each."
 *
 * This is that sentence, measured. Three arms on the same surface, the same
 * arithmetic, compared byte for byte:
 *
 *   gather   the shipped loop: for each row of Y, for each quad, one table
 *            index, four floats from the surface, four to Y
 *   walk     the surface in order, four floats at a time, each quad sent to
 *            the row and channel it belongs to through an INVERSE table
 *            built by enumerating charsiu_acc_index once per shape
 *   walk-f   the same walk with the row and channel computed from the
 *            surface position by the layout's own formula, no table at all
 *            (roleswap2 only, which is what w4a16 prefill reads); checked
 *            against walk on every shape
 *   gather4  the shipped order, four rows at a time: the four rows that
 *            share a 64-byte line of the surface are read off it once and
 *            written to four row streams
 *
 * ⚠ WHAT THE HOST CAN AND CANNOT SAY. The host is aarch64 with caches that
 * dwarf the board's; a surface that fits in its L2 will not show the DRAM
 * argument at all. The shapes are chosen to exceed it (m = 80 by n = 8192 is
 * 2.6 MB a slot, and the loop runs over 24 slots the way a layer does) and the
 * ratio is the thing to carry, not the milliseconds. The vision attention work
 * moved 3.71x here and 10.32x on the card for the same reason: bytes.
 *
 *   build/bench_gather [m ...]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "charsiu.h"

static double now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* the shipped gather, W4G form (per channel scale), assign then add */
static void gather(const float *fo, const uint32_t *mp, unsigned n4,
		   const float *sc, float *yr, int add)
{
	unsigned j;

	if (!add) {
		for (j = 0; j < n4; j++) {
			const float *fp = fo + mp[j], *cp = sc + j * 4;
			float *yp = yr + j * 4;

			yp[0] = fp[0] * cp[0];
			yp[1] = fp[1] * cp[1];
			yp[2] = fp[2] * cp[2];
			yp[3] = fp[3] * cp[3];
		}
	} else {
		for (j = 0; j < n4; j++) {
			const float *fp = fo + mp[j], *cp = sc + j * 4;
			float *yp = yr + j * 4;

			yp[0] += fp[0] * cp[0];
			yp[1] += fp[1] * cp[1];
			yp[2] += fp[2] * cp[2];
			yp[3] += fp[3] * cp[3];
		}
	}
}

/*
 * the walk: quad q of the surface belongs to row inv[q] >> 16 and channel
 * quad inv[q] & 0xffff. Same multiply, same order of operations per element,
 * so the result is the same bits.
 */
static void walk(const float *fo, const uint32_t *inv, unsigned nq,
		 unsigned n, const float *sc, float *Y, int add)
{
	unsigned q;

	if (!add) {
		for (q = 0; q < nq; q++) {
			const float *fp = fo + (size_t)q * 4;
			unsigned r = inv[q] >> 16, j4 = inv[q] & 0xffffu;
			const float *cp = sc + j4 * 4;
			float *yp = Y + (size_t)r * n + j4 * 4;

			yp[0] = fp[0] * cp[0];
			yp[1] = fp[1] * cp[1];
			yp[2] = fp[2] * cp[2];
			yp[3] = fp[3] * cp[3];
		}
	} else {
		for (q = 0; q < nq; q++) {
			const float *fp = fo + (size_t)q * 4;
			unsigned r = inv[q] >> 16, j4 = inv[q] & 0xffffu;
			const float *cp = sc + j4 * 4;
			float *yp = Y + (size_t)r * n + j4 * 4;

			yp[0] += fp[0] * cp[0];
			yp[1] += fp[1] * cp[1];
			yp[2] += fp[2] * cp[2];
			yp[3] += fp[3] * cp[3];
		}
	}
}

/*
 * the walk with the roleswap2 layout inverted by hand:
 *
 *   index = G*m*32 + a*32P + (t/4)*8P + (mi/2)*8 + (mi%2)*4 + t%4
 *
 * so quad q (= index/4) decomposes, P = m/2, as
 *
 *   G   = q / (8m)            a group of 32 channels holds 8m quads
 *   a   = (q / (8P)) % 2      then a half of 16 channels, 8P quads each
 *   tq  = (q / (2P)) % 4      then a channel quad, 2P quads each
 *   hi  = (q / 2) % P         then a row pair
 *   lo  = q % 2               then the row of the pair
 *
 * row = 2*hi + lo, channel quad = G*8 + a*4 + tq.
 */
static void walk_f(const float *fo, unsigned m, unsigned n, const float *sc,
		   float *Y, int add)
{
	unsigned P = m / 2, nq = m * n / 4, q;

	for (q = 0; q < nq; q++) {
		const float *fp = fo + (size_t)q * 4;
		unsigned G = q / (8u * m);
		unsigned a = (q / (8u * P)) % 2u;
		unsigned tq = (q / (2u * P)) % 4u;
		unsigned hi = (q / 2u) % P;
		unsigned lo = q % 2u;
		unsigned r = 2u * hi + lo, j4 = G * 8u + a * 4u + tq;
		const float *cp = sc + j4 * 4;
		float *yp = Y + (size_t)r * n + j4 * 4;

		if (!add) {
			yp[0] = fp[0] * cp[0];
			yp[1] = fp[1] * cp[1];
			yp[2] = fp[2] * cp[2];
			yp[3] = fp[3] * cp[3];
		} else {
			yp[0] += fp[0] * cp[0];
			yp[1] += fp[1] * cp[1];
			yp[2] += fp[2] * cp[2];
			yp[3] += fp[3] * cp[3];
		}
	}
}

/*
 * gather4: FOUR ROWS OFF ONE LINE. In the roleswap2 layout rows 2hi, 2hi+1,
 * 2hi+2, 2hi+3 hold the same channel quad at index, index+4, index+8,
 * index+12 -- one 64-byte line when hi is even and the block bases are
 * line aligned, which they are whenever m is a multiple of 4. The shipped
 * gather reads that line once per row, four times over, and relies on the
 * L2 to have kept it between row passes; this reads it once and writes the
 * four rows' quads to four sequential streams. Same arithmetic per element.
 * Rows left over when m is not a multiple of 4 go through the plain gather.
 */
static void gather4(const float *fo, const uint32_t *map, unsigned m,
		    unsigned n4, const float *sc, float *Y, unsigned n, int add)
{
	unsigned r, j;

	for (r = 0; r + 4 <= m; r += 4) {
		const uint32_t *mp = map + (size_t)r * n4;
		float *y0 = Y + (size_t)r * n, *y1 = y0 + n, *y2 = y1 + n,
		      *y3 = y2 + n;

		if (!add) {
			for (j = 0; j < n4; j++) {
				const float *fp = fo + mp[j], *cp = sc + j * 4;

				y0[j * 4 + 0] = fp[0]  * cp[0];
				y0[j * 4 + 1] = fp[1]  * cp[1];
				y0[j * 4 + 2] = fp[2]  * cp[2];
				y0[j * 4 + 3] = fp[3]  * cp[3];
				y1[j * 4 + 0] = fp[4]  * cp[0];
				y1[j * 4 + 1] = fp[5]  * cp[1];
				y1[j * 4 + 2] = fp[6]  * cp[2];
				y1[j * 4 + 3] = fp[7]  * cp[3];
				y2[j * 4 + 0] = fp[8]  * cp[0];
				y2[j * 4 + 1] = fp[9]  * cp[1];
				y2[j * 4 + 2] = fp[10] * cp[2];
				y2[j * 4 + 3] = fp[11] * cp[3];
				y3[j * 4 + 0] = fp[12] * cp[0];
				y3[j * 4 + 1] = fp[13] * cp[1];
				y3[j * 4 + 2] = fp[14] * cp[2];
				y3[j * 4 + 3] = fp[15] * cp[3];
			}
		} else {
			for (j = 0; j < n4; j++) {
				const float *fp = fo + mp[j], *cp = sc + j * 4;

				y0[j * 4 + 0] += fp[0]  * cp[0];
				y0[j * 4 + 1] += fp[1]  * cp[1];
				y0[j * 4 + 2] += fp[2]  * cp[2];
				y0[j * 4 + 3] += fp[3]  * cp[3];
				y1[j * 4 + 0] += fp[4]  * cp[0];
				y1[j * 4 + 1] += fp[5]  * cp[1];
				y1[j * 4 + 2] += fp[6]  * cp[2];
				y1[j * 4 + 3] += fp[7]  * cp[3];
				y2[j * 4 + 0] += fp[8]  * cp[0];
				y2[j * 4 + 1] += fp[9]  * cp[1];
				y2[j * 4 + 2] += fp[10] * cp[2];
				y2[j * 4 + 3] += fp[11] * cp[3];
				y3[j * 4 + 0] += fp[12] * cp[0];
				y3[j * 4 + 1] += fp[13] * cp[1];
				y3[j * 4 + 2] += fp[14] * cp[2];
				y3[j * 4 + 3] += fp[15] * cp[3];
			}
		}
	}
	for (; r < m; r++)
		gather(fo, map + (size_t)r * n4, n4, sc, Y + (size_t)r * n, add);
}

static int bench(unsigned m, unsigned n, unsigned nslot, int reps)
{
	size_t surf = (size_t)m * n;           /* floats a slot */
	unsigned n4 = n / 4, nq = m * n / 4;
	float *fo = malloc(surf * nslot * sizeof(*fo));
	float *sc = malloc((size_t)n * sizeof(*sc));
	float *Y1 = malloc(surf * sizeof(*Y1));
	float *Y2 = malloc(surf * sizeof(*Y2));
	float *Y3 = malloc(surf * sizeof(*Y3));
	float *Y4 = malloc(surf * sizeof(*Y4));
	uint32_t *map = malloc((size_t)m * n4 * sizeof(*map));
	uint32_t *inv = malloc((size_t)nq * sizeof(*inv));
	double t, tg = 1e30, tw = 1e30, tf = 1e30, t4 = 1e30;
	int rc = 0, k;

	if (!fo || !sc || !Y1 || !Y2 || !Y3 || !Y4 || !map || !inv) {
		fprintf(stderr, "out of memory at m=%u n=%u\n", m, n);
		return 1;
	}
	for (size_t i = 0; i < surf * nslot; i++)
		fo[i] = (float)(i % 9973) * 0.25f;
	for (unsigned j = 0; j < n; j++)
		sc[j] = 1.0f + (float)(j % 7) * 0.125f;
	/* the forward table, exactly as npudev builds it */
	for (unsigned r = 0; r < m; r++)
		for (unsigned j = 0; j < n4; j++)
			map[(size_t)r * n4 + j] =
				(uint32_t)charsiu_acc_index(r, j * 4, m, 1);
	/*
	 * gather4's premise, CHECKED rather than assumed: rows 4h..4h+3 of any
	 * channel quad sit at index, +4, +8, +12. If the layout ever stops
	 * saying so, this says so before a wrong number gets printed.
	 */
	if (m % 4 == 0)
		for (unsigned r = 0; r < m; r += 4)
			for (unsigned j = 0; j < n4; j++)
				for (unsigned lo = 1; lo < 4; lo++)
					if (map[(size_t)(r + lo) * n4 + j] !=
					    map[(size_t)r * n4 + j] + lo * 4) {
						fprintf(stderr, "m=%u n=%u: rows are "
							"not four to a line at r=%u "
							"j=%u\n", m, n, r, j);
						return 1;
					}
	/* and its inverse, by enumeration: a bijection, so every quad lands */
	memset(inv, 0xff, (size_t)nq * sizeof(*inv));
	for (unsigned r = 0; r < m; r++)
		for (unsigned j = 0; j < n4; j++) {
			uint32_t idx = map[(size_t)r * n4 + j];

			if (idx % 4 || idx / 4 >= nq || inv[idx / 4] != 0xffffffffu) {
				fprintf(stderr, "m=%u n=%u: the map is not a quad "
					"bijection at r=%u j=%u (idx %u)\n",
					m, n, r, j, idx);
				return 1;
			}
			inv[idx / 4] = (r << 16) | j;
		}

	for (k = 0; k < reps; k++) {
		/* gather: slot 0 assigns, the rest accumulate, like K slices */
		t = now_ms();
		for (unsigned s = 0; s < nslot; s++)
			for (unsigned r = 0; r < m; r++)
				gather(fo + (size_t)s * surf, map + (size_t)r * n4,
				       n4, sc, Y1 + (size_t)r * n, s != 0);
		t = now_ms() - t;
		if (t < tg) tg = t;

		t = now_ms();
		for (unsigned s = 0; s < nslot; s++)
			walk(fo + (size_t)s * surf, inv, nq, n, sc, Y2, s != 0);
		t = now_ms() - t;
		if (t < tw) tw = t;

		t = now_ms();
		for (unsigned s = 0; s < nslot; s++)
			walk_f(fo + (size_t)s * surf, m, n, sc, Y3, s != 0);
		t = now_ms() - t;
		if (t < tf) tf = t;

		t = now_ms();
		for (unsigned s = 0; s < nslot; s++)
			gather4(fo + (size_t)s * surf, map, m, n4, sc, Y4, n, s != 0);
		t = now_ms() - t;
		if (t < t4) t4 = t;
	}
	if (memcmp(Y1, Y4, surf * sizeof(*Y1))) {
		printf("  m=%3u n=%5u: gather4 DIFFERS from gather\n", m, n);
		rc = 1;
	}
	if (memcmp(Y1, Y2, surf * sizeof(*Y1))) {
		printf("  m=%3u n=%5u: walk DIFFERS from gather\n", m, n);
		rc = 1;
	}
	if (memcmp(Y1, Y3, surf * sizeof(*Y1))) {
		printf("  m=%3u n=%5u: walk-f DIFFERS from gather (the hand "
		       "inverse is wrong)\n", m, n);
		rc = 1;
	}
	{
		double mb = (double)surf * nslot * sizeof(float) / 1e6;

		printf("  m=%3u n=%5u x%u (%6.1f MB): gather %7.2f ms"
		       "  walk %.2fx  walk-f %.2fx  gather4 %7.2f ms (%.2fx)%s\n",
		       m, n, nslot, mb, tg, tg / tw, tg / tf, t4, tg / t4,
		       rc ? "  ⚠ NOT IDENTICAL" : "  identical");
	}
	free(fo); free(sc); free(Y1); free(Y2); free(Y3); free(Y4); free(map); free(inv);
	return rc;
}

int main(int argc, char **argv)
{
	static const unsigned ms[] = { 32, 48, 80 };
	static const unsigned ns[] = { 1024, 2048, 3072, 8192 };
	int rc = 0;

	printf("the batched read back: gather off the table against walking the "
	       "surface\n  (roleswap2, the w4a16 prefill layout; min of 5; the "
	       "ratio is what carries, not the ms)\n");
	if (argc > 1) {
		for (int i = 1; i < argc; i++)
			for (unsigned j = 0; j < sizeof(ns) / sizeof(ns[0]); j++)
				rc |= bench((unsigned)atoi(argv[i]), ns[j], 24, 5);
		return rc;
	}
	for (unsigned i = 0; i < sizeof(ms) / sizeof(ms[0]); i++)
		for (unsigned j = 0; j < sizeof(ns) / sizeof(ns[0]); j++)
			rc |= bench(ms[i], ns[j], 24, 5);
	return rc;
}
