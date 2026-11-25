// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * CHARSIU_NPU_PACK2 against the pack it replaces, byte for byte.
 *
 * batch_pack_slice (src/npudev.c) packs one K slice of X into one device's
 * input object. PACK2 changes three things about HOW and promises the bytes
 * do not move:
 *
 *   rows    the pool splits the packer by rows, four at a time, instead of
 *           by groups (charsiu_pack_input_f16_rows against _groups);
 *   direct  a slice of a cut tensor is packed straight out of X at the row
 *           stride K, instead of gathered into a scratch at stride sk first;
 *   copy    the second device's object is a copy of the first one's -- the
 *           data and its k % 8 block, zeros after -- instead of a second pack.
 *
 * batch_pack_slice needs an open device, which the desk does not have, so the
 * two flows are spelled out here with the same library calls in the same
 * order, and the comparison is on the two whole objects [0, bin_stride) of
 * both devices, poisoned beforehand so a byte nobody wrote shows up. The
 * shapes are the real ones: K 1024, 2048, 3072 and 6144 cut at 1024 and
 * 2048, every row count from 1 to 160, a bank wider than the call (bm > m),
 * a slice that is not a multiple of 8, and the int8 device's packer for the
 * copy, whose fill is the zero point rather than zero.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charsiu.h"

static int fail;
static unsigned long cases;
static uint64_t rng = 0x9e3779b97f4a7c15ull;

static float frand(void)
{
	rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
	/* a wide spread, so the half's exponent, flush and saturation all run */
	switch (rng >> 60) {
	case 0: return 0.0f;
	case 1: return (float)((int64_t)(rng >> 20) % 2000000) * 1e-12f;
	case 2: return (float)((int64_t)(rng >> 20) % 2000000) * 1e-1f;
	default: return (float)((int64_t)(rng >> 16) % 200001 - 100000) * 1e-4f;
	}
}

/* the pool's split as pack_f16_pooled hands it: ranges from charsiu_parallel_for
 * (_grain 4 for rows) -- modelled by cutting [0, n) into `parts` pieces, the
 * starts rounded to `grain` */
static void split(unsigned n, unsigned parts, unsigned grain,
		  unsigned *start, unsigned *len, unsigned *np)
{
	unsigned per = (n + parts - 1) / parts, a = 0;

	if (grain > 1)
		per = (per + grain - 1) / grain * grain;
	*np = 0;
	while (a < n) {
		unsigned l = n - a < per ? n - a : per;

		start[*np] = a; len[*np] = l; (*np)++;
		a += l;
	}
}

/* the old f16 slice: gather when cut, then the group split (or the serial
 * whole buffer entry when the shape has no groups) */
static void old_f16(const struct charsiu_matmul *mm, const float *X, unsigned K,
		    unsigned k0, float *scr, uint8_t *dst, size_t n,
		    unsigned parts)
{
	unsigned sk = mm->k, m = mm->m, ng, st[256], ln[256], np, i;
	const float *src = X + k0;
	size_t stride = K;

	if (sk != K) {
		for (unsigned r = 0; r < m; r++)
			memcpy(scr + (size_t)r * sk, X + (size_t)r * K + k0,
			       sk * sizeof(float));
		src = scr;
		stride = sk;
	}
	ng = charsiu_pack_input_f16_ngroups(mm);
	if (!ng) {
		charsiu_pack_input_f16_stride(mm, src, stride, dst, n);
		return;
	}
	charsiu_pack_input_f16_edges(mm, src, stride, dst, n);
	split(ng, parts, 1, st, ln, &np);
	for (i = 0; i < np; i++)
		charsiu_pack_input_f16_groups(mm, src, stride, dst, st[i], ln[i]);
}

/* the new f16 slice: no gather, the row split */
static void new_f16(const struct charsiu_matmul *mm, const float *X, unsigned K,
		    unsigned k0, uint8_t *dst, size_t n, unsigned parts)
{
	unsigned m = mm->m, st[256], ln[256], np, i;

	if (!charsiu_pack_input_f16_ngroups(mm)) {
		charsiu_pack_input_f16_stride(mm, X + k0, K, dst, n);
		return;
	}
	charsiu_pack_input_f16_edges(mm, X + k0, K, dst, n);
	split(m, parts, 4, st, ln, &np);
	for (i = 0; i < np; i++)
		charsiu_pack_input_f16_rows(mm, X + k0, K, dst, st[i], ln[i]);
}

/* the second device's object, as batch_pack_slice's copy branch writes it */
static void copy_dev(const uint8_t *src, uint8_t *dst, size_t n, unsigned sk,
		     unsigned m, int f16)
{
	size_t used = n;

	if (f16) {
		used = (size_t)(sk / 8) * m * 16 + (sk % 8 ? (size_t)m * 16 : 0);
		if (used > n)
			used = n;
	}
	memcpy(dst, src, used);
	if (n > used)
		memset(dst + used, 0, n - used);
}

static void differ(const char *what, const uint8_t *a, const uint8_t *b,
		   size_t n, unsigned m, unsigned K, unsigned kmax, unsigned ki,
		   unsigned parts)
{
	size_t i;

	if (!memcmp(a, b, n))
		return;
	for (i = 0; i < n && a[i] == b[i]; i++)
		;
	if (fail < 20)
		printf("  %s: m=%u K=%u kmax=%u slice %u parts %u differ at"
		       " byte %zu of %zu\n", what, m, K, kmax, ki, parts, i, n);
	fail++;
}

/* one tensor, every slice, both devices, both flows */
static void tensor(unsigned m, unsigned bm, unsigned K, unsigned kmax,
		   unsigned parts)
{
	unsigned nks = (K + kmax - 1) / kmax, ki;
	size_t bin_stride = (size_t)kmax * bm * 2;
	float *X = malloc((size_t)m * K * sizeof(float));
	float *scr = malloc((size_t)m * kmax * sizeof(float));
	uint8_t *o0 = malloc(bin_stride), *o1 = malloc(bin_stride);
	uint8_t *n0 = malloc(bin_stride), *n1 = malloc(bin_stride);

	for (size_t i = 0; i < (size_t)m * K; i++)
		X[i] = frand();
	for (ki = 0; ki < nks; ki++) {
		unsigned k0 = ki * kmax, sk = K - k0 < kmax ? K - k0 : kmax;
		struct charsiu_matmul mm = { m, sk, 64, CHARSIU_FP16,
					     CHARSIU_FP16 };

		cases++;
		memset(o0, 0xa5, bin_stride); memset(o1, 0x5a, bin_stride);
		memset(n0, 0xc3, bin_stride); memset(n1, 0x3c, bin_stride);
		old_f16(&mm, X, K, k0, scr, o0, bin_stride, parts);
		old_f16(&mm, X, K, k0, scr, o1, bin_stride, parts);
		new_f16(&mm, X, K, k0, n0, bin_stride, parts);
		copy_dev(n0, n1, bin_stride, sk, m, 1);
		differ("f16 device 0", o0, n0, bin_stride, m, K, kmax, ki, parts);
		differ("f16 device 1", o1, n1, bin_stride, m, K, kmax, ki, parts);
	}
	free(X); free(scr); free(o0); free(o1); free(n0); free(n1);
}

/* the int8 device: only the copy changes, and its fill is the zero point */
static void int8_copy(unsigned m, unsigned bm, unsigned sk, uint8_t zp)
{
	size_t n = (size_t)sk * bm;
	struct charsiu_matmul mm = { m, sk, 64, CHARSIU_INT8, CHARSIU_INT8 };
	uint8_t *q = malloc((size_t)m * sk);
	uint8_t *a = malloc(n), *b = malloc(n), *c = malloc(n);

	cases++;
	for (size_t i = 0; i < (size_t)m * sk; i++)
		q[i] = (uint8_t)(frand() * 1e3f);
	memset(a, 0xa5, n); memset(b, 0x5a, n); memset(c, 0x11, n);
	charsiu_pack_input(&mm, q, a, n, zp);
	charsiu_pack_input(&mm, q, b, n, zp);
	copy_dev(a, c, n, sk, m, 0);
	differ("int8 device 1", b, c, n, m, sk, sk, 0, 1);
	free(q); free(a); free(b); free(c);
}

/*
 * charsiu_f2h_x8 against charsiu_f2h_x4 on every float bit pattern: the rows
 * packer uses the first and the old packer the second, so this is the
 * converter half of "the bytes do not move". About ten seconds on the desk.
 */
static void every_float(void)
{
#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
	union { uint32_t u[8]; float f[8]; } b;
	unsigned long bad = 0;
	uint64_t u;

	for (u = 0; u < (1ull << 32); u += 8) {
		uint16_t x[8], y[8];

		for (unsigned i = 0; i < 8; i++)
			b.u[i] = (uint32_t)(u + i);
		vst1q_u16(x, charsiu_f2h_x8(b.f));
		vst1_u16(y, charsiu_f2h_x4(vld1q_f32(b.f)));
		vst1_u16(y + 4, charsiu_f2h_x4(vld1q_f32(b.f + 4)));
		if (memcmp(x, y, sizeof(x))) {
			if (bad < 5)
				printf("  f2h_x8 differs near 0x%08x\n",
				       (unsigned)u);
			bad++;
		}
	}
	printf("  f2h_x8 against f2h_x4: %lu of %llu blocks of 8 differ\n",
	       bad, 1ull << 29);
	if (bad)
		fail++;
#endif
}

int main(void)
{
	static const unsigned Ks[] = { 1024, 2048, 3072, 6144 };
	static const unsigned kmaxs[] = { 1024, 2048 };
	static const unsigned partss[] = { 1, 3, 4 };
	unsigned m, a, b, c;

	for (m = 1; m <= 160; m++)
		for (a = 0; a < sizeof(Ks) / sizeof(*Ks); a++)
			for (b = 0; b < sizeof(kmaxs) / sizeof(*kmaxs); b++) {
				c = m % 3;  /* a different split per row count */
				tensor(m, m, Ks[a], kmaxs[b], partss[c]);
			}
	/* a bank sized for a wider call, so the tail memset and the copy's
	 * zeroing both have something to do */
	for (m = 2; m <= 160; m += 13)
		tensor(m, 160, 3072, 1024, 4);
	/* slices that are not a multiple of the atom: 1000 = 125 groups, and a
	 * K whose last slice is ragged */
	for (m = 1; m <= 40; m += 3) {
		tensor(m, m, 1000, 1000, 4);
		tensor(m, m + 5, 2500, 1024, 3);
		tensor(m, m, 12, 12, 2);
	}
	for (m = 1; m <= 160; m += 7) {
		int8_copy(m, m, 1024, 128);
		int8_copy(m, 160, 1024, 0x7b);
		int8_copy(m, m + 3, 1000, 0x90);
	}
	every_float();
	printf("  pack2 against the old pack: %d mismatched objects over %lu slices\n",
	       fail, cases);
	return fail ? 1 : 0;
}
