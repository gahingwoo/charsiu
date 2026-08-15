// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * Read the int4 weight layout off the hardware, one nibble pattern at a time.
 *
 * WHY A SEPARATE TOOL. int4 is the only 2x left: decode is DRAM bound at 11.9
 * GB/s, so halving the weight bytes halves the token time, 85 ms to 44 ms on
 * Llama-3.2-1B. The registers for it are no longer a guess. Diffing the vendor's
 * own int4 and int8 streams out of a .rkllm (vendor-capture/int4_regs.py) says
 *
 *   0x100c   bit 22 selects int4 weights, bit 29 selects 16 bit activations
 *   0x101c   real bytes, so k * n / 2
 *   0x1020   k / 2, and 0x1030's high half the same, where int8 carries twice
 *   surf     follows the ACTIVATION precision, not the weight precision
 *
 * and every one of those is what charsiu already emits. What is missing is only
 * the LAYOUT, which no file contains: a .rkllm gives geometry, not the order the
 * nibbles sit in. The group of 64 in charsiu_weight_ngroup is copied from the
 * RK3588 notes and has never been confirmed on this silicon.
 *
 * THE PROBE, and why it survives everything else being wrong. The question is
 * which two weights share a byte:
 *
 *   K PAIRING   [n/32][k/64][n%32][k%64], the low nibble is an even k
 *   N PAIRING   [n/64][k/32][n%64][k%32], the low nibble is an even n
 *
 * Fill every low nibble live and every high nibble dead and the two answer
 * differently in a way that no scale, requant or activation precision can
 * disguise, because the signal is WHICH CHANNELS ARE NONZERO and not what value
 * they hold:
 *
 *   k pairing   every output channel sees half its taps live, so they are all
 *               nonzero and all roughly equal
 *   n pairing   half the channels see every tap live and the other half see
 *               none, so the output alternates between live and dead
 *
 * ENTRY 1 IS A LIVENESS CONTROL, not a decode. The vendor only ever runs int4
 * weights with 16 bit activations, and charsiu has never run a 16 bit activation
 * at all, so this asks int4 weights with the int8 activations that are known to
 * work. If everything live produces nothing, that combination does not compute
 * and the layout cannot be read this round, which is a clean answer pointing at
 * fp16 activations first rather than a layout read out of noise.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu.h"

enum pattern { PAT_ALL, PAT_LOW, PAT_HIGH, PAT_FIRST, PAT_DEAD };

static const char *pat_name(enum pattern p)
{
	switch (p) {
	case PAT_ALL:   return "every nibble live";
	case PAT_LOW:   return "LOW nibbles live, high dead";
	case PAT_HIGH:  return "HIGH nibbles live, low dead";
	case PAT_FIRST: return "only the first byte live";
	default:        return "every nibble dead";
	}
}

/*
 * The nibble value. A signed nibble runs -8 to 7 and this project has already
 * learned twice that the hardware reads its operands signed, so dead is 0 and
 * live is 7. If int4 turned out to be an unsigned nibble with a zero point of 8,
 * a dead nibble of 0 would read as -8 rather than 0, which would show as every
 * channel being wrong by a constant rather than as a pattern, and the pattern is
 * what this reads.
 */
#define LIVE 0x7
#define DEAD 0x0

static void fill(uint8_t *dst, size_t bytes, enum pattern p)
{
	size_t i;

	for (i = 0; i < bytes; i++) {
		unsigned lo = DEAD, hi = DEAD;

		switch (p) {
		case PAT_ALL:   lo = LIVE; hi = LIVE; break;
		case PAT_LOW:   lo = LIVE; break;
		case PAT_HIGH:  hi = LIVE; break;
		case PAT_FIRST: if (i == 0) lo = LIVE; break;
		default: break;
		}
		dst[i] = (uint8_t)((hi << 4) | lo);
	}
}

/*
 * ROUND 167 ANSWERED THE FIRST HALF. Low nibbles live and high nibbles live gave
 * the IDENTICAL output, a single value across all 64 channels in both, which
 * rules out every N pairing there is: n with n+1 would have made them
 * complementary on even and odd channels, and n with n+32 would have split them
 * at the group boundary. So both nibbles of a byte feed the SAME output channel
 * at different k, and the group of 64 that charsiu carries from the RK3588 notes
 * is not what this silicon does.
 *
 * WHAT IT DID NOT ANSWER is whether a byte holds k and k+1, adjacent, or k and
 * k+32, the two halves of a group. Both put the pair in the same channel, so the
 * low against high test cannot see the difference.
 *
 * --kmap does. Hold the weights at low nibbles live and high dead, and make the
 * INPUT an impulse at one k. Then the output is nonzero exactly when that k
 * happens to live in a low nibble, and sweeping k reads the map directly:
 *
 *   nonzero for even k only          a byte holds k and k+1
 *   nonzero for k % 64 < 32          a byte holds k and k+32
 *   nonzero for every k              the weights are not being read the way
 *                                    this assumes and the pattern above was
 *                                    reading something else
 *
 * ALSO NOT ANSWERED, and worth saying rather than glossing: the absolute values
 * are wrong. Every nibble dead came back 105 where the output stage says 0, and
 * the low nibble delta was 14 where the tap count says about 6. The layout
 * signal is a difference between patterns so it survives that, but int4 is NOT
 * computing correctly yet and the coefficient buffer, the nibble's own zero
 * point convention and 0x1030's one-times branch have all still to be checked.
 */
static void input_impulse(const struct charsiu_matmul *mm, uint8_t *dst,
			  size_t dst_size, unsigned zp, unsigned k0, unsigned amp)
{
	unsigned atom = charsiu_feature_atom(mm->adtype);
	unsigned kk;

	/* everything at the zero point, which is 0 once biased, so only k0
	 * contributes to any output channel */
	memset(dst, (uint8_t)(zp - 0x80), dst_size);
	for (kk = 0; kk < mm->k; kk++) {
		uint8_t v = (uint8_t)((kk == k0 ? zp + amp : zp) - 0x80);

		dst[(kk / atom) * mm->m * atom + kk % atom] = v;
	}
}

int main(int argc, char **argv)
{
	static const enum pattern pats[] = { PAT_DEAD, PAT_ALL, PAT_LOW,
					     PAT_HIGH, PAT_FIRST };
	struct charsiu_job job = { 0 };
	struct charsiu_device *dev;
	struct charsiu_bo regcmd = { 0 }, in = { 0 }, wt = { 0 }, outbo = { 0 },
			   coef = { 0 };
	uint8_t *a_raw;
	int32_t *bias, *wsums;
	uint32_t in_handles[3], out_handles[1];
	unsigned m, n, k, i, p;
	size_t nreg;
	int ret;

	job.mm.m = argc > 1 ? (unsigned)atoi(argv[1]) : 1;
	job.mm.k = argc > 2 ? (unsigned)atoi(argv[2]) : 64;
	job.mm.n = argc > 3 ? (unsigned)atoi(argv[3]) : 64;
	job.mm.wdtype = CHARSIU_INT4;
	job.mm.adtype = CHARSIU_INT8;
	job.input_scale = 0.02f;
	job.weight_scale = 0.01f;
	job.output_scale = 0.25f;
	job.input_zero_point = 128;
	job.weight_zero_point = 0;      /* a signed nibble is its own zero */
	job.output_zero_point = 0;
	m = job.mm.m; k = job.mm.k; n = job.mm.n;

	printf("int4 layout probe M=%u K=%u N=%u, %zu weight bytes\n",
	       m, k, n, charsiu_weight_bytes(&job.mm));

	dev = charsiu_open(NULL);
	if (!dev) { printf("open FAILED\n"); return 1; }

	a_raw = malloc((size_t)m * k);
	bias = calloc(n, sizeof(*bias));
	wsums = calloc(n, sizeof(*wsums));
	/* Every input the same, so a difference between output channels can only
	 * come from the weights. */
	for (i = 0; i < m * k; i++)
		a_raw[i] = (uint8_t)(job.input_zero_point + 32);

	ret = charsiu_bo_alloc(dev, 4096, &regcmd);
	ret |= charsiu_bo_alloc(dev, (size_t)charsiu_entries_per_row(&job.mm) * 64 * m + 4096, &in);
	ret |= charsiu_bo_alloc(dev, charsiu_weight_bytes(&job.mm) + 4096, &wt);
	ret |= charsiu_bo_alloc(dev, (size_t)m * n + 4096, &outbo);
	ret |= charsiu_bo_alloc(dev, charsiu_coef_bytes(&job.mm) + 4096, &coef);
	if (ret) { printf("bo alloc FAILED %d\n", ret); return 1; }

	job.input_addr = (uint32_t)in.dma_address;
	job.weight_addr = (uint32_t)wt.dma_address;
	job.output_addr = (uint32_t)outbo.dma_address;
	job.coef_addr = (uint32_t)coef.dma_address;

	charsiu_bo_prep(dev, &in, 1000000000);
	charsiu_pack_input(&job.mm, a_raw, in.map, in.size,
			   (uint8_t)job.input_zero_point);
	charsiu_bo_fini(dev, &in);
	charsiu_bo_prep(dev, &coef, 1000000000);
	charsiu_build_coefs(&job, bias, wsums, coef.map);
	charsiu_bo_fini(dev, &coef);
	charsiu_bo_prep(dev, &regcmd, 1000000000);
	nreg = charsiu_emit_job(&job, regcmd.map, 4096 / 8);
	charsiu_bo_fini(dev, &regcmd);
	if (!nreg) { printf("emit FAILED\n"); return 1; }
	printf("register stream: %zu entries\n\n", nreg);

	in_handles[0] = in.handle;
	in_handles[1] = wt.handle;
	in_handles[2] = coef.handle;
	out_handles[0] = outbo.handle;

	for (p = 0; p < sizeof(pats) / sizeof(pats[0]); p++) {
		unsigned live = 0, distinct = 0, seen[256] = { 0 };
		unsigned even = 0, odd = 0;
		uint8_t *o;

		charsiu_bo_prep(dev, &wt, 1000000000);
		fill(wt.map, charsiu_weight_bytes(&job.mm), pats[p]);
		charsiu_bo_fini(dev, &wt);

		charsiu_bo_prep(dev, &outbo, 1000000000);
		memset(outbo.map, 0xa5, (size_t)m * n);
		charsiu_bo_fini(dev, &outbo);

		if (charsiu_submit(dev, &regcmd, (unsigned)nreg, in_handles, 3,
				   out_handles, 1)) {
			printf("  %-28s submit FAILED\n", pat_name(pats[p]));
			continue;
		}
		if (charsiu_bo_prep(dev, &outbo, 2000000000)) {
			printf("  %-28s wait FAILED\n", pat_name(pats[p]));
			continue;
		}
		o = outbo.map;

		for (i = 0; i < m * n; i++) {
			if (o[i] != 0)
				live++;
			if (!seen[o[i]]++)
				distinct++;
			if (i & 1) {
				if (o[i]) odd++;
			} else {
				if (o[i]) even++;
			}
		}
		printf("  %-28s  nonzero %3u/%3u  distinct %3u  even %3u odd %3u\n",
		       pat_name(pats[p]), live, m * n, distinct, even, odd);
		printf("      ");
		for (i = 0; i < 32 && i < m * n; i++)
			printf("%4d", o[i] > 127 ? o[i] - 256 : o[i]);
		printf("\n");
		charsiu_bo_fini(dev, &outbo);
	}

	/*
	 * --kmap: weights fixed at low nibbles live, input an impulse walking k.
	 * The baseline is the same weights with no impulse at all, so the column
	 * that matters is the DIFFERENCE and not the value.
	 */
	if (argc > 4 && !strcmp(argv[4], "--kmap")) {
		static const unsigned ks[] = { 0, 1, 2, 3, 15, 16, 30, 31,
					       32, 33, 62, 63 };
		int base = -1;
		unsigned j;

		printf("\n  --kmap: low nibbles live, an input impulse walking k\n");
		charsiu_bo_prep(dev, &wt, 1000000000);
		fill(wt.map, charsiu_weight_bytes(&job.mm), PAT_LOW);
		charsiu_bo_fini(dev, &wt);

		printf("  %-6s %-8s %-8s\n", "k", "out[0]", "delta");
		for (j = 0; j <= sizeof(ks) / sizeof(ks[0]); j++) {
			uint8_t *o;
			int v;

			charsiu_bo_prep(dev, &in, 1000000000);
			/* j == 0 is the baseline: no impulse anywhere */
			input_impulse(&job.mm, in.map, in.size,
				      job.input_zero_point,
				      j ? ks[j - 1] : (unsigned)-1, 32);
			charsiu_bo_fini(dev, &in);

			if (charsiu_submit(dev, &regcmd, (unsigned)nreg,
					   in_handles, 3, out_handles, 1) ||
			    charsiu_bo_prep(dev, &outbo, 2000000000)) {
				printf("  %-6s submit or wait FAILED\n",
				       j ? "k" : "base");
				continue;
			}
			o = outbo.map;
			v = o[0] > 127 ? o[0] - 256 : o[0];
			if (!j) {
				base = v;
				printf("  %-6s %-8d %-8s\n", "base", v, "-");
			} else {
				printf("  %-6u %-8d %-+8d\n", ks[j - 1], v, v - base);
			}
			charsiu_bo_fini(dev, &outbo);
		}
		printf("\n    nonzero delta on EVEN k only      a byte holds k and k+1\n"
		       "    nonzero delta on k %% 64 < 32       a byte holds k and k+32\n"
		       "    nonzero delta on every k          the weights are not being\n"
		       "                                      read the way this assumes\n");
		charsiu_close(dev);
		return 0;
	}

	printf("\n  READ THE FIRST TWO LINES FIRST.\n"
	       "  every nibble dead should be flat, every nibble live should not.\n"
	       "  If those two are the same, int4 weights with int8 activations do\n"
	       "  not compute and no layout can be read from the rest.\n"
	       "\n  Then LOW against HIGH:\n"
	       "    all channels nonzero and roughly equal   nibbles pair along K\n"
	       "    even and odd channels split              nibbles pair along N\n");
	charsiu_close(dev);
	return 0;
}
