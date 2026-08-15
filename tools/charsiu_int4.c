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

/*
 * --map: ONE nibble live in the whole buffer, and read where it lands.
 *
 * ROUNDS 167 AND 168 ARE WITHDRAWN. Their probes filled the WHOLE weight buffer,
 * every low nibble or every high nibble, and a full fill cannot see a
 * permutation: any layout that is a bijection answers "all channels equal" to
 * it. The --kmap probe had the same defect, testing which half of k sits in the
 * low nibble while still filling everything. Both were invariant to the thing
 * they were built to measure, so their agreement proved nothing, and round 170b
 * showed it: with a scale that actually discriminates, the int4 impulse lights
 * 14 of 64 channels at strided positions where the reference lights nearly all
 * of them.
 *
 * A layout can only be read with a SPARSE probe. That is how the int8 one was
 * read, with position encoded models, and it is what this does.
 *
 * The scales are chosen so the answer is legible without arithmetic. One live
 * nibble of 7, an input whose value at k is k + 1 above the zero point, and a
 * requant multiplier of 1/7, so
 *
 *     the output CHANNEL that lights is n, and its VALUE is k + 1
 *
 * for one probe at a time. Nothing has to be inferred from a pattern.
 */
static void map_probe(struct charsiu_device *dev, struct charsiu_job *job,
		      struct charsiu_bo *wt, struct charsiu_bo *outbo,
		      struct charsiu_bo *regcmd, size_t nreg,
		      const uint32_t *in_h, const uint32_t *out_h,
		      size_t byte, unsigned high)
{
	unsigned i, n = job->mm.n, lit = 0, first_n = 0, first_v = 0;
	uint8_t *o;

	charsiu_bo_prep(dev, wt, 1000000000);
	memset(wt->map, 0, charsiu_weight_bytes(&job->mm));
	/* an int8 weight buffer has no nibbles: the whole byte is the weight,
	 * and the high pass is skipped by the caller */
	((uint8_t *)wt->map)[byte] = job->mm.wdtype == CHARSIU_INT4
		? (uint8_t)(high ? (LIVE << 4) : LIVE)
		: (uint8_t)LIVE;
	charsiu_bo_fini(dev, wt);

	charsiu_bo_prep(dev, outbo, 1000000000);
	memset(outbo->map, 0, (size_t)job->mm.m * n);
	charsiu_bo_fini(dev, outbo);

	if (charsiu_submit(dev, regcmd, (unsigned)nreg, in_h, 3, out_h, 1) ||
	    charsiu_bo_prep(dev, outbo, 2000000000)) {
		printf("  byte %-6zu %-5s submit or wait FAILED\n",
		       byte, high ? "high" : "low");
		return;
	}
	o = outbo->map;
	for (i = 0; i < n; i++) {
		int v = o[i] > 127 ? o[i] - 256 : o[i];

		if (v) {
			if (!lit) { first_n = i; first_v = (unsigned)(v < 0 ? -v : v); }
			lit++;
		}
	}
	if (lit == 1)
		printf("  byte %-6zu %-5s -> n = %-4u v = %-4u\n",
		       byte, high ? "high" : "low", first_n, first_v);
	else if (!lit)
		printf("  byte %-6zu %-5s -> NOTHING LIT\n", byte,
		       high ? "high" : "low");
	else
		printf("  byte %-6zu %-5s -> %u channels lit, first n = %u v = %u\n",
		       byte, high ? "high" : "low", lit, first_n, first_v);
	charsiu_bo_fini(dev, outbo);
}

/*
 * --kpair: sparse on BOTH sides, which is the one thing never tried.
 *
 * Every probe so far has been sparse in one operand and dense in the other. The
 * map fixes one live nibble and feeds a full input, so its output is a SUM over
 * whatever k that nibble touches and its value cannot name a single k. Round
 * 177's impulse has one live nibble per channel and a full input, and came back
 * saturated at the rails, which is what cross talk between channels would do.
 *
 * Both of those are explained by one unmeasured thing: a nibble reaching more
 * than one input. So measure it. Hold ONE live nibble in the whole buffer, make
 * the input ONE HOT at a single k, and sweep k. The output is nonzero exactly
 * when that nibble is paired with that k, so the set of k a nibble touches is
 * read off directly with no arithmetic and no summation to unpick.
 *
 *   exactly one k lights per nibble    the pairing is one to one and the layout
 *                                      is a permutation after all
 *   several k light                    a nibble is broadcast across inputs, and
 *                                      how many says by how much
 *   no k lights                        the nibble is not fetched at that offset
 */
static void kpair_probe(struct charsiu_device *dev, struct charsiu_job *job,
			struct charsiu_bo *wt, struct charsiu_bo *in,
			struct charsiu_bo *outbo, struct charsiu_bo *regcmd,
			size_t nreg, const uint32_t *in_h, const uint32_t *out_h,
			size_t byte, unsigned high, uint8_t *a_raw)
{
	unsigned k, i, hits = 0;
	unsigned firstn = 0;

	charsiu_bo_prep(dev, wt, 1000000000);
	memset(wt->map, 0, charsiu_weight_bytes(&job->mm));
	((uint8_t *)wt->map)[byte] = (uint8_t)(high ? (LIVE << 4) : LIVE);
	charsiu_bo_fini(dev, wt);

	printf("  byte %-5zu %-5s pairs with k =", byte, high ? "high" : "low");
	for (k = 0; k < job->mm.k; k++) {
		uint8_t *o;
		int any = 0;

		for (i = 0; i < job->mm.k; i++)
			a_raw[i] = (uint8_t)(job->input_zero_point + (i == k ? 100 : 0));
		charsiu_bo_prep(dev, in, 1000000000);
		charsiu_pack_input(&job->mm, a_raw, in->map, in->size,
				   (uint8_t)job->input_zero_point);
		charsiu_bo_fini(dev, in);

		charsiu_bo_prep(dev, outbo, 1000000000);
		memset(outbo->map, 0, (size_t)job->mm.m * job->mm.n);
		charsiu_bo_fini(dev, outbo);

		if (charsiu_submit(dev, regcmd, (unsigned)nreg, in_h, 3, out_h, 1) ||
		    charsiu_bo_prep(dev, outbo, 2000000000))
			continue;
		o = outbo->map;
		for (i = 0; i < job->mm.n; i++)
			if (o[i]) { any = 1; if (!hits) firstn = i; break; }
		charsiu_bo_fini(dev, outbo);
		if (any) {
			printf(" %u", k);
			hits++;
		}
	}
	printf("   (%u of %u", hits, job->mm.k);
	if (hits)
		printf(", first channel %u", firstn);
	printf(")\n");
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
	/*
	 * CHARSIU_MAP_W8 maps an INT8 weight buffer with the same probe.
	 *
	 * int8 is byte exact on this hardware, so its map is what a correct one
	 * looks like, and diffing the two says what int4 does differently
	 * instead of leaving it to be inferred. Round 171 and round 173 both
	 * went wrong by fitting a rule to part of a map; this is the method that
	 * has worked every time here, which is to diff against something known
	 * to compute.
	 *
	 * The probe writes raw bytes into the weight buffer either way. For int8
	 * a live byte is 0x07 rather than a nibble, which is a legal small
	 * positive weight in the biased domain the packer would have produced.
	 */
	job.mm.wdtype = getenv("CHARSIU_MAP_W8") ? CHARSIU_INT8 : CHARSIU_INT4;
	job.mm.adtype = CHARSIU_INT8;
	job.input_scale = 0.02f;
	/*
	 * 1.7857 makes the requant multiplier 1/7, so one live nibble of 7
	 * against an input of (k + 1) above the zero point puts exactly k + 1 on
	 * the output. The pattern probes below do not care; --map does.
	 */
	job.weight_scale = 1.7857f;
	job.output_scale = 0.25f;
	job.input_zero_point = 128;
	/*
	 * THE WEIGHT ZERO POINT IS PER PRECISION, and getting it wrong wasted
	 * round 174's whole comparison.
	 *
	 * A nibble is not biased, so int4's is 0 and its B record comes out 0.
	 * An int8 weight IS biased by 0x80, so its zero point is 0x80 and its B
	 * record is likewise 0. Setting 0 for both, which is what this line did,
	 * gives int8 a B of 0x80, and 0x80 * 2080 * mult saturates every channel
	 * at 127 whatever the weight buffer holds. Round 174's int8 map lit all
	 * 64 channels at every offset INCLUDING the all zero one, which is that
	 * and not the hardware.
	 */
	job.weight_zero_point = job.mm.wdtype == CHARSIU_INT4 ? 0 : 0x80;
	job.output_zero_point = 0;
	m = job.mm.m; k = job.mm.k; n = job.mm.n;

	/* the configuration goes in the log, because round 174's whole
	 * comparison was lost to a weight zero point that was right for a nibble
	 * and wrong for a byte, and nothing printed said which was in use */
	printf("layout probe M=%u K=%u N=%u  w%s  wt_zp %u  %zu weight bytes\n",
	       m, k, n, job.mm.wdtype == CHARSIU_INT4 ? "4" : "8",
	       job.weight_zero_point, charsiu_weight_bytes(&job.mm));

	dev = charsiu_open(NULL);
	if (!dev) { printf("open FAILED\n"); return 1; }

	a_raw = malloc((size_t)m * k);
	bias = calloc(n, sizeof(*bias));
	wsums = calloc(n, sizeof(*wsums));
	/* Every input the same, so a difference between output channels can only
	 * come from the weights. */
	/* value k + 1 above the zero point at position k, so the output of a
	 * single live tap names its own k */
	for (i = 0; i < m * k; i++)
		a_raw[i] = (uint8_t)(job.input_zero_point + 1 + (i % k));

	ret = charsiu_bo_alloc(dev, 4096, &regcmd);
	ret |= charsiu_bo_alloc(dev, (size_t)charsiu_entries_per_row(&job.mm) * 64 * m + 4096, &in);
	ret |= charsiu_bo_alloc(dev, charsiu_weight_bytes(&job.mm) + 4096, &wt);
	/* w4a16 writes a FLOAT out, two bytes an element, so the output buffer
	 * is twice what an int8 job needs */
	ret |= charsiu_bo_alloc(dev, (size_t)m * n * 2 + 4096, &outbo);
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
	/*
	 * --map: the sparse read. The scales here are not the caller's; they are
	 * chosen so that one live nibble makes the output value k + 1 on channel
	 * n, so a whole layout can be read off a console log with no arithmetic.
	 */
	/*
	 * --kpair: one live nibble, a one hot input, sweep k. Reads which inputs
	 * a nibble is paired with, which every other probe here has had to infer
	 * from a sum.
	 */
	/*
	 * --f16: the direct test of round 179's conclusion. int4 weights consume
	 * the activation as fp16, so feed a real one.
	 *
	 * One live nibble per channel at k = c mod 16, the same impulse the
	 * integer path runs, but the input is packed as actual halves. The
	 * reference is then a float multiply, which is what the reference for an
	 * fp16 activation has to be: acc = weight * activation, and the requant
	 * turns it into a byte.
	 *
	 *   the output tracks the reference
	 *             int4 computes, and w4a16 is the configuration it needs.
	 *   the output is still at the rails
	 *             the element being 16 bits is measured and its being a
	 *             FLOAT is not, so that is the part that comes down.
	 */
	if (argc > 4 && !strcmp(argv[4], "--f16")) {
		float *af = malloc((size_t)m * k * sizeof(*af));
		unsigned c, j;
		uint8_t *o;

		/* an input a half can hold exactly, so the packing adds no
		 * error of its own: small integers */
		for (i = 0; i < m * k; i++)
			af[i] = (float)((int)(i % 32) - 16);
		charsiu_bo_prep(dev, &in, 1000000000);
		charsiu_pack_input_f16(&job.mm, af, in.map, in.size);
		charsiu_bo_fini(dev, &in);

		/* one live nibble per channel, as the impulse */
		charsiu_bo_prep(dev, &wt, 1000000000);
		memset(wt.map, 0, charsiu_weight_bytes(&job.mm));
		for (c = 0; c < n; c++) {
			size_t row = (size_t)(c / 32) * 512 + (size_t)(c % 32) * 8;
			unsigned kk = c % 16;

			((uint8_t *)wt.map)[row + kk / 2] |=
				(uint8_t)((kk & 1) ? (LIVE << 4) : LIVE);
		}
		charsiu_bo_fini(dev, &wt);

		charsiu_bo_prep(dev, &outbo, 1000000000);
		memset(outbo.map, 0xa5, (size_t)m * n * 2);
		charsiu_bo_fini(dev, &outbo);
		if (charsiu_submit(dev, &regcmd, (unsigned)nreg, in_handles, 3,
				   out_handles, 1) ||
		    charsiu_bo_prep(dev, &outbo, 2000000000)) {
			printf("  f16 submit or wait FAILED\n");
			charsiu_close(dev);
			return 1;
		}
		o = outbo.map;
		printf("\n  --f16: w4a16. A real fp16 activation, one live nibble of\n"
		       "  %d per channel at k = c mod 16, and the output read as HALVES\n"
		       "  because the vendor's own configuration requantises with\n"
		       "  0x40b0 = 1 and 0x40b4 = 0, which is identity.\n\n", LIVE);
		printf("  %-4s %-12s %-12s %-12s %-10s\n",
		       "c", "act[c%%16]", "expected", "npu (half)", "raw");
		for (c = 0; c < 16 && c < n; c++) {
			uint16_t h = (uint16_t)(o[c * 2] | (o[c * 2 + 1] << 8));
			float got = charsiu_half_to_float(h);

			printf("  %-4u %-12.3f %-12.3f %-12.3f 0x%04x\n", c,
			       af[c % 16], af[c % 16] * (float)LIVE, got, h);
		}
		j = 0;
		for (c = 0; c < m * n * 2; c++)
			if (((uint8_t *)outbo.map)[c] == 0xa5) j++;
		printf("\n  %u of %u bytes still hold the sentinel\n", j, m * n * 2);
		/*
		 * THE FIT GOES IN THE LOG, over the channels that came back
		 * finite and varying. Round 181 was read by eyeballing four
		 * numbers off a table and computing the slope by hand, which is
		 * how a sweep stride got read as a row size two rounds earlier.
		 * The expected slope is LIVE and the expected intercept is 0.
		 */
		{
			double sx = 0, sy = 0, sxx = 0, sxy = 0, nn = 0;

			for (c = 0; c < 16 && c < n; c++) {
				uint16_t h = (uint16_t)(o[c * 2] | (o[c * 2 + 1] << 8));
				float got = charsiu_half_to_float(h);

				if (!(got > -1e30f && got < 1e30f) || got == 0.0f)
					continue;       /* nan, inf and the dead ones */
				sx += af[c % 16]; sy += got;
				sxx += (double)af[c % 16] * af[c % 16];
				sxy += (double)af[c % 16] * got;
				nn++;
			}
			if (nn >= 2) {
				double den = nn * sxx - sx * sx;
				double slope = den ? (nn * sxy - sx * sy) / den : 0;

				printf("  fit over %g finite varying channels:"
				       " npu = %.3f * act + %.1f   (want %d * act + 0)\n",
				       nn, slope, (sy - slope * sx) / nn, LIVE);
			} else {
				printf("  fewer than two finite varying channels: no fit\n");
			}
		}
		charsiu_close(dev);
		return 0;
	}

	if (argc > 4 && !strcmp(argv[4], "--kpair")) {
		static const size_t bytes[] = { 0, 1, 2, 3, 4, 7, 8, 512 };
		unsigned j;

		printf("\n  --kpair: ONE live nibble, a ONE HOT input, sweeping k.\n"
		       "  A nibble that pairs with more than one k is broadcast, and\n"
		       "  that would explain both the map's unusable k column and the\n"
		       "  impulse saturating at the rails.\n\n");
		for (j = 0; j < sizeof(bytes) / sizeof(bytes[0]); j++) {
			if (bytes[j] >= charsiu_weight_bytes(&job.mm))
				continue;
			kpair_probe(dev, &job, &wt, &in, &outbo, &regcmd, nreg,
				    in_handles, out_handles, bytes[j], 0, a_raw);
			kpair_probe(dev, &job, &wt, &in, &outbo, &regcmd, nreg,
				    in_handles, out_handles, bytes[j], 1, a_raw);
		}
		charsiu_close(dev);
		return 0;
	}

	if (argc > 4 && !strcmp(argv[4], "--map")) {
		/*
		 * SWEEP, do not sample. Round 171 probed eight byte offsets and
		 * the map inferred from them did not survive its own data: byte
		 * 512 lit channel 32 where n = byte / 8 predicts 64, and every
		 * offset from 1023 up lit nothing at all while channel 32 was
		 * reachable from below. Eight points is not a map, it is an
		 * invitation to theorise, and that is what happened.
		 *
		 * The whole buffer at a stride instead. The default of 8 gives
		 * 256 probes across a 64 by 64 int4 weight buffer, which is one
		 * boot, and the result is data rather than inference.
		 */
		size_t stride = argc > 5 ? (size_t)atoi(argv[5]) : 8;
		size_t wb = charsiu_weight_bytes(&job.mm), b;

		printf("\n  --map: ONE live nibble at a time, every %zu bytes of %zu.\n"
		       "  n is the channel that lights; the value is printed raw\n"
		       "  because round 171's k calibration was wrong and a nibble\n"
		       "  reaches more than one input.\n\n", stride, wb);
		for (b = 0; b < wb; b += stride) {
			map_probe(dev, &job, &wt, &outbo, &regcmd, nreg,
				  in_handles, out_handles, b, 0);
			if (job.mm.wdtype == CHARSIU_INT4)
				map_probe(dev, &job, &wt, &outbo, &regcmd, nreg,
					  in_handles, out_handles, b, 1);
		}
		charsiu_close(dev);
		return 0;
	}

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
