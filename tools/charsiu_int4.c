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
#include <math.h>

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

/*
 * The live nibble is a RUNTIME value on the --f16 path, and a compile time one
 * everywhere else. Every w4a16 round so far ran at a constant 7, so no run has
 * ever shown that the output depends on the weights; CHARSIU_INT4_LIVE is the
 * knob that asks. The other probes are left alone, because one variable.
 */
static unsigned g_live = LIVE;

/*
 * A least squares line and the RMS of what it leaves behind. The residual is
 * the whole point of round 184: two candidate explanatory variables get fitted
 * against the same outputs, and the one that does not explain them shows up
 * here rather than in the slope.
 */
static void fit_line(const char *what, const double *x, const double *y,
		     unsigned nn, double want)
{
	double sx = 0, sy = 0, sxx = 0, sxy = 0, den, slope, icept, res = 0;
	unsigned i;

	if (nn < 2) {
		printf("  fit vs %-18s fewer than two points, no fit\n", what);
		return;
	}
	for (i = 0; i < nn; i++) {
		sx += x[i]; sy += y[i];
		sxx += x[i] * x[i]; sxy += x[i] * y[i];
	}
	den = nn * sxx - sx * sx;
	slope = den ? (nn * sxy - sx * sy) / den : 0;
	icept = (sy - slope * sx) / nn;
	for (i = 0; i < nn; i++) {
		double e = y[i] - (slope * x[i] + icept);

		res += e * e;
	}
	res = sqrt(res / nn);
	printf("  fit vs %-18s npu = %10.3f * x + %10.1f   rms residual %10.3f",
	       what, slope, icept, res);
	if (want != 0)
		printf("   (a real MAC wants a slope of %g)", want);
	printf("\n");
}

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
	/*
	 * ⚠ ROUND 267. This had the same defect --kpair did, and it is the
	 * THIRD place in this file to carry it: it cleared and read m*n BYTES
	 * on the w4a16 path, whose element is a four byte signed integer.
	 * --bmap, --osweep and --pre all read m*n*4. It is correct as written
	 * for int8, where a channel really is a byte, which is how it passed
	 * its int8 validation and kept the bug.
	 *
	 * Read the whole bo against a 0xa5 sentinel, so a word the hardware
	 * wrote 0 into is distinguishable from one it never touched. The count
	 * of written words is the point of round 267: at N of 16, 32 and 64
	 * --kpair measured 16, 24 and 40, which is N/2 + 8, so the hardware
	 * writes FEWER channels than the job declares and at N = 16 that
	 * happens to equal N, which is why every clean result so far is at 16.
	 */
	unsigned i, lit = 0, first_w = 0, wrote = 0, hi = 0, hib = 0;
	int32_t first_v = 0;
	size_t obytes = outbo->size;
	uint8_t *o;

	charsiu_bo_prep(dev, wt, 1000000000);
	memset(wt->map, 0, charsiu_weight_bytes(&job->mm));
	/* an int8 weight buffer has no nibbles: the whole byte is the weight,
	 * and the high pass is skipped by the caller */
	((uint8_t *)wt->map)[byte] = job->mm.wdtype == CHARSIU_INT4
		? (uint8_t)(high ? (g_live << 4) : g_live)
		: (uint8_t)g_live;
	charsiu_bo_fini(dev, wt);

	charsiu_bo_prep(dev, outbo, 1000000000);
	memset(outbo->map, 0xa5, obytes);
	charsiu_bo_fini(dev, outbo);

	if (charsiu_submit(dev, regcmd, (unsigned)nreg, in_h, 3, out_h, 1) ||
	    charsiu_bo_prep(dev, outbo, 2000000000)) {
		printf("  byte %-6zu %-5s submit or wait FAILED\n",
		       byte, high ? "high" : "low");
		return;
	}
	o = outbo->map;
	for (i = 0; (size_t)i + 4 <= obytes; i += 4) {
		uint32_t u;
		int32_t v;
		unsigned b;

		memcpy(&u, o + i, 4);
		if (u == 0xa5a5a5a5u)
			continue;
		wrote++;
		hi = i / 4;
		/*
		 * The highest written BYTE, not just the word. int8 writes one
		 * byte a channel, so a 64 channel int8 job touches 16 words and
		 * 64 bytes; reading only the word count cannot tell that from a
		 * 16 channel job. The int8 arm is the oracle here and it has to
		 * be readable in its own units.
		 */
		for (b = 0; b < 4; b++)
			if (o[i + b] != 0xa5)
				hib = i + b;
		if (!u)
			continue;
		memcpy(&v, &u, 4);
		if (!lit) { first_w = i / 4; first_v = v; }
		lit++;
	}
	if (lit == 1)
		printf("  byte %-6zu %-5s -> w %-4u v %-8d   wrote %u hi %u hib %u\n",
		       byte, high ? "high" : "low", first_w, first_v, wrote, hi, hib);
	else if (!lit)
		printf("  byte %-6zu %-5s -> NOTHING LIT            wrote %u hi %u hib %u\n",
		       byte, high ? "high" : "low", wrote, hi, hib);
	else
		printf("  byte %-6zu %-5s -> %u words lit, first w %u v %d   wrote %u hi %u hib %u\n",
		       byte, high ? "high" : "low", lit, first_w, first_v, wrote, hi, hib);
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
	/*
	 * ⚠ ROUND 265 WIDENED THE READ, and every "channel" this probe printed
	 * before it is an OUTPUT BYTE OFFSET.
	 *
	 * It cleared and read m*n BYTES on a path whose output element is not a
	 * byte. 0x4010 carries a non zero OUT_PRECISION for w4a16, and 0x40ac,
	 * 0x40b0, 0x40b4 are 0, 1, 0, an identity requant, so nothing narrows
	 * the accumulator to a byte. Every other w4a16 probe in this file reads
	 * m*n*4 and only this one read m*n, which is the int8 era reading that
	 * round 189 already removed once from --bmap. The buffer is allocated
	 * m*n*4 under a comment that says two bytes an element, so the file has
	 * carried three different widths at once.
	 *
	 * Two of round 264's three anomalies fall straight out of that. "Two
	 * consecutive channels, and only c mod 4 in {0,1}" is the two low bytes
	 * of one little endian word: a single live nibble of 7 measured +5112,
	 * which is 000013f8, nonzero at offsets 0 and 1 and zero at 2 and 3.
	 * "Everything at b >= 64 is dead" is an element that fell outside a
	 * window m*n bytes wide, and at N of 16, 32 and 64 that window is
	 * exactly where r264's dead rows begin.
	 *
	 * So report the word, its index, and WHICH bytes of it are live. The
	 * word index is the channel candidate. How many bytes light says how
	 * wide an element is. And re running with CHARSIU_INT4_LIVE=15 says
	 * whether the value is a signed 32 bit integer at all, because -4896
	 * lights four bytes where +5112 lights two. The live nibble is g_live
	 * here for that reason; it used to be the LIVE macro, which is why the
	 * environment variable did nothing for --kpair.
	 *
	 * ⚠ ROUND 266 WIDENED IT AGAIN, because 265 left the same hole one
	 * level up. It read m*n*4, which is words 0 to n-1, and the buffer is
	 * allocated m*n*4 + 4096. A byte whose element landed at word n or
	 * beyond would have come back "dead" for the second round running, and
	 * eleven bytes did come back dead. Read the WHOLE bo.
	 *
	 * And fill with 0xa5 rather than 0, which is what every other probe in
	 * this file already does. Zero fill cannot tell a word the hardware
	 * wrote 0 into from one it never touched, so 265 could not say how far
	 * the write actually extends. That extent bounds the channel count and
	 * the element width on its own, without any of the address arithmetic.
	 *
	 * 265 settled the element: +7100 came back 00001bbc lighting two bytes
	 * and -6800 came back ffffe570 lighting four, with both arms agreeing
	 * on every word and every k across 48 points. Four byte signed little
	 * endian, and the word index is the channel. The negative arm has done
	 * its job and is not repeated here.
	 */
	unsigned k, i, hits = 0;
	unsigned firstw = 0, maxwrote = 0, maxhi = 0;
	size_t obytes = outbo->size;

	charsiu_bo_prep(dev, wt, 1000000000);
	memset(wt->map, 0, charsiu_weight_bytes(&job->mm));
	((uint8_t *)wt->map)[byte] = (uint8_t)(high ? (g_live << 4) : g_live);
	charsiu_bo_fini(dev, wt);

	printf("  byte %-5zu %-5s pairs with k =", byte, high ? "high" : "low");
	for (k = 0; k < job->mm.k; k++) {
		uint8_t *o;
		int any = 0;
		unsigned nw = 0, wrote = 0, hi = 0;

		for (i = 0; i < job->mm.k; i++)
			a_raw[i] = (uint8_t)(job->input_zero_point + (i == k ? 100 : 0));
		charsiu_bo_prep(dev, in, 1000000000);
		charsiu_pack_input(&job->mm, a_raw, in->map, in->size,
				   (uint8_t)job->input_zero_point);
		charsiu_bo_fini(dev, in);

		charsiu_bo_prep(dev, outbo, 1000000000);
		memset(outbo->map, 0xa5, obytes);
		charsiu_bo_fini(dev, outbo);

		if (charsiu_submit(dev, regcmd, (unsigned)nreg, in_h, 3, out_h, 1) ||
		    charsiu_bo_prep(dev, outbo, 2000000000))
			continue;

		o = outbo->map;
		wrote = 0;
		hi = 0;
		for (i = 0; (size_t)i + 4 <= obytes; i += 4) {
			uint32_t w;
			unsigned b, mask = 0;

			memcpy(&w, o + i, 4);
			if (w == 0xa5a5a5a5u)
				continue;
			wrote++;
			hi = i / 4;
			if (!w)
				continue;
			for (b = 0; b < 4; b++)
				if (o[i + b])
					mask |= 1u << b;
			if (!any) {
				any = 1;
				if (!hits)
					firstw = i / 4;
				printf(" %u[", k);
			}
			if (nw < 4)
				printf("%sw%u=%08x/b%u%u%u%u", nw ? " " : "",
				       i / 4, w,
				       !!(mask & 1), !!(mask & 2),
				       !!(mask & 4), !!(mask & 8));
			nw++;
		}
		charsiu_bo_fini(dev, outbo);
		if (wrote > maxwrote)
			maxwrote = wrote;
		if (hi > maxhi)
			maxhi = hi;
		if (any) {
			if (nw > 4)
				printf(" ...%u words", nw);
			printf("]");
			hits++;
		}
	}
	printf("   (%u of %u", hits, job->mm.k);
	if (hits)
		printf(", first word %u", firstw);
	printf(", hw wrote %u words, highest %u)\n", maxwrote, maxhi);
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
	if (getenv("CHARSIU_INT4_LIVE"))
		g_live = (unsigned)atoi(getenv("CHARSIU_INT4_LIVE")) & 0xf;
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
	printf("layout probe M=%u K=%u N=%u  w%s  wt_zp %u  %zu weight bytes"
	       "  live nibble %u  activation %s\n",
	       m, k, n, job.mm.wdtype == CHARSIU_INT4 ? "4" : "8",
	       job.weight_zero_point, charsiu_weight_bytes(&job.mm), g_live,
	       getenv("CHARSIU_INT4_PERM") ? "PERMUTED" : "plain ramp");

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
	ret |= charsiu_bo_alloc(dev, (size_t)m * n * 4 + 4096, &outbo);
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

	/*
	 * THE PATTERN PROBES SUBMIT, and that is why round 187's --wedge could
	 * not answer its own question. --wedge asked how many w4a16 jobs it
	 * takes before the NPU cannot start an int8 one, and its N = 0 row had
	 * FIVE behind it, because this loop runs before every sub command and
	 * the weight dtype is int4. Its control caught it. Any sub command now
	 * gets a clean device.
	 */
	if (argc > 4)
		printf("  pattern probes SKIPPED for a sub command: no job has run\n"
		       "  on this device yet\n\n");
	for (p = 0; argc <= 4 && p < sizeof(pats) / sizeof(pats[0]); p++) {
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
	/*
	 * --osweep: read the OUTPUT format the same way the weight layout was
	 * read, by making the input sparse and looking at where the answer
	 * lands. Round 184 spent four runs fitting lines against an output whose
	 * element width was assumed and wrong.
	 *
	 * WHAT ROUND 184 ACTUALLY SHOWED. Read as 16 halves, the outputs fell
	 * into a period of four: index %4==0 moved with the activation, %4==1
	 * was always 0xffff, %4==2 moved with the nibble but not the activation,
	 * and %4==3 was always 0x0000. Pair them up little endian and that is
	 * eight 32 bit integers, alternating sign, with 0xffff and 0x0000 as the
	 * high halves of negative and positive values:
	 *
	 *   [94 f1][ff ff] -> 0xfffff194 = -3692
	 *   [a4 10][00 00] -> 0x000010a4 = +4260
	 *
	 * That also corrects round 181. The registers 0x40b0 = 1 and 0x40b4 = 0
	 * say the stage does not REQUANTISE. "So the output is a float" was
	 * added on top of that and never measured. An un-requantised output is
	 * the raw accumulator, which is an integer.
	 *
	 * SO STOP ASSUMING. One live nibble in the entire weight buffer, in one
	 * channel, and dump the output bytes. Sweep the channel.
	 *
	 *   one run of bytes moves, and its start tracks the channel
	 *             the run length is the element width and the difference
	 *             between starts is the channel stride, both read rather
	 *             than fitted.
	 *   the same bytes move whatever the channel is
	 *             the output is not written per channel and every per
	 *             channel reading in this file is wrong.
	 *   nothing is non zero for any channel
	 *             one nibble against this activation is below whatever the
	 *             stage keeps, and the probe needs a bigger one.
	 *
	 * THE CONTROL THAT CAN FAIL is the first entry, with NO live nibble
	 * anywhere. Round 184 measured that as all zero, so anything non zero
	 * here means the output does not come from the weight buffer and the
	 * whole sweep is unreadable.
	 *
	 * The activation is a single constant, so nothing in the output can
	 * track an activation index; the only thing left varying is the channel.
	 */
	/*
	 * --pre N: submit N w4a16 jobs and exit, so that the int8 job which
	 * follows is charsiu_matmul's OWN, run from the shell as its own
	 * process.
	 *
	 * WHY NOT --wedge, WHICH THIS REPLACES. --wedge built an int8 job here
	 * by hand and ran it in the same process, and its N = 0 control failed
	 * in round 187 and again in round 188 after the pattern probes were
	 * taken out of the way. Zero w4a16 jobs and the hand built int8 job
	 * still came back with its output holding the sentinel, so the thing
	 * being measured with was broken and the table it produced could not be
	 * read. Two rounds of that is enough: the int8 side goes back to the
	 * code path that has been byte exact since round 164 and has not been
	 * touched since.
	 *
	 * The cost is that this is two processes again. That is a real cost,
	 * because round 186 lost its bisection to exactly that. It is paid
	 * differently here: nothing is asked to RECOVER between the two, the
	 * counts run from small to large, and the first row is N = 0, which is
	 * charsiu_matmul on its own and is already known to pass from the
	 * entry above it in the same log.
	 */
	if (argc > 4 && !strcmp(argv[4], "--pre")) {
		unsigned N = argc > 5 ? (unsigned)atoi(argv[5]) : 1;
		size_t obytes = (size_t)m * n * 4;
		unsigned q, wrote = 0, failed = 0, empty = 0;
		float *af = malloc((size_t)m * k * sizeof(*af));

		for (i = 0; i < m * k; i++)
			af[i] = 8.0f;
		charsiu_bo_prep(dev, &in, 1000000000);
		charsiu_pack_input_f16(&job.mm, af, in.map, in.size);
		charsiu_bo_fini(dev, &in);
		charsiu_bo_prep(dev, &wt, 1000000000);
		memset(wt.map, 0, charsiu_weight_bytes(&job.mm));
		((uint8_t *)wt.map)[0] = (uint8_t)g_live;
		charsiu_bo_fini(dev, &wt);

		for (q = 0; q < N; q++) {
			uint8_t *o;
			unsigned touched = 0;

			charsiu_bo_prep(dev, &outbo, 1000000000);
			memset(outbo.map, 0xa5, obytes);
			charsiu_bo_fini(dev, &outbo);
			if (charsiu_submit(dev, &regcmd, (unsigned)nreg, in_handles,
					   3, out_handles, 1) ||
			    charsiu_bo_prep(dev, &outbo, 2000000000)) {
				failed++;
				continue;
			}
			o = outbo.map;
			for (i = 0; i < obytes; i++)
				if (o[i] != 0xa5)
					touched++;
			if (touched)
				wrote++;
			else
				empty++;
			charsiu_bo_fini(dev, &outbo);
		}
		printf("\n  --pre %u: %u w4a16 jobs wrote, %u completed without\n"
		       "  writing, %u failed outright. The int8 job that judges this\n"
		       "  is the next process, not this one.\n", N, wrote, empty, failed);
		charsiu_close(dev);
		return 0;
	}

	/*
	 * --bmap: which output slot each WEIGHT BYTE feeds, on the w4a16 path.
	 *
	 * WHY THIS AND NOT --osweep AGAIN. --osweep placed its live nibble at
	 * `row = (n/32)*512 + (n%32)*8`, and that formula was read off --map in
	 * rounds 176 to 179, when the output still went through the int8 requant
	 * and was read as bytes. The w4a16 path writes 4 byte integers into a
	 * 160 byte region, so the two paths are not known to agree, and the
	 * three dead bands --osweep found (channels 8-15, 24-31, 40-47) are
	 * exactly what a wrong row formula would look like. So drop the formula:
	 * walk the weight buffer a byte at a time and let the hardware say.
	 *
	 * One live nibble in one byte, everything else zero, activation a single
	 * constant, and the output slot recorded. Then a summary per slot: how
	 * many bytes feed it and which range they came from.
	 *
	 *   each slot fed by a contiguous run of bytes
	 *             the run length is the row size and the gap between runs is
	 *             the row stride, both read rather than fitted.
	 *   bytes scattered across the buffer feeding one slot
	 *             the layout is interleaved, and the summary's min and max
	 *             per slot say over what span.
	 *   a large part of the buffer feeding nothing
	 *             the fetch is shorter than the buffer, and how much shorter
	 *             is the number that comes out of this.
	 *
	 * THE CONTROL is the same as everywhere else: the all zero buffer, run
	 * first, which must light nothing.
	 */
	if (argc > 4 && !strcmp(argv[4], "--bmap")) {
		size_t wb = charsiu_weight_bytes(&job.mm), byte;
		size_t obytes = (size_t)m * n * 4;
		unsigned step = argc > 5 ? (unsigned)atoi(argv[5]) : 1;
		unsigned slot_cnt[512], slot_lo[512], slot_hi[512], slot_nz[512];
		int32_t slot_val[512];
		unsigned nzhist[17];
		unsigned lit = 0, dark = 0, printed = 0, s;
		float act = 8.0f;
		float *af = malloc((size_t)m * k * sizeof(*af));

		if (!step)
			step = 1;
		/*
		 * THE ACTIVATION IS A KNOB NOW, because out(w) and out(a) are
		 * the two curves that say what the output actually is, and this
		 * probe has only ever run at one point of each.
		 */
		if (getenv("CHARSIU_BMAP_ACT"))
			act = (float)atof(getenv("CHARSIU_BMAP_ACT"));
		for (s = 0; s < 512; s++) {
			slot_cnt[s] = 0; slot_lo[s] = ~0u; slot_hi[s] = 0;
			slot_val[s] = 0; slot_nz[s] = 0;
		}
		for (s = 0; s < 17; s++)
			nzhist[s] = 0;
		for (i = 0; i < m * k; i++)
			af[i] = act;
		charsiu_bo_prep(dev, &in, 1000000000);
		charsiu_pack_input_f16(&job.mm, af, in.map, in.size);
		charsiu_bo_fini(dev, &in);

		printf("\n  --bmap: ONE live nibble of %u (signed %d) in ONE byte,\n"
		       "  walking %zu weight bytes in steps of %u, activation a\n"
		       "  constant %.3f. The first row is the all zero control and\n"
		       "  must light nothing.\n\n",
		       g_live, g_live > 7 ? (int)g_live - 16 : (int)g_live, wb,
		       step, act);

		for (byte = 0; byte <= wb; byte += step) {
			unsigned written = 0, first = 0, seen = 0, nz = 0;
			uint8_t *o;
			uint32_t u = 0;
			size_t at = byte;    /* byte == wb means the control */

			charsiu_bo_prep(dev, &wt, 1000000000);
			memset(wt.map, 0, wb);
			if (at < wb)
				((uint8_t *)wt.map)[at] = (uint8_t)g_live;
			charsiu_bo_fini(dev, &wt);

			charsiu_bo_prep(dev, &outbo, 1000000000);
			memset(outbo.map, 0xa5, obytes);
			charsiu_bo_fini(dev, &outbo);
			if (charsiu_submit(dev, &regcmd, (unsigned)nreg, in_handles,
					   3, out_handles, 1) ||
			    charsiu_bo_prep(dev, &outbo, 2000000000)) {
				printf("  byte %4zu  submit or wait FAILED\n", at);
				continue;
			}
			o = outbo.map;
			for (i = 0; i < obytes; i++)
				if (o[i] != 0xa5)
					written = i + 1;
			for (i = 0; i < written; i++)
				if (o[i]) {
					nz++;
					if (!seen) { first = i; seen = 1; }
				}
			if (seen) {
				unsigned b = first & ~3u;

				u = (uint32_t)o[b] | ((uint32_t)o[b + 1] << 8) |
				    ((uint32_t)o[b + 2] << 16) |
				    ((uint32_t)o[b + 3] << 24);
			}
			charsiu_bo_fini(dev, &outbo);

			if (at == wb) {
				printf("  THE CONTROL, all zero   written %3u"
				       "  nonzero %3u%s\n\n", written, nz,
				       nz ? "   <- FAILED, read no further" : "");
				continue;
			}
			if (!seen) {
				dark++;
				continue;
			}
			lit++;
			/*
			 * THE OUTPUT BYTE ITSELF, not a slot. Round 188 divided
			 * this by four and the six geometries then contradicted
			 * each other: at K = 16 only even slots appeared and at
			 * K = 32 slot 0's first byte came out as 257. The width
			 * of an element is what is being measured here, so it
			 * cannot also be an input to the measurement.
			 */
			s = first;
			nzhist[nz < 17 ? nz : 16]++;
			if (s < 512) {
				slot_cnt[s]++;
				slot_nz[s] = nz;
				if (byte < slot_lo[s]) slot_lo[s] = (unsigned)byte;
				if (byte > slot_hi[s]) slot_hi[s] = (unsigned)byte;
				slot_val[s] = (int32_t)u;
			}
			/*
			 * AN OUTLIER PRINTS WHATEVER THE LIMIT SAYS. K=16 N=64
			 * came back twice with 159 probes at 2 non zero output
			 * bytes and exactly ONE at 16 or more, the same numbers
			 * both times, so it is a feature of the layout rather
			 * than noise, and which weight byte does it is the whole
			 * question about it.
			 */
			if (printed < 24 || nz > 4) {
				printf("  weight byte %4zu -> output byte %3u,"
				       " %u bytes non zero%s\n", byte, s, nz,
				       nz > 4 ? "   <- OUTLIER" : "");
				if (printed < 24)
					printed++;
			}
		}
		printf("\n  %u weight bytes lit something, %u lit nothing\n", lit, dark);
		/*
		 * THE HISTOGRAM IS THE ELEMENT WIDTH. Every probe writes one
		 * value, so how many output bytes come back non zero is how
		 * wide that value is, with the one caveat that a zero byte
		 * INSIDE a value is invisible here. Round 186's sweep printed
		 * "bytes 0..1" all along and this project read 4 off a hex dump
		 * that was aligned to 4.
		 */
		printf("\n  how many output bytes a single live nibble makes non zero:\n");
		for (s = 1; s < 17; s++)
			if (nzhist[s])
				printf("    %2u byte%s  %6u probes%s\n", s,
				       s == 1 ? " " : "s", nzhist[s],
				       s == 16 ? " or more" : "");
		printf("\n  %-12s %-8s %-10s %-10s %-4s %s\n",
		       "output byte", "weights", "first", "last", "nz", "value");
		for (s = 0; s < 512; s++)
			if (slot_cnt[s])
				printf("  %-12u %-8u %-10u %-10u %-4u %d\n", s,
				       slot_cnt[s], slot_lo[s], slot_hi[s],
				       slot_nz[s], slot_val[s]);
		charsiu_close(dev);
		return 0;
	}

	if (argc > 4 && !strcmp(argv[4], "--osweep")) {
		float *af = malloc((size_t)m * k * sizeof(*af));
		size_t obytes = (size_t)m * n * 4;
		unsigned pi;

		for (i = 0; i < m * k; i++)
			af[i] = 8.0f;
		charsiu_bo_prep(dev, &in, 1000000000);
		charsiu_pack_input_f16(&job.mm, af, in.map, in.size);
		charsiu_bo_fini(dev, &in);

		printf("\n  --osweep: ONE live nibble of %u in the whole weight buffer,\n"
		       "  at k = 0 of one channel, activation a constant 8.0 everywhere,\n"
		       "  and %zu output bytes dumped with NO assumption about how wide\n"
		       "  an element is. The first row has no live nibble at all and is\n"
		       "  the control: it must come back with nothing non zero.\n\n",
		       g_live, obytes);

		/*
		 * ROUND 186 SWEEPS EVERY CHANNEL. Round 185 swept nine and the
		 * nine did not agree: channels 0 to 5 landed at byte 4n, but 16
		 * landed at 32 and 32 at 64, which is 2n, and 63 landed at 156,
		 * which is neither. Nine points cannot carry a rule, and fitting
		 * one to part of a map is what cost rounds 171 and 173. So read
		 * the whole map and let the table say it.
		 *
		 * Entry n == mm->n is the control with NO live nibble anywhere.
		 */
		for (pi = 0; pi <= n; pi++) {
			unsigned n0 = pi == 0 ? n : pi - 1;
			unsigned written = 0, nz = 0, first = 0, last = 0, seen = 0;
			uint8_t *o;
			uint32_t u = 0;

			charsiu_bo_prep(dev, &wt, 1000000000);
			memset(wt.map, 0, charsiu_weight_bytes(&job.mm));
			if (n0 < n) {
				size_t row = (size_t)(n0 / 32) * 512 +
					     (size_t)(n0 % 32) * 8;

				((uint8_t *)wt.map)[row] = (uint8_t)g_live;
			}
			charsiu_bo_fini(dev, &wt);

			charsiu_bo_prep(dev, &outbo, 1000000000);
			memset(outbo.map, 0xa5, obytes);
			charsiu_bo_fini(dev, &outbo);

			if (charsiu_submit(dev, &regcmd, (unsigned)nreg, in_handles,
					   3, out_handles, 1) ||
			    charsiu_bo_prep(dev, &outbo, 2000000000)) {
				printf("  channel %-4u  submit or wait FAILED\n", n0);
				continue;
			}
			o = outbo.map;
			/* the sentinel is itself non zero, so the write extent
			 * has to be found before anything is counted */
			for (i = 0; i < obytes; i++)
				if (o[i] != 0xa5)
					written = i + 1;
			for (i = 0; i < written; i++)
				if (o[i]) {
					nz++;
					if (!seen) { first = i; seen = 1; }
					last = i;
				}
			if (seen) {
				unsigned b = first & ~3u;

				u = (uint32_t)o[b] | ((uint32_t)o[b + 1] << 8) |
				    ((uint32_t)o[b + 2] << 16) |
				    ((uint32_t)o[b + 3] << 24);
			}
			if (n0 >= n)
				printf("  NO live nibble  written %3u  nonzero %3u   "
				       "THE CONTROL: nonzero must be 0\n", written, nz);
			else if (!seen)
				printf("  channel %-3u     written %3u  nothing non zero\n",
				       n0, written);
			else
				printf("  channel %-3u     written %3u  bytes %3u..%-3u"
				       "  slot %2u  0x%08x %12d%s\n",
				       n0, written, first, last, first / 4, u,
				       (int32_t)u, nz > 4 ? "   MORE THAN ONE SLOT" : "");
			charsiu_bo_fini(dev, &outbo);
		}
		/*
		 * The whole buffer once, so the reader can see what the stage
		 * writes into the bytes that are not the answer.
		 */
		printf("\n  and the whole buffer for the last channel swept:\n");
		{
			uint8_t *o = outbo.map;

			charsiu_bo_prep(dev, &outbo, 1000000000);
			o = outbo.map;
			for (i = 0; i < obytes; i += 16) {
				unsigned b;

				printf("      %3u  ", i);
				for (b = i; b < i + 16 && b < obytes; b++)
					printf("%02x%s", o[b], (b & 3) == 3 ? " " : "");
				printf("\n");
			}
			charsiu_bo_fini(dev, &outbo);
		}
		charsiu_close(dev);
		return 0;
	}

	if (argc > 4 && !strcmp(argv[4], "--f16")) {
		/*
		 * ROUND 184 BREAKS A CONFOUND THIS PROBE HAS CARRIED SINCE
		 * ROUND 181, and everything read off it until now is in doubt
		 * until this run comes back.
		 *
		 * The activation was (c mod 16) - 16, which is LINEAR IN THE
		 * CHANNEL INDEX, and the live nibble was a constant 7 in every
		 * run. So "the output varies with the activation" and "the
		 * output varies with the channel index" were the same sentence,
		 * and nothing has ever shown the output depends on the WEIGHTS
		 * at all. Round 181's fit, npu = 319 * act - 2412, can be
		 * rewritten as 319 * (c mod 16) + a constant without touching a
		 * single measured number.
		 *
		 * Two knobs, and each one has a branch that kills the reading.
		 *
		 *   CHARSIU_INT4_PERM=1 permutes which k holds which activation
		 *     VALUE. The value set, the range and the packing are all
		 *     unchanged; only the pairing moves, so this is one
		 *     variable. Both fits are then printed side by side and the
		 *     RESIDUALS decide it, not the slopes.
		 *
		 *   CHARSIU_INT4_LIVE=3 changes the live nibble from 7 to 3.
		 *     A real MAC has to scale with it and the slope has to fall
		 *     from 7 to 3. A slope that does not move means the weights
		 *     are not in the arithmetic, whatever the activation does.
		 */
		static const unsigned perm[16] = {
			9, 2, 14, 5, 11, 0, 7, 13, 3, 15, 6, 8, 1, 12, 4, 10
		};
		int permute = getenv("CHARSIU_INT4_PERM") != NULL;
		float *af = malloc((size_t)m * k * sizeof(*af));
		unsigned c, j, seen16[65536] = { 0 }, distinct = 0;
		uint8_t *o;

		/* an input a half can hold exactly, so the packing adds no
		 * error of its own: small integers. Permuted or not, the values
		 * are the same sixteen. */
		for (i = 0; i < m * k; i++) {
			unsigned slot = i % 32;

			if (permute && slot < 16)
				slot = perm[slot];
			af[i] = (float)((int)slot - 16);
		}
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
				(uint8_t)((kk & 1) ? (g_live << 4) : g_live);
		}
		charsiu_bo_fini(dev, &wt);

		charsiu_bo_prep(dev, &outbo, 1000000000);
		memset(outbo.map, 0xa5, (size_t)m * n * 4);
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
		       "  %u per channel at k = c mod 16, and the output read as HALVES\n"
		       "  because the vendor's own configuration requantises with\n"
		       "  0x40b0 = 1 and 0x40b4 = 0, which is identity.\n",
		       g_live);
		printf("  live nibble %u   activation %s\n\n", g_live,
		       permute ? "PERMUTED, so its value no longer tracks c"
			       : "the plain ramp, which tracks c exactly");
		printf("  %-4s %-12s %-12s %-12s %-10s\n",
		       "c", "act[c%16]", "expected", "npu (half)", "raw");
		for (c = 0; c < 16 && c < n; c++) {
			uint16_t h = (uint16_t)(o[c * 2] | (o[c * 2 + 1] << 8));
			float got = charsiu_half_to_float(h);

			printf("  %-4u %-12.3f %-12.3f %-12.3f 0x%04x\n", c,
			       af[c % 16], af[c % 16] * (float)g_live, got, h);
		}
		j = 0;
		for (c = 0; c < m * n * 4; c++)
			if (((uint8_t *)outbo.map)[c] == 0xa5) j++;
		printf("\n  %u of %u bytes still hold the sentinel\n", j, m * n * 4);
		/*
		 * THE SAME BYTES READ AS 32 BIT INTEGERS. Round 184's halves
		 * came out in a period of four, with 0xffff and 0x0000 sitting
		 * in every second slot, which is what the high half of a signed
		 * 32 bit value looks like. The registers only say the stage does
		 * not requantise; an un-requantised output is the accumulator,
		 * and that is an integer. Both readings are printed so the one
		 * that is wrong can be seen to be wrong.
		 */
		printf("\n  the same bytes as 32 bit integers, which is what an\n"
		       "  un-requantised accumulator would be:\n");
		for (c = 0; c < 8 && c * 4 + 3 < m * n * 4; c++) {
			uint32_t u = (uint32_t)o[c * 4] | ((uint32_t)o[c * 4 + 1] << 8) |
				     ((uint32_t)o[c * 4 + 2] << 16) |
				     ((uint32_t)o[c * 4 + 3] << 24);

			printf("    element %-3u 0x%08x  %12d\n", c, u, (int32_t)u);
		}
		for (c = 0; c < 16 && c < n; c++) {
			uint16_t h = (uint16_t)(o[c * 2] | (o[c * 2 + 1] << 8));

			if (!seen16[h]++)
				distinct++;
		}
		printf("  %u distinct values over the 16 channels%s\n", distinct,
		       distinct < 8 ? "   TOO FLAT TO JUDGE A FIT" : "");
		/*
		 * BOTH FITS GO IN THE LOG. Round 181 was read by eyeballing four
		 * numbers off a table and computing a slope by hand, which is
		 * how a sweep stride got read as a row size two rounds earlier,
		 * and it fitted only the variable it already believed in.
		 */
		{
			double xa[64], xc[64], yy[64];
			unsigned nn = 0;

			for (c = 0; c < 16 && c < n; c++) {
				uint16_t h = (uint16_t)(o[c * 2] | (o[c * 2 + 1] << 8));
				float got = charsiu_half_to_float(h);

				if (!(got > -1e30f && got < 1e30f) || got == 0.0f)
					continue;       /* nan, inf and the dead ones */
				xa[nn] = af[c % 16];
				xc[nn] = (double)c;
				yy[nn] = got;
				nn++;
			}
			/*
			 * NOT "varying". Round 184 called these varying and
			 * four of the eight were constants that did not move
			 * between two runs with different activations, which
			 * poisoned both fits. Whether a slot varies can only be
			 * seen ACROSS runs, never within one.
			 */
			printf("\n  %u finite non zero halves"
			       "  (whether they VARY needs two runs to say)\n", nn);
			fit_line("the activation", xa, yy, nn, (double)g_live);
			fit_line("the channel index", xc, yy, nn, 0);
			printf("\n  THE RULE, WRITTEN BEFORE THE RUN.\n"
			       "  The smaller rms residual names what the output depends on.\n"
			       "  If the channel index fits and the activation does not, this\n"
			       "  path is not computing and rounds 181 to 183 are read wrong.\n"
			       "  And between the two live nibble runs the activation slope\n"
			       "  must fall from 7 to 3, or the weights are not in the MAC.\n");
		}
		charsiu_close(dev);
		return 0;
	}

	if (argc > 4 && !strcmp(argv[4], "--kpair")) {
		/*
		 * ROUND 261. The layout this project has is
		 *
		 *   channel n at (n / 32) * 512 + (n % 32) * 8, eight bytes
		 *   byte b nibble h is k = 2b + h
		 *
		 * and eight bytes is sixteen nibbles, so it describes k = 0 to
		 * 15 and nothing above. At the default 64 by 64 the buffer is
		 * 2048 bytes and that formula accounts for 512 of them, so
		 * three quarters of it is unexplained and k = 16 upward has to
		 * live in there somewhere. The map calls those bytes dead
		 * because it never had an input past k = 15 to pair them with.
		 *
		 * This probe already answers it. One live nibble at a byte, a
		 * one hot input walking k, and it prints BOTH the k it pairs
		 * with and the first channel that lights. The only thing it was
		 * missing is bytes in the unexplained region.
		 *
		 * The bytes below are one per candidate block boundary rather
		 * than a dense sweep, since each costs k submits. 8 and 256
		 * are the two that separate the readings on their own:
		 *
		 *   byte 8    contiguous 16 bytes per channel  channel 0, k=16
		 *             eight bytes per channel          channel 1, k=0
		 *   byte 256  a k block at +256                channel 0, k=16
		 *             a k block at +1024               nothing, or 32
		 */
		/*
		 * ROUND 262. Round 261 established that k = 16 and above are
		 * reachable and live in the first 256 bytes, that the byte to
		 * (k, channel) map does not depend on K at all, and that every
		 * live byte pairs with exactly one k. What it could not do is
		 * name the map, because fourteen points with gaps do not.
		 *
		 * The points that exist, low nibble, both at K = 64 and 128:
		 *
		 *   byte    0  1  2  3  4  7    8   16  256  257  264
		 *   k       0  2  4  6  8 14   16    0   32   34   48
		 *   channel 0  0  0  0  0  0    4    8    0    0    4
		 *
		 * Bytes 248 and 255 are dead, so the live set inside the first
		 * 256 is not contiguous either.
		 *
		 * ⚠ No formula is fitted here on purpose. Every reading tried
		 * against those points has bit 3 of the byte contributing to
		 * BOTH the k and the channel, which is not how a bitfield
		 * splits, and a fit through gaps is exactly what this project
		 * has had to retract before. 9 through 15 and 17 through 31
		 * decide it outright, so they get measured instead.
		 */
		/*
		 * ROUND 263. Two jobs, and the second is the one that matters.
		 *
		 * The rule round 262 fitted, 88 of 88 points and no miss:
		 *
		 *   G = B/128,  b = B%128,  blk = b/8
		 *   b >= 64             dead
		 *   channel = 4*blk + 32*(G & 1)
		 *   k       = 16*(blk & 1) + 2*(b % 8) + 32*(G >> 1), high +1
		 *
		 * It is still a fit. These four bytes were never measured and
		 * the rule's answers for them are written here BEFORE the run,
		 * which is the only thing that turns a fit into a reading:
		 *
		 *   byte  57   channel 28   k 18
		 *   byte 140   channel 36   k 24
		 *   byte 384   channel 32   k 32
		 *   byte 392   channel 36   k 48
		 *   byte 100   dead
		 *   byte 200   dead
		 *
		 * And the probe now prints every channel a nibble lights
		 * rather than the first, because the rule addresses only
		 * channels that are multiples of four, which is 16 of the 64
		 * this shape has.
		 */
		/*
		 * ROUND 264. Half the channels have no home.
		 *
		 * The rule, now confirmed by six predictions for six on the k
		 * side and measured two channels wide:
		 *
		 *   G = B/128,  b = B%128,  blk = b/8
		 *   b >= 64                       dead
		 *   channel = 4*blk + 32*(G & 1)  and that channel + 1
		 *   k = 16*(blk & 1) + 2*(b % 8) + 32*(G >> 1), high +1
		 *
		 * It addresses c mod 4 in {0, 1}, which is 32 of 64, and the
		 * b >= 64 half of every 128 byte region is dead. Two halves
		 * missing, one of channels and one of bytes, and no reason yet
		 * to think they are the same half.
		 *
		 * Two things get swept. The dead half densely, because four
		 * points is not a sweep and 64 and 72 were both blk boundaries.
		 * And N, from the harness, because a map that covers half the
		 * channels at every N is a different animal from one that
		 * covers all of a small N.
		 *
		 * ⚠ No two-option prediction here on purpose. Three rounds
		 * running, the answer was outside the two I wrote down: byte 8
		 * was channel 4 at k 16 when the options were channel 1 or
		 * channel 0, and the channel width was two when the options
		 * were one and four. What is written instead is what each
		 * observation would mean, with room for it to be none of them.
		 */
		static const size_t bytes[] = {
			0, 8, 16, 24, 32, 40, 48, 56,
			64, 65, 66, 72, 80, 88, 96, 104, 112, 120, 127,
			128, 136,
			/*
			 * 256 and 384 are r263's two K invariant rows. 512 was
			 * round 265's control, and it FIRED: the rule then in
			 * force put it at k 64, past the end of a K of 64, and
			 * it came back live at word 16, k 0. What that killed
			 * is the G term. 32*(G>>1) and 32*((G>>1)&1) agree at
			 * G of 0, 1, 2 and 3 and first differ at G of 4, which
			 * is byte 512, and the second one is right.
			 */
			256, 384, 512,
			/*
			 * b MOD 8 INSIDE THE LIVE HALF AT N = 64. Round 265 has
			 * no point at all for this: its only b%8 samples were
			 * bytes 65 and 66, which sit in the half that is dead at
			 * N = 64, so they only ever read at N = 16. The N = 16
			 * rule carried over predicts word 0 throughout with
			 * k = 2*(b%8), which is 2, 4, 6, 8, 10, 12 and 14.
			 */
			1, 2, 3, 4, 5, 6, 7,
			/*
			 * THE DEAD HALF AT A DIFFERENT G. Every dead point in
			 * 265 was in G = 0, so "bit 6 of the address is dead"
			 * is fitted on one region. 192 is G = 1 and 320 is G = 2
			 * at the same b of 64. If the bit is the reason, both
			 * are dark; if only G = 0 has a hole, it is not a bit.
			 */
			192, 320,
			/*
			 * THE G BITS ACROSS THE WHOLE BUFFER. The rule assigns
			 * G bit 0 to word +8, G bit 1 to k +32 and G bits 2 and
			 * up to word +16 each, which was read off five G values
			 * and predicts six more, at b = 0:
			 *
			 *   640  G 5  -> word 24 k 0     1024 G 8  -> word 32 k 0
			 *   768  G 6  -> word 16 k 32    1536 G 12 -> word 48 k 0
			 *   896  G 7  -> word 24 k 32    1920 G 15 -> word 56 k 32
			 *
			 * Six points, and they span to the end of a 2048 byte
			 * buffer rather than stopping a quarter of the way in.
			 */
			640, 768, 896, 1024, 1536, 1920,
		};

		unsigned j;

		printf("\n  --kpair: ONE live nibble, a ONE HOT input, sweeping k.\n"
		       "  Reads the k a nibble pairs with, and the output WORD it\n"
		       "  lights: w<index>=<value>/b<which of the four bytes>.\n"
		       "  The word index is the channel candidate, the byte pattern\n"
		       "  says how wide an element is, and the value says whether it\n"
		       "  is a signed 32 bit integer. Live nibble %x.\n\n", g_live);
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
