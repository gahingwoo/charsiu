// SPDX-License-Identifier: GPL-2.0
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * Does the NPU compute a matmul with MORE THAN ONE ROW of A?
 *
 * charsiu has only ever asked for m=1: npudev.c pins `s->job.mm.m = 1` and
 * every token is a fresh matrix-vector product, so a prompt reads every weight
 * once per token. That is why prefill throughput is flat with prompt length.
 *
 * The job format is a GEMM -- `struct charsiu_matmul` carries m, and
 * charsiu_pack_input has a general m>1 path with the m==1 fast path beside it
 * -- and the emitter differentiates: m=2 doubles the input surface width in
 * 0x1028 and sets the height fields to 1. None of that has ever been run.
 *
 * ⚠ m=1 IS THE CONTROL AND IT RUNS FIRST. If m=1 disagrees with the CPU then
 * this test is wrong and says so, rather than blaming a field it was built to
 * examine. Only m=1 passing makes an m=2 failure mean anything.
 *
 *   npu_gemm_test [K] [N]     sweeps M = 1, 2, 4, 8, 32
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "charsiu.h"

static int run(struct charsiu_device *dev, unsigned m, unsigned k, unsigned n,
	       const uint8_t *A, const uint8_t *B, int32_t *out)
{
	struct charsiu_job job = { 0 };
	struct charsiu_bo wt = { 0 }, in = { 0 }, ob = { 0 }, coef = { 0 }, reg = { 0 };
	size_t nreg, insz;
	int rc = -1;

	job.cbuf_window = (unsigned)charsiu_cbuf_window();
	job.mm.m = m; job.mm.k = k; job.mm.n = n;
	job.mm.wdtype = CHARSIU_INT8;
	job.mm.adtype = CHARSIU_INT8;
	job.input_zero_point = 128;
	job.weight_zero_point = 128;
	job.output_zero_point = 0;
	job.input_scale = 1.0f;
	job.weight_scale = 1.0f;
	job.output_scale = 1.0f;
	job.acc_out = 1;              /* the raw int32 accumulator */

	/*
	 * ⚠ entries_per_row IS PER ROW, and the packed input holds m of them
	 * interleaved: [K/atom][M][atom]. Sizing it without the m factor gave a
	 * 64 byte buffer for a 128 byte pack, which only survived on the slack
	 * and would have failed on the board as something mysterious. Verified
	 * on the host: with the m factor, 0 of 128 elements land outside.
	 */
	insz = (size_t)charsiu_entries_per_row(&job.mm) * 64 * m + 4096;
	if (charsiu_bo_alloc(dev, charsiu_weight_bytes(&job.mm) + 4096, &wt) ||
	    charsiu_bo_alloc(dev, insz, &in) ||
	    charsiu_bo_alloc(dev, (size_t)m * n * 4 + 4096, &ob) ||
	    charsiu_bo_alloc(dev, charsiu_coef_bytes(&job.mm) + 4096, &coef) ||
	    charsiu_bo_alloc(dev, 4096, &reg)) {
		fprintf(stderr, "  a buffer would not allocate\n");
		goto out;
	}
	if (!wt.map || !in.map || !ob.map || !coef.map || !reg.map) {
		fprintf(stderr, "  a buffer allocated but did not map\n");
		goto out;
	}

	charsiu_bo_prep(dev, &wt, 1000000000);
	memset(wt.map, 0, charsiu_weight_bytes(&job.mm));
	charsiu_pack_weights(&job.mm, B, wt.map);
	charsiu_bo_fini(dev, &wt);

	charsiu_bo_prep(dev, &in, 1000000000);
	charsiu_pack_input(&job.mm, A, in.map, insz, job.input_zero_point);
	charsiu_bo_fini(dev, &in);

	/*
	 * ⚠ NEITHER OF THESE MAY BE NULL. charsiu_build_coefs dereferences
	 * bias[oc] and weight_sums[oc] unconditionally, and passing NULL
	 * segfaulted on the board before the control had printed a single
	 * result. Zeros are the right values here anyway: input_zero_point is
	 * 0x80, so the (in_zp - 0x80) factor that multiplies the weight sum is
	 * exactly zero, and there is no bias in this test.
	 */
	{
		int32_t *zero = calloc(n, sizeof(int32_t));

		if (!zero) { fprintf(stderr, "  out of memory\n"); goto out; }
		charsiu_bo_prep(dev, &coef, 1000000000);
		charsiu_build_coefs(&job, zero, zero, coef.map);
		charsiu_bo_fini(dev, &coef);
		free(zero);
	}

	job.input_addr = (uint32_t)in.dma_address;
	job.output_addr = (uint32_t)ob.dma_address;
	job.weight_addr = (uint32_t)wt.dma_address;
	job.coef_addr = (uint32_t)coef.dma_address;

	charsiu_bo_prep(dev, &reg, 1000000000);
	nreg = charsiu_emit_job(&job, reg.map, 4096 / 8);
	charsiu_bo_fini(dev, &reg);
	if (!nreg) {
		fprintf(stderr, "  the register stream came back empty\n");
		goto out;
	}

	charsiu_bo_prep(dev, &ob, 1000000000);
	memset(ob.map, 0, (size_t)m * n * 4);
	charsiu_bo_fini(dev, &ob);

	{
		uint32_t ins[2] = { in.handle, wt.handle };
		uint32_t outs[1] = { ob.handle };

		if (charsiu_submit(dev, &reg, (unsigned)nreg, ins, 2, outs, 1)) {
			fprintf(stderr, "  the submit failed\n");
			goto out;
		}
	}
	charsiu_bo_prep(dev, &ob, 1000000000);
	memcpy(out, ob.map, (size_t)m * n * 4);
	charsiu_bo_fini(dev, &ob);
	rc = 0;
out:
	charsiu_bo_free(dev, &reg); charsiu_bo_free(dev, &coef);
	charsiu_bo_free(dev, &ob); charsiu_bo_free(dev, &in);
	charsiu_bo_free(dev, &wt);
	return rc;
}

/* what the hardware is being asked for: (a - zp) . (b - zp) per output */
/*
 * A byte around 128 that depends on BOTH indices with no short period. Small
 * enough that a dot product of a few thousand terms stays well inside int32.
 */
static uint8_t mix(unsigned a, unsigned b, unsigned span)
{
	uint32_t h = a * 2654435761u ^ (b + 0x9e3779b9u) * 40503u;

	h ^= h >> 13;
	return (uint8_t)(128 + (int)(h % span) - (int)(span / 2));
}

static void reference(unsigned m, unsigned k, unsigned n,
		      const uint8_t *A, const uint8_t *B, int32_t *out)
{
	for (unsigned r = 0; r < m; r++)
		for (unsigned c = 0; c < n; c++) {
			int32_t acc = 0;

			for (unsigned i = 0; i < k; i++)
				acc += ((int)A[(size_t)r * k + i] - 128)
				     * ((int)B[(size_t)c * k + i] - 128);
			out[(size_t)r * n + c] = acc;
		}
}

static int check(const char *what, unsigned m, unsigned n,
		 const int32_t *got, const int32_t *want)
{
	unsigned bad = 0;
	long worst = 0;

	for (size_t i = 0; i < (size_t)m * n; i++) {
		long d = (long)got[i] - (long)want[i];

		if (d) { bad++; if (labs(d) > labs(worst)) worst = d; }
	}
	printf("  %-22s %5u of %5u wrong, worst delta %ld\n",
	       what, bad, m * n, worst);
	if (bad) {
		printf("      first few  got:");
		for (unsigned i = 0; i < 6 && i < m * n; i++) printf(" %d", got[i]);
		printf("\n                 want:");
		for (unsigned i = 0; i < 6 && i < m * n; i++) printf(" %d", want[i]);
		printf("\n");
	}
	return bad != 0;
}

/*
 * ⚠ "WRONG" AND "SOMEWHERE ELSE" ARE DIFFERENT ANSWERS, and only one of them
 * means m>1 is unusable. The first m=2 run had its first four outputs exactly
 * right and then diverged, which is what a layout looks like rather than an
 * arithmetic fault. So before concluding anything, ask whether the numbers are
 * all THERE.
 */
/*
 * WHERE DID EACH WANTED VALUE ACTUALLY LAND?
 *
 * Guessing layouts stopped paying. Round 381 tried nine activation
 * arrangements and five output ones and the best was 15 of 128, while the
 * FIRST FOUR words of the output were exact at m = 2, 4, 8 and 32 alike --
 * four int32 is sixteen bytes, one output atom. Something writes atom zero of
 * row zero where this reads it and everything after somewhere else, and no
 * amount of candidate formulas will say where if none of them is the one.
 *
 * So print the address function instead of hypothesising it: for each of the
 * first few (row, channel) pairs, the flat index in the output buffer that
 * holds the value the CPU computed for it. A layout is then read off the
 * table rather than matched against a list.
 *
 * ⚠ AMBIGUOUS HITS ARE MARKED. A dot product of 256 int8 pairs can repeat, so
 * a value found at three places is three candidates and not a fact; the count
 * is printed so a reader can discount it.
 */
/*
 * How many words the fitted address function spans at this shape. The board
 * writes exactly that many, which is what makes the count a measurement of the
 * strides rather than a symptom.
 */
static size_t g_live_at_n;    /* words the board wrote at the full N */
static size_t g_live_at_m4;   /* and the same at m = 4 */

static size_t addr_range(unsigned n, unsigned m, unsigned R, unsigned S)
{
	size_t hi = 0;

	for (unsigned r = 0; r < m; r++)
		for (unsigned c = 0; c < n; c++) {
			unsigned u = c / 16, v = c % 16;
			size_t f = (size_t)S * (u / 2) + (size_t)R * r
				 + 8 * (v / 4) + 4 * (u % 2) + (v % 4);

			if (f > hi)
				hi = f;
		}
	return hi + 1;
}

static void locate(unsigned m, unsigned n, const int32_t *got,
		   const int32_t *want)
{
	size_t total = (size_t)m * n;
	unsigned shown = 0;

	/*
	 * ⚠ ROW 0 ACROSS THE WHOLE ROW, not its first six channels. Round 381
	 * sampled six, saw them land at 0 1 2 3 8 9 -- which is exactly
	 * [n/4][m][4] -- and then had to explain why the same layout scored
	 * only 15 of 128. Six channels cannot say where a pattern STOPS.
	 */
	/*
	 * ⚠ SAY HOW TRUSTWORTHY THIS TABLE IS. Every row of it reports the
	 * FIRST index holding a value, so a value the reference produces twice
	 * gives an answer that is one of two and reads like one of one.
	 */
	{
		size_t uniq = 0;

		for (size_t i = 0; i < total; i++) {
			size_t j;

			for (j = 0; j < i; j++)
				if (want[j] == want[i])
					break;
			if (j == i)
				uniq++;
		}
		printf("\n  the reference has %zu distinct values in %zu"
		       " (a table below is only as good as this)\n",
		       uniq, total);
	}
	printf("  where each wanted value landed  (flat index in the output)\n");
	printf("    %-10s %-12s %-10s %-8s %s\n",
	       "(row,ch)", "want", "at flat", "hits", "[n/4][m][4] predicts");
	for (unsigned pass = 0; pass < 2; pass++) {
		unsigned r = pass;          /* row 0 in full, then row 1 */
		unsigned step = pass ? (n / 8 ? n / 8 : 1) : 1;

		if (r >= m)
			break;
		for (unsigned c = 0; c < n; c += step) {
			int32_t v = want[(size_t)r * n + c];
			size_t first = total, hits = 0;
			size_t pred = (size_t)(c / 4) * m * 4 + (size_t)r * 4
				      + (c % 4);

			for (size_t i = 0; i < total; i++)
				if (got[i] == v) {
					if (!hits)
						first = i;
					hits++;
				}
			/* row 0 is dense; print it thinned once it is boring */
			if (!pass && c >= 16 && (c % 4))
				continue;
			printf("    (%u,%-3u)    %-12d ", r, c, v);
			if (!hits)
				printf("%-10s %-8s ", "nowhere", "-");
			else
				printf("%-10zu %-8zu ", first, hits);
			printf("%zu%s\n", pred,
			       hits && first == pred ? "  <= matches" : "");
			shown++;
			if (shown > 40)
				break;
		}
		shown = 0;
	}
	printf("    a contiguous [m][n] would put (r,c) at r*%u + c\n", n);

	/*
	 * ⚠ AND THE WHOLE BUFFER, not a sample of it. Thirty sampled rows say
	 * where a pattern holds and cannot say where it STOPS, which is the
	 * only interesting thing left: round 383's samples were exact for
	 * channels 0 to 11 and then broke, and no sampling density would have
	 * found the edge on its own.
	 *
	 * One line per eight words: for each flat index, which (row, channel)
	 * the reference says that value belongs to, or a dot. Read down the
	 * columns and the address function is simply visible.
	 */
	{
		const char *ax = getenv("CHARSIU_M_AXIS");
		const char *ap = getenv("CHARSIU_ENTRY_ATOMICS");

		/*
		 * ⚠ SAY WHICH CONFIGURATION THIS MAP IS OF. Round 384's map
		 * was printed with CHARSIU_ENTRY_ATOMICS left at 8 by the
		 * sweep and round 385's at 4, and the two look nothing alike
		 * -- which read as the hardware being non-deterministic until
		 * the difference turned out to be an environment variable
		 * nobody reset.
		 */
		printf("\n  the whole output, as the (row,channel) each word"
		       " holds\n  (M on the %s, CHARSIU_ENTRY_ATOMICS=%s)\n",
		       ax && (*ax == 'w' || *ax == 'W') ? "width" : "height",
		       ap ? ap : "4");
	}
	for (size_t i = 0; i < total; i += 8) {
		printf("    %4zu ", i);
		for (size_t j = i; j < i + 8 && j < total; j++) {
			size_t hits = 0, at = 0;

			for (size_t q = 0; q < total; q++)
				if (want[q] == got[j]) { if (!hits) at = q; hits++; }
			if (!hits)
				printf("  .    ");
			else if (hits > 1)
				printf(" ?%zu,%-3zu", at / n, at % n);
			else
				printf(" %zu,%-4zu", at / n, at % n);
		}
		printf("\n");
	}
	printf("    a leading ? means that value is not unique in the"
	       " reference\n");

	/*
	 * ⚠⚠ THE QUESTION THIS WHOLE FILE IS ACTUALLY ASKING. If every value
	 * the reference computes appears exactly once in the buffer, then the
	 * ARITHMETIC IS RIGHT and only the read order is wrong -- and a
	 * batched prefill is available today, through a permutation, without
	 * another register. If values are missing, the hardware did not
	 * compute them and no read order will help.
	 *
	 * Counted, not eyeballed: a map that fits every cell it can see can
	 * still be an overfit on the cells it cannot.
	 */
	{
		size_t placed = 0, absent = 0, ambiguous = 0;
		int injective = 1;
		unsigned char *used = calloc(total, 1);

		for (size_t q = 0; q < total; q++) {
			size_t hits = 0, at = 0;

			for (size_t i = 0; i < total; i++)
				if (got[i] == want[q]) { if (!hits) at = i; hits++; }
			if (!hits) absent++;
			else if (hits > 1) ambiguous++;
			else {
				placed++;
				if (used && used[at])
					injective = 0;
				if (used)
					used[at] = 1;
			}
		}
		free(used);
		printf("\n  of %zu reference values: %zu land in exactly one slot,"
		       " %zu in several,\n  %zu are absent from the buffer"
		       " altogether%s\n", total, placed, ambiguous, absent,
		       placed && injective ? "; the unique ones collide with"
					     " nothing" : "");
		if (!absent && injective)
			printf("\n  ⚑ THE ARITHMETIC IS RIGHT AND ONLY THE READ"
			       " ORDER IS WRONG.\n  A batched prefill is"
			       " available through a permutation, with no\n"
			       "  further register work.\n");
		else if (absent)
			printf("\n  %zu values were never computed, so no read"
			       " order recovers them.\n", absent);

		/*
		 * ⚠ AND THE ONE NUMBER THAT SAYS WHAT IT RAN OUT OF. If the
		 * hardware always writes the same COUNT of words whatever N
		 * is, something has a fixed capacity -- a CBUF bank, a task's
		 * output allowance. If it always writes the same FRACTION,
		 * a stride or a count is scaled wrong. Those want different
		 * fixes and one number tells them apart, so print it rather
		 * than leaving it to be inferred from two logs.
		 */
		{
			size_t last = 0, live = 0;

			for (size_t i = 0; i < total; i++) {
				int seen = 0;

				for (size_t q = 0; q < total; q++)
					if (want[q] == got[i]) { seen = 1; break; }
				if (seen) { live++; last = i + 1; }
			}
			printf("  the board wrote %zu of %zu words, the last at"
			       " %zu (%.0f%%)\n\n"
			       "  how many words each extra row is worth:\n",
			       live, total, last,
			       100.0 * (double)live / (double)total);
			g_live_at_n = live;
		}
	}
}

static int layouts(unsigned m, unsigned n, const int32_t *got, const int32_t *want)
{
	size_t total = (size_t)m * n;
	static const unsigned G[] = { 2, 4, 8, 16, 32 };
	unsigned best = 0;
	const char *bestname = NULL;

	/* are the values even present, in any order? */
	{
		int32_t *a = malloc(total * 4), *b = malloc(total * 4);
		size_t i, j, hit = 0;

		memcpy(a, got, total * 4);
		memcpy(b, want, total * 4);
		for (i = 0; i < total; i++)
			for (j = 0; j < total; j++)
				if (b[j] == a[i]) { b[j] = INT32_MIN; hit++; break; }
		printf("\n  as a multiset: %zu of %zu values are present somewhere\n",
		       hit, total);
		free(a); free(b);
	}

	/* [n][m] instead of [m][n] */
	{
		unsigned ok = 0;

		for (unsigned r = 0; r < m; r++)
			for (unsigned c = 0; c < n; c++)
				if (got[(size_t)c * m + r] == want[(size_t)r * n + c]) ok++;
		if (ok > best) { best = ok; bestname = "[n][m], column major"; }
	}

	/* [n/G][m][G]: G output channels, then the rows, then the next G */
	for (unsigned gi = 0; gi < sizeof(G) / sizeof(*G); gi++) {
		unsigned g = G[gi], ok = 0;

		if (n % g)
			continue;
		for (unsigned r = 0; r < m; r++)
			for (unsigned c = 0; c < n; c++) {
				size_t off = (size_t)(c / g) * m * g + (size_t)r * g + (c % g);

				if (off < total && got[off] == want[(size_t)r * n + c]) ok++;
			}
		if (ok > best) {
			static char nm[48];

			snprintf(nm, sizeof(nm), "[n/%u][m][%u]", g, g);
			best = ok; bestname = nm;
		}
	}

	printf("  best layout tried: %s, %u of %zu\n",
	       bestname ? bestname : "(none)", best, total);
	if (best == total) {
		printf("\n  THE ARITHMETIC IS RIGHT. Only the output layout differs\n"
		       "  from [m][n]; m>1 is usable once the reader matches it.\n");
		return 1;
	}
	return 0;
}

/*
 * One pass of the M sweep. `ape` is CHARSIU_ENTRY_ATOMICS: how many channel
 * atoms this tree thinks fit in a CBUF entry.
 *
 * ⚠ 4 IS WHAT THIS TREE HAS ALWAYS SAID AND 8 IS WHAT MESA SAYS. Same
 * function, one constant apart -- rkt_task.c divides by CBUF_ENTRY_SIZE /
 * FEATURE_ATOMIC_SIZE = 128/16 = 8. With 8, charsiu's stream agrees with
 * Mesa's generic RK3576 encoder on ALL 25 geometry words at M = 1, 2, 4 and
 * 32, at every shape tried; with 4 it differs on the three that carry `surf`,
 * and surf enters 0x1028 as `surf * rows`, so the error scales with M.
 *
 * Mesa's encoder is the one this board ran M = 1, 2, 3, 4 and 8 through
 * exactly. 4 stays the default because every correct result in this tree was
 * measured with it; this is the round that asks.
 */
static int sweep(struct charsiu_device *dev, unsigned ape, char axis,
		 const unsigned *MS, unsigned nms, unsigned k, unsigned n,
		 const uint8_t *A, const uint8_t *B,
		 int32_t *got, int32_t *want, unsigned *passed, unsigned *tried)
{
	char apes[8];
	int fail = 0;

	snprintf(apes, sizeof(apes), "%u", ape);
	setenv("CHARSIU_ENTRY_ATOMICS", apes, 1);
	if (axis == 'w')
		setenv("CHARSIU_M_AXIS", "w", 1);
	else
		unsetenv("CHARSIU_M_AXIS");
	printf("\n  M on the %s, CHARSIU_ENTRY_ATOMICS=%u%s\n",
	       axis == 'w' ? "WIDTH " : "height", ape,
	       ape == 8 ? "  (Mesa's constant)" : "");

	*passed = *tried = 0;
	for (unsigned x = 0; x < nms; x++) {
		unsigned m = MS[x];
		char what[32];

		snprintf(what, sizeof(what), "m=%-3u%s", m,
			 m == 1 ? " (the control)" : "");
		reference(m, k, n, A, B, want);
		if (run(dev, m, k, n, A, B, got)) {
			printf("  %-22s submit failed\n", what);
			fail = 1;
			if (m == 1)
				break;
			continue;
		}
		(*tried)++;
		if (check(what, m, n, got, want))
			fail = 1;
		else
			(*passed)++;

		/*
		 * ⚠ m=1 IS THE CONTROL AND NOTHING BELOW IT MEANS ANYTHING. If
		 * the one width this tree has always run disagrees with the
		 * CPU, the pass is broken and says so rather than blaming a
		 * field it was built to examine.
		 */
		if (m == 1 && fail) {
			printf("      m=1 disagrees, so this pass says nothing "
			       "about m>1\n");
			break;
		}
	}
	return fail;
}

int main(int argc, char **argv)
{
	/*
	 * ⚠ M IS A SWEEP NOW, NOT A PAIR. Round 379 ran m=1 and m=2 only and
	 * concluded "m>1 does not work" from one failing width. The register
	 * fix in 2184557 is about the input surface block, which is degenerate
	 * at m=1 and rounds to four at m=2, so those two widths are the two
	 * least informative ones to stop at: m=4, 8 and 32 are where a
	 * batched prefill would actually live.
	 */
	static const unsigned MS[] = { 1, 2, 4, 8, 32 };
	unsigned k = argc > 1 ? (unsigned)atoi(argv[1]) : 256;
	unsigned n = argc > 2 ? (unsigned)atoi(argv[2]) : 64;
	unsigned maxm = MS[sizeof(MS) / sizeof(*MS) - 1];
	struct charsiu_device *dev;
	uint8_t *A, *B;
	int32_t *got, *want;
	unsigned passed = 0, tried = 0;
	int fail = 0;

	dev = charsiu_open(NULL);
	if (!dev) {
		fprintf(stderr, "no /dev/accel/accel0\n");
		return 77;
	}
	A = malloc((size_t)maxm * k);
	B = malloc((size_t)n * k);
	got = malloc((size_t)maxm * n * 4);
	want = malloc((size_t)maxm * n * 4);
	if (!A || !B || !got || !want) {
		fprintf(stderr, "out of memory\n");
		return 1;
	}
	/*
	 * ⚠ EVERY ROW OF A MUST DIFFER, or a wide m could be right by copying
	 * one row over the rest. The stride is coprime with the row length so
	 * no two rows repeat.
	 */
	for (unsigned r = 0; r < maxm; r++)
		for (unsigned i = 0; i < k; i++)
			A[(size_t)r * k + i] = mix(r * 2u + 1u, i, 15);
	/*
	 * ⚠ EVERY OUTPUT CHANNEL MUST BE A DIFFERENT NUMBER, and the obvious
	 * B[c][i] = (c + i) % 9 is not: it has PERIOD NINE in c, so channel 9
	 * computes the same dot product as channel 0. Round 382's locator then
	 * reported the first flat index holding each value, which for a
	 * repeated value is always the earliest channel's slot -- the table
	 * looked like a layout and was an artefact of the test data. A hash of
	 * (c, i) has no period worth finding.
	 */
	for (unsigned c = 0; c < n; c++)
		for (unsigned i = 0; i < k; i++)
			B[(size_t)c * k + i] = mix(c, i, 15);

	printf("K=%u N=%u, int8 weights and activations, raw accumulator\n", k, n);
	/*
	 * ⚠ SAY WHICH GEOMETRY RAN. Round 380's "control" was typed as
	 *
	 *     CHARSIU_M_LEGACY=1
	 *     /opt/charsiu/npu_gemm_test 256 64
	 *
	 * on two lines, which sets a SHELL variable and exports nothing, so the
	 * binary never saw it and the run was a second copy of the default. The
	 * line below is the only thing that would have said so.
	 */
	printf("geometry: Mesa's generic RK3576 encoder, inw=1 inh=M%s\n\n",
	       getenv("CHARSIU_CNA_1098") ? " (CHARSIU_CNA_1098 set)" : "");

	/*
	 * ⚠ FOUR CONFIGURATIONS, AND THE AXIS IS THE INTERESTING ONE. Round
	 * 384's output map showed the second position being spent on output
	 * CHANNEL c+16 rather than row 1, and 0x4020 -- the DPU's output width
	 * -- is 0 at every M on the height axis. CHARSIU_M_AXIS=w is Mesa's
	 * same encoder with inw = ow = M, which tools/mesa_mdiff.py --axis w
	 * matches word for word.
	 */
	{
		static const struct { unsigned ape; char axis; } CFG[] = {
			{ 4, 'h' }, { 8, 'h' }, { 4, 'w' }, { 8, 'w' },
		};
		unsigned nms = sizeof(MS) / sizeof(*MS);
		unsigned best_p = 0, best_t = 0, bi = 0;

		for (unsigned c = 0; c < sizeof(CFG) / sizeof(*CFG); c++) {
			unsigned p = 0, t = 0;

			fail = sweep(dev, CFG[c].ape, CFG[c].axis, MS, nms,
				     k, n, A, B, got, want, &p, &t);
			if (!fail && t > 1) {
				printf("\n  ⚑ THIS ONE WORKS: M on the %s, "
				       "CHARSIU_ENTRY_ATOMICS=%u.\n"
				       "  %u of %u widths exact.\n"
				       "  ⚠ Decode has only ever run at M = 1 "
				       "on the height axis with 4,\n"
				       "  so check TOKENS before changing any "
				       "default.\n",
				       CFG[c].axis == 'w' ? "width" : "height",
				       CFG[c].ape, p, t);
				charsiu_close(dev);
				return 0;
			}
			if (p > best_p) { best_p = p; best_t = t; bi = c; }
		}
		printf("\n  none of the four configurations is exact; best was "
		       "%u of %u\n  (M on the %s, CHARSIU_ENTRY_ATOMICS=%u).\n",
		       best_p, best_t, CFG[bi].axis == 'w' ? "width" : "height",
		       CFG[bi].ape);
		unsetenv("CHARSIU_M_AXIS");
		setenv("CHARSIU_ENTRY_ATOMICS", "4", 1);
		passed = best_p; tried = best_t;
	}
	printf("\n  %u of %u widths exact\n", passed, tried);

	if (passed == tried && tried > 1) {
		printf("\n  M>1 COMPUTES CORRECTLY. A batched prefill has somewhere\n"
		       "  to go: at M=32 the same weight bytes serve 32 rows.\n");
		charsiu_close(dev);
		return 0;
	}

	/*
	 * Only now is the activation layout worth asking about. It was the
	 * first thing round 379 swept and the wrong first thing: at m=1 both
	 * formulas collapse to e = kk, so the control could not choose between
	 * them, and a wrong SURFACE cannot be fixed by any packing of it.
	 */
	printf("\n  m>1 is not exact. Sweeping the activation layout at m=2,\n"
	       "  which is only worth doing now that the geometry is fixed.\n");
	reference(2, k, n, A, B, want);
	{
		static const struct { const char *lay, *gran; } cand[] = {
			{ "0", NULL }, { "0", "1" }, { "0", "2" }, { "0", "4" },
			{ "0", "8" }, { "0", "16" }, { "0", "32" }, { "0", "64" },
			{ "2", NULL },
		};
		unsigned best = 0, bi = 0;

		for (unsigned c = 0; c < sizeof(cand) / sizeof(*cand); c++) {
			unsigned ok = 0;
			char name[32];

			setenv("CHARSIU_A_LAYOUT", cand[c].lay, 1);
			if (cand[c].gran)
				setenv("CHARSIU_A_GRAN", cand[c].gran, 1);
			else
				unsetenv("CHARSIU_A_GRAN");
			snprintf(name, sizeof(name), "lay=%s gran=%s",
				 cand[c].lay, cand[c].gran ? cand[c].gran : "atom");

			if (run(dev, 2, k, n, A, B, got)) {
				printf("    %-18s submit failed\n", name);
				continue;
			}
			for (size_t i = 0; i < (size_t)2 * n; i++)
				if (got[i] == want[i]) ok++;
			printf("    %-18s %4u of %4u correct%s\n", name, ok, 2 * n,
			       ok == 2 * n ? "   <== THIS ONE" : "");
			if (ok > best) { best = ok; bi = c; }
		}
		unsetenv("CHARSIU_A_LAYOUT");
		unsetenv("CHARSIU_A_GRAN");

		if (best == 2 * n) {
			printf("\n  m=2 is correct with lay=%s gran=%s -- the geometry\n"
			       "  needed the layout as well. Re-run the sweep at m=4\n"
			       "  and m=32 before trusting it.\n",
			       cand[bi].lay, cand[bi].gran ? cand[bi].gran : "atom");
			charsiu_close(dev);
			return 1;
		}
		printf("\n  best was %u of %u; no candidate layout makes m=2 correct\n"
		       "  either, so the surface is still not described right.\n",
		       best, 2 * n);
		run(dev, 2, k, n, A, B, got);
		layouts(2, n, got, want);
		locate(2, n, got, want);
	}

	/*
	 * ⚠ AND THE SAME COUNT AT m=4, because two widths ruled out N and two
	 * K values have now ruled out K.
	 *
	 * K=256 has surf 4 and gave a row stride of 20, which is 5 * surf and
	 * looked like an answer. K=512 has surf 8 and gave 20 again -- a ratio
	 * of 2.5, so the stride follows neither N nor K nor the input slice.
	 * What is left that it could be computed from is m itself, or a
	 * literal. One more count says which, and a third constraint makes the
	 * search over-determined rather than merely unique.
	 */
	/*
	 * ⚠⚠ THE SERIES IN m, WHICH IS WHAT THE COUNTS ACTUALLY FIT.
	 *
	 *     N=64 m=2   92 of 128     N=32 m=2   52 of  64
	 *     N=64 m=4  148 of 256
	 *
	 * written = N + inc * (m - 1), with inc = 28 at N=64 and 20 at N=32.
	 * Three points, exact, and the m part is exact on three of them: the
	 * FIRST row gets all N words and every row after it gets 28, not
	 * another 64.
	 *
	 * That is not a stride. A stride puts values in the wrong place and
	 * this leaves them uncomputed; a shape like "one row whole and the
	 * rest a fixed share" is a BUDGET being divided. Mesa bounds exactly
	 * this -- rkt_task.c splits a task's staged rows when
	 * (cbuf_rows + staged) * entries_per_slice > total_entries -- and
	 * charsiu submits all m rows as one task and never checks.
	 *
	 * So extend the series. If it stays linear the increment is the number
	 * to explain; if it flattens, something saturates and the cap is the
	 * number. m=8 predicts 260 of 512 on the linear reading.
	 */
	for (unsigned mm = 2; mm <= 8 && n >= 16; mm *= 2) {
		size_t tot = (size_t)mm * n, livem = 0;

		reference(mm, k, n, A, B, want);
		if (run(dev, mm, k, n, A, B, got))
			continue;
		for (size_t i = 0; i < tot; i++)
			for (size_t q = 0; q < tot; q++)
				if (want[q] == got[i]) { livem++; break; }
		printf("  at N=%u, m=%-2u: wrote %4zu of %4zu, so %4.0f words"
		       " a row after the first\n", n, mm, livem, tot,
		       mm > 1 ? (double)(livem - n) / (mm - 1) : 0.0);
		if (mm == 4)
			g_live_at_m4 = livem;
	}
	if (n >= 16) {
		unsigned n2 = n / 2;
		size_t tot2 = (size_t)2 * n2, live = 0;

		reference(2, k, n2, A, B, want);
		if (!run(dev, 2, k, n2, A, B, got)) {
			for (size_t i = 0; i < tot2; i++)
				for (size_t q = 0; q < tot2; q++)
					if (want[q] == got[i]) { live++; break; }
			printf("  at N=%u, m=2 : wrote %4zu of %4zu"
			       " (%.0f%%)\n", n2, live, tot2,
			       100.0 * (double)live / (double)tot2);
			printf("  against N=%u: the same COUNT would be a"
			       " capacity, the same PERCENTAGE a stride.\n", n);

			/*
			 * ⚠⚠ AND SOLVE FOR THE STRIDES, because two counts
			 * pin them. The address function fitted to the r385
			 * map, 80 of 80 cells:
			 *
			 *   flat = S*(u/2) + R*r + 8*(v/4) + 4*(u%2) + (v%4)
			 *
			 * with c = 16u + v. Its RANGE is what decides how many
			 * words the board writes, so a search over R and S for
			 * the pair that reproduces both counts is a
			 * measurement rather than a fit. Offline on the r387
			 * numbers -- 92 at N=64 and 52 at N=32 -- exactly one
			 * pair survives: R=20, S=40, where an injective
			 * surface needs 32 and 64 at N=64.
			 *
			 * A stride that does not move with N is set from
			 * something that is not N, which is the whole finding.
			 */
			{
				unsigned hits = 0, fr = 0, fs = 0;

				for (unsigned R = 1; R < 256; R++)
					for (unsigned S = 1; S < 512; S++) {
						if (addr_range(n, 2, R, S) != g_live_at_n ||
						    addr_range(n2, 2, R, S) != live)
							continue;
						/* the third constraint, when
						 * the m=4 run produced one */
						if (g_live_at_m4 &&
						    addr_range(n, 4, R, S) != g_live_at_m4)
							continue;
						hits++; fr = R; fs = S;
					}
				if (hits == 1) {
					struct charsiu_matmul sm = {
						1, k, n, CHARSIU_INT8,
						CHARSIU_INT8
					};
					unsigned surf =
						charsiu_entries_per_row(&sm);

					printf("\n  ⚑ exactly one stride pair"
					       " reproduces both counts:\n"
					       "  row stride %u, block stride"
					       " %u -- an injective surface"
					       " needs %u and %u at N=%u.\n",
					       fr, fs, 2 * n / 4, 4 * n / 4, n);
					/*
					 * ⚠ AND surf BESIDE IT, because the
					 * whole question is what the wrong
					 * stride was computed FROM. It does
					 * not move with N -- two widths said
					 * so -- and surf is the obvious thing
					 * that does not either: it depends on
					 * K alone. At K=256 surf is 4 and the
					 * row stride is 20, which is 5 * surf.
					 * Running this at another K is the
					 * one-line test of that: if the stride
					 * follows surf it is computed from the
					 * INPUT slice, and if it stays at 20
					 * it is a constant.
					 */
					printf("  surf at K=%u is %u, so the"
					       " row stride is %.2f * surf;"
					       " re-run at\n  another K to see"
					       " whether that ratio holds.\n",
					       k, surf,
					       surf ? (double)fr / surf : 0.0);
				}
				else
					printf("\n  %u stride pairs fit all"
					       " the counts.\n"
					       "  ⚠ The function that fits the"
					       " m=2 MAP exactly -- 80 of 80\n"
					       "  cells, R=20 S=40 -- predicts"
					       " %zu words at m=4 and the board\n"
					       "  wrote %zu. So the layout"
					       " depends on m in a way one map\n"
					       "  at one m cannot show, and"
					       " counts are too weak to fit it.\n"
					       "  The map at m=4 follows.\n",
					       hits, addr_range(n, 4, 20, 40),
					       g_live_at_m4);
			}
		}
	}

	/*
	 * ⚠ THE MAP AT m=4 AS WELL, because three counts turned out to be too
	 * weak to fit what one map pinned exactly.
	 *
	 * At m=2 the address function is unique: searching a four parameter
	 * family against the 80 unique cells leaves one, R=20 S=40 with the
	 * channel group at 16. It reproduces both m=2 counts and then predicts
	 * 132 words at m=4 where the board wrote 148. A model that fits one
	 * width perfectly and misses the next is a model of that width, so the
	 * m dependence has to be read rather than extrapolated.
	 *
	 * ⚠ And the layout is not the critical path anyway. At m=2 thirty six
	 * reference values were never computed and at m=4 it is a hundred and
	 * eight, so no permutation recovers them; what the second map is for
	 * is which ones are missing and in what pattern.
	 */
	if (n >= 16 && !getenv("CHARSIU_NO_M4_MAP")) {
		reference(4, k, n, A, B, want);
		if (!run(dev, 4, k, n, A, B, got)) {
			printf("\n  ===== the same, at m=4 =====\n");
			locate(4, n, got, want);
		}
	}

	printf("\n  m>1 is NOT usable as it stands: a batched prefill would be wrong.\n");
	charsiu_close(dev);
	return 1;
}
