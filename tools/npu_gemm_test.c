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
	printf("\n  where each wanted value landed  (flat index in the output)\n");
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
static int sweep(struct charsiu_device *dev, unsigned ape,
		 const unsigned *MS, unsigned nms, unsigned k, unsigned n,
		 const uint8_t *A, const uint8_t *B,
		 int32_t *got, int32_t *want, unsigned *passed, unsigned *tried)
{
	char apes[8];
	int fail = 0;

	snprintf(apes, sizeof(apes), "%u", ape);
	setenv("CHARSIU_ENTRY_ATOMICS", apes, 1);
	printf("\n  CHARSIU_ENTRY_ATOMICS=%u  (%s)\n", ape,
	       ape == 8 ? "Mesa's constant" : "this tree's, and every result so far");

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
			A[(size_t)r * k + i] =
				(uint8_t)(128 + (int)((i + 3 * r) % 7) - 3
					  + (int)(r % 3) - 1);
	for (unsigned c = 0; c < n; c++)
		for (unsigned i = 0; i < k; i++)
			B[(size_t)c * k + i] = (uint8_t)(128 + (int)((c + i) % 9) - 4);

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

	{
		unsigned nms = sizeof(MS) / sizeof(*MS);

		fail = sweep(dev, 4, MS, nms, k, n, A, B, got, want,
			     &passed, &tried);
		if (fail && tried > 1) {
			unsigned p8 = 0, t8 = 0;
			int f8 = sweep(dev, 8, MS, nms, k, n, A, B, got, want,
				       &p8, &t8);

			if (!f8) {
				printf("\n  ⚑ MESA'S CONSTANT IS THE ONE.\n"
				       "  %u of %u widths exact at "
				       "CHARSIU_ENTRY_ATOMICS=8, and %u of %u "
				       "at 4.\n"
				       "  Decode has only ever run with 4, so "
				       "check tokens before\n"
				       "  changing the default: "
				       "CHARSIU_ENTRY_ATOMICS=8 charsiu run ...\n",
				       p8, t8, passed, tried);
				charsiu_close(dev);
				return 0;
			}
			printf("\n  8 does not fix it either "
			       "(%u of %u exact). The geometry now\n"
			       "  matches Mesa word for word, so what is left "
			       "is not geometry.\n", p8, t8);
			passed = p8; tried = t8;
		}
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

	printf("\n  m>1 is NOT usable as it stands: a batched prefill would be wrong.\n");
	charsiu_close(dev);
	return 1;
}
