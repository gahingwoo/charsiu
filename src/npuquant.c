// SPDX-License-Identifier: GPL-2.0-or-later
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */

/*
 * The NPU's number format, computed on the CPU.
 *
 * Step 2 of the plan moves a projection onto the NPU. Before any of that is
 * worth writing, one question has to be answered and it can be answered here:
 * does the format the hardware imposes -- signed int8 weights with one scale
 * per output channel, against an activation with ONE scale for the whole
 * vector -- cost the model its output?
 *
 * The CPU decode loop is the oracle, so the way to ask is to run the whole
 * model in exactly that format and compare the tokens.
 *
 * It is deliberately NOT fast. It holds a second copy of every routed tensor in
 * int8 and does a plain widening dot. What it produces is an accuracy answer
 * and the exact buffers a converter will later write to a file: q, scale, and
 * the per channel weight sum the coefficient buffer needs.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu_llm.h"

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
#include <arm_neon.h>
#endif


/*
 * THE WEIGHT CACHE: slow once, fast after, and the user converts nothing.
 *
 * charsiu dequantises 1.24 billion Q8_0 weights to float and re-quantises them
 * at EVERY start, which measured 10 s of pure CPU on the board after round
 * 356's one pass fix and buys nothing the second time. The vendor is fast for
 * exactly one reason: a .rkllm holds weights that are already quantised, so
 * their runtime mmaps and DMAs. This does the same thing for a gguf without
 * asking anyone to convert a model -- the first run writes a sidecar and every
 * run after it reads one.
 *
 * On the SSD this is meant for, 620 MB reads in about a second against 10 s of
 * quantising. On the test SD card it is closer to a wash, which is why round
 * 356 measures the card as well.
 *
 * ⚠ IT MUST NEVER SILENTLY DISAGREE WITH THE CODE. The header carries a format
 * version, the quantiser's own version, the bit width, the group size and a
 * stamp of the model file, and every record re-checks n, k and the group count.
 * Anything that does not match rebuilds from scratch rather than being patched
 * up: a cache that is subtly wrong is worse than no cache, because the failure
 * shows up as a slightly wrong sentence rather than as an error.
 *
 * The nibbles are packed two to a byte on the way out, so an int4 cache is half
 * the size of the codes in memory and half the size of the gguf it replaces.
 */
/* one getenv, cached: this is asked inside loops over the whole tensor */
static int midrise_grid(void)
{
	static int v = -1;

	if (v < 0)
		v = getenv("CHARSIU_NPU_W4_MIDRISE") != NULL;
	return v;
}

#define WCACHE_MAGIC  0x43535743u        /* "CSWC" */
#define WCACHE_FORMAT 1u
/* ⚠ BUMP THIS whenever the quantiser's arithmetic changes, or an old cache
 * will quietly feed the new code the old numbers. */
#define WCACHE_QUANT  2u

struct wcache_head {
	uint32_t magic, format, quant, bits;
	uint64_t group;
	char stamp[64];
};

static struct {
	FILE *f;
	int writing;
	int checked;
} wc;

static void wcache_setup(unsigned bits, uint64_t grp)
{
	const char *path = getenv("CHARSIU_NPU_CACHE");
	const char *stamp = getenv("CHARSIU_NPU_CACHE_STAMP");
	struct wcache_head h, want;

	if (wc.checked)
		return;
	wc.checked = 1;
	if (!path || !*path)
		return;

	memset(&want, 0, sizeof(want));
	want.magic = WCACHE_MAGIC;
	want.format = WCACHE_FORMAT;
	want.quant = WCACHE_QUANT;
	want.bits = bits;
	/* the grid is part of what the codes mean, so it is part of the key */
	want.group = grp | (midrise_grid() ? (1ull << 63) : 0);
	snprintf(want.stamp, sizeof(want.stamp), "%s", stamp ? stamp : "");

	wc.f = fopen(path, "rb");
	if (wc.f) {
		if (fread(&h, sizeof(h), 1, wc.f) == 1 &&
		    !memcmp(&h, &want, sizeof(h))) {
			fprintf(stderr, "charsiu: weight cache %s, reading\n",
				path);
			return;
		}
		fclose(wc.f);
		wc.f = NULL;
		fprintf(stderr, "charsiu: weight cache %s does not match this "
			"model or this quantiser, rebuilding\n", path);
	}
	wc.f = fopen(path, "wb");
	if (!wc.f) {
		fprintf(stderr, "charsiu: cannot write the weight cache %s\n",
			path);
		return;
	}
	if (fwrite(&want, sizeof(want), 1, wc.f) != 1) {
		fclose(wc.f);
		wc.f = NULL;
		return;
	}
	wc.writing = 1;
	fprintf(stderr, "charsiu: weight cache %s, writing\n", path);
}

/* one record's payload size, nibbles packed two to a byte for int4 */
static size_t wcache_qbytes(unsigned bits, uint64_t n, uint64_t k)
{
	return bits == 4 ? (size_t)((n * k + 1) / 2) : (size_t)(n * k);
}

static int wcache_read(struct npu_tensor *t, unsigned bits, const char *name)
{
	char nm[80];
	uint64_t n, k, ngrp;
	size_t qb;

	if (!wc.f || wc.writing)
		return 0;
	if (fread(nm, 1, sizeof(nm), wc.f) != sizeof(nm) ||
	    fread(&n, sizeof(n), 1, wc.f) != 1 ||
	    fread(&k, sizeof(k), 1, wc.f) != 1 ||
	    fread(&ngrp, sizeof(ngrp), 1, wc.f) != 1)
		return 0;
	/*
	 * The records are written in the order the tensors are built and are
	 * read back in the same order, so a mismatch means the two runs did not
	 * ask for the same thing. Stop using the cache rather than hunt for the
	 * record: an out of order cache is a bug, not a case to handle.
	 */
	if (strcmp(nm, name) || n != t->n || k != t->k ||
	    ngrp != (t->kgroup ? (t->k + t->kgroup - 1) / t->kgroup : 1)) {
		fprintf(stderr, "charsiu: weight cache is out of step at %s, "
			"ignoring the rest of it\n", name);
		fclose(wc.f);
		wc.f = NULL;
		return 0;
	}
	qb = wcache_qbytes(bits, n, k);
	if (bits == 4) {
		uint8_t *packed = malloc(qb);
		size_t i;

		if (!packed)
			return 0;
		if (fread(packed, 1, qb, wc.f) != qb ||
		    fread(t->scale, sizeof(float), (size_t)(n * ngrp), wc.f)
			    != (size_t)(n * ngrp)) {
			free(packed);
			return 0;
		}
		for (i = 0; i < (size_t)(n * k); i++) {
			int v = (i & 1) ? (packed[i >> 1] >> 4)
					: (packed[i >> 1] & 0xf);

			t->q[i] = (int8_t)(v < 8 ? v : v - 16);
		}
		free(packed);
	} else {
		if (fread(t->q, 1, qb, wc.f) != qb ||
		    fread(t->scale, sizeof(float), (size_t)(n * ngrp), wc.f)
			    != (size_t)(n * ngrp))
			return 0;
	}
	/* wsum is n int32 and recomputing it is a whole pass over the codes */
	if (fread(t->wsum, sizeof(int32_t), (size_t)n, wc.f) != (size_t)n)
		return 0;
	return 1;
}

static void wcache_write(const struct npu_tensor *t, unsigned bits,
			 const char *name)
{
	char nm[80];
	uint64_t n = t->n, k = t->k;
	uint64_t ngrp = t->kgroup ? (k + t->kgroup - 1) / t->kgroup : 1;
	size_t qb = wcache_qbytes(bits, n, k);

	if (!wc.f || !wc.writing)
		return;
	memset(nm, 0, sizeof(nm));
	snprintf(nm, sizeof(nm), "%s", name);
	if (fwrite(nm, 1, sizeof(nm), wc.f) != sizeof(nm) ||
	    fwrite(&n, sizeof(n), 1, wc.f) != 1 ||
	    fwrite(&k, sizeof(k), 1, wc.f) != 1 ||
	    fwrite(&ngrp, sizeof(ngrp), 1, wc.f) != 1)
		goto bad;
	if (bits == 4) {
		uint8_t *packed = calloc(qb, 1);
		size_t i;

		if (!packed)
			goto bad;
		for (i = 0; i < (size_t)(n * k); i++) {
			uint8_t v = (uint8_t)(t->q[i] & 0xf);

			if (i & 1)
				packed[i >> 1] |= (uint8_t)(v << 4);
			else
				packed[i >> 1] |= v;
		}
		if (fwrite(packed, 1, qb, wc.f) != qb) {
			free(packed);
			goto bad;
		}
		free(packed);
	} else if (fwrite(t->q, 1, qb, wc.f) != qb) {
		goto bad;
	}
	if (fwrite(t->scale, sizeof(float), (size_t)(n * ngrp), wc.f)
	    != (size_t)(n * ngrp) ||
	    fwrite(t->wsum, sizeof(int32_t), (size_t)n, wc.f) != (size_t)n)
		goto bad;
	return;
bad:
	/* a truncated cache would be read back as garbage next time, and the
	 * header would still match. Drop it. */
	fprintf(stderr, "charsiu: the weight cache could not be written; "
		"removing it\n");
	fclose(wc.f);
	wc.f = NULL;
	wc.writing = 0;
	if (getenv("CHARSIU_NPU_CACHE"))
		remove(getenv("CHARSIU_NPU_CACHE"));
}

/*
 * ONE RANGE OF OUTPUT CHANNELS. Rows do not talk to each other: each writes
 * its own k bytes of q, its own ngrp scales and its own wsum, and reads the
 * weight file through gguf_row_f32, which only ever writes the buffer it is
 * handed. So the whole of quantisation splits over the pool that the decode
 * already has -- and a cold start was 7.9 seconds of it on the board, on one
 * core, while three sat idle.
 *
 * ⚠ THE ONE THING THAT DOES CROSS ROWS is the squared error the --info
 * diagnostic accumulates, so a run that asks for it stays serial rather than
 * growing a lock for a number nothing in the decode reads.
 */
struct qrows {
	struct npu_tensor *t;
	const struct gguf_tensor *w;
	uint64_t k, ngrp, grp;
	unsigned bits;
	float qmax;
	int w4sym, w4clip, rms, midrise;
	double se, sw;
};

static void quant_rows(void *vc, uint64_t r0, uint64_t nr)
{
	struct qrows *c = vc;
	struct npu_tensor *t = c->t;
	const struct gguf_tensor *w = c->w;
	const uint64_t k = c->k, ngrp = c->ngrp, grp = c->grp;
	const unsigned bits = c->bits;
	const float qmax = c->qmax;
	const int w4sym = c->w4sym, w4clip = c->w4clip;
	const int rms = c->rms, midrise = c->midrise;
	double se = 0.0, sw = 0.0;
	float *row = malloc((size_t)k * sizeof(float));

	if (!row)
		return;
	for (uint64_t r = r0; r < r0 + nr; r++) {
		int8_t *dst = t->q + r * k;
		int32_t sum = 0;

		gguf_row_f32(w, r, row);
		if (t->kscale)
			for (uint64_t i = 0; i < k; i++)
				row[i] /= t->kscale[i];
		for (uint64_t g = 0; g < ngrp; g++) {
		uint64_t lo = g * grp, hi = lo + grp < k ? lo + grp : k;
		float amax = 0.0f, vmax = 0.0f, d, id;

		/*
		 * ONE PASS, and it carries the SIGNED extreme with it.
		 *
		 * The int4 branch below used to walk the group a SECOND time to
		 * find the signed value of largest magnitude, calling fabsf on
		 * the running maximum every iteration as well. That second pass
		 * is the whole of int4's extra load cost: 3.37 ns a weight
		 * against int8's 2.00, measured over 440 million weights of the
		 * real model with three repeats.
		 */
		for (uint64_t i = lo; i < hi; i++) {
			float a = fabsf(row[i]);

			if (a > amax) {
				amax = a;
				vmax = row[i];
			}
		}
		/*
		 * SYMMETRIC, and 127 rather than 128: the hardware's operand is
		 * a byte biased by 128, so -128 has no positive partner and a
		 * scale built on it would make the largest weight unreachable in
		 * one direction.
		 */
		/*
		 * ⚠ USE ALL SIXTEEN LEVELS. amax/7 is symmetric and throws away
		 * -8, which is one level of the sixteen a nibble has -- and
		 * q4_0, the yardstick this is measured against, takes the
		 * element of largest magnitude and divides by -8 so the range is
		 * [-8, 7]. Round 335's first int4 numbers were a fifteen level
		 * quantiser being compared against a sixteen level one.
		 * CHARSIU_NPU_W4_SYM restores the symmetric version.
		 */
		if (bits == 4 && midrise) {
			/*
			 * THE VENDOR'S GRID, and it has no code for zero.
			 *
			 * Their weight bytes, read straight out of the capture,
			 * are symmetric under u -> ~u: counts 232704 at code 0
			 * against 231852 at code 15, 152658 at 1 against 156566
			 * at 14, all the way in. Read as two's complement s the
			 * mean is -0.5; read as s + 0.5 it is -0.026, and three
			 * separate megabytes agree. So their levels are
			 * +-0.5, +-1.5 ... +-7.5 times a step: sixteen levels
			 * used symmetrically, with no code spent on an exact
			 * zero that a Gaussian almost never lands on.
			 *
			 * w = (s + 0.5) * d, so d is the largest magnitude over
			 * 7.5 and s = w/d - 0.5 rounded. The hardware computes
			 * sum(s * a) either way; the half step becomes
			 * 0.5 * d * sum(a), one number a channel a token, which
			 * npudev adds at the accumulate.
			 *
			 * ⛔ AND IT MEASURED WORSE, so it is OFF by default and
			 * kept only as the record of a refuted idea. On the
			 * host with the real model:
			 *
			 *   RTN     g2048  "Paris is the most populous city...
			 *                   most famous... most beautiful"
			 *   MIDRISE g2048  "a property property one time
			 *                   around. I can make it happen"
			 *   RTN     g1024  "The Eiffel Tower... The Louvre
			 *                   Museum... The Mona Lisa"
			 *   MIDRISE g1024  "the total number of employees of
			 *                   the company is the total number"
			 *
			 * ⚠ AND THE NEGATIVE WAS PREDICTABLE, which is the part
			 * worth remembering. A grid with no zero forces every
			 * near zero weight to +-0.5d, and network weights are
			 * strongly peaked at zero, so midtread beats midrise on
			 * exactly this kind of distribution. I reasoned from
			 * "the vendor's histogram is symmetric, so their grid
			 * must be better" and had the implication backwards.
			 *
			 * Their histogram is still symmetric about the 15/0
			 * boundary. That is as consistent with a CALIBRATED
			 * quantiser whose codes come out shifted as it is with
			 * a midrise grid, and the calibrated reading is the one
			 * that also explains their quality at one scale a row.
			 */
			d = amax / 7.5f;
		} else if (bits == 4 && !w4sym) {
			d = vmax / -8.0f;
			/*
			 * ⚠ absmax IS THE WRONG SCALE FOR FOUR BITS, and it is
			 * the cheapest thing to fix. One outlier in two thousand
			 * weights sets the step for all of them, so every other
			 * weight rounds into a grid that is far too coarse.
			 *
			 * Search the clip instead: try the scale shrunk by a
			 * factor and keep whichever minimises the squared error
			 * of the row. Nothing is stored -- the chosen d is the
			 * per channel scale that would have been stored anyway
			 * -- and nothing changes on the hardware. This is the
			 * cheap half of what a calibrating quantiser does.
			 *
			 * ⚠ MEASURED AND IT IS WORSE: KL went 0.0989 to 0.2084
			 * and 0.3660 to 0.5535 on two prompts. Minimising the
			 * WEIGHT error clips exactly the large weights that
			 * carry the output, which is why AWQ and GPTQ optimise
			 * the OUTPUT error against calibration activations
			 * instead. Off by default, kept as the control that
			 * says a weight space objective is the wrong one.
			 */
			if (w4clip) {
				double bestе = -1.0;
				float bestd = d;
				int ci;

				for (ci = 0; ci <= 20; ci++) {
					float dc = d * (1.0f - 0.025f * ci);
					double err = 0.0;

					if (dc == 0.0f)
						continue;
					for (uint64_t i = lo; i < hi; i++) {
						int v = (int)lrintf(row[i] / dc);
						double e;

						if (v > 7) v = 7;
						if (v < -8) v = -8;
						e = (double)row[i] - (double)v * dc;
						err += e * e;
					}
					if (bestе < 0.0 || err < bestе) {
						bestе = err;
						bestd = dc;
					}
				}
				d = bestd;
			}
		} else {
			d = amax / qmax;
		}
		id = d != 0.0f ? 1.0f / d : 0.0f;
		t->scale[r * ngrp + g] = d;

		for (uint64_t i = lo; i < hi; i++) {
			int v = (int)lrintf(midrise && bits == 4
					    ? row[i] * id - 0.5f
					    : row[i] * id);

			if (bits == 4 && (!w4sym || midrise)) {
				if (v > 7) v = 7;
				if (v < -8) v = -8;
			} else {
				if (v > (int)qmax) v = (int)qmax;
				if (v < -(int)qmax) v = -(int)qmax;
			}
			dst[i] = (int8_t)v;
			sum += v;
			/*
			 * ⚠ A DIAGNOSTIC, AND IT COSTS ABOUT 3%. I guessed a
			 * third before measuring it, and three repeats on the
			 * real model say 3.37 ns a weight against 3.46. Two
			 * doubles multiplied and accumulated for every
			 * weight in the model, 1.24 billion times, to produce
			 * ONE number per tensor that nothing in the decode
			 * reads: t->rms_rel, which is printed by --info.
			 * CHARSIU_NPU_RMS turns it back on for the rounds that
			 * want it.
			 */
			if (rms) {
				double e = (double)row[i]
					 - ((double)v + (midrise && bits == 4
							 ? 0.5 : 0.0)) * d;

				se += e * e;
				sw += (double)row[i] * (double)row[i];
			}
		}
		}
		t->wsum[r] = sum;
	}
	c->se += se;
	c->sw += sw;
	free(row);
}

int npu_tensor_build(struct npu_tensor *t, const struct gguf_tensor *w)
{
	uint64_t n = w->ne[1], k = w->ne[0];
	float *row;
	double se = 0.0, sw = 0.0;

	/*
	 * ⚠ THE ACCURACY QUESTION int4 HAS TO ANSWER BEFORE IT IS WORTH WIRING
	 * IN. The hardware's coefficient buffer carries ONE multiplier per
	 * output channel, so charsiu's NPU weights are quantised per channel --
	 * fine at eight bits, and q4_0 uses a scale every 32 weights precisely
	 * because four bits per channel is not.
	 *
	 * But K is sliced anyway (KMAX), and acc_out makes a K split exact, so
	 * a slice can carry its own scale for free. CHARSIU_NPU_W4_GROUP asks
	 * what that buys: the weights are quantised in groups of that many k,
	 * with a scale each, which is what the runtime would actually get.
	 * CHARSIU_NPU_W4 alone is one scale for the whole row, the worst case.
	 */
	/*
	 * CHARSIU_NPU_W4_ONLY narrows int4 to the tensors whose name contains a
	 * substring, so "ffn" puts the feed forward -- which is 65% of the
	 * bytes a token moves -- at four bits and leaves attention at eight.
	 * Most of the saving for a fraction of the error, if the error turns
	 * out to live in attention.
	 */
	const char *w4only = getenv("CHARSIU_NPU_W4_ONLY");
	/* CHARSIU_NPU_W4V, the int4 DECODE path, implies int4 weights: one
	 * switch cannot select the layout and registers while another leaves
	 * the quantiser at eight bits. */
	unsigned bits = (getenv("CHARSIU_NPU_W4") || getenv("CHARSIU_NPU_W4V"))
		&& (!w4only || strstr(w->name, w4only)) ? 4 : 8;
	uint64_t grp = getenv("CHARSIU_NPU_W4_GROUP")
		? (uint64_t)atoi(getenv("CHARSIU_NPU_W4_GROUP")) : k;
	/*
	 * ⚠⚠ READ ONCE. These two sat inside loops over every weight in the
	 * tensor, and getenv walks the environment with a strcmp per entry.
	 * Round 354's heartbeat split settled where 114 to 144 seconds of
	 * staging went: 4 to 5 s inside charsiu_npu_add and ALL THE REST in
	 * npu_tensor_build. int8 never noticed because "bits == 4 &&" short
	 * circuits before the call, which is also why it looked like an int4
	 * hardware or layout problem for two rounds. It was 1.24 billion
	 * getenv calls.
	 */
	const int w4sym = getenv("CHARSIU_NPU_W4_SYM") != NULL;
	const int w4clip = getenv("CHARSIU_NPU_W4_CLIP") != NULL;
	const int rms = getenv("CHARSIU_NPU_RMS") != NULL;
	const int midrise = midrise_grid();
	uint64_t ngrp;
	float qmax = bits == 4 ? 7.0f : 127.0f;

	if (grp == 0 || grp > k)
		grp = k;
	/*
	 * ⚠⚠ A PARTIAL LAST GROUP IS QUANTISED HERE AND CONSUMED AS THOUGH IT
	 * DID NOT EXIST. npudev's tensor_grouped() requires k % kgroup == 0, so
	 * a tensor with a remainder is treated as UNGROUPED and its scales are
	 * read as scale[row] -- but this had already written them as
	 * scale[row * ngrp + group]. Every row then gets some other row's
	 * scale.
	 *
	 * It never showed up because every llama dimension is a power of two
	 * and divides the 1024 slice exactly. Qwen2.5-1.5B is n_embd 1536 and
	 * n_ff 8960, neither of which does, and on the board it decoded
	 * "otasiculoshci.syič希opol staticollect..." while the same file on the
	 * CPU answered properly.
	 *
	 * One scale a row is what the consumer will apply, so it is what gets
	 * written. Coarser for these shapes, and correct.
	 */
	if (k % grp)
		grp = k;
	ngrp = (k + grp - 1) / grp;

	/*
	 * ⚠ THE k FACTOR, and why it is free.
	 *
	 * charsiu reads the RAW int32 accumulator (acc_out), so the hardware's
	 * per channel multiplier is not in the path at all: the dequantise
	 * happens on the CPU. The only real constraint is that one job's
	 * accumulator sums all of K under one scale a channel.
	 *
	 * But a factor that depends on k ALONE cancels inside the product. Put
	 * c_k into the weights as a divide and into the activation as a
	 * multiply and the accumulator is unchanged, while the weights lose
	 * whatever part of their spread across k is common to every channel --
	 * which is the part one scale a row cannot cover. That is the AWQ and
	 * SmoothQuant trick, and here it costs one multiply a k on a vector of
	 * 2048 and nothing on the hardware.
	 *
	 * CHARSIU_NPU_AWQ is the exponent, 0 for off and 0.5 for the usual
	 * square root balance.
	 */
	double alpha = getenv("CHARSIU_NPU_AWQ")
		? atof(getenv("CHARSIU_NPU_AWQ")) : 0.0;

	memset(t, 0, sizeof(*t));
	snprintf(t->name, sizeof(t->name), "%s", w->name);
	t->n = n;
	t->k = k;
	t->kgroup = grp;
	t->q = malloc((size_t)n * k);
	t->scale = malloc((size_t)n * ngrp * sizeof(float));
	t->wsum = malloc((size_t)n * sizeof(int32_t));
	row = malloc((size_t)k * sizeof(float));
	if (!t->q || !t->scale || !t->wsum || !row) {
		/*
		 * ⚠ SAY WHICH TENSOR AND HOW MUCH. This returns -1 into a
		 * caller that returns NULL into a matvec that quietly runs on
		 * the CPU, so a tensor whose quantised copy would not fit
		 * looked exactly like a tensor nobody had asked to route.
		 *
		 * The copy is one BYTE a weight whatever the bit width -- the
		 * nibbles are packed later, in npudev -- so an output head is
		 * 302 MB here even at four bits.
		 */
		fprintf(stderr, "charsiu: %s stays on the CPU -- its %llu x %llu "
			"quantised copy needs %.0f MB and would not allocate\n",
			w->name, (unsigned long long)n, (unsigned long long)k,
			(double)((size_t)n * k) / 1e6);
		free(row);
		npu_tensor_free(t);
		return -1;
	}

	/*
	 * ⚠ THE CACHE IS SKIPPED WHENEVER SOMETHING ELSE DECIDES THE WEIGHTS.
	 * AWQ folds a per k factor into them and keeps it in t->kscale, and
	 * CHARSIU_W4_FILE replaces them outright; neither is in the record, so
	 * caching either would store weights that cannot be reproduced from
	 * what the header claims.
	 */
	if (alpha == 0.0 && !getenv("CHARSIU_W4_FILE")) {
		wcache_setup(bits, grp);
		if (wcache_read(t, bits, w->name)) {
			free(row);
			return 0;
		}
	}

	if (alpha != 0.0) {
		double *col = calloc(k, sizeof(*col));
		double gm = 0.0;
		const char *sf = getenv("CHARSIU_AWQ_STATS");
		int got = 0;

		if (!col) { free(row); npu_tensor_free(t); return -1; }
		if (sf) {
			FILE *f = fopen(sf, "rb");
			char nm[80];
			uint64_t kk;

			while (f && fread(nm, 1, sizeof(nm), f) == sizeof(nm)
			       && fread(&kk, sizeof(kk), 1, f) == 1) {
				if (!strcmp(nm, w->name) && kk == k) {
					got = fread(col, sizeof(*col), k, f) == k;
					break;
				}
				fseek(f, (long)(kk * sizeof(double)), SEEK_CUR);
			}
			if (f)
				fclose(f);
		}
		if (!got)
			for (uint64_t r = 0; r < n; r++) {
				gguf_row_f32(w, r, row);
				for (uint64_t i = 0; i < k; i++)
					col[i] += fabs((double)row[i]);
			}
		t->kscale = malloc((size_t)k * sizeof(float));
		if (!t->kscale) { free(col); free(row); npu_tensor_free(t); return -1; }
		/*
		 * ⚠ FLOOR THE STATISTIC AND CLAMP THE FACTOR. The first version
		 * did neither and produced KL of 8 to 12, which is not a method
		 * failing, it is a divide by nearly zero: a k whose mean |x| is
		 * tiny gets a tiny factor, the weights of that column are
		 * divided by it, and every one of them clips to the rail. The
		 * floor is relative to the mean so it scales with the tensor,
		 * and the clamp is what AWQ does for the same reason.
		 */
		{
			double mean = 0.0;

			for (uint64_t i = 0; i < k; i++)
				mean += col[i];
			mean /= (double)n * (double)k;
			for (uint64_t i = 0; i < k; i++) {
				double v = col[i] / (double)n;

				if (v < 1e-3 * mean)
					v = 1e-3 * mean;
				col[i] = pow(v, alpha);
				gm += log(col[i]);
			}
		}
		gm = exp(gm / (double)k);            /* keep the mean factor at 1 */
		for (uint64_t i = 0; i < k; i++) {
			double f = col[i] / gm;

			if (f < 0.125) f = 0.125;
			if (f > 8.0) f = 8.0;
			t->kscale[i] = (float)f;
		}
		free(col);
	}

	{
		struct qrows c = { t, w, k, ngrp, grp, bits, qmax,
				   w4sym, w4clip, rms, midrise, 0.0, 0.0 };

		/* the diagnostic is the only thing that crosses rows */
		if (rms)
			quant_rows(&c, 0, n);
		else
			charsiu_parallel_for(quant_rows, &c, n);
		se = c.se;
		sw = c.sw;
	}

	if (alpha == 0.0 && !getenv("CHARSIU_W4_FILE"))
		wcache_write(t, bits, w->name);

	/*
	 * CHARSIU_W4_FILE replaces what was just computed with weights prepared
	 * offline. The format is one record a tensor: an 80 byte name, n and k
	 * as u64, then n*k signed bytes of nibble values and n floats of scale.
	 * Only the VALUES change; the shape, the wsum and everything downstream
	 * is recomputed here from them, so a bad file cannot quietly disagree
	 * with the rest of the pipeline.
	 */
	if (getenv("CHARSIU_W4_FILE")) {
		FILE *f = fopen(getenv("CHARSIU_W4_FILE"), "rb");
		char nm[80];
		uint64_t fn, fk;
		int got = 0;

		while (f && fread(nm, 1, sizeof(nm), f) == sizeof(nm)
		       && fread(&fn, sizeof(fn), 1, f) == 1
		       && fread(&fk, sizeof(fk), 1, f) == 1) {
			if (!strcmp(nm, w->name) && fn == n && fk == k) {
				got = fread(t->q, 1, (size_t)n * k, f)
					      == (size_t)n * k
				      && fread(t->scale, sizeof(float), n, f)
					      == n;
				break;
			}
			fseek(f, (long)((size_t)fn * fk
					+ fn * sizeof(float)), SEEK_CUR);
		}
		if (f)
			fclose(f);
		if (got) {
			t->kgroup = k;               /* one scale a row */
			for (uint64_t r = 0; r < n; r++) {
				int32_t sum = 0;

				for (uint64_t i = 0; i < k; i++)
					sum += t->q[r * k + i];
				t->wsum[r] = sum;
			}
			fprintf(stderr, "w4file: %s loaded\n", w->name);
		} else {
			fprintf(stderr, "w4file: %s NOT in the file, "
				"using the built-in quantiser\n", w->name);
		}
	}

	free(row);
	t->rms_rel = sw > 0.0 ? sqrt(se / sw) : 0.0;
	return 0;
}

void npu_tensor_free(struct npu_tensor *t)
{
	if (!t)
		return;
	free(t->q);
	free(t->scale);
	free(t->kscale);
	free(t->astat);
	free(t->acov);
	free(t->xcal);
	free(t->wsum);
	memset(t, 0, sizeof(*t));
}

static int32_t idot(const int8_t *w, const int8_t *x, uint64_t n)
{
#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
	int32x4_t acc = vdupq_n_s32(0);
	uint64_t i = 0;
	int32_t s;

	for (; i + 16 <= n; i += 16) {
		int8x16_t a = vld1q_s8(w + i), b = vld1q_s8(x + i);

		acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(a),  vget_low_s8(b)));
		acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(a), vget_high_s8(b)));
	}
	s = vaddvq_s32(acc);
	for (; i < n; i++)
		s += (int32_t)w[i] * (int32_t)x[i];
	return s;
#else
	int32_t s = 0;

	for (uint64_t i = 0; i < n; i++)
		s += (int32_t)w[i] * (int32_t)x[i];
	return s;
#endif
}

/*
 * ⚠ THE k FACTOR HAS TO COME FROM THE ACTIVATIONS, NOT THE WEIGHTS.
 *
 * The first version of this took the column means of |w| and it made the KL
 * worse. That is not AWQ: AWQ's whole claim is that the weights worth
 * protecting are the ones multiplying LARGE ACTIVATIONS, so the factor is built
 * from mean |x_k| over a calibration run. Measuring the wrong signal and
 * concluding the method does not work is the mistake, not the method.
 *
 * Two passes. CHARSIU_CALIB=<file> runs the model and writes the per tensor
 * mean |x_k|; CHARSIU_AWQ_STATS=<file> reads it back at build time.
 */
void npu_calib_note(struct npu_tensor *t, const struct charsiu_act *a)
{
	if (!t->astat) {
		t->astat = calloc(t->k, sizeof(*t->astat));
		if (!t->astat)
			return;
	}
	for (uint64_t i = 0; i < t->k; i++)
		t->astat[i] += fabs((double)a->f[i]);
	t->acalls++;

	/*
	 * ⚠ THE ONE MEASUREMENT THAT DECIDES WHETHER GPTQ IS WORTH DAYS.
	 *
	 * GPTQ's whole premise is that the activations are CORRELATED, so the
	 * error made rounding one weight can be pushed onto the others through
	 * the inverse Hessian. If the activations of this model are close to
	 * uncorrelated, H is nearly diagonal, there is nothing to push the
	 * error into, and GPTQ degenerates to round-to-nearest.
	 *
	 * A synthetic testbed cannot answer that, because iid Gaussian inputs
	 * have H = I by construction, which is exactly the case where the
	 * method cannot help. So accumulate the real thing for ONE tensor:
	 * CHARSIU_CALIB_H names it, and the covariance is dumped beside the
	 * means.
	 */
	/*
	 * ⚠ VECTORS, NOT THE COVARIANCE. GPTQ needs H = X^T X, and at k = 8192
	 * that matrix is 512 MB per tensor in doubles, while the vectors it is
	 * built from are 8 MB. Keep the vectors, build H offline where numpy
	 * has BLAS: a Cholesky at k = 8192 is 5.5e11 flops and does not belong
	 * in a hand written loop.
	 */
	{
		const char *xd = getenv("CHARSIU_CALIB_X");
		unsigned cap = getenv("CHARSIU_CALIB_N")
			? (unsigned)atoi(getenv("CHARSIU_CALIB_N")) : 256;

		if (xd && t->nxcal < cap) {
			if (!t->xcal)
				t->xcal = malloc((size_t)cap * t->k
						 * sizeof(*t->xcal));
			if (t->xcal) {
				float *dst = t->xcal + (size_t)t->nxcal * t->k;

				for (uint64_t i = 0; i < t->k; i++)
					dst[i] = a->f[i];
				t->nxcal++;
			}
		}
	}
}

void npu_matvec(const struct npu_tensor *t, const struct charsiu_act *a,
		float *y, uint64_t row0, uint64_t nrows)
{
	if (getenv("CHARSIU_CALIB") && row0 == 0)
		npu_calib_note((struct npu_tensor *)t, a);
	for (uint64_t r = 0; r < nrows; r++) {
		uint64_t n = row0 + r;

		uint64_t grp = t->kgroup ? t->kgroup : t->k;
		uint64_t ngrp = (t->k + grp - 1) / grp;
		double acc = 0.0;
		const int8_t *aq = a->q1;
		int8_t *free_after = NULL;
		float ad = a->d1;

		/*
		 * The k factor rides on the ACTIVATION, so a tensor that has one
		 * needs its own quantisation of the same input vector. On the
		 * board this is one multiply a k before the pack; here it is
		 * done straight so the measurement is honest.
		 */
		if (t->kscale) {
			/*
			 * ⚠ PER CALL, NOT static. npu_matvec runs on the worker
			 * threads, and a static scratch buffer here was a race
			 * and a double free: three of the five exponents in the
			 * first sweep produced no output at all.
			 */
			int8_t *tmp = malloc((size_t)t->k);
			float amax = 0.0f;

			if (!tmp)
				return;
			for (uint64_t i = 0; i < t->k; i++) {
				float v = a->f[i] * t->kscale[i];

				if (fabsf(v) > amax) amax = fabsf(v);
			}
			ad = amax / 127.0f;
			for (uint64_t i = 0; i < t->k; i++) {
				int v = (int)lrintf(a->f[i] * t->kscale[i]
						    / (ad != 0.0f ? ad : 1.0f));

				if (v > 127) v = 127;
				if (v < -127) v = -127;
				tmp[i] = (int8_t)v;
			}
			aq = tmp;
			free_after = tmp;
		}

		/*
		 * ⚠ WHAT PRECISION IS THE ACTIVATION, REALLY.
		 *
		 * charsiu's int4 path already tells the hardware 16 bit
		 * activations: charsiu_effective_adtype returns FP16 and 0x100c
		 * carries bit 29. But the PACKING puts an int8 value in the high
		 * byte of that 16 bit slot, so the container is sixteen bits and
		 * the precision is eight. Every quality number measured so far
		 * used a->q1, which is that eight bit value, so the table is a
		 * measurement of w4a8.
		 *
		 * CHARSIU_NPU_A16 fills the slot instead: the activation stays
		 * at full precision and only the weights are four bits, which is
		 * what the vendor's W4A16 is and what makes a k factor possible
		 * at all, since spreading an eight bit activation's range is
		 * what destroyed it here.
		 */
		if (getenv("CHARSIU_NPU_A16")) {
			for (uint64_t g = 0; g < ngrp; g++) {
				uint64_t lo = g * grp;
				uint64_t len = lo + grp < t->k ? grp : t->k - lo;
				double part = 0.0;

				for (uint64_t i = lo; i < lo + len; i++) {
					double av = a->f[i];

					if (t->kscale)
						av *= t->kscale[i];
					part += (double)t->q[n * t->k + i] * av;
					if (midrise_grid())
						part += 0.5 * av;
				}
				acc += part * t->scale[n * ngrp + g];
			}
			y[n] = (float)acc;
			free(free_after);
			continue;
		}
		for (uint64_t g = 0; g < ngrp; g++) {
			uint64_t lo = g * grp;
			uint64_t len = lo + grp < t->k ? grp : t->k - lo;
			double part = (double)idot(t->q + n * t->k + lo,
						   aq + lo, len);

			/*
			 * ⚠ THE MIDRISE HALF STEP BELONGS HERE TOO. This is the
			 * CPU reference for the same weights, and it computed
			 * sum(s * a) while scaling by a d that means
			 * w = (s + 0.5) * d. The first host test of the grid
			 * produced nothing but <|begin_of_text|> because of it,
			 * which is what a reference that disagrees with its own
			 * quantiser looks like.
			 */
			if (midrise_grid()) {
				double as = 0.0;

				for (uint64_t i = lo; i < lo + len; i++)
					as += (double)aq[i];
				part += 0.5 * as;
			}
			acc += part * t->scale[n * ngrp + g];
		}
		y[n] = (float)(acc * ad);
		free(free_after);
	}
}

static uint64_t npu_cal_calls(void)
{
	static long c = -1;

	if (c < 0) {
		const char *e = getenv("CHARSIU_NPU_CAL");

		c = e ? atol(e) : 32;
		if (c < 1)
			c = 1;
	}
	return (uint64_t)c;
}

int npu_out8_mode(void)
{
	static int m = -1;

	if (m < 0) {
		const char *e = getenv("CHARSIU_NPU_OUT8");

		m = e ? atoi(e) : 0;
	}
	return m;
}

void npu_quantise_output(struct npu_tensor *t, float *y, uint64_t n, int mode)
{
	float amax = 0.0f, d, id;
	uint64_t clipped = 0;

	if (mode <= 0)
		return;

	for (uint64_t i = 0; i < n; i++) {
		float a = fabsf(y[i]);

		if (a > amax)
			amax = a;
	}

	if (t->amax_lo <= 0.0f || amax < t->amax_lo)
		t->amax_lo = amax;
	if (amax > t->amax_hi)
		t->amax_hi = amax;

	if (mode >= 2) {
		/*
		 * A coefficient buffer holds ONE set of numbers for the whole
		 * run, so the scale cannot look at the vector it is about to
		 * quantise.
		 *
		 * mode 2 freezes on the FIRST call, which is the worst possible
		 * calibration and not a fair test of the idea. mode 3 tracks the
		 * running maximum over the first CHARSIU_NPU_CAL calls (32 by
		 * default, so several tokens) and freezes after that, which is
		 * what a calibration pass would actually produce.
		 */
		uint64_t cal = mode >= 3 ? npu_cal_calls() : 1;

		if (t->out_calls < cal) {
			float want = amax / 127.0f;

			if (want > t->out_scale)
				t->out_scale = want;
		}
		d = t->out_scale;
	} else {
		d = amax / 127.0f;
	}

	id = d != 0.0f ? 1.0f / d : 0.0f;
	for (uint64_t i = 0; i < n; i++) {
		int v = (int)lrintf(y[i] * id);

		if (v > 127) { v = 127; clipped++; }
		if (v < -127) { v = -127; clipped++; }
		y[i] = (float)v * d;
	}
	t->out_clip += (double)clipped / (double)n;
	t->out_calls++;
}

/*
 * What a FIXED output scale is up against.
 *
 * The scale a coefficient buffer holds has to cover the largest vector the
 * model will ever produce, and every other vector then uses 127 divided by how
 * much larger that one was. This prints that ratio.
 */
void npu_report(const struct npu_tensor *t, unsigned count)
{
	fprintf(stderr, "\n%-28s  %10s %10s %8s  %8s\n",
		"tensor", "min |y|", "max |y|", "ratio", "clip%");
	for (unsigned i = 0; i < count; i++) {
		const struct npu_tensor *x = &t[i];

		if (!x->out_calls)
			continue;
		fprintf(stderr, "%-28s  %10.3f %10.3f %8.1f  %8.3f\n",
			x->name, (double)x->amax_lo, (double)x->amax_hi,
			x->amax_lo > 0.0f ? (double)(x->amax_hi / x->amax_lo) : 0.0,
			x->out_clip / (double)x->out_calls * 100.0);
	}
}
