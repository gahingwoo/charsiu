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
 * ⚠⚠ FOUR BITS IN MEMORY, NOT EIGHT: WHAT t->q HOLDS, AND WHY THAT IS ONE BOOL.
 *
 * t->q used to keep one whole int8_t per int4 code, which is a second copy of
 * every routed weight at DOUBLE the width of the gguf it was read from and
 * double the width the hardware is handed. Counted over the tensors llama.c
 * actually routes, on the real files: Qwen3-0.6B is 595 984 384 codes, so q was
 * 596.0 MB of it; gemma-3-1b is 999.8 MB and Phi-3.5-mini 3722.4 MB. That is
 * the largest single line in this runtime's memory and more than half of it,
 * on a board with 8 GB shared with everything else -- the 1068 MB Qwen3 peak
 * npudev's output buffer note weighs against Rockchip's 513 has all 596 of
 * these MB inside it.
 *
 * Packed, and measured on the development host, which routes every one of those
 * tensors through the CPU fallback because it has no NPU:
 *
 *   Qwen3-0.6B     953 -> 669 MiB     -284    -29.8%
 *   gemma-3-1b    1682 -> 1205 MiB    -477    -28.4%
 *   Phi-3.5-mini  5679 -> 3904 MiB   -1775    -31.3%
 *
 * with the generated text of all three byte identical to the same binary built
 * before the change, and the decode rate unchanged on the two small ones.
 *
 * So a row of q is packed two codes to a byte, LOW NIBBLE FIRST: column i of a
 * row lives in the low half of byte i/2 when i is even and the high half when i
 * is odd. That is exactly the layout npudev's e->cq already uses for the CPU's
 * share of a split projection, nibble order included -- which is why that copy
 * is now a memcpy of the row instead of a gather.
 *
 * ⚠ ROW STRIDED, ((k + 1) / 2) BYTES A ROW, and NOT the flat (n*k + 1)/2 the
 * weight cache file used to hold. Three reasons, and the third is not a
 * preference:
 *
 *   - every reader of q is a ROW reader, and a byte aligned row start means the
 *     column index alone decides which half of the byte a code is in. Flat
 *     packing puts row r at nibble r*k, so at an odd k the parity alternates
 *     from row to row and every reader needs a starts-odd case. cpu_rows in
 *     npudev records what that costs when it is missed: "at k = 34 with groups
 *     of 17 the test measured 2.26 relative against 1e-6 elsewhere".
 *   - it makes e->cq a memcpy rather than a gather and a shift.
 *   - ⚠⚠ quant_rows RUNS ON THE THREAD POOL, SPLIT BY ROWS. Under flat packing
 *     at an odd k the last code of row r and the first code of row r+1 share a
 *     byte, so two workers read-modify-write the same address and one of the
 *     two nibbles is lost -- silently, and only for the shapes nobody has.
 *     Row striding makes that byte impossible: no two rows ever meet.
 *
 * The cost is one padding nibble a row at odd k: 65 nibbles across the 24
 * shapes the layout was checked at, and EXACTLY ZERO for every real model,
 * whose k is a multiple of 32 because that is gguf's own block. Which is also
 * why the weight cache file did not change length for any model that has one.
 *
 * ⚠ AND IT IS A PROCESS WIDE CONSTANT, DELIBERATELY NARROWER THAN bits == 4.
 *
 * npu_tensor_build picks four bits per TENSOR: CHARSIU_NPU_W4_ONLY narrows int4
 * to the tensors whose name contains a substring. npudev has to read q back and
 * all it holds of a tensor's identity is t->name, an 80 byte copy that a long
 * enough name truncates -- so under W4_ONLY the two files could disagree about
 * the width of the same buffer, and a reader that is one nibble out of step
 * with the writer does not fail. It answers, in fluent sentences. This project
 * has shipped one fast wrong answer already and it cost four commits.
 *
 * So W4_ONLY, which is a diagnostic for WHERE the error lives rather than
 * anything a deployment runs, keeps the old one byte a code array and gives up
 * the saving. Every other int4 run -- which is every real one -- packs, and
 * "is q packed" is then a single bool that both files read from this one
 * function and that nothing per tensor can make disagree.
 */
/*
 * ⚠⚠ AN EMPTY OR ZERO VALUE MEANS OFF, AND HERE IT DID NOT.
 *
 * npudev.c fixed exactly this for the device side and wrote down what it
 * cost: `!= NULL` makes CHARSIU_NPU_W4V= turn int4 ON, there is no way to get
 * int8 past a runner that sets the variable itself, and a board round meant
 * to measure the int8 batched path ran int4 instead -- caught only because it
 * said so in its own report.
 *
 * The quantiser was left on the presence rule, so the two files disagreed
 * about the same variable: CHARSIU_NPU_W4V=0 opened an INT8 device and built
 * INT4 codes for it. Both sides read it the same way now, which is what
 * npu_mode() and act_set() in this tree already do.
 */
static int w4_env(void)
{
	const char *a = getenv("CHARSIU_NPU_W4");
	const char *b = getenv("CHARSIU_NPU_W4V");

	return (a && *a && *a != '0') || (b && *b && *b != '0');
}

/*
 * "0-2", or "4": is this tensor's block inside that range?
 *
 * ⚠ UNSET OR EMPTY MEANS NO, NOT YES. Two knobs read this and both would be
 * dangerous if an unparseable range quietly meant "everything": one of them
 * decides a tensor's WIDTH.
 *
 * A tensor with no "blk.<n>." in its name is not a layer and is never in range.
 */
static int layer_in_range(const char *spec, const char *name)
{
	const char *b = name ? strstr(name, "blk.") : NULL;
	char *end;
	long lo, hi, ln;

	if (!spec || !*spec || !b)
		return 0;
	lo = strtol(spec, &end, 10);
	if (end == spec)
		return 0;
	hi = *end == '-' ? strtol(end + 1, NULL, 10) : lo;
	ln = strtol(b + 4, NULL, 10);
	return ln >= lo && ln <= hi;
}

int npu_q_packed(void)
{
	static int v = -1;

	if (v < 0)
		v = w4_env() && !getenv("CHARSIU_NPU_W4_ONLY");
	return v;
}

/* bytes one row of q occupies, which is what every reader of it strides by */
size_t npu_q_stride(uint64_t k)
{
	return npu_q_packed() ? (size_t)((k + 1) / 2) : (size_t)k;
}

/*
 * THE SAME QUESTION, ASKED OF ONE TENSOR.
 *
 * npu_q_packed is a process-wide answer and that is what made a mixed model
 * unaffordable: CHARSIU_NPU_W4_ONLY turned packing off for EVERY tensor, so a
 * model with eight bits on two layers paid a byte a code on the other fourteen
 * as well -- 1.0 bytes a weight, which is what all int8 costs anyway.
 *
 * The width belongs to the tensor. Both files read t->packed and neither can
 * be a nibble out of step with the other, which is the hazard the note above
 * npu_q_packed describes.
 */
size_t npu_q_stride_t(const struct npu_tensor *t)
{
	return t->packed ? (size_t)((t->k + 1) / 2) : (size_t)t->k;
}

/*
 * ONE CODE OUT OF A ROW OF q, BY COLUMN.
 *
 * pk is npu_q_packed() hoisted out of the caller's loop. It is constant for the
 * life of the process, and these sit in loops that run once per weight in the
 * model -- 1.24 billion times on Llama-3.2-1B, which is the count that made
 * getenv inside a loop worth two rounds of staging time to find.
 */
static inline int q_at(const int8_t *row, uint64_t i, int pk)
{
	unsigned v;

	if (!pk)
		return row[i];
	v = (unsigned)(uint8_t)row[i >> 1];
	v = (i & 1) ? (v >> 4) : (v & 0xfu);
	return v >= 8 ? (int)v - 16 : (int)v;
}

/*
 * ⚠ EVEN COLUMNS ASSIGN, ODD COLUMNS OR, and that is what leaves a row fully
 * defined after ONE ascending pass over its columns. q is malloc'd and not
 * calloc'd, so a read-modify-write on the even half would be reading
 * uninitialised memory, and at an odd k it would also leave the row's trailing
 * padding nibble undefined for the weight cache to write out to a file.
 *
 * Every writer here does walk a row's columns in ascending order and cover all
 * of [0, k): quant_rows goes group by group and the offline weight file is read
 * a row at a time. A writer that did neither would have to clear the row first.
 */
static inline void q_put(int8_t *row, uint64_t i, int v, int pk)
{
	if (!pk)
		row[i] = (int8_t)v;
	else if (i & 1)
		row[i >> 1] = (int8_t)((unsigned)(uint8_t)row[i >> 1]
				       | (((unsigned)v & 0xfu) << 4));
	else
		row[i >> 1] = (int8_t)((unsigned)v & 0xfu);
}

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
 * ⚠ THE RECORD IS A STRAIGHT COPY OF q, which is what it was not. It used to
 * pack the nibbles on the way out and unpack them on the way back in, because
 * q was one byte a code in memory and half that in the file; now that q is
 * packed itself the record is an fwrite and the read is an fread. That is the
 * whole point of the cache getting cheaper rather than only smaller:
 * gemma-3-1b's 999 751 680 codes came back in 710 to 1054 ms of staging when
 * every one of them had to be widened on the way in, and in 85 to 93 ms now.
 *
 * ⚠⚠ WHICH MEANS THE WIDTH IS PART OF THE KEY. A cache written when q was one
 * byte a code cannot be read by a build that packs it -- at an odd k the two
 * are not even the same length -- so npu_q_packed() goes into the group word
 * below. int8 caches are untouched by any of this and stay valid.
 */
/* one getenv, cached: this is asked inside loops over the whole tensor */
static int midrise_grid(void)
{
	static int v = -1;

	if (v < 0)
		v = charsiu_env_flag("CHARSIU_NPU_W4_MIDRISE", 0);
	return v;
}

#define WCACHE_MAGIC  0x43535743u        /* "CSWC" */
#define WCACHE_FORMAT 1u
/* ⚠ BUMP THIS whenever the quantiser's arithmetic changes, or an old cache
 * will quietly feed the new code the old numbers.
 *
 * 2 -> 3 on 2026-09-08: int8 now writes one scale a row where it used to write
 * CHARSIU_NPU_W4_GROUP of them. `group` is in the key, but wcache_setup runs
 * once on the FIRST tensor and int8's group is now that tensor's k -- so on a
 * model whose first staged tensor is 1024 wide, which qwen3's n_embd is, the
 * new key and an old int8 key are the same number and the old file would have
 * been accepted. That is exactly the layout disagreement this week was spent
 * closing, coming back through the cache. */
#define WCACHE_QUANT  3u

struct wcache_head {
	uint32_t magic, format, quant, bits;
	uint64_t group;
	char stamp[64];
};

static struct {
	FILE *f;
	int writing;
	int checked;
	char path[512];
	char stamp[64];
	int overridden;
} wc;

/*
 * ⚠ THE CACHE IS ONE SEQUENTIAL FILE AND ONE STATIC HANDLE. Records go in in
 * the order the tensors are built and come back in the same order, so two
 * graphs staging into it at once would interleave and neither could read the
 * result. That was fine while the language model was the only thing that
 * staged; a vision tower or a whisper encoder in the same process is a second
 * stream.
 *
 * So a caller can claim the cache for its own stretch of staging. It closes
 * whatever was open, which is what makes the previous owner's file complete.
 */
void charsiu_wcache_use(const char *path, const char *stamp)
{
	if (wc.f) {
		fclose(wc.f);
		wc.f = NULL;
	}
	wc.writing = 0;
	wc.checked = 0;
	wc.overridden = path != NULL;
	snprintf(wc.path, sizeof(wc.path), "%s", path ? path : "");
	snprintf(wc.stamp, sizeof(wc.stamp), "%s", stamp ? stamp : "");
}

static void wcache_setup(unsigned bits, uint64_t grp)
{
	const char *path = wc.overridden ? wc.path
					 : getenv("CHARSIU_NPU_CACHE");
	const char *stamp = wc.overridden ? wc.stamp
					  : getenv("CHARSIU_NPU_CACHE_STAMP");
	struct wcache_head h, want;

	if (wc.checked)
		return;
	wc.checked = 1;
	if (!path || !*path)
		return;
	/*
	 * ⚠ THE CACHE HEADER HOLDS ONE `bits` FOR THE WHOLE FILE, and
	 * wcache_read validates a record's name, n, k and ngrp but not its
	 * width. That is safe while every tensor has the same width and is not
	 * safe the moment they do not, so a mixed model does not use the cache
	 * at all rather than use it carefully.
	 */
	if (getenv("CHARSIU_NPU_INT8_LAYERS") || getenv("CHARSIU_NPU_W4_ONLY")) {
		fprintf(stderr, "charsiu: the weight cache is off -- this model "
			"mixes widths and the cache header describes one\n");
		return;
	}

	memset(&want, 0, sizeof(want));
	want.magic = WCACHE_MAGIC;
	want.format = WCACHE_FORMAT;
	want.quant = WCACHE_QUANT;
	want.bits = bits;
	/* the grid is part of what the codes mean, so it is part of the key --
	 * and so is the WIDTH q is held at, because the record is a straight
	 * copy of it. Bit 63 is the grid, bit 62 is npu_q_packed(). */
	want.group = grp | (midrise_grid() ? (1ull << 63) : 0)
			 | (npu_q_packed() ? (1ull << 62) : 0);
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

/* one record's payload size: q goes to the file exactly as it is held */
static size_t wcache_qbytes(const struct npu_tensor *t)
{
	return (size_t)t->n * npu_q_stride_t(t);
}

static int wcache_read(struct npu_tensor *t, const char *name)
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
	qb = wcache_qbytes(t);
	if (fread(t->q, 1, qb, wc.f) != qb ||
	    fread(t->scale, sizeof(float), (size_t)(n * ngrp), wc.f)
		    != (size_t)(n * ngrp))
		return 0;
	/* wsum is n int32 and recomputing it is a whole pass over the codes */
	if (fread(t->wsum, sizeof(int32_t), (size_t)n, wc.f) != (size_t)n)
		return 0;
	return 1;
}

static void wcache_write(const struct npu_tensor *t, const char *name)
{
	char nm[80];
	uint64_t n = t->n, k = t->k;
	uint64_t ngrp = t->kgroup ? (k + t->kgroup - 1) / t->kgroup : 1;
	size_t qb = wcache_qbytes(t);

	if (!wc.f || !wc.writing)
		return;
	memset(nm, 0, sizeof(nm));
	snprintf(nm, sizeof(nm), "%s", name);
	if (fwrite(nm, 1, sizeof(nm), wc.f) != sizeof(nm) ||
	    fwrite(&n, sizeof(n), 1, wc.f) != 1 ||
	    fwrite(&k, sizeof(k), 1, wc.f) != 1 ||
	    fwrite(&ngrp, sizeof(ngrp), 1, wc.f) != 1)
		goto bad;
	if (fwrite(t->q, 1, qb, wc.f) != qb)
		goto bad;
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
	/*
	 * ⚠ THE ROW STRIDE IS NOT k ANY MORE. Read once here rather than per
	 * row: npu_q_stride goes through npu_q_packed, which is a getenv behind
	 * a static, and this struct is what the whole quantiser reads its
	 * constants out of for exactly that reason.
	 */
	size_t stride;
	int pk;
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
	const size_t stride = c->stride;
	const int pk = c->pk;
	double se = 0.0, sw = 0.0;
	float *row = malloc((size_t)k * sizeof(float));

	if (!row)
		return;
	for (uint64_t r = r0; r < r0 + nr; r++) {
		int8_t *dst = t->q + r * stride;
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
			q_put(dst, i, v, pk);
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

/*
 * ⚠ CHARSIU_W4_FILE IS ONE BYTE A CODE AND q MAY NOT BE. The offline format is
 * documented below as n*k signed bytes and nothing about it changes here -- a
 * file prepared by tools/gptq.py months ago still loads -- so the rows are read
 * one at a time into a k byte scratch and packed on the way in. The straight
 * fread this replaced would now write twice the length of the buffer.
 */
static int w4file_codes(FILE *f, struct npu_tensor *t, uint64_t n, uint64_t k)
{
	const int pk = t->packed;
	const size_t stride = npu_q_stride_t(t);
	int8_t *scratch = malloc((size_t)k);
	uint64_t r, i;

	if (!scratch)
		return 0;
	for (r = 0; r < n; r++) {
		if (fread(scratch, 1, (size_t)k, f) != (size_t)k) {
			free(scratch);
			return 0;
		}
		for (i = 0; i < k; i++)
			q_put(t->q + r * stride, i, scratch[i], pk);
	}
	free(scratch);
	return 1;
}

int npu_tensor_build(struct npu_tensor *t, const struct gguf_tensor *w)
{
	uint64_t n = w->ne[1], k = w->ne[0];
	float *row;
	double se = 0.0, sw = 0.0;

	/*
	 * ⚠ THE AWQ PAIR, SET HERE RATHER THAN ASSUMED OF THE CALLER. Both
	 * callers today hand this a zeroed tensor -- npupool calloc's its
	 * array and npu_slice_test writes `= { 0 }` -- and the factor is only
	 * ever allocated further down, so nothing is leaked by this.
	 *
	 * It matters because the two fields are read as a PAIR by three
	 * places now: the group gate, the batched pack, and the input reuse
	 * key. A garbage kshash beside a NULL kscale would not crash; it
	 * would make one device's cached input look like another tensor's
	 * for the rest of the run, and that is a wrong answer in fluent
	 * sentences. A field whose zero value means "no factor" should not
	 * depend on the caller having remembered.
	 */
	t->kscale = NULL;
	t->kshash = 0;

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
	unsigned bits = w4_env()
		&& (!w4only || strstr(w->name, w4only)) ? 4 : 8;

	/*
	 * CHARSIU_NPU_INT8_LAYERS keeps a range of blocks at EIGHT bits while
	 * the rest stay at four. Layers 0 and 1 are 12.5% of the weight bytes
	 * and carry 44% of the four-bit damage, and eight bits on just those
	 * two beats a group of 128 on all sixteen -- ppl 24.28 against 26.94,
	 * two thirds of the whole distance to lossless, for +12.5% of bytes.
	 *
	 * This only became affordable when the width moved onto the tensor:
	 * CHARSIU_NPU_W4_ONLY expresses the same mix but unpacks EVERY tensor,
	 * so it costs a byte a code throughout, which is what all int8 costs.
	 */
	if (bits == 4 && layer_in_range(getenv("CHARSIU_NPU_INT8_LAYERS"),
					w->name))
		bits = 8;
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
	const int w4sym = charsiu_env_flag("CHARSIU_NPU_W4_SYM", 0);
	const int w4clip = charsiu_env_flag("CHARSIU_NPU_W4_CLIP", 0);
	const int rms = charsiu_env_flag("CHARSIU_NPU_RMS", 0);
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
	/*
	 * ⚠⚠ AND THE SAME DISAGREEMENT AGAIN, ON THE OTHER SIDE OF bits.
	 *
	 * tensor_grouped() also requires g->w4. So an INT8 tensor whose k
	 * divides the group exactly walks past the remainder collapse above,
	 * gets its scales written as scale[row * ngrp + group], and is then
	 * read by a consumer that takes scale[row] -- the identical fault the
	 * comment above describes, in the one case that comment's condition
	 * cannot see.
	 *
	 * It is not hypothetical. Every board round exports
	 * CHARSIU_NPU_W4_GROUP=1024 whatever the format, so the accidental
	 * w8a8 arm of round 428 measured ppl 272369 against int4's 75.17 and
	 * was written down as "the int8 path emits noise". The int8 path is
	 * fine; it was being handed scales in a layout it does not read.
	 *
	 * One scale a row is what the int8 consumer applies, so it is what
	 * gets written -- and at eight bits it costs nothing. Host, qwen3,
	 * 200 tokens, CPU reference: group 1024 44.81, one scale a row 43.66.
	 * The row is not worse, it is very slightly BETTER, because eight bits
	 * spans a row's spread on its own and the finer scales only add their
	 * own rounding. Four bits is the opposite -- 91.66 grouped against
	 * 114.22 a row -- which is why the group exists at all.
	 */
	if (bits != 4)
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
	 * CHARSIU_NPU_AWQ is the exponent, 0 for off. 0.5 is the usual square
	 * root balance and is what this tree used from the day the factor was
	 * written; it is NOT the best value here.
	 *
	 * ⚠ THE EXPONENT WAS NEVER SWEPT. Every earlier experiment pinned it
	 * at 0.5 and moved the clamp instead. Swept with the clamp held at its
	 * default 2.0 -- qwen3, host CPU reference, 500 tokens -- it is a clean
	 * single minimum and 0.5 is on the wrong side of it:
	 *
	 *   alpha  0.25   0.30   0.35   0.40   0.45   0.50
	 *   ppl   68.86  65.84  65.12  68.02  75.52  76.36
	 *
	 * 65.12 against 76.36 is 14.7%, and the same ordering holds on the
	 * shorter corpus at 200 tokens (68.07 against 73.77).
	 *
	 * ⚠ THE MINIMUM IS PER MODEL. Llama-3.2-1B, its own calibration, same
	 * corpus and length: off 52.34, then
	 *
	 *   alpha  0.20   0.25   0.30   0.35   0.40   0.50   0.65
	 *   ppl   43.86  46.98  52.55  50.58  52.95  68.26  92.42
	 *
	 * so its minimum is 0.20 and at 0.5 AWQ is WORSE THAN OFF. Sweep it
	 * per model; what both models agree on is only that 0.5 is past the
	 * minimum.
	 */
	double alpha = getenv("CHARSIU_NPU_AWQ")
		? atof(getenv("CHARSIU_NPU_AWQ")) : 0.0;

	/*
	 * CHARSIU_NPU_AWQ_LAYERS restricts the factor to a range of blocks,
	 * "0-2" or "4". The vendor spends its own calibration almost entirely
	 * on the first three: rho = 15 * scale / (max - min), read out of the
	 * scale arrays in its .rkllm, runs 1.5 to 22.3 on layers 0 to 2 of
	 * Llama-3.2-1B and sits within a few percent of 1 on 3 to 15
	 * (tools/rkllm_scales.py rho). AWQ costs decode -- a tensor carrying a
	 * factor cannot share a packed input, so grouped q/k/v drop to single
	 * calls -- so WHERE it can be switched off is worth a knob.
	 *
	 * ⚠ UNSET OR EMPTY MEANS EVERY LAYER, which is what AWQ did before
	 * this existed. A range this cannot parse must not quietly turn the
	 * method off: the arm that says "AWQ on" would then be the arm with
	 * AWQ off, and this tree has already run four of those.
	 *
	 * A tensor with no "blk.<n>." in its name is not a layer and is never
	 * restricted by this.
	 */
	{
		const char *lr = getenv("CHARSIU_NPU_AWQ_LAYERS");

		if (alpha != 0.0 && lr && *lr && strstr(w->name, "blk.") &&
		    !layer_in_range(lr, w->name))
			alpha = 0.0;
	}

	/*
	 * ⚠⚠ AND IT IS A FOUR BIT METHOD, ON A PATH THAT CANNOT SAY SO.
	 *
	 * The factor only cancels because the weights are divided by it and the
	 * activation is multiplied by it. charsiu_npu_matvec does that multiply
	 * on the w4a16 path, where the activation is a float it can scale
	 * before packing. The int8 path packs a->q1, one absmax quantisation of
	 * the whole vector, and there is nowhere in it for a per column factor
	 * -- so the divide happens, the multiply does not, and the factor does
	 * not cancel. Board, qwen3, 600 tokens: int8 27.07, int8 with AWQ
	 * 2162.73. Wrong numbers, not an error.
	 *
	 * Refusing loses nothing measurable. AWQ exists to protect a small
	 * dynamic range and eight bits does not have that problem: on the host
	 * CPU reference, which DOES apply the multiply at eight bits, AWQ makes
	 * int8 slightly worse -- 43.66 off against 44.76 on. So the method is
	 * declined for anything but four bits and says so once.
	 */
	if (alpha != 0.0 && bits != 4) {
		static int said;

		if (!said++)
			fprintf(stderr, "charsiu: CHARSIU_NPU_AWQ is a four bit "
				"method and these weights are %u bit -- "
				"ignoring it. It measured worse at eight bits "
				"even where it is applied correctly.\n", bits);
		alpha = 0.0;
	}

	memset(t, 0, sizeof(*t));
	snprintf(t->name, sizeof(t->name), "%s", w->name);
	t->n = n;
	t->k = k;
	t->kgroup = grp;
	/* the width is this tensor's, decided by its own bits */
	t->packed = (bits == 4);
	t->q = malloc((size_t)n * npu_q_stride_t(t));
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
		 * ⚠ AND THE FIGURE IS THE REAL ONE NOW. The copy used to be one
		 * BYTE a weight whatever the bit width, so gemma3's 262144 by
		 * 1152 output head asked for 302 MB even at four bits and this
		 * line said so, while npudev's own weight_mb for the same tensor
		 * said 151. It is 151 in both places now.
		 *
		 * ⚠ WHICH DOES NOT MEAN THE HEAD IS ROUTED. It is refused by the
		 * maxn gate in charsiu_pool_get long before this, and that is
		 * still 44% of a gemma token on the CPU -- this line is only
		 * about what the message claims when the malloc is what fails.
		 */
		fprintf(stderr, "charsiu: %s stays on the CPU -- its %llu x %llu "
			"quantised copy needs %.0f MB and would not allocate\n",
			w->name, (unsigned long long)n, (unsigned long long)k,
			(double)((size_t)n * npu_q_stride_t(t)) / 1e6);
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
		if (wcache_read(t, w->name)) {
			free(row);
			return 0;
		}
	}

	if (alpha != 0.0) {
		const char *sg = getenv("CHARSIU_NPU_AWQ_SIGN");
		double awq_sign = sg && *sg == '+' ? 1.0 : -1.0;
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
		if (!got) {
			/*
			 * ⚠⚠ AND THIS FALLBACK IS THE REFUTED VARIANT, SAID OUT
			 * LOUD. The column means of |w| are what the first
			 * version of AWQ used here, and the note further down
			 * records why they are wrong: the weights worth
			 * protecting are the ones multiplying LARGE
			 * ACTIVATIONS, which the weights cannot know. Reaching
			 * this quietly is how "AWQ does not work" gets
			 * concluded from measuring something else.
			 */
			static int said;

			/*
			 * ⚠⚠ AND IT IS WORSE THAN NOT RUNNING AWQ AT ALL, so
			 * warning was not enough. Llama-3.2-1B, host CPU
			 * reference, 300 tokens:
			 *
			 *   AWQ off                        41.53
			 *   AWQ 0.20 with statistics       35.20   -15.2%
			 *   AWQ 0.20 with NO statistics    58.35   +40.5%
			 *
			 * Declining costs the caller nothing they had -- it
			 * lands them on the 41.53 they would have had anyway --
			 * and the fallback costs them 40%. So it declines, and
			 * CHARSIU_NPU_AWQ_WEIGHTMEANS=1 keeps the refuted
			 * variant reachable for anyone who wants it as a
			 * control, which is the only thing it is good for.
			 */
			if (!charsiu_env_flag("CHARSIU_NPU_AWQ_WEIGHTMEANS", 0)) {
				if (!said) {
					said = 1;
					fprintf(stderr,
						"charsiu: AWQ is on but no "
						"activation statistics were "
						"found (first: %s) -- DECLINING "
						"it. The fallback measured 40%% "
						"WORSE than leaving AWQ off. "
						"Record them with "
						"CHARSIU_CALIB=<file>, then "
						"point CHARSIU_AWQ_STATS at it "
						"or leave it beside the gguf as "
						"<model>.gguf.awq.\n", w->name);
				}
				free(col);
				alpha = 0.0;
			} else {
				if (!said) {
					said = 1;
					fprintf(stderr,
						"charsiu: AWQ from the column "
						"means of the WEIGHTS, which "
						"measured 40%% worse than off. "
						"This is a control, not a "
						"setting.\n");
				}
				for (uint64_t r = 0; r < n; r++) {
					gguf_row_f32(w, r, row);
					for (uint64_t i = 0; i < k; i++)
						col[i] += fabs((double)row[i]);
				}
			}
		}
		if (alpha == 0.0)
			goto no_awq;
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
				/*
				 * ⚠⚠ NEGATIVE, AND THAT IS THE WHOLE METHOD.
				 *
				 * quant_rows DIVIDES the weights by this factor
				 * and the activation is MULTIPLIED by it, so
				 * W' = W/f and x' = x*f. AWQ asks for the other
				 * one: W' = W*s with s = (mean|x|)^alpha, which
				 * PROTECTS the columns that meet large
				 * activations by giving them more of the int4
				 * grid. With a positive exponent this file
				 * shrinks exactly those columns instead.
				 *
				 * f = 1/s, so the exponent is -alpha.
				 * CHARSIU_NPU_AWQ_SIGN=+ restores the old sense
				 * as the control.
				 */
				col[i] = pow(v, awq_sign * alpha);
				gm += log(col[i]);
			}
		}
		gm = exp(gm / (double)k);            /* keep the mean factor at 1 */
		{
		/*
		 * ⚠⚠ THE CLAMP IS 64x WIDE AND THAT IS THE SUSPECT.
		 *
		 * The factor divides the weights: a k with f = 0.125 has its
		 * column multiplied by EIGHT before rounding. One such column
		 * sets vmax for the whole row, so the int4 step for every other
		 * weight in it goes eight times coarser -- which is the shape
		 * of a collapse, not of a trade.
		 *
		 * Measured with the clamp as it was, qwen3, 200 tokens, on the
		 * CPU reference where no hardware is involved:
		 *
		 *   AWQ off                         114.22
		 *   AWQ 0.5, clamp [0.125, 8]  7252312.56
		 *
		 * Published AWQ keeps the factor near 1 -- the point is to
		 * protect a few salient channels, not to rescale the tensor.
		 * CHARSIU_NPU_AWQ_CLAMP sets the bound, default 2.0, so the
		 * factor lives in [0.5, 2].
		 */
		const char *cl = getenv("CHARSIU_NPU_AWQ_CLAMP");
		double hi = cl && *cl ? atof(cl) : 2.0;
		double lo = hi > 0.0 ? 1.0 / hi : 0.5;

		for (uint64_t i = 0; i < k; i++) {
			double f = col[i] / gm;

			if (f < lo) f = lo;
			if (f > hi) f = hi;
			t->kscale[i] = (float)f;
		}
		/*
		 * A hash of the factor, so charsiu_npu_matvec_group can ask
		 * "is this the same factor?" in one comparison. q, k and v
		 * read one activation and so are handed byte-identical
		 * statistics, which makes their factors byte-identical too --
		 * checked on this model's own calibration file, all sixteen
		 * layers, and the same for gate against up.
		 */
		t->kshash = 1469598103934665603ull;
		for (uint64_t i = 0; i < k; i++) {
			uint32_t b;

			memcpy(&b, &t->kscale[i], sizeof(b));
			t->kshash = (t->kshash ^ b) * 1099511628211ull;
		}
		}
		free(col);
	}
no_awq:

	{
		struct qrows c = { t, w, k, ngrp, grp, bits, qmax,
				   w4sym, w4clip, rms, midrise,
				   npu_q_stride_t(t), t->packed, 0.0, 0.0 };

		/* the diagnostic is the only thing that crosses rows */
		if (rms)
			quant_rows(&c, 0, n);
		else
			charsiu_parallel_for(quant_rows, &c, n);
		se = c.se;
		sw = c.sw;
	}

	if (alpha == 0.0 && !getenv("CHARSIU_W4_FILE"))
		wcache_write(t, w->name);

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
				got = w4file_codes(f, t, n, k)
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
			const int pk = t->packed;
			const size_t stride = npu_q_stride_t(t);

			t->kgroup = k;               /* one scale a row */
			for (uint64_t r = 0; r < n; r++) {
				const int8_t *qr = t->q + r * stride;
				int32_t sum = 0;

				for (uint64_t i = 0; i < k; i++)
					sum += q_at(qr, i, pk);
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
 * THE SAME DOT PRODUCT AGAINST A PACKED ROW: sum over [lo, lo+len) of
 * code(row, i) * x[i], with row and x both indexed by absolute column.
 *
 * ⚠⚠ THIS IS THE FALLBACK, WHICH IS THE ONE THAT HAS TO BE RIGHT. Everything
 * routed to the hardware can arrive back here at run time -- a width the batch
 * refuses, a buffer object that would not allocate, a job that timed out -- and
 * llama.c's matvec_again takes that path SILENTLY on purpose, because a run
 * that degrades to the CPU still finishes and still says which shape stopped
 * answering. Which means a reader one nibble out of step with q_put would not
 * raise anything. It would answer, in whole sentences, slightly wrong.
 *
 * ⚠ THE VECTOR PATH READS BYTE i/2 AND TAKES ITS LOW NIBBLE AS COLUMN i, so it
 * only means that when the start is EVEN. Groups begin at multiples of
 * CHARSIU_NPU_W4_GROUP, which is 1024 on the board and even at every shape this
 * has ever run -- but an odd group size that divides an odd k starts a group on
 * an odd column: at k = 33 with groups of 11, columns 11 and 22 both do. One
 * scalar step first fixes the alignment. That is the same correction cpu_rows
 * in npudev carries, and it is there because the test caught it: 2.26 relative
 * at k = 34 with groups of 17, against 1e-6 everywhere else.
 */
static int32_t qdot(const int8_t *row, const int8_t *x, uint64_t lo,
		    uint64_t len)
{
	uint64_t i = lo, hi = lo + len;
	int32_t s = 0;

	if ((i & 1) && i < hi) {
		s += (int32_t)q_at(row, i, 1) * (int32_t)x[i];
		i++;
	}
#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
	{
	int32x4_t acc = vdupq_n_s32(0);

	for (; i + 16 <= hi; i += 16) {
		uint8x8_t b = vld1_u8((const uint8_t *)row + (i >> 1));
		/*
		 * The low nibbles sign extended by a shift left into the top of
		 * the byte and an arithmetic shift back, the high nibbles by
		 * the arithmetic shift alone, then zipped back into column
		 * order -- val[0] is columns i..i+7 and val[1] is i+8..i+15.
		 * Same unpack as cpu_rows; this one accumulates in integers
		 * because the activation here is already the int8 q1 form.
		 */
		int8x8_t l = vshr_n_s8(vshl_n_s8(
			vreinterpret_s8_u8(vand_u8(b, vdup_n_u8(0x0f))), 4), 4);
		int8x8_t h = vshr_n_s8(vreinterpret_s8_u8(b), 4);
		int8x8x2_t z = vzip_s8(l, h);
		int8x16_t w = vcombine_s8(z.val[0], z.val[1]);
		int8x16_t v = vld1q_s8(x + i);

		/* |code| <= 8 and |x| <= 127, so the widening product is 1016
		 * at worst and int16 has room for it before the pairwise add */
		acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w),  vget_low_s8(v)));
		acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(w), vget_high_s8(v)));
	}
	s += vaddvq_s32(acc);
	}
#endif
	for (; i < hi; i++)
		s += (int32_t)q_at(row, i, 1) * (int32_t)x[i];
	return s;
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
	const int pk = t->packed;
	const size_t stride = npu_q_stride_t(t);

	/*
	 * The k factor rides on the ACTIVATION, so a tensor that has one needs
	 * its own quantisation of the same input vector. On the board this is
	 * one multiply a k before the pack; here it is done straight so the
	 * measurement is honest.
	 *
	 * ⚠ PER CALL, NOT static. npu_matvec runs on the worker threads, and a
	 * static scratch buffer here was a race and a double free: three of
	 * the five exponents in the first sweep produced no output at all.
	 *
	 * ⚠⚠ AND IT WAS PER ROW, WHICH THE PARAGRAPH ABOVE ALREADY SAID IT
	 * SHOULD NOT BE. The scaled activation depends on `a` and on
	 * `t->kscale` and on nothing else -- `r` does not appear in it -- so
	 * every row of a tensor was mallocing k bytes, taking a maximum over
	 * k, quantising k values and freeing, to arrive at the same buffer the
	 * previous row had just built. On the output head that is 128256
	 * identical reconstructions of one vector, and it is why an AWQ arm of
	 * a ppl sweep runs several times longer than the arm with AWQ off.
	 *
	 * Hoisted, it is exactly the "per call" the note asked for. Same
	 * arithmetic in the same order, so the answer must not move by a
	 * digit -- which is the only test that can prove it.
	 */
	const int8_t *aq = a->q1;
	int8_t *free_after = NULL;
	float ad = a->d1;

	if (getenv("CHARSIU_CALIB") && row0 == 0)
		npu_calib_note((struct npu_tensor *)t, a);
	if (t->kscale) {
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
	for (uint64_t r = 0; r < nrows; r++) {
		uint64_t n = row0 + r;
		const int8_t *qr = t->q + n * stride;

		uint64_t grp = t->kgroup ? t->kgroup : t->k;
		uint64_t ngrp = (t->k + grp - 1) / grp;
		double acc = 0.0;

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
		if (charsiu_env_flag("CHARSIU_NPU_A16", 0)) {
			for (uint64_t g = 0; g < ngrp; g++) {
				uint64_t lo = g * grp;
				uint64_t len = lo + grp < t->k ? grp : t->k - lo;
				double part = 0.0;

				for (uint64_t i = lo; i < lo + len; i++) {
					double av = a->f[i];

					if (t->kscale)
						av *= t->kscale[i];
					part += (double)q_at(qr, i, pk) * av;
					if (midrise_grid())
						part += 0.5 * av;
				}
				acc += part * t->scale[n * ngrp + g];
			}
			y[n] = (float)acc;
			continue;
		}
		for (uint64_t g = 0; g < ngrp; g++) {
			uint64_t lo = g * grp;
			uint64_t len = lo + grp < t->k ? grp : t->k - lo;
			double part = pk
				? (double)qdot(qr, aq, lo, len)
				: (double)idot(qr + lo, aq + lo, len);

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
	}
	/* ⚠ ONCE, AFTER THE LOOP. It used to be freed at the bottom of every
	 * iteration, which was consistent with allocating at the top of every
	 * iteration and is a double free the moment either one moves. */
	free(free_after);
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

void npu_quantise_output(struct npu_tensor *t, float *y, uint64_t n, int mode,
			 float a_scale)
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

		if (t->out_calls < cal && a_scale > 0.0f)
			t->out_ascale = a_scale;
		if (t->out_calls < cal) {
			float want = amax / 127.0f;

			if (want > t->out_scale)
				t->out_scale = want;
		}
		d = t->out_scale;
		/*
		 * ⚠⚠ MODE 4: THE FROZEN PART IS PER CHANNEL, THE MOVING PART
		 * IS THE ACTIVATION'S OWN SCALE.
		 *
		 * The note above says a coefficient buffer cannot look at the
		 * vector it is about to quantise, and that is true. It does not
		 * need to. The output of a projection is
		 *
		 *     sum(w_q * a_q) * w_scale * a_scale
		 *
		 * so its magnitude is PROPORTIONAL to a_scale, which is the
		 * activation's own absmax over 127 and is computed at pack
		 * time, before the dispatch. Freeze the rest and let a_scale
		 * carry the per-token movement -- which is exactly the 2971x
		 * that job.c refused a byte for.
		 *
		 * The calibration therefore records amax / a_scale rather than
		 * amax, and this multiplies it back.
		 */
		if (mode >= 4 && t->out_ascale > 0.0f && a_scale > 0.0f)
			d = t->out_scale * (a_scale / t->out_ascale);
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
