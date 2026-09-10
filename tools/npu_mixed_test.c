// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * CAN ONE OPEN DEVICE ALTERNATE w8a8 AND w4a16 JOBS AND GET BOTH RIGHT?
 *
 * This is the whole of what blocks CHARSIU_NPU_INT8_LAYERS from reaching the
 * hardware, asked without writing the change it would justify.
 *
 * The knob keeps the first blocks at eight bits and the rest at four, and it
 * is worth having: on Llama-3.2-1B it takes the host reference from 41.53 to
 * 26.07 for 12.5% of the bytes, and on Qwen3-0.6B from 110.05 to 85.09 for
 * 7.1%. On the board it does nothing at all, because charsiu_npu_add refuses
 * an eight-bit tensor on a device opened for four and those tensors fall back
 * to the CPU.
 *
 * ⚠⚠ AND THE CHEAP WAY ROUND IT IS DOMINATED, which is why this probe exists
 * rather than a patch. The OTHER direction of the mismatch already works and
 * is shipped: an int8 DEVICE reading int4 codes is what every vision tower
 * does (pack_rows tests t->packed and g->w4 separately and has since it was
 * written). So a mixed model could simply be opened as int8 today, with no
 * new hardware path -- and that is strictly worse than not bothering:
 *
 *     all int8                       ppl 17.98,  int8 weight bytes
 *     INT8_LAYERS=0-1 as int8        ppl 26.07,  int8 weight bytes
 *
 * Same bytes on the wire, worse answer. The knob is only worth anything if
 * the int4 DEVICE runs the int8 tensors, because its entire value is int8's
 * quality at int4's bandwidth. So the question is the one in the title.
 *
 * ⚠ THE ACTIVATION COSTS ALMOST NOTHING AT FOUR BITS, measured on the host
 * reference the same day: w4a8 41.5289, w4a16 40.7016, a 2.0% gap. That
 * matters here because a mixed dispatch would hand the int4 tensors whatever
 * activation the int8 ones need if sharing one pack turned out to be easier
 * than two. It is worth 2%, so it is not a reason to refuse.
 *
 *   npu_mixed_test [K] [N] [--loop N] [--dry]
 *
 * --dry needs no hardware: it builds both jobs, emits both register streams
 * and checks they are two distinct programs. Run it before carrying the tool
 * to the board.
 *
 * ⚠⚠ USE A REAL SHAPE. K=16 N=8 and K=64 N=8 WEDGE THE NPU -- see the note at
 * the top of npu_fp16_test.c, which cost six wrong explanations. The defaults
 * here are K=256 N=64, which ran 32 of 32 with a clean dmesg.
 *
 * The order matters and is the point:
 *
 *   1  int8 alone          the control. If this is wrong, nothing below means
 *   2  int4 alone          anything, and the same for its own arm.
 *   3  int8, int4, int8    the question. A device that carries state between
 *      ... alternating     jobs -- a cached descriptor, a mode latched in a
 *                          register the emitter only writes on a change --
 *                          fails HERE and passes 1 and 2.
 *
 * ⚠ AN ALTERNATION THAT PASSES ONCE HAS PROVED LITTLE. --loop runs the
 * alternation N times; a latch that takes two switches to show, or a state
 * that only leaks when a buffer is reused, needs the repeats.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "charsiu.h"

/* A byte around 128 that depends on both indices with no short period. */
static uint8_t mix8(unsigned a, unsigned b, unsigned span)
{
	uint32_t h = a * 2654435761u ^ (b + 0x9e3779b9u) * 40503u;

	h ^= h >> 13;
	return (uint8_t)(128 + (int)(h % span) - (int)(span / 2));
}

/* a signed int4 code, [-8, 7] */
static int mix4(unsigned a, unsigned b)
{
	uint32_t h = a * 2246822519u ^ (b + 0x85ebca6bu) * 2654435761u;

	h ^= h >> 15;
	return (int)(h % 16u) - 8;
}

/*
 * ⚠ THE ACTIVATION IS EXACTLY REPRESENTABLE IN fp16 ON PURPOSE. The question
 * is whether the two programs coexist, not what fp16 rounds to, and a
 * reference that disagrees in the last bit would be read as the former.
 * Sixteenths in [-8, 8) are exact in fp16 and in float.
 */
static float mixf(unsigned a, unsigned b)
{
	uint32_t h = a * 40503u ^ (b + 0x27d4eb2fu) * 2246822519u;

	h ^= h >> 11;
	return (float)((int)(h % 256u) - 128) / 16.0f;
}

struct bufs {
	struct charsiu_bo wt, in, ob, coef, reg;
};

static void bufs_free(struct charsiu_device *dev, struct bufs *b)
{
	charsiu_bo_free(dev, &b->reg);  charsiu_bo_free(dev, &b->coef);
	charsiu_bo_free(dev, &b->ob);   charsiu_bo_free(dev, &b->in);
	charsiu_bo_free(dev, &b->wt);
}

/*
 * Build one job's buffers and fill the ones that never change. int4 and int8
 * differ in three places and this is all three: the dtypes, the byte the
 * packer is handed for a weight, and whether the input is packed as fp16 or
 * as an unsigned byte around 128.
 */
static int build(struct charsiu_device *dev, struct charsiu_job *job,
		 struct bufs *b, int w4, unsigned k, unsigned n,
		 const float *Af, const uint8_t *A8, const int8_t *W)
{
	size_t insz;
	int32_t *zero;

	memset(job, 0, sizeof(*job));
	memset(b, 0, sizeof(*b));
	job->cbuf_window = (unsigned)charsiu_cbuf_window();
	job->mm.m = 1;
	job->mm.k = k;
	job->mm.n = n;
	job->mm.wdtype = w4 ? CHARSIU_INT4 : CHARSIU_INT8;
	job->mm.adtype = w4 ? CHARSIU_FP16 : CHARSIU_INT8;
	job->input_zero_point = 128;
	job->weight_zero_point = 128;
	job->output_zero_point = 0;
	job->input_scale = 1.0f;
	job->weight_scale = 1.0f;
	job->output_scale = 1.0f;
	job->acc_out = 1;

	insz = (size_t)charsiu_entries_per_row(&job->mm) * 64 + 4096;
	if (charsiu_bo_alloc(dev, charsiu_weight_bytes(&job->mm) + 4096, &b->wt) ||
	    charsiu_bo_alloc(dev, insz, &b->in) ||
	    charsiu_bo_alloc(dev, (size_t)n * 4 + 4096, &b->ob) ||
	    charsiu_bo_alloc(dev, charsiu_coef_bytes(&job->mm) + 4096, &b->coef) ||
	    charsiu_bo_alloc(dev, 4096, &b->reg)) {
		fprintf(stderr, "  a buffer would not allocate\n");
		return -1;
	}
	if (!b->wt.map || !b->in.map || !b->ob.map || !b->coef.map || !b->reg.map) {
		fprintf(stderr, "  a buffer allocated but did not map\n");
		return -1;
	}

	/*
	 * ⚠ THE BYTE, NOT THE VALUE. int8 wants the code around a zero point
	 * of 128; int4 wants the signed code in the low nibble, which two's
	 * complement already puts there for anything in [-8, 7]. This is
	 * pack_rows' own rule, kept in one place there and repeated here on
	 * purpose: if the two ever disagree, this probe is measuring itself.
	 */
	{
		uint8_t *tmp = malloc((size_t)k * n);

		if (!tmp) { fprintf(stderr, "  out of memory\n"); return -1; }
		for (size_t i = 0; i < (size_t)k * n; i++)
			tmp[i] = w4 ? (uint8_t)((unsigned)W[i] & 0xfu)
				    : (uint8_t)((int)W[i] + 128);
		charsiu_bo_prep(dev, &b->wt, 1000000000);
		memset(b->wt.map, 0, charsiu_weight_bytes(&job->mm));
		charsiu_pack_weights(&job->mm, tmp, b->wt.map);
		charsiu_bo_fini(dev, &b->wt);
		free(tmp);
	}

	charsiu_bo_prep(dev, &b->in, 1000000000);
	if (w4)
		charsiu_pack_input_f16(&job->mm, Af, b->in.map, insz);
	else
		charsiu_pack_input(&job->mm, A8, b->in.map, insz,
				   job->input_zero_point);
	charsiu_bo_fini(dev, &b->in);

	/*
	 * Zeros for both, and for the same reason npu_gemm_test gives: the
	 * (in_zp - 0x80) factor that multiplies the weight sum is exactly zero
	 * at a zero point of 128, and there is no bias here. int4 never wants
	 * the sum at all -- add_slice takes it only when !g->w4.
	 */
	zero = calloc(n, sizeof(int32_t));
	if (!zero) { fprintf(stderr, "  out of memory\n"); return -1; }
	charsiu_bo_prep(dev, &b->coef, 1000000000);
	charsiu_build_coefs(job, zero, zero, b->coef.map);
	charsiu_bo_fini(dev, &b->coef);
	free(zero);

	job->input_addr = (uint32_t)b->in.dma_address;
	job->output_addr = (uint32_t)b->ob.dma_address;
	job->weight_addr = (uint32_t)b->wt.dma_address;
	job->coef_addr = (uint32_t)b->coef.dma_address;
	return 0;
}

/* one dispatch of an already-built job; out is n words (int32 or float) */
static int fire(struct charsiu_device *dev, struct charsiu_job *job,
		struct bufs *b, unsigned n, void *out)
{
	size_t nreg;

	charsiu_bo_prep(dev, &b->reg, 1000000000);
	nreg = charsiu_emit_job(job, b->reg.map, 4096 / 8);
	charsiu_bo_fini(dev, &b->reg);
	if (!nreg) {
		fprintf(stderr, "  the register stream came back empty\n");
		return -1;
	}
	charsiu_bo_prep(dev, &b->ob, 1000000000);
	memset(b->ob.map, 0, (size_t)n * 4);
	charsiu_bo_fini(dev, &b->ob);
	{
		uint32_t ins[2] = { b->in.handle, b->wt.handle };
		uint32_t outs[1] = { b->ob.handle };

		if (charsiu_submit(dev, &b->reg, (unsigned)nreg, ins, 2, outs, 1)) {
			fprintf(stderr, "  the submit failed\n");
			return -1;
		}
	}
	charsiu_bo_prep(dev, &b->ob, 1000000000);
	memcpy(out, b->ob.map, (size_t)n * 4);
	charsiu_bo_fini(dev, &b->ob);
	return 0;
}

int main(int argc, char **argv)
{
	unsigned k = 256, n = 64, loop = 8;
	struct charsiu_device *dev;
	struct charsiu_job j8, j4;
	struct bufs b8, b4;
	float *Af = NULL, *want4 = NULL, *got4 = NULL;
	uint8_t *A8 = NULL;
	int8_t *W = NULL;
	int32_t *want8 = NULL, *got8 = NULL;
	int rc = 1, pos = 1, bad = 0, ran = 0, dry = 0, npos = 0;

	/* ⚠ POSITION COUNTED, NOT INFERRED FROM argv INDEX. "--loop 3 512 128"
	 * read 512 as N and left K at its default, silently. */
	while (pos < argc) {
		if (!strcmp(argv[pos], "--loop") && pos + 1 < argc)
			loop = (unsigned)strtoul(argv[++pos], NULL, 10);
		else if (!strcmp(argv[pos], "--dry"))
			dry = 1;
		else if (npos++ == 0)
			k = (unsigned)strtoul(argv[pos], NULL, 10);
		else
			n = (unsigned)strtoul(argv[pos], NULL, 10);
		pos++;
	}
	if (k < 128 || n < 32) {
		fprintf(stderr, "npu_mixed_test: K >= 128 and N >= 32. "
			"Small shapes wedge the NPU -- npu_fp16_test.c has the "
			"table.\n");
		return 2;
	}
	printf("one device, K=%u N=%u, %u alternations\n", k, n, loop);

	/*
	 * ⚠ THE HALF A DESK CAN DECIDE, and it is worth having separately: a
	 * board round that dies because the emitter refuses one of the two
	 * dtypes has spent the round on a question the host could have
	 * answered. --dry builds both jobs with no buffer objects at all and
	 * emits both register streams. It proves the emitter produces two
	 * DIFFERENT programs of sane length; it proves nothing about the
	 * silicon, which is the whole rest of this file.
	 */
	if (dry) {
		uint64_t r8[4096 / 8], r4[4096 / 8];
		size_t n8, n4, diff = 0;

		memset(&j8, 0, sizeof(j8));
		memset(&j4, 0, sizeof(j4));
		for (int w4 = 0; w4 < 2; w4++) {
			struct charsiu_job *j = w4 ? &j4 : &j8;

			j->cbuf_window = (unsigned)charsiu_cbuf_window();
			j->mm.m = 1; j->mm.k = k; j->mm.n = n;
			j->mm.wdtype = w4 ? CHARSIU_INT4 : CHARSIU_INT8;
			j->mm.adtype = w4 ? CHARSIU_FP16 : CHARSIU_INT8;
			j->input_zero_point = 128;
			j->weight_zero_point = 128;
			j->input_scale = j->weight_scale = j->output_scale = 1.0f;
			j->acc_out = 1;
		}
		n8 = charsiu_emit_job(&j8, r8, sizeof(r8) / sizeof(r8[0]));
		n4 = charsiu_emit_job(&j4, r4, sizeof(r4) / sizeof(r4[0]));
		printf("  int8 program  %zu words, %.1f MB of weights\n",
		       n8, (double)charsiu_weight_bytes(&j8.mm) / 1e6);
		printf("  int4 program  %zu words, %.1f MB of weights\n",
		       n4, (double)charsiu_weight_bytes(&j4.mm) / 1e6);
		for (size_t i = 0; i < (n8 < n4 ? n8 : n4); i++)
			if (r8[i] != r4[i])
				diff++;
		printf("  %zu words differ over the shorter stream\n", diff);
		if (!n8 || !n4) {
			puts("⚠⚠ AN EMITTER REFUSED ONE OF THE TWO. There is "
			     "nothing for the board to answer yet.");
			return 1;
		}
		if (!diff && n8 == n4) {
			puts("⚠⚠ THE TWO PROGRAMS ARE IDENTICAL, which cannot "
			     "be right: the dtype fields alone must differ. "
			     "This probe is not building what it says.");
			return 1;
		}
		puts("→ the emitter produces two distinct programs. The "
		     "silicon question is the rest of this tool, on a board.");
		return 0;
	}

	Af = malloc((size_t)k * sizeof(*Af));
	A8 = malloc((size_t)k);
	W = malloc((size_t)k * n);
	want4 = malloc((size_t)n * sizeof(*want4));
	got4 = malloc((size_t)n * sizeof(*got4));
	want8 = malloc((size_t)n * sizeof(*want8));
	got8 = malloc((size_t)n * sizeof(*got8));
	if (!Af || !A8 || !W || !want4 || !got4 || !want8 || !got8) {
		fprintf(stderr, "out of memory\n");
		goto out;
	}
	for (unsigned i = 0; i < k; i++) {
		Af[i] = mixf(0, i);
		A8[i] = mix8(0, i, 64);
	}
	/*
	 * ⚠ ONE W, TWO READINGS OF IT. The same codes go into both jobs --
	 * as a byte around 128 for int8 and as a nibble for int4 -- so the two
	 * references are the same arithmetic and a difference between the arms
	 * cannot be the test data.
	 */
	for (unsigned c = 0; c < n; c++)
		for (unsigned i = 0; i < k; i++)
			W[(size_t)c * k + i] = (int8_t)mix4(c, i);

	for (unsigned c = 0; c < n; c++) {
		double a4 = 0.0;
		int32_t a8 = 0;

		for (unsigned i = 0; i < k; i++) {
			a4 += (double)W[(size_t)c * k + i] * Af[i];
			a8 += (int32_t)W[(size_t)c * k + i]
			    * ((int)A8[i] - 128);
		}
		want4[c] = (float)a4;
		want8[c] = a8;
	}

	dev = charsiu_open(NULL);
	if (!dev) { fprintf(stderr, "no NPU device\n"); goto out; }
	if (build(dev, &j8, &b8, 0, k, n, Af, A8, W) ||
	    build(dev, &j4, &b4, 1, k, n, Af, A8, W)) {
		charsiu_close(dev);
		goto out;
	}

#define CHECK8(what) do {                                                     \
		unsigned w_ = 0; long worst_ = 0;                             \
		for (unsigned c = 0; c < n; c++) {                            \
			long d = (long)got8[c] - (long)want8[c];              \
			if (d) { w_++; if (labs(d) > labs(worst_)) worst_ = d; } \
		}                                                             \
		printf("  %-26s %4u of %4u wrong, worst %ld\n", what, w_, n, worst_); \
		if (w_) bad++;                                                \
		ran++;                                                        \
	} while (0)
	/*
	 * ⚠ AN ABSOLUTE TOLERANCE, because the reference is a double and the
	 * hardware accumulates the fp16 products. The activations are exact
	 * sixteenths and the codes are integers, so every product is exact and
	 * only the SUM can drift; 1e-3 on a dot product of a few hundred terms
	 * whose magnitudes are single digits is far below one code.
	 */
#define CHECK4(what) do {                                                     \
		unsigned w_ = 0; double worst_ = 0.0;                         \
		for (unsigned c = 0; c < n; c++) {                            \
			double d = (double)got4[c] - (double)want4[c];        \
			if (fabs(d) > 1e-3) {                                 \
				w_++;                                         \
				if (fabs(d) > fabs(worst_)) worst_ = d;       \
			}                                                     \
		}                                                             \
		printf("  %-26s %4u of %4u wrong, worst %.4f\n", what, w_, n, worst_); \
		if (w_) bad++;                                                \
		ran++;                                                        \
	} while (0)

	puts("\ncontrols, each program on its own");
	if (!fire(dev, &j8, &b8, n, got8)) CHECK8("int8 alone");
	if (!fire(dev, &j4, &b4, n, got4)) CHECK4("int4 alone");
	if (bad) {
		puts("\n⚠⚠ A CONTROL FAILED. Nothing below this line means "
		     "anything: fix the arm that is wrong before reading the "
		     "alternation.");
		charsiu_close(dev);
		goto done;
	}

	puts("\nalternating on the same open device");
	for (unsigned i = 0; i < loop; i++) {
		char lbl[40];

		snprintf(lbl, sizeof(lbl), "int8 after int4 (%u)", i);
		if (!fire(dev, &j8, &b8, n, got8)) CHECK8(lbl);
		snprintf(lbl, sizeof(lbl), "int4 after int8 (%u)", i);
		if (!fire(dev, &j4, &b4, n, got4)) CHECK4(lbl);
	}
	bufs_free(dev, &b4);
	bufs_free(dev, &b8);
	charsiu_close(dev);
done:
	printf("\n%d of %d dispatches wrong\n", bad, ran);
	if (!bad && ran > 2)
		puts("→ one open device runs both programs. The per-tensor "
		     "width refactor in npudev.c is worth doing: "
		     "CHARSIU_NPU_INT8_LAYERS can reach the hardware.");
	else if (bad)
		puts("→ the two programs do NOT coexist on one open device. "
		     "INT8_LAYERS stays a host-side research knob unless a "
		     "second device is opened for the other width.");
	rc = bad != 0;
out:
	free(got8); free(want8); free(got4); free(want4);
	free(W); free(A8); free(Af);
	return rc;
}
