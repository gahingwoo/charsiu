// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * WHICH TENSORS COULD HAVE THEIR OUTPUT READ BACK AS fp16, AND WHAT SHARE OF
 * THE READ THAT IS.
 *
 * charsiu reads the raw int32 accumulator (job.acc_out) on every dispatch:
 * four bytes an output element, every K slice, every row of a prefill. Round
 * 165 priced it -- prefill reads 0.94 ms a row at four bytes an element, and
 * two bytes would be 0.47 -- against a TTFT gap to the vendor of 0.67 ms a row
 * on int8. So the whole gap is inside one read width.
 *
 * ⚠⚠ THE BOUND IS A STATIC PROPERTY OF THE WEIGHTS, WHICH IS WHY THIS TOOL
 * EXISTS. At int8 the group is the whole row, so the DPU's per-channel requant
 * can apply exactly the scale the CPU applies now, and the hardware would emit
 * the requantised value in units of the activation scale. Every |a_q| <= 127,
 * so the largest magnitude a channel can produce is
 *
 *     127 * sum_k |w[c][k]|
 *
 * with w the DEQUANTISED weight. If that fits fp16's 65504 the narrow read is
 * exact; if it does not, the overflow is an inf and the token is destroyed. So
 * only the worst case can gate it, and the worst case is computable here.
 *
 * ⚠ IT IS A WORST CASE AND SAYS SO. It assumes every |a_q| is 127 and every
 * sign agrees. Real activations run about a third of that, so a tensor over
 * the line would probably not overflow in practice -- and "probably" is not a
 * thing to gate an inf on.
 *
 * ⚠ AND IT IS AN int8 QUESTION. At four bits the group is 1024 and the requant
 * is per channel, so the hardware cannot apply a per-group scale and the raw
 * accumulator is the only correct read. Nothing here applies to w4a16.
 *
 * ⚠⚠ AND THE K SPLIT IS A SECOND GATE, BUT ONLY FOR ONE OF THE TWO WIDTHS.
 * This is worth getting right because the obvious version of it is wrong.
 *
 * What makes a K split free today is `acc_out`: the hardware writes the raw
 * int32 accumulator and the CPU adds the slices, so a tensor wider than KMAX
 * costs nothing at all. A requantised output is a different object, and what
 * it costs depends on whether its precision is RELATIVE or ABSOLUTE.
 *
 *   fp16   relative, about 2^-11 of whatever the partial sum is. Four
 *          partials each carrying that, summed, still carry about that. The
 *          split is fine and only the 65504 bound gates it.
 *   int8   absolute, against a scale that has to be fixed before the
 *          dispatch. A partial sum of a quarter of K is smaller than the
 *          total by roughly the square root of four, so a scale chosen for
 *          the total spends two of the eight bits on range the partial never
 *          uses -- and the four roundings add. A four-way split is nearer six
 *          bits than eight.
 *
 * So the second percentage below gates the INT8 read and not the fp16 one,
 * and it is the interesting number because one byte saves 0.70 ms a row where
 * two saves 0.47 and the gap is 0.67.
 *
 * ⚠ THE WEIGHTS ARE THE GGUF'S, NOT charsiu'S QUANTISATION OF THEM. The
 * hardware would emit sum a_q * w_q * w_scale and this sums |w| off the file.
 * They differ by the quantiser's error on a sum of thousands of magnitudes,
 * which is far below the 2.1x margins that decide anything here -- but if a
 * tensor ever lands within a few percent of the line, that is the reason not
 * to trust this tool about it.
 *
 *   out16_bound MODEL.gguf [MODEL.gguf ...] [--kmax N]
 *
 * The last column is the one that decides whether the feature is worth a board
 * round: what share of the output elements a prefill reads are on tensors that
 * could be read narrow.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "charsiu_llm.h"

#define FP16_MAX 65504.0

/*
 * The tensors charsiu routes to the NPU: a 2-D weight it multiplies by.
 *
 * ⚠⚠ BY PROPERTY, NOT BY SPELLING, and the first version of this was by
 * spelling. A list of eight suffixes -- attn_q, attn_k, attn_v, ... -- read
 * Phi-3.5 as having no attention weights at all, because Phi fuses them into
 * `attn_qkv.weight` and no suffix matched. It reported four kinds where the
 * model has five and said nothing about the missing one, which for a tool
 * whose whole output is a coverage percentage is the worst possible failure.
 *
 * So: any 2-D tensor big enough to be a projection, minus the one named
 * exception.
 *
 * ⚠⚠ AND THE EXCEPTION IS CONDITIONAL, which the first version of it was not.
 * token_embd is a LOOKUP and output.weight is the head -- EXCEPT where the
 * model ties them, and then llama.c does `m->output = m->tok_embd` and
 * token_embd IS the head. Excluding it there drops the single widest tensor
 * in the model: Llama-3.2-1B's is 128256 channels against 8192 for the next
 * biggest, so a percentage computed without it is not slightly off, it is
 * missing most of its own denominator.
 */
static int routed(const struct gguf_tensor *t, int tied)
{
	if (t->n_dims != 2 || t->ne[0] < 32 || t->ne[1] < 32)
		return 0;
	/*
	 * ⚠ EXACT, NOT A SUBSTRING. gemma4's per_layer_token_embd CONTAINS
	 * "token_embd" and is a lookup in every model that has it, tied or
	 * not -- llama.c reads a row of it per token. A substring test made
	 * it the head on gemma4 and put a vocabulary-wide tensor into the
	 * denominator of the percentage this tool exists to print.
	 */
	if (!strcmp(t->name, "token_embd.weight"))
		return tied;      /* the head, when there is no output.weight */
	if (strstr(t->name, "token_embd"))
		return 0;         /* per_layer_token_embd and friends: lookups */
	return 1;
}

/* "blk.13.ffn_down.weight" -> "ffn_down", "output.weight" -> "output" */
static void kindof(const char *name, char *out, size_t max)
{
	const char *b = strstr(name, "blk.");
	const char *s = b ? strchr(b + 4, '.') : NULL;
	const char *from = s ? s + 1 : name;
	const char *dot = strrchr(from, '.');
	size_t n = dot ? (size_t)(dot - from) : strlen(from);

	if (n >= max) n = max - 1;
	memcpy(out, from, n);
	out[n] = '\0';
}

struct kind {
	char name[48];
	double worst;        /* the largest 127*sum|w| over every channel */
	char worst_in[80];
	uint64_t nchan;      /* output elements a prefill row reads, per slice */
	uint64_t nchan_ok;   /* of them, on tensors under the line */
	uint64_t nchan_1ks;  /* of them, on tensors with ONE K slice */
	unsigned ntensor, nover, nsplit;
};

static struct kind *find(struct kind *k, unsigned *n, const char *name)
{
	for (unsigned i = 0; i < *n; i++)
		if (!strcmp(k[i].name, name))
			return &k[i];
	snprintf(k[*n].name, sizeof(k[*n].name), "%s", name);
	return &k[(*n)++];
}

static int one(const char *path, unsigned kmax)
{
	struct gguf g;
	struct kind kinds[16];
	unsigned nk = 0;
	uint64_t tot = 0, ok = 0, one_ks = 0;
	float *row = NULL;
	uint64_t rowmax = 0;
	int tied;

	memset(kinds, 0, sizeof(kinds));
	if (gguf_open(&g, path)) {
		fprintf(stderr, "%s: will not open\n", path);
		return 1;
	}
	/* llama.c's own rule: no output.weight, or an empty one, means tied */
	{
		const struct gguf_tensor *o = gguf_tensor(&g, "output.weight");

		tied = !o || !o->nbytes;
	}
	printf("\n%s%s\n", path, tied ? "   (tied head: token_embd IS the head)" : "");
	for (uint64_t i = 0; i < g.n_tensors; i++) {
		const struct gguf_tensor *t = &g.t[i];
		char kn[48];
		struct kind *k;
		double worst = 0.0;

		if (!routed(t, tied))
			continue;
		if (t->ne[0] > rowmax) {
			free(row);
			rowmax = t->ne[0];
			row = malloc((size_t)rowmax * sizeof(*row));
			if (!row) {
				fprintf(stderr, "out of memory\n");
				gguf_close(&g);
				return 1;
			}
		}
		for (uint64_t c = 0; c < t->ne[1]; c++) {
			double s = 0.0;

			gguf_row_f32(t, c, row);
			for (uint64_t j = 0; j < t->ne[0]; j++)
				s += fabs((double)row[j]);
			s *= 127.0;
			if (s > worst)
				worst = s;
		}
		kindof(t->name, kn, sizeof(kn));
		k = find(kinds, &nk, kn);
		k->ntensor++;
		/*
		 * ⚠ PER K SLICE, because that is what is READ. Every slice
		 * writes the tensor's full n outputs and the CPU sums them,
		 * so a tensor cut four ways is read four times over -- which
		 * is exactly why ffn_down dominates the read and exactly why
		 * it is the one a narrow output cannot have.
		 */
		{
			uint64_t ks = (t->ne[0] + kmax - 1) / kmax;

			k->nchan += t->ne[1] * ks;
			tot += t->ne[1] * ks;
			if (ks == 1) {
				k->nchan_1ks += t->ne[1];
				one_ks += t->ne[1];
			} else {
				k->nsplit++;
			}
			if (worst <= FP16_MAX) {
				k->nchan_ok += t->ne[1] * ks;
				ok += t->ne[1] * ks;
			} else {
				k->nover++;
			}
		}
		if (worst > k->worst) {
			k->worst = worst;
			snprintf(k->worst_in, sizeof(k->worst_in), "%s", t->name);
		}
	}
	free(row);
	gguf_close(&g);
	if (!tot) {
		puts("  no routed tensors -- is this a language model?");
		return 0;
	}
	printf("  %-14s %7s %5s %6s  %12s  %6s  %s\n",
	       "kind", "tensors", "over", "split", "worst 127S|w|", "of max",
	       "worst in");
	for (unsigned i = 0; i < nk; i++) {
		struct kind *k = &kinds[i];

		printf("  %-14s %7u %5u %6u  %12.0f  %5.2fx  %s\n",
		       k->name, k->ntensor, k->nover, k->nsplit, k->worst,
		       k->worst / FP16_MAX, k->nover ? k->worst_in : "-");
	}
	printf("  ----\n");
	printf("  of the output elements a prefill row reads, at KMAX %u:\n", kmax);
	printf("    %5.1f%%  are on tensors under fp16's 65504\n",
	       100.0 * (double)ok / (double)tot);
	printf("    %5.1f%%  are on tensors with ONE K slice -- the share an "
	       "int8 read could\n            take at full precision; fp16 is "
	       "relative and does not need this\n",
	       100.0 * (double)one_ks / (double)tot);
	return 0;
}

int main(int argc, char **argv)
{
	int bad = 0, nfile = 0;
	unsigned kmax = 2048;

	if (argc < 2) {
		fprintf(stderr, "usage: out16_bound MODEL.gguf [...] [--kmax N]\n");
		return 2;
	}
	for (int i = 1; i < argc; i++)
		if (!strcmp(argv[i], "--kmax") && i + 1 < argc)
			kmax = (unsigned)strtoul(argv[++i], NULL, 10);
	puts("The worst case a channel can emit, against fp16's 65504.");
	puts("⚠ WORST CASE: every |a_q| = 127 and every sign agreeing. Real "
	     "activations run about a third of that -- but an overflow is an "
	     "inf, so only the worst case can gate it.");
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--kmax")) { i++; continue; }
		bad |= one(argv[i], kmax);
		nfile++;
	}
	if (!nfile) {
		fprintf(stderr, "no model given\n");
		return 2;
	}
	return bad;
}
