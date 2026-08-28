/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * CLIP's other tower: text into the same space the pictures land in.
 *
 * A CLIP gguf is two encoders and two projections. The vision half is a ViT and
 * charsiu_vision already runs it; this is the text half, which is the same
 * twelve blocks with three differences that all matter:
 *
 *   - the attention is CAUSAL. A picture's patches all see each other; a
 *     sentence's tokens see only what came before them.
 *   - the sentence's embedding is the row at the END OF TEXT token, not a
 *     pooled average and not the first row.
 *   - its own tokenizer, which is in the gguf and is not the language model's.
 */
#ifndef CHARSIU_CLIP_H
#define CHARSIU_CLIP_H

#include <stdint.h>
#include <stdio.h>
#include "charsiu_llm.h"

struct charsiu_clip_layer {
	const struct gguf_tensor *ln1_w, *ln1_b;
	const struct gguf_tensor *q_w, *q_b, *k_w, *k_b, *v_w, *v_b;
	const struct gguf_tensor *o_w, *o_b;
	const struct gguf_tensor *ln2_w, *ln2_b;
	const struct gguf_tensor *fc1_w, *fc1_b, *fc2_w, *fc2_b;
};

struct charsiu_clip_text {
	struct gguf g;
	int opened;

	uint32_t n_embd, n_ff, n_head, n_layer, n_ctx, proj_dim;
	int use_gelu;
	float eps;

	const struct gguf_tensor *tok_embd, *pos_embd;
	const struct gguf_tensor *post_ln_w, *post_ln_b, *proj_w;
	struct charsiu_clip_layer *layer;

	/* the vocabulary, in merge order: see charsiu_clip_tokenize */
	char **vocab;
	uint32_t n_vocab;
	int32_t sot, eot;

	char missing[24][80];
	unsigned n_missing;
	char why[160];
};

int charsiu_clip_text_open(struct charsiu_clip_text *t, const char *path);
void charsiu_clip_text_close(struct charsiu_clip_text *t);
const char *charsiu_clip_text_why_not(const struct charsiu_clip_text *t);
void charsiu_clip_text_describe(const struct charsiu_clip_text *t, FILE *out);

/*
 * CLIP's BPE, out of the vocabulary alone.
 *
 * ⚠ THE gguf CARRIES NO MERGE LIST. clip.cpp never ran the text tower, so its
 * converter wrote tokenizer.ggml.tokens and nothing else. The ranks are
 * recoverable anyway, because CLIP's vocabulary IS the merge order: 256 byte
 * symbols, then the same 256 with the end of word marker, then every merge in
 * the order it was learned. So "merge the adjacent pair whose concatenation has
 * the lowest index" is the same algorithm as merging by rank, and
 * tests/clip_tokenizer_cross.py checks that against the real merges.txt.
 *
 * ⚠ ASCII WORD SPLITTING. CLIP's pattern is a unicode one and this splits on
 * ASCII letter, digit and everything-else runs. English prompts, which is what
 * these models are trained on, tokenize identically; a prompt with accents or
 * CJK in it will not, and charsiu_clip_tokenize says so rather than guessing.
 *
 * Writes at most `max` ids INCLUDING the two markers. Returns the count, or -1.
 */
int charsiu_clip_tokenize(const struct charsiu_clip_text *t, const char *text,
			  int32_t *ids, int max, int *non_ascii);

/* The sentence's embedding, proj_dim wide. */
int charsiu_clip_encode_text(struct charsiu_clip_text *t, const int32_t *ids,
			     int n, float *out);

/* a . b / (|a| |b|) */
float charsiu_cosine(const float *a, const float *b, unsigned n);

#endif /* CHARSIU_CLIP_H */
