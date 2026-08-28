/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * The vision tower: a ViT read out of llama.cpp's `mmproj-*.gguf`, producing
 * the embeddings a multimodal model splices into its token stream.
 *
 * ⚠ WHY THIS IS NOT A CONVOLUTION. A patch embedding is a k x k convolution
 * with stride k, which is to say the patches do not overlap -- so it is a
 * gather into rows followed by one matmul, and charsiu's job encoder, which
 * only ever speaks (m, k, n), can express the whole tower. The real 2D
 * convolution lives in Mesa and is not needed here.
 *
 * ⚠ AND THE TOWER WANTS int8. An image arrives as 256 to 1600 patches AT ONCE,
 * which is exactly the batched matmul measured on 2026-08-27 at 2.94x. w4a16
 * computes exactly one row whatever it is asked for, so an int4 tower would run
 * those patches one at a time -- 256 dispatches where int8 does one.
 */
#ifndef CHARSIU_VISION_H
#define CHARSIU_VISION_H

#include <stdint.h>
#include "charsiu_llm.h"

enum charsiu_proj {
	CHARSIU_PROJ_NONE = 0,
	CHARSIU_PROJ_MLP,          /* two matmuls with a GELU between them */
	CHARSIU_PROJ_UNKNOWN,
};

struct charsiu_vision_layer {
	const struct gguf_tensor *ln1_w, *ln1_b;
	const struct gguf_tensor *q_w, *q_b, *k_w, *k_b, *v_w, *v_b;
	const struct gguf_tensor *o_w, *o_b;
	const struct gguf_tensor *ln2_w, *ln2_b;
	const struct gguf_tensor *up_w, *up_b, *down_w, *down_b;
};

/*
 * ⚠ EVERY NAME IN HERE IS UNVERIFIED AGAINST A REAL FILE. They are llama.cpp's
 * clip names as this tree understands them, and the whole point of the loader
 * below is that a name it does not find is REPORTED rather than skipped.
 *
 * gemma4 is why. Three of its per layer tensors have names shorter than the
 * model wide ones they belong to, the guess found nothing three times, the
 * whole per layer path was silently skipped, and the model loaded, ran,
 * answered, and was missing the half of itself its name is about.
 */
struct charsiu_vision {
	struct gguf g;
	int opened;

	uint32_t image_size, patch_size, n_embd, n_ff, n_head, n_layer;
	uint32_t proj_dim, grid, n_patches;
	float eps;
	float mean[3], std[3];
	enum charsiu_proj proj;

	const struct gguf_tensor *patch_w, *patch_b, *pos_embd;
	const struct gguf_tensor *pre_ln_w, *pre_ln_b, *post_ln_w, *post_ln_b;
	const struct gguf_tensor *mm_w[2], *mm_b[2];
	struct charsiu_vision_layer *layer;

	/* what it wanted and did not find, in the order it wanted them */
	char missing[24][80];
	unsigned n_missing;
	char why[160];
};

/*
 * Open an mmproj gguf and bind every tensor the tower needs.
 *
 * Returns 0 when the tower is complete and usable, -1 otherwise, and in BOTH
 * cases fills v->missing and v->why. A caller that only wants to report -- the
 * install's model check, `charsiu show` -- reads those after either result.
 */
int charsiu_vision_open(struct charsiu_vision *v, const char *path);
void charsiu_vision_close(struct charsiu_vision *v);

/* NULL when the tower is usable, otherwise a short phrase. */
const char *charsiu_vision_why_not(const struct charsiu_vision *v);

/* One line per hparam and per missing tensor, for a person to read. */
void charsiu_vision_describe(const struct charsiu_vision *v, FILE *out);

#endif /* CHARSIU_VISION_H */
