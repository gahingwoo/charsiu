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
	CHARSIU_PROJ_IDEFICS3,     /* a pixel shuffle, then one fc */
	CHARSIU_PROJ_CLIP,         /* the class token, then one projection */
	CHARSIU_PROJ_UNKNOWN,
};

/*
 * ⚠ fc1 AND fc2, NOT up AND down. In a real mmproj the feed forward is named
 * the other way round from the language model's: v.blk.N.ffn_down is the FIRST
 * matmul, n_embd -> n_ff, and ffn_up is the second. Reading a real
 * SmolVLM-256M mmproj is what said so -- ffn_down.weight is (768, 3072) and
 * ffn_up.weight is (3072, 768).
 *
 * So these are bound BY SHAPE, not by name: whichever of the two contracts over
 * n_embd is fc1. A square feed forward would make the two indistinguishable,
 * and there is no such thing in a ViT.
 */
struct charsiu_vision_layer {
	const struct gguf_tensor *ln1_w, *ln1_b;
	const struct gguf_tensor *q_w, *q_b, *k_w, *k_b, *v_w, *v_b;
	const struct gguf_tensor *o_w, *o_b;
	const struct gguf_tensor *ln2_w, *ln2_b;
	const struct gguf_tensor *fc1_w, *fc1_b, *fc2_w, *fc2_b;
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
	/*
	 * ⚠ AN IMAGE IS NOT ONE TOKEN PER PATCH. idefics3 rearranges the patch
	 * grid by scale_factor before the projector -- 32 x 32 patches at
	 * scale 4 is 64 embeddings of 768 * 16, not 1024 of 768 -- so the
	 * number of tokens an image costs the language model is n_patches
	 * divided by scale_factor squared.
	 */
	uint32_t scale;
	int use_gelu;
	float eps;
	float mean[3], std[3];
	enum charsiu_proj proj;

	const struct gguf_tensor *patch_w, *patch_b, *pos_embd;
	const struct gguf_tensor *pre_ln_w, *pre_ln_b, *post_ln_w, *post_ln_b;
	const struct gguf_tensor *mm_w[2], *mm_b[2];   /* the mlp projector */
	const struct gguf_tensor *fc_w, *fc_b;         /* idefics3's single fc */
	/*
	 * ⚠ CLIP PREPENDS A CLASS TOKEN, so the position embedding has one more
	 * row than there are patches and the image's single embedding is that
	 * token's, post normalised and projected -- not the patches at all.
	 * A tower for a language model hands over every patch; a tower for
	 * retrieval hands over one vector. The same twelve blocks either way.
	 */
	const struct gguf_tensor *class_embd, *vproj_w;
	struct charsiu_vision_layer *layer;

	/* what it wanted and did not find, in the order it wanted them */
	char missing[24][80];
	unsigned n_missing;
	char why[160];

	/*
	 * ⚠ THE TOWER'S OWN NPU POOL, opened in int8 whatever the environment
	 * says. A picture is 1024 patches against weights that do not change,
	 * which is the batched matmul; w4a16 makes one row, so an int4 device
	 * here would dispatch those patches one at a time and be slower than
	 * the CPU it was moved off.
	 */
	struct charsiu_npu_pool pool;
	int npu;
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

/* Where the time went, under CHARSIU_STAGES. */
void charsiu_vision_stages(FILE *out);

/* One line per hparam and per missing tensor, for a person to read. */
void charsiu_vision_describe(const struct charsiu_vision *v, FILE *out);

/* How many embeddings one image becomes, and how wide each one is. */
unsigned charsiu_vision_tokens(const struct charsiu_vision *v);
unsigned charsiu_vision_width(const struct charsiu_vision *v);

/*
 * One image in, its embeddings out.
 *
 * `px` is [3][image_size][image_size] f32 ALREADY NORMALISED -- (v - mean) / std
 * with the tower's own numbers, which charsiu_vision_normalise does. `out` holds
 * charsiu_vision_tokens() * charsiu_vision_width() floats.
 *
 * ⚠ NO CAUSAL MASK. A ViT's attention is full: patch 0 sees patch 255. Carrying
 * the language model's mask in here would be a picture that can only see up and
 * to the left of itself, which is a plausible looking answer about the wrong
 * image.
 */
int charsiu_vision_encode(struct charsiu_vision *v, const float *px, float *out);

/*
 * The tower's self attention on its own: q, k, v and o are [n][W] with the
 * heads laid side by side across W, and `scale` is 1/sqrt(head_dim).
 *
 * ⚠ EXPOSED BECAUSE IT IS THE HALF OF THE ENCODE THAT NEEDS MEASURING and the
 * stage table cannot see it clearly on a host: here the matmuls are CPU and
 * dwarf it, on the board they are NPU and it is half the run. tools/vattn_bench
 * drives this directly so a change can be timed at the board's shape without
 * the other six stages moving underneath it.
 */
void charsiu_vision_attention(const float *q, const float *k, const float *v,
			      float *o, unsigned n, unsigned W,
			      unsigned n_head, float scale);

/* (v - mean) / std, per channel, in place. */
void charsiu_vision_normalise(const struct charsiu_vision *v, float *px);

/*
 * A jpeg, png, bmp or gif off the disk as [3][side][side] f32 in 0..1, resized
 * bilinearly on the half pixel centres. NOT normalised: pass it through
 * charsiu_vision_normalise, which knows the tower's own mean and std.
 *
 * Returns a malloc'd buffer, or NULL with a sentence in `err`.
 */
float *charsiu_image_load(const char *path, unsigned side, char *err,
			  size_t errlen);

#endif /* CHARSIU_VISION_H */
