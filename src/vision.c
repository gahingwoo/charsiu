// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * Reading a vision tower out of an mmproj gguf.
 *
 * ⚠ THIS FILE'S ONE JOB IS TO BE LOUD. Every tensor name here is llama.cpp's
 * clip naming as this tree understands it, and none of it has been checked
 * against a real file yet. So the loader binds what it can and REPORTS what it
 * could not, by name, in the order it wanted them -- because the failure this
 * is written against is not a crash. It is gemma4: a guessed name found
 * nothing, the path was skipped in silence, and the model loaded, ran and
 * answered while missing the half of itself its name is about.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu_llm.h"
#include "charsiu_vision.h"

/*
 * Bind one tensor by name, recording a miss. `opt` marks the ones a tower can
 * legitimately be without -- a pre-layernorm, a second projector matmul -- so
 * that "absent" and "missing" stay different things.
 */
static const struct gguf_tensor *bind(struct charsiu_vision *v,
				      const char *fmt, int idx, int opt)
{
	const struct gguf_tensor *t;
	char name[80];

	if (idx >= 0)
		snprintf(name, sizeof(name), fmt, idx);
	else
		snprintf(name, sizeof(name), "%s", fmt);
	t = gguf_tensor(&v->g, name);
	if (!t && !opt && v->n_missing < sizeof(v->missing) / sizeof(*v->missing))
		snprintf(v->missing[v->n_missing++], sizeof(v->missing[0]),
			 "%s", name);
	return t;
}

/* ⚠ gguf_get_u32 RETURNS 0 ON SUCCESS. Reading it as a truth value inverts
 * every check in this file, which is how the first run of the synthetic test
 * reported "no vision encoder" about a file that had one. */
static int need(struct charsiu_vision *v, const char *key, uint32_t *out)
{
	if (gguf_get_u32(&v->g, key, out) == 0)
		return 0;
	if (v->n_missing < sizeof(v->missing) / sizeof(*v->missing))
		snprintf(v->missing[v->n_missing++], sizeof(v->missing[0]),
			 "%s (a key, not a tensor)", key);
	return -1;
}

static void read_arr3(struct charsiu_vision *v, const char *key, float *out,
		      float dflt)
{
	const struct gguf_kv *kv = gguf_find(&v->g, key);
	unsigned i;

	for (i = 0; i < 3; i++)
		out[i] = dflt;
	if (!kv || kv->arr_type != GGUF_V_F32 || kv->arr_len < 3)
		return;
	for (i = 0; i < 3; i++)
		memcpy(&out[i], kv->arr + i * sizeof(float), sizeof(float));
}

int charsiu_vision_open(struct charsiu_vision *v, const char *path)
{
	uint32_t i, has = 0;

	memset(v, 0, sizeof(*v));
	if (gguf_open(&v->g, path) < 0) {
		snprintf(v->why, sizeof(v->why), "%s will not open as a gguf",
			 path);
		return -1;
	}
	v->opened = 1;

	/*
	 * ⚠ AN LLM GGUF OPENS FINE HERE. `charsiu foo.gguf --image` pointed at
	 * the language half is the mistake a person actually makes, and without
	 * this it would come back as twenty missing tensors rather than one
	 * sentence.
	 */
	if (gguf_get_u32(&v->g, "clip.has_vision_encoder", &has) != 0 || !has) {
		snprintf(v->why, sizeof(v->why),
			 "no vision encoder in %s -- an mmproj file is a "
			 "separate download from the model", path);
		return -1;
	}

	need(v, "clip.vision.image_size", &v->image_size);
	need(v, "clip.vision.patch_size", &v->patch_size);
	need(v, "clip.vision.embedding_length", &v->n_embd);
	need(v, "clip.vision.feed_forward_length", &v->n_ff);
	need(v, "clip.vision.block_count", &v->n_layer);
	need(v, "clip.vision.attention.head_count", &v->n_head);
	if (gguf_get_u32(&v->g, "clip.vision.projection_dim", &v->proj_dim) != 0)
		v->proj_dim = 0;
	if (gguf_get_f32(&v->g, "clip.vision.attention.layer_norm_epsilon",
			 &v->eps) != 0)
		v->eps = 1e-6f;

	/*
	 * ⚠ THE PREPROCESSING IS PART OF THE MODEL. A tower trained on one
	 * normalisation and fed another does not fail, it answers about a
	 * different picture. CLIP's numbers are the default because a file
	 * without the keys is almost certainly CLIP's.
	 */
	read_arr3(v, "clip.vision.image_mean", v->mean, 0.48145466f);
	read_arr3(v, "clip.vision.image_std", v->std, 0.26862954f);

	if (v->patch_size && v->image_size) {
		v->grid = v->image_size / v->patch_size;
		v->n_patches = v->grid * v->grid;
	}

	v->patch_w   = bind(v, "v.patch_embd.weight", -1, 0);
	v->patch_b   = bind(v, "v.patch_embd.bias", -1, 1);
	v->pos_embd  = bind(v, "v.position_embd.weight", -1, 0);
	v->pre_ln_w  = bind(v, "v.pre_ln.weight", -1, 1);
	v->pre_ln_b  = bind(v, "v.pre_ln.bias", -1, 1);
	v->post_ln_w = bind(v, "v.post_ln.weight", -1, 1);
	v->post_ln_b = bind(v, "v.post_ln.bias", -1, 1);

	v->mm_w[0] = bind(v, "mm.0.weight", -1, 0);
	v->mm_b[0] = bind(v, "mm.0.bias", -1, 1);
	v->mm_w[1] = bind(v, "mm.2.weight", -1, 1);
	v->mm_b[1] = bind(v, "mm.2.bias", -1, 1);
	v->proj = v->mm_w[0] ? CHARSIU_PROJ_MLP : CHARSIU_PROJ_UNKNOWN;

	if (v->n_layer) {
		v->layer = calloc(v->n_layer, sizeof(*v->layer));
		if (!v->layer) {
			snprintf(v->why, sizeof(v->why),
				 "%u layers will not allocate", v->n_layer);
			return -1;
		}
	}
	for (i = 0; i < v->n_layer; i++) {
		struct charsiu_vision_layer *L = &v->layer[i];

		L->ln1_w  = bind(v, "v.blk.%d.ln1.weight", (int)i, 0);
		L->ln1_b  = bind(v, "v.blk.%d.ln1.bias", (int)i, 1);
		L->q_w    = bind(v, "v.blk.%d.attn_q.weight", (int)i, 0);
		L->q_b    = bind(v, "v.blk.%d.attn_q.bias", (int)i, 1);
		L->k_w    = bind(v, "v.blk.%d.attn_k.weight", (int)i, 0);
		L->k_b    = bind(v, "v.blk.%d.attn_k.bias", (int)i, 1);
		L->v_w    = bind(v, "v.blk.%d.attn_v.weight", (int)i, 0);
		L->v_b    = bind(v, "v.blk.%d.attn_v.bias", (int)i, 1);
		L->o_w    = bind(v, "v.blk.%d.attn_out.weight", (int)i, 0);
		L->o_b    = bind(v, "v.blk.%d.attn_out.bias", (int)i, 1);
		L->ln2_w  = bind(v, "v.blk.%d.ln2.weight", (int)i, 0);
		L->ln2_b  = bind(v, "v.blk.%d.ln2.bias", (int)i, 1);
		L->up_w   = bind(v, "v.blk.%d.ffn_up.weight", (int)i, 0);
		L->up_b   = bind(v, "v.blk.%d.ffn_up.bias", (int)i, 1);
		L->down_w = bind(v, "v.blk.%d.ffn_down.weight", (int)i, 0);
		L->down_b = bind(v, "v.blk.%d.ffn_down.bias", (int)i, 1);

		/*
		 * ⚠ ONE LAYER'S WORTH OF MISSES IS THE WHOLE STORY. Twenty four
		 * layers of the same wrong guess is twenty four identical lines
		 * and a full table, which pushes the hparams off the screen.
		 */
		if (v->n_missing >= sizeof(v->missing) / sizeof(*v->missing))
			break;
	}

	if (v->n_missing) {
		snprintf(v->why, sizeof(v->why),
			 "%u of the tower's tensors are not in this file under "
			 "the names this reads", v->n_missing);
		return -1;
	}
	if (!v->n_patches) {
		snprintf(v->why, sizeof(v->why),
			 "image_size %u and patch_size %u give no patch grid",
			 v->image_size, v->patch_size);
		return -1;
	}
	return 0;
}

void charsiu_vision_close(struct charsiu_vision *v)
{
	free(v->layer);
	v->layer = NULL;
	if (v->opened)
		gguf_close(&v->g);
	v->opened = 0;
}

const char *charsiu_vision_why_not(const struct charsiu_vision *v)
{
	return v->why[0] ? v->why : NULL;
}

void charsiu_vision_describe(const struct charsiu_vision *v, FILE *out)
{
	unsigned i;

	fprintf(out, "vision tower\n");
	fprintf(out, "  image        %u x %u, patches of %u -> %u x %u = %u\n",
		v->image_size, v->image_size, v->patch_size, v->grid, v->grid,
		v->n_patches);
	fprintf(out, "  width        %u, ffn %u, heads %u, layers %u\n",
		v->n_embd, v->n_ff, v->n_head, v->n_layer);
	fprintf(out, "  projection   %s, to %u\n",
		v->proj == CHARSIU_PROJ_MLP ? "mlp" : "unrecognised",
		v->proj_dim);
	fprintf(out, "  norm eps     %g\n", (double)v->eps);
	fprintf(out, "  pixels       mean %.4f %.4f %.4f  std %.4f %.4f %.4f\n",
		(double)v->mean[0], (double)v->mean[1], (double)v->mean[2],
		(double)v->std[0], (double)v->std[1], (double)v->std[2]);
	if (!v->n_missing) {
		fprintf(out, "  complete     every tensor this needs is here\n");
		return;
	}
	fprintf(out, "  MISSING      %u, under the names this reads:\n",
		v->n_missing);
	for (i = 0; i < v->n_missing; i++)
		fprintf(out, "    %s\n", v->missing[i]);
}
