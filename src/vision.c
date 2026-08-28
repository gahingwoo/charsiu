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
#include <math.h>
#include <stdarg.h>

#include "charsiu.h"
#include "charsiu_llm.h"
#include "charsiu_vision.h"

/*
 * Bind one tensor by name, recording a miss. `opt` marks the ones a tower can
 * legitimately be without -- a pre-layernorm, a second projector matmul -- so
 * that "absent" and "missing" stay different things.
 */
static void miss(struct charsiu_vision *v, const char *fmt, ...)
{
	va_list ap;

	if (v->n_missing >= sizeof(v->missing) / sizeof(*v->missing))
		return;
	va_start(ap, fmt);
	vsnprintf(v->missing[v->n_missing++], sizeof(v->missing[0]), fmt, ap);
	va_end(ap);
}

/* the contraction axis and the row count of a 2D or 4D weight */
static void wshape(const struct gguf_tensor *t, uint64_t *in, uint64_t *out)
{
	unsigned d;

	*in = 1;
	for (d = 0; d + 1 < (t->n_dims ? t->n_dims : 1); d++)
		*in *= t->ne[d];
	*out = t->n_dims ? t->ne[t->n_dims - 1] : 1;
}

/*
 * Bind one tensor by name and CHECK ITS SHAPE.
 *
 * ⚠ A NAME THAT EXISTS WITH THE WRONG SHAPE IS THE DANGEROUS CASE. A missing
 * name is loud on its own; a present one that contracts over the wrong axis
 * produces finite, plausible numbers all the way to a sentence. `in` or `out`
 * of 0 means "do not check that side".
 *
 * `opt` marks the ones a tower can legitimately be without -- a pre-layernorm,
 * a projector bias -- so that absent and missing stay different things.
 */
static const struct gguf_tensor *bind(struct charsiu_vision *v,
				      const char *fmt, int idx, int opt,
				      uint64_t in, uint64_t out)
{
	const struct gguf_tensor *t;
	uint64_t gi, go;
	char name[80];

	if (idx >= 0)
		snprintf(name, sizeof(name), fmt, idx);
	else
		snprintf(name, sizeof(name), "%s", fmt);
	t = gguf_tensor(&v->g, name);
	if (!t) {
		if (!opt)
			miss(v, "%s", name);
		return NULL;
	}
	wshape(t, &gi, &go);
	if ((in && gi != in) || (out && go != out)) {
		miss(v, "%s is %llux%llu, wanted %llux%llu", name,
		     (unsigned long long)gi, (unsigned long long)go,
		     (unsigned long long)in, (unsigned long long)out);
		return NULL;
	}
	return t;
}

/*
 * The same lookup with the reporting off: "is there a tensor of this name with
 * this shape". Used where two names could carry the same matrix and the shape
 * is what decides, so that asking does not itself count as a miss.
 */
static const struct gguf_tensor *probe(struct charsiu_vision *v,
				       const char *fmt, int idx,
				       uint64_t in, uint64_t out)
{
	const struct gguf_tensor *t;
	uint64_t gi, go;
	char name[80];

	snprintf(name, sizeof(name), fmt, idx);
	t = gguf_tensor(&v->g, name);
	if (!t)
		return NULL;
	wshape(t, &gi, &go);
	if ((in && gi != in) || (out && go != out))
		return NULL;
	return t;
}

/* a 1D tensor: a bias, or a norm's gain */
static const struct gguf_tensor *bind1(struct charsiu_vision *v,
				       const char *fmt, int idx, int opt,
				       uint64_t n)
{
	return bind(v, fmt, idx, opt, 0, n);
}

/* ⚠ gguf_get_u32 RETURNS 0 ON SUCCESS. Reading it as a truth value inverts
 * every check in this file, which is how the first run of the synthetic test
 * reported "no vision encoder" about a file that had one. */
static int need(struct charsiu_vision *v, const char *key, uint32_t *out)
{
	if (gguf_get_u32(&v->g, key, out) == 0)
		return 0;
	miss(v, "%s (a key, not a tensor)", key);
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

	{
		uint32_t W = v->n_embd, pin = 3u * v->patch_size * v->patch_size;

		v->patch_w   = bind(v, "v.patch_embd.weight", -1, 0, pin, W);
		v->patch_b   = bind1(v, "v.patch_embd.bias", -1, 1, W);
		v->class_embd = probe(v, "v.class_embd", -1, 0, W)
			      ? bind1(v, "v.class_embd", -1, 1, W) : NULL;
		v->pos_embd  = bind(v, "v.position_embd.weight", -1, 0, W,
				    v->n_patches + (v->class_embd ? 1 : 0));
		v->pre_ln_w  = bind1(v, "v.pre_ln.weight", -1, 1, W);
		v->pre_ln_b  = bind1(v, "v.pre_ln.bias", -1, 1, W);
		v->post_ln_w = bind1(v, "v.post_ln.weight", -1, 1, W);
		v->post_ln_b = bind1(v, "v.post_ln.bias", -1, 1, W);
	}

	/*
	 * ⚠ THE PROJECTOR IS THE ONE PART THAT IS NOT A ViT, and it differs per
	 * model family. clip.projector_type says which; a file without the key
	 * gets the mlp shape, which is what llava writes.
	 */
	{
		char kind[32] = "";

		gguf_get_str(&v->g, "clip.projector_type", kind, sizeof(kind));
		if (gguf_tensor(&v->g, "visual_projection.weight")) {
			v->proj = CHARSIU_PROJ_CLIP;
			v->scale = 1;
			v->vproj_w = bind(v, "visual_projection.weight", -1, 0,
					  v->n_embd, v->proj_dim);
			if (!v->class_embd)
				miss(v, "v.class_embd (a CLIP tower pools the "
				     "class token and there is none)");
		} else if (!strcmp(kind, "idefics3")) {
			uint32_t s2;

			v->proj = CHARSIU_PROJ_IDEFICS3;
			if (gguf_get_u32(&v->g, "clip.vision.projector.scale_factor",
					 &v->scale) != 0 || !v->scale)
				v->scale = 1;
			s2 = v->scale * v->scale;
			/*
			 * ⚠ THE PIXEL SHUFFLE IS WHY THE fc IS SO WIDE. It
			 * folds scale_factor squared patches into one, so the
			 * fc contracts over n_embd * scale^2 -- 768 * 16 =
			 * 12288 on SmolVLM-256M -- and an image becomes
			 * n_patches / scale^2 tokens.
			 */
			v->fc_w = bind(v, "mm.model.fc.weight", -1, 0,
				       (uint64_t)v->n_embd * s2, v->proj_dim);
			v->fc_b = bind1(v, "mm.model.fc.bias", -1, 1,
					v->proj_dim);
			if (v->n_patches % s2)
				miss(v, "%u patches do not divide by scale %u squared",
				     v->n_patches, v->scale);
		} else {
			v->proj = CHARSIU_PROJ_MLP;
			v->scale = 1;
			v->mm_w[0] = bind(v, "mm.0.weight", -1, 0, v->n_embd, 0);
			v->mm_b[0] = bind(v, "mm.0.bias", -1, 1, 0, 0);
			v->mm_w[1] = bind(v, "mm.2.weight", -1, 1, 0, 0);
			v->mm_b[1] = bind(v, "mm.2.bias", -1, 1, 0, 0);
		}
	}

	/* clip.use_gelu is a key rather than a guess; a ViT that says so is 1. */
	{
		uint32_t g = 1;

		gguf_get_u32(&v->g, "clip.use_gelu", &g);
		v->use_gelu = g ? 1 : 0;
	}

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

		uint32_t W = v->n_embd, F = v->n_ff;

		L->ln1_w  = bind1(v, "v.blk.%d.ln1.weight", (int)i, 0, W);
		L->ln1_b  = bind1(v, "v.blk.%d.ln1.bias", (int)i, 1, W);
		L->q_w    = bind(v, "v.blk.%d.attn_q.weight", (int)i, 0, W, W);
		L->q_b    = bind1(v, "v.blk.%d.attn_q.bias", (int)i, 1, W);
		L->k_w    = bind(v, "v.blk.%d.attn_k.weight", (int)i, 0, W, W);
		L->k_b    = bind1(v, "v.blk.%d.attn_k.bias", (int)i, 1, W);
		L->v_w    = bind(v, "v.blk.%d.attn_v.weight", (int)i, 0, W, W);
		L->v_b    = bind1(v, "v.blk.%d.attn_v.bias", (int)i, 1, W);
		L->o_w    = bind(v, "v.blk.%d.attn_out.weight", (int)i, 0, W, W);
		L->o_b    = bind1(v, "v.blk.%d.attn_out.bias", (int)i, 1, W);
		L->ln2_w  = bind1(v, "v.blk.%d.ln2.weight", (int)i, 0, W);
		L->ln2_b  = bind1(v, "v.blk.%d.ln2.bias", (int)i, 1, W);

		/*
		 * ⚠ BY SHAPE, NOT BY NAME. In this file ffn_down is the FIRST
		 * matmul, n_embd -> n_ff, and ffn_up is the second -- the
		 * opposite of the language model's use of the same two words.
		 * Binding by name would have contracted fc1 over 3072 where
		 * the activation is 768 wide. So ask for the shape and let
		 * whichever name carries it answer.
		 */
		if (probe(v, "v.blk.%d.ffn_down.weight", (int)i, W, F)) {
			L->fc1_w = bind(v, "v.blk.%d.ffn_down.weight", (int)i, 0, W, F);
			L->fc1_b = bind1(v, "v.blk.%d.ffn_down.bias", (int)i, 1, F);
			L->fc2_w = bind(v, "v.blk.%d.ffn_up.weight", (int)i, 0, F, W);
			L->fc2_b = bind1(v, "v.blk.%d.ffn_up.bias", (int)i, 1, W);
		} else {
			L->fc1_w = bind(v, "v.blk.%d.ffn_up.weight", (int)i, 0, W, F);
			L->fc1_b = bind1(v, "v.blk.%d.ffn_up.bias", (int)i, 1, F);
			L->fc2_w = bind(v, "v.blk.%d.ffn_down.weight", (int)i, 0, F, W);
			L->fc2_b = bind1(v, "v.blk.%d.ffn_down.bias", (int)i, 1, W);
		}

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

	/*
	 * ⚠ int8, AND ONLY IF CHARSIU_NPU ASKED. The tower is eight weights a
	 * layer plus the projector; the widest contraction is the projector's,
	 * which on a pixel shuffled one is n_embd * scale^2.
	 */
	/* ⚠ nothing else starts it outside the language model */
	charsiu_threads_start(0);

	if (getenv("CHARSIU_NPU")) {
		unsigned wide = v->n_embd > v->n_ff ? v->n_embd : v->n_ff;
		unsigned rows = v->n_ff > v->proj_dim ? v->n_ff : v->proj_dim;

		if (v->proj == CHARSIU_PROJ_IDEFICS3 && v->scale)
			wide = wide > v->n_embd * v->scale * v->scale
			     ? wide : v->n_embd * v->scale * v->scale;
		if (rows < v->n_embd)
			rows = v->n_embd;
		if (!charsiu_pool_init(&v->pool, v->n_layer * 8 + 4, wide,
				       rows, 0))
			v->npu = v->pool.dev != NULL;
		if (!v->npu && charsiu_diag())
			fprintf(stderr, "charsiu: the vision tower stays on "
				"the CPU\n");
		if (v->npu) {
			/*
			 * ⚠ ALL OF IT, NOW, AND INTO A CACHE. A board round
			 * measured 75 s of this tower's 82 s inside the
			 * quantiser -- the matmuls underneath were about 8 s
			 * against the CPU's 148. Staging lazily also
			 * interleaves with the language model's, and the cache
			 * is one ordered file.
			 */
			const struct gguf_tensor *list[26 * 8 + 4];
			unsigned nl = 0, li;
			char cbuf[512];
			const char *cache = charsiu_cache_path(path, cbuf,
							       sizeof(cbuf));
			char stamp[64];

			snprintf(stamp, sizeof(stamp), "%u:%u:%u:%u",
				 v->n_embd, v->n_ff, v->n_layer, v->proj_dim);
			for (li = 0; li < v->n_layer &&
			     nl + 8 < sizeof(list) / sizeof(*list); li++) {
				struct charsiu_vision_layer *L = &v->layer[li];

				list[nl++] = L->q_w; list[nl++] = L->k_w;
				list[nl++] = L->v_w; list[nl++] = L->o_w;
				list[nl++] = L->fc1_w; list[nl++] = L->fc2_w;
			}
			if (v->fc_w)
				list[nl++] = v->fc_w;
			if (v->vproj_w)
				list[nl++] = v->vproj_w;
			if (v->mm_w[0])
				list[nl++] = v->mm_w[0];
			if (v->mm_w[1])
				list[nl++] = v->mm_w[1];
			charsiu_pool_stage_all(&v->pool, list, nl, cache, stamp);
		}
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
	if (v->npu && charsiu_diag())
		charsiu_pool_report(&v->pool, stderr);
	charsiu_pool_fini(&v->pool);
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
	if (v->proj == CHARSIU_PROJ_CLIP)
		fprintf(out, "  projection   CLIP, the class token pooled to "
			"one embedding of %u\n", charsiu_vision_width(v));
	else if (v->proj == CHARSIU_PROJ_IDEFICS3)
		fprintf(out, "  projection   idefics3, pixel shuffle by %u so "
			"%u patches become %u tokens of %u\n",
			v->scale, v->n_patches, charsiu_vision_tokens(v),
			charsiu_vision_width(v));
	else
		fprintf(out, "  projection   %s, %u tokens of %u\n",
			v->proj == CHARSIU_PROJ_MLP ? "mlp" : "unrecognised",
			charsiu_vision_tokens(v), charsiu_vision_width(v));
	fprintf(out, "  norm eps     %g, activation %s\n", (double)v->eps,
		v->use_gelu ? "gelu (tanh)" : "gelu quick");
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

/* ---- the forward pass ---------------------------------------------------- */

/*
 * ⚠ A 4D PATCH EMBEDDING IS A 2D MATMUL, and gguf_matvec reads ne[0] as the
 * columns and ne[1] as the rows. llama.cpp stores the patch weight as a
 * convolution kernel, [kw][kh][in_c][out], so handed to gguf_matvec unchanged
 * it would contract over kw alone. The data is contiguous and in the right
 * order already; only the shape is wrong, so flatten the shape.
 */
static struct gguf_tensor flat2d(const struct gguf_tensor *w)
{
	struct gguf_tensor t = *w;
	uint64_t in = 1;
	unsigned d;

	if (w->n_dims <= 2)
		return t;
	for (d = 0; d + 1 < w->n_dims; d++)
		in *= w->ne[d];
	t.n_dims = 2;
	t.ne[0] = in;
	t.ne[1] = w->ne[w->n_dims - 1];
	t.ne[2] = t.ne[3] = 1;
	return t;
}

static uint64_t rows_of(const struct gguf_tensor *w)
{
	struct gguf_tensor t = flat2d(w);

	return t.ne[1];
}

/*
 * ⚠ ITS OWN SOFTMAX, because llama.c's is static and this one runs over a
 * different axis: every patch against every patch, with no mask.
 */
static void vsoftmax(float *x, unsigned n)
{
	float mx = x[0], sum = 0.0f;
	unsigned i;

	for (i = 1; i < n; i++)
		if (x[i] > mx)
			mx = x[i];
	for (i = 0; i < n; i++) {
		x[i] = expf(x[i] - mx);
		sum += x[i];
	}
	for (i = 0; i < n; i++)
		x[i] /= sum;
}

/* x[n] <- (x - mean) / sqrt(var + eps) * w + b, the mean subtracting kind. */
static void layernorm(float *out, const float *x, const float *w,
		      const float *b, unsigned n, float eps)
{
	float mean = 0.0f, var = 0.0f, inv;
	unsigned i;

	for (i = 0; i < n; i++)
		mean += x[i];
	mean /= (float)n;
	for (i = 0; i < n; i++) {
		float d = x[i] - mean;

		var += d * d;
	}
	var /= (float)n;
	inv = 1.0f / sqrtf(var + eps);
	for (i = 0; i < n; i++) {
		float y = (x[i] - mean) * inv;

		out[i] = y * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
	}
}

/*
 * ⚠⚠ clip.use_gelu PICKS AN ACTIVATION, IT DOES NOT SWITCH ONE OFF.
 *
 * 1 is the tanh approximation, which is ggml's GGML_OP_GELU and llama.c's
 * gelu_mul. 0 is GELU QUICK, x * sigmoid(1.702 x), which is what OpenAI's CLIP
 * uses -- not the absence of a nonlinearity. Reading it as a boolean meant
 * every CLIP feed forward would have run as two matmuls with nothing between
 * them: finite, plausible, and a completely different model. SmolVLM has
 * use_gelu 1, which is why the first tower this ran on could not show it.
 *
 * Neither is gated: a ViT's feed forward is fc1 -> activation -> fc2, with no
 * second branch to multiply against.
 */
static void gelu(float *x, unsigned n, int tanh_form)
{
	unsigned i;

	for (i = 0; i < n; i++) {
		float v = x[i];

		if (tanh_form)
			x[i] = 0.5f * v *
			       (1.0f + tanhf(0.7978845608028654f *
					     (v + 0.044715f * v * v * v)));
		else
			x[i] = v / (1.0f + expf(-1.702f * v));
	}
}

/* one row of a 1D tensor, or nothing when the tensor is absent */
static const float *row1(const struct gguf_tensor *t, float *buf, unsigned n)
{
	if (!t)
		return NULL;
	gguf_row_f32(t, 0, buf);
	(void)n;
	return buf;
}

/* Y[m][nout] = X[m][k] * W, with the bias added. */
static struct charsiu_npu_pool *rows_pool;

/*
 * ⚠ ONLY THE 2D WEIGHTS GO TO THE HARDWARE, and the one that does not is the
 * patch embedding: it is a 4D convolution kernel, the pool keys on the tensor
 * POINTER, and a flattened view is a different address every call. It is 1.8 of
 * the tower's 90 G-mac, so this is two percent left on the CPU rather than a
 * gap worth a persistent view.
 */
static void rows_mul(const struct gguf_tensor *w, const float *bias,
		     const float *X, unsigned m, unsigned k, float *Y,
		     unsigned nout, struct charsiu_act *a)
{
	struct gguf_tensor t = flat2d(w);
	unsigned r, i;

	if (rows_pool && m > 1 && w->n_dims <= 2 &&
	    !charsiu_pool_rows(rows_pool, w, X, m, Y)) {
		if (bias)
			for (r = 0; r < m; r++)
				for (i = 0; i < nout; i++)
					Y[(size_t)r * nout + i] += bias[i];
		return;
	}

	for (r = 0; r < m; r++) {
		float *y = Y + (size_t)r * nout;

		charsiu_act_set(a, X + (size_t)r * k, (int)k);
		gguf_matvec(&t, a, y, 0, nout);
		if (bias)
			for (i = 0; i < nout; i++)
				y[i] += bias[i];
	}
}

void charsiu_vision_normalise(const struct charsiu_vision *v, float *px)
{
	unsigned c, i, hw = v->image_size * v->image_size;

	for (c = 0; c < 3; c++) {
		float m = v->mean[c], s = v->std[c] != 0.0f ? v->std[c] : 1.0f;

		for (i = 0; i < hw; i++) {
			float *p = &px[(size_t)c * hw + i];

			*p = (*p - m) / s;
		}
	}
}

unsigned charsiu_vision_tokens(const struct charsiu_vision *v)
{
	unsigned s2 = v->scale ? v->scale * v->scale : 1;

	/* ⚠ ONE, not one per patch: a retrieval tower pools to a single vector */
	if (v->proj == CHARSIU_PROJ_CLIP)
		return 1;
	return s2 > 1 ? v->n_patches / s2 : v->n_patches;
}

unsigned charsiu_vision_width(const struct charsiu_vision *v)
{
	if (v->proj == CHARSIU_PROJ_CLIP)
		return v->vproj_w ? (unsigned)rows_of(v->vproj_w) : 0;
	if (v->proj == CHARSIU_PROJ_IDEFICS3)
		return v->fc_w ? (unsigned)rows_of(v->fc_w) : 0;
	if (v->mm_w[1])
		return (unsigned)rows_of(v->mm_w[1]);
	if (v->mm_w[0])
		return (unsigned)rows_of(v->mm_w[0]);
	return v->n_embd;
}

/*
 * idefics3's pixel shuffle: fold a scale x scale block of neighbouring patches
 * into one embedding scale^2 times as wide, so an image costs the language
 * model n_patches / scale^2 tokens instead of n_patches.
 *
 * ⚠ THE INDEX MAPPING IS THE WHOLE OF IT, and it is two reshapes with a
 * transpose between them rather than a plain block gather. Written out from
 * transformers' Idefics3 pixel_shuffle:
 *
 *   out[h2][w2][e3] = x[h2*s + e3/(E*s)][w2*s + (e3 % (E*s))/E][e3 % E]
 *
 * Get it wrong and every number stays finite and the sentence stays fluent.
 */
static void pixel_shuffle(const float *x, float *out, unsigned grid,
			  unsigned E, unsigned s)
{
	unsigned g2 = grid / s, h2, w2, e3;

	for (h2 = 0; h2 < g2; h2++)
		for (w2 = 0; w2 < g2; w2++)
			for (e3 = 0; e3 < E * s * s; e3++) {
				unsigned d = e3 / (E * s), r = e3 % (E * s);
				unsigned h = h2 * s + d, w = w2 * s + r / E;

				out[((size_t)h2 * g2 + w2) * E * s * s + e3] =
					x[((size_t)h * grid + w) * E + r % E];
			}
}

/* one (head, patch) each; see the note at the call site */
struct vattn {
	const float *q, *k, *v;
	float *o;
	unsigned n, W, hd;
	float scale;
};

/*
 * ⚠ A BLOCK OF PATCHES AT A TIME. One query reads every key and every value, so
 * doing it per query streams 2 * n * head_dim floats off DRAM 1025 times a head
 * -- and the board measured this shape as bandwidth bound rather than
 * arithmetic bound, which is why vectorising it bought 2.9x on a host and 1.24x
 * there. See the long note in whisper.c.
 */
#define VATTN_QB 8

static void vattn_rows(void *ctx, uint64_t r0, uint64_t n)
{
	const struct vattn *c = ctx;
	unsigned nb = (c->n + VATTN_QB - 1) / VATTN_QB;
	float *att = malloc((size_t)VATTN_QB * c->n * sizeof(float));
	uint64_t r;

	if (!att)
		return;
	for (r = r0; r < r0 + n; r++) {
		unsigned h = (unsigned)(r / nb), b = (unsigned)(r % nb);
		unsigned i0 = b * VATTN_QB, off = h * c->hd;
		unsigned nq = c->n - i0 < VATTN_QB ? c->n - i0 : VATTN_QB;
		unsigned j, u, e;

		for (j = 0; j < c->n; j++) {
			const float *kj = c->k + (size_t)j * c->W + off;

			for (u = 0; u < nq; u++)
				att[u * c->n + j] = charsiu_dot_f32(
					c->q + (size_t)(i0 + u) * c->W + off,
					kj, c->hd) * c->scale;
		}
		for (u = 0; u < nq; u++) {
			float *o = c->o + (size_t)(i0 + u) * c->W + off;

			vsoftmax(att + u * c->n, c->n);
			for (e = 0; e < c->hd; e++)
				o[e] = 0.0f;
		}
		for (j = 0; j < c->n; j++) {
			const float *vj = c->v + (size_t)j * c->W + off;

			for (u = 0; u < nq; u++)
				charsiu_axpy_f32(c->o + (size_t)(i0 + u) *
						 c->W + off, vj,
						 att[u * c->n + j], c->hd);
		}
	}
	free(att);
}

int charsiu_vision_encode(struct charsiu_vision *v, const float *px, float *out)
{
	unsigned np = v->n_patches, W = v->n_embd, P = v->patch_size;
	unsigned pin = 3u * P * P, hd, l, p, i;
	/*
	 * ⚠ THE SEQUENCE IS NOT THE PATCHES. CLIP prepends a class token, so
	 * everything from the position embedding to the attention runs over
	 * np + 1 rows, and only the gather and the pixel shuffle are about
	 * patches. Sizing the buffers by np was a read one row past every one
	 * of them.
	 */
	unsigned cls, nt;
	unsigned nff = v->n_ff;
	unsigned wide = W > nff ? W : nff;
	float *patch = NULL, *x = NULL, *xb = NULL, *q = NULL, *k = NULL;
	float *val = NULL, *att = NULL, *ff = NULL, *tmp = NULL, *gain = NULL;
	float *bias = NULL, scale;
	struct charsiu_act a;
	int rc = -1;

	if (!v->opened || v->n_missing || !np || !v->n_head)
		return -1;
	cls = v->class_embd ? 1u : 0u;
	nt = np + cls;
	hd = W / v->n_head;
	if (!hd || hd * v->n_head != W)
		return -1;
	scale = 1.0f / sqrtf((float)hd);

	/*
	 * ⚠ THE WIDEST CONTRACTION IS THE PROJECTOR, not the feed forward. A
	 * pixel shuffled fc contracts over n_embd * scale^2 -- 12288 where
	 * n_ff is 3072 -- and an activation buffer sized for the ffn would be
	 * overrun by four times.
	 */
	if (v->proj == CHARSIU_PROJ_IDEFICS3 && v->scale)
		wide = wide > W * v->scale * v->scale
		     ? wide : W * v->scale * v->scale;
	if (charsiu_act_alloc(&a, (int)(pin > wide ? pin : wide)))
		return -1;
	/*
	 * ⚠ A FILE SCOPE POINTER, SET FOR THE LENGTH OF ONE CALL. rows_mul is
	 * the one place a weight is multiplied and it is called from eleven
	 * sites; threading a pool through all of them would be eleven more
	 * arguments for one bit of information. It is cleared on the way out so
	 * a second tower cannot inherit the first one's device.
	 */
	rows_pool = v->npu ? &v->pool : NULL;
	patch = malloc((size_t)np * pin * sizeof(float));
	x     = malloc((size_t)nt * W * sizeof(float));
	xb    = malloc((size_t)nt * W * sizeof(float));
	q     = malloc((size_t)nt * W * sizeof(float));
	k     = malloc((size_t)nt * W * sizeof(float));
	val   = malloc((size_t)nt * W * sizeof(float));
	ff    = malloc((size_t)nt * (size_t)nff * sizeof(float));
	att   = malloc((size_t)nt * sizeof(float));
	tmp   = malloc((size_t)W * sizeof(float));
	gain  = malloc((size_t)wide * sizeof(float));
	bias  = malloc((size_t)wide * sizeof(float));
	if (!patch || !x || !xb || !q || !k || !val || !ff || !att || !tmp ||
	    !gain || !bias)
		goto out;

	/*
	 * ⚠ THE PATCH GATHER IS THE CONVOLUTION. Stride equals kernel, so the
	 * patches do not overlap and each one is just its own pixels in the
	 * kernel's own order: x fastest, then y, then channel, which is the
	 * order ggml stores [kw][kh][in_c] in. Get this order wrong and every
	 * number downstream is still finite and still plausible.
	 */
	for (p = 0; p < np; p++) {
		unsigned gy = p / v->grid, gx = p % v->grid;
		float *dst = patch + (size_t)p * pin;
		unsigned c, y;

		for (c = 0; c < 3; c++)
			for (y = 0; y < P; y++)
				for (i = 0; i < P; i++)
					dst[(c * P + y) * P + i] =
						px[((size_t)c * v->image_size +
						    gy * P + y) * v->image_size +
						   gx * P + i];
	}

	rows_mul(v->patch_w, row1(v->patch_b, bias, W), patch, np, pin,
		 x + (size_t)cls * W, W, &a);
	if (cls)
		gguf_row_f32(v->class_embd, 0, x);

	/* the position embedding is one row per patch */
	for (p = 0; p < nt; p++) {
		gguf_row_f32(v->pos_embd, p, tmp);
		for (i = 0; i < W; i++)
			x[(size_t)p * W + i] += tmp[i];
	}

	if (v->pre_ln_w) {
		const float *g = row1(v->pre_ln_w, gain, W);
		const float *b = row1(v->pre_ln_b, bias, W);

		for (p = 0; p < nt; p++)
			layernorm(x + (size_t)p * W, x + (size_t)p * W, g, b,
				  W, v->eps);
	}

	for (l = 0; l < v->n_layer; l++) {
		struct charsiu_vision_layer *L = &v->layer[l];
		float lw[1];

		(void)lw;
		{
			float *g = malloc((size_t)W * sizeof(float));
			float *b = malloc((size_t)W * sizeof(float));

			if (!g || !b) { free(g); free(b); goto out; }
			row1(L->ln1_w, g, W);
			for (p = 0; p < nt; p++)
				layernorm(xb + (size_t)p * W, x + (size_t)p * W,
					  L->ln1_w ? g : NULL,
					  L->ln1_b ? row1(L->ln1_b, b, W) : NULL,
					  W, v->eps);
			free(g); free(b);
		}

		rows_mul(L->q_w, row1(L->q_b, bias, W), xb, nt, W, q, W, &a);
		rows_mul(L->k_w, row1(L->k_b, bias, W), xb, nt, W, k, W, &a);
		rows_mul(L->v_w, row1(L->v_b, bias, W), xb, nt, W, val, W, &a);

		/*
		 * ⚠ FULL ATTENTION, EVERY PATCH AGAINST EVERY PATCH, and on a
		 * thread each. whisper's identical loop measured 62% of a
		 * transcription and came down 3.46x on four cores; this one is
		 * 1025 against 1025, twelve heads, twelve layers.
		 *
		 * It is not a matmul against a weight, so nothing built for the
		 * NPU pool touches it.
		 */
		{
			struct vattn c;

			c.q = q; c.k = k; c.v = val; c.o = xb;
			c.n = nt; c.W = W; c.hd = hd; c.scale = scale;
			charsiu_parallel_for(vattn_rows, &c,
					     (uint64_t)v->n_head *
					     ((nt + VATTN_QB - 1) / VATTN_QB));
		}

		rows_mul(L->o_w, row1(L->o_b, bias, W), xb, nt, W, q, W, &a);
		for (i = 0; i < nt * W; i++)
			x[i] += q[i];

		{
			float *g = malloc((size_t)W * sizeof(float));
			float *b = malloc((size_t)W * sizeof(float));

			if (!g || !b) { free(g); free(b); goto out; }
			row1(L->ln2_w, g, W);
			for (p = 0; p < nt; p++)
				layernorm(xb + (size_t)p * W, x + (size_t)p * W,
					  L->ln2_w ? g : NULL,
					  L->ln2_b ? row1(L->ln2_b, b, W) : NULL,
					  W, v->eps);
			free(g); free(b);
		}

		rows_mul(L->fc1_w, row1(L->fc1_b, bias, nff), xb, nt, W, ff,
			 nff, &a);
		gelu(ff, nt * nff, v->use_gelu);
		rows_mul(L->fc2_w, row1(L->fc2_b, bias, W), ff, nt, nff, q, W,
			 &a);
		for (i = 0; i < nt * W; i++)
			x[i] += q[i];
	}

	if (v->post_ln_w) {
		float *g = malloc((size_t)W * sizeof(float));
		float *b = malloc((size_t)W * sizeof(float));

		if (!g || !b) { free(g); free(b); goto out; }
		/*
		 * ⚠ CLIP POST NORMALISES THE POOLED TOKEN ONLY. HF takes
		 * last_hidden_state[:, 0] and then applies post_layernorm; a
		 * tower feeding a language model applies it to every patch.
		 * Same tensor, different number of rows.
		 */
		unsigned pn = v->proj == CHARSIU_PROJ_CLIP ? 1 : nt;

		row1(v->post_ln_w, g, W);
		for (p = 0; p < pn; p++)
			layernorm(x + (size_t)p * W, x + (size_t)p * W, g,
				  v->post_ln_b ? row1(v->post_ln_b, b, W) : NULL,
				  W, v->eps);
		free(g); free(b);
	}

	if (v->proj == CHARSIU_PROJ_CLIP) {
		rows_mul(v->vproj_w, NULL, x, 1, W, out,
			 (unsigned)rows_of(v->vproj_w), &a);
	} else if (v->proj == CHARSIU_PROJ_IDEFICS3) {
		unsigned s2 = v->scale * v->scale;
		unsigned tok = np / s2, wide2 = W * s2;
		float *sh = malloc((size_t)tok * wide2 * sizeof(float));

		if (!sh)
			goto out;
		pixel_shuffle(x, sh, v->grid, W, v->scale);
		rows_mul(v->fc_w, row1(v->fc_b, bias, v->proj_dim), sh, tok,
			 wide2, out, (unsigned)rows_of(v->fc_w), &a);
		free(sh);
	} else if (v->mm_w[0]) {
		unsigned d0 = (unsigned)rows_of(v->mm_w[0]);
		float *h0 = malloc((size_t)np * d0 * sizeof(float));

		if (!h0)
			goto out;
		rows_mul(v->mm_w[0], row1(v->mm_b[0], bias, d0), x, np, W, h0,
			 d0, &a);
		if (v->mm_w[1]) {
			unsigned d1 = (unsigned)rows_of(v->mm_w[1]);

			gelu(h0, np * d0, v->use_gelu);
			rows_mul(v->mm_w[1], row1(v->mm_b[1], bias, d1), h0, np,
				 d0, out, d1, &a);
		} else {
			memcpy(out, h0, (size_t)np * d0 * sizeof(float));
		}
		free(h0);
	} else {
		memcpy(out, x, (size_t)nt * W * sizeof(float));
	}
	rc = 0;
out:
	rows_pool = NULL;
	free(patch); free(x); free(xb); free(q); free(k); free(val);
	free(ff); free(att); free(tmp); free(gain); free(bias);
	charsiu_act_free(&a);
	return rc;
}
