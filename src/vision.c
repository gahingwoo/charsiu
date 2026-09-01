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
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
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

/* ---- where the time goes -------------------------------------------------- */

/*
 * ⚠ THE SAME INSTRUMENT THAT SETTLED WHISPER. Its stage table said the matmuls
 * were 26% and the attention 63%, after four commits aimed at the matmuls. This
 * tower has had no such table and the board says it is 15.5 s against a vendor's
 * 768 ms for the same model, of which 3.1 s is the hardware.
 */
/*
 * ⚠⚠ AND "feed forward" WAS TWO DIFFERENT MACHINES UNDER ONE ROW.
 *
 * The board's table read 2398 ms against a 5.7 s encode -- the largest single
 * stage -- and that row was a pair of NPU matmuls AND a scalar libm loop added
 * together. The two do not respond to the same thing and cannot be reasoned
 * about as one number: the matmuls are 384 hardware dispatches whose cost is
 * fixed by the 64 row chunk, and the activation is 3.1 million tanhf calls a
 * layer on one core. Reading them as one row is how a memory-traffic story got
 * told about a stage that was arithmetic.
 *
 * ⚠ AND THE RESIDUALS AND THE PATCH GATHER WERE IN NO ROW AT ALL. The row
 * called "patch gather + embed" timed the embed only -- the gather ran above
 * the line that turns the clock on -- and the two residual adds a layer, 18.9
 * million elements over the tower, were never counted anywhere. A table whose
 * rows do not add up to the encode is an instrument that hides its own
 * remainder, which is exactly the failure it exists to prevent.
 */
enum { V_PATCH, V_QKV, V_ATTN, V_PROJ, V_FFN, V_ACT, V_NORM, V_RESID,
       V_SHUF, V_N };
static const char *const vstage_name[V_N] = {
	"patch gather + embed", "q k v", "attention", "out proj",
	"feed forward matmuls", "ffn bias + activation", "layernorms",
	"residual + position", "pixel shuffle + projector",
};
static double vstage_ms[V_N];
static int vstage_on = -1;

static double vnow(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/*
 * ⚠ VARIADIC BECAUSE A BLOCK IS NOT ONE MACRO ARGUMENT. Braces do not protect a
 * comma from the preprocessor the way parentheses do, so `unsigned gy = a, gx =
 * b;` inside a timed loop is two arguments and the build stops. Taking the rest
 * as __VA_ARGS__ and pasting it back is what lets the patch gather -- which was
 * never in the table at all -- be timed where it stands instead of being moved
 * into a function for the instrument's convenience.
 */
#define VSTAGE(i, ...) do { \
	double _t = vstage_on ? vnow() : 0.0; \
	__VA_ARGS__; \
	if (vstage_on) vstage_ms[i] += vnow() - _t; \
} while (0)

void charsiu_vision_stages(FILE *out)
{
	double tot = 0.0;
	int i;

	for (i = 0; i < V_N; i++)
		tot += vstage_ms[i];
	if (tot <= 0.0)
		return;
	fprintf(out, "charsiu vision: %.1f s accounted for\n", tot / 1e3);
	for (i = 0; i < V_N; i++)
		fprintf(out, "  %-26s %8.0f ms  %5.1f%%\n", vstage_name[i],
			vstage_ms[i], 100.0 * vstage_ms[i] / tot);
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
	float mx = x[0], sum, inv;
	unsigned i;

	for (i = 1; i < n; i++)
		if (x[i] > mx)
			mx = x[i];
	/*
	 * ⚠ THE EXPONENTIAL IS THE ARITHMETIC NOBODY COUNTED, and it is a
	 * shared kernel now: a picture asks for n^2 per head per layer, 151
	 * million of them at this tower's shape. See charsiu_expsum_f32.
	 */
	sum = charsiu_expsum_f32(x, n, mx);
	inv = sum > 0.0f ? 1.0f / sum : 0.0f;
	for (i = 0; i < n; i++)
		x[i] *= inv;
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
/*
 * ⚠⚠ AND BOTH OF THEM ARE ONE EXPONENTIAL, WHICH IS THE WHOLE OF THIS ROUND.
 *
 * The tanh form was written as written, with a tanhf per element, and that is
 * 0.5 x (1 + tanh y). Since tanh y = 1 - 2/(e^2y + 1),
 *
 *     0.5 (1 + tanh y) = 1 / (1 + e^-2y)
 *
 * so gelu(x) = x / (1 + e^(-2 k (x + 0.044715 x^3))) -- the SAME function, one
 * exponential instead of a hyperbolic tangent, and the same shape as gelu
 * quick, which was already written that way. Not an approximation of the tanh
 * form: an algebraic identity, and the measured disagreement over 3.1 million
 * elements is 4.77e-07 absolute at x = 2.43, where gelu is 2.41 -- one float
 * ulp, which is the summation order and nothing else. A dense sweep of
 * [-20, 20] finds the same worst absolute and no larger.
 *
 * ⚠ GELU QUICK'S FORMULA DID NOT CHANGE AT ALL -- only its expf became
 * charsiu_vexpq -- so the whole of the risk on that branch is the polynomial,
 * and a dense sweep of [-25, 25] puts it at 2.24e-07 relative, one ulp again.
 * It is checked separately because no model on this desk uses it: SmolVLM has
 * clip.use_gelu 1, and a branch nothing runs is a branch nothing catches.
 *
 * ⚠ THE PRICE THIS WAS PAYING. A picture is 1024 patches, the feed forward is
 * 3072 wide and there are twelve layers: 37.7 MILLION activations. On this
 * development host, one layer's 3145728 elements:
 *
 *     bias pass, then tanhf     34.65 ms      <- what shipped
 *     bias pass, then expf       2.66 ms
 *     bias FOLDED IN, NEON       1.97 ms
 *     the same across 6 threads  0.76 ms
 *
 * 45x on the pass, and there is nothing left here: 0.76 ms is what a bias pass
 * ALONE costs on this host (0.77 ms), so what remains is moving 25.2 MB and no
 * arithmetic worth naming.
 *
 * ⚠ AND IN SITU, in the tower's own table at SmolVLM-256M's shape, which is
 * the reading that counts because it is the one a board can reproduce. One
 * binary, one environment variable each, best of five interleaved:
 *
 *     CHARSIU_EXACT_GELU=1 CHARSIU_THREADS=1     373 ms   the old arithmetic
 *     CHARSIU_EXACT_GELU=1                       103 ms
 *     CHARSIU_THREADS=1                           26 ms   the identity alone
 *     (default)                                   11 ms
 *
 * So 14x of it is the identity, which does not depend on how many cores there
 * are and is the part certain to transfer, and 2.4x more is the pool on this
 * host's six. What shipped was the 373 with a separate bias pass on top.
 *
 * ⚠ AND DO NOT READ THE WHOLE ENCODE OFF THIS HOST. The ffn matmul row beside
 * these four numbers came back 4930, 4937, 5016, 5157, 5208, 5328, 5431, 5453,
 * 5923, 6560 and 6637 ms for the SAME work -- this machine is shared and the
 * spread is three times the thing being measured. Only the row moves; the wall
 * clock cannot see it.
 *
 * ⚠ THE ANSWER MOVED BY ONE ULP AND THE CONTROL PROVES IT IS ONLY THIS. Over
 * the 36864 embeddings a 512x512 image leaves as, against the binary from
 * before this change: worst 3.33e-06 absolute on an output whose rms is 1.53,
 * and 5.06e-07 of that rms rms-for-rms. With CHARSIU_EXACT_GELU set the two
 * binaries agree on all 36864 values BIT FOR BIT -- which is what says the
 * folded bias, the threading and the stage split changed nothing, and the
 * exponential is the only thing here that touches the numbers.
 *
 * ⚠ THE BOARD'S SHARE OF THAT IS AN ESTIMATE, not a reading, and this commit
 * adds the row that turns it into one. Its pool reported 2680 ms of hardware
 * across all 73 matmuls, and the two stages that are matmul-and-bias only --
 * q k v at 1057 ms and out proj at 380 -- price the (1024, 768, 768) shape at
 * about 29 ms, so the feed forward's own 24 matmuls come to roughly 1270 ms
 * and the rest of its 2398 ms row, about 1130 ms, is this function plus the
 * bias pass. That is 27 ns an element, against the 23 ns round 367 measured
 * for glibc's expf on the same board -- consistent, and tanhf is the slower
 * of the two everywhere it has been asked.
 *
 * ⚠ THE OVERFLOW BEHAVIOUR IS NOT THE SAME CODE BUT IS THE SAME ANSWER, and it
 * was checked rather than assumed. x^3 overflows f32 above 4.6e12; the cube
 * then carries an infinity into the exponent, charsiu_vexpq clamps its input
 * to [-88, 88] so the result stays finite, and x / (1 + e^-88) is x -- which
 * is what gelu does out there. The scalar tail gets expf(-inf) = 0 and the
 * same x. Neither path can make a NaN out of a finite input.
 *
 * CHARSIU_EXACT_GELU puts the tanhf form back, as the control that the change
 * has to be diffed against. It is the same switch CHARSIU_EXACT_SILU and
 * CHARSIU_EXACT_SOFTMAX are, for the same reason.
 */
static int gelu_exact(void)
{
	static int v = -1;

	if (v < 0)
		v = getenv("CHARSIU_EXACT_GELU") != NULL;
	return v;
}

/*
 * y[0..n) <- act(y + b), for ONE row, with the bias folded into the same pass.
 *
 * ⚠ FOLDED, NOT SEQUENCED, AND THAT IS A SECOND SAVING ON TOP OF THE FIRST.
 * rows_mul used to add the bias in a pass of its own and this function then
 * read the whole intermediate back. At this tower's shape the intermediate is
 * 1024 x 3072 x 4 = 12.6 MB, which is not in any cache on this board, so that
 * was 12.6 MB read and 12.6 MB written per layer for an add -- 302 MB over the
 * tower, about 42 ms of the board's 7.13 GB/s, thrown away to visit the same
 * numbers twice. It is worth 0.69 ms of the 2.66 above.
 *
 * ⚠ b MAY BE NULL and the test is inside the vector loop on purpose: it is one
 * perfectly predicted branch against a divide and a six term polynomial. The
 * hand-hoisted form was written and timed interleaved against this one -- 2.13
 * ms to this one's 2.09 over 3145728 elements -- so removing the branch is not
 * worth two copies of the kernel.
 */
static void act_span(float *y, const float *b, unsigned n, int tanh_form)
{
	const float k2 = 2.0f * 0.7978845608028654f;   /* 2 sqrt(2/pi) */
	unsigned i = 0;

	if (gelu_exact()) {
		for (; i < n; i++) {
			float v = y[i] + (b ? b[i] : 0.0f);

			if (tanh_form)
				y[i] = 0.5f * v *
				       (1.0f + tanhf(0.7978845608028654f *
						     (v + 0.044715f * v * v * v)));
			else
				y[i] = v / (1.0f + expf(-1.702f * v));
		}
		return;
	}

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
	{
		const float32x4_t one = vdupq_n_f32(1.0f);
		const float32x4_t c = vdupq_n_f32(0.044715f);
		const float32x4_t kk = vdupq_n_f32(tanh_form ? -k2 : -1.702f);

		for (; i + 4 <= n; i += 4) {
			float32x4_t v = vld1q_f32(y + i), e;

			if (b)
				v = vaddq_f32(v, vld1q_f32(b + i));
			if (tanh_form) {
				float32x4_t v3 = vmulq_f32(vmulq_f32(v, v), v);

				e = vmulq_f32(kk, vmlaq_f32(v, c, v3));
			} else {
				e = vmulq_f32(kk, v);
			}
			vst1q_f32(y + i,
				  vdivq_f32(v, vaddq_f32(one, charsiu_vexpq(e))));
		}
	}
#endif
	for (; i < n; i++) {
		float v = y[i] + (b ? b[i] : 0.0f);

		if (tanh_form)
			v = v / (1.0f + expf(-k2 * (v + 0.044715f * v * v * v)));
		else
			v = v / (1.0f + expf(-1.702f * v));
		y[i] = v;
	}
}

/*
 * The same over a block of m rows sharing one bias, across the thread pool.
 *
 * ⚠ A ROW EACH, BECAUSE THE THING BEING DIVIDED IS BYTES. Every row is
 * independent and every row is n floats read and n written; there is no shared
 * state, no reduction and no barrier inside the pass. That makes this exactly
 * the shape the attention round found transfers -- the win there was moving
 * 11.81 GB to 1.12 GB across L1, and it was 3.71x on this host and 10.32x on
 * the board, because the board is the one that is short of bandwidth and has
 * cores sitting idle to pull more of it.
 *
 * ⚠ AND A SIZE FLOOR, because npudev has the counter-example written down: a
 * gather put on this same pool lost twice, 190 ms both times, at about 0.84 ms
 * of barrier per dispatch with not enough work behind it. Here it is twelve
 * dispatches for a whole picture with 3.1 million elements behind each, so the
 * barrier is under a percent -- but a small tower or a narrow projector must
 * not pay it, hence the floor.
 */
struct vact {
	float *y;
	const float *b;
	unsigned n;
	int tanh_form;
};

static void vact_rows(void *ctx, uint64_t r0, uint64_t nr)
{
	const struct vact *c = ctx;
	uint64_t r;

	for (r = r0; r < r0 + nr; r++)
		act_span(c->y + r * c->n, c->b, c->n, c->tanh_form);
}

static void bias_act(float *y, unsigned m, unsigned n, const float *b,
		     int tanh_form)
{
	struct vact c;
	unsigned r;

	if ((uint64_t)m * n < 65536u) {
		for (r = 0; r < m; r++)
			act_span(y + (size_t)r * n, b, n, tanh_form);
		return;
	}
	c.y = y;
	c.b = b;
	c.n = n;
	c.tanh_form = tanh_form;
	charsiu_parallel_for(vact_rows, &c, m);
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

/*
 * q, k and v of one input, as one pooled call that packs each chunk once.
 * The bias for each is read into its own row, because row1() fills the one
 * scratch it is handed and three projections need three.
 */
static void rows_mul3(const struct charsiu_vision_layer *L, const float *X, unsigned m,
		      unsigned W, float *q, float *k, float *v, float *bias,
		      struct charsiu_act *a)
{
	float *bk = malloc((size_t)W * sizeof(float));
	float *bv = malloc((size_t)W * sizeof(float));
	const float *bq_r = row1(L->q_b, bias, W);
	const float *bk_r = bk ? row1(L->k_b, bk, W) : NULL;
	const float *bv_r = bv ? row1(L->v_b, bv, W) : NULL;
	unsigned r, i;

	if (bk && bv && rows_pool && m > 1 &&
	    !charsiu_pool_rows3(rows_pool, L->q_w, L->k_w, L->v_w, X, m,
				q, k, v)) {
		for (r = 0; r < m; r++)
			for (i = 0; i < W; i++) {
				if (bq_r) q[(size_t)r * W + i] += bq_r[i];
				if (bk_r) k[(size_t)r * W + i] += bk_r[i];
				if (bv_r) v[(size_t)r * W + i] += bv_r[i];
			}
		free(bk); free(bv);
		return;
	}
	/* the old three calls, which fall back a row at a time themselves */
	rows_mul(L->q_w, row1(L->q_b, bias, W), X, m, W, q, W, a);
	rows_mul(L->k_w, row1(L->k_b, bias, W), X, m, W, k, W, a);
	rows_mul(L->v_w, row1(L->v_b, bias, W), X, m, W, v, W, a);
	free(bk); free(bv);
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

/* one query block each; see the schedules below */
struct vattn {
	const float *q, *k, *v;
	float *o;
	unsigned n, W, hd, nh, nb, qb, kt, fused;
	float scale;
};

/*
 * ⚠ A BLOCK OF PATCHES AT A TIME. One query reads every key and every value, so
 * doing it per query streams 2 * n * head_dim floats off DRAM 1025 times a head
 * -- and the board measured this shape as bandwidth bound rather than
 * arithmetic bound, which is why vectorising it bought 2.9x on a host and 1.24x
 * there. See the long note in whisper.c.
 *
 * QB queries share one pass over the keys and one over the values, so the K and
 * V traffic divides by QB and the arithmetic is unchanged. What stops it going
 * up for ever is the scores: they are QB * n floats and they are written once,
 * read three times by the softmax and read again by the value pass, so past the
 * point where they stop fitting in cache the block starts paying for itself.
 *
 * ⚠ THE NUMBER IS MEASURED, NOT CHOSEN, AND IT IS THE MACHINE'S ANSWER. It is a
 * runtime value so vattn_bench -Q can sweep it in ONE process against the same
 * interference, and so a board can re-ask without a rebuild; nq was already a
 * runtime bound, so nothing in the inner loops changed shape.
 */
#ifndef VATTN_QB
#define VATTN_QB 64
#endif
#ifndef VATTN_KT
#define VATTN_KT 16
#endif

/*
 * ⚠⚠ AND THE SCORES ARE WHAT CAPS THE BLOCK, NOT THE KEYS.
 *
 * QB queries share one pass over K and one over V, so raising QB divides the
 * K/V traffic -- but the scores it has to hold are QB * n floats, written once,
 * read three times by the softmax and read a fourth time by the value pass. At
 * n = 1024 and head_dim = 64 the two cross at QB = 32:
 *
 *   QB    K+V per query   scores resident   scores traffic
 *    8       64.0 KB           32 KB            160 KB
 *   16       32.0 KB           64 KB            320 KB
 *   32       16.0 KB          128 KB            640 KB   <- past K+V's 512 KB
 *   64        8.0 KB          256 KB           1280 KB
 *
 * which is exactly where the measured sweep turns round. So the cap is not the
 * keys and it is not the arithmetic: it is a scratch array that only exists
 * because the softmax was written as three passes over a whole row.
 *
 * VATTN_FUSED takes the cap away. Tile the KEYS as well, carry a running max
 * and a running sum, and rescale the accumulator when the max moves -- the
 * scores then live QB * KT at a time, a few kilobytes, and never leave L1. The
 * keys and values are still read exactly once per query block, so raising QB
 * keeps dividing their traffic with nothing growing to pay for it.
 *
 * ⚠ IT IS NOT BIT IDENTICAL AND THAT IS THE WHOLE OF THE RISK. Everything else
 * in this stage reorders only the ISSUE of the arithmetic; this reorders the
 * arithmetic. exp(x - m) for a running m, rescaled, is the same number in exact
 * arithmetic and a few ulp away in f32. This tree has shipped one fast wrong
 * answer, so vattn_bench -c prints the worst element wise disagreement against
 * the exact kernel rather than a checksum, and vision_cross still has to pass.
 */
static void vattn_block_fused(const struct vattn *c, float *sc, unsigned off,
			      unsigned b)
{
	unsigned i0 = b * c->qb, hd = c->hd, kt = c->kt;
	unsigned nq = c->n - i0 < c->qb ? c->n - i0 : c->qb;
	float *s = sc;                       /* qb * kt, the tile of scores */
	float *acc = s + (size_t)c->qb * kt; /* qb * hd, the running output */
	float *mx = acc + (size_t)c->qb * hd, *sum = mx + c->qb;
	unsigned j0, u, e;

	for (u = 0; u < nq; u++) {
		mx[u] = -INFINITY;
		sum[u] = 0.0f;
		for (e = 0; e < hd; e++)
			acc[u * hd + e] = 0.0f;
	}
	for (j0 = 0; j0 < c->n; j0 += kt) {
		unsigned nk = c->n - j0 < kt ? c->n - j0 : kt, j;

		charsiu_qk_f32(s, kt, c->q + (size_t)i0 * c->W + off, c->W,
			       c->k + (size_t)j0 * c->W + off, c->W,
			       nq, nk, hd, c->scale);
		for (u = 0; u < nq; u++) {
			float *su = s + u * kt, m = mx[u], t;

			for (j = 0; j < nk; j++)
				if (su[j] > m)
					m = su[j];
			/*
			 * ⚠ ONLY WHEN THE MAX ACTUALLY MOVED. After the first
			 * few tiles it usually has not, and the rescale is a
			 * pass over the accumulator -- the one piece of work
			 * this kernel adds that the three pass form does not.
			 */
			if (m > mx[u]) {
				float r = expf(mx[u] - m);

				for (e = 0; e < hd; e++)
					acc[u * hd + e] *= r;
				sum[u] *= r;
				mx[u] = m;
			}
			t = charsiu_expsum_f32(su, nk, m);
			sum[u] += t;
		}
		charsiu_pv_f32(acc, hd, c->v + (size_t)j0 * c->W + off, c->W,
			       s, kt, nq, nk, hd);
	}
	for (u = 0; u < nq; u++) {
		float *o = c->o + (size_t)(i0 + u) * c->W + off;
		float inv = sum[u] > 0.0f ? 1.0f / sum[u] : 0.0f;

		for (e = 0; e < hd; e++)
			o[e] = acc[u * hd + e] * inv;
	}
}

/* one head's slice of one block of queries; `att` is c->qb * n scratch */
static void vattn_block_exact(const struct vattn *c, float *att, unsigned off,
			      unsigned b)
{
	unsigned i0 = b * c->qb;
	unsigned nq = c->n - i0 < c->qb ? c->n - i0 : c->qb;
	unsigned u, e;

	charsiu_qk_f32(att, c->n, c->q + (size_t)i0 * c->W + off, c->W,
		       c->k + off, c->W, nq, c->n, c->hd, c->scale);
	for (u = 0; u < nq; u++) {
		float *o = c->o + (size_t)(i0 + u) * c->W + off;

		vsoftmax(att + u * c->n, c->n);
		for (e = 0; e < c->hd; e++)
			o[e] = 0.0f;
	}
	charsiu_pv_f32(c->o + (size_t)i0 * c->W + off, c->W, c->v + off, c->W,
		       att, c->n, nq, c->n, c->hd);
}

/* how many floats of scratch one worker needs, for whichever kernel is on */
static size_t vattn_scratch(const struct vattn *c)
{
	if (!c->fused)
		return (size_t)c->qb * c->n;
	return (size_t)c->qb * c->kt + (size_t)c->qb * c->hd + 2u * c->qb;
}

static void vattn_block(const struct vattn *c, float *sc, unsigned off,
			unsigned b)
{
	if (c->fused)
		vattn_block_fused(c, sc, off, b);
	else
		vattn_block_exact(c, sc, off, b);
}

/*
 * ⚠⚠ WHICH THREAD IS ON WHICH HEAD, AND IT IS THE WHOLE OF THIS STAGE ON THE
 * BOARD.
 *
 * A head's keys and values are n * head_dim * 2 * 4 bytes -- half a megabyte at
 * n = 1024 -- and every one of the 128 query blocks of that head reads all of
 * it. Whether that half megabyte is read once or 128 times is decided by
 * nothing but which items the pool hands to which thread.
 *
 * SCHED_FLAT, the original, is item = head * nb + block, one flat span cut into
 * contiguous chunks. Six threads land on six DIFFERENT heads and hold three
 * megabytes of live stream against a board with about one megabyte of L2 for
 * the whole A72 cluster. Nothing stays resident and every block pays DRAM.
 *
 * SCHED_HEADWISE dispatches each head separately, so every thread is on the
 * same half megabyte at the same instant. It is the strongest form and it costs
 * n_head barriers a layer instead of one -- and a barrier makes every core wait
 * for the slowest, which on a big.LITTLE board is an A53 holding up four A72s,
 * twelve times a layer instead of once.
 *
 * SCHED_SHARE is the same locality without the barriers. The item numbering
 * becomes head = item % n_head, block = item / n_head, so a thread's contiguous
 * chunk is 1/threads of the BLOCKS and ALL of the heads; the thread then walks
 * its own chunk head major. Every thread starts at head 0 with its own slice of
 * the blocks, and they advance through the heads together because each does the
 * same work per head -- no barrier, and they only drift as far apart as the
 * cores differ in speed.
 */
enum { SCHED_FLAT, SCHED_HEADWISE, SCHED_SHARE };

static void vattn_flat(void *ctx, uint64_t r0, uint64_t n)
{
	const struct vattn *c = ctx;
	float *att = malloc(vattn_scratch(c) * sizeof(float));
	uint64_t r;

	if (!att)
		return;
	for (r = r0; r < r0 + n; r++)
		vattn_block(c, att, (unsigned)(r / c->nb) * c->hd,
			    (unsigned)(r % c->nb));
	free(att);
}

/* the head is fixed by the caller; the range is blocks */
static void vattn_one_head(void *ctx, uint64_t r0, uint64_t n)
{
	const struct vattn *c = ctx;
	float *att = malloc(vattn_scratch(c) * sizeof(float));
	uint64_t r;

	if (!att)
		return;
	for (r = r0; r < r0 + n; r++)
		vattn_block(c, att, c->nh, (unsigned)r);
	free(att);
}

/*
 * ⚠ THE RANGE IS WALKED OUT OF ORDER ON PURPOSE. The items are the same items;
 * visiting the ones with the same head together is what makes all the threads
 * read one head's keys at one time. The stride is n_head because that is the
 * numbering, and the first item of head h is the first r >= r0 with r % nh == h.
 */
static void vattn_share(void *ctx, uint64_t r0, uint64_t n)
{
	const struct vattn *c = ctx;
	float *att = malloc(vattn_scratch(c) * sizeof(float));
	uint64_t r, end = r0 + n;
	unsigned h;

	if (!att)
		return;
	for (h = 0; h < c->nh; h++) {
		unsigned first = (unsigned)((h + c->nh - r0 % c->nh) % c->nh);

		for (r = r0 + first; r < end; r += c->nh)
			vattn_block(c, att, h * c->hd,
				    (unsigned)(r / c->nh));
	}
	free(att);
}

/*
 * ⚠ PUBLIC SO IT CAN BE TIMED WITHOUT THE TOWER AROUND IT. On the board this
 * is half the encode; on this development host it is under a tenth of it,
 * because the board's matmuls go to the NPU and the host's do not. Measuring it
 * through charsiu_vision_encode means reading a 6% row of a stage table and
 * calling the difference a result. tools/vattn_bench.c calls this directly.
 *
 * ⚠ THE SCHEDULE IS A KNOB BECAUSE THE ANSWER IS THE MACHINE'S. None of the
 * three touches the arithmetic or the order within it, so all three are bit
 * identical to each other and to the unblocked form -- which vattn_bench -c
 * checks with a checksum rather than assuming.
 */
static int vattn_sched = -1;
static unsigned vattn_qb;

unsigned charsiu_vision_attn_qb(void)
{
	if (!vattn_qb) {
		const char *e = getenv("CHARSIU_VATTN_QB");
		int v = e ? atoi(e) : 0;

		vattn_qb = v > 0 ? (unsigned)v : VATTN_QB;
	}
	return vattn_qb;
}

void charsiu_vision_attn_qb_set(unsigned qb)
{
	vattn_qb = qb ? qb : VATTN_QB;
}

/*
 * The key tile the fused kernel holds scores for. It wants to be the largest
 * that keeps qb * kt floats of scores plus qb * head_dim of accumulator inside
 * L1, because staying there is the entire reason the kernel exists.
 */
static unsigned vattn_kt;

unsigned charsiu_vision_attn_kt(void)
{
	if (!vattn_kt) {
		const char *e = getenv("CHARSIU_VATTN_KT");
		int v = e ? atoi(e) : 0;

		vattn_kt = v > 0 ? (unsigned)v : VATTN_KT;
	}
	return vattn_kt;
}

void charsiu_vision_attn_kt_set(unsigned kt)
{
	vattn_kt = kt ? kt : VATTN_KT;
}

/*
 * ⚠ THE ONE KNOB THAT CHANGES THE ANSWER. Every other choice in this stage
 * reorders the issue of the same arithmetic and is bit identical; the fused
 * kernel reorders the arithmetic itself, and a wrong answer that arrives faster
 * is the failure this tree has already shipped once. Default on, with the exact
 * kernel one environment variable away as the control.
 */
static int vattn_fused = -1;

int charsiu_vision_attn_fused(void)
{
	if (vattn_fused < 0) {
		const char *e = getenv("CHARSIU_VATTN_FUSED");

		vattn_fused = !(e && (!strcmp(e, "0") || !strcmp(e, "no")));
	}
	return vattn_fused;
}

void charsiu_vision_attn_fused_set(int on)
{
	vattn_fused = on ? 1 : 0;
}

int charsiu_vision_attn_sched_get(void)
{
	if (vattn_sched < 0) {
		const char *e = getenv("CHARSIU_VATTN_SCHED");

		vattn_sched = SCHED_SHARE;
		if (e && !strcmp(e, "flat"))
			vattn_sched = SCHED_FLAT;
		else if (e && !strcmp(e, "headwise"))
			vattn_sched = SCHED_HEADWISE;
	}
	return vattn_sched;
}

/*
 * ⚠ SETTABLE BECAUSE THE COMPARISON HAS TO HAPPEN IN ONE PROCESS. This host
 * runs six cores shared with an editor and two other agents, and two builds
 * timed one after the other disagreed by 1.8x with the SAME binary on both
 * sides. Interleaving the schedules inside one run puts them on the same cores,
 * the same cache state and the same interference.
 */
void charsiu_vision_attn_sched_set(int sched)
{
	vattn_sched = sched;
}

const char *charsiu_vision_attn_sched_name(int sched)
{
	static const char *const nm[] = { "flat", "headwise", "share" };

	return sched >= 0 && sched < 3 ? nm[sched] : "?";
}

void charsiu_vision_attention(const float *q, const float *k, const float *v,
			      float *o, unsigned n, unsigned W,
			      unsigned n_head, float scale)
{
	unsigned qb = charsiu_vision_attn_qb();
	uint64_t nb = (n + qb - 1) / qb;
	struct vattn c;
	unsigned h;

	c.q = q; c.k = k; c.v = v; c.o = o;
	c.n = n; c.W = W; c.hd = W / n_head; c.scale = scale;
	c.nh = n_head; c.nb = (unsigned)nb; c.qb = qb;
	c.kt = charsiu_vision_attn_kt();
	c.fused = (unsigned)charsiu_vision_attn_fused();
	if (c.kt > n)
		c.kt = n;
	switch (charsiu_vision_attn_sched_get()) {
	case SCHED_FLAT:
		charsiu_parallel_for(vattn_flat, &c, (uint64_t)n_head * nb);
		break;
	case SCHED_HEADWISE:
		for (h = 0; h < n_head; h++) {
			c.nh = h * c.hd;         /* reused as the head offset */
			charsiu_parallel_for(vattn_one_head, &c, nb);
		}
		break;
	default:
		charsiu_parallel_for(vattn_share, &c, (uint64_t)n_head * nb);
		break;
	}
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
	/*
	 * ⚠ THE CLOCK GOES ON HERE, NOT BELOW THE GATHER. It used to be turned
	 * on after this loop, so the row named "patch gather + embed" timed the
	 * embed and the gather ran outside every row in the table.
	 */
	if (vstage_on < 0)
		vstage_on = getenv("CHARSIU_STAGES") != NULL;

	VSTAGE(V_PATCH, for (p = 0; p < np; p++) {
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
	});

	VSTAGE(V_PATCH, rows_mul(v->patch_w, row1(v->patch_b, bias, W), patch,
				 np, pin, x + (size_t)cls * W, W, &a));
	if (cls)
		gguf_row_f32(v->class_embd, 0, x);

	/* the position embedding is one row per patch */
	VSTAGE(V_RESID, for (p = 0; p < nt; p++) {
		gguf_row_f32(v->pos_embd, p, tmp);
		for (i = 0; i < W; i++)
			x[(size_t)p * W + i] += tmp[i];
	});

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
			VSTAGE(V_NORM, for (p = 0; p < nt; p++)
				layernorm(xb + (size_t)p * W, x + (size_t)p * W,
					  L->ln1_w ? g : NULL,
					  L->ln1_b ? row1(L->ln1_b, b, W) : NULL,
					  W, v->eps));
			free(g); free(b);
		}

		VSTAGE(V_QKV, rows_mul3(L, xb, nt, W, q, k, val, bias, &a));

		/*
		 * ⚠ FULL ATTENTION, EVERY PATCH AGAINST EVERY PATCH, and on a
		 * thread each. whisper's identical loop measured 62% of a
		 * transcription and came down 3.46x on four cores; this one is
		 * 1025 against 1025, twelve heads, twelve layers.
		 *
		 * It is not a matmul against a weight, so nothing built for the
		 * NPU pool touches it.
		 */
		VSTAGE(V_ATTN, charsiu_vision_attention(q, k, val, xb, nt, W,
						       v->n_head, scale));

		VSTAGE(V_PROJ, rows_mul(L->o_w, row1(L->o_b, bias, W), xb, nt, W, q, W, &a));
		VSTAGE(V_RESID, for (i = 0; i < nt * W; i++)
			x[i] += q[i]);

		{
			float *g = malloc((size_t)W * sizeof(float));
			float *b = malloc((size_t)W * sizeof(float));

			if (!g || !b) { free(g); free(b); goto out; }
			row1(L->ln2_w, g, W);
			VSTAGE(V_NORM, for (p = 0; p < nt; p++)
				layernorm(xb + (size_t)p * W, x + (size_t)p * W,
					  L->ln2_w ? g : NULL,
					  L->ln2_b ? row1(L->ln2_b, b, W) : NULL,
					  W, v->eps));
			free(g); free(b);
		}

		/*
		 * ⚠⚠ THE FEED FORWARD IS NOT BANDWIDTH BOUND, AND THAT IS THE
		 * FINDING, not the speedup beside it.
		 *
		 * Counted out at SmolVLM-256M's shape -- 1024 patches, 768
		 * wide, 3072 in the middle, twelve layers, int8 on the NPU with
		 * charsiu_pool_rows chunking at 64 rows, so sixteen hardware
		 * dispatches per matmul. Per layer, in MB:
		 *
		 *   fc1  weights 2.36 x 16 dispatches         37.7
		 *        activation read f32 / written int8    3.9
		 *        NPU reads it back                     0.8
		 *        NPU writes int32 1024 x 3072         12.6
		 *        the gather reads it -- 16 byte runs
		 *        off 64 byte lines, so four times     50.3
		 *        the gather writes ff                 12.6
		 *        bias pass, then activation pass      50.4  -> 25.2
		 *   fc2  weights 2.36 x 16                    37.7
		 *        ff read f32 / written int8           15.8
		 *        NPU reads it back / writes int32      6.3
		 *        the gather, again x4                 15.8
		 *        bias pass                             6.3
		 *                                            -----
		 *                                            249.7  -> 224.5
		 *
		 * Twelve layers is 3.00 GB, and 2.69 after the fusion below.
		 * The board measures 7.13 GB/s, so the ROOF for this stage is
		 * about 420 ms and it was reported at 2398. It is five and a
		 * half times off the roof: there is no byte argument for this
		 * stage's time, and the two things that fill the gap are the
		 * activation (about 1030 ms, which is what this round removes)
		 * and 384 hardware dispatches at roughly 3.3 ms each against
		 * about 1.4 ms of bytes in one.
		 *
		 * ⚠ AND 905 MB OF THE 3.00 GB IS THE WEIGHTS READ SIXTEEN
		 * TIMES. 30% of the traffic exists only because the batch is
		 * cut into 64 row chunks -- at one chunk it would be 57 MB. It
		 * is not a thing to fix here: 80 is the last width whose output
		 * is identical to two rows and 96 is the first that is not, and
		 * that bound is the board's, written down in npupool.c.
		 */
		/*
		 * ⚠ THE BIAS GOES TO bias_act, NOT TO rows_mul, and that is the
		 * whole of the fusion. rows_mul's own bias loop is a separate
		 * full pass over an intermediate that is 12.6 MB at this shape;
		 * handing it to the activation instead visits those numbers
		 * once. `bias` is the shared scratch row and fc2's row1 below
		 * overwrites it, which is why fb is read out before that.
		 */
		{
			const float *fb = row1(L->fc1_b, bias, nff);

			VSTAGE(V_FFN, rows_mul(L->fc1_w, NULL, xb, nt, W, ff,
					       nff, &a));
			VSTAGE(V_ACT, bias_act(ff, nt, nff, fb, v->use_gelu));
		}
		VSTAGE(V_FFN, rows_mul(L->fc2_w, row1(L->fc2_b, bias, W), ff,
				       nt, nff, q, W, &a));
		VSTAGE(V_RESID, for (i = 0; i < nt * W; i++)
			x[i] += q[i]);
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
		VSTAGE(V_SHUF, pixel_shuffle(x, sh, v->grid, W, v->scale));
		VSTAGE(V_SHUF, rows_mul(v->fc_w, row1(v->fc_b, bias,
				v->proj_dim), sh, tok, wide2, out,
				(unsigned)rows_of(v->fc_w), &a));
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

			bias_act(h0, np, d0, NULL, v->use_gelu);
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
