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
 * ⚠ THE TANH APPROXIMATION, matching gelu_mul in llama.c and ggml's GGML_OP_GELU.
 * This one is NOT gated: a ViT's feed forward is fc1 -> GELU -> fc2, with no
 * second branch to multiply against.
 */
static void gelu(float *x, unsigned n)
{
	unsigned i;

	for (i = 0; i < n; i++) {
		float v = x[i];

		x[i] = 0.5f * v *
		       (1.0f + tanhf(0.7978845608028654f *
				     (v + 0.044715f * v * v * v)));
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
static void rows_mul(const struct gguf_tensor *w, const float *bias,
		     const float *X, unsigned m, unsigned k, float *Y,
		     unsigned nout, struct charsiu_act *a)
{
	struct gguf_tensor t = flat2d(w);
	unsigned r, i;

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
	return v->n_patches;
}

unsigned charsiu_vision_width(const struct charsiu_vision *v)
{
	if (v->mm_w[1])
		return (unsigned)rows_of(v->mm_w[1]);
	if (v->mm_w[0])
		return (unsigned)rows_of(v->mm_w[0]);
	return v->n_embd;
}

int charsiu_vision_encode(struct charsiu_vision *v, const float *px, float *out)
{
	unsigned np = v->n_patches, W = v->n_embd, P = v->patch_size;
	unsigned pin = 3u * P * P, hd, l, p, i, j, h, e;
	unsigned nff = v->n_ff, wide = W > nff ? W : nff;
	float *patch = NULL, *x = NULL, *xb = NULL, *q = NULL, *k = NULL;
	float *val = NULL, *att = NULL, *ff = NULL, *tmp = NULL, *gain = NULL;
	float *bias = NULL, scale;
	struct charsiu_act a;
	int rc = -1;

	if (!v->opened || v->n_missing || !np || !v->n_head)
		return -1;
	hd = W / v->n_head;
	if (!hd || hd * v->n_head != W)
		return -1;
	scale = 1.0f / sqrtf((float)hd);

	if (charsiu_act_alloc(&a, (int)(pin > wide ? pin : wide)))
		return -1;
	patch = malloc((size_t)np * pin * sizeof(float));
	x     = malloc((size_t)np * W * sizeof(float));
	xb    = malloc((size_t)np * W * sizeof(float));
	q     = malloc((size_t)np * W * sizeof(float));
	k     = malloc((size_t)np * W * sizeof(float));
	val   = malloc((size_t)np * W * sizeof(float));
	ff    = malloc((size_t)np * (size_t)nff * sizeof(float));
	att   = malloc((size_t)np * sizeof(float));
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

	rows_mul(v->patch_w, row1(v->patch_b, bias, W), patch, np, pin, x, W, &a);

	/* the position embedding is one row per patch */
	for (p = 0; p < np; p++) {
		gguf_row_f32(v->pos_embd, p, tmp);
		for (i = 0; i < W; i++)
			x[(size_t)p * W + i] += tmp[i];
	}

	if (v->pre_ln_w) {
		const float *g = row1(v->pre_ln_w, gain, W);
		const float *b = row1(v->pre_ln_b, bias, W);

		for (p = 0; p < np; p++)
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
			for (p = 0; p < np; p++)
				layernorm(xb + (size_t)p * W, x + (size_t)p * W,
					  L->ln1_w ? g : NULL,
					  L->ln1_b ? row1(L->ln1_b, b, W) : NULL,
					  W, v->eps);
			free(g); free(b);
		}

		rows_mul(L->q_w, row1(L->q_b, bias, W), xb, np, W, q, W, &a);
		rows_mul(L->k_w, row1(L->k_b, bias, W), xb, np, W, k, W, &a);
		rows_mul(L->v_w, row1(L->v_b, bias, W), xb, np, W, val, W, &a);

		/* full attention, every patch against every patch */
		for (h = 0; h < v->n_head; h++) {
			unsigned off = h * hd;

			for (i = 0; i < np; i++) {
				const float *qi = q + (size_t)i * W + off;
				float *o = xb + (size_t)i * W + off;

				for (j = 0; j < np; j++) {
					const float *kj = k + (size_t)j * W + off;
					float d = 0.0f;

					for (e = 0; e < hd; e++)
						d += qi[e] * kj[e];
					att[j] = d * scale;
				}
				vsoftmax(att, np);
				for (e = 0; e < hd; e++)
					o[e] = 0.0f;
				for (j = 0; j < np; j++) {
					const float *vj = val + (size_t)j * W + off;
					float w = att[j];

					for (e = 0; e < hd; e++)
						o[e] += w * vj[e];
				}
			}
		}

		rows_mul(L->o_w, row1(L->o_b, bias, W), xb, np, W, q, W, &a);
		for (i = 0; i < np * W; i++)
			x[i] += q[i];

		{
			float *g = malloc((size_t)W * sizeof(float));
			float *b = malloc((size_t)W * sizeof(float));

			if (!g || !b) { free(g); free(b); goto out; }
			row1(L->ln2_w, g, W);
			for (p = 0; p < np; p++)
				layernorm(xb + (size_t)p * W, x + (size_t)p * W,
					  L->ln2_w ? g : NULL,
					  L->ln2_b ? row1(L->ln2_b, b, W) : NULL,
					  W, v->eps);
			free(g); free(b);
		}

		rows_mul(L->up_w, row1(L->up_b, bias, nff), xb, np, W, ff, nff,
			 &a);
		gelu(ff, np * nff);
		rows_mul(L->down_w, row1(L->down_b, bias, W), ff, np, nff, q, W,
			 &a);
		for (i = 0; i < np * W; i++)
			x[i] += q[i];
	}

	if (v->post_ln_w) {
		float *g = malloc((size_t)W * sizeof(float));
		float *b = malloc((size_t)W * sizeof(float));

		if (!g || !b) { free(g); free(b); goto out; }
		row1(v->post_ln_w, g, W);
		for (p = 0; p < np; p++)
			layernorm(x + (size_t)p * W, x + (size_t)p * W, g,
				  v->post_ln_b ? row1(v->post_ln_b, b, W) : NULL,
				  W, v->eps);
		free(g); free(b);
	}

	if (v->mm_w[0]) {
		unsigned d0 = (unsigned)rows_of(v->mm_w[0]);
		float *h0 = malloc((size_t)np * d0 * sizeof(float));

		if (!h0)
			goto out;
		rows_mul(v->mm_w[0], row1(v->mm_b[0], bias, d0), x, np, W, h0,
			 d0, &a);
		if (v->mm_w[1]) {
			unsigned d1 = (unsigned)rows_of(v->mm_w[1]);

			gelu(h0, np * d0);
			rows_mul(v->mm_w[1], row1(v->mm_b[1], bias, d1), h0, np,
				 d0, out, d1, &a);
		} else {
			memcpy(out, h0, (size_t)np * d0 * sizeof(float));
		}
		free(h0);
	} else {
		memcpy(out, x, (size_t)np * W * sizeof(float));
	}
	rc = 0;
out:
	free(patch); free(x); free(xb); free(q); free(k); free(val);
	free(ff); free(att); free(tmp); free(gain); free(bias);
	charsiu_act_free(&a);
	return rc;
}
