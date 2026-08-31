// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * CLIP's text tower, and the tokenizer that is not the language model's.
 *
 * See charsiu_clip.h for what differs from the vision tower. The short of it is
 * a causal mask, a pooled row that is the END OF TEXT position, and a BPE whose
 * merge list is not in the file.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>

#include "charsiu_llm.h"
#include "charsiu_clip.h"

/* ---- the loader, shape checked the same way the vision one is ------------ */

static void miss(struct charsiu_clip_text *t, const char *fmt, ...)
{
	va_list ap;

	if (t->n_missing >= sizeof(t->missing) / sizeof(*t->missing))
		return;
	va_start(ap, fmt);
	vsnprintf(t->missing[t->n_missing++], sizeof(t->missing[0]), fmt, ap);
	va_end(ap);
}

static void wshape(const struct gguf_tensor *g, uint64_t *in, uint64_t *out)
{
	unsigned d;

	*in = 1;
	for (d = 0; d + 1 < (g->n_dims ? g->n_dims : 1); d++)
		*in *= g->ne[d];
	*out = g->n_dims ? g->ne[g->n_dims - 1] : 1;
}

static const struct gguf_tensor *bind(struct charsiu_clip_text *t,
				      const char *fmt, int idx, int opt,
				      uint64_t in, uint64_t out)
{
	const struct gguf_tensor *g;
	uint64_t gi, go;
	char name[80];

	if (idx >= 0)
		snprintf(name, sizeof(name), fmt, idx);
	else
		snprintf(name, sizeof(name), "%s", fmt);
	g = gguf_tensor(&t->g, name);
	if (!g) {
		if (!opt)
			miss(t, "%s", name);
		return NULL;
	}
	wshape(g, &gi, &go);
	if ((in && gi != in) || (out && go != out)) {
		miss(t, "%s is %llux%llu, wanted %llux%llu", name,
		     (unsigned long long)gi, (unsigned long long)go,
		     (unsigned long long)in, (unsigned long long)out);
		return NULL;
	}
	return g;
}

static const struct gguf_tensor *bind1(struct charsiu_clip_text *t,
				       const char *fmt, int idx, int opt,
				       uint64_t n)
{
	return bind(t, fmt, idx, opt, 0, n);
}

/* ---- the vocabulary ------------------------------------------------------ */

/*
 * ⚠ A HASH OVER THE PIECES, because the tokenizer asks "is this string a token"
 * once per adjacent pair per merge step, and a linear scan of 49408 entries
 * would make a three word prompt take longer than the twelve transformer
 * blocks it feeds.
 */
struct vslot { const char *s; int32_t id; };
static struct vslot *vtab;
static uint32_t vmask;

static uint32_t vhash(const char *s)
{
	uint32_t h = 2166136261u;

	while (*s)
		h = (h ^ (uint8_t)*s++) * 16777619u;
	return h;
}

static void vtab_build(char **vocab, uint32_t n)
{
	uint32_t cap = 1, i;

	while (cap < n * 2)
		cap <<= 1;
	free(vtab);
	vtab = calloc(cap, sizeof(*vtab));
	vmask = cap - 1;
	if (!vtab)
		return;
	for (i = 0; i < n; i++) {
		uint32_t h = vhash(vocab[i]) & vmask;

		while (vtab[h].s)
			h = (h + 1) & vmask;
		vtab[h].s = vocab[i];
		vtab[h].id = (int32_t)i;
	}
}

static int32_t vlookup(const char *s)
{
	uint32_t h;

	if (!vtab)
		return -1;
	h = vhash(s) & vmask;
	while (vtab[h].s) {
		if (!strcmp(vtab[h].s, s))
			return vtab[h].id;
		h = (h + 1) & vmask;
	}
	return -1;
}

static const uint8_t *arr_str(const uint8_t *p, uint64_t *n)
{
	memcpy(n, p, sizeof(*n));
	return p + sizeof(*n);
}

static int load_vocab(struct charsiu_clip_text *t)
{
	const struct gguf_kv *kv = gguf_find(&t->g, "tokenizer.ggml.tokens");
	const uint8_t *p;
	uint32_t i;

	if (!kv || kv->type != GGUF_V_ARRAY || kv->arr_type != GGUF_V_STRING) {
		miss(t, "tokenizer.ggml.tokens (a key, not a tensor)");
		return -1;
	}
	t->n_vocab = (uint32_t)kv->arr_len;
	t->vocab = calloc(t->n_vocab, sizeof(*t->vocab));
	if (!t->vocab)
		return -1;
	p = kv->arr;
	for (i = 0; i < t->n_vocab; i++) {
		uint64_t n;

		p = arr_str(p, &n);
		t->vocab[i] = malloc(n + 1);
		if (!t->vocab[i])
			return -1;
		memcpy(t->vocab[i], p, n);
		t->vocab[i][n] = 0;
		p += n;
	}
	vtab_build(t->vocab, t->n_vocab);
	t->sot = vlookup("<|startoftext|>");
	t->eot = vlookup("<|endoftext|>");
	if (t->sot < 0 || t->eot < 0)
		miss(t, "the start and end of text markers are not in the vocabulary");
	return 0;
}

int charsiu_clip_text_open(struct charsiu_clip_text *t, const char *path)
{
	uint32_t i, has = 0;

	memset(t, 0, sizeof(*t));
	if (gguf_open(&t->g, path) < 0) {
		snprintf(t->why, sizeof(t->why), "%s will not open as a gguf",
			 path);
		return -1;
	}
	t->opened = 1;
	if (gguf_get_u32(&t->g, "clip.has_text_encoder", &has) != 0 || !has) {
		snprintf(t->why, sizeof(t->why),
			 "%s has no text encoder. An mmproj for a language "
			 "model carries the vision half only; a two tower CLIP "
			 "carries both", path);
		return -1;
	}
	if (gguf_get_u32(&t->g, "clip.text.embedding_length", &t->n_embd) ||
	    gguf_get_u32(&t->g, "clip.text.feed_forward_length", &t->n_ff) ||
	    gguf_get_u32(&t->g, "clip.text.block_count", &t->n_layer) ||
	    gguf_get_u32(&t->g, "clip.text.attention.head_count", &t->n_head)) {
		miss(t, "one of clip.text.{embedding_length, "
		     "feed_forward_length, block_count, head_count}");
	}
	if (gguf_get_u32(&t->g, "clip.text.context_length", &t->n_ctx) != 0)
		t->n_ctx = 77;
	if (gguf_get_u32(&t->g, "clip.text.projection_dim", &t->proj_dim) != 0)
		t->proj_dim = t->n_embd;
	if (gguf_get_f32(&t->g, "clip.text.attention.layer_norm_epsilon",
			 &t->eps) != 0)
		t->eps = 1e-5f;
	{
		uint32_t ug = 1;

		gguf_get_u32(&t->g, "clip.use_gelu", &ug);
		t->use_gelu = ug ? 1 : 0;
	}

	if (load_vocab(t) && !t->n_missing)
		miss(t, "the vocabulary would not load");

	{
		uint32_t W = t->n_embd, F = t->n_ff;

		t->tok_embd  = bind(t, "t.token_embd.weight", -1, 0, W, 0);
		t->pos_embd  = bind(t, "t.position_embd.weight", -1, 0, W,
				    t->n_ctx);
		t->post_ln_w = bind1(t, "t.post_ln.weight", -1, 0, W);
		t->post_ln_b = bind1(t, "t.post_ln.bias", -1, 1, W);
		t->proj_w    = bind(t, "text_projection.weight", -1, 0, W,
				    t->proj_dim);

		t->layer = t->n_layer ? calloc(t->n_layer, sizeof(*t->layer))
				      : NULL;
		if (t->n_layer && !t->layer) {
			snprintf(t->why, sizeof(t->why),
				 "%u layers will not allocate", t->n_layer);
			return -1;
		}
		for (i = 0; i < t->n_layer; i++) {
			struct charsiu_clip_layer *L = &t->layer[i];

			L->ln1_w = bind1(t, "t.blk.%d.ln1.weight", (int)i, 0, W);
			L->ln1_b = bind1(t, "t.blk.%d.ln1.bias", (int)i, 1, W);
			L->q_w = bind(t, "t.blk.%d.attn_q.weight", (int)i, 0, W, W);
			L->q_b = bind1(t, "t.blk.%d.attn_q.bias", (int)i, 1, W);
			L->k_w = bind(t, "t.blk.%d.attn_k.weight", (int)i, 0, W, W);
			L->k_b = bind1(t, "t.blk.%d.attn_k.bias", (int)i, 1, W);
			L->v_w = bind(t, "t.blk.%d.attn_v.weight", (int)i, 0, W, W);
			L->v_b = bind1(t, "t.blk.%d.attn_v.bias", (int)i, 1, W);
			L->o_w = bind(t, "t.blk.%d.attn_out.weight", (int)i, 0, W, W);
			L->o_b = bind1(t, "t.blk.%d.attn_out.bias", (int)i, 1, W);
			L->ln2_w = bind1(t, "t.blk.%d.ln2.weight", (int)i, 0, W);
			L->ln2_b = bind1(t, "t.blk.%d.ln2.bias", (int)i, 1, W);
			/* the same backwards naming as the vision tower */
			L->fc1_w = bind(t, "t.blk.%d.ffn_down.weight", (int)i, 0, W, F);
			L->fc1_b = bind1(t, "t.blk.%d.ffn_down.bias", (int)i, 1, F);
			L->fc2_w = bind(t, "t.blk.%d.ffn_up.weight", (int)i, 0, F, W);
			L->fc2_b = bind1(t, "t.blk.%d.ffn_up.bias", (int)i, 1, W);
			if (t->n_missing >= sizeof(t->missing) / sizeof(*t->missing))
				break;
		}
	}

	if (t->n_missing) {
		snprintf(t->why, sizeof(t->why),
			 "%u of the text tower's pieces are not in this file "
			 "under the names this reads", t->n_missing);
		return -1;
	}
	return 0;
}

void charsiu_clip_text_close(struct charsiu_clip_text *t)
{
	uint32_t i;

	for (i = 0; i < t->n_vocab && t->vocab; i++)
		free(t->vocab[i]);
	free(t->vocab);
	t->vocab = NULL;
	free(t->layer);
	t->layer = NULL;
	free(vtab);
	vtab = NULL;
	if (t->opened)
		gguf_close(&t->g);
	t->opened = 0;
}

const char *charsiu_clip_text_why_not(const struct charsiu_clip_text *t)
{
	return t->why[0] ? t->why : NULL;
}

void charsiu_clip_text_describe(const struct charsiu_clip_text *t, FILE *out)
{
	unsigned i;

	fprintf(out, "text tower\n");
	fprintf(out, "  width        %u, ffn %u, heads %u, layers %u\n",
		t->n_embd, t->n_ff, t->n_head, t->n_layer);
	fprintf(out, "  context      %u, projection to %u\n", t->n_ctx,
		t->proj_dim);
	fprintf(out, "  vocabulary   %u, start %d, end %d\n", t->n_vocab,
		t->sot, t->eot);
	if (!t->n_missing) {
		fprintf(out, "  complete     every piece this needs is here\n");
		return;
	}
	fprintf(out, "  MISSING      %u:\n", t->n_missing);
	for (i = 0; i < t->n_missing; i++)
		fprintf(out, "    %s\n", t->missing[i]);
}

/* ---- the tokenizer ------------------------------------------------------- */

/*
 * ⚠ THE BYTE ENCODER, which is GPT-2's and CLIP inherits it. Bytes 33..126,
 * 161..172 and 174..255 stand for themselves as codepoints; every other byte
 * gets 256 + n, in order. The vocabulary holds the UTF-8 of those codepoints,
 * so a byte outside the printable range is not itself in the file.
 *
 * For a word made only of ASCII 33..126 -- which after the whitespace split is
 * every English word -- this is the identity, and that is the case the cross
 * check covers.
 */
static void byte_to_piece(uint8_t b, char *out)
{
	unsigned cp;

	if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) {
		cp = b;
	} else {
		unsigned n = 0, i;

		for (i = 0; i < b; i++)
			if (!((i >= 33 && i <= 126) || (i >= 161 && i <= 172) ||
			      i >= 174))
				n++;
		cp = 256 + n;
	}
	if (cp < 0x80) {
		out[0] = (char)cp;
		out[1] = 0;
	} else {
		out[0] = (char)(0xC0 | (cp >> 6));
		out[1] = (char)(0x80 | (cp & 0x3F));
		out[2] = 0;
	}
}

/*
 * ⚠ BIG ENOUGH FOR A MERGED SYMBOL, which is the whole word by the end. The
 * first version sized these at eight bytes, which is fine for the single
 * characters a word starts as and truncates the moment two merges have
 * happened -- "photograph</w>" is fourteen. A truncated symbol is not found in
 * the vocabulary, so the merge stops early and the sentence tokenizes into
 * more, shorter pieces. Every one of them is a real token and the model
 * answers, which is why this has to be sized rather than noticed.
 */
#define MAXSYM 264
#define SYMLEN 72

/*
 * One word into ids, by the merge rule described in charsiu_clip.h: repeatedly
 * join the adjacent pair whose concatenation sits EARLIEST in the vocabulary.
 */
static int bpe_word(const struct charsiu_clip_text *t, const char *w,
		    int32_t *ids, int max)
{
	(void)t;   /* the vocabulary is in vtab, built at load */
	char sym[MAXSYM][SYMLEN];
	char join[2 * SYMLEN];
	int n = 0, i, out = 0;

	for (i = 0; w[i] && n < MAXSYM - 1; i++)
		byte_to_piece((uint8_t)w[i], sym[n++]);
	if (!n)
		return 0;
	/* the last symbol carries the end of word marker */
	strncat(sym[n - 1], "</w>", sizeof(sym[0]) - strlen(sym[n - 1]) - 1);

	for (;;) {
		int best = -1;
		int32_t bestid = 0;

		for (i = 0; i + 1 < n; i++) {
			size_t la = strlen(sym[i]), lb = strlen(sym[i + 1]);
			int32_t id;

			/* ⚠ a pair that will not fit is not merged rather than
			 * merged truncated: a short symbol is a real token and
			 * would be accepted silently. */
			if (la + lb >= SYMLEN)
				continue;
			memcpy(join, sym[i], la);
			memcpy(join + la, sym[i + 1], lb + 1);
			id = vlookup(join);
			if (id >= 0 && (best < 0 || id < bestid)) {
				best = i;
				bestid = id;
			}
		}
		if (best < 0)
			break;
		{
			size_t la = strlen(sym[best]);
			size_t lb = strlen(sym[best + 1]);

			memcpy(sym[best] + la, sym[best + 1], lb + 1);
		}
		for (i = best + 1; i + 1 < n; i++)
			memcpy(sym[i], sym[i + 1], sizeof(sym[0]));
		n--;
	}

	for (i = 0; i < n && out < max; i++) {
		int32_t id = vlookup(sym[i]);

		if (id >= 0)
			ids[out++] = id;
	}
	return out;
}

static int is_alpha(char c) { return c >= 'a' && c <= 'z'; }
static int is_digit(char c) { return c >= '0' && c <= '9'; }

int charsiu_clip_tokenize(const struct charsiu_clip_text *t, const char *text,
			  int32_t *ids, int max, int *non_ascii)
{
	static const char *contract[] = { "'s", "'t", "'re", "'ve", "'m",
					  "'ll", "'d", NULL };
	int n = 0;
	const char *p;
	char *low;
	size_t i, len = strlen(text);

	if (non_ascii)
		*non_ascii = 0;
	if (max < 3)
		return -1;
	low = malloc(len + 1);
	if (!low)
		return -1;
	for (i = 0; i < len; i++) {
		unsigned char c = (unsigned char)text[i];

		if (c >= 0x80 && non_ascii)
			*non_ascii = 1;
		low[i] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : (char)c;
	}
	low[len] = 0;

	ids[n++] = t->sot;
	p = low;
	while (*p && n < max - 1) {
		char word[256];
		size_t w = 0;
		int k;

		if (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
			p++;
			continue;
		}
		/* the contractions come first, as they do in CLIP's pattern */
		for (k = 0; contract[k]; k++) {
			size_t cl = strlen(contract[k]);

			if (!strncmp(p, contract[k], cl)) {
				memcpy(word, p, cl);
				word[cl] = 0;
				p += cl;
				w = cl;
				break;
			}
		}
		if (!w) {
			if (is_alpha(*p)) {
				while (is_alpha(*p) && w < sizeof(word) - 1)
					word[w++] = *p++;
			} else if (is_digit(*p)) {
				/*
				 * ⚠ ONE DIGIT AT A TIME. CLIP's pattern is
				 * [\p{N}] and not [\p{N}]+, so "2024" is four
				 * words and four tokens, and reading it as a
				 * run gives a different sentence.
				 */
				word[w++] = *p++;
			} else {
				while (*p && *p != ' ' && *p != '\t' &&
				       *p != '\n' && *p != '\r' &&
				       !is_alpha(*p) && !is_digit(*p) &&
				       w < sizeof(word) - 1)
					word[w++] = *p++;
			}
		}
		word[w] = 0;
		if (!w)
			break;
		n += bpe_word(t, word, ids + n, max - 1 - n);
	}
	ids[n++] = t->eot;
	free(low);
	return n;
}

float charsiu_cosine(const float *a, const float *b, unsigned n)
{
	float d = 0.0f, na = 0.0f, nb = 0.0f;
	unsigned i;

	for (i = 0; i < n; i++) {
		d += a[i] * b[i];
		na += a[i] * a[i];
		nb += b[i] * b[i];
	}
	if (na <= 0.0f || nb <= 0.0f)
		return 0.0f;
	return d / (sqrtf(na) * sqrtf(nb));
}

/* ---- the forward pass ---------------------------------------------------- */

static void tlayernorm(float *out, const float *x, const float *w,
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
	for (i = 0; i < n; i++)
		out[i] = (x[i] - mean) * inv * (w ? w[i] : 1.0f) +
			 (b ? b[i] : 0.0f);
}

/*
 * ⚠⚠ THE TANH GELU IS A SIGMOID, WHICH IS AN IDENTITY AND NOT AN
 * APPROXIMATION OF AN APPROXIMATION. tanh y = 1 - 2/(e^2y + 1), so
 *
 *     0.5 (1 + tanh y) = 1 / (1 + e^-2y)
 *
 * and the activation is x / (1 + e^(-2k(x + 0.044715 x^3))), k = sqrt(2/pi):
 * the same function with one exponential in place of a hyperbolic tangent,
 * which is the shape gelu quick was already written in. It goes four at a time
 * through the charsiu_vexpq the language model's SiLU built. The vision tower
 * priced the identical loop at its own shape -- 34.65 ms with tanhf against
 * 2.66 with an exponential over 3145728 elements -- and this is the third and
 * last copy of it in the tree.
 *
 * ⚠ GELU QUICK'S FORMULA DID NOT CHANGE, only its expf became charsiu_vexpq,
 * so the whole of the risk on that branch is the polynomial. It is worth
 * saying out loud that nothing here exercises it: clip.use_gelu is 1 on every
 * checkpoint on this desk and on the synthetic model clip_cross.py builds, so
 * that branch is one no test catches. It is written to match the tanh branch
 * line for line for exactly that reason.
 *
 * ⚠⚠ AND IT IS DELIBERATELY NOT THREADED, WHICH IS A MEASUREMENT AND NOT AN
 * OVERSIGHT. The vision tower puts its activation on charsiu_parallel_for and
 * whisper's encoder does too, because a picture is 1024 patches by 3072 wide
 * and a mel window is 1500 frames by 1536; A CAPTION IS NEITHER. This tower's
 * feed forward is n_ff wide -- 2048 on a ViT-B text encoder, 3072 on a ViT-L
 * -- and n is the number of tokens the caption ACTUALLY HAS, not the context
 * length, so one call is a few thousand elements. On this six core host, NEON
 * alone against the same kernel over the pool, best of two hundred:
 *
 *     14336   (a 7 token caption)   0.009 ms   0.040 ms   the pool loses 4.4x
 *    157696   (n_ctx = 77, the       0.088      0.086     a dead heat
 *              longest input this
 *              tower can be given)
 *
 * A pool dispatch costs about 0.040 ms of broadcast and barrier on this host
 * whatever is behind it, and the LONGEST caption CLIP can be handed only just
 * pays for one. So the floor whisper needs -- 262144 elements, swept there --
 * would exclude every call this tower ever makes, and the code that would
 * implement it is code that never runs.
 *
 * ⚠ AND IT WOULD NOT EVEN HAVE RUN. charsiu_parallel_for on an unstarted pool
 * is a plain call on one thread -- it looks exactly like success -- and
 * nothing on the text tower's path calls charsiu_threads_start. The vision
 * tower and whisper both do it in their open. A parallel_for added here would
 * have been serial, silently, and measured as a win against tanhf either way.
 *
 * ⚠ WHAT THE VECTOR KERNEL ALONE IS WORTH, in situ and not by arithmetic. A
 * ViT-B/32 text tower is 512 wide, 2048 in the middle and twelve layers deep;
 * one seven token caption through it, best of thirty, three times interleaved
 * against a binary built from before this change:
 *
 *     before                       21.686  21.600  21.543 ms
 *     CHARSIU_EXACT_GELU=1         21.582  21.409  21.414     the control
 *     shipped                      20.066  20.228  20.144
 *
 * so about 1.5 ms of a 21.6 ms encode, 6.7%, for one branch and no threads --
 * and the control lands on the old number, which is what says the 1.5 ms is
 * the exponential and not the restructuring.
 *
 * CHARSIU_EXACT_GELU puts the tanhf form and libm's expf back, on both
 * branches, as the control this has to be diffed against. It is the same
 * variable the vision tower and the language model take, so one switch covers
 * every gelu in the runtime -- and with it set this tower's projection is bit
 * identical to the binary from before the change.
 */
static int gelu_exact(void)
{
	static int v = -1;

	if (v < 0)
		v = getenv("CHARSIU_EXACT_GELU") != NULL;
	return v;
}

static void tgelu(float *x, unsigned n, int tanh_form)
{
	const float k2 = 2.0f * 0.7978845608028654f;   /* 2 sqrt(2/pi) */
	unsigned i = 0;

	if (gelu_exact()) {
		for (; i < n; i++) {
			float v = x[i];

			if (tanh_form)
				x[i] = 0.5f * v *
				       (1.0f + tanhf(0.7978845608028654f *
						     (v + 0.044715f * v * v * v)));
			else
				x[i] = v / (1.0f + expf(-1.702f * v));
		}
		return;
	}

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
	{
		const float32x4_t one = vdupq_n_f32(1.0f);
		const float32x4_t c = vdupq_n_f32(0.044715f);
		const float32x4_t kk = vdupq_n_f32(tanh_form ? -k2 : -1.702f);

		/*
		 * ⚠ THE BRANCH STAYS INSIDE THE LOOP, as it does in the
		 * tower. It is one perfectly predicted test against a divide
		 * and a six term polynomial, and hoisting it means two copies
		 * of the kernel for a difference the tower measured at 2.13 ms
		 * against 2.09 over three million elements.
		 */
		for (; i + 4 <= n; i += 4) {
			float32x4_t v = vld1q_f32(x + i), e;

			if (tanh_form) {
				float32x4_t v3 = vmulq_f32(vmulq_f32(v, v), v);

				e = vmulq_f32(kk, vmlaq_f32(v, c, v3));
			} else {
				e = vmulq_f32(kk, v);
			}
			vst1q_f32(x + i,
				  vdivq_f32(v, vaddq_f32(one, charsiu_vexpq(e))));
		}
	}
#endif
	for (; i < n; i++) {
		float v = x[i];

		if (tanh_form)
			x[i] = v / (1.0f + expf(-k2 * (v + 0.044715f * v * v * v)));
		else
			x[i] = v / (1.0f + expf(-1.702f * v));
	}
}

static void tsoftmax(float *x, unsigned n)
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

static struct gguf_tensor tflat2d(const struct gguf_tensor *w)
{
	struct gguf_tensor g = *w;
	uint64_t in = 1;
	unsigned d;

	if (w->n_dims <= 2)
		return g;
	for (d = 0; d + 1 < w->n_dims; d++)
		in *= w->ne[d];
	g.n_dims = 2;
	g.ne[0] = in;
	g.ne[1] = w->ne[w->n_dims - 1];
	g.ne[2] = g.ne[3] = 1;
	return g;
}

static void trows(const struct gguf_tensor *w, const float *bias,
		  const float *X, unsigned m, unsigned k, float *Y,
		  unsigned nout, struct charsiu_act *a)
{
	struct gguf_tensor g = tflat2d(w);
	unsigned r, i;

	for (r = 0; r < m; r++) {
		float *y = Y + (size_t)r * nout;

		charsiu_act_set(a, X + (size_t)r * k, (int)k);
		gguf_matvec(&g, a, y, 0, nout);
		if (bias)
			for (i = 0; i < nout; i++)
				y[i] += bias[i];
	}
}

static const float *trow1(const struct gguf_tensor *g, float *buf)
{
	if (!g)
		return NULL;
	gguf_row_f32(g, 0, buf);
	return buf;
}

int charsiu_clip_encode_text(struct charsiu_clip_text *t, const int32_t *ids,
			     int n, float *out)
{
	unsigned W = t->n_embd, F = t->n_ff, hd, l, i, j, h, e;
	unsigned wide = W > F ? W : F;
	float *x = NULL, *xb = NULL, *q = NULL, *k = NULL, *v = NULL;
	float *ff = NULL, *att = NULL, *tmp = NULL, *g1 = NULL, *b1 = NULL;
	float scale;
	struct charsiu_act a;
	int rc = -1;

	if (!t->opened || t->n_missing || n < 1 || (unsigned)n > t->n_ctx)
		return -1;
	hd = W / t->n_head;
	if (!hd || hd * t->n_head != W)
		return -1;
	scale = 1.0f / sqrtf((float)hd);

	if (charsiu_act_alloc(&a, (int)wide))
		return -1;
	x   = malloc((size_t)n * W * sizeof(float));
	xb  = malloc((size_t)n * W * sizeof(float));
	q   = malloc((size_t)n * W * sizeof(float));
	k   = malloc((size_t)n * W * sizeof(float));
	v   = malloc((size_t)n * W * sizeof(float));
	ff  = malloc((size_t)n * F * sizeof(float));
	att = malloc((size_t)n * sizeof(float));
	tmp = malloc((size_t)W * sizeof(float));
	g1  = malloc((size_t)wide * sizeof(float));
	b1  = malloc((size_t)wide * sizeof(float));
	if (!x || !xb || !q || !k || !v || !ff || !att || !tmp || !g1 || !b1)
		goto out;

	for (i = 0; i < (unsigned)n; i++) {
		gguf_row_f32(t->tok_embd, (uint64_t)ids[i], x + (size_t)i * W);
		gguf_row_f32(t->pos_embd, i, tmp);
		for (j = 0; j < W; j++)
			x[(size_t)i * W + j] += tmp[j];
	}

	for (l = 0; l < t->n_layer; l++) {
		struct charsiu_clip_layer *L = &t->layer[l];

		for (i = 0; i < (unsigned)n; i++)
			tlayernorm(xb + (size_t)i * W, x + (size_t)i * W,
				   trow1(L->ln1_w, g1), trow1(L->ln1_b, b1),
				   W, t->eps);
		trows(L->q_w, trow1(L->q_b, b1), xb, n, W, q, W, &a);
		trows(L->k_w, trow1(L->k_b, b1), xb, n, W, k, W, &a);
		trows(L->v_w, trow1(L->v_b, b1), xb, n, W, v, W, &a);

		/*
		 * ⚠ CAUSAL. This is the one structural difference from the
		 * vision tower, and it is invisible in the output: with full
		 * attention every embedding still comes back finite and the
		 * cosine similarities still order themselves plausibly. The
		 * END OF TEXT row, which is the one that becomes the sentence,
		 * is the row a missing mask changes least.
		 */
		for (h = 0; h < t->n_head; h++) {
			unsigned off = h * hd;

			for (i = 0; i < (unsigned)n; i++) {
				const float *qi = q + (size_t)i * W + off;
				float *o = xb + (size_t)i * W + off;

				for (j = 0; j <= i; j++)
					att[j] = charsiu_dot_f32(qi,
						k + (size_t)j * W + off, hd) *
						scale;
				tsoftmax(att, i + 1);
				for (e = 0; e < hd; e++)
					o[e] = 0.0f;
				for (j = 0; j <= i; j++)
					charsiu_axpy_f32(o,
						v + (size_t)j * W + off,
						att[j], hd);
			}
		}
		trows(L->o_w, trow1(L->o_b, b1), xb, n, W, q, W, &a);
		for (i = 0; i < (unsigned)n * W; i++)
			x[i] += q[i];

		for (i = 0; i < (unsigned)n; i++)
			tlayernorm(xb + (size_t)i * W, x + (size_t)i * W,
				   trow1(L->ln2_w, g1), trow1(L->ln2_b, b1),
				   W, t->eps);
		trows(L->fc1_w, trow1(L->fc1_b, b1), xb, n, W, ff, F, &a);
		tgelu(ff, (unsigned)n * F, t->use_gelu);
		trows(L->fc2_w, trow1(L->fc2_b, b1), ff, n, F, q, W, &a);
		for (i = 0; i < (unsigned)n * W; i++)
			x[i] += q[i];
	}

	/*
	 * ⚠ THE SENTENCE IS THE LAST ROW, and it is the last row because that
	 * is where the END OF TEXT token sits -- not because it is the end of
	 * the buffer. charsiu_clip_tokenize puts eot last, and a caller that
	 * builds ids some other way has to keep that true.
	 */
	tlayernorm(xb, x + (size_t)(n - 1) * W, trow1(t->post_ln_w, g1),
		   trow1(t->post_ln_b, b1), W, t->eps);
	trows(t->proj_w, NULL, xb, 1, W, out, t->proj_dim, &a);
	rc = 0;
out:
	free(x); free(xb); free(q); free(k); free(v);
	free(ff); free(att); free(tmp); free(g1); free(b1);
	charsiu_act_free(&a);
	return rc;
}
