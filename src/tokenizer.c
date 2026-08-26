// SPDX-License-Identifier: GPL-2.0-or-later
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */

/*
 * Byte level BPE, the GPT-2 flavour that Llama 3 uses.
 *
 * Three pieces, and the whole thing is wrong if any one of them is:
 *
 *   1. the byte to codepoint table, which is what makes every byte spellable
 *      as a printable character so the merge table can be plain text;
 *   2. the pre-tokenizer, which cuts the input into words BEFORE any merging,
 *      and whose exact cut decides the token ids;
 *   3. the merge loop, lowest rank first.
 *
 * The pre-tokenizer is the part most worth doubting: it is specified as a
 * Unicode regex, and what is here is a hand written scanner for the same
 * alternation. It is exact for ASCII. For everything above U+007F it leans on
 * a coarse table of which blocks are letters, which is right for ordinary
 * Latin, Greek, Cyrillic and CJK text and can be wrong for symbol blocks. That
 * is a stated approximation, not a claim.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu_llm.h"
#include "unicode_tables.h"

/* ---- byte <-> codepoint (the GPT-2 table) -------------------------------- */

/*
 * Bytes that are already printable ASCII or printable Latin-1 map to
 * themselves. The remaining 68 map to U+0100 upwards, in byte order. Built
 * rather than tabulated so the rule is visible.
 */
static uint32_t g_b2u[256];
static int16_t g_u2b[512];   /* codepoint - 0 .. 511 -> byte, or -1 */

static void build_byte_table(void)
{
	int n = 0;
	int used[256] = { 0 };

	for (int b = '!'; b <= '~'; b++) used[b] = 1;
	for (int b = 0xa1; b <= 0xac; b++) used[b] = 1;
	for (int b = 0xae; b <= 0xff; b++) used[b] = 1;

	for (int i = 0; i < 512; i++)
		g_u2b[i] = -1;

	for (int b = 0; b < 256; b++) {
		if (used[b])
			g_b2u[b] = (uint32_t)b;
		else
			g_b2u[b] = (uint32_t)(256 + n++);
		if (g_b2u[b] < 512)
			g_u2b[g_b2u[b]] = (int16_t)b;
	}
}

/* ---- utf8 ---------------------------------------------------------------- */

static int utf8_decode(const char *s, size_t n, uint32_t *cp)
{
	const uint8_t *u = (const uint8_t *)s;

	if (!n)
		return 0;
	if (u[0] < 0x80) { *cp = u[0]; return 1; }
	if ((u[0] & 0xe0) == 0xc0 && n >= 2) {
		*cp = ((uint32_t)(u[0] & 0x1f) << 6) | (u[1] & 0x3f);
		return 2;
	}
	if ((u[0] & 0xf0) == 0xe0 && n >= 3) {
		*cp = ((uint32_t)(u[0] & 0x0f) << 12) | ((uint32_t)(u[1] & 0x3f) << 6) | (u[2] & 0x3f);
		return 3;
	}
	if ((u[0] & 0xf8) == 0xf0 && n >= 4) {
		*cp = ((uint32_t)(u[0] & 0x07) << 18) | ((uint32_t)(u[1] & 0x3f) << 12) |
		      ((uint32_t)(u[2] & 0x3f) << 6) | (u[3] & 0x3f);
		return 4;
	}
	*cp = 0xfffd;
	return 1;
}

static int utf8_encode(uint32_t cp, char *out)
{
	if (cp < 0x80) { out[0] = (char)cp; return 1; }
	if (cp < 0x800) {
		out[0] = (char)(0xc0 | (cp >> 6));
		out[1] = (char)(0x80 | (cp & 0x3f));
		return 2;
	}
	if (cp < 0x10000) {
		out[0] = (char)(0xe0 | (cp >> 12));
		out[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
		out[2] = (char)(0x80 | (cp & 0x3f));
		return 3;
	}
	out[0] = (char)(0xf0 | (cp >> 18));
	out[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
	out[2] = (char)(0x80 | ((cp >> 6) & 0x3f));
	out[3] = (char)(0x80 | (cp & 0x3f));
	return 4;
}

/* ---- the character classes the pre-tokenizer needs ----------------------- */

static int cp_is_space(uint32_t c)
{
	return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0b ||
	       c == 0x0c || c == 0x85 || c == 0xa0 || c == 0x1680 ||
	       (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 ||
	       c == 0x202f || c == 0x205f || c == 0x3000;
}

/*
 * \p{L} and \p{N}, from the generated tables.
 *
 * These were an approximation by block until the fuzzer put Hebrew and Arabic
 * punctuation next to an apostrophe: U+05CF read as a letter, so the leading
 * quote went to the wrong side of the cut and every id after it moved. A block
 * is not a category, and the pre-tokenizer needs the category.
 */
static int in_ranges(const uint32_t *r, size_t n, uint32_t c)
{
	size_t lo = 0, hi = n / 2;

	while (lo < hi) {
		size_t mid = (lo + hi) / 2;

		if (c < r[2 * mid])
			hi = mid;
		else if (c > r[2 * mid + 1])
			lo = mid + 1;
		else
			return 1;
	}
	return 0;
}

static int cp_is_digit(uint32_t c)
{
	return in_ranges(g_number_ranges,
			 sizeof(g_number_ranges) / sizeof(g_number_ranges[0]), c);
}

static int cp_is_letter(uint32_t c)
{
	return in_ranges(g_letter_ranges,
			 sizeof(g_letter_ranges) / sizeof(g_letter_ranges[0]), c);
}

/* ---- a small open addressed string map ----------------------------------- */

struct smap {
	const char **key;
	uint64_t *klen;
	int32_t *val;
	size_t cap;      /* a power of two */
	size_t used;
};

static uint64_t fnv(const char *s, size_t n)
{
	uint64_t h = 1469598103934665603ull;

	for (size_t i = 0; i < n; i++) {
		h ^= (uint8_t)s[i];
		h *= 1099511628211ull;
	}
	return h;
}

static int smap_init(struct smap *m, size_t want)
{
	size_t cap = 16;

	while (cap < want * 2)
		cap <<= 1;
	m->key = calloc(cap, sizeof(*m->key));
	m->klen = calloc(cap, sizeof(*m->klen));
	m->val = calloc(cap, sizeof(*m->val));
	m->cap = cap;
	m->used = 0;
	if (!m->key || !m->klen || !m->val)
		return -1;
	for (size_t i = 0; i < cap; i++)
		m->val[i] = -1;
	return 0;
}

static void smap_free(struct smap *m)
{
	free(m->key);
	free(m->klen);
	free(m->val);
	memset(m, 0, sizeof(*m));
}

static void smap_put(struct smap *m, const char *k, uint64_t n, int32_t v)
{
	size_t i = fnv(k, n) & (m->cap - 1);

	while (m->val[i] >= 0) {
		if (m->klen[i] == n && !memcmp(m->key[i], k, n))
			return;                   /* first spelling wins */
		i = (i + 1) & (m->cap - 1);
	}
	m->key[i] = k;
	m->klen[i] = n;
	m->val[i] = v;
	m->used++;
}

static int32_t smap_get(const struct smap *m, const char *k, uint64_t n)
{
	size_t i = fnv(k, n) & (m->cap - 1);

	while (m->val[i] >= 0) {
		if (m->klen[i] == n && !memcmp(m->key[i], k, n))
			return m->val[i];
		i = (i + 1) & (m->cap - 1);
	}
	return -1;
}

/* ---- the tokenizer ------------------------------------------------------- */

struct tokenizer {
	char **tok;            /* n_vocab byte-encoded spellings, NUL terminated */
	uint32_t *toklen;
	int32_t *type;
	uint32_t n_vocab;

	struct smap vocab;     /* spelling -> id */
	struct smap merges;    /* "left right" -> rank */
	char *merge_store;     /* the merge keys, one after another */

	int32_t *special;      /* ids with a literal spelling to match */
	uint32_t n_special;

	int32_t bos, eos, eot;
	int add_bos;

	/*
	 * ⚠ TWO FAMILIES, NOT ONE. A gguf that says tokenizer.ggml.model = gpt2
	 * carries BPE merges; one that says llama carries SentencePiece pieces
	 * with a SCORE each and no merges at all, and the two are segmented by
	 * different algorithms. Phi-3.5 and Gemma are the second kind, and this
	 * used to stop at "the file has no merge table".
	 */
	float *score;          /* n_vocab, higher is better; SPM only */
	int spm;
	int32_t unk;
};

static const uint8_t *arr_next_str(const uint8_t *p, uint64_t *len)
{
	uint64_t n;

	memcpy(&n, p, 8);
	*len = n;
	return p + 8;
}

struct tokenizer *tokenizer_from_gguf(const struct gguf *g)
{
	const struct gguf_kv *toks = gguf_find(g, "tokenizer.ggml.tokens");
	const struct gguf_kv *types = gguf_find(g, "tokenizer.ggml.token_type");
	const struct gguf_kv *merges = gguf_find(g, "tokenizer.ggml.merges");
	struct tokenizer *tk;
	const uint8_t *p;
	uint32_t v;

	build_byte_table();

	if (!toks || toks->type != GGUF_V_ARRAY || toks->arr_type != GGUF_V_STRING) {
		fprintf(stderr, "tokenizer: the file has no token list\n");
		return NULL;
	}

	tk = calloc(1, sizeof(*tk));
	if (!tk)
		return NULL;
	tk->n_vocab = (uint32_t)toks->arr_len;
	tk->tok = calloc(tk->n_vocab, sizeof(*tk->tok));
	tk->toklen = calloc(tk->n_vocab, sizeof(*tk->toklen));
	tk->type = calloc(tk->n_vocab, sizeof(*tk->type));
	if (!tk->tok || !tk->toklen || !tk->type)
		goto fail;

	p = toks->arr;
	for (uint32_t i = 0; i < tk->n_vocab; i++) {
		uint64_t n;

		p = arr_next_str(p, &n);
		tk->tok[i] = malloc(n + 1);
		if (!tk->tok[i])
			goto fail;
		memcpy(tk->tok[i], p, n);
		tk->tok[i][n] = 0;
		tk->toklen[i] = (uint32_t)n;
		p += n;
	}

	if (types && types->type == GGUF_V_ARRAY && types->arr_len == toks->arr_len)
		for (uint32_t i = 0; i < tk->n_vocab; i++)
			memcpy(&tk->type[i], types->arr + i * 4, 4);

	if (smap_init(&tk->vocab, tk->n_vocab) < 0)
		goto fail;
	for (uint32_t i = 0; i < tk->n_vocab; i++)
		smap_put(&tk->vocab, tk->tok[i], tk->toklen[i], (int32_t)i);

	/* the special list: anything the file marks CONTROL or USER_DEFINED */
	tk->special = calloc(tk->n_vocab ? tk->n_vocab : 1, sizeof(*tk->special));
	if (!tk->special)
		goto fail;
	for (uint32_t i = 0; i < tk->n_vocab; i++)
		if (tk->type[i] == 3 || tk->type[i] == 4)
			tk->special[tk->n_special++] = (int32_t)i;

	if (merges && merges->type == GGUF_V_ARRAY && merges->arr_type == GGUF_V_STRING) {
		uint64_t total = 0;
		const uint8_t *q = merges->arr;
		char *w;

		for (uint64_t i = 0; i < merges->arr_len; i++) {
			uint64_t n;

			q = arr_next_str(q, &n);
			total += n + 1;
			q += n;
		}
		tk->merge_store = malloc(total ? total : 1);
		if (!tk->merge_store || smap_init(&tk->merges, merges->arr_len) < 0)
			goto fail;

		q = merges->arr;
		w = tk->merge_store;
		for (uint64_t i = 0; i < merges->arr_len; i++) {
			uint64_t n;

			q = arr_next_str(q, &n);
			memcpy(w, q, n);
			w[n] = 0;
			smap_put(&tk->merges, w, n, (int32_t)i);
			w += n + 1;
			q += n;
		}
	} else {
		/*
		 * ⚠ NO MERGES IS NOT NECESSARILY A BROKEN FILE. SentencePiece
		 * carries a score per piece instead, and the segmentation is a
		 * search over those rather than a sequence of merges. Take that
		 * path when the scores are there, and only give up when neither
		 * is.
		 */
		const struct gguf_kv *sc = gguf_find(g, "tokenizer.ggml.scores");

		if (!sc || sc->type != GGUF_V_ARRAY || sc->arr_len != toks->arr_len) {
			fprintf(stderr, "tokenizer: the file has neither a merge "
				"table nor piece scores\n");
			goto fail;
		}
		tk->score = malloc((size_t)tk->n_vocab * sizeof(float));
		if (!tk->score)
			goto fail;
		for (uint64_t i = 0; i < tk->n_vocab; i++)
			memcpy(&tk->score[i], sc->arr + i * 4, 4);
		tk->spm = 1;
	}

	tk->bos = tk->eos = tk->eot = -1;
	tk->unk = -1;
	if (!gguf_get_u32(g, "tokenizer.ggml.unknown_token_id", &v))
		tk->unk = (int32_t)v;
	if (!gguf_get_u32(g, "tokenizer.ggml.bos_token_id", &v))
		tk->bos = (int32_t)v;
	if (!gguf_get_u32(g, "tokenizer.ggml.eos_token_id", &v))
		tk->eos = (int32_t)v;
	if (!gguf_get_u32(g, "tokenizer.ggml.eot_token_id", &v))
		tk->eot = (int32_t)v;
	tk->add_bos = 1;
	if (!gguf_get_u32(g, "tokenizer.ggml.add_bos_token", &v))
		tk->add_bos = (int)v;

	/*
	 * Llama 3 ends a chat turn on <|eot_id|> and the file does not always
	 * name it. Look it up by spelling so a generate loop can stop.
	 */
	if (tk->eot < 0)
		tk->eot = smap_get(&tk->vocab, "<|eot_id|>", 10);

	return tk;

fail:
	tokenizer_free(tk);
	return NULL;
}

void tokenizer_free(struct tokenizer *tk)
{
	if (!tk)
		return;
	if (tk->tok) {
		for (uint32_t i = 0; i < tk->n_vocab; i++)
			free(tk->tok[i]);
		free(tk->tok);
	}
	free(tk->toklen);
	free(tk->type);
	free(tk->score);
	free(tk->special);
	free(tk->merge_store);
	smap_free(&tk->vocab);
	smap_free(&tk->merges);
	free(tk);
}

/*
 * A CONTROL token is one the model steers with and nobody reads: Llama 3's
 * <|eot_id|>, <|start_header_id|> and the reserved specials are all type 3.
 *
 * It matters because tokenizer_decode hands back a control token's literal
 * SPELLING, and the generation loop used to write whatever came back straight
 * to stdout. Only end-of-generation was ever intercepted, so a sampled
 * <|start_header_id|> was printed as those nineteen characters -- which is
 * exactly what round 372's 384 token arm shows, the model closing a turn it
 * was never given and the tags landing in the text.
 *
 * ⚠ TYPE 3 ONLY, not 4. USER_DEFINED tokens are added vocabulary that is
 * usually meant to be seen; the encoder's special list takes both because it
 * is matching literal spellings in a prompt, which is a different question.
 */
/*
 * The id of a token by its exact spelling, or -1. Used to find the two header
 * markers, which cannot be hard coded: they are Llama 3's names and another
 * family spells its roles differently.
 */
int32_t tokenizer_find(const struct tokenizer *tk, const char *spelling)
{
	return smap_get(&tk->vocab, spelling, strlen(spelling));
}

enum chat_fmt chat_format_of(const struct tokenizer *tk)
{
	if (tokenizer_find(tk, "<|im_start|>") >= 0) return CHAT_CHATML;
	/* phi3: <|user|> ... <|end|> and no header markers at all */
	if (tokenizer_find(tk, "<|end|>") >= 0 &&
	    tokenizer_find(tk, "<|assistant|>") >= 0) return CHAT_PHI3;
	return CHAT_LLAMA3;
}

size_t chat_turn(char *out, size_t max, enum chat_fmt f,
		 const char *role, const char *text)
{
	/*
	 * ⚠ AN EMPTY TURN IS NO TURN. phi3's own template guards its system
	 * message with `if role == 'system' and message['content']`, and this
	 * wrote the markers with nothing between them.
	 */
	if (!text || !*text) {
		if (max) out[0] = 0;
		return 0;
	}

	if (f == CHAT_PHI3)
		return (size_t)snprintf(out, max, "<|%s|>\n%s<|end|>\n", role, text);
	if (f == CHAT_CHATML)
		return (size_t)snprintf(out, max, "<|im_start|>%s\n%s<|im_end|>\n",
					role, text);
	return (size_t)snprintf(out, max,
		"<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>", role, text);
}

size_t chat_open(char *out, size_t max, enum chat_fmt f, const char *role)
{
	if (f == CHAT_PHI3)
		return (size_t)snprintf(out, max, "<|%s|>\n", role);
	if (f == CHAT_CHATML)
		return (size_t)snprintf(out, max, "<|im_start|>%s\n", role);
	return (size_t)snprintf(out, max,
		"<|start_header_id|>%s<|end_header_id|>\n\n", role);
}

size_t chat_close(char *out, size_t max, enum chat_fmt f)
{
	return (size_t)snprintf(out, max,
				f == CHAT_PHI3   ? "<|end|>\n" :
				f == CHAT_CHATML ? "<|im_end|>\n" : "<|eot_id|>");
}

int tokenizer_is_control(const struct tokenizer *tk, int32_t id)
{
	return id >= 0 && (uint32_t)id < tk->n_vocab && tk->type[id] == 3;
}

int32_t tokenizer_bos(const struct tokenizer *tk) { return tk->bos; }
int32_t tokenizer_eos(const struct tokenizer *tk) { return tk->eos; }
uint32_t tokenizer_n_vocab(const struct tokenizer *tk) { return tk->n_vocab; }

int tokenizer_is_eog(const struct tokenizer *tk, int32_t id)
{
	return id == tk->eos || (tk->eot >= 0 && id == tk->eot);
}

/* ---- the pre-tokenizer --------------------------------------------------- */

/*
 * The alternation, in the order the regex writes it. The first branch that
 * matches at `i` wins, which is what a PCRE alternation does.
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)
 * | [^\r\n\p{L}\p{N}]?\p{L}+
 * | \p{N}{1,3}
 * |  ?[^\s\p{L}\p{N}]+[\r\n]*
 * | \s*[\r\n]+
 * | \s+(?!\S)
 * | \s+
 */
static size_t pretok_next(const uint32_t *cp, size_t n, size_t i)
{
	size_t j;

	/* 1. contractions */
	if (cp[i] == '\'' && i + 1 < n) {
		uint32_t a = cp[i + 1] | 32;
		uint32_t b = i + 2 < n ? (cp[i + 2] | 32) : 0;

		if (a == 's' || a == 't' || a == 'm' || a == 'd')
			return i + 2;
		if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
		    (a == 'l' && b == 'l'))
			return i + 3;
	}

	/* 2. an optional lead-in character, then a run of letters */
	{
		size_t s = i;

		if (!cp_is_letter(cp[s]) && cp[s] != '\r' && cp[s] != '\n' &&
		    !cp_is_digit(cp[s]))
			s++;
		if (s < n && cp_is_letter(cp[s])) {
			j = s;
			while (j < n && cp_is_letter(cp[j]))
				j++;
			return j;
		}
	}

	/* 3. one to three digits */
	if (cp_is_digit(cp[i])) {
		j = i;
		while (j < n && j < i + 3 && cp_is_digit(cp[j]))
			j++;
		return j;
	}

	/* 4. an optional space, then punctuation, then any trailing newlines */
	{
		size_t s = i;

		if (cp[s] == ' ')
			s++;
		if (s < n && !cp_is_space(cp[s]) && !cp_is_letter(cp[s]) &&
		    !cp_is_digit(cp[s])) {
			j = s;
			while (j < n && !cp_is_space(cp[j]) && !cp_is_letter(cp[j]) &&
			       !cp_is_digit(cp[j]))
				j++;
			while (j < n && (cp[j] == '\r' || cp[j] == '\n'))
				j++;
			return j;
		}
	}

	/*
	 * 5. \s*[\r\n]+ -- whitespace ending in a newline.
	 *
	 * \s* is greedy over ALL whitespace and then gives back only what
	 * [\r\n]+ needs, so the match runs to the LAST newline in the run and
	 * not to the first one. "\n \n" is one token, not two: reading \s* as
	 * "the spaces before a newline" cut it in half.
	 */
	if (cp_is_space(cp[i])) {
		size_t e = i, last = (size_t)-1;

		while (e < n && cp_is_space(cp[e])) {
			if (cp[e] == '\r' || cp[e] == '\n')
				last = e;
			e++;
		}
		if (last != (size_t)-1)
			return last + 1;
	}

	/* 6 and 7. a whitespace run, giving the last space back to the next word */
	if (cp_is_space(cp[i])) {
		j = i;
		while (j < n && cp_is_space(cp[j]))
			j++;
		if (j < n && j - i > 1)
			return j - 1;      /* the (?!\S) */
		return j;
	}

	return i + 1;   /* nothing above matched; never loop */
}

/* ---- the merge loop ------------------------------------------------------ */

struct sym {
	const char *text;
	size_t n;
	int prev, next;
};

struct cand {
	int left, right;
	int32_t rank;
	size_t size;
};

struct heap {
	struct cand *a;
	int n, cap;
};

static int cand_better(const struct cand *x, const struct cand *y)
{
	if (x->rank != y->rank)
		return x->rank < y->rank;
	return x->left < y->left;
}

static void heap_push(struct heap *h, struct cand c)
{
	int i;

	if (h->n == h->cap) {
		h->cap = h->cap ? h->cap * 2 : 64;
		h->a = realloc(h->a, (size_t)h->cap * sizeof(*h->a));
	}
	h->a[h->n] = c;
	i = h->n++;
	while (i > 0) {
		int p = (i - 1) / 2;

		if (!cand_better(&h->a[i], &h->a[p]))
			break;
		c = h->a[i]; h->a[i] = h->a[p]; h->a[p] = c;
		i = p;
	}
}

static int heap_pop(struct heap *h, struct cand *out)
{
	int i = 0;

	if (!h->n)
		return 0;
	*out = h->a[0];
	h->a[0] = h->a[--h->n];
	for (;;) {
		int l = 2 * i + 1, r = l + 1, b = i;

		if (l < h->n && cand_better(&h->a[l], &h->a[b])) b = l;
		if (r < h->n && cand_better(&h->a[r], &h->a[b])) b = r;
		if (b == i)
			break;
		{
			struct cand t = h->a[i];

			h->a[i] = h->a[b];
			h->a[b] = t;
		}
		i = b;
	}
	return 1;
}

static void try_bigram(const struct tokenizer *tk, struct heap *h,
		       const struct sym *sy, int l, int r)
{
	char key[512];
	size_t kn;
	int32_t rank;

	if (l < 0 || r < 0)
		return;
	kn = sy[l].n + 1 + sy[r].n;
	if (kn >= sizeof(key))
		return;
	memcpy(key, sy[l].text, sy[l].n);
	key[sy[l].n] = ' ';
	memcpy(key + sy[l].n + 1, sy[r].text, sy[r].n);

	rank = smap_get(&tk->merges, key, kn);
	if (rank < 0)
		return;

	heap_push(h, (struct cand){ .left = l, .right = r, .rank = rank,
				    .size = sy[l].n + sy[r].n });
}

/* Merge one pre-tokenized word and append its ids. */
static int bpe_word(const struct tokenizer *tk, const char *w, size_t wn,
		    int32_t *out, int max, int cnt)
{
	struct sym *sy;
	struct heap h = { 0 };
	int nsym = 0;
	size_t i = 0;
	struct cand c;

	if (!wn)
		return cnt;

	/* one symbol per codepoint to start with, so the count is known up front */
	sy = malloc(wn * sizeof(*sy));
	if (!sy)
		return -1;

	while (i < wn) {
		uint32_t cp;
		int l = utf8_decode(w + i, wn - i, &cp);

		sy[nsym].text = w + i;
		sy[nsym].n = (size_t)l;
		sy[nsym].prev = nsym - 1;
		sy[nsym].next = nsym + 1;
		nsym++;
		i += (size_t)l;
	}
	sy[nsym - 1].next = -1;

	for (int j = 1; j < nsym; j++)
		try_bigram(tk, &h, sy, j - 1, j);

	while (heap_pop(&h, &c)) {
		struct sym *L = &sy[c.left];
		struct sym *R = &sy[c.right];

		if (!L->n || !R->n || L->n + R->n != c.size)
			continue;
		if (L->text + L->n != R->text)
			continue;

		L->n += R->n;
		R->n = 0;
		L->next = R->next;
		if (R->next >= 0)
			sy[R->next].prev = c.left;

		try_bigram(tk, &h, sy, L->prev, c.left);
		try_bigram(tk, &h, sy, c.left, L->next);
	}

	for (int j = 0; j >= 0; j = sy[j].next) {
		int32_t id;

		if (!sy[j].n) {
			if (sy[j].next < 0)
				break;
			continue;
		}
		id = smap_get(&tk->vocab, sy[j].text, sy[j].n);
		if (id < 0) {
			fprintf(stderr, "tokenizer: no id for a %zu byte symbol\n", sy[j].n);
			free(h.a);
			free(sy);
			return -1;
		}
		if (cnt >= max) {
			free(h.a);
			free(sy);
			return -1;
		}
		out[cnt++] = id;
		if (sy[j].next < 0)
			break;
	}

	free(h.a);
	free(sy);
	return cnt;
}

/* ---- encode -------------------------------------------------------------- */

/* The longest special token whose spelling starts at `s`, or -1. */
static int32_t special_at(const struct tokenizer *tk, const char *s, size_t n,
			  size_t *len)
{
	int32_t best = -1;
	size_t bestlen = 0;

	for (uint32_t i = 0; i < tk->n_special; i++) {
		int32_t id = tk->special[i];
		size_t l = tk->toklen[id];

		if (l && l <= n && l > bestlen && !memcmp(s, tk->tok[id], l)) {
			best = id;
			bestlen = l;
		}
	}
	*len = bestlen;
	return best;
}

/*
 * SentencePiece: the best-scoring way to cut the text into pieces.
 *
 * Unigram segmentation is a shortest-path problem, not a sequence of merges.
 * best[i] is the score of the best segmentation of the first i bytes; for each
 * end i, every piece that could finish there is tried against the best way of
 * reaching its start. One pass, and the pieces are recovered by walking the
 * back pointers.
 *
 * ⚠ SPM SPELLS A SPACE AS U+2581, and prepends one to the text. Feeding raw
 * spaces finds no piece at all, since the vocabulary contains none: every
 * word-initial piece begins with the marker.
 *
 * ⚠ AND A BYTE THAT NO PIECE COVERS IS NOT AN ERROR. The vocabulary carries
 * <0x00> through <0xff> for exactly that, so an unmatched byte becomes its own
 * token rather than <unk>, and no input is unrepresentable.
 */
static int32_t spm_byte(const struct tokenizer *tk, unsigned char b)
{
	char name[8];

	snprintf(name, sizeof(name), "<0x%02X>", b);
	return smap_get(&tk->vocab, name, strlen(name));
}

/*
 * SentencePiece, as the gguf format actually means it: a GREEDY BIGRAM MERGE,
 * not a Viterbi over the piece scores.
 *
 * ⚠ THE SCORES ARE NOT ALWAYS LOG PROBABILITIES, and that is the whole reason
 * this is not the shortest-path search it looks like it should be. Llama 2 and
 * phi3 store real unigram log probabilities, where adding them along a
 * segmentation means something. Gemma stores what is effectively MINUS A RANK:
 * measured in gemma-3-1b-it-Q4_0, "▁The" scores -175 while "▁T" and "he" score
 * -64 and -5, so the additive objective prefers the two small pieces by a wide
 * margin and the prompt comes out as " T | he | c | ap | it | al". It reads
 * like a broken vocabulary and is a correct search over the wrong objective.
 *
 * llama.cpp merges instead: seed a max-heap with every adjacent pair whose
 * concatenation is in the vocabulary, keyed on the MERGED piece's score, then
 * pop and merge until the heap is empty, pushing the two new neighbours each
 * time. Under that rule "he" merges first, then "▁T", then the pair of them
 * into "▁The" -- the low score does not lose the merge, it only delays it.
 * Ties go to the leftmost pair.
 */
struct spm_sym {
	const char *text;
	size_t n;             /* 0 once this symbol has been merged away */
	int prev, next;
};

struct spm_bigram {
	int left, right;
	float score;
	size_t size;          /* what left->n + right->n was when pushed */
};

/* higher score first; equal scores, leftmost first */
static int spm_better(const struct spm_bigram *a, const struct spm_bigram *b)
{
	if (a->score != b->score)
		return a->score > b->score;
	return a->left < b->left;
}

static void spm_heap_push(struct spm_bigram *h, int *n, struct spm_bigram v)
{
	int i = (*n)++;

	h[i] = v;
	while (i > 0) {
		int p = (i - 1) / 2;

		if (!spm_better(&h[i], &h[p]))
			break;
		v = h[p]; h[p] = h[i]; h[i] = v;
		i = p;
	}
}

static struct spm_bigram spm_heap_pop(struct spm_bigram *h, int *n)
{
	struct spm_bigram top = h[0], t;
	int i = 0;

	h[0] = h[--(*n)];
	for (;;) {
		int l = 2 * i + 1, r = l + 1, b = i;

		if (l < *n && spm_better(&h[l], &h[b]))
			b = l;
		if (r < *n && spm_better(&h[r], &h[b]))
			b = r;
		if (b == i)
			break;
		t = h[b]; h[b] = h[i]; h[i] = t;
		i = b;
	}
	return top;
}

static int encode_span_spm(const struct tokenizer *tk, const char *s, size_t n,
			   int32_t *out, int max, int cnt, int at_start)
{
	static const char MARK[3] = { (char)0xe2, (char)0x96, (char)0x81 };
	char *buf;
	size_t bn = 0, i;
	struct spm_sym *sym = NULL;
	struct spm_bigram *heap = NULL;
	int nsym = 0, nheap = 0, heapcap, rc = cnt;

	if (!n)
		return cnt;
	/* the text with every space, and one prepended, written as U+2581 */
	buf = malloc(n * 3 + 3);
	if (!buf)
		return -1;
	/*
	 * ⚠ THE PREPENDED MARKER IS FOR THE START OF THE TEXT, NOT EVERY SPAN.
	 * Prepending it to each run between special tokens put a space after
	 * every one of them: the prompt echoed as "<|system|> \nYou are..."
	 * and the model, handed a template it had never seen, answered nothing
	 * at all.
	 */
	if (at_start) { memcpy(buf + bn, MARK, 3); bn += 3; }
	for (i = 0; i < n; i++) {
		if (s[i] == ' ') { memcpy(buf + bn, MARK, 3); bn += 3; }
		else buf[bn++] = s[i];
	}

	/* one symbol per utf-8 character, in a doubly linked list */
	sym = malloc((bn + 1) * sizeof(*sym));
	if (!sym) {
		free(buf);
		return -1;
	}
	for (i = 0; i < bn; ) {
		unsigned char c = (unsigned char)buf[i];
		size_t len = c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;

		if (i + len > bn)
			len = bn - i;
		sym[nsym].text = buf + i;
		sym[nsym].n = len;
		sym[nsym].prev = nsym - 1;
		sym[nsym].next = (i + len == bn) ? -1 : nsym + 1;
		nsym++;
		i += len;
	}

	/*
	 * Every merge pops one and pushes at most two, and there are fewer
	 * than nsym merges, so 3 * nsym is a bound rather than a guess.
	 */
	heapcap = 3 * nsym + 8;
	heap = malloc((size_t)heapcap * sizeof(*heap));
	if (!heap) {
		free(buf); free(sym);
		return -1;
	}

	/*
	 * ⚠ THE ARGUMENTS ARE COPIED FIRST, and the local is not called bg.
	 * The obvious spelling of this macro declares a `struct spm_bigram bg`
	 * and is then called as TRY_BIGRAM(bg.left, ...) from a loop that has
	 * its own bg: the inner one shadows it, `bg_.left = (L)` becomes an
	 * uninitialised self-assignment, and the merge is applied to two
	 * symbols picked at random. It shows up as a chain with holes in it --
	 * "The capital of France is" tokenising to four pieces -- rather than
	 * as a crash.
	 */
#define TRY_BIGRAM(L, R) do {                                                 \
		int l_ = (L), r_ = (R);                                       \
									      \
		if (l_ >= 0 && r_ >= 0 && nheap < heapcap) {                  \
			size_t sz = sym[l_].n + sym[r_].n;                    \
			int32_t id = smap_get(&tk->vocab, sym[l_].text, sz);  \
									      \
			if (id >= 0) {                                        \
				struct spm_bigram bg_;                        \
									      \
				bg_.left = l_; bg_.right = r_;                \
				bg_.score = tk->score[id];                    \
				bg_.size = sz;                                \
				spm_heap_push(heap, &nheap, bg_);             \
			}                                                     \
		}                                                             \
	} while (0)

	for (int k = 1; k < nsym; k++)
		TRY_BIGRAM(k - 1, k);

	while (nheap > 0) {
		struct spm_bigram bg = spm_heap_pop(heap, &nheap);
		struct spm_sym *l = &sym[bg.left], *r = &sym[bg.right];

		/* one of the two has already been merged into something else */
		if (!l->n || !r->n || l->n + r->n != bg.size)
			continue;

		l->n += r->n;
		r->n = 0;
		l->next = r->next;
		if (r->next >= 0)
			sym[r->next].prev = bg.left;

		TRY_BIGRAM(l->prev, bg.left);
		TRY_BIGRAM(bg.left, l->next);
	}
#undef TRY_BIGRAM

	if (getenv("CHARSIU_SPM_DEBUG")) {
		fprintf(stderr, "spm: nsym=%d bn=%zu chain:", nsym, bn);
		for (int k = 0; k != -1; k = sym[k].next)
			fprintf(stderr, " [%d]'%.*s'(%zu)", k,
				(int)sym[k].n, sym[k].text, sym[k].n);
		fprintf(stderr, "\n");
	}
	for (int k = 0; k != -1; k = sym[k].next) {
		int32_t id = smap_get(&tk->vocab, sym[k].text, sym[k].n);

		if (id >= 0) {
			if (rc >= max) { rc = -1; break; }
			out[rc++] = id;
			continue;
		}
		/*
		 * A symbol the vocabulary does not carry. Only an unmerged
		 * character can get here -- every merge required a hit -- and
		 * the file spells those out a byte at a time as <0xNN>.
		 */
		for (size_t j = 0; j < sym[k].n; j++) {
			int32_t b = spm_byte(tk, (unsigned char)sym[k].text[j]);

			if (b < 0)
				continue;
			if (rc >= max) { rc = -1; break; }
			out[rc++] = b;
		}
		if (rc < 0)
			break;
	}

	free(buf); free(sym); free(heap);
	return rc;
}

/* Byte-encode a raw span, then pre-tokenize and merge it. */
static int encode_span(const struct tokenizer *tk, const char *s, size_t n,
		       int32_t *out, int max, int cnt, int at_start)
{
	uint32_t *cp;

	if (tk->spm)
		return encode_span_spm(tk, s, n, out, max, cnt, at_start);

	size_t ncp = 0, i = 0;
	char *word;
	int rc = cnt;

	if (!n)
		return cnt;

	/*
	 * Codepoints first, because the pre-tokenizer's classes are Unicode.
	 * The byte encoding happens after the cut, per word.
	 */
	cp = malloc(n * sizeof(*cp));
	if (!cp)
		return -1;
	while (i < n) {
		uint32_t c;
		int l = utf8_decode(s + i, n - i, &c);

		cp[ncp++] = c;
		i += (size_t)l;
	}

	word = malloc(n * 4 + 8);
	if (!word) {
		free(cp);
		return -1;
	}

	i = 0;
	while (i < ncp) {
		size_t j = pretok_next(cp, ncp, i);
		size_t wn = 0;

		if (j <= i)
			j = i + 1;
		for (size_t k = i; k < j; k++) {
			char raw[4];
			int rl = utf8_encode(cp[k], raw);

			for (int b = 0; b < rl; b++)
				wn += (size_t)utf8_encode(g_b2u[(uint8_t)raw[b]], word + wn);
		}
		rc = bpe_word(tk, word, wn, out, max, rc);
		if (rc < 0)
			break;
		i = j;
	}

	free(word);
	free(cp);
	return rc;
}

int tokenizer_encode(const struct tokenizer *tk, const char *text,
		     int add_bos, int32_t *out, int max)
{
	size_t n = strlen(text), i = 0, run = 0;
	int cnt = 0;

	if (add_bos < 0)
		add_bos = tk->add_bos;
	if (add_bos && tk->bos >= 0) {
		if (cnt >= max)
			return -1;
		out[cnt++] = tk->bos;
	}

	while (i < n) {
		size_t slen = 0;
		int32_t sid = special_at(tk, text + i, n - i, &slen);

		if (sid < 0) {
			run++;
			i++;
			continue;
		}
		cnt = encode_span(tk, text + i - run, run, out, max, cnt,
				  text + i - run == text);
		if (cnt < 0)
			return -1;
		if (cnt >= max)
			return -1;
		out[cnt++] = sid;
		i += slen;
		run = 0;
	}
	cnt = encode_span(tk, text + i - run, run, out, max, cnt,
			  text + i - run == text);
	return cnt;
}

static int hexval(char c)
{
	if (c >= '0' && c <= '9') return c - '0';
	if (c >= 'a' && c <= 'f') return c - 'a' + 10;
	if (c >= 'A' && c <= 'F') return c - 'A' + 10;
	return -1;
}

/* ---- decode -------------------------------------------------------------- */

const char *tokenizer_decode(const struct tokenizer *tk, int32_t id, int *len)
{
	static __thread char buf[512];
	const char *s;
	size_t n, i = 0, w = 0;

	if (id < 0 || (uint32_t)id >= tk->n_vocab) {
		*len = 0;
		return "";
	}
	s = tk->tok[id];
	n = tk->toklen[id];

	/* a control token has no byte spelling; hand it back as written */
	if (tk->type[id] == 3) {
		*len = (int)n;
		return s;
	}

	/*
	 * ⚠ SentencePiece PIECES ARE NOT BYTE-ENCODED. They are the text
	 * itself, with U+2581 standing in for a space, and a byte the
	 * vocabulary could not spell written as the four characters <0xNN>.
	 * Running them through the GPT-2 byte map would mangle every one.
	 */
	if (tk->spm) {
		if (n == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
			int hi = hexval(s[3]), lo = hexval(s[4]);

			if (hi >= 0 && lo >= 0) {
				buf[0] = (char)(hi * 16 + lo);
				*len = 1;
				return buf;
			}
		}
		while (i < n && w + 4 < sizeof(buf)) {
			if (n - i >= 3 && (unsigned char)s[i] == 0xe2 &&
			    (unsigned char)s[i + 1] == 0x96 &&
			    (unsigned char)s[i + 2] == 0x81) {
				buf[w++] = ' ';
				i += 3;
				continue;
			}
			buf[w++] = s[i++];
		}
		buf[w] = 0;
		*len = (int)w;
		return buf;
	}

	while (i < n && w + 4 < sizeof(buf)) {
		uint32_t cp;
		int l = utf8_decode(s + i, n - i, &cp);

		if (cp < 512 && g_u2b[cp] >= 0)
			buf[w++] = (char)g_u2b[cp];
		else
			w += (size_t)utf8_encode(cp, buf + w);
		i += (size_t)l;
	}
	buf[w] = 0;
	*len = (int)w;
	return buf;
}
