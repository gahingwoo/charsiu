// SPDX-License-Identifier: GPL-2.0-or-later
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */

/*
 * The decode loop, on the CPU, in f32.
 *
 * Nothing here is fast and nothing here is meant to be. It is the reference:
 * the sequence of tokens this produces is what a version with the NPU under
 * the projections has to reproduce exactly, and the intermediate tensors are
 * what a disagreement is bisected against.
 *
 * Every matmul goes through gguf_matvec(), one call per weight tensor, so
 * there is exactly one place to change when a projection moves to the NPU.
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "charsiu_llm.h"

/* ---- a fixed pool, so a token is not 144 thread creations ---------------- */

struct pool {
	int n;
	pthread_t *th;
	pthread_mutex_t mu;
	pthread_cond_t cv_work, cv_done;
	int gen, done, stop;

	const struct gguf_tensor *w;
	const struct npu_tensor *nt;      /* set instead of w in NPU quant mode */
	const struct charsiu_act *a;
	float *y;
	uint64_t nrows;
};

static struct pool g_pool;

static void *worker(void *arg)
{
	long id = (long)arg;
	int mygen = 0;

	for (;;) {
		uint64_t per, r0, n;

		pthread_mutex_lock(&g_pool.mu);
		while (g_pool.gen == mygen && !g_pool.stop)
			pthread_cond_wait(&g_pool.cv_work, &g_pool.mu);
		if (g_pool.stop) {
			pthread_mutex_unlock(&g_pool.mu);
			return NULL;
		}
		mygen = g_pool.gen;
		pthread_mutex_unlock(&g_pool.mu);

		per = (g_pool.nrows + (uint64_t)g_pool.n - 1) / (uint64_t)g_pool.n;
		r0 = per * (uint64_t)id;
		n = r0 >= g_pool.nrows ? 0 : g_pool.nrows - r0;
		if (n > per)
			n = per;
		if (n) {
			if (g_pool.nt)
				npu_matvec(g_pool.nt, g_pool.a, g_pool.y, r0, n);
			else
				gguf_matvec(g_pool.w, g_pool.a, g_pool.y, r0, n);
		}

		pthread_mutex_lock(&g_pool.mu);
		if (++g_pool.done == g_pool.n)
			pthread_cond_signal(&g_pool.cv_done);
		pthread_mutex_unlock(&g_pool.mu);
	}
}

static void pool_start(int nthreads)
{
	const char *env = getenv("CHARSIU_THREADS");

	if (g_pool.n)
		return;
	if (nthreads < 1 && env)
		nthreads = atoi(env);
	if (nthreads < 1) {
		long c = sysconf(_SC_NPROCESSORS_ONLN);

		nthreads = c > 0 ? (int)c : 1;
	}
	g_pool.n = nthreads;
	if (nthreads == 1)
		return;

	pthread_mutex_init(&g_pool.mu, NULL);
	pthread_cond_init(&g_pool.cv_work, NULL);
	pthread_cond_init(&g_pool.cv_done, NULL);
	g_pool.th = calloc((size_t)nthreads, sizeof(*g_pool.th));
	for (long i = 0; i < nthreads; i++)
		pthread_create(&g_pool.th[i], NULL, worker, (void *)i);
}

static void matvec_again(struct llama_state *s, const struct gguf_tensor *w,
			 float *y);

/*
 * NPU quantisation mode: every routed tensor gets a second copy in the format
 * the hardware takes, built on first use. A linear lookup over at most 145
 * entries, which is nothing next to the matmul it is about to do.
 */
static unsigned npu_maxn(void)
{
	const char *e = getenv("CHARSIU_NPU_MAXN");

	return e ? (unsigned)atoi(e) : 8192;
}

static int npu_mode(void)
{
	static int m = -1;

	if (m < 0) {
		const char *e = getenv("CHARSIU_NPU_QUANT");

		m = e && *e != '0';
	}
	return m;
}

static const struct npu_tensor *npu_get(struct llama_state *s,
					const struct gguf_tensor *w)
{
	for (unsigned i = 0; i < s->n_npu; i++)
		if (s->npu_key[i] == w)
			return &s->npu[i];

	if (s->n_npu == s->npu_cap)
		return NULL;
	if (npu_tensor_build(&s->npu[s->n_npu], w) < 0)
		return NULL;
	/*
	 * And onto the hardware, if this run asked for it and this tensor is
	 * one of the ones asked for. CHARSIU_NPU_ONLY narrows it to names
	 * containing a substring, which is how a disagreement gets bisected to
	 * one projection instead of a hundred and thirteen.
	 */
	s->npu_id[s->n_npu] = -1;
	if (s->dev) {
		const char *only = getenv("CHARSIU_NPU_ONLY");

		if ((!only || strstr(w->name, only)) &&
		    w->ne[1] <= (uint64_t)npu_maxn())
			s->npu_id[s->n_npu] = charsiu_npu_add(s->dev, &s->npu[s->n_npu]);
		if (getenv("CHARSIU_NPU_VERBOSE"))
			fprintf(stderr, "  -> %s\n",
				s->npu_id[s->n_npu] >= 0 ? "on the NPU" : "on the CPU");
	}
	if (getenv("CHARSIU_NPU_VERBOSE"))
		fprintf(stderr, "npu-quant  %-28s  %llu x %llu  rms %.4f%%\n",
			w->name, (unsigned long long)w->ne[1],
			(unsigned long long)w->ne[0],
			s->npu[s->n_npu].rms_rel * 100.0);
	snprintf(s->npu[s->n_npu].name, sizeof(s->npu[s->n_npu].name), "%s", w->name);
	s->npu_key[s->n_npu] = w;
	return &s->npu[s->n_npu++];
}

/*
 * y = W x, over all of W's rows.
 *
 * The activation is quantised ONCE here, before the fan out, so its cost is
 * paid per matvec rather than per row and every thread reads the same buffer.
 */
static void matvec(struct llama_state *s, const struct gguf_tensor *w,
		   const float *x, float *y)
{
	charsiu_act_set(&s->act, x, (int)w->ne[0]);
	matvec_again(s, w, y);
}

/*
 * The same, for a weight that multiplies the activation the PREVIOUS call just
 * quantised. Q, K and V all read one RMSNorm output, and so do gate and up, so
 * six of the nine quantisations in a layer are of a vector already done.
 *
 * Only valid directly after matvec() on the same x, which is why it is a
 * separate name rather than a cache keyed on a pointer: the buffer is reused
 * between the attention and the feed forward halves, so pointer equality would
 * be wrong in exactly the place it looks right.
 */
/*
 * Several projections of ONE activation, in one submit and one fence.
 *
 * q, k and v all multiply the same RMSNorm output, and so do gate and up. Round
 * 321 measured the fence at 94% of the hardware path, so a fence removed is
 * worth more than a submit removed: 113 fences a token becomes 65.
 *
 * It falls back to one at a time whenever anything is not on the hardware, so
 * the arithmetic is the same either way and the tokens have to be identical.
 * CHARSIU_NPU_NOGROUP forces the fallback, which is the control.
 */
static int group_off(void)
{
	static int m = -1;

	if (m < 0)
		m = getenv("CHARSIU_NPU_NOGROUP") != NULL;
	return m;
}

static void matvec_pair(struct llama_state *s, const float *x,
			const struct gguf_tensor *wa, float *ya,
			const struct gguf_tensor *wb, float *yb,
			const struct gguf_tensor *wc, float *yc)
{
	const struct gguf_tensor *w[3] = { wa, wb, wc };
	float *y[3] = { ya, yb, yc };
	const struct npu_tensor *nt[3];
	int ids[3];
	unsigned n = wc ? 3 : 2, i;

	charsiu_act_set(&s->act, x, (int)wa->ne[0]);

	if (s->dev && !group_off() && npu_mode() && s->act.q1_valid) {
		for (i = 0; i < n; i++) {
			nt[i] = npu_get(s, w[i]);
			ids[i] = -1;
			if (nt[i])
				for (unsigned j = 0; j < s->n_npu; j++)
					if (&s->npu[j] == nt[i]) {
						ids[i] = s->npu_id[j];
						break;
					}
			if (ids[i] < 0)
				break;
		}
		if (i == n &&
		    !charsiu_npu_matvec_group(s->dev, ids, n, &s->act, y))
			return;
	}

	for (i = 0; i < n; i++)
		matvec_again(s, w[i], y[i]);
}

static void matvec_again(struct llama_state *s, const struct gguf_tensor *w,
			 float *y)
{
	struct charsiu_act *a = &s->act;
	const struct npu_tensor *nt = NULL;

	/* f32 and f16 stay where they are: the NPU takes int8 operands */
	if (npu_mode() && a->q1_valid && w->type != GGML_F32 && w->type != GGML_F16)
		nt = npu_get(s, w);

	/* the hardware path is one submit and does not fan out */
	if (nt) {
		int id = -1;

		for (unsigned i = 0; i < s->n_npu; i++)
			if (&s->npu[i] == nt) {
				id = s->npu_id[i];
				break;
			}
		/*
		 * A failure here FALLS BACK rather than aborting. Round 313
		 * wedged the block on the first feed forward projection and the
		 * loop kept resubmitting a job that timed out every 1.9 seconds
		 * until the board was power cycled. A run that degrades to the
		 * CPU still finishes and still reports which shape stopped
		 * answering, which is worth more than either a hang or a crash.
		 */
		if (id >= 0 && !charsiu_npu_matvec(s->dev, id, a, y)) {
			npu_quantise_output((struct npu_tensor *)nt, y, nt->n,
					    npu_out8_mode());
			return;
		}
	}

	if (g_pool.n <= 1) {
		if (nt) {
			npu_matvec(nt, a, y, 0, nt->n);
			npu_quantise_output((struct npu_tensor *)nt, y, nt->n,
					    npu_out8_mode());
		} else {
			gguf_matvec(w, a, y, 0, w->ne[1]);
		}
		return;
	}

	pthread_mutex_lock(&g_pool.mu);
	g_pool.w = w;
	g_pool.nt = nt;
	g_pool.a = a;
	g_pool.y = y;
	g_pool.nrows = w->ne[1];
	g_pool.done = 0;
	g_pool.gen++;
	pthread_cond_broadcast(&g_pool.cv_work);
	while (g_pool.done < g_pool.n)
		pthread_cond_wait(&g_pool.cv_done, &g_pool.mu);
	pthread_mutex_unlock(&g_pool.mu);

	/* after the fan in, because the scale is a property of the whole vector */
	if (nt)
		npu_quantise_output((struct npu_tensor *)nt, y, nt->n,
				    npu_out8_mode());
}

/* ---- the small pieces ---------------------------------------------------- */

static void rmsnorm(float *out, const float *x, const struct gguf_tensor *g,
		    uint32_t n, float eps)
{
	float ss = 0.0f, scale;
	float gw[1];

	(void)gw;
	for (uint32_t i = 0; i < n; i++)
		ss += x[i] * x[i];
	scale = 1.0f / sqrtf(ss / (float)n + eps);

	/*
	 * The gain is one row of a tensor, and it is f32 in every file seen so
	 * far. Read it through gguf_row_f32 anyway rather than assuming.
	 */
	{
		static float *buf;
		static uint32_t bufn;

		if (bufn < n) {
			buf = realloc(buf, n * sizeof(float));
			bufn = n;
		}
		gguf_row_f32(g, 0, buf);
		for (uint32_t i = 0; i < n; i++)
			out[i] = x[i] * scale * buf[i];
	}
}

static void softmax(float *x, int n)
{
	float mx = x[0], sum = 0.0f;

	for (int i = 1; i < n; i++)
		if (x[i] > mx)
			mx = x[i];
	for (int i = 0; i < n; i++) {
		x[i] = expf(x[i] - mx);
		sum += x[i];
	}
	for (int i = 0; i < n; i++)
		x[i] /= sum;
}

/*
 * RoPE, the interleaved form: element 2i and element 2i+1 of a head are one
 * complex number. llama.cpp's convert step permutes Q and K so this is the
 * right pairing for a gguf, even though the HF checkpoint it came from pairs
 * i with i + d/2 instead.
 */
static void rope(float *v, uint32_t nheads, uint32_t hd, int pos,
		 float base, const float *freq_factors)
{
	for (uint32_t h = 0; h < nheads; h++) {
		float *p = v + h * hd;

		for (uint32_t i = 0; i < hd / 2; i++) {
			float theta = (float)pos * powf(base, -2.0f * (float)i / (float)hd);
			float c, s, x0, x1;

			if (freq_factors)
				theta /= freq_factors[i];
			c = cosf(theta);
			s = sinf(theta);
			x0 = p[2 * i];
			x1 = p[2 * i + 1];
			p[2 * i]     = x0 * c - x1 * s;
			p[2 * i + 1] = x0 * s + x1 * c;
		}
	}
}

/* ---- load ---------------------------------------------------------------- */

static const struct gguf_tensor *need(const struct gguf *g, const char *name)
{
	const struct gguf_tensor *t = gguf_tensor(g, name);

	if (!t)
		fprintf(stderr, "llama: the file has no tensor %s\n", name);
	else if (!t->nbytes)
		fprintf(stderr, "llama: %s is %s, which this build cannot read\n",
			name, ggml_type_name(t->type));
	return t && t->nbytes ? t : NULL;
}

int llama_load(struct llama_model *m, const char *path)
{
	char arch[64] = "llama";
	char key[128];
	uint32_t v;
	float f;

	memset(m, 0, sizeof(*m));
	if (gguf_open(&m->gguf, path) < 0)
		return -1;

	gguf_get_str(&m->gguf, "general.architecture", arch, sizeof(arch));
	if (strcmp(arch, "llama")) {
		fprintf(stderr, "llama: architecture %s is not supported\n", arch);
		goto fail;
	}

#define GETU(suffix, dst, dflt) do {                                    \
		snprintf(key, sizeof(key), "%s." suffix, arch);          \
		if (gguf_get_u32(&m->gguf, key, &v))                     \
			v = (dflt);                                      \
		(dst) = v;                                               \
	} while (0)
#define GETF(suffix, dst, dflt) do {                                    \
		snprintf(key, sizeof(key), "%s." suffix, arch);          \
		if (gguf_get_f32(&m->gguf, key, &f))                     \
			f = (dflt);                                      \
		(dst) = f;                                               \
	} while (0)

	GETU("embedding_length", m->n_embd, 0);
	GETU("block_count", m->n_layer, 0);
	GETU("attention.head_count", m->n_head, 0);
	GETU("attention.head_count_kv", m->n_head_kv, m->n_head);
	GETU("feed_forward_length", m->n_ff, 0);
	GETU("context_length", m->n_ctx_train, 2048);
	GETU("rope.dimension_count", m->head_dim, m->n_head ? m->n_embd / m->n_head : 0);
	GETF("attention.layer_norm_rms_epsilon", m->rms_eps, 1e-5f);
	GETF("rope.freq_base", m->rope_base, 10000.0f);
#undef GETU
#undef GETF

	if (!m->n_embd || !m->n_layer || !m->n_head || !m->n_ff) {
		fprintf(stderr, "llama: the file is missing a shape key\n");
		goto fail;
	}
	if (m->head_dim & 1) {
		fprintf(stderr, "llama: an odd head dimension has no rope pairing\n");
		goto fail;
	}

	m->tk = tokenizer_from_gguf(&m->gguf);
	if (!m->tk)
		goto fail;
	m->n_vocab = tokenizer_n_vocab(m->tk);

	m->tok_embd = need(&m->gguf, "token_embd.weight");
	m->out_norm = need(&m->gguf, "output_norm.weight");
	if (!m->tok_embd || !m->out_norm)
		goto fail;

	/* Llama 3.2 1B ties the output head to the embedding; larger ones do not */
	m->output = gguf_tensor(&m->gguf, "output.weight");
	if (!m->output || !m->output->nbytes)
		m->output = m->tok_embd;

	m->rope_freqs = gguf_tensor(&m->gguf, "rope_freqs.weight");
	if (m->rope_freqs && !m->rope_freqs->nbytes)
		m->rope_freqs = NULL;

	m->layers = calloc(m->n_layer, sizeof(*m->layers));
	if (!m->layers)
		goto fail;
	for (uint32_t l = 0; l < m->n_layer; l++) {
		struct llama_layer *L = &m->layers[l];

#define T(field, suffix) do {                                            \
		snprintf(key, sizeof(key), "blk.%u." suffix ".weight", l); \
		L->field = need(&m->gguf, key);                          \
		if (!L->field)                                           \
			goto fail;                                       \
	} while (0)
		T(attn_norm, "attn_norm");
		T(wq, "attn_q");
		T(wk, "attn_k");
		T(wv, "attn_v");
		T(wo, "attn_output");
		T(ffn_norm, "ffn_norm");
		T(gate, "ffn_gate");
		T(up, "ffn_up");
		T(down, "ffn_down");
#undef T
	}

	return 0;

fail:
	llama_free(m);
	return -1;
}

void llama_free(struct llama_model *m)
{
	if (!m)
		return;
	free(m->layers);
	tokenizer_free(m->tk);
	gguf_close(&m->gguf);
	memset(m, 0, sizeof(*m));
}

/* ---- state --------------------------------------------------------------- */

struct llama_state *llama_state_new(const struct llama_model *m, int n_ctx)
{
	struct llama_state *s = calloc(1, sizeof(*s));
	size_t kvn;

	if (!s)
		return NULL;
	s->m = m;
	/*
	 * NOT the training context by default. Llama 3.2 trains at 131072, and
	 * a KV cache that long is 8.6 GB for a 1B model -- more than the board
	 * has. calloc gets away with it on a machine with overcommit because
	 * the pages are never touched, which is exactly the kind of thing that
	 * works until it is measured somewhere real.
	 */
	s->n_ctx = n_ctx > 0 ? n_ctx : (int)m->n_ctx_train;
	if (n_ctx <= 0 && s->n_ctx > CHARSIU_DEFAULT_CTX)
		s->n_ctx = CHARSIU_DEFAULT_CTX;
	s->pos = 0;

	kvn = (size_t)m->n_layer * (size_t)s->n_ctx * m->n_head_kv * m->head_dim;
	s->kcache = calloc(kvn, sizeof(float));
	s->vcache = calloc(kvn, sizeof(float));
	s->x   = calloc(m->n_embd, sizeof(float));
	s->xb  = calloc(m->n_embd, sizeof(float));
	s->xb2 = calloc(m->n_embd, sizeof(float));
	s->hb  = calloc(m->n_ff, sizeof(float));
	s->hb2 = calloc(m->n_ff, sizeof(float));
	s->q   = calloc((size_t)m->n_head * m->head_dim, sizeof(float));
	s->k   = calloc((size_t)m->n_head_kv * m->head_dim, sizeof(float));
	s->v   = calloc((size_t)m->n_head_kv * m->head_dim, sizeof(float));
	s->att = calloc((size_t)m->n_head * s->n_ctx, sizeof(float));
	s->logits = calloc(m->n_vocab, sizeof(float));

	{
		uint32_t widest = m->n_embd > m->n_ff ? m->n_embd : m->n_ff;

		if (charsiu_act_alloc(&s->act, (int)widest) < 0) {
			llama_state_free(s);
			return NULL;
		}
	}

	/* nine per layer plus the output head */
	s->npu_cap = m->n_layer * 9 + 2;
	s->npu = calloc(s->npu_cap, sizeof(*s->npu));
	s->npu_key = calloc(s->npu_cap, sizeof(*s->npu_key));
	s->npu_id = calloc(s->npu_cap, sizeof(*s->npu_id));
	if (!s->npu || !s->npu_key || !s->npu_id) {
		llama_state_free(s);
		return NULL;
	}

	if (getenv("CHARSIU_NPU")) {
		const char *e = getenv("CHARSIU_NPU_MAXN");
		unsigned maxn = e ? (unsigned)atoi(e) : 8192;
		unsigned widest = m->n_embd > m->n_ff ? m->n_embd : m->n_ff;

		if (maxn > m->n_vocab)
			maxn = m->n_vocab;
		s->dev = charsiu_npu_open(widest, maxn, s->npu_cap);
		if (!s->dev) {
			fprintf(stderr, "charsiu: no NPU; staying on the CPU\n");
		} else {
			fprintf(stderr, "charsiu: NPU open, routing tensors with "
				"n <= %u\n", maxn);
		}
	}

	if (!s->kcache || !s->vcache || !s->x || !s->xb || !s->xb2 || !s->hb ||
	    !s->hb2 || !s->q || !s->k || !s->v || !s->att || !s->logits) {
		llama_state_free(s);
		return NULL;
	}

	pool_start(0);
	return s;
}

void llama_state_free(struct llama_state *s)
{
	if (!s)
		return;
	free(s->kcache); free(s->vcache);
	free(s->x); free(s->xb); free(s->xb2);
	free(s->hb); free(s->hb2);
	free(s->q); free(s->k); free(s->v);
	free(s->att); free(s->logits);
	charsiu_act_free(&s->act);
	if (s->npu && s->n_npu && getenv("CHARSIU_NPU_REPORT"))
		npu_report(s->npu, s->n_npu);
	/*
	 * The calibration dump: one record a tensor, name then k then the sum
	 * of |x| over every token the run saw. Written here because this is the
	 * only place that knows the run has finished.
	 */
	if (getenv("CHARSIU_CALIB") && s->npu) {
		FILE *f = fopen(getenv("CHARSIU_CALIB"), "wb");
		unsigned wrote = 0;

		for (unsigned i = 0; f && i < s->n_npu; i++) {
			struct npu_tensor *t = &s->npu[i];

			if (!t->astat || !t->acalls)
				continue;
			for (uint64_t j = 0; j < t->k; j++)
				t->astat[j] /= (double)t->acalls;
			fwrite(t->name, 1, sizeof(t->name), f);
			fwrite(&t->k, sizeof(t->k), 1, f);
			fwrite(t->astat, sizeof(double), t->k, f);
			wrote++;
			if (t->acov) {
				char hp[256];
				FILE *hf;

				snprintf(hp, sizeof(hp), "%s.cov",
					 getenv("CHARSIU_CALIB"));
				hf = fopen(hp, "wb");
				if (hf) {
					fwrite(&t->k, sizeof(t->k), 1, hf);
					fwrite(t->acov, sizeof(double),
					       (size_t)t->k * t->k, hf);
					fclose(hf);
					fprintf(stderr, "calib: covariance of %s"
						" (%llu^2) to %s\n", t->name,
						(unsigned long long)t->k, hp);
				}
			}
		}
		if (f) {
			fclose(f);
			fprintf(stderr, "calib: wrote %u tensors to %s\n",
				wrote, getenv("CHARSIU_CALIB"));
		}
	}
	llama_stages_report();
	if (s->dev)
		charsiu_npu_report(s->dev);
	charsiu_npu_close(s->dev);
	if (s->npu) {
		for (unsigned i = 0; i < s->n_npu; i++)
			npu_tensor_free(&s->npu[i]);
		free(s->npu);
	}
	free(s->npu_key);
	free(s->npu_id);
	free(s);
}

/* ---- where the token's time goes ----------------------------------------- *
 *
 * The hardware path is measured to the microsecond by npudev and the CPU's
 * share of a token is not measured at all: it is 21 ms by SUBTRACTION, which is
 * the kind of number this project has had to withdraw five times. Every stage
 * of the forward pass is stamped here instead, so the next thing moved off the
 * CPU is the one that is actually large.
 *
 * CHARSIU_STAGES turns it on. Two clock reads a stage, about 5 us a token, and
 * off by default so it cannot flatter or slow a timing round by accident.
 */
enum {
	ST_EMBD, ST_NORM1, ST_QKV, ST_ROPE, ST_ATTN, ST_WO, ST_RES1,
	ST_NORM2, ST_GATEUP, ST_SILU, ST_DOWN, ST_RES2, ST_NORMF, ST_HEAD,
	ST_N
};

static const char *stage_name[ST_N] = {
	"token embedding", "attn rmsnorm", "q k v", "rope + kv copy",
	"attention", "o proj", "residual", "ffn rmsnorm", "gate + up",
	"silu * up", "down", "residual", "final rmsnorm", "output head",
};

static double stage_ms[ST_N];
static unsigned stage_tok;
static int stage_on = -1;

static double now_ms(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

void llama_stages_reset(void)
{
	memset(stage_ms, 0, sizeof(stage_ms));
	stage_tok = 0;
}

void llama_stages_report(void)
{
	double tot = 0;
	unsigned i;

	if (stage_on <= 0 || !stage_tok)
		return;
	for (i = 0; i < ST_N; i++)
		tot += stage_ms[i];
	printf("charsiu stages: %u tokens, %.1f ms a token\n",
	       stage_tok, tot / stage_tok);
	for (i = 0; i < ST_N; i++)
		printf("  %-16s %8.2f ms a token   %5.1f%%\n", stage_name[i],
		       stage_ms[i] / stage_tok, 100.0 * stage_ms[i] / tot);
}

/* ---- the forward pass ---------------------------------------------------- */

const float *llama_forward(struct llama_state *s, int32_t token, int pos)
{
	const struct llama_model *m = s->m;
	uint32_t hd = m->head_dim;
	uint32_t kvdim = m->n_head_kv * hd;
	uint32_t gqa = m->n_head / m->n_head_kv;
	float scale = 1.0f / sqrtf((float)hd);
	static float *freqbuf;
	const float *freqf = NULL;

	double t0, t1;

	if (pos >= s->n_ctx)
		return NULL;

	if (stage_on < 0)
		stage_on = getenv("CHARSIU_STAGES") != NULL;
#define STAGE(i) do { if (stage_on) { t1 = now_ms(); stage_ms[i] += t1 - t0; \
			      t0 = t1; } } while (0)
	t0 = t1 = stage_on ? now_ms() : 0.0;

	if (m->rope_freqs) {
		if (!freqbuf)
			freqbuf = calloc(hd, sizeof(float));
		gguf_row_f32(m->rope_freqs, 0, freqbuf);
		freqf = freqbuf;
	}

	gguf_row_f32(m->tok_embd, (uint64_t)token, s->x);
	STAGE(ST_EMBD);

	for (uint32_t l = 0; l < m->n_layer; l++) {
		const struct llama_layer *L = &m->layers[l];
		float *kc = s->kcache + ((size_t)l * s->n_ctx + pos) * kvdim;
		float *vc = s->vcache + ((size_t)l * s->n_ctx + pos) * kvdim;

		rmsnorm(s->xb, s->x, L->attn_norm, m->n_embd, m->rms_eps);
		STAGE(ST_NORM1);

		matvec_pair(s, s->xb, L->wq, s->q, L->wk, s->k, L->wv, s->v);
		STAGE(ST_QKV);

		rope(s->q, m->n_head, hd, pos, m->rope_base, freqf);
		rope(s->k, m->n_head_kv, hd, pos, m->rope_base, freqf);

		memcpy(kc, s->k, kvdim * sizeof(float));
		memcpy(vc, s->v, kvdim * sizeof(float));
		STAGE(ST_ROPE);

		for (uint32_t h = 0; h < m->n_head; h++) {
			const float *qh = s->q + h * hd;
			float *att = s->att + (size_t)h * s->n_ctx;
			uint32_t kvh = h / gqa;
			float *out = s->xb + h * hd;

			for (int t = 0; t <= pos; t++) {
				const float *kt = s->kcache +
					((size_t)l * s->n_ctx + t) * kvdim + kvh * hd;
				float a = 0.0f;

				for (uint32_t i = 0; i < hd; i++)
					a += qh[i] * kt[i];
				att[t] = a * scale;
			}
			softmax(att, pos + 1);

			memset(out, 0, hd * sizeof(float));
			for (int t = 0; t <= pos; t++) {
				const float *vt = s->vcache +
					((size_t)l * s->n_ctx + t) * kvdim + kvh * hd;
				float a = att[t];

				for (uint32_t i = 0; i < hd; i++)
					out[i] += a * vt[i];
			}
		}

		STAGE(ST_ATTN);

		matvec(s, L->wo, s->xb, s->xb2);
		STAGE(ST_WO);
		for (uint32_t i = 0; i < m->n_embd; i++)
			s->x[i] += s->xb2[i];
		STAGE(ST_RES1);

		rmsnorm(s->xb, s->x, L->ffn_norm, m->n_embd, m->rms_eps);
		STAGE(ST_NORM2);
		matvec_pair(s, s->xb, L->gate, s->hb, L->up, s->hb2, NULL, NULL);
		STAGE(ST_GATEUP);
		for (uint32_t i = 0; i < m->n_ff; i++) {
			float g = s->hb[i];

			g *= 1.0f / (1.0f + expf(-g));       /* SiLU */
			s->hb[i] = g * s->hb2[i];
		}
		STAGE(ST_SILU);
		matvec(s, L->down, s->hb, s->xb2);
		STAGE(ST_DOWN);
		for (uint32_t i = 0; i < m->n_embd; i++)
			s->x[i] += s->xb2[i];
		STAGE(ST_RES2);
	}

	rmsnorm(s->xb, s->x, m->out_norm, m->n_embd, m->rms_eps);
	STAGE(ST_NORMF);
	matvec(s, m->output, s->xb, s->logits);
	STAGE(ST_HEAD);
	if (stage_on)
		stage_tok++;
#undef STAGE

	s->pos = pos + 1;
	return s->logits;
}

/* ---- sampling ------------------------------------------------------------ */

int32_t llama_argmax(const float *logits, uint32_t n)
{
	int32_t best = 0;

	for (uint32_t i = 1; i < n; i++)
		if (logits[i] > logits[best])
			best = (int32_t)i;
	return best;
}

struct pair { float p; int32_t i; };

static int pair_desc(const void *a, const void *b)
{
	const struct pair *x = a, *y = b;

	return x->p < y->p ? 1 : x->p > y->p ? -1 : 0;
}

int32_t llama_sample(const float *logits, uint32_t n, float temp, float top_p,
		     uint64_t *rng)
{
	struct pair *p;
	float sum = 0.0f, mx = logits[0], cum = 0.0f, r;
	int32_t out;
	uint32_t keep;

	if (temp <= 0.0f)
		return llama_argmax(logits, n);

	p = malloc((size_t)n * sizeof(*p));
	if (!p)
		return llama_argmax(logits, n);

	for (uint32_t i = 1; i < n; i++)
		if (logits[i] > mx)
			mx = logits[i];
	for (uint32_t i = 0; i < n; i++) {
		p[i].p = expf((logits[i] - mx) / temp);
		p[i].i = (int32_t)i;
		sum += p[i].p;
	}
	for (uint32_t i = 0; i < n; i++)
		p[i].p /= sum;

	qsort(p, n, sizeof(*p), pair_desc);

	if (top_p <= 0.0f || top_p >= 1.0f) {
		keep = n;
	} else {
		keep = n;
		for (uint32_t i = 0; i < n; i++) {
			cum += p[i].p;
			if (cum >= top_p) {
				keep = i + 1;
				break;
			}
		}
	}

	cum = 0.0f;
	for (uint32_t i = 0; i < keep; i++)
		cum += p[i].p;

	/* xorshift64*, so a seed reproduces a run exactly */
	*rng ^= *rng >> 12; *rng ^= *rng << 25; *rng ^= *rng >> 27;
	r = (float)((*rng * 2685821657736338717ull) >> 11) / 9007199254740992.0f;
	r *= cum;

	out = p[keep - 1].i;
	cum = 0.0f;
	for (uint32_t i = 0; i < keep; i++) {
		cum += p[i].p;
		if (r < cum) {
			out = p[i].i;
			break;
		}
	}

	free(p);
	return out;
}
