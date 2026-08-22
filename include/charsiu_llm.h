/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
#ifndef CHARSIU_LLM_H
#define CHARSIU_LLM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The CPU decode loop.
 *
 * This half of charsiu has NO NPU in it, on purpose. It exists to be the
 * oracle: a run of it is a known-correct sequence of tokens and a known
 * sequence of intermediate tensors, and every later version that moves a
 * matmul onto the NPU is diffed against it rather than against a guess.
 *
 * The project's own record is the argument for building it first. Every board
 * round that had an oracle was readable; every round that guessed at a form
 * got refuted by the next one.
 */

/* ---- GGUF ---------------------------------------------------------------- */

enum gguf_vtype {
	GGUF_V_U8 = 0, GGUF_V_I8, GGUF_V_U16, GGUF_V_I16, GGUF_V_U32,
	GGUF_V_I32, GGUF_V_F32, GGUF_V_BOOL, GGUF_V_STRING, GGUF_V_ARRAY,
	GGUF_V_U64, GGUF_V_I64, GGUF_V_F64,
};

/* ggml tensor types, only the ones this reads. */
enum ggml_type {
	GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3,
	GGML_Q8_0 = 8, GGML_Q6_K = 14,
};

struct gguf_kv {
	char *key;
	uint32_t type;
	union { uint64_t u; int64_t i; double f; } val;   /* scalars */
	const char *str;                                  /* GGUF_V_STRING */
	uint64_t str_len;
	uint32_t arr_type;                                /* GGUF_V_ARRAY */
	uint64_t arr_len;
	const uint8_t *arr;                               /* into the mapping */
};

struct gguf_tensor {
	char name[80];
	unsigned n_dims;
	uint64_t ne[4];
	uint32_t type;
	uint64_t offset;
	const void *data;      /* into the mapping */
	uint64_t nbytes;
};

struct gguf {
	int fd;
	const uint8_t *map;
	size_t map_size;
	uint32_t version;
	struct gguf_kv *kv;
	uint64_t n_kv;
	struct gguf_tensor *t;
	uint64_t n_tensors;
	const uint8_t *data;
	uint64_t alignment;
};

int  gguf_open(struct gguf *g, const char *path);
void gguf_close(struct gguf *g);

const struct gguf_kv *gguf_find(const struct gguf *g, const char *key);
int gguf_get_u32(const struct gguf *g, const char *key, uint32_t *out);
int gguf_get_f32(const struct gguf *g, const char *key, float *out);
int gguf_get_str(const struct gguf *g, const char *key, char *out, size_t max);
const struct gguf_tensor *gguf_tensor(const struct gguf *g, const char *name);
const char *ggml_type_name(uint32_t type);

/*
 * The activation vector, in both the forms a weight type might want it.
 *
 * `f` is always the f32 original. `q`/`d` are the same vector quantised to
 * signed int8 in blocks of 32 with a scale each, which is what lets a
 * quantised weight be multiplied WITHOUT converting every weight to a float
 * first: the whole dot product becomes integer, and one multiply at the end of
 * a block puts it back on scale.
 *
 * The quantisation happens ONCE per matvec and is shared by every row and
 * every thread, so its cost is amortised over N rows.
 *
 * This is also the shape the NPU wants -- charsiu_job carries an
 * input_scale and an input_zero_point for exactly this -- so the CPU path and
 * the NPU path can take the same operand.
 */
struct charsiu_act {
	const float *f;      /* the f32 original, always valid */
	int n;
	int nb;              /* blocks of 32, n rounded up */
	int8_t *q;           /* nb * 32 */
	float *d;            /* nb scales */
	float *bs;           /* nb block sums of f, for the types with an offset */
	int quantised;

	/*
	 * The SAME vector under ONE scale, which is the shape the NPU takes:
	 * charsiu_job carries a single input_scale and input_zero_point, not a
	 * scale per 32 weights. Filled only in NPU quantisation mode, because
	 * it is a coarser quantisation and there is no reason to pay for it on
	 * the CPU path.
	 */
	int8_t *q1;
	float d1;
	int q1_valid;
};

/* `max_n` is the widest vector this will ever hold: one allocation serves
 * every matvec in a model. */
int  charsiu_act_alloc(struct charsiu_act *a, int max_n);
void charsiu_act_free(struct charsiu_act *a);
/* Fill q/d/bs from x[0..n). Skipped when CHARSIU_NO_QACT is set: the control. */
void charsiu_act_set(struct charsiu_act *a, const float *x, int n);

/* ---- the NPU's number format, on the CPU ---------------------------------- */

/*
 * A weight tensor as the NPU would take it: signed int8 with ONE scale per
 * OUTPUT CHANNEL, which is what the coefficient buffer applies, against an
 * activation with one scale for the whole vector. Both zero points are 128 in
 * the hardware's unsigned bytes, so the arithmetic the DPU does is exactly
 *
 *     acc[n] = sum_k a_q[k] * w_q[n][k]
 *     y[n]   = acc[n] * a_scale * w_scale[n]
 *
 * This exists to answer, on the host and before any NPU plumbing, whether that
 * number format costs the model anything. If it does, no amount of correct
 * register streams will fix it.
 */
struct npu_tensor {
	int8_t *q;         /* [n][k], row major */
	float *scale;      /* n, per output channel */
	int32_t *wsum;     /* n, sum of q over k: the coefficient buffer wants it */
	uint64_t n, k;
	double rms_rel;    /* what the quantisation cost this tensor */
	float out_scale;   /* CHARSIU_NPU_OUT8>=2: calibrated, then frozen */
	double out_clip;   /* how much of the output the frozen scale clipped */
	uint64_t out_calls;
	float amax_lo, amax_hi;   /* the spread of |y| across calls: the outliers */
	char name[80];
};

int  npu_tensor_build(struct npu_tensor *t, const struct gguf_tensor *w);
void npu_tensor_free(struct npu_tensor *t);
void npu_matvec(const struct npu_tensor *t, const struct charsiu_act *a,
		float *y, uint64_t row0, uint64_t nrows);

/*
 * What the hardware does to the RESULT.
 *
 * charsiu's int8 path requantises through the coefficient buffer and writes a
 * BYTE, so a projection would leave the NPU as int8 with a scale that is baked
 * into the coefficients and cannot depend on the token. Whether a model
 * survives that is the question that decides whether the wide output the w4a16
 * path uses is needed at all, and it is answerable here rather than on a board.
 *
 *   mode 1  a per call optimal scale. The BEST case int8 output can ever be;
 *           if the text breaks here, a fixed scale cannot save it.
 *   mode 2  a scale calibrated on the first call and then frozen, which is what
 *           a coefficient buffer actually holds. Records what it clips.
 */
void npu_quantise_output(struct npu_tensor *t, float *y, uint64_t n, int mode);

/* ---- and the same thing on the hardware ---------------------------------- */

struct charsiu_npu;

struct charsiu_npu *charsiu_npu_open(unsigned max_k, unsigned max_n,
				     unsigned max_tensors);
void charsiu_npu_close(struct charsiu_npu *g);
int  charsiu_npu_add(struct charsiu_npu *g, const struct npu_tensor *t);
int  charsiu_npu_matvec(struct charsiu_npu *g, int id,
			const struct charsiu_act *a, float *y);
/* several independent projections of the same activation, one submit, one fence */
int  charsiu_npu_matvec_group(struct charsiu_npu *g, const int *ids, unsigned n,
			      const struct charsiu_act *a, float **ys);
unsigned long charsiu_npu_submits(const struct charsiu_npu *g);
void charsiu_npu_report(const struct charsiu_npu *g);
void npu_report(const struct npu_tensor *t, unsigned count);
int  npu_out8_mode(void);

/*
 * The one primitive the whole model is built out of: y[row] = dot(W[row], x).
 * W is [ne[1] rows][ne[0] columns] and x has ne[0] elements.
 *
 * This is EXACTLY the operation step 2 replaces with charsiu_emit_job(), so
 * keeping every model matmul behind this one call is deliberate.
 */
void gguf_matvec(const struct gguf_tensor *w, const struct charsiu_act *a,
		 float *y, uint64_t row0, uint64_t nrows);

/* Dequantise one whole row into f32. Used for the token embedding lookup. */
void gguf_row_f32(const struct gguf_tensor *w, uint64_t row, float *dst);

/* ---- tokenizer ----------------------------------------------------------- */

struct tokenizer;

struct tokenizer *tokenizer_from_gguf(const struct gguf *g);
void tokenizer_free(struct tokenizer *tk);

/*
 * Byte level BPE. `text` may contain literal <|...|> control tokens; when the
 * exact spelling is a control token in the vocabulary it is emitted as that one
 * id rather than being split, which is how a chat prompt is built without a
 * template engine.
 *
 * Returns the token count, or -1. `out` must hold `max` ids.
 */
int tokenizer_encode(const struct tokenizer *tk, const char *text,
		     int add_bos, int32_t *out, int max);

/* The token's bytes. Not NUL terminated in general; `len` is the truth. */
const char *tokenizer_decode(const struct tokenizer *tk, int32_t id, int *len);

int32_t tokenizer_bos(const struct tokenizer *tk);
int32_t tokenizer_eos(const struct tokenizer *tk);
int tokenizer_is_eog(const struct tokenizer *tk, int32_t id);
uint32_t tokenizer_n_vocab(const struct tokenizer *tk);

/* ---- the model ----------------------------------------------------------- */

/* what llama_state_new() uses when the caller does not say */
#define CHARSIU_DEFAULT_CTX 4096

struct llama_layer {
	const struct gguf_tensor *attn_norm;
	const struct gguf_tensor *wq, *wk, *wv, *wo;
	const struct gguf_tensor *ffn_norm;
	const struct gguf_tensor *gate, *up, *down;
};

struct llama_model {
	struct gguf gguf;
	struct tokenizer *tk;

	uint32_t n_embd, n_layer, n_head, n_head_kv, n_ff, n_vocab;
	uint32_t head_dim, n_ctx_train;
	float rms_eps, rope_base;

	const struct gguf_tensor *tok_embd;
	const struct gguf_tensor *out_norm;
	const struct gguf_tensor *output;      /* may alias tok_embd (tied) */
	const struct gguf_tensor *rope_freqs;  /* llama 3.1 style scaling, or NULL */
	struct llama_layer *layers;
};

/*
 * Everything that changes as tokens are produced. Split from the model so the
 * weights stay read only and a second state is just another allocation.
 */
struct llama_state {
	const struct llama_model *m;
	int n_ctx;
	int pos;               /* how many tokens are in the cache */

	float *kcache;         /* [n_layer][n_ctx][n_head_kv * head_dim] */
	float *vcache;

	float *x, *xb, *xb2;   /* n_embd */
	float *hb, *hb2;       /* n_ff */
	float *q;              /* n_head * head_dim */
	float *k, *v;          /* n_head_kv * head_dim */
	float *att;            /* n_head * n_ctx */
	float *logits;         /* n_vocab */

	struct charsiu_act act; /* the activation, quantised once per matvec */

	/* CHARSIU_NPU_QUANT: a second copy of each routed tensor, in the
	 * format the hardware takes. Built on first use. */
	struct npu_tensor *npu;
	const struct gguf_tensor **npu_key;
	int *npu_id;              /* >= 0 when the tensor is on the hardware */
	unsigned n_npu, npu_cap;
	struct charsiu_npu *dev;  /* CHARSIU_NPU=1 */
};

int  llama_load(struct llama_model *m, const char *path);
void llama_free(struct llama_model *m);

struct llama_state *llama_state_new(const struct llama_model *m, int n_ctx);
void llama_state_free(struct llama_state *s);

/* One token in, a full logit vector out. `pos` is where it goes in the cache. */
const float *llama_forward(struct llama_state *s, int32_t token, int pos);

/* argmax, which is the only sampler an oracle is allowed. */
int32_t llama_argmax(const float *logits, uint32_t n);

/* Temperature plus top-p, for when a human is reading the output. */
int32_t llama_sample(const float *logits, uint32_t n, float temp, float top_p,
		     uint64_t *rng);

#ifdef __cplusplus
}
#endif

#endif /* CHARSIU_LLM_H */
