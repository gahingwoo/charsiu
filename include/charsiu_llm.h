/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
#ifndef CHARSIU_LLM_H
#define CHARSIU_LLM_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
#include <arm_neon.h>
/*
 * e^x four at a time, the cephes range reduction: n = round(x/ln2), then a
 * degree six polynomial on the remainder and a shift of the exponent field.
 * About 1e-7 relative, which is under a float's own last bit for these
 * magnitudes.
 *
 * It exists for SiLU. The feed forward is 8192 wide and there are 16 of them,
 * so a token asks for 131072 exponentials, and round 367 measured that at 3.08
 * ms on the board -- 23 ns an element, more than the rmsnorms and the residuals
 * and the rope put together.
 */
static inline float32x4_t charsiu_vexpq(float32x4_t x)
{
	const float32x4_t log2e = vdupq_n_f32(1.44269504088896341f);
	const float32x4_t ln2hi = vdupq_n_f32(0.693359375f);
	const float32x4_t ln2lo = vdupq_n_f32(-2.12194440e-4f);
	float32x4_t n, r, rr, y;
	int32x4_t k;

	x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-88.0f)), vdupq_n_f32(88.0f));
	n = vrndaq_f32(vmulq_f32(x, log2e));
	r = vmlsq_f32(vmlsq_f32(x, n, ln2hi), n, ln2lo);
	rr = vmulq_f32(r, r);

	y = vdupq_n_f32(1.9875691500e-4f);
	y = vmlaq_f32(vdupq_n_f32(1.3981999507e-3f), y, r);
	y = vmlaq_f32(vdupq_n_f32(8.3334519073e-3f), y, r);
	y = vmlaq_f32(vdupq_n_f32(4.1665795894e-2f), y, r);
	y = vmlaq_f32(vdupq_n_f32(1.6666665459e-1f), y, r);
	y = vmlaq_f32(vdupq_n_f32(5.0000001201e-1f), y, r);
	y = vaddq_f32(vmlaq_f32(r, y, rr), vdupq_n_f32(1.0f));

	k = vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127));
	return vmulq_f32(y, vreinterpretq_f32_s32(vshlq_n_s32(k, 23)));
}
#endif

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
	GGML_Q8_0 = 8, GGML_Q6_K = 14, GGML_BF16 = 30,
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
	/*
	 * Whether the NPU could take this activation at all -- NPU quantisation
	 * mode is on, the buffer exists, and nothing switched it off. It used
	 * to be q1_valid that answered that, which stopped working when q1
	 * became lazy: "the hardware may have it" and "it has been computed"
	 * are different questions and only one of them gates routing.
	 */
	int npu_ok;
};

/* `max_n` is the widest vector this will ever hold: one allocation serves
 * every matvec in a model. */
int  charsiu_act_alloc(struct charsiu_act *a, int max_n);
void charsiu_act_free(struct charsiu_act *a);
/*
 * Record x[0..n). NOTHING is quantised here: the two forms below are filled on
 * demand, because a fully routed int4 run reads neither and round 376 measured
 * that at 8.09 ms a token. CHARSIU_ACT_EAGER fills both up front, which is the
 * control; CHARSIU_NO_QACT still switches the blocks off entirely.
 */
void charsiu_act_set(struct charsiu_act *a, const float *x, int n);
/*
 * ⚠ CALL THESE ON THE CALLING THREAD, BEFORE ANY FAN OUT. They fill a buffer
 * shared by every worker, so realising one from inside gguf_matvec or
 * npu_matvec -- both of which run on the pool -- is a race.
 */
void charsiu_act_blocks(struct charsiu_act *a);
void charsiu_act_q1(struct charsiu_act *a);

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
	/*
	 * ⚠ TWO LAYOUTS, AND npu_q_packed() IN src/npuquant.c DECIDES WHICH.
	 * [n][k] one signed byte a code for int8, or [n][(k+1)/2] with two int4
	 * codes a byte -- low nibble the even column, each ROW byte aligned so
	 * a reader strides by npu_q_stride() and the column alone picks the
	 * half. Row alignment is not free (one padding nibble per row at odd k,
	 * and zero for every real model, whose k is a multiple of 32) and it is
	 * what keeps quant_rows' per-row thread split from having two workers
	 * read-modify-write one shared byte.
	 */
	int8_t *q;
	float *scale;      /* n * ngroup, per output channel per k group */
	uint64_t kgroup;   /* k per scale; 0 or k means one scale a row */
	float *kscale;     /* k, a factor shared by every channel; NULL when off */
	double *astat;     /* k, the running sum of |x| over a calibration run */
	uint64_t acalls;
	double *acov;      /* k*k, upper triangle, one named tensor only */
	float *xcal;       /* nxcal * k input vectors, for offline GPTQ */
	unsigned nxcal;
	int32_t *wsum;     /* n, sum of q over k: the coefficient buffer wants it */
	uint64_t n, k;
	double rms_rel;    /* what the quantisation cost this tensor */
	float out_scale;   /* CHARSIU_NPU_OUT8>=2: calibrated, then frozen */
	double out_clip;   /* how much of the output the frozen scale clipped */
	uint64_t out_calls;
	float amax_lo, amax_hi;   /* the spread of |y| across calls: the outliers */
	char name[80];
};

/*
 * Run fn over [0, n) split across the decode's thread pool, and wait. The
 * pool lives in llama.c because that is what starts it; quantisation wants it
 * too, and starting a second one would fight the first for the same cores.
 * With one thread, or before the pool exists, fn simply runs inline.
 */
void charsiu_parallel_for(void (*fn)(void *ctx, uint64_t r0, uint64_t n),
			  void *ctx, uint64_t n);

/*
 * Start the worker pool. llama_state_new does this; a graph that is not the
 * language model has to do it itself or every parallel_for runs on one core --
 * quietly, because the single thread path is an ordinary call.
 * 0 asks CHARSIU_THREADS, then the machine.
 */
void charsiu_threads_start(int nthreads);
int charsiu_threads(void);

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

/*
 * want_w4: -1 asks CHARSIU_NPU_W4V, 0 forces int8, 1 forces int4.
 *
 * ⚠ A CALLER THAT BATCHES MUST FORCE int8. w4a16 makes exactly one row whatever
 * it is asked for, so a device opened in int4 turns a 1024 row tower into 1024
 * dispatches -- correct, and slower than the CPU it was moved off.
 */
/* Whether this device takes a batch at all: int8 does, w4a16 makes one row. */
int charsiu_npu_batches(const struct charsiu_npu *g);

struct charsiu_npu *charsiu_npu_open_mode(unsigned max_k, unsigned max_n,
					  unsigned max_tensors, int want_w4);

struct charsiu_npu *charsiu_npu_open(unsigned max_k, unsigned max_n,
				     unsigned max_tensors);
void charsiu_npu_close(struct charsiu_npu *g);
int  charsiu_npu_add(struct charsiu_npu *g, const struct npu_tensor *t);
/* int4 takes the float activation; int8 needs q1 realised first */
int  charsiu_npu_needs_q1(const struct charsiu_npu *g);
int  charsiu_npu_matvec(struct charsiu_npu *g, int id,
			const struct charsiu_act *a, float *y);
/*
 * M rows through one set of weights, which is the whole of prefill. X is m by
 * the tensor's K and Y is m by its N, both row major. Decode does not use this
 * and charsiu_npu_matvec is unchanged.
 */
int  charsiu_npu_matmul(struct charsiu_npu *g, int id, const float *X,
			unsigned m, float *Y);
/*
 * The same call, with the INPUT DECLARED UNCHANGED since the previous batched
 * call: same X contents, same m. q, k and v multiply one RMSNorm output, and so
 * do gate and up, and the pack of that input -- gather, quantise, permute, and
 * the cache clean that makes it visible to the hardware, on both cores -- is
 * 26% of a batched matmul. This skips it when X, m and K match what the input
 * BO already holds and packs as usual when they do not. The caller vouches for
 * the CONTENTS: a buffer rewritten between two calls at the same address is
 * exactly the misuse this cannot see.
 */
int  charsiu_npu_matmul_same(struct charsiu_npu *g, int id, const float *X,
			     unsigned m, float *Y);
/* how often the declaration was honoured, and how often it had to pack anyway */
void charsiu_npu_reuse_stats(const struct charsiu_npu *g, unsigned long *hits,
			     unsigned long *misses);
/* whether CHARSIU_NPU_REUSE=1 is set: matmul_same packs like matmul without it */
int  charsiu_npu_reuse_on(void);

/* what the batched calls spent, in ms: packing, submitting, the fence, reading */
void charsiu_npu_batch_split(struct charsiu_npu *g, double *pack, double *sub,
			     double *fence, double *read, int reset);
/* the fifth segment: buffers and the output zero, before any packing */
double charsiu_npu_batch_prep(struct charsiu_npu *g, int reset);
double charsiu_npu_batch_wall(struct charsiu_npu *g, int reset);
/* how much of prep is the output buffer allocation, and how often */
double charsiu_npu_batch_alloc(struct charsiu_npu *g, unsigned *n, int reset);
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
/* the widest batch a prefill chunk will use; sizes a small on-stack array */
#define CHARSIU_BATCH_MAX 16

/* M activations against one weight, reading the weight once. y is [m][nrows]
 * with stride ystride between tokens. */
void gguf_matmul(const struct gguf_tensor *w, const struct charsiu_act *a,
		 unsigned m, float *y, uint64_t ystride,
		 uint64_t row0, uint64_t nrows);
void gguf_matvec(const struct gguf_tensor *w, const struct charsiu_act *a,
		 float *y, uint64_t row0, uint64_t nrows);

/*
 * The two kernels an attention is made of, which is not a matmul against a
 * weight and so has none of the machinery above. NEON where there is NEON.
 * ⚠ The summation order is not the scalar loop's.
 */
float charsiu_dot_f32(const float *a, const float *b, uint64_t n);
void charsiu_axpy_f32(float *y, const float *x, float a, uint64_t n);

/*
 * x[i] <- e^(x[i] - m), in place, returning the sum -- the middle pass of every
 * softmax in this tree, done in one read of the row instead of two.
 *
 * ⚠⚠ THE EXPONENTIAL IS THE ARITHMETIC NOBODY COUNTED. A ViT softmax is n^2 of
 * them a head, and the vision tower is 1024 against 1024 over twelve heads and
 * twelve layers: 151 million for one picture, against a feed forward's 131072 a
 * token. glibc's expf measured 23 ns an element on the board.
 *
 * CHARSIU_EXACT_SOFTMAX forces glibc's correctly rounded expf back, which is
 * the control the polynomial has to be diffed against.
 */
float charsiu_expsum_f32(float *x, uint64_t n, float m);

/*
 * acc[u][0..hd) += sum over j < nk of s[u][j] * v[j][0..hd), for a block of
 * nq queries at once. Every operand takes its own row stride: the values arrive
 * as [n][n_embd] and the scores as [nq][key_tile], and the accumulator is
 * either a contiguous block or a slice of the output rows.
 *
 * ⚠ THE POINT IS THE ACCUMULATOR STAYING IN REGISTERS. One axpy per (query,
 * key) pays two vector loads and a store for every FMA; holding four
 * accumulators across the tile pays half a memory operation per FMA. Bit
 * identical: the sum over j for any one output element is in the same order.
 */
void charsiu_pv_f32(float *acc, uint64_t astride, const float *v,
		    uint64_t vstride, const float *s, uint64_t sstride,
		    unsigned nq, unsigned nk, unsigned hd);

/*
 * s[u][j] = (q[u] . k[j]) * scale, for a block of queries against a block of
 * keys. Same reason as charsiu_pv_f32 and the same shape of answer: a dot
 * product per pair reloads both operands for every FMA, and four queries
 * against two keys reuses each.
 *
 * ⚠ BIT IDENTICAL, INCLUDING THE LANES. dot_f32 sums in two vectors and
 * reduces at the end; this keeps two accumulators per pair so it reproduces
 * that exactly, which is what caps the block at 4 x 2 rather than 4 x 4.
 */
void charsiu_qk_f32(float *s, uint64_t sstride, const float *q,
		    uint64_t qstride, const float *k, uint64_t kstride,
		    unsigned nq, unsigned nk, unsigned hd, float scale);

/* CHARSIU_PLAIN_ATTN: one pair at a time, the control for both of the above */
void charsiu_attn_plain_set(int on);
void charsiu_softmax_exact_set(int on);

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

/*
 * ⚠ THE CHAT TEMPLATE IS NOT THE SAME FOR EVERY MODEL, and both tools wrote
 * the Llama 3 one for all of them. SmolLM2, which is what the setup wizard
 * downloads by default, uses ChatML, saw the Llama headers as ordinary text
 * and answered with them. The gguf carries a Jinja template in
 * tokenizer.chat_template, which is far more than this can run, but the
 * markers themselves are tokens, so ask the vocabulary which family it is.
 */
enum chat_fmt {
	CHAT_LLAMA3 = 0, CHAT_CHATML = 1, CHAT_PHI3 = 2, CHAT_GEMMA = 3,
	/*
	 * ⚠ gemma4 IS NOT gemma3's FORMAT. It opens a turn with <|turn> and
	 * closes it with <turn|> -- the same four characters mirrored -- where
	 * gemma3 used <start_of_turn> and <end_of_turn>, and neither of those
	 * is in gemma4's vocabulary at all. A file whose markers are not found
	 * falls through to Llama 3's headers, which is what E2B was being
	 * handed: 75 tokens of <|start_header_id|> that the model has never
	 * seen, and it still answered correctly, which is exactly how this
	 * kind of mistake survives.
	 */
	CHAT_GEMMA4 = 4,
};
enum chat_fmt chat_format_of(const struct tokenizer *tk);
/* one complete turn */
size_t chat_turn(char *out, size_t max, enum chat_fmt f,
		 const char *role, const char *text);
/* the opening of a turn with no content, for the model to continue */
size_t chat_open(char *out, size_t max, enum chat_fmt f, const char *role);
/* close the turn the model just generated, before starting the next one */
size_t chat_close(char *out, size_t max, enum chat_fmt f);

/* the id of an exact spelling, or -1 */
int32_t tokenizer_find(const struct tokenizer *tk, const char *spelling);
/* type 3, CONTROL: the model steers with it and it must not be printed */
int  tokenizer_is_control(const struct tokenizer *tk, int32_t id);
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
	/*
	 * ⚠ qwen2 IS A LLAMA WITH THREE MORE TENSORS. Measured against
	 * Qwen2.5-1.5B-Instruct-Q4_0: identical nine weights a layer, plus a
	 * bias on each of Q, K and V, f32 and one dimensional, sized exactly
	 * like the projections they follow. NULL on llama, which has none.
	 */
	const struct gguf_tensor *bq, *bk, *bv;
	/*
	 * ⚠ qwen3 DROPPED THE BIASES AND ADDED THESE. One gain of head_dim
	 * each, shared by every head, applied to q and k AFTER the projection
	 * and BEFORE rope -- the same order the bias has to keep, and for the
	 * same reason: normalising a rotated vector is not normalising it.
	 *
	 * NULL on llama, qwen2 and phi3, which have neither.
	 */
	const struct gguf_tensor *q_norm, *k_norm;
	/*
	 * ⚠ gemma NORMALISES THE BRANCH, NOT JUST THE INPUT TO IT. These sit
	 * between the projection and the residual add -- after attn_output and
	 * after ffn_down -- which is a second norm a layer that no llama has.
	 * NULL everywhere else.
	 */
	const struct gguf_tensor *attn_post_norm, *ffn_post_norm;
	/*
	 * ⚠ gemma4's PER LAYER EMBEDDING, which is where its "E2B" name comes
	 * from: the model carries more parameters than it activates, and this
	 * is the path that decides which. After the feed forward's residual a
	 * layer gates itself against a slice of a second embedding table, and
	 * the result is added back.
	 *
	 * NULL on every other architecture, including gemma3.
	 */
	const struct gguf_tensor *pl_inp_gate, *pl_proj, *pl_post_norm;
	/* one scalar the whole layer output is multiplied by, gemma4 only */
	const struct gguf_tensor *out_scale;
	/*
	 * ⚠ PER LAYER ROPE FACTORS, and only the FULL attention layers have
	 * them. gemma4 gives its window layers a plain rotation and its global
	 * ones a scaled one, which is a second axis on top of the two bases
	 * gemma3 already needed.
	 */
	const struct gguf_tensor *rope_freqs;
	/*
	 * ⚠ gemma4 GIVES EVERY LAYER ITS OWN SHAPE. feed_forward_length is an
	 * ARRAY -- 6144 for its first fifteen layers and 12288 after -- and a
	 * window layer's head is 256 where a full layer's is 512. Nothing
	 * before this had either, so n_ff and head_dim were model wide and are
	 * still the fallback: a layer that says nothing takes the model's.
	 */
	uint32_t n_ff, head_dim;
	/*
	 * ⚠ WHICH LAYER'S KV THIS ONE READS. gemma4 shares the cache: the
	 * layers past attention.shared_kv_layers have no wk or wv at all and
	 * attend against an earlier layer's. -1 means its own.
	 */
	int kv_from;
	/* 1 when this layer's attention only sees the last n_swa positions */
	int swa;
	const struct gguf_tensor *ffn_norm;
	const struct gguf_tensor *gate, *up, *down;
	/*
	 * ⚠ phi3 STACKS ITS PROJECTIONS: one attn_qkv holding q, k and v, and
	 * one ffn_up holding gate and up. A row range of a row-major tensor is
	 * contiguous, so the halves are the same bytes at an offset -- these
	 * are the descriptors the pointers above are aimed at, not copies.
	 */
	struct gguf_tensor split[5];
};

struct llama_model {
	struct gguf gguf;
	struct tokenizer *tk;

	uint32_t n_embd, n_layer, n_head, n_head_kv, n_ff, n_vocab;
	uint32_t head_dim, n_ctx_train;
	/*
	 * ⚠ n_head * head_dim, WHICH IS NOT ALWAYS n_embd. It is on llama,
	 * qwen2 and phi3, and assuming so is what sized every buffer here for
	 * three architectures. Qwen3-0.6B is 16 heads of 128 against an
	 * embedding of 1024, so attention produces 2048 floats and feeds them
	 * to attn_output -- twice what an n_embd buffer holds.
	 */
	uint32_t n_embd_attn;
	/* which of the two RoPE pairings this file's weights were saved for */
	int rope_neox;

	/*
	 * Sliding window attention, gemma3's shape of it: most layers see only
	 * the last n_swa positions and every swa_pattern'th one sees all of
	 * them. n_swa 0 means every layer is a full one, which is every
	 * architecture before this.
	 *
	 * ⚠ The two kinds of layer also ROTATE DIFFERENTLY. A window layer
	 * uses rope_base_swa (10000 in the files measured) and a full one
	 * rope_base (1000000), so a token needs two angle tables, not one.
	 */
	uint32_t n_swa, swa_pattern;
	/*
	 * ⚠ gemma4 writes sliding_window_pattern as an ARRAY, one flag a
	 * layer, where gemma3 writes a scalar period. When this is set it wins:
	 * gemma4's pattern is not periodic and a period cannot express it.
	 */
	const struct gguf_kv *swa_arr;
	/* and feed_forward_length, which gemma4 also writes one a layer */
	const struct gguf_kv *ff_arr;
	float rope_base_swa;

	/* logits -> tanh(logits / c) * c. 0 turns it off. */
	float final_softcap;
	/* the embedding is multiplied by this on the way in. 1 turns it off. */
	float embd_scale;
	/* gemma's feed forward is GELU where llama's is SiLU */
	int ffn_gelu;
	/* gemma4 RMS normalises V as well, with no gain */
	int v_norm;

	/*
	 * ⚠ THE ATTENTION SCALE IS NOT ALWAYS 1/sqrt(head_dim). gemma4 sets it
	 * to 1.0 -- its python calls that self.scaling = 1.0 -- and folds the
	 * scaling into the QK norms instead. Getting this wrong does not crash
	 * and does not look wrong; it flattens or sharpens every softmax in
	 * the model by a constant.
	 */
	float attn_scale;

	/* ---- gemma4's per layer embeddings, or 0 and NULL everywhere else --- */
	uint32_t n_embd_pl;                    /* per layer embedding width */
	const struct gguf_tensor *pl_tok_embd; /* [n_embd_pl * n_layer][vocab] */
	const struct gguf_tensor *pl_model_proj;
	const struct gguf_tensor *pl_proj_norm;
	/*
	 * ⚠ A SWA LAYER MAY HAVE A DIFFERENT HEAD LENGTH from a full one.
	 * gemma4 declares attention.key_length_swa separately, and the KV cache
	 * has to be sized for the larger of the two.
	 */
	uint32_t head_dim_swa;
	/* layers from this index up read an earlier layer's KV cache */
	uint32_t n_layer_kv;
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
/*
 * WHAT IS ON THE HARDWARE, and how a weight gets there.
 *
 * ⚠ THIS USED TO BE FIVE FIELDS INSIDE struct llama_state, which meant the only
 * graph that could reach the NPU was the language model. The vision tower, CLIP
 * and whisper are not llama and were therefore all on the CPU -- measured on the
 * board at 0.6 G-mac/s, three times, by three graphs that never touched the
 * hardware they were running on.
 *
 * ⚠ ONE STAGING PATH, NOT TWO. Everything a tensor needs to reach the NPU --
 * the requantised copy, the width refusals, CHARSIU_NPU_ONLY, the maxn gate that
 * kept an output head on the CPU for a fortnight while saying nothing -- lives
 * here once. A second copy for the towers is how the two drift.
 */
struct charsiu_npu_pool {
	struct npu_tensor *t;
	const struct gguf_tensor **key;
	/*
	 * ⚠ THE WEIGHTS THAT WERE STAGED, because the key is an address and a
	 * caller building a temporary tensor on the stack reuses one address
	 * for several different weights. See charsiu_pool_get.
	 */
	const void **src;
	int *id;                  /* >= 0 when the tensor is on the hardware */

	/*
	 * ⚠ WHAT ACTUALLY WENT TO THE HARDWARE, counted rather than assumed.
	 * A board round subtracted a staging figure from a wall clock and
	 * announced 17x; the next round dropped that staging by 19 seconds and
	 * the wall clock did not move. Neither number was wrong -- the
	 * SUBTRACTION was, because nothing was counting how much of the work
	 * reached the NPU at all.
	 */
	unsigned long calls, hw, fell_back;
	unsigned long rows_hw;
	double hw_ms;
	unsigned n, cap;
	struct charsiu_npu *dev;  /* CHARSIU_NPU=1 */
};

/*
 * `max_tensors` slots, and a device opened for weights up to max_k by max_n.
 * A pool with no device still quantises, which is the CPU path's own fast form.
 */
/*
 * Claim the weight cache for the staging that follows, or NULL to hand it back
 * to CHARSIU_NPU_CACHE. See the note in npuquant.c: it is one sequential file
 * and two graphs cannot share it.
 */
void charsiu_wcache_use(const char *path, const char *stamp);

/*
 * Stage every one of `n` tensors now, into `cache` if it is given.
 *
 * ⚠ EAGERLY, BECAUSE THE CACHE IS ORDERED. Lazy staging interleaves with
 * whatever else is running and the records come back in a different order than
 * they went in; and a board round measured 75 s of the vision tower's 82 s
 * inside the quantiser, so this is also where the time is.
 */
int charsiu_pool_stage_all(struct charsiu_npu_pool *p,
			   const struct gguf_tensor *const *w, unsigned n,
			   const char *cache, const char *stamp);

/* One line: how much of the work actually reached the hardware. */
void charsiu_pool_report(const struct charsiu_npu_pool *p, FILE *out);

/*
 * The five-way split of a batched matmul plus the share with no name.
 * Called by charsiu_pool_report; separate so a caller can ask for the
 * breakdown without the pool line.
 */
void charsiu_pool_report_batch(const struct charsiu_npu_pool *p, FILE *out);

/* $XDG_CACHE_HOME/charsiu/<name>.wq, made if it is not there. NULL on failure. */
const char *charsiu_cache_path(const char *model, char *buf, size_t max);

extern double charsiu_pool_stage_ms;

int charsiu_pool_init(struct charsiu_npu_pool *p, unsigned max_tensors,
		      unsigned max_k, unsigned max_n, int want_w4);
void charsiu_pool_fini(struct charsiu_npu_pool *p);

/* The staged form of `w`, staging it on first sight. NULL if it will not go. */
const struct npu_tensor *charsiu_pool_get(struct charsiu_npu_pool *p,
					  const struct gguf_tensor *w);

/*
 * Y[m][n] = X[m][k] * w, on the hardware. Returns 0, or -1 when this tensor is
 * not on the NPU or the batch will not go -- in which case the caller does what
 * it did before, one row at a time, which is correct and slower.
 */
int charsiu_pool_rows(struct charsiu_npu_pool *p, const struct gguf_tensor *w,
		      const float *X, unsigned m, float *Y);
/*
 * Three projections of ONE input, chunked together: q, k and v. Each chunk is
 * packed once and the other two reuse it. Returns 0 with all three Y filled,
 * or -1 having filled none of them, so the caller falls back for all three.
 */
/* the general form: up to 8 projections of one input, packed once a chunk */
int charsiu_pool_rowsn(struct charsiu_npu_pool *p,
		       const struct gguf_tensor *const *ws, unsigned nw,
		       const float *X, unsigned m, float *const *Ys);
int charsiu_pool_rows3(struct charsiu_npu_pool *p,
		       const struct gguf_tensor *w0, const struct gguf_tensor *w1,
		       const struct gguf_tensor *w2, const float *X, unsigned m,
		       float *Y0, float *Y1, float *Y2);


struct llama_state {
	const struct llama_model *m;
	int n_ctx;
	int pos;               /* how many tokens are in the cache */

	float *kcache;         /* [n_layer][n_ctx][n_head_kv * head_dim] */
	float *vcache;

	float *x, *xb, *xb2;   /* n_embd */
	/*
	 * ⚠ AN EMBEDDING THAT DID NOT COME FROM THE VOCABULARY. A picture
	 * enters the model as rows in this space rather than as token ids, so
	 * llama_forward_embd parks one here and llama_forward uses it INSTEAD
	 * of the table lookup, for exactly one call. Doing it this way rather
	 * than as a second copy of the forward pass means the image path and
	 * the text path cannot drift.
	 */
	const float *embd_in;
	/*
	 * ⚠ THE BATCHED PREFILL'S OWN ROWS, allocated only if a prompt takes
	 * that path and never touched by a decode, which is one row and uses
	 * the three above.
	 */
	float *bx, *bxb, *bhb, *bhb2, *bxo, *bcs;
	/*
	 * The batched q k v and the attention's output.
	 * ⚠ bq AND bao ARE n_head * head_dim WIDE, NOT n_embd. Qwen3 0.6B is
	 * 16 heads of 128 against an embedding of 1024 and the two are not the
	 * same number; a buffer sized by n_embd truncates every row.
	 */
	float *bq, *bk, *bv, *bao;
	float *bfreq;          /* m->rope_freqs, read once a prompt */
	/*
	 * gemma4's per layer embeddings, batched: bpl is [n][n_layer][n_embd_pl]
	 * and bplg is the gate, [n][n_embd_pl], reused a layer at a time.
	 *
	 * ⚠ ONE SET A ROW, NOT ONE A CHUNK. s->pl above is per TOKEN -- it is
	 * looked up from the token id and projected from that token's own
	 * embedding -- so a chunk of 32 rows needs 32 of them, not one.
	 */
	float *bpl, *bplg;
	/*
	 * ⚠ MEASURED AND ABANDONED: one activation a row, so gguf_matmul could
	 * read each weight row once for all m. It is four times SLOWER than n
	 * calls to llama's own matvec and it changes the text, because matvec
	 * is not gguf_matvec -- see the note in matmul_rows. Kept as fields
	 * that are never allocated would be worse than kept as a comment.
	 */
	unsigned bx_n;
	float *hb, *hb2;       /* n_ff */
	float *q;              /* n_head * head_dim */
	float *k, *v;          /* n_head_kv * head_dim */
	float *att;            /* n_head * n_ctx */
	float *logits;         /* n_vocab */
	/* gemma4's per layer embeddings: [n_layer][n_embd_pl], and scratch */
	float *pl, *plb, *plc;

	struct charsiu_act act; /* the activation, quantised once per matvec */

	/* CHARSIU_NPU_QUANT: a second copy of each routed tensor, in the
	 * format the hardware takes. Built on first use. */
	struct charsiu_npu_pool pool;
};

int  llama_load(struct llama_model *m, const char *path);
void llama_free(struct llama_model *m);

struct llama_state *llama_state_new(const struct llama_model *m, int n_ctx);
void llama_state_free(struct llama_state *s);

/*
 * Milliseconds spent turning weights into what the hardware takes, which
 * happens lazily inside the FIRST forward pass that touches each tensor.
 *
 * ⚠ THAT LANDS INSIDE THE PROMPT AND IT IS NOT PREFILL. A gemma3 board round
 * read "prompt 6 tok in 6516 ms, 0.92 tok/s" where the six tokens were 678 ms
 * of it and the rest was staging 182 tensors. Prefill is the number this
 * project has left to move, so it has to be reported without staging in it.
 */
double llama_stage_ms(void);

/*
 * What batching buys on this board: m calls to matvec against one call to
 * matmul, over the model's own staged tensors, checked for agreement first.
 */
int llama_batch_probe(struct llama_state *s, const struct llama_model *m,
		      unsigned mrows);

/*
 * A prompt in chunks of n, batching the feed forward across the rows. Returns
 * 0 having advanced s->pos and left the LAST row's logits in s->logits, or -1
 * if it will not take this model -- in which case the caller loops
 * llama_forward, which is correct for every architecture and merely slower.
 */
int llama_prefill_batch(struct llama_state *s, const struct llama_model *m,
			const int32_t *toks, int n, int pos0);

/*
 * Why llama_prefill_batch will refuse this model, as a short phrase, or NULL
 * if it will take it. A caller that falls back should SAY which of the two
 * happened: two runs at the same rate mean nothing without it.
 */
const char *llama_batch_why_not(const struct llama_model *m);

/* One token in, a full logit vector out. `pos` is where it goes in the cache. */
void llama_stages_reset(void);
void llama_stages_report(void);
const float *llama_forward(struct llama_state *s, int32_t token, int pos);

/*
 * One embedding in, a full logit vector out. The same forward pass, entered
 * past the vocabulary lookup: `embd` is n_embd floats, which is what a vision
 * projector produces.
 *
 * ⚠ NOT SCALED BY embd_scale. That factor belongs to the token embedding table
 * -- gemma multiplies its lookup by sqrt(n_embd) -- and a projector's output is
 * already in the model's own space. UNVERIFIED against a gemma vision model,
 * because the only mmproj this has been run against is llama shaped and has
 * embd_scale 1, where the two readings are indistinguishable.
 */
const float *llama_forward_embd(struct llama_state *s, const float *embd,
				int pos);

/*
 * The batched forward with the output head on EVERY row, for a speculative
 * pass: logits_all is n * n_vocab floats. Advances s->pos by n; the caller
 * rolls it back to what it accepted. -1 if the model is refused, in which case
 * nothing was touched.
 */
int llama_verify_batch(struct llama_state *s, const struct llama_model *m,
		       const int32_t *toks, int n, int pos0, float *logits_all);

/*
 * Speculative decoding: k drafted tokens verified in one batched pass. Greedy
 * only, and bit-identical to the plain greedy loop by construction -- the
 * drafts decide how many tokens one weight read yields, never which tokens.
 * The long note over llama_spec_step in llama.c has the argument.
 */
struct llama_spec {
	int k;                  /* drafts a pass, 1..5 (rows are 1 + k, even) */
	int ngram;              /* longest n-gram the lookup tries first */
	int off;                /* the batched forward refused: plain from here */
	int junk;               /* CHARSIU_SPEC_JUNK: drafts are noise (control) */
	uint32_t n_vocab;
	int32_t *hist;          /* everything seen or said, for the lookup */
	int n_hist, cap_hist;
	float *logits_all;      /* (k + 2) rows of n_vocab */
	unsigned long passes, plain, drafted, accepted, committed;
};
int  llama_spec_init(struct llama_spec *sp, const struct llama_model *m, int k,
		     int n_ctx);
void llama_spec_free(struct llama_spec *sp);
void llama_spec_push(struct llama_spec *sp, int32_t tok);   /* the prompt */
void llama_spec_reset(struct llama_spec *sp);
/*
 * One pass. `tok` is the last committed token, not yet in the cache. Fills
 * out[] with the tokens greedy decoding would produce next and returns how
 * many (at least 1), or -1 at the end of the context. The caller prints them,
 * then calls again with the LAST of them.
 */
int  llama_spec_step(struct llama_spec *sp, struct llama_state *s,
		     const struct llama_model *m, int32_t tok, int32_t *out,
		     int max_out);
void llama_spec_report(const struct llama_spec *sp, FILE *out);

/* argmax, which is the only sampler an oracle is allowed. */
/*
 * The clock of the first pinned CPU, in MHz, or 0 where sysfs has none. Read it
 * AFTER the work: at startup ondemand has not seen any and reports idle.
 */
long charsiu_cpu_mhz(void);

int32_t llama_argmax(const float *logits, uint32_t n);

/* Temperature plus top-p, for when a human is reading the output. */
int32_t llama_sample(const float *logits, uint32_t n, float temp, float top_p,
		     uint64_t *rng);

#ifdef __cplusplus
}
#endif

#endif /* CHARSIU_LLM_H */
