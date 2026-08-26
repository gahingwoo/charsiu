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
#define _GNU_SOURCE                 /* sched_setaffinity, for the big cores */
#include <sched.h>

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "charsiu.h"
#include "charsiu_llm.h"

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
static inline float32x4_t vexpq(float32x4_t x)
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

/*
 * The plain paths, for a round that wants its own before and after in one boot.
 * Read once: this is asked per feed forward, not per element.
 */
static int cpu_plain(void)
{
	static int v = -1;

	if (v < 0)
		v = getenv("CHARSIU_CPU_PLAIN") != NULL;
	return v;
}

/*
 * CHARSIU_KV_POSMAJOR puts the cache back to [layer][position][kv dim], which
 * is what it was before round 374. A layout change moves no values, so there
 * is nothing else that could tell the two apart and the round needs a control.
 */
static int kv_posmajor(void)
{
	static int v = -1;

	if (v < 0)
		v = getenv("CHARSIU_KV_POSMAJOR") != NULL;
	return v;
}

static int attn_perhead(void)
{
	static int v = -1;

	if (v < 0)
		v = getenv("CHARSIU_ATTN_PERHEAD") != NULL;
	return v;
}

static int attn_pool(void)
{
	static int v = -1;

	if (v < 0)
		v = getenv("CHARSIU_ATTN_POOL") != NULL && !cpu_plain();
	return v;
}

/*
 * ⚠ OPT IN, AND IT MOVES TOKENS.
 *
 * vexpq is accurate to about one last bit and glibc's expf is correctly
 * rounded, so the vector path is the slightly WORSE of the two. That is
 * nothing next to int4 weights -- but greedy decoding turns a near tie into a
 * different word, and on the host it did: "Claude Monet painted the water
 * lilies" became "Claude Monet was born in Paris" at token 25.
 *
 * Both continuations are fine and neither is a bug. What it costs is the
 * anchor: every round since 352 has quoted the same Louvre sentence, and a
 * sentence that changes for a reason unrelated to the hardware makes the next
 * regression harder to see, not easier. 3 ms of 100 is not worth that, so it
 * stays behind a switch until a round measures what it buys.
 */
#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
/*
 * ON BY DEFAULT SINCE ROUND 372, and the board is why.
 *
 * Vectorising the q.k dot product adds it up in a different ORDER, which is a
 * last bit, and a last bit is enough to move a token at a near tie. It did on
 * the host. It did NOT on the board, twice: round 370 ran it against the
 * unvectorised arm and the two wrote the same sentence word for word, as round
 * 368 had already found for the exponential.
 *
 * Two boards rounds agreeing is not a proof that no prompt will ever diverge --
 * nothing here can prove that, and the rounds said so before they ran. What it
 * is, is enough to stop paying 2.1 ms a token for a switch nobody turns on.
 *
 * CHARSIU_EXACT_ATTN puts the sequential sum back.
 */
static int fast_attn(void)
{
	static int v = -1;

	if (v < 0)
		v = getenv("CHARSIU_EXACT_ATTN") == NULL && !cpu_plain();
	return v;
}

/*
 * The same, for the exponential: 1.0 ms a token, one last bit against glibc's
 * correctly rounded expf, and the same sentence on the board in rounds 368 and
 * 370. CHARSIU_EXACT_SILU goes back to expf.
 */
static int fast_silu(void)
{
	static int v = -1;

	if (v < 0)
		v = getenv("CHARSIU_EXACT_SILU") == NULL && !cpu_plain();
	return v;
}
#endif

/* hb = silu(hb) * hb2, which is the gate and the up projection joined */
static void silu_mul(float *hb, const float *hb2, uint32_t n)
{
	uint32_t i = 0;

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
	if (fast_silu())
		for (; i + 4 <= n; i += 4) {
			float32x4_t g = vld1q_f32(hb + i);
			float32x4_t d = vaddq_f32(vdupq_n_f32(1.0f),
						  vexpq(vnegq_f32(g)));

			vst1q_f32(hb + i, vmulq_f32(vdivq_f32(g, d),
						    vld1q_f32(hb2 + i)));
		}
#endif
	for (; i < n; i++) {
		float g = hb[i];

		g *= 1.0f / (1.0f + expf(-g));
		hb[i] = g * hb2[i];
	}
}

/*
 * hb = gelu(hb) * hb2, which is the same join with gemma's activation.
 *
 * ⚠ THE TANH APPROXIMATION, not the exact erf one. ggml's GGML_OP_GELU is the
 * tanh form and that is what the gemma files were quantised against; the two
 * differ by about 1e-3 in the middle of the range, which is small and is not
 * nothing when it is applied 3072 times a layer.
 */
static void gelu_mul(float *hb, const float *hb2, uint32_t n)
{
	const float k = 0.7978845608028654f;   /* sqrt(2/pi) */

	for (uint32_t i = 0; i < n; i++) {
		float g = hb[i];

		g = 0.5f * g * (1.0f + tanhf(k * (g + 0.044715f * g * g * g)));
		hb[i] = g * hb2[i];
	}
}

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

	/*
	 * A MATVEC IS NOT THE ONLY THING WORTH SPLITTING FOUR WAYS.
	 *
	 * In NPU mode the pool barely runs: every projection is routed to the
	 * hardware and returns from the calling thread, so the whole CPU side of
	 * a token -- 13 ms of it in round 367 -- is one core's work while three
	 * sit idle. attention is 7.7 ms of that and splits over heads exactly,
	 * because each head writes its own slice of xb and its own row of att.
	 */
	void (*fn)(void *ctx, uint64_t r0, uint64_t n);
	void *ctx;
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
			if (g_pool.fn)
				g_pool.fn(g_pool.ctx, r0, n);
			else if (g_pool.nt)
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

/*
 * WHICH CORES. RK3576 is four Cortex-A53 at 0..3 and four Cortex-A72 at 4..7,
 * and nothing in this runtime has ever said which it wants.
 *
 * It matters more in NPU mode than it looks. Every projection is routed to the
 * hardware and returns from the CALLING thread, so the pool is nearly idle and
 * the CPU's whole 13 ms a token -- attention, the rmsnorms, the rope, the sum
 * over slices -- is one thread's work. If the scheduler has parked that thread
 * on an A53 it is running at roughly half the rate it could.
 *
 * CHARSIU_CPUS takes a list like "4-7" or "0,2,4" and applies it before the
 * pool exists, so the workers inherit it. Opt in: which cores a deployment
 * wants is its call, and a wrong guess baked in as a default would be a
 * regression nobody could see.
 */
/*
 * ⚠ SAY WHAT THE CORES WERE DOING, because a tokens-per-second number is not
 * comparable without it.
 *
 * Round 389 measured 15.91 tok/s where an earlier board run on a different
 * rootfs had 17.1 at the same model, and the difference could not be attributed
 * to anything: the governor, the clock and the token count were in none of the
 * logs. Eighty percent of this token is the NPU sitting on the DRAM roof and
 * cannot move, so the whole question is about the twelve milliseconds the CPU
 * spends -- and a core parked at half its clock is a bigger effect than
 * anything in that twelve.
 *
 * One read of sysfs at startup. If the files are not there, say nothing rather
 * than pretend.
 */
static void cpu_clock_report(const cpu_set_t *set)
{
	char path[128], gov[32] = "";
	long khz = 0;
	int first = -1;

	for (int c = 0; c < CPU_SETSIZE && first < 0; c++)
		if (CPU_ISSET((size_t)c, set))
			first = c;
	if (first < 0)
		return;

	snprintf(path, sizeof(path),
		 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", first);
	{
		FILE *f = fopen(path, "r");

		if (f) {
			if (fgets(gov, sizeof(gov), f)) {
				char *nl = strchr(gov, '\n');

				if (nl)
					*nl = 0;
			}
			fclose(f);
		}
	}
	snprintf(path, sizeof(path),
		 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", first);
	{
		FILE *f = fopen(path, "r");

		if (f) {
			if (fscanf(f, "%ld", &khz) != 1)
				khz = 0;
			fclose(f);
		}
	}
	if (gov[0] || khz)
		fprintf(stderr, "charsiu: cpu%d governor %s, %ld MHz\n",
			first, gov[0] ? gov : "(unknown)", khz / 1000);
}

static void cpus_pin(void)
{
	const char *spec = getenv("CHARSIU_CPUS");
	cpu_set_t set;
	const char *p;

	if (!spec || !*spec)
		return;
	CPU_ZERO(&set);
	for (p = spec; *p; ) {
		char *end;
		long a = strtol(p, &end, 10), b;

		if (end == p)
			break;
		p = end;
		b = a;
		if (*p == '-') {
			b = strtol(p + 1, &end, 10);
			p = end;
		}
		for (long c = a; c <= b && c < CPU_SETSIZE; c++)
			if (c >= 0)
				CPU_SET((size_t)c, &set);
		while (*p == ',' || *p == ' ')
			p++;
	}
	if (CPU_COUNT(&set) && sched_setaffinity(0, sizeof(set), &set))
		fprintf(stderr, "charsiu: CHARSIU_CPUS=%s did not apply\n", spec);
	else
		fprintf(stderr, "charsiu: pinned to CPUs %s, %d of them\n",
			spec, CPU_COUNT(&set));
	cpu_clock_report(&set);
}

static void pool_start(int nthreads)
{
	const char *env = getenv("CHARSIU_THREADS");

	if (g_pool.n)
		return;
	cpus_pin();
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

/*
 * Fan a range out over the pool and wait. The single thread case runs it
 * inline rather than paying a broadcast and a condition variable to reach
 * itself.
 */
static void pool_run(void (*fn)(void *, uint64_t, uint64_t), void *ctx,
		     uint64_t n)
{
	if (g_pool.n <= 1) {
		fn(ctx, 0, n);
		return;
	}
	pthread_mutex_lock(&g_pool.mu);
	g_pool.fn = fn;
	g_pool.ctx = ctx;
	g_pool.nrows = n;
	g_pool.done = 0;
	g_pool.gen++;
	pthread_cond_broadcast(&g_pool.cv_work);
	while (g_pool.done < g_pool.n)
		pthread_cond_wait(&g_pool.cv_done, &g_pool.mu);
	g_pool.fn = NULL;
	pthread_mutex_unlock(&g_pool.mu);
}

void charsiu_parallel_for(void (*fn)(void *ctx, uint64_t r0, uint64_t n),
			  void *ctx, uint64_t n)
{
	pool_run(fn, ctx, n);
}

/*
 * ⚠ NOT A STAGE, A SLICE THROUGH THEM. charsiu_act_set runs at the top of
 * every matvec, inside whichever row the stage table is counting, so it is
 * accumulated separately and printed as a note rather than a row.
 *
 * It exists because round 375 finally split the residue correctly: 8 ms a
 * token that npudev does not do and the hardware does not do, and CONSTANT at
 * both context lengths, so it is neither attention's nor the fence's. This is
 * the first suspect -- 231 thousand elements a token quantised into blocks of
 * 32, for an int4 path that reads the FLOAT activation and never looks at the
 * result.
 */
static double act_ms;
static int stage_on = -1;

static double now_ms(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

static void act_set_timed(struct charsiu_act *a, const float *x, int n)
{
	double t;

	if (stage_on <= 0) {
		charsiu_act_set(a, x, n);
		return;
	}
	t = now_ms();
	charsiu_act_set(a, x, n);
	act_ms += now_ms() - t;
}

/*
 * ⚠ THE REALISERS ARE TIMED TOO, and they have to be. Once the work became
 * lazy, timing only charsiu_act_set would read 0.00 whether the quantisation
 * VANISHED or merely MOVED to a fallback -- and those are the two answers this
 * number exists to tell apart. The instrument follows the work.
 */
/*
 * ⚠⚠ THE UNTIMED BRANCH CALLS THE REAL FUNCTION, NOT ITSELF. Both of these
 * recursed instead, which is one word in a wrapper whose whole body is four
 * lines, and it was fatal in one place and silent in the other.
 *
 * On the NPU path act_q1_timed is reached on every routed projection, so with
 * CHARSIU_STAGES unset -- which is every ordinary run -- the first forward
 * recursed until the stack ran out. On the board that is a bare "Segmentation
 * fault" after "pinned to CPUs", with nothing to say where.
 *
 * On the CPU path act_blocks_timed is reached for every block quantised
 * weight, and there it was worse than a crash: an infinite recursion with no
 * side effect is undefined behaviour, so an optimising build is entitled to
 * delete it, and this one did. The blocks were then never realised, gguf_matvec
 * quietly took its float fallback, and the answers stayed CORRECT while the
 * quantised path this project exists to measure never ran.
 *
 * Found with ASan, which does not tail-call away the recursion and so reported
 * the stack overflow by name. An -O2 build cannot: it turns one of these into
 * a hang and the other into nothing at all.
 */
static void act_q1_timed(struct charsiu_act *a)
{
	double t;

	if (stage_on <= 0) {
		charsiu_act_q1(a);
		return;
	}
	t = now_ms();
	charsiu_act_q1(a);
	act_ms += now_ms() - t;
}

static void act_blocks_timed(struct charsiu_act *a)
{
	double t;

	if (stage_on <= 0) {
		charsiu_act_blocks(a);
		return;
	}
	t = now_ms();
	charsiu_act_blocks(a);
	act_ms += now_ms() - t;
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
	/*
	 * ⚠ w->name IS INSIDE THE MAPPED FILE for a whole tensor and inside
	 * the layer for one of phi3's slices; both outlive a crash.
	 */
	charsiu_note(w->name, (unsigned long)w->ne[1], (unsigned long)w->ne[0]);
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
	act_set_timed(&s->act, x, (int)w->ne[0]);
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

	act_set_timed(&s->act, x, (int)wa->ne[0]);

	if (s->dev && !group_off() && npu_mode() && s->act.npu_ok) {
		/* int8 takes q1; int4 takes the float and never looks */
		if (charsiu_npu_needs_q1(s->dev))
			act_q1_timed(&s->act);
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
	if (npu_mode() && a->npu_ok && w->type != GGML_F32 && w->type != GGML_F16)
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
		if (id >= 0 && charsiu_npu_needs_q1(s->dev))
			act_q1_timed(a);
		if (id >= 0 && !charsiu_npu_matvec(s->dev, id, a, y)) {
			npu_quantise_output((struct npu_tensor *)nt, y, nt->n,
					    npu_out8_mode());
			return;
		}
	}

	/*
	 * ⚠ REALISE BEFORE THE FAN OUT, not inside the kernels: gguf_matvec and
	 * npu_matvec both run on the pool, and filling a shared buffer from
	 * four threads would be a race. Everything below this line is a
	 * FALLBACK -- the hardware path returned above -- so it is also the
	 * only place the CPU forms are needed at all.
	 */
	if (nt)
		act_q1_timed(a);
	else
		act_blocks_timed(a);

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
	g_pool.fn = NULL;
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
 * RoPE, in BOTH pairings, because a gguf can want either and the wrong one is
 * not a crash.
 *
 * The HF checkpoints pair element i of a head with element i + d/2. For llama
 * and smollm3 the convert step PERMUTES Q and K so that pairing becomes the
 * interleaved (2i, 2i+1) one, and the runtime rotates interleaved. For qwen2,
 * qwen3 and phi3 it does not permute, so the runtime has to rotate halves.
 *
 * ⚠ THE WRONG PAIRING STILL PRODUCES WORDS. Every element is still rotated by
 * an angle from the right table, just partnered with the wrong neighbour, so
 * the output stays inside the model's vocabulary and reads as English -- it
 * just loses track of position and repeats. That is what "the the capital of
 * of France" was: not a broken norm, a rotation against the wrong partner.
 */
/*
 * ONE ANGLE TABLE A TOKEN, NOT ONE A HEAD A LAYER.
 *
 * theta depends on the dimension pair and the position and on nothing else, so
 * the powf, the cosf and the sinf are the same numbers for every head of every
 * layer. This model asked for them 20480 times a token -- 40 head passes times
 * 32 pairs times 16 layers -- and needs 32. Round 367 measured the difference
 * as 1.79 ms a token, second only to attention on the CPU side.
 *
 * The values are identical, not approximated: same theta, same libm call, so
 * the tokens cannot move.
 */
static void rope_table(float *cs, uint32_t hd, int pos, float base,
		       const float *freq_factors)
{
	for (uint32_t i = 0; i < hd / 2; i++) {
		float theta = (float)pos * powf(base, -2.0f * (float)i / (float)hd);

		if (freq_factors)
			theta /= freq_factors[i];
		cs[2 * i]     = cosf(theta);
		cs[2 * i + 1] = sinf(theta);
	}
}

/*
 * y += bias. f32 and one dimensional in every qwen2 file measured, so this
 * refuses anything else rather than reading a quantised block as floats.
 */
static void add_bias(float *y, const struct gguf_tensor *b, uint32_t n)
{
	const float *v = (const float *)b->data;
	uint32_t i;

	if (b->type != 0 || b->nbytes < (uint64_t)n * sizeof(float))
		return;
	for (i = 0; i < n; i++)
		y[i] += v[i];
}

/*
 * RMS normalise EACH HEAD on its own, against one gain shared by all of them.
 *
 * This is qwen3's QK norm. It is not the layer norm applied to a longer
 * vector: the sum of squares is taken over one head's head_dim elements, so a
 * head with large values is not scaled down by its quiet neighbours. Doing it
 * over the whole q vector instead produces text that still reads fluently,
 * which is why this is written out rather than routed through rmsnorm().
 */
static void qk_norm(float *v, uint32_t nheads, uint32_t hd,
		    const struct gguf_tensor *g, float eps)
{
	static float *gain;
	static uint32_t gainn;
	static const struct gguf_tensor *cached;

	if (gainn < hd) {
		gain = realloc(gain, hd * sizeof(float));
		gainn = hd;
		cached = NULL;
	}
	if (!gain)
		return;
	/*
	 * The gain is re-read once per call rather than once per head: it is
	 * the same row for all of them, and a layer asks for two of these.
	 */
	if (cached != g) {
		gguf_row_f32(g, 0, gain);
		cached = g;
	}

	for (uint32_t h = 0; h < nheads; h++) {
		float *p = v + (size_t)h * hd;
		float ss = 0.0f, scale;

		for (uint32_t i = 0; i < hd; i++)
			ss += p[i] * p[i];
		scale = 1.0f / sqrtf(ss / (float)hd + eps);
		for (uint32_t i = 0; i < hd; i++)
			p[i] = p[i] * scale * gain[i];
	}
}

static void rope(float *v, uint32_t nheads, uint32_t hd, const float *cs,
		 int neox)
{
	for (uint32_t h = 0; h < nheads; h++) {
		float *p = v + h * hd;

		for (uint32_t i = 0; i < hd / 2; i++) {
			float c = cs[2 * i], s = cs[2 * i + 1];
			uint32_t a = neox ? i : 2 * i;
			uint32_t b = neox ? i + hd / 2 : 2 * i + 1;
			float x0 = p[a];
			float x1 = p[b];

			p[a] = x0 * c - x1 * s;
			p[b] = x0 * s + x1 * c;
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

/*
 * A row range of a row-major tensor, without copying: rows are contiguous and
 * every quantised type here has a fixed number of bytes per row, so a slice is
 * the same data pointer moved forward with fewer rows.
 */
static int subtensor(struct gguf_tensor *dst, const struct gguf_tensor *src,
		     uint64_t row0, uint64_t nrows, const char *part)
{
	uint64_t per;

	if (!src || !src->ne[1] || src->nbytes % src->ne[1])
		return -1;
	per = src->nbytes / src->ne[1];
	if (row0 + nrows > src->ne[1])
		return -1;
	*dst = *src;
	/*
	 * ⚠ A SLICE NEEDS ITS OWN NAME. The weight cache is keyed on name, n
	 * and k, and phi3's three slices of attn_qkv agree on all three: q's
	 * weights would come back for k and for v. Nothing reads it today
	 * because the cache is opt-in, which is exactly the kind of bug that
	 * waits.
	 */
	snprintf(dst->name, sizeof(dst->name), "%.*s.%s",
		 (int)(sizeof(dst->name) - strlen(part) - 2), src->name, part);
	dst->ne[1] = nrows;
	dst->data = (const uint8_t *)src->data + row0 * per;
	dst->nbytes = nrows * per;
	return 0;
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
	/*
	 * ⚠ qwen2 RUNS ON THE LLAMA GRAPH. Read out of the files rather than
	 * assumed: Qwen2.5-1.5B-Instruct declares the same nine weights a
	 * layer under the same names, ties its output head to the embedding
	 * the way Llama 3.2 1B does, and carries no softcapping or sliding
	 * window key at all. The whole difference is a bias on Q, K and V.
	 *
	 * ⚠ THE ARCH NAME IS ALSO THE KEY PREFIX, so it has to stay whatever
	 * the file said: qwen2.embedding_length, not llama.embedding_length.
	 *
	 * ⚠ qwen3 IS THE SAME GRAPH AGAIN, with the biases gone and a norm on
	 * Q and K instead. It is also the first one here whose head is not
	 * n_embd / n_head: 16 heads of 128 against an embedding of 1024.
	 *
	 * gemma2 is refused, and not over tensor names -- it declares
	 * attn_logit_softcapping 50.0, final_logit_softcapping 30.0 and a 4096
	 * sliding window, none of which this graph does.
	 */
	if (strcmp(arch, "llama") && strcmp(arch, "qwen2") &&
	    strcmp(arch, "qwen3") && strcmp(arch, "gemma3") &&
	    strcmp(arch, "phi3") && strcmp(arch, "smollm3")) {
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
	/*
	 * ⚠ READ attention.key_length RATHER THAN TRUSTING rope.dimension_count
	 * TO BE THE HEAD. They agree on every architecture here -- qwen3 rotates
	 * the whole head -- but they are different questions, and a model that
	 * rotates part of a head would size the KV cache wrong if this used the
	 * rope width for it.
	 */
	{
		uint32_t kl;

		snprintf(key, sizeof(key), "%s.attention.key_length", arch);
		if (!gguf_get_u32(&m->gguf, key, &kl) && kl && !(kl & 1))
			m->head_dim = kl;
	}
	m->n_embd_attn = m->n_head * m->head_dim;
	/*
	 * ⚠ BY ARCHITECTURE, and there is no key in the file that says it --
	 * llama.cpp carries the same thing as a switch over the architecture
	 * enum. llama and smollm3 are the permuted, interleaved ones; qwen2,
	 * qwen3 and phi3 are not.
	 */
	m->rope_neox = !strcmp(arch, "qwen2") || !strcmp(arch, "qwen3") ||
		       !strcmp(arch, "gemma3") || !strcmp(arch, "phi3");
	/*
	 * The control. The two pairings are the whole difference between
	 * "Paris. It is the largest city in France" and "a country in the
	 * world. The capital of the world", and a round that wants to see that
	 * for itself should not have to rebuild.
	 */
	{
		const char *e = getenv("CHARSIU_ROPE_NEOX");

		if (e)
			m->rope_neox = atoi(e);
	}

	/*
	 * ---- what gemma3 adds, all of it optional and absent elsewhere ----
	 *
	 * Read out of the file rather than switched on the architecture name
	 * wherever the file says it: a gemma3 with no sliding_window key is a
	 * full attention model and this makes it one.
	 */
	m->embd_scale = 1.0f;
	/*
	 * ⚠ 10000, NOT rope_base, WHEN THE FILE DOES NOT SAY. gemma-3-1b-it
	 * carries sliding_window but no rope.freq_base_swa, and llama.cpp's
	 * gemma3 path -- unlike its gemma2, olmo2 and cohere2 paths -- does
	 * NOT seed the field from the model's own base first, so it keeps the
	 * struct default of 10000. That is the documented Gemma 3 design: the
	 * local layers rotate at 10k and the global ones at 1M. Defaulting to
	 * rope_base instead is invisible over a short prompt, where theta is
	 * small either way, and diverges as the context grows.
	 */
	m->rope_base_swa = 10000.0f;
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
	GETU("attention.sliding_window", m->n_swa, 0);
	/*
	 * ⚠ THE PATTERN COUNTS FROM THE WINDOW LAYERS. llama.cpp's
	 * set_swa_pattern is `il % n < n - 1`, so with the gemma default of 6
	 * layers 0..4 slide and layer 5 sees everything -- five windows and a
	 * full one, not one window in six.
	 */
	GETU("attention.sliding_window_pattern", m->swa_pattern, 6);
	GETF("rope.freq_base_swa", m->rope_base_swa, 10000.0f);
	GETF("final_logit_softcapping", m->final_softcap, 0.0f);
#undef GETU
#undef GETF
	if (!m->n_swa)
		m->swa_pattern = 0;
	/*
	 * The controls, and they are real ones. CHARSIU_SWA=0 makes every
	 * layer a full one; a large CHARSIU_SWA_PATTERN makes every layer a
	 * window one, because the rule is `l % P < P - 1` and no layer index
	 * reaches P - 1. The second is what tells a long-context result apart
	 * from luck: with all 26 layers windowed to 512 there is no path for a
	 * fact 950 tokens back, so a run that still answers was never using
	 * the window in the first place.
	 */
	{
		const char *e = getenv("CHARSIU_SWA");

		if (e) {
			m->n_swa = (uint32_t)atoi(e);
			if (!m->n_swa)
				m->swa_pattern = 0;
			else if (!m->swa_pattern)
				m->swa_pattern = 6;
		}
		e = getenv("CHARSIU_SWA_PATTERN");
		if (e && m->n_swa)
			m->swa_pattern = (uint32_t)atoi(e);
	}
	if (!strcmp(arch, "gemma3")) {
		/*
		 * ⚠ sqrt(n_embd) ON THE EMBEDDING, and it is not a detail: at
		 * n_embd 1152 it is a factor of 34, so leaving it out feeds
		 * the first norm a vector 34 times too small.
		 *
		 * The norm GAINS need no such treatment. Gemma stores them as
		 * (w - 1) and the convert step adds the 1 back -- see
		 * conversion/gemma.py -- so what is in the gguf is already the
		 * gain this code multiplies by.
		 */
		m->embd_scale = sqrtf((float)m->n_embd);
		m->ffn_gelu = 1;
	}

	/*
	 * ⚠ SAY WHICH MODEL, because the output is the only clue otherwise. A
	 * board decoding nonsense looked like a broken runtime until it turned
	 * out to be a particular file; the name and the architecture would have
	 * said so on the first line. The basename only: the full path is in the
	 * config and this is a status line, not a report.
	 */
	{
		const char *base = strrchr(path, '/');

		fprintf(stderr, "charsiu: %s (%s, %u layers)\n",
			base ? base + 1 : path, arch, m->n_layer);
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
	/*
	 * ⚠ phi3 CALLS THEM SOMETHING ELSE, and has two sets: short factors for
	 * a context within its original 4096 and long ones beyond. rope_table
	 * already divides theta by factor[i] over hd/2, which is exactly the
	 * 48 floats phi3 stores, so the short set plugs straight in.
	 *
	 * ⚠ Only the short set. A context past the original length wants the
	 * long one and would be wrong here; charsiu's default is 2048 and this
	 * refuses to pretend otherwise.
	 */
	if (!m->rope_freqs)
		m->rope_freqs = gguf_tensor(&m->gguf, "rope_factors_short.weight");
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
		if (!strcmp(arch, "phi3")) {
			const struct gguf_tensor *qkv, *fu;
			uint64_t hq = (uint64_t)m->n_head * m->head_dim;
			uint64_t hk = (uint64_t)m->n_head_kv * m->head_dim;

			snprintf(key, sizeof(key), "blk.%u.attn_qkv.weight", l);
			qkv = need(&m->gguf, key);
			snprintf(key, sizeof(key), "blk.%u.ffn_up.weight", l);
			fu = need(&m->gguf, key);
			if (!qkv || !fu)
				goto fail;
			/* q, then k, then v; gate, then up */
			if (subtensor(&L->split[0], qkv, 0, hq, "q") ||
			    subtensor(&L->split[1], qkv, hq, hk, "k") ||
			    subtensor(&L->split[2], qkv, hq + hk, hk, "v") ||
			    subtensor(&L->split[3], fu, 0, m->n_ff, "gate") ||
			    subtensor(&L->split[4], fu, m->n_ff, m->n_ff, "up")) {
				fprintf(stderr, "llama: %s will not split into "
					"q k v and gate up\n", qkv->name);
				goto fail;
			}
			L->wq = &L->split[0];
			L->wk = &L->split[1];
			L->wv = &L->split[2];
			L->gate = &L->split[3];
			L->up = &L->split[4];
		} else {
			T(wq, "attn_q");
			T(wk, "attn_k");
			T(wv, "attn_v");
		}
		T(wo, "attn_output");
		/* optional: llama has none, qwen2 has all three */
#define B(field, suffix) do {                                            \
		snprintf(key, sizeof(key), "blk.%u." suffix ".bias", l); \
		L->field = gguf_tensor(&m->gguf, key);                   \
		if (L->field && !L->field->nbytes)                       \
			L->field = NULL;                                 \
	} while (0)
		B(bq, "attn_q");
		B(bk, "attn_k");
		B(bv, "attn_v");
#undef B
		/*
		 * optional WEIGHTS, not biases: qwen3 has both of these and
		 * nothing older has either.
		 */
#define OW(field, suffix) do {                                             \
		snprintf(key, sizeof(key), "blk.%u." suffix ".weight", l); \
		L->field = gguf_tensor(&m->gguf, key);                     \
		if (L->field && !L->field->nbytes)                         \
			L->field = NULL;                                   \
	} while (0)
		OW(q_norm, "attn_q_norm");
		OW(k_norm, "attn_k_norm");
		OW(attn_post_norm, "post_attention_norm");
		OW(ffn_post_norm, "post_ffw_norm");
#undef OW
		if (!L->q_norm != !L->k_norm) {
			fprintf(stderr, "llama: layer %u norms one of q and k "
				"and not the other\n", l);
			goto fail;
		}
		T(ffn_norm, "ffn_norm");
		if (strcmp(arch, "phi3")) {
			T(gate, "ffn_gate");
			T(up, "ffn_up");
		}
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

/*
 * The widest vector any matvec in this model takes as its INPUT: n_embd for
 * the projections, n_ff for ffn_down, and n_head * head_dim for attn_output,
 * which is the one that is not always n_embd.
 */
static uint32_t state_widest(const struct llama_model *m)
{
	uint32_t w = m->n_embd;

	if (m->n_ff > w)
		w = m->n_ff;
	if (m->n_embd_attn > w)
		w = m->n_embd_attn;
	return w;
}

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
	/*
	 * ⚠ xb IS BOTH the normalised embedding and the attention output, and
	 * those are two different widths. Attention writes n_head * head_dim,
	 * which on qwen3 is larger than n_embd; sizing this by n_embd alone
	 * was correct for three architectures and is a heap overflow on the
	 * fourth.
	 */
	s->xb  = calloc(m->n_embd > m->n_embd_attn ? m->n_embd : m->n_embd_attn,
			sizeof(float));
	s->xb2 = calloc(m->n_embd, sizeof(float));
	s->hb  = calloc(m->n_ff, sizeof(float));
	s->hb2 = calloc(m->n_ff, sizeof(float));
	s->q   = calloc((size_t)m->n_head * m->head_dim, sizeof(float));
	s->k   = calloc((size_t)m->n_head_kv * m->head_dim, sizeof(float));
	s->v   = calloc((size_t)m->n_head_kv * m->head_dim, sizeof(float));
	s->att = calloc((size_t)m->n_head * s->n_ctx, sizeof(float));
	s->logits = calloc(m->n_vocab, sizeof(float));

	{
		uint32_t widest = state_widest(m);

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
		unsigned widest = state_widest(m);

		if (maxn > m->n_vocab)
			maxn = m->n_vocab;
		s->dev = charsiu_npu_open(widest, maxn, s->npu_cap);
		if (!s->dev) {
			fprintf(stderr, "charsiu: no NPU; staying on the CPU\n");
		} else {
			/*
			 * ⚠ SAY HOW MANY ARE COMING. Staging is about twenty
			 * seconds of silence and the heartbeat below only
				 * counts up, so a caller drawing a progress bar
			 * has no denominator. Seven projections a layer --
			 * q k v o gate up down -- plus the output head, which
			 * is 113 for the 16 layer model every board round uses
			 * and matches its logs exactly.
			 */
			fprintf(stderr, "charsiu: NPU open, routing tensors with "
				"n <= %u, staging %u\n",
				maxn, m->n_layer * 7 + 1);
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
		unsigned wrote = 0, xwrote = 0;

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
			if (t->xcal && t->nxcal) {
				char xp[256];
				FILE *xf;

				snprintf(xp, sizeof(xp), "%s.x",
					 getenv("CHARSIU_CALIB"));
				/* ⚠ truncate on the FIRST tensor that writes, not on tensor 0:
				 * keyed on i, a run whose first tensor has no vectors
				 * appends to the previous run's file. */
				xf = fopen(xp, xwrote++ ? "ab" : "wb");
				if (xf) {
					fwrite(t->name, 1, sizeof(t->name), xf);
					fwrite(&t->n, sizeof(t->n), 1, xf);
					fwrite(&t->k, sizeof(t->k), 1, xf);
					fwrite(&t->nxcal, sizeof(t->nxcal), 1, xf);
					fwrite(t->xcal, sizeof(float),
					       (size_t)t->nxcal * t->k, xf);
					fclose(xf);
				}
			}
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

void llama_stages_reset(void)
{
	memset(stage_ms, 0, sizeof(stage_ms));
	act_ms = 0.0;
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
	printf("  %-16s %8.2f ms a token          (inside the rows above)\n",
	       "quantising x", act_ms / stage_tok);
}

/* ---- the forward pass ---------------------------------------------------- */

/*
 * ATTENTION, ONE RANGE OF HEADS AT A TIME.
 *
 * Each head reads the whole K and V cache but writes only its own hd floats of
 * xb and its own row of att, so the heads are independent and the split needs
 * no locking and changes no arithmetic: the sum over t inside a head keeps its
 * order, which is what would have moved the tokens.
 */
struct attn_job {
	struct llama_state *s;
	uint32_t l;
	int pos;
	/*
	 * The OLDEST position this layer may look at. 0 on a full layer, which
	 * is every layer of every architecture but gemma3's window ones.
	 */
	int t0;
	uint32_t hd, kvdim, gqa, nkv;
	float scale;
};

/*
 * ⚠ ONE PASS OVER THE KV CACHE PER GROUP, NOT PER HEAD.
 *
 * This model has 32 query heads and 8 key/value heads, so four queries share
 * every kv row -- and the loop below used to walk the whole cache once for each
 * of them, reading every row four times. That was invisible at the sixty-odd
 * positions every round since 352 has run at, and round 372 made it visible:
 * at 384 tokens attention was 36.90 ms a token, 33% of the whole thing and the
 * largest single row in the table, against 4.53 ms at 64.
 *
 * It also grew 8.15x while the average position grew 5.27x. Superlinear is the
 * signature of falling out of cache: the kv cache is 4.6 MB at position 70 and
 * 25.6 MB at 390, so the redundant reads stop being free exactly when there
 * start to be a lot of them.
 *
 * Processing a group together reads each row once. The ARITHMETIC IS
 * UNTOUCHED: every head still sums over the same i in the same order and over
 * the same t in the same order, and only the interleaving changes, so this
 * cannot move a token.
 */
static void attn_heads(void *vj, uint64_t h0, uint64_t nh)
{
	const struct attn_job *j = vj;
	struct llama_state *s = j->s;
	uint32_t hd = j->hd, gqa = j->gqa, nkv = j->nkv;
	int pos = j->pos, t0 = j->t0;
	uint64_t h = h0, end = h0 + nh;
	const float *kbase, *vbase;
	size_t kstride;

	while (h < end) {
		uint32_t kvh = (uint32_t)h / gqa;
		/* the queries of this kv head that fall inside the range */
		uint64_t g0 = h, g1 = (uint64_t)(kvh + 1) * gqa;
		unsigned n, q;

		/*
		 * CHARSIU_ATTN_PERHEAD takes the group back down to one, which
		 * IS the old walk: dot loop, softmax, accumulate loop, per
		 * head, reading every kv row once for each of the four queries
		 * that share it. It is here so the round can measure this
		 * change rather than assume it -- the arithmetic is identical
		 * either way, so nothing else could tell them apart.
		 */
		if (attn_perhead())
			g1 = g0 + 1;
		if (g1 > end)
			g1 = end;

		/*
		 * ONE BRANCH A GROUP, NOT ONE A POSITION. Both layouts are a
		 * base and a stride; picking them here keeps the t loop free
		 * of the question.
		 */
		if (kv_posmajor()) {
			size_t b = (size_t)j->l * s->n_ctx * j->kvdim
				 + (size_t)kvh * hd;

			kbase = s->kcache + b;
			vbase = s->vcache + b;
			kstride = j->kvdim;
		} else {
			size_t b = (size_t)(j->l * nkv + kvh) * s->n_ctx * hd;

			kbase = s->kcache + b;
			vbase = s->vcache + b;
			kstride = hd;
		}
		n = (unsigned)(g1 - g0);
		h = g1;

		for (int t = t0; t <= pos; t++) {
			const float *kt = kbase + (size_t)t * kstride;

			for (q = 0; q < n; q++) {
				const float *qh = s->q + (g0 + q) * hd;
				float a = 0.0f;
				uint32_t i = 0;

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
				/*
				 * ⚠ OPT OUT ONLY. A dot product is a reduction
				 * and four lanes add it up in a different
				 * ORDER; that is a last bit, and a last bit
				 * moved a token on the host. It did not on the
				 * board in round 370, and it is the default
				 * since 372. CHARSIU_EXACT_ATTN goes back.
				 */
				if (fast_attn()) {
					float32x4_t a0 = vdupq_n_f32(0.0f);
					float32x4_t a1 = vdupq_n_f32(0.0f);

					for (; i + 8 <= hd; i += 8) {
						a0 = vfmaq_f32(a0,
							vld1q_f32(qh + i),
							vld1q_f32(kt + i));
						a1 = vfmaq_f32(a1,
							vld1q_f32(qh + i + 4),
							vld1q_f32(kt + i + 4));
					}
					a = vaddvq_f32(vaddq_f32(a0, a1));
				}
#endif
				for (; i < hd; i++)
					a += qh[i] * kt[i];
				s->att[(size_t)(g0 + q) * s->n_ctx + t] =
					a * j->scale;
			}
		}

		for (q = 0; q < n; q++) {
			/*
			 * ⚠ SOFTMAX OVER THE WINDOW, not over the cache. The
			 * positions before t0 were never scored, so including
			 * them would normalise against whatever the buffer
			 * happens to hold from an earlier token.
			 */
			softmax(s->att + (size_t)(g0 + q) * s->n_ctx + t0,
				pos + 1 - t0);
			memset(s->xb + (g0 + q) * hd, 0, hd * sizeof(float));
		}

		for (int t = t0; t <= pos; t++) {
			const float *vt = vbase + (size_t)t * kstride;

			for (q = 0; q < n; q++) {
				float *out = s->xb + (g0 + q) * hd;
				float a = s->att[(size_t)(g0 + q) * s->n_ctx + t];
				uint32_t i = 0;

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
				/*
				 * Each i lands in its own accumulator, so there
				 * is no reduction and no order to change. The
				 * barrier keeps the multiply and the add
				 * separately rounded, which is what the scalar
				 * line below compiles to.
				 */
				if (!cpu_plain()) {
					float32x4_t av = vdupq_n_f32(a);

					for (; i + 4 <= hd; i += 4) {
						float32x4_t p = vmulq_f32(av,
							vld1q_f32(vt + i));

						__asm__("" : "+w"(p));
						vst1q_f32(out + i,
							vaddq_f32(vld1q_f32(out + i),
								  p));
					}
				}
#endif
				for (; i < hd; i++)
					out[i] += a * vt[i];
			}
		}
	}
}

const float *llama_forward(struct llama_state *s, int32_t token, int pos)
{
	const struct llama_model *m = s->m;
	uint32_t hd = m->head_dim;
	uint32_t kvdim = m->n_head_kv * hd;
	uint32_t gqa = m->n_head / m->n_head_kv;
	float scale = 1.0f / sqrtf((float)hd);
	static float *freqbuf;
	static float *ropecs, *ropecs_swa;
	const float *freqf = NULL;

	double t0, t1;
	unsigned long cur_layer = 0;

	if (pos >= s->n_ctx)
		return NULL;

	if (stage_on < 0)
		stage_on = getenv("CHARSIU_STAGES") != NULL;
	/*
	 * ⚠ THE STAGE IS ALSO THE BREADCRUMB. A crash anywhere in the forward
	 * pass used to arrive as "Segmentation fault" with nothing else, and
	 * once charsiu_note started clearing itself on the way out of the NPU
	 * code, everything that was NOT the NPU became "something outside the
	 * NPU code (0, 0)" -- true, and one bit of information.
	 *
	 * One store a stage, sixteen stages a layer, is nothing next to a
	 * projection, and it turns the same crash into a layer number and the
	 * name of the step. The note points at a string in stage_name[], which
	 * is static and outlives any crash.
	 */
#define STAGE(i) do { charsiu_note(stage_name[i], cur_layer, (unsigned long)pos); \
		      if (stage_on) { t1 = now_ms(); stage_ms[i] += t1 - t0; \
			      t0 = t1; } } while (0)
	t0 = t1 = stage_on ? now_ms() : 0.0;

	if (m->rope_freqs) {
		if (!freqbuf)
			freqbuf = calloc(hd, sizeof(float));
		gguf_row_f32(m->rope_freqs, 0, freqbuf);
		freqf = freqbuf;
	}

	if (!ropecs)
		ropecs = calloc(hd, sizeof(float));
	rope_table(ropecs, hd, pos, m->rope_base, freqf);
	/*
	 * The second table exists only when the two bases differ, which is
	 * gemma3 and nothing else. Both are still one table a TOKEN, not one a
	 * layer: theta depends on the base, the pair and the position.
	 */
	if (m->swa_pattern && m->rope_base_swa != m->rope_base) {
		if (!ropecs_swa)
			ropecs_swa = calloc(hd, sizeof(float));
		rope_table(ropecs_swa, hd, pos, m->rope_base_swa, freqf);
	}

	gguf_row_f32(m->tok_embd, (uint64_t)token, s->x);
	if (m->embd_scale != 1.0f)
		for (uint32_t i = 0; i < m->n_embd; i++)
			s->x[i] *= m->embd_scale;
	STAGE(ST_EMBD);

	for (uint32_t l = 0; l < m->n_layer; l++) {
		const struct llama_layer *L = &m->layers[l];

		cur_layer = l;
		/*
		 * ⚠ `il % n < n - 1` IS THE PATTERN, taken from llama.cpp's
		 * set_swa_pattern: with the gemma default of 6 that is five
		 * window layers and then a full one, not one window in six.
		 */
		int swa = m->swa_pattern &&
			  (l % m->swa_pattern) < m->swa_pattern - 1;
		const float *cs = swa && ropecs_swa ? ropecs_swa : ropecs;
		/*
		 * A window layer may look at the last n_swa positions
		 * INCLUDING this one: llama.cpp masks when pos - t >= n_swa.
		 */
		/*
		 * ⚠⚠ NOT t0, AND THIS ONE COST FOUR BOARD ROUNDS.
		 *
		 * llama_forward's timing variable is a double called t0 and the
		 * STAGE macro assigns to it. An int t0 declared inside the
		 * layer loop shadows it, so every stage handed the sliding
		 * window's start a millisecond timestamp converted to int --
		 * and attention then calls
		 *
		 *     softmax(s->att + (g0 + q) * n_ctx + t0, pos + 1 - t0)
		 *
		 * on a pointer megabytes past the buffer. That is a SIGSEGV in
		 * the attention block of layer 0, which is exactly where the
		 * breadcrumbs put it.
		 *
		 * It needs CHARSIU_STAGES to trigger, and the board has it on,
		 * so "off by default" was never the protection it looked like.
		 * It arrived with sliding window attention in the gemma3
		 * commit and every NPU run since then died on it.
		 */
		int tlo = swa && pos + 1 > (int)m->n_swa ?
			  pos + 1 - (int)m->n_swa : 0;

		rmsnorm(s->xb, s->x, L->attn_norm, m->n_embd, m->rms_eps);
		STAGE(ST_NORM1);

		matvec_pair(s, s->xb, L->wq, s->q, L->wk, s->k, L->wv, s->v);
		/*
		 * ⚠ BEFORE ROPE, NOT AFTER. The bias is part of the projection;
		 * rotating a biased vector is not the same as biasing a rotated
		 * one, and the wrong order is the kind of mistake that still
		 * produces fluent-looking text.
		 */
		if (L->bq) {
			add_bias(s->q, L->bq, m->n_head * hd);
			add_bias(s->k, L->bk, m->n_head_kv * hd);
			add_bias(s->v, L->bv, m->n_head_kv * hd);
		}
		/*
		 * ⚠ AFTER THE BIAS AND BEFORE ROPE, which is the one order
		 * that is not interchangeable: rope mixes element 2i with
		 * 2i+1, so normalising afterwards divides a rotated pair by a
		 * sum of squares that rotation already changed. V is NOT
		 * normed -- only q and k are, because only they meet in a dot
		 * product.
		 */
		if (L->q_norm) {
			qk_norm(s->q, m->n_head, hd, L->q_norm, m->rms_eps);
			qk_norm(s->k, m->n_head_kv, hd, L->k_norm, m->rms_eps);
		}
		STAGE(ST_QKV);

		charsiu_note("rope on q", cur_layer, (unsigned long)m->n_head);
		rope(s->q, m->n_head, hd, cs, m->rope_neox);
		charsiu_note("rope on k", cur_layer,
			     (unsigned long)m->n_head_kv);
		rope(s->k, m->n_head_kv, hd, cs, m->rope_neox);
		charsiu_note("the kv cache copy", cur_layer,
			     (unsigned long)kvdim);

		/*
		 * ⚠ HEAD MAJOR: [layer][kv head][position][head dim].
		 *
		 * The cache used to be [layer][position][kv dim], which puts
		 * one kv head's consecutive positions 2048 bytes apart -- and
		 * attention walks exactly that way, one head over every
		 * position. Round 373 measured the walk at 2.65 GB/s where
		 * this CPU reads sequentially at 7.13, with 4.9 ms a token in
		 * it at 384 tokens.
		 *
		 * Head major makes the walk contiguous. The cost is that the
		 * write is now one memcpy a kv head rather than one a layer:
		 * eight copies of 256 bytes instead of one of 2048, 128 a
		 * token, which is nothing against 12.9 MB of reading.
		 *
		 * The VALUES are untouched, so this cannot move a token.
		 */
		if (kv_posmajor()) {
			size_t off = ((size_t)l * s->n_ctx + pos) * kvdim;

			memcpy(s->kcache + off, s->k, kvdim * sizeof(float));
			memcpy(s->vcache + off, s->v, kvdim * sizeof(float));
		} else {
			for (uint32_t kh = 0; kh < m->n_head_kv; kh++) {
				size_t off = ((size_t)(l * m->n_head_kv + kh)
					      * s->n_ctx + pos) * hd;

				memcpy(s->kcache + off, s->k + kh * hd,
				       hd * sizeof(float));
				memcpy(s->vcache + off, s->v + kh * hd,
				       hd * sizeof(float));
			}
		}
		STAGE(ST_ROPE);

		{
			struct attn_job aj = { s, l, pos, tlo, hd, kvdim, gqa,
					       m->n_head_kv, scale };

			charsiu_note("attention: entering", cur_layer,
				     (unsigned long)m->n_head);

			/*
			 * ⚠ SERIAL, AND ROUND 368 IS WHY.
			 *
			 * Splitting these heads over the pool was measured at
			 * 22.70 ms a token against 7.75 serial when the
			 * process was left where the scheduler put it, and at
			 * 7.42 against 7.75 when it was pinned to the four
			 * A72s. So it costs 15 ms in the ordinary case and
			 * buys 0.33 ms in the best one: the work per layer is
			 * two milliseconds and a fan out and fan in around it
			 * is not free.
			 *
			 * The path stays, because attention grows with the
			 * context and 38 positions is not where this question
			 * gets settled. CHARSIU_ATTN_POOL turns it on.
			 */
			if (attn_pool()) {
				charsiu_note("attention over the pool",
					     cur_layer, (unsigned long)g_pool.n);
				pool_run(attn_heads, &aj, m->n_head);
			} else {
				attn_heads(&aj, 0, m->n_head);
			}
			charsiu_note("attention: done", cur_layer,
				     (unsigned long)pos);
		}

		STAGE(ST_ATTN);

		matvec(s, L->wo, s->xb, s->xb2);
		STAGE(ST_WO);
		/*
		 * ⚠ ON THE BRANCH, BEFORE THE RESIDUAL ADD. Normalising after
		 * the add would normalise the residual stream as well, which
		 * is a different model.
		 */
		if (L->attn_post_norm)
			rmsnorm(s->xb2, s->xb2, L->attn_post_norm, m->n_embd,
				m->rms_eps);
		for (uint32_t i = 0; i < m->n_embd; i++)
			s->x[i] += s->xb2[i];
		STAGE(ST_RES1);

		rmsnorm(s->xb, s->x, L->ffn_norm, m->n_embd, m->rms_eps);
		STAGE(ST_NORM2);
		matvec_pair(s, s->xb, L->gate, s->hb, L->up, s->hb2, NULL, NULL);
		STAGE(ST_GATEUP);
		if (m->ffn_gelu)
			gelu_mul(s->hb, s->hb2, m->n_ff);
		else
			silu_mul(s->hb, s->hb2, m->n_ff);
		STAGE(ST_SILU);
		matvec(s, L->down, s->hb, s->xb2);
		STAGE(ST_DOWN);
		if (L->ffn_post_norm)
			rmsnorm(s->xb2, s->xb2, L->ffn_post_norm, m->n_embd,
				m->rms_eps);
		for (uint32_t i = 0; i < m->n_embd; i++)
			s->x[i] += s->xb2[i];
		STAGE(ST_RES2);
	}

	rmsnorm(s->xb, s->x, m->out_norm, m->n_embd, m->rms_eps);
	STAGE(ST_NORMF);
	matvec(s, m->output, s->xb, s->logits);
	/*
	 * Squash the logits into (-c, c) with a tanh. gemma2 does this to the
	 * attention scores as well; gemma3 caps only the output, and the files
	 * measured leave even that off, so this is here for the ones that
	 * declare it rather than as something every gemma needs.
	 */
	if (m->final_softcap > 0.0f)
		for (uint32_t i = 0; i < m->n_vocab; i++)
			s->logits[i] = tanhf(s->logits[i] / m->final_softcap) *
				       m->final_softcap;
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
