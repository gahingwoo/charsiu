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
 * ⚠ e^x FOUR AT A TIME NOW LIVES IN THE HEADER, as charsiu_vexpq. It was
 * written here for SiLU and the vision tower's softmax turned out to want it
 * far more: a picture asks for 151 million exponentials against a feed
 * forward's 131072 a token.
 */
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

/*
 * ⚠⚠ AND THE TANH GELU IS THE SAME SIGMOID, WHICH IS AN IDENTITY RATHER THAN
 * AN APPROXIMATION.
 *
 *   0.5 * (1 + tanh y)  ==  1 / (1 + e^-2y)
 *
 * exact to machine epsilon over the whole range, so gemma's activation is
 * x / (1 + e^(-2k(x + 0.044715 x^3))) and goes through the same charsiu_vexpq
 * silu already uses. It was one tanhf an element on the per token path.
 *
 * The vision tower priced the identical loop at its own shape: 34.65 ms with
 * tanhf against 2.66 with an exponential, one layer of 3145728 elements, and
 * 14x of the whole win there was this line alone, core count independent.
 *
 * CHARSIU_EXACT_GELU is the control, and it is the same variable the tower
 * takes so one switch covers both.
 */
static int fast_gelu(void)
{
	static int v = -1;

	if (v < 0)
		v = getenv("CHARSIU_EXACT_GELU") == NULL && !cpu_plain();
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
						  charsiu_vexpq(vnegq_f32(g)));

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
	uint32_t i = 0;

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
	if (fast_gelu())
		for (; i + 4 <= n; i += 4) {
			float32x4_t g = vld1q_f32(hb + i);
			float32x4_t c = vmulq_f32(vmulq_f32(g, g), g);
			float32x4_t y = vmulq_f32(vdupq_n_f32(-2.0f * k),
					vaddq_f32(g, vmulq_f32(
						vdupq_n_f32(0.044715f), c)));
			float32x4_t d = vaddq_f32(vdupq_n_f32(1.0f),
						  charsiu_vexpq(y));

			vst1q_f32(hb + i, vmulq_f32(vdivq_f32(g, d),
						    vld1q_f32(hb2 + i)));
		}
#endif
	for (; i < n; i++) {
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
static int g_pinned_cpu = -1;

/*
 * ⚠⚠ scaling_cur_freq AT STARTUP IS AN IDLE CPU, and ondemand has not seen any
 * work yet. Two consecutive board runs of the same command reported 2208 MHz
 * and 1200 MHz from this line, both with the governor at ondemand, purely
 * because of when it was read. A number that swings by a factor of two on the
 * same machine doing the same thing is not a measurement of anything.
 *
 * charsiu_cpu_mhz() is what a caller uses AFTER the work, which is the reading
 * that decides whether a slow run was a slow clock.
 */
long charsiu_cpu_mhz(void)
{
	char path[128];
	FILE *f;
	long khz = 0;

	if (g_pinned_cpu < 0)
		return 0;
	snprintf(path, sizeof(path),
		 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq",
		 g_pinned_cpu);
	f = fopen(path, "r");
	if (!f)
		return 0;
	if (fscanf(f, "%ld", &khz) != 1)
		khz = 0;
	fclose(f);
	return khz / 1000;
}

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
	g_pinned_cpu = first;

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
	if ((gov[0] || khz) && charsiu_diag())
		fprintf(stderr, "charsiu: cpu%d governor %s, %ld MHz idle\n",
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
	/* ⚠ the FAILURE still speaks. A pin that did not apply changes the
	 * numbers and is not a running commentary. */
	if (CPU_COUNT(&set) && sched_setaffinity(0, sizeof(set), &set))
		fprintf(stderr, "charsiu: CHARSIU_CPUS=%s did not apply\n", spec);
	else if (charsiu_diag())
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
 * ⚠ THE POOL IS STARTED BY llama_state_new AND NOTHING ELSE STARTED IT. A
 * whisper transcription or a vision tower has no llama_state, so every
 * charsiu_parallel_for in those graphs ran on one core -- silently, because
 * pool_run's single thread path is a plain call and looks like success.
 */
void charsiu_threads_start(int nthreads)
{
	pool_start(nthreads);
}

int charsiu_threads(void)
{
	return g_pool.n ? g_pool.n : 1;
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
/* ⚠ NOT stage_ms: that name is the per stage table further down */
/* the staging clock lives in npupool.c with the staging */

double llama_stage_ms(void)
{
	return charsiu_pool_stage_ms;
}
static int stage_on = -1;

/* CHARSIU_DBG_LAYERS: the RMS of the residual stream after every layer */
static int dbg_layers(void)
{
	static int v = -1;

	if (v < 0)
		v = getenv("CHARSIU_DBG_LAYERS") != NULL;
	return v;
}

static double now_ms(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

/*
 * WHAT BATCHING BUYS ON THIS BOARD, AND WHETHER IT IS RIGHT, which the first
 * run said it is not.
 *
 * Not a synthetic matmul: the staged tensors of a real model, walked once so
 * the weights come from memory the way a forward pass makes them.
 *
 * ⚠ IT SWEEPS m, and that is the point rather than a convenience. m = 2 and
 * m = 4 are the widths charsiu_acc_index was solved on and m = 8 is the one it
 * was confirmed on. 32 extrapolates both that expression and 0x40b8 = 3 * rows,
 * which was swept to m = 4. If the small widths agree and the large ones do
 * not, the plumbing is right and one of those two formulas stops somewhere. If
 * m = 2 disagrees, it is the plumbing.
 *
 * ⚠ AND IT SAYS WHICH ROWS. A wrong permutation puts a neighbour's value in a
 * slot; a wrong scale multiplies every slot. "Row 0 agrees and the rest do
 * not" and "everything is off by a factor" are different faults, and one
 * worst-case number cannot tell them apart.
 */
int llama_batch_probe(struct llama_state *s, const struct llama_model *m,
		      unsigned mmax)
{
	/*
	 * ⚠ THE WIDTHS THE PREFILL WILL ACTUALLY USE, not just the ones the
	 * read order was solved on. roleswap2 is exact at 2, 4, 16 and 32 and
	 * m = 8 is 871 of 904, so the question is now where else it bends --
	 * and a chunk of 48, 64 or 80 is what a real prompt hands it.
	 * --batch-probe N still caps this.
	 */
	static const unsigned MS_DEFAULT[] = { 2, 4, 8, 16, 32, 48, 64, 80 };
	/*
	 * ⚠⚠ THE WIDTHS A REAL PROMPT HANDS IT ARE NOT THIS LIST.
	 *
	 * phi3's prompt is 87 tokens, which at a chunk of 32 is 32, 32 and
	 * TWENTY THREE -- a width this sweep has never asked about, on a model
	 * whose text is wrong on the board while every width here is exact.
	 * m = 8 was only ever found because 8 happened to be in the list.
	 *
	 * CHARSIU_PROBE_WIDTHS takes a space or comma separated list, so a
	 * round can ask about 23, or sweep 2..32 dense and stop finding broken
	 * widths by luck.
	 */
	unsigned ms_buf[64];
	const unsigned *MS = MS_DEFAULT;
	unsigned n_ms = sizeof(MS_DEFAULT) / sizeof(MS_DEFAULT[0]);
	const char *wenv = getenv("CHARSIU_PROBE_WIDTHS");
	/*
	 * ⚠⚠ CHARSIU_PROBE_MAXT: HOW MANY STAGED TENSORS A WIDTH COSTS, which
	 * is what stops a dense sweep from being affordable.
	 *
	 * The one row reference is O(m) matvecs per tensor and it runs over
	 * every staged tensor, so a width costs m * 225 submits before the
	 * batched side is even asked. At m = 32 the reference alone was
	 * 5864 ms, and 2..64 dense is sixty three widths -- days, not a round.
	 * Capping the tensors is the only term that can come down without
	 * dropping widths, and dropping widths is the whole point of the
	 * sweep.
	 *
	 * ⚠ IT TAKES THE FIRST N STAGED TENSORS, AND THAT IS A BIAS WITH A
	 * SHAPE. The pool fills lazily in the order the first forward pass
	 * touches things, so the first N are layer 0's projections -- q, k, v,
	 * o, then gate, up, down -- and N = 8 reaches into layer 1. That is
	 * deliberate: it still covers every DISTINCT (k, n) an ordinary layer
	 * has, including the n = 8192 gate/up pair that m = 8 misses, so a
	 * width broken the way m = 8 is broken still shows.
	 *
	 * What it drops is the OUTPUT HEAD, which is staged last and is the
	 * one shape nothing else in the model resembles: 128256 channels
	 * against 8192. A width that is wrong only on the head reads as PASS
	 * under a cap, and the round after a capped sweep has to be an uncapped
	 * one on whatever widths it left standing. The table says the cap, on
	 * every row, so that argument is never made from a number that has
	 * forgotten it.
	 */
	unsigned maxt = 0;
	const char *tenv = getenv("CHARSIU_PROBE_MAXT");
	float *X = NULL, *Y = NULL, *Yref = NULL;
	struct charsiu_act a;
	unsigned widest = 0, n_staged = 0;
	int rc = -1;

	if (!s->pool.dev) {
		fprintf(stderr, "charsiu: no NPU staged; nothing to batch\n");
		return -1;
	}
	if (wenv && *wenv) {
		unsigned n = 0, top = 0;
		char *p = (char *)wenv;

		const unsigned cap = sizeof(ms_buf) / sizeof(ms_buf[0]);

		/*
		 * ⚠⚠ EVERY WAY THIS CAN GO WRONG REFUSES OUT LOUD, and none of
		 * them shortens the list quietly.
		 *
		 * The first draft stopped on a full buffer and stopped on the
		 * first character it did not like, both without a word. So
		 * "2 4 x 8" would have measured 2 and 4 and said it had done
		 * what was asked, and a list of seventy widths would have lost
		 * the last six -- which are the widest, the slowest to reach
		 * and the whole reason for passing a list by hand.
		 *
		 * This project has now lost four rounds to output that was cut
		 * without saying so. A probe whose ONE job is to ask an exact
		 * question must not answer a smaller one.
		 */
		for (;;) {
			while (*p == ' ' || *p == ',' || *p == '\t')
				p++;
			if (!*p)
				break;
			if (*p < '0' || *p > '9') {
				fprintf(stderr, "charsiu: CHARSIU_PROBE_WIDTHS "
					"stops making sense at \"%s\" -- "
					"refusing rather than measuring the "
					"part before it\n", p);
				return -1;
			}
			if (n == cap) {
				fprintf(stderr, "charsiu: CHARSIU_PROBE_WIDTHS "
					"holds at most %u widths and there are "
					"more after \"%s\" -- refusing rather "
					"than dropping the widest ones\n",
					cap, p);
				return -1;
			}
			ms_buf[n] = (unsigned)strtoul(p, &p, 10);
			if (ms_buf[n] < 2) {
				fprintf(stderr, "charsiu: CHARSIU_PROBE_WIDTHS "
					"asks for width %u, and 1 is the decode "
					"path rather than a batch -- refusing\n",
					ms_buf[n]);
				return -1;
			}
			if (ms_buf[n] > top)
				top = ms_buf[n];
			n++;
		}
		/*
		 * ⚠ AND THE LIST RAISES THE CAP. --batch-probe caps the sweep,
		 * and a round that asks for width 87 under a cap of 80 gets 80
		 * and a table that does not mention it. Asking explicitly is
		 * the whole point of this switch.
		 */
		if (top > mmax)
			mmax = top;
		if (!n) {
			fprintf(stderr, "charsiu: CHARSIU_PROBE_WIDTHS=\"%s\" "
				"parsed to nothing; refusing to fall back to "
				"the default list, because a round that "
				"silently asks a different question is the "
				"failure this probe exists to avoid\n", wenv);
			return -1;
		}
		MS = ms_buf;
		n_ms = n;
	}
	if (tenv && *tenv) {
		maxt = (unsigned)strtoul(tenv, NULL, 10);
		if (!maxt) {
			fprintf(stderr, "charsiu: CHARSIU_PROBE_MAXT=\"%s\" "
				"parsed to zero, which would check no tensors "
				"at all; refusing rather than quietly walking "
				"all of them, because a round that silently "
				"asks a different question is the failure this "
				"probe exists to avoid\n", tenv);
			return -1;
		}
	}
	for (unsigned i = 0; i < s->pool.n; i++) {
		if (s->pool.id[i] < 0)
			continue;
		n_staged++;
		if (s->pool.t[i].k > widest)
			widest = (unsigned)s->pool.t[i].k;
	}
	/*
	 * ⚠ ONCE, NOT ONCE A ROW. The first version allocated and blocked the
	 * activation inside the timing loop and charged all of it to the one
	 * row path, which is the side it was trying to beat.
	 *
	 * ⚠ AND IT IS SIZED OFF THE WIDEST STAGED TENSOR, NOT THE WIDEST
	 * CHECKED ONE, so CHARSIU_PROBE_MAXT does not reach it. The cap is
	 * about time; sizing this to the first N as well would make one switch
	 * quietly change what the tensors it did not check could have been.
	 */
	if (charsiu_act_alloc(&a, (int)widest) < 0)
		return -1;

	/*
	 * ⚠ SAY WHICH WIDTHS, because the caller caps this and the two have
	 * already disagreed. MS[] was widened to 80 and the board script was
	 * still passing --batch-probe 32, so a round that was run to reach 48,
	 * 64 and 80 stopped at 32 and its header said 32 while the reason for
	 * running it said otherwise. A list that prints cannot do that.
	 */
	printf("\n  batching %u layers, checked before it is timed; widths",
	       m->n_layer);
	for (unsigned i = 0; i < n_ms; i++)
		if (MS[i] <= mmax)
			printf(" %u", MS[i]);
	printf("   (--batch-probe caps at %u)\n", mmax);
	/*
	 * ⚠⚠ A CAPPED ROUND MUST NEVER BE READABLE AS A FULL ONE. It is said
	 * here in words and again in the tensors column on every single row,
	 * because the line that gets pasted out of a round is a table row and
	 * not a header, and "225" and "8" are both just a number until one of
	 * them is written as a fraction of the other.
	 */
	if (maxt && maxt < n_staged)
		printf("  ⚠ CHARSIU_PROBE_MAXT=%u: only the FIRST %u of %u"
		       " staged tensors are checked. Those are layer 0's"
		       " projections and the start of layer 1, so every shape"
		       " an ordinary layer has is covered -- but NOT the output"
		       " head, which is staged last and is the one shape"
		       " nothing else resembles. A width that is exact here is"
		       " exact on a layer, not on the model.\n", maxt, maxt,
		       n_staged);
	else
		printf("  all %u staged tensors are checked"
		       " (CHARSIU_PROBE_MAXT caps this)\n", n_staged);
	printf("    m    tensors   worst rel   rows that agree"
	       "     one row    batched  speedup  us a row    GB/s"
	       "   where the batched time went, ms (prep is buffers and the"
	       " output zero; rest is what none of them caught)\n");

	/*
	 * ⚠⚠ TWO AXES AND SEVEN READINGS, at m = 2 on one tensor, before any
	 * timing.
	 *
	 * charsiu_acc_index is solved and confirmed on the int8 accumulator on
	 * the HEIGHT axis: exact at every N to 2048 and every m to 8. The
	 * runtime's path is w4a16 int4, and job.c has known since round 347
	 * that the vendor puts M on the WIDTH axis for that one, and that at
	 * M = 1 the two axes collapse -- which is why decode never noticed and
	 * why the comment saying so sat there while I looked everywhere else.
	 *
	 * So the axis and the reading are one question with two knobs, and this
	 * asks both at once rather than fitting either.
	 *
	 * ⚠ ROW 0 IS NOT A CONTROL HERE. On the wrong axis it is wrong too --
	 * the last round read "row 0 is exact" off six of two thousand values
	 * and spent itself on the one term row 0 cannot see. Both rows are
	 * counted and both are printed.
	 */
	if (getenv("CHARSIU_BATCH_SWEEP")) {
		static const char *READS[] = { "acc", "flat", "2", "4", "8",
					       "16", "32" };
		static const char *AXES[] = { "h", "w" };
		unsigned i0 = 0;

		while (i0 < s->pool.n && s->pool.id[i0] < 0)
			i0++;
		if (i0 < s->pool.n) {
			const struct npu_tensor *t = &s->pool.t[i0];
			size_t nx = (size_t)2 * t->k, ny = (size_t)2 * t->n;

			X = realloc(X, nx * sizeof(*X));
			Y = realloc(Y, ny * sizeof(*Y));
			Yref = realloc(Yref, ny * sizeof(*Yref));
			if (X && Y && Yref) {
				for (size_t j = 0; j < nx; j++)
					X[j] = (float)(((j * 2654435761u) >> 9)
						       & 0xff) / 255.0f - 0.5f;
				for (unsigned r = 0; r < 2; r++) {
					charsiu_act_set(&a, X + (size_t)r * t->k,
							(int)t->k);
					charsiu_act_blocks(&a);
					/*
					 * ⚠ int8's matvec READS a->q1 and the
					 * float path does not, so without this
					 * the reference came back all zeros on
					 * an int8 model -- and the run that
					 * finally produced both batched rows
					 * scored 0 of 224 against it.
					 */
					if (charsiu_npu_needs_q1(s->pool.dev))
						charsiu_act_q1(&a);
					charsiu_npu_matvec(s->pool.dev, s->pool.id[i0],
						&a, Yref + (size_t)r * t->n);
				}
				printf("\n  axis and reading, %s at m=2,"
				       " %u channels a row\n",
				       t->name, (unsigned)t->n);
				printf("    axis  read     row0        row1\n");
				setenv("CHARSIU_BATCH_ROWSTEP", "0", 1);
				for (unsigned ax = 0; ax < 2; ax++) {
					if (AXES[ax][0] == 'w')
						setenv("CHARSIU_M_AXIS", "w", 1);
					else
						unsetenv("CHARSIU_M_AXIS");
					for (unsigned q = 0; q < sizeof(READS)/sizeof(*READS); q++) {
						unsigned ok0 = 0, ok1 = 0;

						setenv("CHARSIU_BATCH_READ",
						       READS[q], 1);
						if (charsiu_npu_matmul(s->pool.dev,
							s->pool.id[i0], X, 2, Y))
							continue;
						for (unsigned j = 0; j < (unsigned)t->n; j++) {
							double w0 = Yref[j];
							double w1 = Yref[t->n + j];

							if (fabs(Y[j] - w0) <= (fabs(w0) > 1e-3 ? fabs(w0)*1e-3 : 1e-3)) ok0++;
							if (fabs(Y[t->n+j] - w1) <= (fabs(w1) > 1e-3 ? fabs(w1)*1e-3 : 1e-3)) ok1++;
						}
						printf("      %s  %-5s  %5u/%-5u  %5u/%-5u%s\n",
						       AXES[ax], READS[q],
						       ok0, (unsigned)t->n,
						       ok1, (unsigned)t->n,
						       (ok0 == t->n && ok1 == t->n)
						       ? "   <== both rows" : "");
					}
				}
				/*
				 * ⚠ AND THE ROW TERM, on the reading that just
				 * returned a whole row. Row 0 contributes
				 * nothing to it, so it is a control that
				 * cannot move: a step that changes row 0 is
				 * measuring something else.
				 */
				unsetenv("CHARSIU_M_AXIS");
				setenv("CHARSIU_BATCH_READ", "4", 1);
				printf("\n    row step on the height axis,"
				       " atom 4\n");
				{
				static const int STEPS[] = { 1, 2, 4, 8, 16, 32,
							     64, 128, 256, 512,
							     1024, 2048 };
				for (unsigned q = 0; q < sizeof(STEPS)/sizeof(*STEPS); q++) {
					char b[16];
					unsigned ok0 = 0, ok1 = 0;

					snprintf(b, sizeof(b), "%d", STEPS[q]);
					setenv("CHARSIU_BATCH_ROWSTEP", b, 1);
					if (charsiu_npu_matmul(s->pool.dev,
						s->pool.id[i0], X, 2, Y))
						continue;
					for (unsigned j = 0; j < (unsigned)t->n; j++) {
						double w0 = Yref[j];
						double w1 = Yref[t->n + j];

						if (fabs(Y[j] - w0) <= (fabs(w0) > 1e-3 ? fabs(w0)*1e-3 : 1e-3)) ok0++;
						if (fabs(Y[t->n+j] - w1) <= (fabs(w1) > 1e-3 ? fabs(w1)*1e-3 : 1e-3)) ok1++;
					}
					printf("      step %5d   row0 %5u/%-5u"
					       "   row1 %5u/%-5u%s\n",
					       STEPS[q], ok0, (unsigned)t->n,
					       ok1, (unsigned)t->n,
					       ok1 == t->n ? "   <== that is it" : "");
				}
				}
				unsetenv("CHARSIU_BATCH_ROWSTEP");
				/*
				 * ⚠ AND THE INPUT. Row 1 came back the right
				 * magnitude and the wrong number at every row
				 * step, which is a real dot product of the
				 * wrong activation rather than a misplaced
				 * one. Row 0 stays a control for pack 0; for
				 * the others it moves, and that is information
				 * too.
				 */
				printf("\n    input packing, height axis,"
				       " atom 4 read\n");
				for (int pk = 0; pk < 3; pk++) {
					char b[8];
					unsigned ok0 = 0, ok1 = 0;

					snprintf(b, sizeof(b), "%d", pk);
					setenv("CHARSIU_BATCH_PACK", b, 1);
					if (charsiu_npu_matmul(s->pool.dev,
						s->pool.id[i0], X, 2, Y))
						continue;
					for (unsigned j = 0; j < (unsigned)t->n; j++) {
						double w0 = Yref[j];
						double w1 = Yref[t->n + j];

						if (fabs(Y[j] - w0) <= (fabs(w0) > 1e-3 ? fabs(w0)*1e-3 : 1e-3)) ok0++;
						if (fabs(Y[t->n+j] - w1) <= (fabs(w1) > 1e-3 ? fabs(w1)*1e-3 : 1e-3)) ok1++;
					}
					printf("      pack %d %-22s row0 %5u/%-5u"
					       "   row1 %5u/%-5u%s\n", pk,
					       pk == 0 ? "[k/atom][m][atom]" :
					       pk == 1 ? "rows contiguous" :
							 "rows at the CBUF stride",
					       ok0, (unsigned)t->n,
					       ok1, (unsigned)t->n,
					       (ok0 == t->n && ok1 == t->n)
					       ? "   <== both rows" : "");
				}
				unsetenv("CHARSIU_BATCH_PACK");
				/*
				 * ⚠⚠ AND 0x40b8, WHICH IS THE ONE REGISTER
				 * WITH A KNOWN m DEPENDENCE.
				 *
				 * Everything else is now confirmed on this
				 * path: the input surface (pack 0 is the only
				 * one that gets row 0 at all), the weights, the
				 * coefficients, the output group stride, and
				 * the whole of row 0. Row 1 is absent at every
				 * row step and every reading, which is not a
				 * misplaced row, it is a row that was never
				 * produced.
				 *
				 * 0x40b8 is ow * (2 * full_oh - win_orows), an
				 * output height quantity, and 3 * rows was
				 * swept on the int8 accumulator. w4a16 has its
				 * own output stage -- wide8 is forced to zero
				 * for int4 and w4_dpu takes over -- and nothing
				 * has swept it there.
				 *
				 * ⚠ Row 0 is the control and it is a real one:
				 * 2048 of 2048, and it has survived twelve row
				 * steps and three packings without moving.
				 */
				printf("\n    DPU 0x40b8 on the int4 path,"
				       " atom 4 read (6 = 3*rows today)\n");
				for (unsigned v = 0; v <= 16; v++) {
					char b[16];
					unsigned ok0 = 0, ok1 = 0;

					snprintf(b, sizeof(b), "%u", v);
					setenv("CHARSIU_DPU_40B8", b, 1);
					if (charsiu_npu_matmul(s->pool.dev,
						s->pool.id[i0], X, 2, Y))
						continue;
					for (unsigned j = 0; j < (unsigned)t->n; j++) {
						double w0 = Yref[j];
						double w1 = Yref[t->n + j];

						if (fabs(Y[j] - w0) <= (fabs(w0) > 1e-3 ? fabs(w0)*1e-3 : 1e-3)) ok0++;
						if (fabs(Y[t->n+j] - w1) <= (fabs(w1) > 1e-3 ? fabs(w1)*1e-3 : 1e-3)) ok1++;
					}
					printf("      0x40b8 %3u   row0 %5u/%-5u"
					       "   row1 %5u/%-5u%s%s\n", v,
					       ok0, (unsigned)t->n,
					       ok1, (unsigned)t->n,
					       v == 6 ? "   <- today" : "",
					       (ok0 == t->n && ok1 == t->n)
					       ? "   <== both rows" : "");
				}
				unsetenv("CHARSIU_DPU_40B8");
				/*
				 * ⚠⚠ THE LAST REGISTER THAT COUNTS ROWS, and
				 * the second one on this path chosen at a width
				 * where it cannot show.
				 *
				 * 0x301c is lines, which is rows - 1, so at
				 * M = 1 both halves of the word are zero and
				 * the w4v form and the int8 form are the same
				 * word. The w4v form puts M in the LOW half,
				 * from a vendor capture that is M = 32 on the
				 * WIDTH axis; everything else this file emits
				 * is the height axis. CHARSIU_W4_301C=high puts
				 * it in the half the rest of the stream uses.
				 */
				printf("\n    CORE 0x301c half, atom 4 read\n");
				for (unsigned q = 0; q < 2; q++) {
					unsigned ok0 = 0, ok1 = 0;

					if (q)
						setenv("CHARSIU_W4_301C", "high", 1);
					else
						unsetenv("CHARSIU_W4_301C");
					if (charsiu_npu_matmul(s->pool.dev,
						s->pool.id[i0], X, 2, Y))
						continue;
					for (unsigned j = 0; j < (unsigned)t->n; j++) {
						double w0 = Yref[j];
						double w1 = Yref[t->n + j];

						if (fabs(Y[j] - w0) <= (fabs(w0) > 1e-3 ? fabs(w0)*1e-3 : 1e-3)) ok0++;
						if (fabs(Y[t->n+j] - w1) <= (fabs(w1) > 1e-3 ? fabs(w1)*1e-3 : 1e-3)) ok1++;
					}
					printf("      M in the %-4s half   row0 %5u/%-5u"
					       "   row1 %5u/%-5u%s%s\n",
					       q ? "high" : "low",
					       ok0, (unsigned)t->n,
					       ok1, (unsigned)t->n,
					       q ? "" : "   <- today",
					       (ok0 == t->n && ok1 == t->n)
					       ? "   <== both rows" : "");
				}
				unsetenv("CHARSIU_W4_301C");
				/*
				 * ⚠⚠ THE CNA, ONE WORD AT A TIME, AGAINST A
				 * STREAM THAT PRODUCES TWO ROWS.
				 *
				 * Five knobs are swept and settled and row 1 is
				 * still absent, so stop turning knobs. The int8
				 * accumulator computes m = 2 at this shape and
				 * its DPU and RDMA blocks are identical to
				 * int4's, so the difference is seven CNA words.
				 * Each is put back to the int8 value on its
				 * own, which is round 260's method.
				 */
				/*
				 * ⚠⚠ IS ROW 1 PRODUCED AT ALL? Ask before
				 * explaining why it is wrong.
				 *
				 * Feed both rows the SAME activation. If the
				 * block computes two rows, row 1 must come back
				 * equal to row 0, which is already known exact.
				 * If it stays absent, row 1 was never produced
				 * and no amount of reading or placing it will
				 * help. I asserted that for two rounds without
				 * testing it.
				 */
				{
					unsigned same = 0;

					memcpy(X + t->k, X, t->k * sizeof(*X));
					if (!charsiu_npu_matmul(s->pool.dev,
						s->pool.id[i0], X, 2, Y)) {
						for (unsigned j = 0; j < (unsigned)t->n; j++) {
							double w = Yref[j];

							if (fabs(Y[t->n+j] - w) <= (fabs(w) > 1e-3 ? fabs(w)*1e-3 : 1e-3)) same++;
						}
						printf("\n    both rows fed the SAME"
						       " activation: row1 matches"
						       " row0 in %u of %u\n",
						       same, (unsigned)t->n);
						printf("      %s\n", same > 100
						       ? "so two rows ARE produced and row 1 is misplaced"
						       : "so row 1 is NOT produced at all");
					}
					for (size_t j = 0; j < nx; j++)
						X[j] = (float)(((j * 2654435761u) >> 9)
							       & 0xff) / 255.0f - 0.5f;
				}

				printf("\n    CNA geometry against the int8"
				       " stream, one word at a time\n");
				setenv("CHARSIU_BATCH_CNADIFF", "-1", 1);
				charsiu_npu_matmul(s->pool.dev, s->pool.id[i0], X, 2, Y);
				for (int q = 0; q < 4; q++) {
					char b[8];
					unsigned ok0 = 0, ok1 = 0, well = 0;

					snprintf(b, sizeof(b), "%d", q);
					setenv("CHARSIU_BATCH_CNADIFF", b, 1);
					if (charsiu_npu_matmul(s->pool.dev,
						s->pool.id[i0], X, 2, Y))
						continue;
					for (unsigned j = 0; j < (unsigned)t->n; j++) {
						double w0 = Yref[j];
						double w1 = Yref[t->n + j];

						if (fabs(Y[j] - w0) <= (fabs(w0) > 1e-3 ? fabs(w0)*1e-3 : 1e-3)) ok0++;
						if (fabs(Y[t->n+j] - w1) <= (fabs(w1) > 1e-3 ? fabs(w1)*1e-3 : 1e-3)) ok1++;
					}
					/*
					 * ⚠⚠ A HEALTH CHECK BETWEEN STEPS, and
					 * this sweep exists because there was
					 * not one. One override faulted the
					 * IOMMU, the reset left the block with
					 * MMU_DTE_ADDR not functioning, and the
					 * seven steps after it returned the
					 * same two numbers off a corpse. A
					 * sweep without a liveness check is a
					 * sweep that reports its first crash
					 * seven more times.
					 */
					unsetenv("CHARSIU_BATCH_CNADIFF");
					charsiu_act_set(&a, X, (int)t->k);
					charsiu_act_blocks(&a);
					if (charsiu_npu_needs_q1(s->pool.dev))
						charsiu_act_q1(&a);
					if (!charsiu_npu_matvec(s->pool.dev,
						s->pool.id[i0], &a, Y))
						for (unsigned j = 0; j < (unsigned)t->n; j++) {
							double w = Yref[j];

							if (fabs(Y[j] - w) <= (fabs(w) > 1e-3 ? fabs(w)*1e-3 : 1e-3)) well++;
						}
					printf("      cnadiff %d   row0 %5u/%-5u"
					       "   row1 %5u/%-5u   alive %u/%u%s\n", q,
					       ok0, (unsigned)t->n,
					       ok1, (unsigned)t->n,
					       well, (unsigned)t->n,
					       well < t->n ? "   ⚠ THE BLOCK IS GONE, stop reading here"
					       : ok1 > 100 ? "   <== rows appear" : "");
					if (well < t->n)
						break;
				}
				unsetenv("CHARSIU_BATCH_CNADIFF");
				unsetenv("CHARSIU_BATCH_READ");
			}
		}
	}

	for (unsigned y = 0; y < n_ms; y++) {
		unsigned mr = MS[y], tested = 0, rows_ok = 0, rows_tot = 0;
		unsigned seen = 0;		/* staged tensors reached, capped */
		unsigned nbad = 0;		/* misses named, capped */
		int whered = 0;			/* the where-did-it-go scan, once a width */
		double t_one = 0.0, t_bat = 0.0, worst = 0.0, mb = 0.0;

		if (mr > mmax)
			break;
		{
			double z;

			charsiu_npu_batch_split(s->pool.dev, &z, &z, &z, &z, 1);
			charsiu_npu_batch_prep(s->pool.dev, 1);
			charsiu_npu_batch_alloc(s->pool.dev, NULL, 1);
		}
		for (unsigned i = 0; i < s->pool.n; i++) {
			const struct npu_tensor *t = &s->pool.t[i];
			size_t nx, ny;

			if (s->pool.id[i] < 0)
				continue;
			/*
			 * ⚠ COUNTED HERE AND NOT OFF `tested`, which is
			 * incremented at the BOTTOM of the body and is skipped
			 * by the `continue` a failed matmul takes. A cap read
			 * off it would let a width whose matmuls are all being
			 * refused walk the whole pool paying the one row
			 * reference for every tensor -- the single most
			 * expensive thing here, spent on a width that measured
			 * nothing.
			 */
			if (maxt && seen >= maxt)
				break;
			seen++;
			nx = (size_t)mr * t->k;
			ny = (size_t)mr * t->n;
			X = realloc(X, nx * sizeof(*X));
			Y = realloc(Y, ny * sizeof(*Y));
			Yref = realloc(Yref, ny * sizeof(*Yref));
			if (!X || !Y || !Yref)
				goto out;
			/* every row different, or a batch could be right by
			 * copying one row over the rest */
			for (size_t j = 0; j < nx; j++)
				X[j] = (float)(((j * 2654435761u) >> 9) & 0xff)
				     / 255.0f - 0.5f;

			{
				double t0 = now_ms();

				for (unsigned r = 0; r < mr; r++) {
					charsiu_act_set(&a, X + (size_t)r * t->k,
							(int)t->k);
					charsiu_act_blocks(&a);
					/*
					 * ⚠ int8's matvec READS a->q1 and the
					 * float path does not, so without this
					 * the reference came back all zeros on
					 * an int8 model -- and the run that
					 * finally produced both batched rows
					 * scored 0 of 224 against it.
					 */
					if (charsiu_npu_needs_q1(s->pool.dev))
						charsiu_act_q1(&a);
					charsiu_npu_matvec(s->pool.dev, s->pool.id[i],
						&a, Yref + (size_t)r * t->n);
				}
				t_one += now_ms() - t0;
			}
			{
				double t0 = now_ms();

				if (charsiu_npu_matmul(s->pool.dev, s->pool.id[i], X,
						       mr, Y))
					continue;
				t_bat += now_ms() - t0;
			}
			for (unsigned r = 0; r < mr; r++) {
				int ok = 1;
				double rworst = 0;

				for (unsigned j = 0; j < (unsigned)t->n; j++) {
					size_t o = (size_t)r * t->n + j;
					double d = fabs((double)Y[o]
							- (double)Yref[o]);
					double sc = fabs((double)Yref[o]);
					double rel = sc > 1e-3 ? d / sc : d;

					if (rel > worst)
						worst = rel;
					if (rel > rworst)
						rworst = rel;
					if (rel > 1e-3)
						ok = 0;
				}
				rows_ok += ok;
				rows_tot++;
				/*
				 * ⚠ NAME THE TENSOR AND THE ROW when a width
				 * that is otherwise exact loses a few. m = 8
				 * comes back 871 of 904 -- 33 rows, not a
				 * whole tensor and not a whole row of them --
				 * where 4 and 16 either side are perfect, and
				 * a count of 33 cannot say whether it is one
				 * shape, one row index, or scattered. Eight
				 * lines is enough to tell those apart.
				 *
				 * ⚠ FORTY, NOT EIGHT, AND THE ARITHMETIC IS
				 * WHY. 33 was written down as "every ffn_gate
				 * and ffn_up in the model, once each" and that
				 * is 32 -- there is a thirty third miss nobody
				 * has ever seen, because the cap stopped at
				 * eight. Whether it is the output head, whose
				 * slices are also 8192 wide, or something with
				 * no 8192 in it at all is the difference
				 * between two different rules, and one line
				 * settles it. Forty covers m = 8 whole and is
				 * still bounded if a width breaks outright.
				 */
				if (!ok && nbad < 40) {
					printf("      MISS %-24s k=%-5u n=%-5u"
					       " row %u of %u, worst rel %.2e\n",
					       t->name, (unsigned)t->k,
					       (unsigned)t->n, r, mr, rworst);
					nbad++;
				}
				/*
				 * ⚠⚠ ABSENT OR MISPLACED, ASKED OF THE ROW
				 * THAT MISSED. The count above says a row is
				 * wrong and cannot say why, and the two
				 * answers want different work.
				 *
				 * m = 8 loses row 0 of the n = 8192 tensors
				 * and nothing else, at that one width. The
				 * read order is a bijection there and takes no
				 * n, so it cannot be selective by n and the
				 * other seven rows being exact leaves row 0's
				 * values nowhere to be but its own slots. That
				 * is a deduction; this is the measurement it
				 * predicts, and it can come back either way:
				 *
				 *   absent      the block never wrote them and
				 *               no reading recovers them --
				 *               the surface is short
				 *   somewhere   they were written and put in
				 *               the wrong place, and the
				 *               deduction above is wrong
				 *
				 * ⚠ ONE ROW OF ONE TENSOR A WIDTH, AND ON A
				 * WORK BUDGET. Each wanted value is looked for
				 * in the whole batch, so the cost is
				 * (values scanned) * m * n. The whole row at
				 * 8 by 8192 is 5.4e8 and takes about a second;
				 * the whole row of the 128256 wide head at
				 * m = 80 would be 1.3e12 and never finish. So
				 * the budget fixes how many of the row's
				 * channels are asked about and the line says
				 * how many that was -- a smaller sample of a
				 * question that can still be answered beats a
				 * whole one that hangs the round.
				 *
				 * The zero count is one pass and is over the
				 * whole row either way.
				 */
				if (!ok && !whered) {
					size_t live = 0, zero = 0;
					size_t tot = (size_t)mr * t->n;
					size_t look = 1000000000u / (tot ? tot : 1);

					whered = 1;
					if (look < 64)
						look = 64;
					if (look > (size_t)t->n)
						look = (size_t)t->n;
					for (unsigned j = 0; j < (unsigned)t->n; j++)
						if (Y[(size_t)r * t->n + j] == 0.0f)
							zero++;
					for (size_t j = 0; j < look; j++) {
						double w = Yref[(size_t)r * t->n + j];
						double sc = fabs(w);

						for (size_t o = 0; o < tot; o++)
							if (fabs((double)Y[o] - w)
							    <= (sc > 1e-3 ? sc * 1e-3
									  : 1e-3)) {
								live++;
								break;
							}
					}
					printf("      row %u of %s at m=%u:"
					       " %zu of the first %zu wanted"
					       " values are somewhere in the"
					       " batch, and %zu of the row's"
					       " %u slots came back exactly"
					       " zero\n",
					       r, t->name, mr, live, look, zero,
					       (unsigned)t->n);
					printf("        wanted ");
					for (unsigned j = 0; j < 6 && j < t->n; j++)
						printf("%11.4g",
						       Yref[(size_t)r * t->n + j]);
					printf("\n        got    ");
					for (unsigned j = 0; j < 6 && j < t->n; j++)
						printf("%11.4g",
						       Y[(size_t)r * t->n + j]);
					printf("\n");
				}
			}
			/*
			 * ⚠⚠ WRONG PLACE OR WRONG NUMBER, which are different
			 * faults and the row count above cannot tell apart.
			 *
			 * This is the question that cracked the accumulator's
			 * layout twice: are the wanted values IN the buffer at
			 * all? If they are, the arithmetic is right and only
			 * the order is wrong, and an order is something a map
			 * can be read for. If they are not, the order is
			 * irrelevant until the numbers are fixed.
			 *
			 * Only the first tensor, and only at the narrowest m,
			 * because it is O(n^2) in the row.
			 */
			if (!tested && mr == MS[0]) {
				size_t live = 0, tot = (size_t)mr * t->n;

				for (size_t q = 0; q < tot; q++)
					for (size_t o = 0; o < tot; o++) {
						double d = fabs((double)Y[o]
							  - (double)Yref[q]);
						double sc = fabs((double)Yref[q]);

						if (d <= (sc > 1e-3 ? sc * 1e-3
								    : 1e-3)) {
							live++;
							break;
						}
					}
				printf("  %s at m=%u: %zu of %zu wanted values"
				       " are somewhere in the batch\n",
				       t->name, mr, live, tot);
				/*
				 * ⚠ BOTH ROWS OF BOTH PATHS. Row 0 came back
				 * exact and the run still agreed on nothing,
				 * so the question is what row 1 is: a copy of
				 * row 0 means the activation was packed once,
				 * and numbers belonging to neither row mean it
				 * was computed wrong.
				 */
				for (unsigned r = 0; r < mr && r < 3; r++) {
					printf("    row%u batched  ", r);
					for (unsigned j = 0; j < 6 && j < t->n; j++)
						printf("%9.3f",
						       Y[(size_t)r * t->n + j]);
					printf("\n    row%u one row  ", r);
					for (unsigned j = 0; j < 6 && j < t->n; j++)
						printf("%9.3f",
						       Yref[(size_t)r * t->n + j]);
					printf("\n");
				}
				/*
				 * ⚠ AND WHERE THE MISSING ONES ARE. 74% present
				 * with row 0 exact says the loss is not spread
				 * evenly, and a per row count says which row
				 * lost them.
				 */
				for (unsigned r = 0; r < mr && r < 4; r++) {
					size_t have = 0;

					for (unsigned j = 0; j < (unsigned)t->n; j++) {
						double w = Yref[(size_t)r * t->n + j];
						double lim = fabs(w) > 1e-3
							   ? fabs(w) * 1e-3 : 1e-3;

						for (size_t o = 0; o < tot; o++)
							if (fabs((double)Y[o] - w) <= lim) {
								have++;
								break;
							}
					}
					printf("    row%u: %zu of %u of its values"
					       " are in the batch\n",
					       r, have, (unsigned)t->n);
				}
				/*
				 * ⚠⚠ AND WHERE EACH ONE LANDED, which is the
				 * permutation itself rather than a count of it.
				 *
				 * This is what npu_gemm_test --read does for
				 * the int8 accumulator, and it is how that
				 * layout was solved twice. There is no
				 * equivalent for w4a16: npu_gemm_test has no
				 * int4 at all and charsiu_int4 runs w4a8, so
				 * the runtime's own format has never had its
				 * output surface mapped at m > 1. This is that
				 * map, on the real path, with the real weights.
				 *
				 * ⚠ Y IS ALREADY READ THROUGH charsiu_acc_index.
				 * The table below is therefore the permutation
				 * that is LEFT after this tree's read order, so
				 * "landed at (r, c) itself" is what correct
				 * looks like and anything else is the residue.
				 *
				 * ⚠ THE WHOLE OF ROW 0, not its first six.
				 * Round 381 sampled six, saw 0 1 2 3 8 9, and
				 * had to explain afterwards why a layout that
				 * fit them scored 15 of 128. Six cannot say
				 * where a pattern stops.
				 *
				 * ⚠ AND HOW TRUSTWORTHY IT IS. Every row
				 * reports the FIRST slot holding the value, so
				 * a value the reference produces twice gives
				 * one answer out of two and reads like one out
				 * of one. The distinct count is printed first
				 * for exactly that reason.
				 */
				/*
				 * ⚠⚠ POSITION BY POSITION FIRST, because the
				 * search below is fuzzy and this is not.
				 *
				 * "Is the value somewhere in the batch" matches
				 * on 1e-3 relative, which for a value near zero
				 * is 1e-3 ABSOLUTE -- so every small output
				 * matches every other small output and both the
				 * present count and the landed-at column are
				 * inflated by it. The board says how much:
				 * this reference has 1456 unique values in
				 * 4096, and a table is only as good as that.
				 *
				 * Comparing Y[r][c] against Yref[r][c] has no
				 * such freedom. It cannot say where a value
				 * WENT, but it says exactly where a row stops
				 * being right, which is the question row 0 has
				 * been raising for three rounds by being exact
				 * in its first six channels and agreeing on no
				 * row at all.
				 */
				for (unsigned r = 0; r < mr && r < 4; r++) {
					size_t agree = 0;
					long first = -1;

					for (unsigned c = 0; c < (unsigned)t->n; c++) {
						double w = Yref[(size_t)r * t->n + c];
						double d = fabs((double)Y[(size_t)r * t->n + c] - w);
						double lim = fabs(w) > 1e-3
							   ? fabs(w) * 1e-3 : 1e-3;

						if (d <= lim)
							agree++;
						else if (first < 0)
							first = c;
					}
					printf("    row%u in place: %zu of %u channels agree",
					       r, agree, (unsigned)t->n);
					if (first < 0)
						printf(", the whole row\n");
					else
						printf(", first wrong at channel %ld\n",
						       first);
				}
				/*
				 * ⚠ AN OFFLINE SWEEP OF THE READ ORDER WAS
				 * WRITTEN HERE AND REMOVED, and the reason is
				 * worth keeping.
				 *
				 * It reconstructed the raw buffer from Y and
				 * scored candidate index functions against it
				 * without a board round. Y is not raw: the
				 * batched read is `yr[j] += fo[mp[j]] * sc[j]`
				 * and sc is PER OUTPUT CHANNEL, so a value
				 * scored at a different channel carries the
				 * wrong scale and the whole table is off by a
				 * ratio wherever a candidate crosses a group
				 * boundary. Nearly sound is exactly the kind of
				 * instrument this tree has been burned by.
				 *
				 * The family is two members anyway. Of a * A
				 * and the two halves swapped, only A = 4 and
				 * the swap are permutations at all -- 128 of
				 * 128 distinct slots against 68, 80, 96 and 64
				 * for A of 8, 1, 2 and 0 -- and A = 4 is the
				 * control. So CHARSIU_ACC_A picks between them
				 * and the board scores both exactly, through
				 * the real read path with the real scales.
				 */
				{
					size_t uniq = 0, q, o;
					/* ⚠ NOT A MULTIPLE OF 32, or every sampled
					 * channel has a = 0 and the half that is
					 * wrong is never looked at. The last round
					 * stepped by 64 and every one of its 32
					 * samples was in the good half. */
					unsigned step = t->n >= 128
						      ? (unsigned)t->n / 32 + 4 : 4;

					for (q = 0; q < tot; q++) {
						double w = Yref[q];
						double lim = fabs(w) > 1e-3
							   ? fabs(w) * 1e-3 : 1e-3;
						size_t same = 0;

						for (o = 0; o < tot; o++)
							if (fabs((double)Yref[o] - w) <= lim)
								same++;
						if (same == 1)
							uniq++;
					}
					printf("    the reference has %zu unique"
					       " values in %zu (the table is only"
					       " as good as that)\n", uniq, tot);
					printf("    %-10s %-12s %-12s %-6s %s\n",
					       "(row,ch)", "want", "landed at",
					       "hits", "correct is (row,ch) itself");
					/*
					 * ⚠ THE FIRST 64 CHANNELS IN FULL, then
					 * a coarse tail. The structure repeats
					 * every 32 -- a is (c%32)/16 and t is
					 * c%16 -- so one pair of super groups
					 * holds all of it, and the half that is
					 * WRONG is in there. A coarse sweep
					 * alone cannot solve a layout; that is
					 * how the last round sampled 32 channels
					 * and every one of them was in the good
					 * half.
					 */
					for (unsigned r = 0; r < mr && r < 2; r++) {
						unsigned st = r ? step * 4 : step;
						unsigned c;

						for (c = 0; c < (unsigned)t->n;
						     c = c < 64 ? c + 1 : c + st) {
							double w = Yref[(size_t)r * t->n + c];
							double lim = fabs(w) > 1e-3
								   ? fabs(w) * 1e-3 : 1e-3;
							size_t at = tot, hits = 0;

							for (o = 0; o < tot; o++)
								if (fabs((double)Y[o] - w) <= lim) {
									if (at == tot)
										at = o;
									hits++;
								}
							if (at == tot)
								printf("    (%u,%-5u) %12.4f %-12s %-6s\n",
								       r, c, w, "nowhere", "-");
							else
								printf("    (%u,%-5u) %12.4f (%zu,%-5zu) %-6zu%s\n",
								       r, c, w, at / t->n,
								       at % t->n, hits,
								       (at == (size_t)r * t->n + c)
								       ? "  <= correct" : "");
						}
					}
				}
			}
			/* what the hardware moved: the weights, once */
			mb += (double)t->k * t->n / 1e6;
			tested++;
		}
		if (!tested)
			continue;
		/*
		 * ⚠ GB/s, BECAUSE A SPEEDUP CANNOT SAY WHETHER THERE IS ROOM
		 * LEFT. 3.73x against a one row loop sounds finished; the same
		 * run at 1.3 GB/s against a 9.5 GB/s hardware path says most of
		 * the time is not the hardware at all.
		 */
		{
			double pk, sb, fn, rd, pr, al;
			unsigned an = 0;
			char tcol[24];

			charsiu_npu_batch_split(s->pool.dev, &pk, &sb, &fn, &rd, 1);
			pr = charsiu_npu_batch_prep(s->pool.dev, 1);
			al = charsiu_npu_batch_alloc(s->pool.dev, &an, 1);
			/*
			 * ⚠ AND WHAT IS STILL MISSING. The five segments are
			 * printed with the remainder beside them, because the
			 * four of them came to 451 ms of a 606 ms matmul and
			 * nobody noticed until the shares were added up by
			 * hand. A breakdown that does not say what it failed to
			 * account for is an invitation to optimise the wrong
			 * third.
			 */
			/*
			 * ⚠ THE TENSOR COUNT IS A FRACTION, ALWAYS. An
			 * uncapped row reads 225/225 and a capped one 8/225,
			 * so the two can never be confused for each other by
			 * anyone reading a single pasted line.
			 */
			snprintf(tcol, sizeof(tcol), "%u/%u", tested, n_staged);
			printf("  %3u  %9s  %10.2e  %6u of %-6u  %7.0f ms"
			       " %7.0f ms  %5.2fx  %7.1f  %6.2f"
			       "   prep %4.0f (alloc %4.0f x%u)  pack %4.0f"
			       "  submit %3.0f  fence %5.0f  read %4.0f"
			       "  rest %4.0f\n",
			       mr, tcol, worst, rows_ok, rows_tot, t_one,
			       t_bat, t_bat > 0 ? t_one / t_bat : 0.0,
			       t_bat * 1e3 / (tested * (double)mr),
			       t_bat > 0 ? mb / t_bat : 0.0,
			       pr, al, an, pk, sb, fn, rd,
			       t_bat - (pr + pk + sb + fn + rd));
		}
		rc = 0;
	}
	printf("\n  ⚠ a speed with rows that do not agree is the speed of a"
	       " wrong answer.\n  the bar is relative and 1e-3: these are float"
	       " sums of thousands of\n  terms in two orders and will not be bit"
	       " identical.\n");
out:
	charsiu_act_free(&a);
	free(X); free(Y); free(Yref);
	return rc;
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

static int npu_mode(void)
{
	static int m = -1;

	if (m < 0) {
		const char *e = getenv("CHARSIU_NPU_QUANT");

		m = e && *e != '0';
	}
	return m;
}

/*
 * ⚠ THE BODY OF THIS MOVED TO src/npupool.c, so that a graph which is not the
 * language model can stage a weight the same way. What is left is the shape the
 * rest of this file calls it in.
 */
static const struct npu_tensor *npu_get(struct llama_state *s,
					const struct gguf_tensor *w)
{
	return charsiu_pool_get(&s->pool, w);
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

	if (s->pool.dev && !group_off() && npu_mode() && s->act.npu_ok) {
		/* int8 takes q1; int4 takes the float and never looks */
		if (charsiu_npu_needs_q1(s->pool.dev))
			act_q1_timed(&s->act);
		for (i = 0; i < n; i++) {
			nt[i] = npu_get(s, w[i]);
			ids[i] = -1;
			if (nt[i])
				for (unsigned j = 0; j < s->pool.n; j++)
					if (&s->pool.t[j] == nt[i]) {
						ids[i] = s->pool.id[j];
						break;
					}
			if (ids[i] < 0)
				break;
		}
		if (i == n &&
		    !charsiu_npu_matvec_group(s->pool.dev, ids, n, &s->act, y))
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

		for (unsigned i = 0; i < s->pool.n; i++)
			if (&s->pool.t[i] == nt) {
				id = s->pool.id[i];
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
		if (id >= 0 && charsiu_npu_needs_q1(s->pool.dev))
			act_q1_timed(a);
		if (id >= 0 && !charsiu_npu_matvec(s->pool.dev, id, a, y)) {
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
	 * ⚠ NO GAIN IS A REAL CASE, not a missing argument. gemma4 normalises
	 * V with a bare RMS and no weight at all, so a NULL here means one
	 * rather than nothing.
	 */
	if (!g) {
		for (uint32_t h = 0; h < nheads; h++) {
			float *p = v + (size_t)h * hd;
			float ss = 0.0f, sc;

			for (uint32_t i = 0; i < hd; i++)
				ss += p[i] * p[i];
			sc = 1.0f / sqrtf(ss / (float)hd + eps);
			for (uint32_t i = 0; i < hd; i++)
				p[i] *= sc;
		}
		return;
	}
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
	    strcmp(arch, "gemma4") &&
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
	snprintf(key, sizeof(key), "%s.feed_forward_length", arch);
	{
		const struct gguf_kv *ff = gguf_find(&m->gguf, key);

		if (ff && ff->type == GGUF_V_ARRAY &&
		    ff->arr_len >= m->n_layer) {
			uint32_t i32, mx = 0;

			m->ff_arr = ff;
			/*
			 * ⚠ THE LARGEST, not the first. Every buffer
			 * below is sized from m->n_ff and the widest
			 * layer is the one that has to fit; taking
			 * arr[0] gives 6144 where a later layer wants
			 * 12288, and the overrun is silent.
			 */
			for (uint32_t q = 0; q < m->n_layer; q++) {
				memcpy(&i32, (const uint8_t *)ff->arr
					     + (size_t)q * 4, 4);
				if (i32 > mx)
					mx = i32;
			}
			if (mx)
				m->n_ff = mx;
		}
	}
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
	m->head_dim_swa = m->head_dim;
	m->attn_scale = 1.0f / sqrtf((float)m->head_dim);
	m->n_layer_kv = m->n_layer;
	/*
	 * ⚠ BY ARCHITECTURE, and there is no key in the file that says it --
	 * llama.cpp carries the same thing as a switch over the architecture
	 * enum. llama and smollm3 are the permuted, interleaved ones; qwen2,
	 * qwen3 and phi3 are not.
	 */
	/*
	 * ⚠ EVERY BUFFER TAKES THE WIDER HEAD. gemma4's window layers are 256
	 * where its full ones are 512, and q, the KV cache and xb are one
	 * allocation each for all of them.
	 */
	if (m->head_dim_swa > m->head_dim)
		m->head_dim = m->head_dim_swa;
	m->n_embd_attn = m->n_head * m->head_dim;

	m->rope_neox = !strcmp(arch, "qwen2") || !strcmp(arch, "qwen3") ||
		       !strcmp(arch, "gemma3") || !strcmp(arch, "gemma4") ||
		       !strcmp(arch, "phi3");
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
	/*
	 * ⚠⚠ gemma4 WRITES THE SAME KEY AS AN ARRAY, one flag a layer, where
	 * gemma3 writes a scalar period. llama.cpp reads it with
	 * get_key_or_arr into is_swa_impl[] for exactly that reason.
	 *
	 * A scalar period cannot express what gemma4 does -- its pattern is
	 * not periodic -- so read the array where there is one and fall back
	 * to the period where there is not. The per layer flag is what the
	 * forward pass actually asks, so both forms end up in the same place.
	 */
	{
		const struct gguf_kv *sw;

		snprintf(key, sizeof(key), "%s.attention.sliding_window_pattern",
			 arch);
		sw = gguf_find(&m->gguf, key);
		if (sw && sw->type == GGUF_V_ARRAY && sw->arr_len >= m->n_layer)
			m->swa_arr = sw;
	}
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
	if (!strcmp(arch, "gemma4")) {
		uint32_t v2;

		/*
		 * ⚠ gemma4 SETS THE ATTENTION SCALE TO ONE. Its python is
		 * `self.scaling = 1.0`; the factor that would be 1/sqrt(head)
		 * is folded into the QK norms instead. Leaving 1/sqrt(head)
		 * here does not crash and does not read as wrong -- it
		 * flattens every softmax in the model by a constant.
		 */
		m->attn_scale = 1.0f;
		/*
		 * ⚠ THE KEY IS embedding_length_per_layer_INPUT. llama.cpp's
		 * LLM_KV_EMBEDDING_LENGTH_PER_LAYER renders to that string, and
		 * the shorter name it reads like is in no file.
		 */
		snprintf(key, sizeof(key),
			 "%s.embedding_length_per_layer_input", arch);
		if (gguf_get_u32(&m->gguf, key, &v2))
			v2 = 0;
		m->n_embd_pl = v2;
		/*
		 * ⚠ A WINDOW LAYER MAY HAVE A SHORTER HEAD. gemma4 declares
		 * attention.key_length_swa on its own, and the KV cache has to
		 * be sized for whichever of the two is larger.
		 */
		snprintf(key, sizeof(key), "%s.attention.key_length_swa", arch);
		if (gguf_get_u32(&m->gguf, key, &v2) || !v2 || (v2 & 1))
			v2 = m->head_dim;
		m->head_dim_swa = v2;
		/*
		 * ⚠ THE LAYERS PAST THIS SHARE AN EARLIER LAYER'S KV, and have
		 * no wk or wv of their own. The key counts the SHARED ones, so
		 * the first that shares is n_layer minus it.
		 */
		snprintf(key, sizeof(key), "%s.attention.shared_kv_layers", arch);
		if (gguf_get_u32(&m->gguf, key, &v2))
			v2 = 0;
		m->n_layer_kv = v2 < m->n_layer ? m->n_layer - v2 : m->n_layer;
		m->v_norm = 1;
	}
	if (!strcmp(arch, "gemma3") || !strcmp(arch, "gemma4")) {
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

	/*
	 * gemma4's per layer embedding tables. Absent everywhere else, and
	 * absent on a gemma4 that declares no per layer width.
	 */
	if (m->n_embd_pl) {
		m->pl_tok_embd = need(&m->gguf, "per_layer_token_embd.weight");
		m->pl_model_proj = need(&m->gguf, "per_layer_model_proj.weight");
		m->pl_proj_norm = need(&m->gguf, "per_layer_proj_norm.weight");
		if (!m->pl_tok_embd || !m->pl_model_proj || !m->pl_proj_norm)
			goto fail;
	}

	m->layers = calloc(m->n_layer, sizeof(*m->layers));
	if (!m->layers)
		goto fail;
	int last_kv[2] = { -1, -1 };   /* [0] full attention, [1] window */

	for (uint32_t l = 0; l < m->n_layer; l++) {
		struct llama_layer *L = &m->layers[l];

		/*
		 * ⚠ RESOLVE THE WINDOW FLAG ONCE, HERE. The forward pass asks
		 * it every layer of every token and there are two ways a file
		 * can say it; deciding in the loop would put the two forms in
		 * the hot path and in two places.
		 */
		if (m->swa_arr) {
			const uint8_t *a = m->swa_arr->arr;
			unsigned w = m->swa_arr->arr_type == GGUF_V_U32 ||
				     m->swa_arr->arr_type == GGUF_V_I32 ? 4 :
				     m->swa_arr->arr_type == GGUF_V_U16 ||
				     m->swa_arr->arr_type == GGUF_V_I16 ? 2 : 1;
			uint32_t v32 = 0;

			memcpy(&v32, a + (size_t)l * w, w);
			L->swa = m->n_swa && v32 != 0;
		} else {
			L->swa = m->swa_pattern &&
				 (l % m->swa_pattern) < m->swa_pattern - 1;
		}
		/*
		 * ⚠ WHOSE KV THIS LAYER READS. gemma4's last few layers carry
		 * no wk or wv and attend against the last layer that had them.
		 * -1 is "its own", which is every layer of everything else.
		 */
		L->kv_from = -1;
		/*
		 * ⚠ THE LAYER'S OWN SHAPES, resolved here so the forward pass
		 * reads a field instead of a rule. Everything before gemma4
		 * takes the model's, which is what the fallback is.
		 */
		L->n_ff = m->n_ff;
		if (m->ff_arr) {
			uint32_t v32 = 0;

			memcpy(&v32, (const uint8_t *)m->ff_arr->arr
				     + (size_t)l * 4, 4);
			if (v32)
				L->n_ff = v32;
		}
		L->head_dim = L->swa ? m->head_dim_swa : m->head_dim;

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
		} else if (!strcmp(arch, "gemma4")) {
			/*
			 * ⚠ ONLY wq IS ALWAYS THERE.
			 *
			 * gemma4's last layers share an earlier layer's KV and
			 * carry no attn_k at all; and attn_v is optional in
			 * EVERY layer, where its absence means V is K -- which
			 * llama.cpp spells `Vcur = Kcur` rather than as a
			 * separate projection.
			 */
			T(wq, "attn_q");
			snprintf(key, sizeof(key), "blk.%u.attn_k.weight", l);
			L->wk = gguf_tensor(&m->gguf, key);
			if (L->wk && !L->wk->nbytes)
				L->wk = NULL;
			snprintf(key, sizeof(key), "blk.%u.attn_v.weight", l);
			L->wv = gguf_tensor(&m->gguf, key);
			if (L->wv && !L->wv->nbytes)
				L->wv = NULL;
			/*
			 * ⚠⚠ THE LAST LAYER OF THE SAME KIND, not simply the
			 * last one.
			 *
			 * gemma4 keeps two caches, one for its window layers
			 * and one for its full ones -- llama.cpp builds it with
			 * build_attn_inp_kv_iswa -- and they are not even the
			 * same shape here: a window layer's head is 256 and a
			 * full one's is 512. E2B's first shared layer is 15,
			 * which is a WINDOW layer, and the last layer with a
			 * KV of its own is 14, which is a FULL one. Pointing
			 * the first at the second has it read 256 floats out
			 * of a slot written as 512, which is not a crash and
			 * not obviously wrong in the residual stream: the
			 * norms stayed between 0.75 and 2 the whole way down.
			 */
			if (L->wk) {
				last_kv[L->swa ? 1 : 0] = (int)l;
			} else if (last_kv[L->swa ? 1 : 0] < 0) {
				fprintf(stderr, "llama: layer %u shares a %s KV "
					"and no earlier layer has one\n", l,
					L->swa ? "window" : "full");
				goto fail;
			} else {
				L->kv_from = last_kv[L->swa ? 1 : 0];
			}
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
		/*
		 * ⚠ gemma4 only, and the names are SHORTER than the model wide
		 * ones they belong to: per_layer_token_embd and
		 * per_layer_model_proj sit at the top level, but a layer's
		 * three are blk.N.inp_gate, blk.N.proj and blk.N.post_norm.
		 * Guessing per_layer_* here found nothing, three times, and a
		 * tensor that is not found is simply not used -- the whole
		 * per-layer path was skipped and the model answered in a
		 * different language every token.
		 */
		OW(pl_inp_gate, "inp_gate");
		OW(pl_proj, "proj");
		OW(pl_post_norm, "post_norm");
		OW(out_scale, "layer_output_scale");
		OW(rope_freqs, "rope_freqs");
#undef OW
		/*
		 * ⚠ EXCEPT WHERE THERE IS NO K. gemma4's shared KV layers carry
		 * neither attn_k nor attn_k_norm, and layer 15 of E2B is the
		 * first of them, so the symmetry this checks is the wrong
		 * symmetry there: it is q_norm against a K THAT EXISTS.
		 */
		if (L->wk && !L->q_norm != !L->k_norm) {
			fprintf(stderr, "llama: layer %u norms one of q and k "
				"and not the other\n", l);
			goto fail;
		}
		/*
		 * ⚠ REFUSE A MIXTURE OF EXPERTS RATHER THAN COMPUTE HALF OF IT.
		 *
		 * A gemma4 MoE layer runs a dense MLP and an expert branch in
		 * PARALLEL and adds them. The dense half is the nine weights
		 * this graph already knows, so loading it and ignoring
		 * ffn_gate_inp would produce a model that runs, answers, and
		 * is quietly missing half of every expert layer. gemma-4-26B
		 * -A4B is the one that has them; E2B and E4B are dense.
		 */
		snprintf(key, sizeof(key), "blk.%u.ffn_gate_inp.weight", l);
		if (gguf_tensor(&m->gguf, key)) {
			fprintf(stderr, "llama: layer %u is a mixture of "
				"experts, which this graph does not build\n", l);
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

/*
 * ⚠⚠ A WIDER K SLICE IS FREE ON SOME MODELS AND BUYS QUALITY ON OTHERS, and
 * which one a model is can be read off two integers.
 *
 * The read back is m * n * ks and ks is ceil(K / KMAX), so a wider slice is
 * directly less work -- the board measured KMAX 4096 taking Phi-3.5's read from
 * 11776 ms to 3484. But npudev's own note closes the free version of it: ONE
 * DISPATCH CANNOT COVER K WIDER THAN ONE QUANTISATION GROUP, so KMAX and
 * CHARSIU_NPU_W4_GROUP move together, and moving them changes how the weights
 * were quantised.
 *
 * ⚠ EXCEPT WHERE THEY WERE NEVER GROUPED. npuquant falls back to one scale a
 * row when K % group is non-zero. A model whose every K misses every candidate
 * width is on that per-row path at all of them, so widening changes the
 * SLICING and not one weight -- and the board says so: gemma-3-1b (1152, 6912)
 * came back byte identical at 1024, 2048 and 4096 on three prompts, while
 * Phi-3.5 (3072, 8192) and SmolLM2-1.7B (2048, 8192) degraded, the latter to
 * "cold.  .  .  .  ." at 4096.
 *
 * ⚠ AND THE COUNTING PROMPT COULD NOT SEE ANY OF THAT. An earlier round swept
 * KMAX with "1 2 3 ... 256" and reported text identical on all eight models,
 * because continuing a count is the least quantisation sensitive thing a
 * language model does. The degradation above was found only once the probe
 * asked something else.
 *
 * So: widen only when NO tensor's grouping changes, which for a model that is
 * ungrouped at the baseline means it is ungrouped everywhere. Anything else
 * keeps 1024 and can still be overridden by hand.
 */
static void llama_auto_kmax(const struct llama_model *m)
{
	/*
	 * ⚠ 2048 ONLY, AND 4096 IS DELIBERATELY NOT HERE. Phase 13 put the
	 * batched path against the model's own token loop at 1024, 2048, 3072
	 * and 4096, on the three models whose K divides none of them -- so the
	 * quantiser emits the same bytes across the sweep and only the slicing
	 * moves. 2048 agrees on all three -- and it was never broken. A commit
	 * once credited a CBUF fix for it; that fix was in a function only a
	 * dump tool calls, job.c already had the rule, and 2048 had simply
	 * never been compared against the token loop before. Above it:
	 *
	 *   slice 2816   WRONG   (Qwen2.5 at KMAX 3072, surf 88)
	 *   slice 3072   right   (gemma-3-1b at KMAX 3072, surf 96)
	 *   slice 4096   WRONG   (both, surf 128)
	 *
	 * which is not a size threshold -- 3072 is larger than 2816 and works.
	 * Something else is wrong above 2048 and it has not been found, so the
	 * candidate list stops at the width the board has actually verified.
	 * CHARSIU_NPU_KMAX by hand still reaches anything.
	 *
	 * 2048 is also what the vendor uses for 81% of its own int4 dispatches.
	 *
	 * ⚠⚠ AND IT IS THE OPTIMUM, NOT A COMPROMISE, under the surface ceiling
	 * npudev enforces: (slice / 32) * m <= 5120, so a wider slice buys
	 * itself a narrower chunk. Scored on Qwen2.5-1.5B, where read work is
	 * sum over tensors of ceil(K / KMAX) * n and does not depend on m:
	 *
	 *   KMAX    max m    read      calls at that m
	 *   1024     160     100%      1.00x
	 *   2048      80      51%      1.00x
	 *   3072      53      46%      1.51x
	 *   4096      40      46%      2.00x
	 *
	 * Nearly all of the read is bought going 1024 -> 2048, and it is the
	 * last width that still allows a chunk of 80. Against the board's own
	 * measurement -- 19109 ms of batched matmul, 4925 read, 8003 fence, of
	 * which about 30% is the per call term -- 4096 at m = 40 saves 266 ms
	 * of read and pays about 2400 ms of call overhead for it.
	 *
	 * So this axis is finished. More would need the surface ceiling lifted,
	 * and that is the third CBUF window state nothing on disk has shown.
	 */
	static const unsigned cand[] = { 2048 };
	unsigned base = 1024, i, j;
	uint64_t k[8];
	char buf[16];
	unsigned nk = 0;

	if (getenv("CHARSIU_NPU_KMAX") || getenv("CHARSIU_NPU_W4_GROUP"))
		return;                 /* asked for by hand, leave it alone */

	/*
	 * ⚠⚠ SET THE BASELINE FIRST, ALWAYS, BEFORE DECIDING ANYTHING. Falling
	 * out of this function without setting them does NOT leave 1024: the
	 * code defaults are CHARSIU_NPU_KMAX 4096 in npudev.c and, in
	 * npuquant.c, a group of k -- one absmax over a whole row. No board
	 * round has ever run that pair; every probe pinned 1024/1024, which is
	 * exactly why nobody noticed the defaults had drifted away from it.
	 *
	 * Unpinning the probes and shipping this function together turned that
	 * into eight models of nine answering wrongly, and the six that this
	 * function DECLINED to widen were the six that got the untouched
	 * defaults. A function that decides not to act still has to say so.
	 */
	setenv("CHARSIU_NPU_KMAX", "1024", 1);
	setenv("CHARSIU_NPU_W4_GROUP", "1024", 1);

	/*
	 * ⚠ THE WIDENING IS ON, AND WHAT TURNED IT ON WAS ONE PHASE. Phase 13
	 * compares the batched path against the model's own TOKEN LOOP at each
	 * width -- phases 10 and 12 compare batched against batched and are
	 * blind to a batched-path fault by construction, which is how a wide
	 * slice was called "identical" on three models and shipped wrong.
	 * CHARSIU_NPU_KMAX_AUTO=0 turns it off.
	 *
	 * The reasoning below is sound as far as the QUANTISER goes and the
	 * board agreed with it: at 1024, 2048 and 4096 the three models whose
	 * every K misses every width came back byte identical on three prompts.
	 * But that comparison, and the KMAX sweep in phase 10, put a BATCHED
	 * run against another BATCHED run. Phase 2 puts the batched path
	 * against the model's own token loop, and there Qwen2.5 and gemma-3-1b
	 * DISAGREE at 4096 -- both of them models this function had cleared,
	 * and both of them models phase 12 had called identical.
	 *
	 * So the weights are fine and something in the batched path is not, at
	 * a K slice wider than 1024. Until that is found, the widest safe slice
	 * is the one the board has always run.
	 */
	{
		const char *a = getenv("CHARSIU_NPU_KMAX_AUTO");

		if (a && *a == '0')
			return;
	}

	k[nk++] = m->n_embd;
	if (m->n_ff)
		k[nk++] = m->n_ff;
	/*
	 * ⚠⚠ EVERY LAYER'S OWN WIDTH, NOT THE MODEL'S. gemma4 gives each layer
	 * its own feed_forward_length -- 6144 for its first fifteen and 12288
	 * after -- and the model wide n_ff is only the fallback. Checking that
	 * one number would clear a model whose per layer widths include a
	 * grouped one, and this decision is only safe when it has seen every K
	 * that will be staged.
	 */
	if (m->layers)
		for (i = 0; i < m->n_layer; i++) {
			uint64_t f = m->layers[i].n_ff ? m->layers[i].n_ff
						       : m->n_ff;
			int seen = 0;

			for (j = 0; j < nk; j++)
				if (k[j] == f)
					seen = 1;
			if (seen || !f)
				continue;
			if (nk == sizeof(k) / sizeof(k[0]))
				return;   /* more widths than room: do not guess */
			k[nk++] = f;
		}

	for (i = 0; i < sizeof(cand) / sizeof(cand[0]); i++) {
		int safe = 1;

		for (j = 0; j < nk; j++) {
			/*
			 * Grouped at the baseline, or grouped at the candidate:
			 * either way the scale layout is not what it was.
			 */
			if (k[j] % base == 0 || k[j] % cand[i] == 0)
				safe = 0;
		}
		if (!safe)
			continue;
		snprintf(buf, sizeof(buf), "%u", cand[i]);
		setenv("CHARSIU_NPU_KMAX", buf, 1);
		setenv("CHARSIU_NPU_W4_GROUP", buf, 1);
		if (charsiu_diag())
			fprintf(stderr, "charsiu: K slices at %u -- no tensor "
				"of this model is grouped at any candidate "
				"width, so the weights are unchanged\n",
				cand[i]);
		return;
	}
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
	/*
	 * gemma4's per layer embeddings: one slice a layer, built once per
	 * token, plus two scratch vectors of the per layer width.
	 */
	if (m->n_embd_pl) {
		s->pl = calloc((size_t)m->n_embd_pl * m->n_layer, sizeof(float));
		s->plb = calloc((size_t)m->n_embd_pl * m->n_layer, sizeof(float));
		s->plc = calloc(m->n_embd_pl, sizeof(float));
		if (!s->pl || !s->plb || !s->plc) {
			llama_state_free(s);
			return NULL;
		}
	}

	{
		uint32_t widest = state_widest(m);

		if (charsiu_act_alloc(&s->act, (int)widest) < 0) {
			llama_state_free(s);
			return NULL;
		}
	}

	/* nine per layer plus the output head */
	s->pool.cap = m->n_layer * 9 + 2;
	s->pool.t = calloc(s->pool.cap, sizeof(*s->pool.t));
	s->pool.key = calloc(s->pool.cap, sizeof(*s->pool.key));
	s->pool.src = calloc(s->pool.cap, sizeof(*s->pool.src));
	s->pool.id = calloc(s->pool.cap, sizeof(*s->pool.id));
	if (!s->pool.t || !s->pool.key || !s->pool.src || !s->pool.id) {
		llama_state_free(s);
		return NULL;
	}

	if (getenv("CHARSIU_NPU")) {
		const char *e = getenv("CHARSIU_NPU_MAXN");
		unsigned maxn = e ? (unsigned)atoi(e) : 8192;
		unsigned widest = state_widest(m);

		if (maxn > m->n_vocab)
			maxn = m->n_vocab;
		llama_auto_kmax(m);
		s->pool.dev = charsiu_npu_open(widest, maxn, s->pool.cap);
		if (!s->pool.dev) {
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
	free(s->pl);
	free(s->plb);
	free(s->plc);
	/*
	 * ⚠ THE BATCHED PREFILL'S ROWS WERE NEVER FREED. They are allocated
	 * lazily on the first prompt that takes that path and nothing here ever
	 * mentioned them -- six buffers, and at a 32 row chunk of a wide feed
	 * forward that is megabytes a state. It never showed because a run
	 * makes one state and then exits, which is the kind of leak that waits
	 * for the server.
	 */
	free(s->bx); free(s->bxb); free(s->bxo);
	free(s->bhb); free(s->bhb2); free(s->bcs);
	free(s->bq); free(s->bk); free(s->bv); free(s->bao);
	free(s->bfreq);
	free(s->bpl); free(s->bplg);

	free(s->att); free(s->logits);
	charsiu_act_free(&s->act);
	if (s->pool.t && s->pool.n && getenv("CHARSIU_NPU_REPORT"))
		npu_report(s->pool.t, s->pool.n);
	/*
	 * The calibration dump: one record a tensor, name then k then the sum
	 * of |x| over every token the run saw. Written here because this is the
	 * only place that knows the run has finished.
	 */
	if (getenv("CHARSIU_CALIB") && s->pool.t) {
		FILE *f = fopen(getenv("CHARSIU_CALIB"), "wb");
		unsigned wrote = 0, xwrote = 0;

		for (unsigned i = 0; f && i < s->pool.n; i++) {
			struct npu_tensor *t = &s->pool.t[i];

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
	if (s->pool.dev) {
		charsiu_npu_report(s->pool.dev);
		/*
		 * ⚠ llama had NO batched breakdown at all. The five counters
		 * behind this have existed since they were written with no
		 * caller anywhere, and vision and whisper reach them only
		 * through charsiu_pool_report, which nothing on this path
		 * calls. It self-suppresses when no prompt took the batched
		 * path, so a decode-only run prints nothing new.
		 */
		charsiu_pool_report_batch(&s->pool, stderr);
	}
	charsiu_npu_close(s->pool.dev);
	if (s->pool.t) {
		for (unsigned i = 0; i < s->pool.n; i++)
			npu_tensor_free(&s->pool.t[i]);
		free(s->pool.t);
	}
	free(s->pool.key);
	free(s->pool.src);
	free(s->pool.id);
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
/*
 * ⚠ THE BATCHED PROMPT HAD NO STAGES. The fourteen above are the token
 * loop's; batch_layers, which is where every prompt token goes, was never
 * instrumented, so on a board where Qwen3's prompt takes 1298 ms and its
 * batched matmul entry about 300 of them, nothing had ever said where the
 * other thousand went. Kept apart from the token loop's numbers because a
 * row of a batch and a decoded token do not cost the same thing.
 */
static double bstage_ms[ST_N];
static unsigned bstage_rows, bstage_chunks;

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

	if (stage_on <= 0 || (!stage_tok && !bstage_rows))
		return;
	if (bstage_rows) {
		double bt = 0;

		for (i = 0; i < ST_N; i++)
			bt += bstage_ms[i];
		printf("charsiu batched stages: %u rows in %u chunks, %.2f ms a row"
		       " (%.0f ms)\n", bstage_rows, bstage_chunks,
		       bt / bstage_rows, bt);
		for (i = 0; i < ST_N; i++)
			if (bstage_ms[i] > 0.0)
				printf("  %-16s %8.2f ms a row     %5.1f%%\n",
				       stage_name[i], bstage_ms[i] / bstage_rows,
				       100.0 * bstage_ms[i] / bt);
		printf("  (\"residual\" after o proj carries the ffn rmsnorm too, and"
		       " \"rope + kv copy\" only the rope)\n");
	}
	if (!stage_tok)
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
	/*
	 * ⚠ hd IS THIS LAYER'S HEAD AND hdmax IS THE CACHE'S STRIDE, and on
	 * gemma4 they differ: its window layers have a 256 long head and its
	 * full ones 512, while the KV cache is one allocation with one stride
	 * for all of them. Indexing the cache by the live head would make
	 * layer 4 read where layer 0 wrote.
	 */
	uint32_t hd, hdmax, kvdim, gqa, nkv;
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
	uint32_t hd = j->hd, hdmax = j->hdmax, gqa = j->gqa, nkv = j->nkv;
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
			size_t b = (size_t)(j->l * nkv + kvh) * s->n_ctx
				 * hdmax;

			kbase = s->kcache + b;
			vbase = s->vcache + b;
			kstride = hdmax;
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

/*
 * A PROMPT IN CHUNKS, WHICH IS WHERE THE BATCHED MATMUL PAYS.
 *
 * Every projection in a prompt is the same weights against a different row, so
 * a token at a time re-streams the whole model per token. The probe measures
 * 5.14x at m = 32 on the projections alone.
 *
 * ⚠⚠ THIS IS A SECOND COPY OF THE LAYER LOOP AND IT IS DELIBERATELY BLIND.
 *
 * llama_forward carries seven architectures: gemma3's sliding window and two
 * rope bases, gemma4's per layer embeddings and shared KV, qwen3's q and k
 * norms, biases, post norms, the softcaps. Reimplementing all of that here is
 * how two copies quietly stop agreeing. So this handles the plain case and
 * REFUSES the rest, once and up front, and the caller falls back to the token
 * loop -- which is correct for all seven and merely slower.
 *
 * ⚠ WHAT IS BATCHED: the feed forward, 63% of the projection time (gate and up
 * 39%, down 24%). q, k, v and o stay one row at a time, because batching those
 * means duplicating the attention half too, and that is where the architectures
 * differ. Rows are still walked in order inside a layer: row r's attention
 * reads the KV that the rows before it wrote.
 *
 * ⚠ AND THE HEAD IS NOT BATCHED AT ALL. A prompt needs logits for its last
 * token and no other, so the widest projection in the model is skipped n - 1
 * times rather than made n times wider.
 */
static int npu_id_for(struct llama_state *s, const struct gguf_tensor *w)
{
	const struct npu_tensor *nt;

	if (!npu_mode() || !s->pool.dev || w->type == GGML_F32 || w->type == GGML_F16)
		return -1;
	nt = npu_get(s, w);
	if (!nt)
		return -1;
	for (unsigned i = 0; i < s->pool.n; i++)
		if (&s->pool.t[i] == nt)
			return s->pool.id[i];
	return -1;
}

/* n rows through one set of weights, or the same thing a row at a time */
/* whether the hardware would take a batch of this tensor, without trying one */
static int will_batch(struct llama_state *s, const struct gguf_tensor *w)
{
	return npu_id_for(s, w) >= 0 && charsiu_npu_batches(s->pool.dev);
}

/*
 * ⚠ OFF BY DEFAULT, and the default is the one the board preferred. See the
 * long note at the call site: fewer fences lost to more weight refetches.
 */
static int prefill_grouped(void)
{
	static int v = -1;

	if (v < 0)
		v = getenv("CHARSIU_PREFILL_GROUPED") != NULL;
	return v;
}

/*
 * Returns 1 when the hardware took the whole batch and 0 when this fell back to
 * a row at a time.
 *
 * ⚠ THE CALLER NEEDS TO KNOW. q, k and v all multiply one RMSNorm output, and
 * matvec_pair sends the three of them in ONE submit -- round 321 measured the
 * fence at 94% of the hardware path and that grouping took 113 fences a token
 * down to 65. Calling matmul_rows three times instead turns n grouped submits
 * into 3n separate ones the moment the batch is refused, which on int4 is
 * always. I did that to this loop two commits ago.
 */
static int matmul_rows(struct llama_state *s, const struct gguf_tensor *w,
		       const float *X, int n, float *Y, uint32_t k, uint32_t nout)
{
	int id = npu_id_for(s, w);

	if (id >= 0 && !charsiu_npu_matmul(s->pool.dev, id, X, (unsigned)n, Y))
		return 1;
	/*
	 * ⚠ AND IT HAS TO WORK WITHOUT THE NPU, or the loop restructuring
	 * above can only ever be checked on the board.
	 *
	 * ⚠⚠ BUT NOT ONE ROW AT A TIME. This fell back to n calls to matvec,
	 * which reads every weight row n times -- and the board's own numbers
	 * say what that costs: with CHARSIU_NPU_W4V=1, which is the default and
	 * what Rockchip's table compares against, charsiu_npu_matmul REFUSES
	 * every batch because w4a16 makes one row, so this fallback IS the
	 * prefill. Qwen3 0.6B prefilled at 46 ms a token against a decode of
	 * 63: a token loop with the output head skipped, which is exactly what
	 * the numbers looked like.
	 *
	 * ⚠⚠ AND gguf_matmul IS NOT THE ANSWER, MEASURED. It reads each weight
	 * row once for all m rows, which is the right idea and the wrong
	 * function: matvec here is LLAMA'S matvec, not gguf_matvec. It uses the
	 * requantised NPU copy when npu_mode is on and the activation's own q1
	 * form, and gguf_matmul knows about neither -- with the activations
	 * unquantised it falls into the scalar float path and dequantises the
	 * weight per row anyway.
	 *
	 *   tinyllama   4880 ms against 1203        FOUR TIMES SLOWER
	 *   qwen2.5     1723 ms against  775
	 *   qwen3       2141 ms against  940, and the text CHANGED
	 *
	 * Two of the four also stopped matching the token loop, which is what
	 * "a different function" looks like from outside. Batching this path
	 * properly means teaching llama's own matvec to take m rows, not
	 * routing around it.
	 */
	for (int r = 0; r < n; r++)
		matvec(s, w, X + (size_t)r * k, Y + (size_t)r * nout);
	return 0;
}

/*
 * The same call with the input declared unchanged since the previous batched
 * one -- k and v after q, up after gate. The CPU fallback is the same loop.
 */
/*
 * ⚠ WHICH OF THE THREE SITES MAY REUSE, so a board round can bisect. Every
 * site reuses unless CHARSIU_REUSE_SITES names a subset: any of k, v, up,
 * comma separated. Phase 2 broke Phi-3.5 and gemma4 with all three on, and
 * phase 22 walking the sites is what found the key with no expiry
 * (reusekey.h): k and up broke, and only on the core the leader had not
 * packed.
 */
static int reuse_site(char which)
{
	static const char *sites = NULL;
	static int asked = 0;

	if (!asked) {
		sites = getenv("CHARSIU_REUSE_SITES");
		asked = 1;
	}
	if (!sites || !*sites)
		return 1;
	switch (which) {
	case 'k': return strstr(sites, "k") != NULL && !strstr(sites, "kv-");
	case 'v': return strstr(sites, "v") != NULL;
	case 'u': return strstr(sites, "up") != NULL;
	}
	return 1;
}

static int matmul_rows_same(struct llama_state *s, const struct gguf_tensor *w,
			    const float *X, int n, float *Y, uint32_t k,
			    uint32_t nout, char site)
{
	int id = npu_id_for(s, w);

	if (!reuse_site(site))
		return matmul_rows(s, w, X, n, Y, k, nout);
	if (id >= 0 && !charsiu_npu_matmul_same(s->pool.dev, id, X, (unsigned)n, Y))
		return 1;
	for (int r = 0; r < n; r++)
		matvec(s, w, X + (size_t)r * k, Y + (size_t)r * nout);
	return 0;
}

/*
 * ⚠ WHY IT WILL NOT, NOT JUST THAT IT WILL NOT. A refusal that returns 0 makes
 * the caller fall back silently, and a board log then shows a batched run and a
 * control run at the same rate with nothing to say which of the two things that
 * means: the flag did nothing, or the architecture was never batchable and both
 * halves ran the same code. That cost a whole round on Phi-3.5, where wk and wv
 * are fused and this refuses on the first layer.
 *
 * The string is the reason and NULL means it will batch.
 */
const char *llama_batch_why_not(const struct llama_model *m)
{
	/*
	 * ⚠⚠ EVERY REASON, NOT THE FIRST ONE. Returning the first costs a
	 * board round each time it is fixed: gemma4 came back "a value norm",
	 * and the round after that would have said "KV shared between layers",
	 * and the one after that something else again. One line should say the
	 * whole distance to a batched prompt.
	 */
	static char buf[256];
	size_t n = 0;

	/*
	 * ⚠ THE MODEL IS UNUSED TODAY AND THE PARAMETER STAYS. Every refusal
	 * that ever lived here was a property of the model, and the next one
	 * will be too; taking it out would make putting one back an API change
	 * across four call sites and two tools, at the moment somebody is
	 * trying to write down a fault rather than refactor.
	 */
	(void)m;
	buf[0] = 0;
#define WHY(cond, txt) do {                                                 \
		if (cond) {                                                 \
			int w_ = snprintf(buf + n, sizeof(buf) - n, "%s%s",  \
					  n ? ", " : "", txt);              \
			if (w_ > 0 && (size_t)w_ < sizeof(buf) - n)         \
				n += (size_t)w_;                            \
		}                                                           \
	} while (0)

	WHY(kv_posmajor(), "a position major KV cache");
	/*
	 * 🏁 GEMMA4 AND PHI3 CAME OFF THIS LIST ON 2026-08-30, and what they
	 * were refused for was never true of either of them.
	 *
	 * gemma4 was refused for its per layer embeddings and phi3 for having
	 * q, k and v as views of one attn_qkv. Both notes said, in as many
	 * words, that this was a hypothesis about the cause and only a fact
	 * about the model. Both hypotheses are dead:
	 *
	 *  - phi3's batched matmul is EXACT at every even width: 225 tensors,
	 *    18000 of 18000 rows at m = 80, worst relative 1.61e-04. Being a
	 *    view had nothing to do with it.
	 *  - gemma4's failures were every one of them an ODD last chunk. Its
	 *    88 token prompt tails at 24 and came back clean ten times; the
	 *    rounds that called it wrong were reading a 89 token prompt --
	 *    one trailing space in a test script -- which tails at 25.
	 *
	 * What was actually wrong is two things, and neither is a property of
	 * an architecture:
	 *
	 *  1. an odd batch width has no expression on the accumulator surface,
	 *     which is organised in pairs of rows. The chunker only emits even
	 *     widths now and npudev refuses the rest.
	 *  2. the two NPU cores corrupt each other when their submits overlap.
	 *     phi3 at width 24: 13 of 16 wrong overlapped, 0 of 16 with one
	 *     core, 0 of 16 with the submits serialised. Serialised is the
	 *     default now.
	 *
	 * ⚠ WHAT WOULD PUT SOMETHING BACK HERE. This list is empty but for a
	 * debug switch, so the next architecture will not be refused by it --
	 * it will be MISSED by it, the way blk.N.layer_output_scale was: a
	 * tensor this loop did not apply, on a model nothing refused. A new
	 * architecture means reading llama_forward against this loop line by
	 * line, and then board_text_all.sh on the card, which is the check
	 * that found phi3 on its first run.
	 */
#undef WHY
	if (n)
		return buf;
	/*
	 * ⚠ FOUR REFUSALS LEFT THIS LIST ON 2026-08-28, and what they cost is
	 * why. Rockchip publish Qwen3-0.6B at 468 ms to the first token on this
	 * board; charsiu took 7354, because a query norm sent the whole prompt
	 * through the token loop. Of the five models in their table that this
	 * tree can run, FOUR were refused -- biases, a query norm, a fused K
	 * and V, per layer embeddings -- and only TinyLLAMA batched.
	 *
	 * The four removed are the ones that are the SAME PER ROW OPERATION the
	 * token loop already does, applied in the same order: a bias, a query
	 * and key norm, the two post norms, and the logit softcap. Refusing
	 * them was right while they were unwritten and became the dominant cost
	 * the moment there was a scoreboard.
	 *
	 * ⚠ AN ABSENT K OR V AND A SHARED KV LEFT THIS LIST on 2026-08-29, and
	 * neither was a different computation either. gemma4 makes attn_v
	 * optional in every layer, where its absence means V IS K, and drops
	 * attn_k entirely in its last ones, where the layer attends against
	 * L->kv_from's cache -- both of which the token loop has done since
	 * gemma4 landed. What the batched loop had to learn was to ask, three
	 * times, the question the token loop asks: skip the projections that
	 * are not there, skip the cache write, and read the cache the layer
	 * names.
	 *
	 * ⚠ A VARYING FEED FORWARD WIDTH LEFT THIS LIST on 2026-08-29, and it
	 * was never a computation at all -- it was one buffer sized from
	 * layers[0] instead of from the widest layer. m->n_ff has been the max
	 * since gemma4 landed, so the fix was to allocate from it and read
	 * L->n_ff in the loop, which is what the token loop already did.
	 *
	 * ⚠ PER LAYER EMBEDDINGS LEFT THIS LIST on 2026-08-29 and they WERE a
	 * computation -- the only one of the five that was. They are a second
	 * embedding table, a projection of the first, a norm a layer slice, and
	 * a gated residual at the bottom of every layer, which is the whole of
	 * what "E2B" means. What made them batchable was that all of it is per
	 * ROW: two matmuls that take n rows and scalar work that does not.
	 *
	 * ⚠⚠ THE LIST IS NOW EMPTY BUT FOR A DEBUG SWITCH, so the next
	 * architecture will not be refused by it -- it will be MISSED by it,
	 * the way blk.N.layer_output_scale was: a tensor this loop did not
	 * apply, on a model nothing else refused. A new architecture means
	 * reading llama_forward against this loop line by line, not trusting
	 * a NULL from here.
	 *
	 * ⚠ THE SLIDING WINDOW LEFT THIS LIST TOO, and it was the reason Phi3
	 * and Gemma4 took 23.6 s and 17.6 s to a first token against
	 * Rockchip's 1.8 and 1.2. It is not a mask this loop had to learn: the
	 * attention already takes t0, the oldest position a layer may look at,
	 * because the token loop has passed it since gemma3 landed. What was
	 * missing was passing it, and choosing the window layers' own rope
	 * table and head.
	 */
	return NULL;
}

static int batch_ok(const struct llama_model *m)
{
	const char *why = llama_batch_why_not(m);

	if (!why)
		return 1;
	/*
	 * ⚠ A REFUSAL WITH NO WAY PAST IT CANNOT BE TESTED.
	 *
	 * gemma4 and phi3 are refused because the board says their batched
	 * prompt is wrong, and each refusal names the property that
	 * distinguishes the model -- per layer embeddings, and weights that are
	 * views. Neither is proven guilty, and the standing alternative is that
	 * neither is guilty at all: m = 8 was four orders out on EVERY model
	 * until CHARSIU_NPU_ONEDEV made it bit exact, so this hardware has one
	 * known way to be wrong at m > 1 that has nothing to do with the model.
	 * If a forced phi3 comes back right on one core, both refusals are
	 * about the wrong thing.
	 *
	 * That experiment needs the refused arm to REACH the hardware, the same
	 * way CHARSIU_NPU_W4_BATCH=height lets the wrong axis through. It says
	 * so on stderr every run: this is a probe switch, and a number measured
	 * under it is a number about a model that is still refused.
	 */
	if (getenv("CHARSIU_BATCH_FORCE")) {
		static int said;

		if (!said++)
			fprintf(stderr, "charsiu: CHARSIU_BATCH_FORCE -- "
				"batching a model that is REFUSED (%s). Its "
				"output is not trusted; check the text.\n", why);
		return 1;
	}
	return 0;
}

/*
 * ⚠ THE LAYERS, AND ONLY THE LAYERS. Everything from the embedding lookup to
 * the last residual add, over n rows at pos0, with the KV cache written for
 * every row. What it leaves behind is s->bx: n rows of the final residual
 * stream, unnormed. Two callers finish it two ways -- llama_prefill_batch runs
 * the output head on the LAST row, which is all a prompt needs, and
 * llama_verify_batch runs it on EVERY row, which is what a speculative pass
 * needs to score its drafts. Neither owns a copy of the layer loop, so the two
 * cannot drift.
 *
 * s->pos is NOT advanced here; the callers do that, because the verify pass
 * advances it by how many drafts were accepted rather than by n.
 */
static int batch_layers(struct llama_state *s, const struct llama_model *m,
			const int32_t *toks, int n, int pos0)
{
	/*
	 * ⚠ hdmax IS THE CACHE'S STRIDE AND L->head_dim IS THE LAYER'S HEAD,
	 * and they are two different numbers. gemma4's window layers have a 256
	 * long head and its full ones 512 while the KV cache is one allocation
	 * with one stride, so indexing the cache by the live head makes layer 4
	 * read where layer 0 wrote.
	 *
	 * This loop used ONE hd for both, which is right for every architecture
	 * it currently accepts -- per layer heads only come with a sliding
	 * window and that is still refused -- and would be wrong the day the
	 * window is written. It is the same shape as the buffer stride that
	 * just cost a round on qwen3: correct by a coincidence of the models
	 * being run rather than by construction.
	 */
	uint32_t hdmax = m->head_dim ? m->head_dim : m->n_embd / m->n_head;
	uint32_t kvdim = m->n_head_kv * hdmax;
	/*
	 * ⚠ THE WIDEST LAYER'S FEED FORWARD, NOT LAYER ZERO'S. gemma4 states a
	 * width PER LAYER and E2B uses two of them, so a buffer sized from
	 * layers[0] and then written L->n_ff floats deep is a heap overflow on
	 * the first layer that disagrees with the first.
	 *
	 * m->n_ff is already the largest of them -- llama_load takes the max
	 * over the array for exactly this reason and says so -- so the
	 * allocation asks for that and the loop below uses the layer's own.
	 */
	uint32_t nffmax = m->n_ff;
	uint32_t gqa = m->n_head / m->n_head_kv;
	const float *freqf = NULL;
	float scale = m->attn_scale != 0.0f ? m->attn_scale
					    : 1.0f / sqrtf((float)hdmax);

	if (n < 2 || !batch_ok(m))
		return -1;
	if (s->bx_n < (unsigned)n) {
		free(s->bx); free(s->bxb); free(s->bhb);
		free(s->bhb2); free(s->bxo); free(s->bcs);
		s->bx = malloc((size_t)n * m->n_embd * sizeof(float));
		s->bxb = malloc((size_t)n * m->n_embd * sizeof(float));
		s->bxo = malloc((size_t)n * m->n_embd * sizeof(float));
		s->bhb = malloc((size_t)n * nffmax * sizeof(float));
		s->bhb2 = malloc((size_t)n * nffmax * sizeof(float));
		s->bcs = malloc((size_t)hdmax * sizeof(float));
		/*
		 * ⚠ q IS n_head * head_dim WIDE AND THAT IS NOT n_embd. Qwen3
		 * 0.6B is 16 heads of 128 against an embedding of 1024, so a
		 * buffer sized by n_embd is half of what the projection
		 * writes -- the same trap qwen3's attn_output buffer fell into
		 * when the architecture first landed.
		 */
		free(s->bq); free(s->bk); free(s->bv); free(s->bao);
		s->bq = malloc((size_t)n * m->n_head * hdmax * sizeof(float));
		s->bk = malloc((size_t)n * kvdim * sizeof(float));
		s->bv = malloc((size_t)n * kvdim * sizeof(float));
		/*
		 * ⚠⚠ AND THE ATTENTION'S OUTPUT IS n_head * head_dim WIDE TOO,
		 * WHICH IS THE SAME TRAP A SECOND TIME. Qwen3 0.6B is 16 heads
		 * of 128 against an embedding of 1024, so attention produces
		 * 2048 floats and hands them to wo, which contracts over 2048.
		 * Keeping them in a buffer whose rows are n_embd apart
		 * truncates every row and overlaps the next -- and the model
		 * still answered, in fluent English, about the wrong thing.
		 *
		 * The first time this was qwen3's own attn_output buffer when
		 * the architecture landed. It is written down in this file and
		 * I did it again in the batched copy of the same loop.
		 */
		s->bao = malloc((size_t)n * m->n_head * hdmax * sizeof(float));
		/*
		 * ⚠ gemma4's PER LAYER EMBEDDINGS ARE PER ROW. s->pl is one
		 * token's -- looked up by that token's id and projected from
		 * that token's own embedding -- so a chunk needs n of them.
		 * Sharing one would give every row of the prompt the last
		 * row's, which is a model that runs and answers.
		 */
		free(s->bpl); free(s->bplg);
		s->bpl = s->bplg = NULL;
		if (m->n_embd_pl) {
			s->bpl = malloc((size_t)n * m->n_embd_pl * m->n_layer
					* sizeof(float));
			s->bplg = malloc((size_t)n * m->n_embd_pl
					 * sizeof(float));
		}

		if (!s->bx || !s->bxb || !s->bxo || !s->bhb || !s->bhb2 ||
		    !s->bcs || !s->bq || !s->bk || !s->bv || !s->bao ||
		    (m->n_embd_pl && (!s->bpl || !s->bplg))) {
			s->bx_n = 0;
			return -1;
		}
		s->bx_n = n;
	}

	/*
	 * ⚠⚠ THE ROPE FREQUENCY FACTORS, WHICH THIS LOOP HAS NEVER READ. The
	 * token loop pulls m->rope_freqs into a buffer and hands it to
	 * rope_table; the batched copy passed NULL from the day it was written.
	 * Every model that carries that tensor -- Phi-3.5-mini's longrope is
	 * the one that showed it -- has been prefilled with a rotation that is
	 * not the one its decode uses.
	 *
	 * It went unnoticed because none of the models the batched path was
	 * checked against has the tensor: llama 3.2, qwen2, qwen3, SmolVLM. The
	 * control was right and the models were not varied enough.
	 */
	if (m->rope_freqs) {
		if (!s->bfreq)
			s->bfreq = calloc(hdmax, sizeof(float));
		if (s->bfreq) {
			gguf_row_f32(m->rope_freqs, 0, s->bfreq);
			freqf = s->bfreq;
		}
	}

	for (int r = 0; r < n; r++) {
		float *xr = s->bx + (size_t)r * m->n_embd;

		gguf_row_f32(m->tok_embd, (uint64_t)toks[r], xr);
		if (m->embd_scale != 1.0f)
			for (uint32_t i = 0; i < m->n_embd; i++)
				xr[i] *= m->embd_scale;
	}

	/*
	 * ⚠ gemma4's PER LAYER EMBEDDINGS, built once for the whole chunk:
	 *
	 *   pl[r][l][j] = ( proj[r][l][j] + tok[r][l][j] * sqrt(n_embd_pl) )
	 *                 / sqrt(2)
	 *
	 * where proj is per_layer_model_proj applied to the SCALED embedding
	 * and divided by sqrt(n_embd), then RMS normalised a layer at a time
	 * against per_layer_proj_norm, and tok is a row of a second embedding
	 * table looked up by the same token. Every line of it is the token
	 * loop's, in the token loop's order, with one index added.
	 *
	 * ⚠ THE NORM IS PER LAYER SLICE, not over the whole vector: the gain
	 * is n_embd_pl long and llama.cpp reshapes to [n_embd_pl][n_layer]
	 * before normalising. Over the concatenation every layer would be
	 * divided by every other layer's magnitude.
	 *
	 * ⚠ THE PROJECTION IS THE ONLY PART THAT BATCHES. The rest reads one
	 * row's table entry and normalises one row's slices, which is n times
	 * the same scalar code and not a matmul.
	 */
	if (m->n_embd_pl) {
		uint32_t np = m->n_embd_pl, nl = m->n_layer, l, j;
		float ts = sqrtf((float)np);
		float ps = 1.0f / sqrtf((float)m->n_embd);
		float half = 1.0f / sqrtf(2.0f);

		/*
		 * ⚠ READ ONCE. The token loop reads this gain inside its layer
		 * loop, which is the same vector every time; here that would be
		 * n_layer reads a row. Same values, so it cannot move a token.
		 */
		gguf_row_f32(m->pl_proj_norm, 0, s->plc);
		matmul_rows(s, m->pl_model_proj, s->bx, n, s->bpl, m->n_embd,
			    np * nl);
		for (int r = 0; r < n; r++) {
			gguf_row_f32(m->pl_tok_embd, (uint64_t)toks[r], s->plb);
			for (l = 0; l < nl; l++) {
				float *row = s->bpl
					   + ((size_t)r * nl + l) * np;
				float ss = 0.0f, sc;

				for (j = 0; j < np; j++) {
					row[j] *= ps;
					ss += row[j] * row[j];
				}
				sc = 1.0f / sqrtf(ss / (float)np + m->rms_eps);
				for (j = 0; j < np; j++)
					row[j] = (row[j] * sc * s->plc[j]
						  + s->plb[(size_t)l * np + j]
						    * ts) * half;
			}
		}
	}

	/*
	 * The stage clock for the batched path: the same fourteen names as
	 * the token loop, accumulated per row of the chunk. CHARSIU_STAGES
	 * turns it on; a prompt-only run reaches here before any token, so
	 * it is read here too rather than only in llama_forward.
	 */
	if (stage_on < 0)
		stage_on = getenv("CHARSIU_STAGES") != NULL;
	double bt0 = stage_on > 0 ? now_ms() : 0.0, bt1;
#define BSTAGE(i) do { if (stage_on > 0) { bt1 = now_ms();               \
			bstage_ms[i] += bt1 - bt0; bt0 = bt1; } } while (0)
	if (stage_on > 0) {
		bstage_rows += (unsigned)n;
		bstage_chunks++;
	}

	for (uint32_t l = 0; l < m->n_layer; l++) {
		const struct llama_layer *L = &m->layers[l];
		uint32_t hd = L->head_dim ? L->head_dim : hdmax;
		/* ⚠ THIS LAYER'S WIDTH; m->n_ff is only the fallback */
		uint32_t nff = L->n_ff ? L->n_ff : m->n_ff;
		int swa = L->swa;

		if (l == 0)
			BSTAGE(ST_EMBD);

		/*
		 * ⚠⚠ THE PROJECTIONS BATCH, THE ATTENTION DOES NOT. Only
		 * gate, up and down were batched here, which is three of the
		 * seven matmuls a layer does -- and the board showed what that
		 * left on the table: Qwen3 0.6B prefilled at 50 ms a token
		 * against a decode of 65, a 23% saving on work that is four
		 * fifths batchable.
		 *
		 * q, k, v and o are ordinary weight matmuls over n rows and
		 * batch exactly like the feed forward. What CANNOT batch is
		 * what sits between them: rope needs the position, the cache
		 * has to be written in order, and a row's attention reads the
		 * cache rows before it. So the loop below is the same as it
		 * was with the projections lifted out of it.
		 */
		for (int r = 0; r < n; r++)
			rmsnorm(s->bxb + (size_t)r * m->n_embd,
				s->bx + (size_t)r * m->n_embd, L->attn_norm,
				m->n_embd, m->rms_eps);
		BSTAGE(ST_NORM1);
		/*
		 * ⚠⚠ TENSOR MAJOR OR ROW MAJOR, AND THE BOARD SAYS TENSOR.
		 *
		 * Three calls to matmul_rows do all n rows of q, then all n of
		 * k, then all n of v -- the same weight, n submits in a row.
		 * matvec_pair instead does q, k and v for ONE row in one
		 * submit, which is fewer fences and a different weight every
		 * time. Round 321 measured the fence at 94% of the hardware
		 * path, so grouping should win, and on the board it LOST:
		 *
		 *   tensor major   38608 submits   TTFT 5239 ms
		 *   grouped        35528 submits   TTFT 6302 ms
		 *
		 * Eight percent fewer submits and twenty percent slower. The
		 * only reading that fits is that consecutive submits of the
		 * SAME weight do not pay for it twice, and grouping threw that
		 * away to save a fence.
		 *
		 * ⚠ It is a reading of two runs on a warming board, so it is a
		 * switch rather than a deletion: CHARSIU_PREFILL_GROUPED=1
		 * restores the grouping and the two can be compared in one
		 * session.
		 *
		 * ⚠ AND THIS IS WHAT THE VENDOR DOES TOO. Their own w4a16
		 * RK3576 model -- the model in the row this is measured
		 * against -- has 9296 four bit weight matmuls and NOT ONE of
		 * them is above M = 1. Every batched op in it is fp16 and is
		 * the attention. Batching an int4 weight matmul is not the
		 * mechanism behind their 469 ms.
		 */
		/*
		 * ⚠ ASKED, NOT TRIED. A first version wrote this as
		 * `!grouped() || matmul_rows(wq)`, and the short circuit meant
		 * that in the tensor major case matmul_rows was never called
		 * for wq at all -- q was simply not computed, and all three
		 * models came back DIFFERENT. The control caught it in one run.
		 */
		/*
		 * ⚠ A SHARED KV LAYER PROJECTS ONLY Q, and there is nothing to
		 * group. gemma4's last layers carry no attn_k and attend
		 * against an earlier layer's cache, so handing matmul_rows
		 * L->wk here would dereference NULL -- the same three cases the
		 * token loop spells out, in the same order, because the two
		 * loops disagreeing about which projections a layer has is the
		 * kind of difference that still produces fluent text.
		 */
		if (!L->wk) {
			matmul_rows(s, L->wq, s->bxb, n, s->bq, m->n_embd,
				    m->n_head * hd);
		} else if (!prefill_grouped() || will_batch(s, L->wq)) {
			matmul_rows(s, L->wq, s->bxb, n, s->bq, m->n_embd,
				    m->n_head * hd);
			/* ⚠ nothing writes bxb between these three: the
			 * declaration below is only true because of that */
			matmul_rows_same(s, L->wk, s->bxb, n, s->bk, m->n_embd,
					 m->n_head_kv * hd, 'k');
			if (L->wv)
				matmul_rows_same(s, L->wv, s->bxb, n, s->bv,
						 m->n_embd, m->n_head_kv * hd, 'v');
		} else {
			for (int r = 0; r < n; r++)
				matvec_pair(s, s->bxb + (size_t)r * m->n_embd,
					    L->wq, s->bq + (size_t)r *
						   m->n_head * hd,
					    L->wk, s->bk + (size_t)r *
						   m->n_head_kv * hd,
					    L->wv, s->bv + (size_t)r *
						   m->n_head_kv * hd);
		}
		/*
		 * ⚠ WHERE attn_v IS ABSENT AND attn_k IS NOT, V IS K. That is
		 * llama.cpp's `Vcur = Kcur` and not a projection this file
		 * failed to find, so it is a copy of the rows just computed and
		 * not a third matmul. gemma4 declares attn_v optional in EVERY
		 * layer, so this is not only the shared ones.
		 */
		if (L->wk && !L->wv)
			memcpy(s->bv, s->bk,
			       (size_t)n * m->n_head_kv * hd * sizeof(float));

		for (int r = 0; r < n; r++) {
			int pos = pos0 + r;

			/*
			 * ⚠ A WINDOW LAYER ROTATES AT ITS OWN BASE AND ITS OWN
			 * HEAD. gemma3's window layers turn at 10000 and its
			 * full ones at the model's own 1000000, and the file
			 * carries no key saying so.
			 *
			 * ⚠⚠ AND WITH NO FREQUENCY FACTORS. llama.cpp gives
			 * rope_freqs to the FULL layers only; this handed them
			 * to both, which scales a frequency table a window
			 * layer was never built for. It could not show on
			 * gemma3, which carries no such tensor, and gemma4 is
			 * the first model to arrive here with both a window and
			 * a shorter window head.
			 *
			 * The condition is the token loop's, word for word:
			 * a second table exists only where the base or the head
			 * actually differs, and where it does not a window
			 * layer takes the full one, factors and all.
			 */
			BSTAGE(ST_QKV);   /* row 0: the three projections; rows after: nothing */
			int swatab = swa && (m->swa_pattern || m->swa_arr) &&
				     (m->rope_base_swa != m->rope_base ||
				      m->head_dim_swa != m->head_dim);

			rope_table(s->bcs,
				   swatab ? m->head_dim_swa : hdmax, pos,
				   swatab ? m->rope_base_swa : m->rope_base,
				   swatab ? NULL : freqf);
			memcpy(s->q, s->bq + (size_t)r * m->n_head * hd,
			       (size_t)m->n_head * hd * sizeof(float));
			/* ⚠ bk AND bv HOLD NOTHING WHEN THERE IS NO wk, so
			 * every line below that touches k or v is asked the
			 * same question. The token loop leaves s->k and s->v
			 * from the previous layer and never reads them; this
			 * leaves them untouched for the same reason. */
			if (L->wk) {
				memcpy(s->k, s->bk + (size_t)r * m->n_head_kv
				       * hd,
				       (size_t)m->n_head_kv * hd *
				       sizeof(float));
				memcpy(s->v, s->bv + (size_t)r * m->n_head_kv
				       * hd,
				       (size_t)m->n_head_kv * hd *
				       sizeof(float));
			}
			/*
			 * ⚠ BIAS, THEN NORM, THEN ROPE -- the one order in
			 * here that is not interchangeable, and it is copied
			 * from the token loop rather than reasoned about
			 * again. Rope mixes element 2i with 2i+1, so norming
			 * after it divides a rotated pair by a sum of squares
			 * the rotation already changed; and rotating a biased
			 * vector is not biasing a rotated one.
			 */
			if (L->bq) {
				add_bias(s->q, L->bq, m->n_head * hd);
				/* ⚠ NO K MEANS NO K BIAS. The token loop adds
				 * all three unguarded and would dereference
				 * NULL here; no file in the zoo has both a
				 * shared KV and attention biases, so neither
				 * loop has ever reached it. */
				if (L->wk) {
					add_bias(s->k, L->bk,
						 m->n_head_kv * hd);
					add_bias(s->v, L->bv,
						 m->n_head_kv * hd);
				}
			}
			if (L->q_norm) {
				qk_norm(s->q, m->n_head, hd, L->q_norm,
					m->rms_eps);
				if (L->wk)
					qk_norm(s->k, m->n_head_kv, hd,
						L->k_norm, m->rms_eps);
			}
			/*
			 * ⚠ THE VALUE NORM IS THE SAME CALL, and refusing a
			 * model for it was right only while this line did not
			 * exist. gemma4 norms V with no gain -- llama.cpp
			 * writes it as a bare rms_norm, so there is no weight
			 * to find and the only place it exists is the graph --
			 * and it is qk_norm with a NULL gain, per row, in the
			 * same position the token loop puts it.
			 */
			if (m->v_norm && L->wk)
				qk_norm(s->v, m->n_head_kv, hd, NULL,
					m->rms_eps);
			rope(s->q, m->n_head, hd, s->bcs, m->rope_neox);
			if (L->wk)
				rope(s->k, m->n_head_kv, hd, s->bcs,
				     m->rope_neox);
			BSTAGE(ST_ROPE);
			/* ⚠ the cache is strided by hdmax, written at hd, and
			 * a shared KV layer has nothing of its own to store:
			 * it reads what L->kv_from wrote. Writing here would
			 * put this layer's q-only garbage over the slot the
			 * layer it borrows from just filled. */
			if (L->wk)
				for (uint32_t kh = 0; kh < m->n_head_kv; kh++) {
					size_t off = ((size_t)(l * m->n_head_kv
						      + kh)
						      * s->n_ctx + pos) * hdmax;

					memcpy(s->kcache + off, s->k + kh * hd,
					       hd * sizeof(float));
					memcpy(s->vcache + off, s->v + kh * hd,
					       hd * sizeof(float));
				}
			{
				/*
				 * ⚠⚠ t0 IS THE OLDEST POSITION THIS LAYER MAY
				 * READ, and the batched loop passed 0 for every
				 * layer -- which is a full attention wearing a
				 * window layer's weights. The token loop's own
				 * comment says this one cost four board rounds
				 * when it was first got wrong.
				 *
				 * A window layer sees the last n_swa positions
				 * INCLUDING this one, so the bound is
				 * pos + 1 - n_swa.
				 */
				int tlo = swa && pos + 1 > (int)m->n_swa
					? pos + 1 - (int)m->n_swa : 0;
				/*
				 * ⚠ WHOSE CACHE, and it is not always this
				 * layer's. gemma4's shared layers name an
				 * earlier one in L->kv_from; -1 is "its own",
				 * which is every layer of everything else.
				 * The head is still THIS layer's, and that is
				 * sound because kv_from is chosen among layers
				 * of the same window kind, which is the only
				 * thing head_dim depends on.
				 */
				struct attn_job aj = { s,
						       L->kv_from >= 0
						       ? (uint32_t)L->kv_from
						       : l,
						       pos, tlo, hd,
						       hdmax,
						       m->n_head_kv * hdmax,
						       gqa, m->n_head_kv,
						       scale };

				attn_heads(&aj, 0, m->n_head);
			}
			/* the attention's own output, kept for one batched
			 * o projection over every row */
			memcpy(s->bao + (size_t)r * m->n_head * hd, s->xb,
			       (size_t)m->n_head * hd * sizeof(float));
			BSTAGE(ST_ATTN);
		}

		matmul_rows(s, L->wo, s->bao, n, s->bxo, m->n_head * hd,
			    m->n_embd);
		BSTAGE(ST_WO);
		for (int r = 0; r < n; r++) {
			float *xr = s->bx + (size_t)r * m->n_embd;
			float *o = s->bxo + (size_t)r * m->n_embd;

			/* ⚠ on the branch, before the residual add: after it
			 * would normalise the residual stream too. */
			if (L->attn_post_norm)
				rmsnorm(o, o, L->attn_post_norm, m->n_embd,
					m->rms_eps);
			for (uint32_t i = 0; i < m->n_embd; i++)
				xr[i] += o[i];
			rmsnorm(s->bxb + (size_t)r * m->n_embd, xr, L->ffn_norm,
				m->n_embd, m->rms_eps);
		}
		BSTAGE(ST_RES1);   /* the residual add and the ffn rmsnorm */

		/* gate and up read one norm as well: the same choice */
		if (!prefill_grouped() || will_batch(s, L->gate)) {
			matmul_rows(s, L->gate, s->bxb, n, s->bhb, m->n_embd,
				    nff);
			matmul_rows_same(s, L->up, s->bxb, n, s->bhb2,
					 m->n_embd, nff, 'u');
		} else {
			for (int r = 0; r < n; r++)
				matvec_pair(s, s->bxb + (size_t)r * m->n_embd,
					    L->gate, s->bhb + (size_t)r * nff,
					    L->up, s->bhb2 + (size_t)r * nff,
					    NULL, NULL);
		}
		BSTAGE(ST_GATEUP);
		for (int r = 0; r < n; r++) {
			if (m->ffn_gelu)
				gelu_mul(s->bhb + (size_t)r * nff,
					 s->bhb2 + (size_t)r * nff, nff);
			else
				silu_mul(s->bhb + (size_t)r * nff,
					 s->bhb2 + (size_t)r * nff, nff);
		}
		BSTAGE(ST_SILU);
		matmul_rows(s, L->down, s->bhb, n, s->bxo, nff, m->n_embd);
		BSTAGE(ST_DOWN);
		for (int r = 0; r < n; r++) {
			float *xr = s->bx + (size_t)r * m->n_embd;
			float *o = s->bxo + (size_t)r * m->n_embd;

			if (L->ffn_post_norm)
				rmsnorm(o, o, L->ffn_post_norm, m->n_embd,
					m->rms_eps);
			for (uint32_t i = 0; i < m->n_embd; i++)
				xr[i] += o[i];
		}
		BSTAGE(ST_RES2);

		/*
		 * ⚠ gemma4's PER LAYER EMBEDDING, A RESIDUAL OF ITS OWN and not
		 * a replacement:
		 *
		 *   g = gelu(per_layer_inp_gate . x)      [n_embd_pl]
		 *   g = g * pl[l]                          elementwise
		 *   x = x + rmsnorm(per_layer_proj . g, per_layer_post_norm)
		 *
		 * This is the whole of what "E2B" means, so a batched prefill
		 * that skipped it would prompt a different model from the one
		 * that then decodes -- fluent, and answering about the wrong
		 * thing. Both matmuls batch; the gelu and the elementwise
		 * multiply are per row because the gate is.
		 */
		if (L->pl_inp_gate) {
			uint32_t np = m->n_embd_pl, i;

			matmul_rows(s, L->pl_inp_gate, s->bx, n, s->bplg,
				    m->n_embd, np);
			for (int r = 0; r < n; r++) {
				float *g = s->bplg + (size_t)r * np;
				const float *plr = s->bpl
						 + ((size_t)r * m->n_layer + l)
						   * np;

				/* ⚠ the same gate, so the same function:
				 * two open coded copies of this loop were
				 * still calling tanhf an element after
				 * gelu_mul stopped. */
				gelu_mul(g, plr, np);
			}
			/*
			 * ⚠ bxo IS FREE HERE. The feed forward's down
			 * projection went into it and was added into bx on the
			 * loop above; nothing reads it again this layer, and it
			 * is already n rows of n_embd, which is this
			 * projection's shape exactly.
			 */
			matmul_rows(s, L->pl_proj, s->bplg, n, s->bxo, np,
				    m->n_embd);
			for (int r = 0; r < n; r++) {
				float *xr = s->bx + (size_t)r * m->n_embd;
				float *o = s->bxo + (size_t)r * m->n_embd;

				if (L->pl_post_norm)
					rmsnorm(o, o, L->pl_post_norm,
						m->n_embd, m->rms_eps);
				for (i = 0; i < m->n_embd; i++)
					xr[i] += o[i];
			}
		}
		/*
		 * ⚠ ONE SCALAR THE WHOLE LAYER OUTPUT IS MULTIPLIED BY, and
		 * this loop never had it. It is not in llama_batch_why_not
		 * either, so a model carrying blk.N.layer_output_scale and
		 * none of the listed refusals would have been prefilled
		 * without it and decoded with it -- silently, since the two
		 * differ by a constant per layer and nothing here reads a
		 * magnitude. gemma4 is the only architecture that has it, and
		 * gemma4 was refused for other reasons until now.
		 *
		 * Read through gguf_row_f32 rather than cast, the way every
		 * other gain in this file is read.
		 */
		if (L->out_scale) {
			float sc = 1.0f;

			gguf_row_f32(L->out_scale, 0, &sc);
			for (int r = 0; r < n; r++) {
				float *xr = s->bx + (size_t)r * m->n_embd;

				for (uint32_t i = 0; i < m->n_embd; i++)
					xr[i] *= sc;
			}
		}
		/*
		 * ⚠ THE LAST ROW, BECAUSE THAT IS THE ONE THE TOKEN LOOP CAN BE
		 * HELD AGAINST. CHARSIU_DBG_LAYERS makes llama_forward print
		 * this line for every layer of every token; run a prompt of
		 * exactly one chunk and its final n_layer lines are the same
		 * token as these, so a batched path that has gone wrong says
		 * WHICH LAYER it went wrong at instead of only that the text
		 * changed. Both loops fall back to the same matvec on a machine
		 * with no NPU, so the two columns are expected to agree to
		 * every printed digit and not approximately.
		 */
		if (dbg_layers()) {
			const float *xr = s->bx + (size_t)(n - 1) * m->n_embd;
			double n2 = 0.0;
			uint32_t i;

			for (i = 0; i < m->n_embd; i++)
				n2 += (double)xr[i] * xr[i];
			fprintf(stderr, "  layer %2u  swa=%d hd=%3u ff=%5u "
				"kv=%2d  |x| = %.4f\n", l, L->swa, hd, nff,
				L->kv_from, sqrt(n2 / m->n_embd));
		}
	}

	return 0;
}

int llama_prefill_batch(struct llama_state *s, const struct llama_model *m,
			const int32_t *toks, int n, int pos0)
{
	if (batch_layers(s, m, toks, n, pos0))
		return -1;
	rmsnorm(s->xb, s->bx + (size_t)(n - 1) * m->n_embd, m->out_norm,
		m->n_embd, m->rms_eps);
	matvec(s, m->output, s->xb, s->logits);
	if (m->final_softcap > 0.0f)
		for (uint32_t i = 0; i < m->n_vocab; i++)
			s->logits[i] = tanhf(s->logits[i] / m->final_softcap) *
				       m->final_softcap;
	s->pos = pos0 + n;
	return 0;
}

/*
 * ⚠⚠ THE SAME FORWARD, WITH THE HEAD ON EVERY ROW.
 *
 * A speculative pass feeds the last committed token as row 0 and k drafted
 * tokens as rows 1..k. Whether draft i was right is decided by the logits of
 * row i -- the model's own next token given everything up to and including
 * draft i-1 -- so every row needs its logits, not the last one. The head is
 * one more batched matmul over n rows, and on the hardware it is the same
 * batched call a prompt makes; on a machine with no NPU matmul_rows runs it a
 * row at a time, which is correct and only slower.
 *
 * On return s->pos is pos0 + n and the cache holds all n rows. The CALLER
 * rolls s->pos back to pos0 + 1 + accepted: the rows past that are stale
 * entries that the next pass overwrites, and nothing reads a position above
 * s->pos, so rolling back is one assignment.
 */
int llama_verify_batch(struct llama_state *s, const struct llama_model *m,
		       const int32_t *toks, int n, int pos0, float *logits_all)
{
	if (batch_layers(s, m, toks, n, pos0))
		return -1;
	for (int r = 0; r < n; r++)
		rmsnorm(s->bxb + (size_t)r * m->n_embd,
			s->bx + (size_t)r * m->n_embd, m->out_norm,
			m->n_embd, m->rms_eps);
	matmul_rows(s, m->output, s->bxb, n, logits_all, m->n_embd,
		    m->n_vocab);
	if (m->final_softcap > 0.0f)
		for (size_t i = 0; i < (size_t)n * m->n_vocab; i++)
			logits_all[i] = tanhf(logits_all[i] / m->final_softcap)
					* m->final_softcap;
	s->pos = pos0 + n;
	return 0;
}

/*
 * ⚠ ONE CALL, THEN IT CLEARS ITSELF. A left-over embd_in would silently turn
 * the next real token into the previous picture, which is a fluent answer about
 * nothing that was asked.
 */
const float *llama_forward_embd(struct llama_state *s, const float *embd,
				int pos)
{
	s->embd_in = embd;
	return llama_forward(s, 0, pos);
}

const float *llama_forward(struct llama_state *s, int32_t token, int pos)
{
	const struct llama_model *m = s->m;
	uint32_t hdmax = m->head_dim;
	uint32_t gqa = m->n_head / m->n_head_kv;
	float scale = m->attn_scale;
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
			freqbuf = calloc(hdmax, sizeof(float));
		gguf_row_f32(m->rope_freqs, 0, freqbuf);
		freqf = freqbuf;
	}

	if (!ropecs)
		ropecs = calloc(hdmax, sizeof(float));
	rope_table(ropecs, hdmax, pos, m->rope_base, freqf);
	/*
	 * The second table exists only when the two bases differ, which is
	 * gemma3 and nothing else. Both are still one table a TOKEN, not one a
	 * layer: theta depends on the base, the pair and the position.
	 */
	/*
	 * ⚠ THE WINDOW TABLE IS ITS OWN LENGTH AND HAS NO FACTORS. gemma4
	 * rotates 256 of a window layer's head against 512 of a full one, and
	 * llama.cpp gives rope_freqs to the FULL layers only -- a window layer
	 * gets the plain rotation. Handing it the full layers' factors would
	 * scale a frequency table it was never built for.
	 */
	if ((m->swa_pattern || m->swa_arr) &&
	    (m->rope_base_swa != m->rope_base ||
	     m->head_dim_swa != m->head_dim)) {
		if (!ropecs_swa)
			ropecs_swa = calloc(hdmax, sizeof(float));
		rope_table(ropecs_swa, m->head_dim_swa, pos, m->rope_base_swa,
			   NULL);
	}

	/* see llama_forward_embd: a picture enters here, past the lookup */
	if (s->embd_in) {
		memcpy(s->x, s->embd_in, m->n_embd * sizeof(float));
		s->embd_in = NULL;
	} else {
		gguf_row_f32(m->tok_embd, (uint64_t)token, s->x);
		if (m->embd_scale != 1.0f)
			for (uint32_t i = 0; i < m->n_embd; i++)
				s->x[i] *= m->embd_scale;
	}

	/*
	 * ⚠ gemma4's PER LAYER EMBEDDINGS, built once for the whole token.
	 *
	 *   pl[l][j] = ( proj[l][j] + tok[l][j] * sqrt(n_embd_pl) ) / sqrt(2)
	 *
	 * where proj is per_layer_model_proj applied to the SCALED embedding
	 * and divided by sqrt(n_embd), then RMS normalised a layer at a time
	 * against per_layer_proj_norm. tok is a row of a second embedding
	 * table, n_embd_pl * n_layer wide, looked up by the same token.
	 *
	 * ⚠ THE NORM IS PER LAYER SLICE, not over the whole vector: the gain
	 * is n_embd_pl long and llama.cpp reshapes to [n_embd_pl][n_layer]
	 * before normalising. Doing it over the concatenation would divide
	 * every layer by every other layer's magnitude.
	 */
	if (m->n_embd_pl) {
		uint32_t np = m->n_embd_pl, nl = m->n_layer, l, j;
		float ts = sqrtf((float)np);
		float ps = 1.0f / sqrtf((float)m->n_embd);
		float half = 1.0f / sqrtf(2.0f);

		gguf_row_f32(m->pl_tok_embd, (uint64_t)token, s->plb);
		charsiu_act_set(&s->act, s->x, (int)m->n_embd);
		matvec_again(s, m->pl_model_proj, s->pl);
		for (l = 0; l < nl; l++) {
			float *row = s->pl + (size_t)l * np;
			float ss = 0.0f, sc;

			for (j = 0; j < np; j++) {
				row[j] *= ps;
				ss += row[j] * row[j];
			}
			sc = 1.0f / sqrtf(ss / (float)np + m->rms_eps);
			gguf_row_f32(m->pl_proj_norm, 0, s->plc);
			for (j = 0; j < np; j++)
				row[j] = (row[j] * sc * s->plc[j]
					  + s->plb[(size_t)l * np + j] * ts)
					 * half;
		}
	}
	STAGE(ST_EMBD);

	for (uint32_t l = 0; l < m->n_layer; l++) {
		const struct llama_layer *L = &m->layers[l];

		cur_layer = l;
		/*
		 * ⚠ `il % n < n - 1` IS THE PATTERN, taken from llama.cpp's
		 * set_swa_pattern: with the gemma default of 6 that is five
		 * window layers and then a full one, not one window in six.
		 */
		/*
		 * ⚠ RESOLVED AT LOAD, not here. gemma3 states a period and
		 * gemma4 states one flag a layer, and the forward pass should
		 * not have to know which form the file used.
		 */
		int swa = L->swa;
		/*
		 * ⚠ THIS LAYER'S HEAD, not the model's. gemma4 is the first
		 * architecture here whose layers disagree about it.
		 */
		uint32_t hd = L->head_dim;
		uint32_t kvdim = m->n_head_kv * hd;
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

		/*
		 * ⚠ A SHARED KV LAYER PROJECTS ONLY Q. gemma4's last layers
		 * carry no attn_k and attend against an earlier layer's cache,
		 * so asking matvec_pair for k and v would dereference NULL.
		 * And where attn_v is absent but attn_k is not, V IS K -- that
		 * is llama.cpp's `Vcur = Kcur`, not a missing projection.
		 */
		if (!L->wk) {
			matvec(s, L->wq, s->xb, s->q);
		} else if (!L->wv) {
			matvec_pair(s, s->xb, L->wq, s->q, L->wk, s->k,
				    NULL, NULL);
			memcpy(s->v, s->k,
			       (size_t)m->n_head_kv * hd * sizeof(float));
		} else {
			matvec_pair(s, s->xb, L->wq, s->q, L->wk, s->k,
				    L->wv, s->v);
		}
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
			if (L->wk)
				qk_norm(s->k, m->n_head_kv, hd, L->k_norm,
					m->rms_eps);
		}
		/*
		 * ⚠ gemma4 NORMS V TOO, AND WITH NO GAIN. llama.cpp writes it
		 * as a bare ggml_rms_norm rather than a build_norm, so there is
		 * no weight to look for and nothing in the tensor list to
		 * notice it by -- the only place it exists is the graph.
		 */
		if (m->v_norm && L->wk)
			qk_norm(s->v, m->n_head_kv, hd, NULL, m->rms_eps);
		STAGE(ST_QKV);

		charsiu_note("rope on q", cur_layer, (unsigned long)m->n_head);
		rope(s->q, m->n_head, hd, cs, m->rope_neox);
		charsiu_note("rope on k", cur_layer,
			     (unsigned long)m->n_head_kv);
		if (L->wk)
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
		if (!L->wk) {
			/* nothing of its own to store: it reads L->kv_from's */
		} else if (kv_posmajor()) {
			size_t off = ((size_t)l * s->n_ctx + pos) * kvdim;

			memcpy(s->kcache + off, s->k, kvdim * sizeof(float));
			memcpy(s->vcache + off, s->v, kvdim * sizeof(float));
		} else {
			for (uint32_t kh = 0; kh < m->n_head_kv; kh++) {
				size_t off = ((size_t)(l * m->n_head_kv + kh)
					      * s->n_ctx + pos) * hdmax;

				memcpy(s->kcache + off, s->k + kh * hd,
				       hd * sizeof(float));
				memcpy(s->vcache + off, s->v + kh * hd,
				       hd * sizeof(float));
			}
		}
		STAGE(ST_ROPE);

		{
			struct attn_job aj = { s,
					       L->kv_from >= 0
					       ? (uint32_t)L->kv_from : l,
					       pos, tlo, hd, hdmax, kvdim,
					       gqa, m->n_head_kv, scale };

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
			gelu_mul(s->hb, s->hb2, L->n_ff);
		else
			silu_mul(s->hb, s->hb2, L->n_ff);
		STAGE(ST_SILU);
		matvec(s, L->down, s->hb, s->xb2);
		STAGE(ST_DOWN);
		if (L->ffn_post_norm)
			rmsnorm(s->xb2, s->xb2, L->ffn_post_norm, m->n_embd,
				m->rms_eps);
		for (uint32_t i = 0; i < m->n_embd; i++)
			s->x[i] += s->xb2[i];

		/*
		 * ⚠ gemma4's PER LAYER EMBEDDING, and it is a RESIDUAL of its
		 * own rather than a replacement:
		 *
		 *   g = gelu(per_layer_inp_gate . x)      [n_embd_pl]
		 *   g = g * pl[l]                          elementwise
		 *   x = x + rmsnorm(per_layer_proj . g, per_layer_post_norm)
		 *
		 * This is the whole of what "E2B" means -- the model carries
		 * far more parameters than it activates and this is the path
		 * that chooses, per layer and per token, which of them matter.
		 * Leaving it out gives a model that loads, runs and answers,
		 * with every layer missing the half of itself that the name is
		 * about.
		 */
		if (L->pl_inp_gate) {
			uint32_t np = m->n_embd_pl, i;
			const float *plr = s->pl + (size_t)l * np;

			matvec(s, L->pl_inp_gate, s->x, s->plc);
			gelu_mul(s->plc, plr, np);
			matvec(s, L->pl_proj, s->plc, s->xb2);
			if (L->pl_post_norm)
				rmsnorm(s->xb2, s->xb2, L->pl_post_norm,
					m->n_embd, m->rms_eps);
			for (i = 0; i < m->n_embd; i++)
				s->x[i] += s->xb2[i];
		}
		/*
		 * One scalar the whole layer output is multiplied by. f32 and
		 * one element; read it through gguf_row_f32 rather than
		 * assuming the type, the way every other gain here is read.
		 */
		if (L->out_scale) {
			float sc = 1.0f;

			gguf_row_f32(L->out_scale, 0, &sc);
			for (uint32_t i = 0; i < m->n_embd; i++)
				s->x[i] *= sc;
		}
		if (dbg_layers()) {
			double n2 = 0.0;
			uint32_t i;

			for (i = 0; i < m->n_embd; i++)
				n2 += (double)s->x[i] * s->x[i];
			fprintf(stderr, "  layer %2u  swa=%d hd=%3u ff=%5u "
				"kv=%2d  |x| = %.4f\n", l, L->swa, hd, L->n_ff,
				L->kv_from, sqrt(n2 / m->n_embd));
		}
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

/*
 * ⚠⚠ SPECULATIVE DECODING, AND WHY IT IS THE ONE ALGORITHM THAT CAN GO PAST
 * THE VENDOR RATHER THAN TO IT.
 *
 * At m = 1 every token reads every weight, and that is the whole cost of a
 * decode step on this hardware: Qwen3-0.6B is 298 MB of int4 and the board
 * reads it at 16 GB/s across the two cores, which is the 52 ms the token
 * takes. Rockchip reads the same bytes at 40 ms. Nothing inside a matmul
 * changes the bytes, and the NPU has no int2 or int3, so the only way to more
 * tokens a second is more tokens a READ -- which is what a batch does. The
 * batched path reads the weights ONCE for m rows, and the board has it exact at
 * m = 2, 4 and 6.
 *
 * So: guess k tokens by some cheap means, feed the last committed token and
 * the k guesses as one batch of 1 + k rows, and read off every row's logits.
 * Row i's argmax is what greedy decoding would have produced after guess i-1.
 * Accept guesses while they match; the first row that disagrees supplies the
 * token greedy would have produced there instead. Every committed token is
 * therefore exactly the token the plain loop would have committed -- the
 * drafts only decide how many of them one weight read yields. THE TEXT IS
 * BIT-IDENTICAL TO GREEDY BY CONSTRUCTION, and tests/spec_identity.sh holds it
 * to that with a control whose drafts are junk.
 *
 * The drafter here is prompt lookup: the last few tokens are searched for in
 * everything the model has seen or said, and what followed them last time is
 * proposed. It costs nothing, needs no second model, and is strong wherever the
 * output repeats its input -- quoting, summarising, code, structured answers --
 * and weak on open prose, where it mostly proposes nothing and the pass
 * degrades to a plain forward. It is the drafter to start with because it is
 * the one with no model to get wrong; the pass itself does not care where the
 * drafts come from.
 *
 * ⚠ WHAT THIS DOES NOT DO. Sampling at a temperature is left to the plain
 * loop: lossless speculative SAMPLING exists (accept with p_target(d), else
 * draw from the residual) and is not written here, so with --temp the runner
 * says so once and does not speculate. And a pass at m = 4 has never been
 * PRICED on the board -- the argument that it costs about one decode step is
 * an argument about bytes, and the fence and the read back both grow with m.
 * The acceptance statistics this prints are a property of the model and the
 * prompt and are measured on any machine; the pass cost is the one number
 * that needs the hardware.
 */

static void spec_push(struct llama_spec *sp, int32_t tok)
{
	if (sp->n_hist < sp->cap_hist)
		sp->hist[sp->n_hist++] = tok;
}

int llama_spec_init(struct llama_spec *sp, const struct llama_model *m, int k,
		    int n_ctx)
{
	memset(sp, 0, sizeof(*sp));
	if (k < 1)
		k = 1;
	/*
	 * ⚠ AT MOST 5, BECAUSE THE ROWS ARE 1 + k AND THE BOARD REFUSES 8 AND
	 * 10. npudev's dense sweep has 2, 4 and 6 exact on both cores and 8
	 * and 10 missing row 0 of the wide projections, so a pass is 2, 4 or 6
	 * rows and nothing between 6 and 12 is asked for.
	 */
	if (k > 5)
		k = 5;
	sp->k = k;
	sp->ngram = 3;
	sp->n_vocab = m->n_vocab;
	sp->cap_hist = n_ctx;
	sp->hist = malloc((size_t)n_ctx * sizeof(*sp->hist));
	/* k drafts, row 0, and one row of padding to keep the width even */
	sp->logits_all = malloc((size_t)(k + 2) * m->n_vocab * sizeof(float));
	sp->junk = getenv("CHARSIU_SPEC_JUNK") != NULL;
	if (!sp->hist || !sp->logits_all) {
		llama_spec_free(sp);
		return -1;
	}
	return 0;
}

void llama_spec_free(struct llama_spec *sp)
{
	free(sp->hist);
	free(sp->logits_all);
	memset(sp, 0, sizeof(*sp));
}

void llama_spec_push(struct llama_spec *sp, int32_t tok)
{
	spec_push(sp, tok);
}

void llama_spec_reset(struct llama_spec *sp)
{
	sp->n_hist = 0;
}

/*
 * Prompt lookup: the longest n-gram ending at the present that has occurred
 * before, and up to k of what followed it. Newest occurrence first, because a
 * repeated structure is more likely to continue the way it most recently did.
 */
static int spec_draft(const struct llama_spec *sp, int32_t *out, int k)
{
	int n = sp->n_hist;

	/*
	 * ⚠ THE CONTROL. Junk drafts must be rejected every time and the text
	 * must not move; a run where they are accepted, or where the text
	 * changes, is a verifier that is not verifying. Deterministic in the
	 * history length so a run reproduces.
	 */
	if (sp->junk) {
		for (int i = 0; i < k; i++)
			out[i] = (int32_t)(((unsigned)n * 104729u + (unsigned)i
					    * 7919u + 17u) % (unsigned)sp->n_vocab);
		return k;
	}
	for (int ng = sp->ngram; ng >= 1; ng--) {
		const int32_t *pat;

		if (n < ng + 1)
			continue;
		pat = sp->hist + n - ng;
		for (int i = n - ng - 1; i >= 0; i--) {
			int avail, d;

			if (memcmp(sp->hist + i, pat, (size_t)ng * sizeof(*pat)))
				continue;
			avail = n - (i + ng);
			d = avail < k ? avail : k;
			memcpy(out, sp->hist + i + ng, (size_t)d * sizeof(*out));
			return d;
		}
	}
	return 0;
}

int llama_spec_step(struct llama_spec *sp, struct llama_state *s,
		    const struct llama_model *m, int32_t tok, int32_t *out,
		    int max_out)
{
	int32_t rows[8], drafts[8];
	int d, n, a, nc, pos0 = s->pos;

	spec_push(sp, tok);
	d = sp->off ? 0 : spec_draft(sp, drafts, sp->k);
	/* every row must fit the cache, or the pass would write past it */
	if (d && pos0 + 1 + d + 1 > s->n_ctx)
		d = 0;
	if (d == 0) {
		const float *lg = llama_forward(s, tok, s->pos);

		if (!lg)
			return -1;
		out[0] = llama_argmax(lg, m->n_vocab);
		sp->passes++;
		sp->plain++;
		return 1;
	}
	rows[0] = tok;
	for (int i = 0; i < d; i++)
		rows[1 + i] = drafts[i];
	n = 1 + d;
	/*
	 * ⚠ AN ODD WIDTH HAS NO EXPRESSION ON THE SURFACE, so pad to even with
	 * a row that is never read: it sits after the last draft, nothing
	 * before it can see it, and the roll back below discards it.
	 */
	if (n & 1)
		rows[n++] = drafts[d - 1];
	if (llama_verify_batch(s, m, rows, n, pos0, sp->logits_all)) {
		/*
		 * ⚠ REFUSED, AND SAID ONCE. batch_layers refuses before it
		 * touches the cache, so nothing needs undoing; the rest of the
		 * run is the plain loop and the report line says so, because
		 * a speculative run that quietly ran plain would read as
		 * "speculation gained nothing" and that is not what happened.
		 */
		if (!sp->off)
			fprintf(stderr, "charsiu: speculation is off -- the "
				"batched forward refused this model (%s)\n",
				llama_batch_why_not(m));
		sp->off = 1;
		s->pos = pos0;
		return llama_spec_step(sp, s, m, tok, out, max_out);
	}
	a = 0;
	for (int i = 0; i < d; i++) {
		int32_t x = llama_argmax(sp->logits_all
					 + (size_t)i * m->n_vocab, m->n_vocab);

		if (x != drafts[i])
			break;
		a++;
	}
	/*
	 * The committed tokens: the a drafts that matched, and then the
	 * model's own token from row a -- the correction where a draft was
	 * wrong, or the free extra token where all of them were right. Their
	 * count is at most d + 1, which is at most 6, and max_out is 16.
	 */
	nc = 0;
	for (int i = 0; i < a && nc < max_out; i++)
		out[nc++] = drafts[i];
	if (nc < max_out)
		out[nc++] = llama_argmax(sp->logits_all + (size_t)a * m->n_vocab,
					 m->n_vocab);
	/*
	 * Rows 1..a hold accepted tokens and their cache entries are right.
	 * The correction is not in the cache yet: it is the next pass's row 0.
	 */
	s->pos = pos0 + 1 + a;
	/*
	 * ⚠ EVERY TOKEN ENTERS THE HISTORY EXACTLY ONCE. The caller feeds the
	 * LAST committed token back as the next pass's `tok`, which pushes it
	 * then; everything before it is pushed here.
	 */
	for (int i = 0; i + 1 < nc; i++)
		spec_push(sp, out[i]);
	sp->passes++;
	sp->drafted += (unsigned long)d;
	sp->accepted += (unsigned long)a;
	sp->committed += (unsigned long)nc;
	return nc;
}

void llama_spec_report(const struct llama_spec *sp, FILE *out)
{
	unsigned long spec_passes = sp->passes - sp->plain;

	fprintf(out, "[spec k=%d: %lu passes, %lu plain, %lu speculative; "
		"drafted %lu, accepted %lu (%.0f%%), committed %lu; "
		"%.2f tok/pass%s%s]\n",
		sp->k, sp->passes, sp->plain, spec_passes,
		sp->drafted, sp->accepted,
		sp->drafted ? 100.0 * sp->accepted / sp->drafted : 0.0,
		sp->committed + sp->plain,
		sp->passes ? (double)(sp->committed + sp->plain) / sp->passes
			   : 0.0,
		sp->off ? ", REFUSED by the batched forward" : "",
		sp->junk ? ", JUNK drafts (control)" : "");
}
