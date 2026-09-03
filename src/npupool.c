// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * What is on the hardware, for any graph.
 *
 * ⚠ THIS WAS FIVE FIELDS INSIDE struct llama_state, and that is why the vision
 * tower, CLIP and whisper were all on the CPU: staging a weight was something
 * only the language model could do. Measured on the board at 0.6 G-mac/s, three
 * times, by three graphs that never touched the hardware they ran on.
 *
 * ⚠ ONE STAGING PATH. Everything a weight needs to reach the NPU is here once --
 * the requantised copy, CHARSIU_NPU_ONLY, and the maxn gate that kept an output
 * head on the CPU for a fortnight while saying nothing about it. A second copy
 * for the towers is how the two drift, and the maxn refusal is the proof: it
 * cost more and said less than any other refusal in this tree.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>

#include <time.h>

#include "charsiu.h"
#include "charsiu_llm.h"

static double now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

double charsiu_pool_stage_ms;

static unsigned pool_maxn(void)
{
	const char *e = getenv("CHARSIU_NPU_MAXN");

	return e ? (unsigned)atoi(e) : 8192;
}

int charsiu_pool_init(struct charsiu_npu_pool *p, unsigned max_tensors,
		      unsigned max_k, unsigned max_n, int want_w4)
{
	memset(p, 0, sizeof(*p));
	p->cap = max_tensors;
	p->t = calloc(p->cap, sizeof(*p->t));
	p->key = calloc(p->cap, sizeof(*p->key));
	p->src = calloc(p->cap, sizeof(*p->src));
	p->id = calloc(p->cap, sizeof(*p->id));
	if (!p->t || !p->key || !p->src || !p->id) {
		charsiu_pool_fini(p);
		return -1;
	}
	if (max_k && max_n)
		p->dev = charsiu_npu_open_mode(max_k, max_n, p->cap, want_w4);
	return 0;
}

void charsiu_pool_fini(struct charsiu_npu_pool *p)
{
	unsigned i;

	for (i = 0; i < p->n; i++)
		npu_tensor_free(&p->t[i]);
	free(p->t);
	free(p->key);
	free(p->src);
	free(p->id);
	if (p->dev)
		charsiu_npu_close(p->dev);
	memset(p, 0, sizeof(*p));
}

const struct npu_tensor *charsiu_pool_get(struct charsiu_npu_pool *p,
					  const struct gguf_tensor *w)
{
	double t_stage;
	unsigned i;

	for (i = 0; i < p->n; i++)
		if (p->key[i] == w) {
			/*
			 * ⚠⚠ THE SAME ADDRESS IS NOT THE SAME TENSOR. This
			 * keys on the pointer, and a caller that builds a
			 * temporary struct gguf_tensor on the stack hands over
			 * the SAME address with different weights behind it
			 * every call. whisper's conv1d3 does exactly that --
			 * three taps and two convolutions through one stack
			 * slot -- so the first tap was staged and then used for
			 * all six. The transcript came back EMPTY, which is the
			 * loud version; a smaller error would have come back as
			 * words.
			 *
			 * Refuse, once, out loud. A caller whose weights move
			 * belongs on the CPU.
			 */
			if (p->src[i] != w->data) {
				static int said;

				if (!said++)
					fprintf(stderr, "charsiu: %s is staged "
						"with different weights behind "
						"the same address -- it stays "
						"on the CPU\n", w->name);
				return NULL;
			}
			return &p->t[i];
		}

	if (p->n == p->cap) {
		static int said;

		if (!said++)
			fprintf(stderr, "charsiu: %s stays on the CPU -- all %u "
				"tensor slots are taken\n", w->name, p->cap);
		return NULL;
	}
	/*
	 * ⚠ w->name IS INSIDE THE MAPPED FILE for a whole tensor and inside
	 * the layer for one of phi3's slices; both outlive a crash.
	 */
	charsiu_note(w->name, (unsigned long)w->ne[1], (unsigned long)w->ne[0]);
	t_stage = now_ms();
	/* npu_tensor_build says why on its own way out */
	if (npu_tensor_build(&p->t[p->n], w) < 0) {
		charsiu_pool_stage_ms += now_ms() - t_stage;
		return NULL;
	}
	p->id[p->n] = -1;
	if (p->dev) {
		const char *only = getenv("CHARSIU_NPU_ONLY");

		if ((!only || strstr(w->name, only)) &&
		    w->ne[1] <= (uint64_t)pool_maxn()) {
			p->id[p->n] = charsiu_npu_add(p->dev, &p->t[p->n]);
		} else if (!only && w->ne[1] > (uint64_t)pool_maxn()) {
			/*
			 * ⚠ THE ONE REFUSAL THAT COST THE MOST AND SAID THE
			 * LEAST. Every other way onto the hardware whines when
			 * it declines; this one only spoke under
			 * CHARSIU_NPU_VERBOSE, and it is the gate the output
			 * head hits -- gemma3's is 262144 rows against a
			 * default of 8192. That head is 44% of the token.
			 */
			static int said;

			if (!said++)
				fprintf(stderr, "charsiu: %s stays on the CPU "
					"-- %llu rows is over CHARSIU_NPU_MAXN=%u "
					"(charsiu-config, [npu] maxn)\n",
					w->name,
					(unsigned long long)w->ne[1], pool_maxn());
		}
		if (getenv("CHARSIU_NPU_VERBOSE"))
			fprintf(stderr, "  -> %s\n",
				p->id[p->n] >= 0 ? "on the NPU" : "on the CPU");
	}
	if (getenv("CHARSIU_NPU_VERBOSE"))
		fprintf(stderr, "npu-quant  %-28s  %llu x %llu  rms %.4f%%\n",
			w->name, (unsigned long long)w->ne[1],
			(unsigned long long)w->ne[0], p->t[p->n].rms_rel * 100.0);
	snprintf(p->t[p->n].name, sizeof(p->t[p->n].name), "%s", w->name);
	p->key[p->n] = w;
	p->src[p->n] = w->data;
	charsiu_pool_stage_ms += now_ms() - t_stage;
	return &p->t[p->n++];
}

/*
 * ⚠⚠ THE BATCH IS CHUNKED, AND 32 IS THE ONLY WIDTH THAT HAS EVER BEEN CHECKED.
 *
 * charsiu_npu_matmul was verified at m = 2 to 32, against a CPU reference, value
 * for value. The towers hand it 1024 patches and 1500 encoder positions, and the
 * first board round that did so came back with a picture the model called "I am
 * not sure" and a transcript that was EMPTY -- while CLIP, whose tower is fifty
 * rows, was right. Fifty is inside the range and a thousand is not.
 *
 * Mesa's own arithmetic says where this goes wrong and it is written in the
 * README already: an LLM matmul reaches the encoder one column wide, so the
 * CBUF budget test fires above m = 320 and the row split above m = 640. Those
 * are the shapes Mesa splits and charsiu never has.
 *
 * So: chunk. CHARSIU_NPU_ROWS_MAX moves the bound, which is what the sweep that
 * finds the real one needs, and 32 is the default because it is the number that
 * has evidence behind it rather than the number that looks safe.
 */
/*
 * ⚠ 64, AND THE BOARD SAID WHERE THE EDGE IS: 80 is the last width whose output
 * is IDENTICAL to two rows, and 96 is the first that is not.
 *
 *     4 8 16 32 48 64 80   0.000000   identical
 *     96 112 128 ... 1024  56 to 95   a different tower
 *
 * ⚠ AND THE RATE IS FLAT ACROSS ALL OF THEM -- 78 to 81 s at every width, so
 * there is nothing to buy by sitting next to the edge. 64 is inside it with a
 * whole step to spare, and what the sweep also showed is that the time is not
 * here at all: 75 of the tower's 82 s was the quantiser.
 *
 * ⚠ AND 80 WAS MEASURED ON ONE TOWER, at K = 768 and 3072. Whether the limit is
 * m alone or m against K is not known, which is the other reason not to sit at
 * the edge.
 */
/*
 * ⚠ 80 NOW, NOT 64, AND ON TWO MEASUREMENTS. The sweep above has 80 identical
 * at K = 768 and 3072, and board_verify phase 18 has K = 3072 at 80 rows EXACT
 * against the row loop on the height axis (surface 7680; the next cell up,
 * 10240, is wrong and phase 19 walks the gap). When the tower took 82 s the
 * dispatch count did not matter; at 5.4 s it does -- 1024 rows is 16 calls of
 * 64 or 13 of 80, and every call carries its own pack, fence and read.
 */
static unsigned rows_max(void)
{
	const char *e = getenv("CHARSIU_NPU_ROWS_MAX");
	int v = e ? atoi(e) : 80;

	return v > 0 ? (unsigned)v : 80;
}

/*
 * ⚠⚠ THE HEIGHT AXIS HAS A CEILING OF ITS OWN, AND PHASE 19 WALKED IT:
 *
 *   surf x rows   4096  6144  7680  8192   exact
 *                 8960 10240               WRONG, every row, every channel
 *
 * with surf alone (128 at m = 32) and the output size (122880 at K = 3072)
 * both exact, so the quantity is the input surface (K / 32) * rows and the
 * line is in (8192, 8960]. 80 rows is only safe up to K = 3276: SmolVLM's
 * K = 3072 sits at 7680 and passed by 512; whisper medium's K = 4096 at 80
 * rows is 10240 and would be wrong on every row. So the chunk is the
 * smaller of rows_max and what 8192 entries allow at this K.
 */
static unsigned rows_fit(const struct charsiu_npu_pool *p, uint64_t k)
{
	unsigned cap = rows_max();
	/*
	 * ⚠ THE SLICE, NOT THE TENSOR. npudev slices a tensor wider than its
	 * kmax into dispatches of at most kmax, and the surface ceiling is per
	 * DISPATCH. The first version of this took the whole K: SmolVLM's
	 * idefics3 projector is K = 12288, which is three slices of 4096 and
	 * 8192 entries at 64 rows -- exactly at the line and exact on the board
	 * all day -- and this computed 21 rows for it, chunked 64 into
	 * 21 + 21 + 21 + 1, the batched path refused the row of one, and the
	 * whole projector went to the CPU: 20 ms became 512 and the pool said
	 * "1 fell back". A rule about a dispatch has to be applied to a dispatch.
	 */
	uint64_t slice = charsiu_npu_kmax(p->dev);
	uint64_t ks = k < slice ? k : slice;
	unsigned by_surface = ks >= 32 ? (unsigned)((8192u * 32u) / ks) : cap;

	if (by_surface < 2)
		by_surface = 2;
	return by_surface < cap ? by_surface : cap;
}

/*
 * The next chunk: never leave a tail of one row, which the batched path
 * refuses and which would send the whole call to the CPU.
 */
static unsigned rows_next(unsigned rem, unsigned chunk)
{
	unsigned c = rem < chunk ? rem : chunk;

	if (rem - c == 1 && c > 2)
		c--;
	return c;
}

static int pool_id(struct charsiu_npu_pool *p, const struct gguf_tensor *w,
		   unsigned *id)
{
	unsigned i;

	if (!p->dev || !charsiu_pool_get(p, w))
		return -1;
	for (i = 0; i < p->n; i++)
		if (p->key[i] == w) {
			if (p->id[i] < 0)
				return -1;
			*id = (unsigned)p->id[i];
			return 0;
		}
	return -1;
}

int charsiu_pool_rowsn(struct charsiu_npu_pool *p,
		       const struct gguf_tensor *const *ws, unsigned nw,
		       const float *X, unsigned m, float *const *Ys)
{
	unsigned ids[8], chunk, done = 0, i;
	uint64_t k, ns[8];
	double t0;

	if (nw < 1 || nw > 8)
		return -1;
	p->calls += nw;
	for (i = 0; i < nw; i++)
		if (pool_id(p, ws[i], &ids[i]))
			return -1;
	k = ws[0]->ne[0];
	for (i = 0; i < nw; i++) {
		if (ws[i]->ne[0] != k)
			return -1;     /* one input means one K */
		ns[i] = ws[i]->n_dims ? ws[i]->ne[ws[i]->n_dims - 1] : 1;
	}
	chunk = rows_fit(p, k);
	t0 = now_ms();
	while (done < m) {
		unsigned c = rows_next(m - done, chunk);
		const float *x = X + (size_t)done * k;

		/*
		 * ⚠ THE CHUNK IS THE UNIT OF REUSE. Whole-tensor calls in a
		 * row would pack every chunk once per tensor, because the
		 * input BO only ever holds the LAST chunk packed; all the
		 * projections on one chunk before the next is what makes the
		 * packs after the first vanish.
		 */
		for (i = 0; i < nw; i++) {
			float *y = Ys[i] + (size_t)done * ns[i];
			int rc = i == 0
			       ? charsiu_npu_matmul(p->dev, (int)ids[0], x, c, y)
			       : charsiu_npu_matmul_same(p->dev, (int)ids[i], x,
							 c, y);

			if (rc) {
				p->fell_back += nw;
				p->hw_ms += now_ms() - t0;
				return -1;
			}
		}
		done += c;
	}
	p->hw += nw;
	p->rows_hw += nw * m;
	p->hw_ms += now_ms() - t0;
	return 0;
}

int charsiu_pool_rows3(struct charsiu_npu_pool *p,
		       const struct gguf_tensor *w0, const struct gguf_tensor *w1,
		       const struct gguf_tensor *w2, const float *X, unsigned m,
		       float *Y0, float *Y1, float *Y2)
{
	const struct gguf_tensor *ws[3] = { w0, w1, w2 };
	float *Ys[3] = { Y0, Y1, Y2 };

	return charsiu_pool_rowsn(p, ws, 3, X, m, Ys);
}

int charsiu_pool_rows(struct charsiu_npu_pool *p, const struct gguf_tensor *w,
		      const float *X, unsigned m, float *Y)
{
	unsigned i, id = 0, found = 0, chunk, done = 0;
	uint64_t k, n;
	double t0;

	p->calls++;
	if (!p->dev || !charsiu_pool_get(p, w))
		return -1;
	for (i = 0; i < p->n; i++)
		if (p->key[i] == w) {
			if (p->id[i] < 0)
				return -1;
			id = (unsigned)p->id[i];
			found = 1;
			break;
		}
	if (!found)
		return -1;

	k = w->ne[0];
	n = w->n_dims ? w->ne[w->n_dims - 1] : 1;
	chunk = rows_fit(p, k);
	t0 = now_ms();
	while (done < m) {
		unsigned c = rows_next(m - done, chunk);

		if (charsiu_npu_matmul(p->dev, (int)id, X + (size_t)done * k, c,
				       Y + (size_t)done * n)) {
			/*
			 * ⚠ A REFUSAL PART WAY THROUGH LEAVES HALF AN ANSWER,
			 * and the caller redoes the whole thing on the CPU, so
			 * the rows already written are overwritten and nothing
			 * is lost. It is counted separately because "it fell
			 * back" and "it never tried" are different facts.
			 */
			p->fell_back++;
			p->hw_ms += now_ms() - t0;
			return -1;
		}
		done += c;
	}
	p->hw++;
	p->rows_hw += m;
	p->hw_ms += now_ms() - t0;
	return 0;
}

/*
 * ⚠ THE ONE LINE THAT SAYS WHETHER ANY OF THIS IS HAPPENING. Without it a run
 * that quietly fell back to the CPU looks exactly like a run that did not, and
 * the only visible difference is a wall clock that did not move -- which is how
 * a 17x got announced from a subtraction.
 */
void charsiu_pool_report(const struct charsiu_npu_pool *p, FILE *out)
{
	unsigned i, on = 0;

	for (i = 0; i < p->n; i++)
		if (p->id[i] >= 0)
			on++;
	fprintf(out, "charsiu NPU pool: %u of %u tensors on the hardware; "
		"%lu matmuls of %lu rows in %.0f ms, %lu asked and %lu fell "
		"back\n", on, p->n, p->hw, p->rows_hw, p->hw_ms, p->calls,
		p->fell_back);
	charsiu_pool_report_batch(p, out);
}

/*
 * ⚠⚠ WHERE A BATCHED MATMUL'S TIME GOES, AND THE PART WITH NO NAME.
 *
 * charsiu_npu_batch_split and charsiu_npu_batch_prep have counted five shares
 * of a batched call since they were written and NOTHING HAS EVER CALLED THEM.
 * That is the same shape as the switch that was "written, legal, default off"
 * and turned out to corrupt the heap the first time hardware ran it: an
 * instrument nobody runs measures nothing, and its absence is why the only
 * figure anyone can quote about prefill is a whole-run wall clock.
 *
 * It matters now because the two gaps to the vendor are not the same size. On
 * Qwen3-0.6B decode is 19.70 tok/s against 24.85, which is 1.26x, and time to
 * first token is 1588 ms against 469, which is 3.39x. The distance is in
 * prefill, and this driver's own note says the NPU is idle for 91% of a
 * batched matmul -- so the work is on this side of the ioctl and this is the
 * only thing that can say which part.
 *
 * ⚠ THE UNNAMED ROW IS THE POINT, not the five named ones. A previous round
 * named a 44% share and a 26% one had no name at all; optimising the first
 * while the second is unaccounted is how this tree has been caught before. The
 * denominator is charsiu_npu_batch_wall, the clock around every
 * charsiu_npu_matmul call, so whatever the five do not add up to is printed
 * rather than left out.
 *
 * ⚠ IT IS NOT THE POOL'S hw_ms, WHICH WOULD HAVE BEEN ZERO HERE. hw_ms is only
 * incremented by charsiu_pool_rows, and only vision and whisper call that --
 * llama calls charsiu_npu_matmul directly. Dividing by it would have printed
 * nan or a divide by zero on exactly the workload this was written to explain.
 */
void charsiu_pool_report_batch(const struct charsiu_npu_pool *p, FILE *out)
{
	double pack, sub, fence, read, prep, named, other, wall;
	unsigned nbuf = 0;
	double alloc;

	if (!p->dev)
		return;
	/* reset = 0: reading this must not disturb a run that is still going */
	charsiu_npu_batch_split(p->dev, &pack, &sub, &fence, &read, 0);
	prep = charsiu_npu_batch_prep(p->dev, 0);
	alloc = charsiu_npu_batch_alloc(p->dev, &nbuf, 0);
	wall = charsiu_npu_batch_wall(p->dev, 0);
	named = pack + sub + fence + read + prep;
	if (named <= 0.0 || wall <= 0.0)
		return;              /* nothing took the batched path */
	other = wall - named;

	fprintf(out, "charsiu NPU batched: %.0f ms in the matmul entry, "
		"accounted:\n", wall);
	fprintf(out, "    prep  %8.1f ms  %5.1f%%  buffers, output alloc, "
		"the memset of Y\n", prep, 100.0 * prep / wall);
	fprintf(out, "    pack  %8.1f ms  %5.1f%%  gather, quantise, pack into "
		"the input BO\n", pack, 100.0 * pack / wall);
	{
		double emit = 0.0, fini = 0.0;

		charsiu_npu_batch_pack_split(p->dev, &emit, &fini, 0);
		if (emit + fini > 0.0)
			fprintf(out, "          of which %8.1f ms emitting register "
				"streams, %8.1f ms in FINI ioctls, %8.1f ms the packing\n",
				emit, fini, pack - emit - fini);
	}
	fprintf(out, "    sub   %8.1f ms  %5.1f%%  the submit ioctls\n",
		sub, 100.0 * sub / wall);
	fprintf(out, "    fence %8.1f ms  %5.1f%%  waiting on the hardware\n",
		fence, 100.0 * fence / wall);
	fprintf(out, "    read  %8.1f ms  %5.1f%%  reading the accumulators "
		"back\n", read, 100.0 * read / wall);
	fprintf(out, "    ----\n");
	fprintf(out, "    other %8.1f ms  %5.1f%%  %s\n", other,
		100.0 * other / wall,
		/*
		 * ⚠ COMPARED AGAINST THE SHARES ANYONE WOULD ACTUALLY TARGET,
		 * which is pack and read. The first board round tested it
		 * against `prep` too, and prep collapsed to 0.2% once the
		 * buffers went into a pool -- so a 3% unnamed share was
		 * "LARGER THAN A NAMED SHARE" on six models of eight, which is
		 * a warning that fires when nothing is wrong and teaches the
		 * reader to skip the line.
		 */
		other > pack || other > read
		? "⚠ LARGER THAN A SHARE WORTH OPTIMISING -- name it first"
		: "unaccounted");
	if (nbuf)
		fprintf(out, "    (%u batch buffer allocations, %.1f ms, inside "
			"prep -- a pool that quietly reallocates is the old "
			"bug in new clothes)\n", nbuf, alloc);
	{
		unsigned long hits = 0, misses = 0;

		charsiu_npu_reuse_stats(p->dev, &hits, &misses);
		/*
		 * ⚠ SAID EVEN WHEN ZERO. A reuse that silently never fired
		 * would read as "pack did not move", which is a different
		 * fact from "the declaration was never honoured".
		 */
		if (hits || misses)
			fprintf(out, "    input reused %lu times, packed anyway "
				"%lu times when declared the same\n",
				hits, misses);
		else if (!charsiu_npu_reuse_on())
			fprintf(out, "    input reuse OFF (CHARSIU_NPU_REUSE=0 "
				"is set)\n");
		else
			fprintf(out, "    input reuse never asked for: no "
				"batched call declared its input unchanged\n");
	}
}

/*
 * ⚠ WHERE A CACHE GOES. Not beside the model: that directory is the user's and
 * an 85 MB file appearing in it unasked is a surprise. XDG_CACHE_HOME is the
 * place for a file that can be deleted without losing anything, and this one
 * can -- it rebuilds in the time it saves.
 */
const char *charsiu_cache_path(const char *model, char *buf, size_t max)
{
	const char *xdg = getenv("XDG_CACHE_HOME");
	const char *home = getenv("HOME");
	const char *base;
	char dir[400];

	if (getenv("CHARSIU_NO_WCACHE"))
		return NULL;
	if (xdg && *xdg)
		snprintf(dir, sizeof(dir), "%s/charsiu", xdg);
	else if (home && *home)
		snprintf(dir, sizeof(dir), "%s/.cache/charsiu", home);
	else
		return NULL;
	if (mkdir(dir, 0755) && errno != EEXIST)
		return NULL;
	base = strrchr(model, '/');
	base = base ? base + 1 : model;
	snprintf(buf, max, "%s/%s.wq", dir, base);
	return buf;
}

int charsiu_pool_stage_all(struct charsiu_npu_pool *p,
			   const struct gguf_tensor *const *w, unsigned n,
			   const char *cache, const char *stamp)
{
	unsigned i, staged = 0;
	double t0 = now_ms();

	if (!p->dev)
		return -1;
	if (cache)
		charsiu_wcache_use(cache, stamp);
	for (i = 0; i < n; i++)
		if (w[i] && charsiu_pool_get(p, w[i]))
			staged++;
	/* ⚠ STAGED IS NOT ROUTED. charsiu_pool_get returns the quantised copy
	 * whether or not the hardware took it, and the first version of this
	 * message said "on the NPU" about both. */
	{
		unsigned k2, on = 0;

		for (k2 = 0; k2 < p->n; k2++)
			if (p->id[k2] >= 0)
				on++;
		staged = on;
	}
	/*
	 * ⚠ HAND IT BACK. The language model stages after this in the same
	 * process when a picture is part of a prompt, and it has its own file.
	 */
	if (cache)
		charsiu_wcache_use(NULL, NULL);
	if (charsiu_diag())
		fprintf(stderr, "charsiu: %u of %u tensors ON THE HARDWARE, "
			"staged in %.0f ms%s\n", staged, n, now_ms() - t0,
			cache ? "" : " (no cache)");
	return 0;
}
