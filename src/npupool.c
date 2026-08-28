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
static unsigned rows_max(void)
{
	const char *e = getenv("CHARSIU_NPU_ROWS_MAX");
	int v = e ? atoi(e) : 32;

	return v > 0 ? (unsigned)v : 32;
}

int charsiu_pool_rows(struct charsiu_npu_pool *p, const struct gguf_tensor *w,
		      const float *X, unsigned m, float *Y)
{
	unsigned i, id = 0, found = 0, chunk = rows_max(), done = 0;
	uint64_t k, n;

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
	while (done < m) {
		unsigned c = m - done < chunk ? m - done : chunk;

		if (charsiu_npu_matmul(p->dev, (int)id, X + (size_t)done * k, c,
				       Y + (size_t)done * n))
			return -1;
		done += c;
	}
	return 0;
}
