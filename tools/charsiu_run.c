// SPDX-License-Identifier: GPL-2.0-or-later
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */

/*
 * charsiu_run - the CPU decode loop, which is the oracle.
 *
 * Its job is to be checkable, not fast:
 *
 *   --tokens   prints the tokenization and stops, so the tokenizer can be
 *              diffed against a reference without running the model;
 *   --logits N prints the top N logits after the prompt, which is the surface
 *              a version with the NPU under the projections is compared on;
 *   greedy by default, because a sampler that reaches for a random number is
 *              not something two runs can be diffed against.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "charsiu_llm.h"

static double now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void usage(void)
{
	fprintf(stderr,
"usage: charsiu_run MODEL.gguf [options]\n"
"  -p TEXT       the prompt (default: a short one)\n"
"  -f FILE       read the prompt from a file\n"
"  --chat        wrap the prompt in the Llama 3 chat template\n"
"  --sys TEXT    the system message, with --chat\n"
"  -n N          how many tokens to generate (default 64)\n"
"  -c N          context, in tokens (default: the model's training context)\n"
"  -t N          threads (default: one per online cpu)\n"
"  --temp T      sample at temperature T instead of greedy\n"
"  --top-p P     nucleus cutoff, with --temp\n"
"  --seed S      the sampler seed\n"
"  --tokens      print the tokenization and stop\n"
"  --logits N    print the top N logits after the prompt and stop\n"
"  --info        print what the file says about the model and stop\n"
"  --no-bos      do not prepend the begin-of-text token\n"
"  --ignore-eos  keep going past end-of-generation, for a longer diff\n"
"  -q            do not echo the prompt\n");
}

int main(int argc, char **argv)
{
	const char *path = NULL, *prompt = NULL, *promptfile = NULL;
	const char *sys = "You are a helpful assistant.";
	int n_gen = 64, n_ctx = 0, nthreads = 0, chat = 0, quiet = 0;
	int show_tokens = 0, show_logits = 0, show_info = 0, add_bos = 1;
	int ignore_eos = 0;
	float temp = 0.0f, top_p = 0.9f;
	uint64_t seed = 1234;
	struct llama_model m;
	struct llama_state *st;
	int32_t *ids;
	int n_ids, max_ids;
	char *text = NULL;
	double t_load, t0, t_prompt;
	int i;

	for (i = 1; i < argc; i++) {
		const char *a = argv[i];

#define NEXT() (i + 1 < argc ? argv[++i] : (usage(), exit(2), ""))
		if (a[0] != '-' && !path) path = a;
		else if (!strcmp(a, "-p")) prompt = NEXT();
		else if (!strcmp(a, "-f")) promptfile = NEXT();
		else if (!strcmp(a, "--sys")) sys = NEXT();
		else if (!strcmp(a, "--chat")) chat = 1;
		else if (!strcmp(a, "-n")) n_gen = atoi(NEXT());
		else if (!strcmp(a, "-c")) n_ctx = atoi(NEXT());
		else if (!strcmp(a, "-t")) nthreads = atoi(NEXT());
		else if (!strcmp(a, "--temp")) temp = (float)atof(NEXT());
		else if (!strcmp(a, "--top-p")) top_p = (float)atof(NEXT());
		else if (!strcmp(a, "--seed")) seed = strtoull(NEXT(), NULL, 0);
		else if (!strcmp(a, "--tokens")) show_tokens = 1;
		else if (!strcmp(a, "--logits")) show_logits = atoi(NEXT());
		else if (!strcmp(a, "--info")) show_info = 1;
		else if (!strcmp(a, "--no-bos")) add_bos = 0;
		else if (!strcmp(a, "--ignore-eos")) ignore_eos = 1;
		else if (!strcmp(a, "-q")) quiet = 1;
		else { usage(); return 2; }
#undef NEXT
	}
	if (!path) {
		usage();
		return 2;
	}
	if (nthreads > 0) {
		char buf[32];

		/* the pool reads this on first use */
		snprintf(buf, sizeof(buf), "%d", nthreads);
		setenv("CHARSIU_THREADS", buf, 1);
	}

	t0 = now_ms();
	if (llama_load(&m, path) < 0)
		return 1;
	t_load = now_ms() - t0;

	if (show_info) {
		printf("file        %s\n", path);
		printf("gguf        v%u, %llu tensors, %llu keys\n", m.gguf.version,
		       (unsigned long long)m.gguf.n_tensors,
		       (unsigned long long)m.gguf.n_kv);
		printf("n_embd      %u\n", m.n_embd);
		printf("n_layer     %u\n", m.n_layer);
		printf("n_head      %u  (kv %u, head_dim %u)\n", m.n_head, m.n_head_kv, m.head_dim);
		printf("n_ff        %u\n", m.n_ff);
		printf("n_vocab     %u\n", m.n_vocab);
		printf("n_ctx_train %u\n", m.n_ctx_train);
		printf("rms_eps     %g\n", (double)m.rms_eps);
		printf("rope_base   %g%s\n", (double)m.rope_base,
		       m.rope_freqs ? "  (with a rope_freqs table)" : "");
		printf("tok_embd    %s\n", ggml_type_name(m.tok_embd->type));
		printf("attn_q[0]   %s\n", ggml_type_name(m.layers[0].wq->type));
		printf("output      %s%s\n", ggml_type_name(m.output->type),
		       m.output == m.tok_embd ? "  (tied to the embedding)" : "");
		printf("load        %.0f ms\n", t_load);
		llama_free(&m);
		return 0;
	}

	if (promptfile) {
		FILE *fp = fopen(promptfile, "rb");
		long n;

		if (!fp) {
			perror(promptfile);
			return 1;
		}
		fseek(fp, 0, SEEK_END);
		n = ftell(fp);
		fseek(fp, 0, SEEK_SET);
		text = malloc((size_t)n + 1);
		if (fread(text, 1, (size_t)n, fp) != (size_t)n) {
			perror(promptfile);
			return 1;
		}
		text[n] = 0;
		fclose(fp);
		prompt = text;
	}
	if (!prompt)
		prompt = "The capital of France is";

	if (chat) {
		size_t n = strlen(prompt) + strlen(sys) + 256;
		char *c = malloc(n);

		snprintf(c, n,
			 "<|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|>"
			 "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>"
			 "<|start_header_id|>assistant<|end_header_id|>\n\n",
			 sys, prompt);
		prompt = c;
	}

	max_ids = (int)strlen(prompt) + 64;
	ids = malloc((size_t)max_ids * sizeof(*ids));
	n_ids = tokenizer_encode(m.tk, prompt, add_bos, ids, max_ids);
	if (n_ids < 0) {
		fprintf(stderr, "charsiu_run: the prompt did not tokenize\n");
		return 1;
	}

	if (show_tokens) {
		printf("%d tokens\n", n_ids);
		for (i = 0; i < n_ids; i++) {
			int len;
			const char *s = tokenizer_decode(m.tk, ids[i], &len);

			printf("%5d  %6d  '", i, ids[i]);
			fwrite(s, 1, (size_t)len, stdout);
			printf("'\n");
		}
		llama_free(&m);
		return 0;
	}

	if (!n_ids) {
		fprintf(stderr, "charsiu_run: the prompt is empty\n");
		return 1;
	}

	st = llama_state_new(&m, n_ctx);
	if (!st) {
		fprintf(stderr, "charsiu_run: no room for a %d token context\n",
			n_ctx ? n_ctx : (int)m.n_ctx_train);
		return 1;
	}
	if (n_ids >= st->n_ctx) {
		fprintf(stderr, "charsiu_run: the prompt is longer than the context\n");
		return 1;
	}

	if (!quiet) {
		for (i = 0; i < n_ids; i++) {
			int len;
			const char *s = tokenizer_decode(m.tk, ids[i], &len);

			fwrite(s, 1, (size_t)len, stdout);
		}
		fflush(stdout);
	}

	t0 = now_ms();
	const float *logits = NULL;

	for (i = 0; i < n_ids; i++)
		logits = llama_forward(st, ids[i], i);
	t_prompt = now_ms() - t0;
	/*
	 * ⚠ THE STAGE TIMERS START HERE, NOT AT THE FIRST TOKEN. Round 327
	 * read its own stage table as a decode split and it was 86% the
	 * PROMPT: six tokens that fault 1.3 GB of weights in off the card
	 * and build the NPU tensors take 31 seconds, and dividing by 38
	 * spreads that over every stage. The host check I built it against
	 * had the same defect and I quoted it anyway.
	 */
	llama_stages_reset();

	/*
	 * CHARSIU_LOGIT_DUMP writes the WHOLE logit vector, because the top few
	 * are not a metric: in this model ranks two to five sit within one
	 * point of each other, so their ORDER moves under any perturbation and
	 * says nothing about how large the perturbation is. A quantisation is
	 * compared against the f32 run over the whole vector or not at all.
	 */
	{
		const char *dp = getenv("CHARSIU_LOGIT_DUMP");

		if (dp) {
			FILE *f = fopen(dp, "wb");

			if (f) {
				fwrite(logits, sizeof(float), m.n_vocab, f);
				fclose(f);
				fprintf(stderr, "logits: wrote %u floats to %s\n",
					m.n_vocab, dp);
			}
		}
	}
	if (show_logits) {
		int n = show_logits;

		if (n > (int)m.n_vocab)
			n = (int)m.n_vocab;
		printf("\n--- top %d after %d prompt tokens ---\n", n, n_ids);
		{
			int *idx = malloc((size_t)m.n_vocab * sizeof(int));
			float *lc = malloc((size_t)m.n_vocab * sizeof(float));

			memcpy(lc, logits, (size_t)m.n_vocab * sizeof(float));
			for (uint32_t j = 0; j < m.n_vocab; j++)
				idx[j] = (int)j;
			for (int r = 0; r < n; r++) {
				int b = r;

				for (uint32_t j = (uint32_t)r + 1; j < m.n_vocab; j++)
					if (lc[idx[j]] > lc[idx[b]])
						b = (int)j;
				{ int t = idx[r]; idx[r] = idx[b]; idx[b] = t; }
				{
					int len;
					const char *s = tokenizer_decode(m.tk, idx[r], &len);

					printf("%2d  %6d  %12.6f  '", r, idx[r],
					       (double)lc[idx[r]]);
					fwrite(s, 1, (size_t)len, stdout);
					printf("'\n");
				}
			}
			free(idx);
			free(lc);
		}
		llama_state_free(st);
		llama_free(&m);
		return 0;
	}

	t0 = now_ms();
	int produced = 0;

	for (i = 0; i < n_gen; i++) {
		int32_t tok = temp > 0.0f
			? llama_sample(logits, m.n_vocab, temp, top_p, &seed)
			: llama_argmax(logits, m.n_vocab);
		int len;
		const char *s;

		if (!ignore_eos && tokenizer_is_eog(m.tk, tok))
			break;
		s = tokenizer_decode(m.tk, tok, &len);
		fwrite(s, 1, (size_t)len, stdout);
		fflush(stdout);
		produced++;

		if (st->pos >= st->n_ctx)
			break;
		logits = llama_forward(st, tok, st->pos);
		if (!logits)
			break;
	}

	{
		double t_gen = now_ms() - t0;
		long hwm = 0;
		FILE *st = fopen("/proc/self/status", "r");

		if (st) {
			char line[256];

			while (fgets(line, sizeof(line), st))
				if (!strncmp(line, "VmHWM:", 6)) {
					hwm = strtol(line + 6, NULL, 10);
					break;
				}
			fclose(st);
		}

		printf("\n\n[load %.0f ms | prompt %d tok in %.0f ms, %.2f tok/s"
		       " | gen %d tok in %.0f ms, %.2f tok/s | peak %ld MB]\n",
		       t_load, n_ids, t_prompt, n_ids * 1000.0 / (t_prompt ? t_prompt : 1),
		       produced, t_gen, produced * 1000.0 / (t_gen ? t_gen : 1),
		       hwm / 1024);
	}

	llama_state_free(st);
	llama_free(&m);
	free(text);
	return 0;
}
