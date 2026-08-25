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
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

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
"  --bos         prepend it even if the file says not to\n"
"  --show-special  print control tokens instead of hiding them\n"
"  --ignore-eos  keep going past end-of-generation, for a longer diff\n"
"  -i, --interactive  a conversation: the model and the NPU tensors stay\n"
"                 staged, so every turn after the first costs a token and\n"
"                 not the twenty seconds staging takes\n"
"  --hold-secs N  after generating, stay alive N seconds with the model\n"
"                 still mapped and the NPU still open, so an idle session\n"
"                 can be measured from outside\n"
"  -q            do not echo the prompt\n");
}

int main(int argc, char **argv)
{
	const char *path = NULL, *prompt = NULL, *promptfile = NULL;
	const char *sys = "You are a helpful assistant.";
	int n_gen = 64, n_ctx = 0, nthreads = 0, chat = 0, quiet = 0;
	int show_tokens = 0, show_logits = 0, show_info = 0;
	/*
	 * ⚠ -1 MEANS ASK THE FILE. tokenizer_encode has always taken that and
	 * gguf's tokenizer.ggml.add_bos_token has always been read into the
	 * tokenizer, but this tool passed a hard 1, so a model whose file says
	 * not to prepend one got one anyway. Llama 3 says true, which is why it
	 * never showed.
	 */
	int add_bos = -1, show_special = 0;
	int ignore_eos = 0;
	float temp = 0.0f, top_p = 0.9f;
	uint64_t seed = 1234;
	struct llama_model m;
	struct llama_state *st;
	int32_t *ids;
	int n_ids, max_ids;
	char *text = NULL;
	/*
	 * --cache writes the quantised weights beside the model the first time
	 * and reads them back after, which is the difference between 20 seconds
	 * of startup and about five. It is OPT IN because it puts 620 MB next
	 * to somebody's model file and doing that unasked is rude.
	 */
	const char *cache = NULL;
	int hold_secs = 0;
	int interactive = 0, turn;
	char cachepath[1024];
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
		else if (!strcmp(a, "--bos")) add_bos = 1;
		else if (!strcmp(a, "--show-special")) show_special = 1;
		else if (!strcmp(a, "--ignore-eos")) ignore_eos = 1;
		else if (!strcmp(a, "--hold-secs")) hold_secs = atoi(NEXT());
		else if (!strcmp(a, "-i") || !strcmp(a, "--interactive"))
			interactive = 1;
		else if (!strcmp(a, "--cache")) cache = "";
		else if (!strcmp(a, "--cache-at")) cache = NEXT();
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
	if (cache) {
		struct stat sb;
		char stamp[64];

		if (!*cache) {
			snprintf(cachepath, sizeof(cachepath), "%s.wcache",
				 path ? path : "model");
			cache = cachepath;
		}
		/*
		 * The stamp is the model's size and mtime. A cache whose header
		 * does not match it is rebuilt rather than trusted: the failure
		 * mode of a stale cache is a slightly wrong sentence, which is
		 * far worse than a slow start.
		 */
		if (path && !stat(path, &sb))
			snprintf(stamp, sizeof(stamp), "%llu:%llu",
				 (unsigned long long)sb.st_size,
				 (unsigned long long)sb.st_mtime);
		else
			snprintf(stamp, sizeof(stamp), "nostat");
		setenv("CHARSIU_NPU_CACHE", cache, 1);
		setenv("CHARSIU_NPU_CACHE_STAMP", stamp, 1);
	}

	enum chat_fmt fmt = CHAT_LLAMA3;

	if (llama_load(&m, path) < 0)
		return 1;
	t_load = now_ms() - t0;
	fmt = chat_format_of(m.tk);

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
	/* whether -p was actually given decides if turn 0 asks for a line */
	int prompt_given = prompt != NULL;

	if (!prompt)
		prompt = "The capital of France is";

	/*
	 * ⚠ INTERACTIVE IMPLIES THE CHAT TEMPLATE. A conversation fed as raw
	 * completion text has no turn structure for the model to close, which
	 * is exactly how rounds 372-377 ended up with a model that opens a
	 * header and never shuts it.
	 */
	if (interactive)
		chat = 1;

	if (chat && !interactive) {
		size_t n = strlen(prompt) + strlen(sys) + 256;
		char *c = malloc(n);

		size_t k = chat_turn(c, n, fmt, "system", sys);

		k += chat_turn(c + k, n - k, fmt, "user", prompt);
		chat_open(c + k, n - k, fmt, "assistant");
		prompt = c;
	}

	/*
	 * ⚠ THE STATE IS BUILT ONCE AND EVERY TURN SHARES IT. That is the whole
	 * point of -i: staging the NPU tensors takes about twenty seconds, so a
	 * conversation that reloaded per turn would be unusable. The kv cache
	 * carries the history, which is also why the second turn feeds its
	 * tokens at st->pos rather than at zero.
	 */
	st = llama_state_new(&m, n_ctx);
	if (!st) {
		fprintf(stderr, "charsiu_run: no room for a %d token context\n",
			n_ctx ? n_ctx : (int)m.n_ctx_train);
		return 1;
	}

	max_ids = 8192;
	ids = malloc((size_t)max_ids * sizeof(*ids));
	if (!ids)
		return 1;

	char *turnbuf = NULL;
	size_t turnbuf_n = 0;

	for (turn = 0; ; turn++) {
	const char *feed = prompt;

	if (interactive) {
		char line[4096];

		if (turn > 0 || !prompt_given) {
			fputs("\n> ", stdout);
			fflush(stdout);
			if (!fgets(line, sizeof(line), stdin)) {
				fputs("\n", stdout);
				break;
			}
			line[strcspn(line, "\n")] = 0;
			if (!*line)
				continue;
			if (!strcmp(line, "/quit") || !strcmp(line, "/exit"))
				break;
		} else {
			snprintf(line, sizeof(line), "%s", prompt);
		}

		/*
		 * ⚠ Turn 0 opens with the system message; later turns must
		 * first CLOSE the assistant turn the model just finished. The
		 * generation loop breaks ON the end-of-turn token without
		 * feeding it, so <|eot_id|> is not in the cache yet and the
		 * next turn has to supply it.
		 */
		size_t need = strlen(line) + strlen(sys) + 512;
		if (need > turnbuf_n) {
			free(turnbuf);
			turnbuf = malloc(need);
			if (!turnbuf)
				break;
			turnbuf_n = need;
		}
		size_t k = 0;

		/* turn two onwards has to close the reply the model just gave:
		 * generation stops AT the end marker without emitting it. */
		if (turn > 0)
			k = chat_close(turnbuf, turnbuf_n, fmt);
		else
			k = chat_turn(turnbuf, turnbuf_n, fmt, "system", sys);
		k += chat_turn(turnbuf + k, turnbuf_n - k, fmt, "user", line);
		chat_open(turnbuf + k, turnbuf_n - k, fmt, "assistant");
		feed = turnbuf;
	}

	/* ⚠ add_bos only once: a second one mid-conversation is a new document */
	n_ids = tokenizer_encode(m.tk, feed, turn == 0 ? add_bos : 0, ids, max_ids);
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

	/* ⚠ what is LEFT, not what the context is: turn five starts at st->pos */
	if (st->pos + n_ids + n_gen >= st->n_ctx) {
		if (interactive) {
			printf("\n[the %d token context is full -- /quit and start again]\n",
			       st->n_ctx);
			break;
		}
		fprintf(stderr, "charsiu_run: the prompt is longer than the context\n");
		return 1;
	}

	/*
	 * Echo the prompt in one-shot mode so a log reads as one piece of text.
	 * In a conversation the user just typed it, and echoing it back with the
	 * template markers around it would be noise.
	 */
	if (!quiet && !interactive) {
		for (i = 0; i < n_ids; i++) {
			int len;
			const char *s = tokenizer_decode(m.tk, ids[i], &len);

			fwrite(s, 1, (size_t)len, stdout);
		}
		fflush(stdout);
	}

	t0 = now_ms();
	const float *logits = NULL;

	/* ⚠ st->pos, not i: turn two starts where turn one stopped */
	for (i = 0; i < n_ids; i++)
		logits = llama_forward(st, ids[i], st->pos);
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
	/*
	 * ⚠ A ROLE MARKER IS THREE TOKENS AND ONLY TWO OF THEM ARE CONTROL.
	 *
	 * Llama 3 writes a turn header as <|start_header_id|>assistant
	 * <|end_header_id|>, and the middle one is an ORDINARY TEXT token.
	 * Round 374 hid the two markers by type and left the word, so from
	 * round 374 to 377 the long generation arms printed a bare "assistant"
	 * glued to the previous sentence -- which reads as if the model had
	 * said it, and is worse than showing the tags was.
	 *
	 * The whole header is suppressed now, markers and content.
	 *
	 * ⚠ AND IT BUFFERS RATHER THAN DROPS, because the first two attempts at
	 * this both lost text. Pushed past its own end with --ignore-eos on a
	 * prompt that was never in chat format, this model opens a header and
	 * NEVER CLOSES IT: the stream reads
	 * <|eot_id|><|start_header_id|>assistant | and then plain text, with
	 * <|end_header_id|> appearing ZERO times in 384 tokens.
	 *
	 * Suppressing until the close latched and ate everything after. Giving
	 * up after a budget ate the budget. So the tokens after the marker are
	 * HELD: if the close arrives they are dropped, and if it does not they
	 * are printed. A state machine reading model output must assume the
	 * closing token may never come, and must not lose anything when it
	 * does not.
	 */
	int32_t hdr0 = tokenizer_find(m.tk, "<|start_header_id|>");
	int32_t hdr1 = tokenizer_find(m.tk, "<|end_header_id|>");
	int in_header = 0, eog_at = -1, nheld = 0, heldn = 0;
	const int HDR_MAX = 8;
	/*
	 * ⚠ THE BYTES, NOT THE POINTER. tokenizer_decode hands back a pointer
	 * into a single reused per-thread buffer for anything that is not a
	 * control token, so holding the pointer and printing it later prints
	 * whatever the LATEST call left there. The second attempt at this did
	 * exactly that and emitted "el\0\0\0ribeel\0\0el" where the text
	 * should have been.
	 */
	char held[1024];

	for (i = 0; i < n_gen; i++) {
		int32_t tok = temp > 0.0f
			? llama_sample(logits, m.n_vocab, temp, top_p, &seed)
			: llama_argmax(logits, m.n_vocab);
		int len;
		const char *s;

		if (tokenizer_is_eog(m.tk, tok)) {
			if (!ignore_eos)
				break;
			/*
			 * ⚠ WHERE THE MODEL WANTED TO STOP. Past this point
			 * the text is a continuation of a finished turn, and
			 * greedy decoding on an unchanged context regenerates
			 * near-identical paragraphs -- which is what every 384
			 * token arm since round 372 has shown and what
			 * --ignore-eos means. Reporting it stops the log from
			 * reading as though the model had lost the thread.
			 */
			if (eog_at < 0)
				eog_at = i;
		}
		s = tokenizer_decode(m.tk, tok, &len);
		if (tok == hdr0 && !show_special) {
			in_header = 1;
			nheld = heldn = 0;
			produced++;
			goto next;
		}
		if (in_header) {
			if (tok == hdr1) {          /* a whole header: drop it */
				in_header = 0;
				nheld = heldn = 0;
				produced++;
				goto next;
			}
			if (nheld < HDR_MAX &&
			    heldn + len <= (int)sizeof(held)) {
				memcpy(held + heldn, s, (size_t)len);
				heldn += len;
				nheld++;
				produced++;
				goto next;
			}
			/* it never closed -- print what was held and carry on */
			in_header = 0;
			fwrite(held, 1, (size_t)heldn, stdout);
			nheld = heldn = 0;
		}
		/*
		 * ⚠ A CONTROL TOKEN IS NOT TEXT. decode hands back its literal
		 * spelling, and this loop used to write whatever came back, so
		 * a sampled <|eot_id|> or <|start_header_id|> was printed as
		 * those characters. It STILL COUNTS and it still goes back into
		 * the model: it cost a forward pass and it is real context.
		 * Only the printing is suppressed. --show-special prints it
		 * anyway, which is how this was diagnosed.
		 */
		if (show_special || !tokenizer_is_control(m.tk, tok)) {
			fwrite(s, 1, (size_t)len, stdout);
			fflush(stdout);
		}
		produced++;
next:

		if (st->pos >= st->n_ctx)
			break;
		logits = llama_forward(st, tok, st->pos);
		if (!logits)
			break;
	}

	{
		double t_gen = now_ms() - t0;
		long hwm = 0;
		/* ⚠ NOT `st`: that is the llama state, and shadowing it here
		 * made the interactive report read st->pos off a FILE. */
		FILE *pf = fopen("/proc/self/status", "r");

		if (pf) {
			char line[256];

			while (fgets(line, sizeof(line), pf))
				if (!strncmp(line, "VmHWM:", 6)) {
					hwm = strtol(line + 6, NULL, 10);
					break;
				}
			fclose(pf);
		}

		if (interactive)
			printf("\n[%d tok, %.1f tok/s, %d/%d context]\n",
			       produced, produced * 1000.0 / (t_gen ? t_gen : 1),
			       st->pos, st->n_ctx);
		else
		printf("\n\n[load %.0f ms | prompt %d tok in %.0f ms, %.2f tok/s"
		       " | gen %d tok in %.0f ms, %.2f tok/s | peak %ld MB]\n",
		       t_load, n_ids, t_prompt, n_ids * 1000.0 / (t_prompt ? t_prompt : 1),
		       produced, t_gen, produced * 1000.0 / (t_gen ? t_gen : 1),
		       hwm / 1024);
		if (!interactive && eog_at >= 0)
			printf("[the model reached its own end at token %d; "
			       "the other %d are --ignore-eos continuing a "
			       "finished turn]\n", eog_at + 1,
			       produced - eog_at - 1);
	}

	if (!interactive)
		break;
	}   /* the turn loop */
	free(turnbuf);

	/*
	 * A battery device spends most of its life here: the model is loaded,
	 * the NPU file descriptors are open, the worker pool exists, and nobody
	 * is talking. Whether the rails drop in this state is the whole question
	 * for an idle session, and it cannot be asked from outside a process
	 * that has already exited. Hold before freeing anything.
	 */
	if (hold_secs > 0) {
		printf("[holding %d s with the model resident]\n", hold_secs);
		fflush(stdout);
		sleep((unsigned int)hold_secs);
		printf("[hold done]\n");
		fflush(stdout);
	}

	llama_state_free(st);
	llama_free(&m);
	free(text);
	return 0;
}
