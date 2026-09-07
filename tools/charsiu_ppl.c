/*
 * charsiu_ppl -- perplexity, so a quantiser change stops being a matter of
 * reading one paragraph and deciding it looks fine.
 *
 * ⚠⚠ IT HAS TO RUN ON THE BOARD. The host has no NPU, so npu_get returns NULL
 * and every matvec falls back to the gguf weights -- which measures llama.cpp's
 * q4_0 and not charsiu's int4 at all. On the card the staged, requantised
 * weights are the ones in the loop, which is the thing under test.
 *
 * The number is exp of the mean negative log likelihood of each token given
 * everything before it, over the token loop, one position at a time. No
 * batching: the batched path is a different arithmetic and this is meant to
 * price the WEIGHTS.
 *
 * ⚠ The first token has no prediction to score and is skipped. A run reports
 * how many positions it actually scored, because a comparison between two
 * arms is only a comparison if both scored the same ones.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charsiu_llm.h"

int main(int argc, char **argv)
{
	struct llama_model m;
	struct llama_state *st;
	struct tokenizer *tk;
	int32_t *ids = NULL;
	char *text = NULL;
	size_t tlen = 0, cap = 0;
	int n_ctx = 512, want = 0, n = 0, i, scored = 0;
	double nll = 0.0;
	const char *path = NULL, *tfile = NULL;
	FILE *f;

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-c") && i + 1 < argc)
			n_ctx = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-n") && i + 1 < argc)
			want = atoi(argv[++i]);
		else if (!path)
			path = argv[i];
		else
			tfile = argv[i];
	}
	if (!path || !tfile) {
		fprintf(stderr, "usage: charsiu_ppl MODEL.gguf TEXT [-c ctx]"
			" [-n tokens]\n");
		return 2;
	}
	f = fopen(tfile, "rb");
	if (!f) { perror(tfile); return 1; }
	while (1) {
		if (tlen + 65536 + 1 > cap) {
			cap = cap ? cap * 2 : 262144;
			text = realloc(text, cap);
			if (!text) { fprintf(stderr, "oom\n"); return 1; }
		}
		size_t got = fread(text + tlen, 1, 65536, f);
		tlen += got;
		if (got < 65536) break;
	}
	fclose(f);
	text[tlen] = 0;

	if (llama_load(&m, path) < 0) { fprintf(stderr, "load failed\n"); return 1; }
	tk = tokenizer_from_gguf(&m.gguf);
	if (!tk) { fprintf(stderr, "no tokenizer\n"); return 1; }
	ids = malloc((tlen + 8) * sizeof(*ids));
	n = tokenizer_encode(tk, text, 1, ids, (int)(tlen + 8));
	if (n < 2) { fprintf(stderr, "text too short: %d tokens\n", n); return 1; }
	if (want && n > want) n = want;
	if (n > n_ctx) n = n_ctx;

	st = llama_state_new(&m, n_ctx);
	if (!st) { fprintf(stderr, "state failed\n"); return 1; }

	for (i = 0; i + 1 < n; i++) {
		const float *lg = llama_forward(st, ids[i], i);
		float mx = lg[0], sum = 0.0f;
		uint32_t j;

		if (!lg) { fprintf(stderr, "forward failed at %d\n", i); return 1; }
		/* log softmax, shifted by the max so the exp cannot overflow */
		for (j = 1; j < m.n_vocab; j++)
			if (lg[j] > mx) mx = lg[j];
		for (j = 0; j < m.n_vocab; j++)
			sum += expf(lg[j] - mx);
		nll -= (double)(lg[ids[i + 1]] - mx) - log((double)sum);
		scored++;
		if ((i & 31) == 31)
			fprintf(stderr, "\r  %d/%d  ppl %.4f", scored, n - 1,
				exp(nll / scored));
	}
	fprintf(stderr, "\r%*s\r", 40, "");
	printf("ppl %.4f  over %d scored positions of %d tokens  (%s)\n",
	       exp(nll / scored), scored, n, path);
	llama_state_free(st);
	tokenizer_free(tk);
	llama_free(&m);
	free(ids); free(text);
	return 0;
}
