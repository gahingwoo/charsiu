/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * Print llama.cpp's own logits after a prompt. This is the reference half of
 * tests/forward_cross.py: llama-completion prints text, and text only shows a
 * disagreement once it has already changed a token, so a number is needed.
 *
 * Built against a llama.cpp checkout, not part of charsiu's own build:
 *
 *   cd llama.cpp && cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
 *       -DLLAMA_CURL=OFF && ninja -C build llama llama-completion
 *   gcc -O2 -I include -I ggml/include -o ref_logits path/to/ref_logits.c \
 *       -Lbuild/bin -lllama -lggml -lggml-base -lm -Wl,-rpath,$PWD/build/bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama.h"

int main(int argc, char **argv)
{
	const char *path = argv[1];
	const char *text = argv[2];
	int topk = argc > 3 ? atoi(argv[3]) : 8;
	int dumpn = argc > 4 ? atoi(argv[4]) : 0;

	llama_backend_init();

	struct llama_model_params mp = llama_model_default_params();
	struct llama_model *model = llama_model_load_from_file(path, mp);
	if (!model) { fprintf(stderr, "load failed\n"); return 1; }
	const struct llama_vocab *vocab = llama_model_get_vocab(model);

	struct llama_context_params cp = llama_context_default_params();
	cp.n_ctx = 2048; cp.n_batch = 2048; cp.n_ubatch = 2048; cp.n_threads = 6;
	cp.n_threads_batch = 6;
	struct llama_context *ctx = llama_init_from_model(model, cp);
	if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

	llama_token toks[4096];
	int n = llama_tokenize(vocab, text, (int)strlen(text), toks, 4096, true, true);
	if (n < 0) { fprintf(stderr, "tokenize failed %d\n", n); return 1; }

	printf("%d prompt tokens:", n);
	for (int i = 0; i < n; i++) printf(" %d", toks[i]);
	printf("\n");

	if (llama_decode(ctx, llama_batch_get_one(toks, n))) {
		fprintf(stderr, "decode failed\n"); return 1;
	}
	float *lg = llama_get_logits_ith(ctx, n - 1);
	int nv = llama_vocab_n_tokens(vocab);

	if (dumpn) {
		FILE *f = fopen("/tmp/reflogits/logits.bin", "wb");
		fwrite(lg, 4, (size_t)nv, f);
		fclose(f);
		printf("wrote %d logits\n", nv);
	}

	for (int r = 0; r < topk; r++) {
		int b = 0;
		for (int i = 1; i < nv; i++) if (lg[i] > lg[b]) b = i;
		char buf[64];
		int l = llama_token_to_piece(vocab, b, buf, sizeof(buf) - 1, 0, true);
		if (l < 0) l = 0;
		buf[l] = 0;
		printf("%2d  %6d  %12.6f  '%s'\n", r, b, (double)lg[b], buf);
		lg[b] = -1e30f;
	}
	return 0;
}
