// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * What is in a whisper.cpp model, and what a clip of audio looks like to it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu_llm.h"
#include "charsiu_whisper.h"

/*
 * ⚠ A GENERATED SIGNAL, for the same reason the vision probe generates its
 * pixels: a cross check needs both sides to hold the same audio, and passing it
 * costs a decoder before either side has been shown right. Integers only, so
 * the reference computes the identical samples.
 */
static void fake_audio(float *pcm, size_t n)
{
	size_t i;

	for (i = 0; i < n; i++)
		pcm[i] = ((float)(int)(i % 1009u) / 1009.0f) * 2.0f - 1.0f;
}

int main(int argc, char **argv)
{
	struct charsiu_whisper w;
	const char *model = NULL, *audio = NULL;
	int i, want_mel = 0, want_enc = 0, secs = 1, rc = 1;
	float *pcm = NULL, *mel = NULL, *enc = NULL;
	size_t n = 0;

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--mel"))
			want_mel = 1;
		else if (!strcmp(argv[i], "--encode"))
			want_enc = want_mel = 1;
		else if (!strcmp(argv[i], "--audio") && i + 1 < argc)
			audio = argv[++i];
		else if (!strcmp(argv[i], "--seconds") && i + 1 < argc)
			secs = atoi(argv[++i]);
		else if (argv[i][0] != '-' && !model)
			model = argv[i];
	}
	if (!model) {
		fprintf(stderr,
			"usage: charsiu_whisper MODEL.bin [--mel] "
			"[--encode]\n"
			"       [--audio FILE.wav] [--seconds N]\n"
			"\n"
			"  MODEL.bin is a whisper.cpp ggml model, not a gguf.\n");
		return 2;
	}
	if (charsiu_whisper_open(&w, model)) {
		fprintf(stderr, "charsiu_whisper: %s\n",
			charsiu_whisper_why_not(&w));
		charsiu_whisper_describe(&w, stderr);
		charsiu_whisper_close(&w);
		return 1;
	}
	if (!want_mel) {
		charsiu_whisper_describe(&w, stdout);
		charsiu_whisper_close(&w);
		return 0;
	}

	if (audio) {
		char err[256] = "";

		pcm = charsiu_wav_load(audio, &n, err, sizeof(err));
		if (!pcm) {
			fprintf(stderr, "charsiu_whisper: %s\n", err);
			goto done;
		}
	} else {
		n = (size_t)secs * WHISPER_SAMPLE_RATE;
		pcm = malloc(n * sizeof(float));
		if (!pcm)
			goto done;
		fake_audio(pcm, n);
	}
	mel = malloc((size_t)w.n_mels * WHISPER_N_FRAMES * sizeof(float));
	if (!mel || charsiu_whisper_mel(&w, pcm, n, mel)) {
		fprintf(stderr, "charsiu_whisper: the spectrogram failed\n");
		goto done;
	}
	if (!want_enc) {
		printf("mel %d %d\n", w.n_mels, WHISPER_N_FRAMES);
		for (i = 0; i < w.n_mels * WHISPER_N_FRAMES; i++)
			printf("%.7g\n", (double)mel[i]);
		rc = 0;
		goto done;
	}

	enc = malloc((size_t)w.n_audio_ctx * w.n_audio_state * sizeof(float));
	if (!enc || charsiu_whisper_encode(&w, mel, enc)) {
		fprintf(stderr, "charsiu_whisper: the encoder failed\n");
		goto done;
	}
	printf("encoder %d %d\n", w.n_audio_ctx, w.n_audio_state);
	for (i = 0; i < w.n_audio_ctx * w.n_audio_state; i++)
		printf("%.7g\n", (double)enc[i]);
	rc = 0;
done:
	free(pcm);
	free(mel);
	free(enc);
	charsiu_whisper_close(&w);
	return rc;
}
