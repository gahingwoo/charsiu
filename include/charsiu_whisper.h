/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * Whisper: a board that hears.
 *
 * ⚠ NOT A gguf. whisper.cpp has its own container and every model anybody
 * actually has is in it, so this reads that: a magic, eleven int32 hparams, the
 * MEL FILTERBANK, the vocabulary, and then the tensors back to back. Two of
 * those are a gift -- the filterbank and the vocabulary are in the file, so
 * neither has to be recomputed or guessed.
 *
 * The tensors are wrapped in struct gguf_tensor, whose ftype 0 and 1 already
 * mean f32 and f16, so gguf_matvec and gguf_row_f32 work on them unchanged and
 * the encoder is the same matmul the rest of this tree is built out of.
 *
 * ⚠ AND THE ENCODER IS THE BATCHED CASE AGAIN. Thirty seconds of audio is 3000
 * mel frames and 1500 encoder positions, all present at once -- the same shape
 * as an image's patches and a prompt's tokens.
 */
#ifndef CHARSIU_WHISPER_H
#define CHARSIU_WHISPER_H

#include <stdint.h>
#include <stdio.h>
#include "charsiu_llm.h"

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP         160
#define WHISPER_CHUNK_SEC   30
/* 30 s at 16 kHz in hops of 160 */
#define WHISPER_N_FRAMES    3000

struct whisper_block {
	const struct gguf_tensor *attn_ln_w, *attn_ln_b;
	const struct gguf_tensor *q_w, *q_b, *k_w, *v_w, *v_b, *o_w, *o_b;
	/* decoder only */
	const struct gguf_tensor *x_ln_w, *x_ln_b;
	const struct gguf_tensor *xq_w, *xq_b, *xk_w, *xv_w, *xv_b, *xo_w, *xo_b;
	const struct gguf_tensor *mlp_ln_w, *mlp_ln_b;
	const struct gguf_tensor *fc1_w, *fc1_b, *fc2_w, *fc2_b;
};

struct charsiu_whisper {
	int fd;
	const uint8_t *map;
	size_t map_size;

	int32_t n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer;
	int32_t n_text_ctx, n_text_state, n_text_head, n_text_layer;
	int32_t n_mels, ftype;

	/* [n_mel][n_fft_bins], straight out of the file */
	const float *mel_filters;
	int32_t n_mel_filt, n_fft_bins;

	char **vocab;
	int32_t n_vocab_file;
	/*
	 * ⚠ THE SPECIAL IDS ARE NOT IN THE FILE. whisper.cpp assigns them by
	 * arithmetic on n_vocab, and the ENGLISH ONLY models shift them by one
	 * from the multilingual ones. Getting this wrong is a decoder that
	 * never stops, or one that emits timestamps as words.
	 */
	int32_t tok_eot, tok_sot, tok_prev, tok_nosp, tok_not, tok_beg;
	int multilingual;

	struct gguf_tensor *t;
	size_t n_tensors;
	void **owned;            /* tensors copied for alignment */
	size_t n_owned;

	const struct gguf_tensor *conv1_w, *conv1_b, *conv2_w, *conv2_b;
	const struct gguf_tensor *e_pos, *e_ln_w, *e_ln_b;
	const struct gguf_tensor *d_pos, *d_tok, *d_ln_w, *d_ln_b;
	struct whisper_block *enc, *dec;

	char missing[24][80];
	unsigned n_missing;
	char why[192];
};

int charsiu_whisper_open(struct charsiu_whisper *w, const char *path);
void charsiu_whisper_close(struct charsiu_whisper *w);
const char *charsiu_whisper_why_not(const struct charsiu_whisper *w);
void charsiu_whisper_describe(const struct charsiu_whisper *w, FILE *out);

/*
 * 16 kHz mono f32 samples in, log mel spectrogram out.
 *
 * `out` is [n_mels][WHISPER_N_FRAMES], padded with the pad value where the
 * audio ran out. Returns 0, or -1.
 *
 * ⚠ THE NORMALISATION IS PART OF THE MODEL. log10, then clamped to eight
 * decades below the LOUDEST BIN IN THE WHOLE SPECTROGRAM, then (x + 4) / 4.
 * That maximum makes the transform depend on the entire clip, so a frame does
 * not have a value until the last frame has been computed.
 */
int charsiu_whisper_mel(const struct charsiu_whisper *w, const float *pcm,
			size_t n, float *out);

/* A 16 bit PCM WAV, resampled to 16 kHz mono. Returns samples, or NULL. */
float *charsiu_wav_load(const char *path, size_t *n, char *err, size_t errlen);

#endif /* CHARSIU_WHISPER_H */
