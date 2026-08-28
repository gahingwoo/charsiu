// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * Reading whisper.cpp's container, and the mel spectrogram in front of it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdarg.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "charsiu.h"
#include "charsiu_llm.h"
#include "charsiu_whisper.h"

static void miss(struct charsiu_whisper *w, const char *fmt, ...)
{
	va_list ap;

	if (w->n_missing >= sizeof(w->missing) / sizeof(*w->missing))
		return;
	va_start(ap, fmt);
	vsnprintf(w->missing[w->n_missing++], sizeof(w->missing[0]), fmt, ap);
	va_end(ap);
}

/* ---- the container ------------------------------------------------------- */

struct cur { const uint8_t *p, *end; int bad; };

static int32_t ci32(struct cur *c)
{
	int32_t v = 0;

	if (c->p + 4 > c->end) {
		c->bad = 1;
		return 0;
	}
	memcpy(&v, c->p, 4);
	c->p += 4;
	return v;
}

static const struct gguf_tensor *find(struct charsiu_whisper *w,
				      const char *fmt, int idx, int opt,
				      uint64_t in, uint64_t out)
{
	char name[80];
	size_t i;

	if (idx >= 0)
		snprintf(name, sizeof(name), fmt, idx);
	else
		snprintf(name, sizeof(name), "%s", fmt);
	for (i = 0; i < w->n_tensors; i++) {
		const struct gguf_tensor *t = &w->t[i];
		uint64_t gi = 1, go;
		unsigned d;

		if (strcmp(t->name, name))
			continue;
		for (d = 0; d + 1 < (t->n_dims ? t->n_dims : 1); d++)
			gi *= t->ne[d];
		go = t->n_dims ? t->ne[t->n_dims - 1] : 1;
		/*
		 * ⚠ SHAPE AS WELL AS NAME, for the same reason the vision
		 * loader does it: a name that exists with the wrong shape
		 * contracts over the wrong axis and produces a transcript
		 * rather than an error.
		 */
		if ((in && gi != in) || (out && go != out)) {
			miss(w, "%s is %llux%llu, wanted %llux%llu", name,
			     (unsigned long long)gi, (unsigned long long)go,
			     (unsigned long long)in, (unsigned long long)out);
			return NULL;
		}
		return t;
	}
	if (!opt)
		miss(w, "%s", name);
	return NULL;
}

static void block(struct charsiu_whisper *w, struct whisper_block *B,
		  const char *who, int i, int cross)
{
	char f[80];
	uint32_t W = (uint32_t)(cross ? w->n_text_state : w->n_audio_state);
	uint32_t F = 4 * W;

#define NM(suffix) (snprintf(f, sizeof(f), "%s.blocks.%%d." suffix, who), f)
	B->attn_ln_w = find(w, NM("attn_ln.weight"), i, 0, 0, W);
	B->attn_ln_b = find(w, NM("attn_ln.bias"), i, 0, 0, W);
	B->q_w = find(w, NM("attn.query.weight"), i, 0, W, W);
	B->q_b = find(w, NM("attn.query.bias"), i, 0, 0, W);
	/*
	 * ⚠ THE KEY PROJECTION HAS NO BIAS. Whisper leaves it out -- query,
	 * value and out have one and key does not -- so asking for it as
	 * required would report a miss on every layer of every model.
	 */
	B->k_w = find(w, NM("attn.key.weight"), i, 0, W, W);
	B->v_w = find(w, NM("attn.value.weight"), i, 0, W, W);
	B->v_b = find(w, NM("attn.value.bias"), i, 0, 0, W);
	B->o_w = find(w, NM("attn.out.weight"), i, 0, W, W);
	B->o_b = find(w, NM("attn.out.bias"), i, 0, 0, W);

	if (cross) {
		B->x_ln_w = find(w, NM("cross_attn_ln.weight"), i, 0, 0, W);
		B->x_ln_b = find(w, NM("cross_attn_ln.bias"), i, 0, 0, W);
		B->xq_w = find(w, NM("cross_attn.query.weight"), i, 0, W, W);
		B->xq_b = find(w, NM("cross_attn.query.bias"), i, 0, 0, W);
		B->xk_w = find(w, NM("cross_attn.key.weight"), i, 0, W, W);
		B->xv_w = find(w, NM("cross_attn.value.weight"), i, 0, W, W);
		B->xv_b = find(w, NM("cross_attn.value.bias"), i, 0, 0, W);
		B->xo_w = find(w, NM("cross_attn.out.weight"), i, 0, W, W);
		B->xo_b = find(w, NM("cross_attn.out.bias"), i, 0, 0, W);
	}

	B->mlp_ln_w = find(w, NM("mlp_ln.weight"), i, 0, 0, W);
	B->mlp_ln_b = find(w, NM("mlp_ln.bias"), i, 0, 0, W);
	B->fc1_w = find(w, NM("mlp.0.weight"), i, 0, W, F);
	B->fc1_b = find(w, NM("mlp.0.bias"), i, 0, 0, F);
	B->fc2_w = find(w, NM("mlp.2.weight"), i, 0, F, W);
	B->fc2_b = find(w, NM("mlp.2.bias"), i, 0, 0, W);
#undef NM
}

int charsiu_whisper_open(struct charsiu_whisper *w, const char *path)
{
	struct cur c;
	struct stat sb;
	int32_t magic, i;
	size_t misaligned = 0, cap = 0;

	memset(w, 0, sizeof(*w));
	w->fd = open(path, O_RDONLY);
	if (w->fd < 0 || fstat(w->fd, &sb) < 0) {
		snprintf(w->why, sizeof(w->why), "%s will not open", path);
		return -1;
	}
	w->map_size = (size_t)sb.st_size;
	w->map = mmap(NULL, w->map_size, PROT_READ, MAP_PRIVATE, w->fd, 0);
	if (w->map == MAP_FAILED) {
		w->map = NULL;
		snprintf(w->why, sizeof(w->why), "%s will not map", path);
		return -1;
	}
	c.p = w->map;
	c.end = w->map + w->map_size;
	c.bad = 0;

	magic = ci32(&c);
	if (magic != 0x67676d6c) {
		snprintf(w->why, sizeof(w->why),
			 "%s is not a whisper.cpp model: the magic is 0x%08x, "
			 "not 0x67676d6c", path, (unsigned)magic);
		return -1;
	}
	w->n_vocab       = ci32(&c);
	w->n_audio_ctx   = ci32(&c);
	w->n_audio_state = ci32(&c);
	w->n_audio_head  = ci32(&c);
	w->n_audio_layer = ci32(&c);
	w->n_text_ctx    = ci32(&c);
	w->n_text_state  = ci32(&c);
	w->n_text_head   = ci32(&c);
	w->n_text_layer  = ci32(&c);
	w->n_mels        = ci32(&c);
	w->ftype         = ci32(&c);

	w->n_mel_filt = ci32(&c);
	w->n_fft_bins = ci32(&c);
	if (c.bad || w->n_mel_filt <= 0 || w->n_fft_bins <= 0) {
		snprintf(w->why, sizeof(w->why), "%s: the header is short", path);
		return -1;
	}
	w->mel_filters = (const float *)c.p;
	c.p += (size_t)w->n_mel_filt * w->n_fft_bins * sizeof(float);

	w->n_vocab_file = ci32(&c);
	if (w->n_vocab_file <= 0 || c.bad) {
		snprintf(w->why, sizeof(w->why), "%s: no vocabulary", path);
		return -1;
	}
	w->vocab = calloc((size_t)w->n_vocab, sizeof(*w->vocab));
	if (!w->vocab)
		return -1;
	for (i = 0; i < w->n_vocab_file; i++) {
		uint32_t n;

		if (c.p + 4 > c.end)
			break;
		memcpy(&n, c.p, 4);
		c.p += 4;
		if (c.p + n > c.end)
			break;
		w->vocab[i] = malloc(n + 1);
		if (!w->vocab[i])
			return -1;
		memcpy(w->vocab[i], c.p, n);
		w->vocab[i][n] = 0;
		c.p += n;
	}

	/*
	 * ⚠ THE ENGLISH ONLY MODELS SHIFT EVERY SPECIAL ID BY ONE. whisper.cpp
	 * decides on n_vocab == 51865, and tiny.en is 51864. Off by one here is
	 * a decoder that never emits its own end of text.
	 */
	w->multilingual = (w->n_vocab == 51865);
	w->tok_eot = 50256 + w->multilingual;
	w->tok_sot = 50257 + w->multilingual;
	w->tok_prev = 50360 + w->multilingual;
	w->tok_nosp = 50361 + w->multilingual;
	w->tok_not = 50362 + w->multilingual;
	w->tok_beg = 50363 + w->multilingual;

	/* the tensors, back to back, each one its own little header */
	while (c.p < c.end && !c.bad) {
		struct gguf_tensor t;
		int32_t nd, nl, ft, d;
		uint64_t nel = 1;
		size_t nb;

		nd = ci32(&c);
		nl = ci32(&c);
		ft = ci32(&c);
		if (c.bad || nd < 1 || nd > 4 || nl < 1 || nl > 79)
			break;
		memset(&t, 0, sizeof(t));
		t.n_dims = (unsigned)nd;
		for (d = 0; d < nd; d++) {
			t.ne[d] = (uint64_t)(uint32_t)ci32(&c);
			nel *= t.ne[d];
		}
		for (d = nd; d < 4; d++)
			t.ne[d] = 1;
		if (c.p + nl > c.end)
			break;
		memcpy(t.name, c.p, (size_t)nl);
		t.name[nl] = 0;
		c.p += nl;
		t.type = (uint32_t)ft;
		nb = (size_t)nel * (ft == 0 ? 4 : 2);
		if (c.p + nb > c.end)
			break;
		t.data = c.p;
		t.nbytes = nb;
		/*
		 * ⚠ THE NAMES ARE VARIABLE LENGTH AND NOTHING IS PADDED, so a
		 * tensor's data can begin at an odd address. An f16 read off an
		 * odd pointer is undefined and, on the machines where it is
		 * not, slow. Copy only the ones that need it and say how many.
		 */
		if ((uintptr_t)t.data & 3u) {
			void *aligned = malloc(nb);

			if (!aligned)
				return -1;
			memcpy(aligned, t.data, nb);
			t.data = aligned;
			if (w->n_owned == cap) {
				cap = cap ? cap * 2 : 16;
				w->owned = realloc(w->owned,
						   cap * sizeof(*w->owned));
				if (!w->owned)
					return -1;
			}
			w->owned[w->n_owned++] = aligned;
			misaligned++;
		}
		c.p += nb;

		w->t = realloc(w->t, (w->n_tensors + 1) * sizeof(*w->t));
		if (!w->t)
			return -1;
		w->t[w->n_tensors++] = t;
	}

	{
		uint32_t A = (uint32_t)w->n_audio_state;
		uint32_t T = (uint32_t)w->n_text_state;

		/*
		 * ⚠ conv1 IS [kernel][in][out] AND THAT IS A REAL CONVOLUTION.
		 * Unlike a patch embedding, whose stride equals its kernel,
		 * these overlap: kernel 3, stride 1 then 2, padding 1. It is
		 * three matmuls summed, not one.
		 */
		w->conv1_w = find(w, "encoder.conv1.weight", -1, 0,
				  3u * (uint32_t)w->n_mels, A);
		w->conv1_b = find(w, "encoder.conv1.bias", -1, 0, 0, A);
		w->conv2_w = find(w, "encoder.conv2.weight", -1, 0, 3u * A, A);
		w->conv2_b = find(w, "encoder.conv2.bias", -1, 0, 0, A);
		w->e_pos   = find(w, "encoder.positional_embedding", -1, 0, A,
				  (uint64_t)w->n_audio_ctx);
		w->e_ln_w  = find(w, "encoder.ln_post.weight", -1, 0, 0, A);
		w->e_ln_b  = find(w, "encoder.ln_post.bias", -1, 0, 0, A);
		w->d_pos   = find(w, "decoder.positional_embedding", -1, 0, T,
				  (uint64_t)w->n_text_ctx);
		w->d_tok   = find(w, "decoder.token_embedding.weight", -1, 0, T,
				  (uint64_t)w->n_vocab);
		w->d_ln_w  = find(w, "decoder.ln.weight", -1, 0, 0, T);
		w->d_ln_b  = find(w, "decoder.ln.bias", -1, 0, 0, T);
	}

	w->enc = calloc((size_t)w->n_audio_layer, sizeof(*w->enc));
	w->dec = calloc((size_t)w->n_text_layer, sizeof(*w->dec));
	if (!w->enc || !w->dec)
		return -1;
	for (i = 0; i < w->n_audio_layer; i++)
		block(w, &w->enc[i], "encoder", i, 0);
	for (i = 0; i < w->n_text_layer; i++)
		block(w, &w->dec[i], "decoder", i, 1);

	if (misaligned)
		snprintf(w->why, sizeof(w->why),
			 "%zu tensors were copied to align them", misaligned);

	/*
	 * ⚠ int8, AND SIZED FOR THE ENCODER. n_audio_state by 4 * n_audio_state
	 * covers every 2D weight in it; the decoder's tied output head is 51864
	 * rows and is m = 1 anyway, so leaving it out of max_n costs nothing
	 * and keeps the device small.
	 */
	if (getenv("CHARSIU_NPU") && !w->n_missing) {
		unsigned A = (unsigned)w->n_audio_state;
		unsigned nt = (unsigned)(w->n_audio_layer + w->n_text_layer);

		if (!charsiu_pool_init(&w->pool, nt * 12 + 8, 4 * A, 4 * A, 0))
			w->npu = w->pool.dev != NULL;
	}
	if (w->n_missing) {
		snprintf(w->why, sizeof(w->why),
			 "%u of the model's tensors are not in this file under "
			 "the names this reads", w->n_missing);
		return -1;
	}
	w->why[0] = 0;
	return 0;
}

void charsiu_whisper_close(struct charsiu_whisper *w)
{
	int32_t i;
	size_t k;

	charsiu_pool_fini(&w->pool);

	for (i = 0; i < w->n_vocab && w->vocab; i++)
		free(w->vocab[i]);
	free(w->vocab);
	for (k = 0; k < w->n_owned; k++)
		free(w->owned[k]);
	free(w->owned);
	free(w->t);
	free(w->enc);
	free(w->dec);
	if (w->map)
		munmap((void *)w->map, w->map_size);
	if (w->fd >= 0)
		close(w->fd);
	memset(w, 0, sizeof(*w));
	w->fd = -1;
}

const char *charsiu_whisper_why_not(const struct charsiu_whisper *w)
{
	return w->why[0] ? w->why : NULL;
}

void charsiu_whisper_describe(const struct charsiu_whisper *w, FILE *out)
{
	unsigned i;

	fprintf(out, "whisper model\n");
	fprintf(out, "  audio        %d wide, %d heads, %d layers, %d positions\n",
		w->n_audio_state, w->n_audio_head, w->n_audio_layer,
		w->n_audio_ctx);
	fprintf(out, "  text         %d wide, %d heads, %d layers, %d positions\n",
		w->n_text_state, w->n_text_head, w->n_text_layer, w->n_text_ctx);
	fprintf(out, "  mel          %d bands over %d fft bins\n",
		w->n_mel_filt, w->n_fft_bins);
	fprintf(out, "  vocabulary   %d, %d in the file, %s\n", w->n_vocab,
		w->n_vocab_file,
		w->multilingual ? "multilingual" : "English only");
	fprintf(out, "  markers      sot %d, eot %d, no timestamps %d\n",
		w->tok_sot, w->tok_eot, w->tok_not);
	fprintf(out, "  tensors      %zu\n", w->n_tensors);
	if (!w->n_missing) {
		fprintf(out, "  complete     every tensor this needs is here\n");
		return;
	}
	fprintf(out, "  MISSING      %u:\n", w->n_missing);
	for (i = 0; i < w->n_missing; i++)
		fprintf(out, "    %s\n", w->missing[i]);
}

/* ---- the mel spectrogram ------------------------------------------------- */

/*
 * A discrete Fourier transform, split radix where the length is even and a
 * direct sum where it is not.
 *
 * ⚠ 400 IS NOT A POWER OF TWO. 400 = 2^4 * 25, so this recurses four times and
 * finishes with a 25 point sum, which is 625 multiplies out of the 3000 frames'
 * worth of work and is not worth a mixed radix kernel. whisper.cpp does the
 * same thing for the same reason.
 */
static void dft_naive(const float *in, int n, float *out)
{
	int k, t;

	for (k = 0; k < n; k++) {
		double re = 0.0, im = 0.0;

		for (t = 0; t < n; t++) {
			/*
			 * ⚠⚠ REDUCE k * t MODULO n BEFORE THE ANGLE. At k = t =
			 * 24 the unreduced angle is -145 radians, and an f32
			 * cannot hold that to better than a part in 10^5 -- so
			 * cosf returns a faithful cosine of the WRONG angle.
			 * Measured: the spectrum came out a hundred times less
			 * accurate than numpy's own f32 transform, 2.06e-04
			 * against 2.10e-06, and the mel spectrogram inherited
			 * every bit of it.
			 *
			 * The reduction is exact in integers and the angle is
			 * then at most 2 pi.
			 */
			double a = -2.0 * M_PI * (double)((k * t) % n) /
				   (double)n;

			re += (double)in[t] * cos(a);
			im += (double)in[t] * sin(a);
		}
		out[2 * k] = (float)re;
		out[2 * k + 1] = (float)im;
	}
}

static void fft_rec(const float *in, int n, float *out, float *scratch)
{
	float *even = scratch, *odd = scratch + n / 2;
	float *eo = out + 2 * n, *oo = out + 2 * n + n;
	int i, k;

	if (n == 1) {
		out[0] = in[0];
		out[1] = 0.0f;
		return;
	}
	if (n % 2) {
		dft_naive(in, n, out);
		return;
	}
	for (i = 0; i < n / 2; i++) {
		even[i] = in[2 * i];
		odd[i] = in[2 * i + 1];
	}
	fft_rec(even, n / 2, eo, scratch + n);
	fft_rec(odd, n / 2, oo, scratch + n);

	for (k = 0; k < n / 2; k++) {
		/* the same care, though this angle is already inside 2 pi */
		double a = -2.0 * M_PI * (double)k / (double)n;
		float cr = (float)cos(a), ci = (float)sin(a);
		float re = cr * oo[2 * k] - ci * oo[2 * k + 1];
		float im = cr * oo[2 * k + 1] + ci * oo[2 * k];

		out[2 * k] = eo[2 * k] + re;
		out[2 * k + 1] = eo[2 * k + 1] + im;
		out[2 * (k + n / 2)] = eo[2 * k] - re;
		out[2 * (k + n / 2) + 1] = eo[2 * k + 1] - im;
	}
}

int charsiu_whisper_mel(const struct charsiu_whisper *w, const float *pcm,
			size_t n, float *out)
{
	const int nfft = WHISPER_N_FFT, hop = WHISPER_HOP;
	const int bins = w->n_fft_bins, nmel = w->n_mel_filt;
	float *hann = NULL, *frame = NULL, *spec = NULL, *scratch = NULL;
	float *power = NULL;
	double mx = -1e30;
	int f, i, m, rc = -1;

	hann    = malloc((size_t)nfft * sizeof(float));
	frame   = malloc((size_t)nfft * sizeof(float));
	/* the recursion writes its children past the parent's own 2n floats */
	spec    = calloc((size_t)nfft * 2 * 12, sizeof(float));
	scratch = calloc((size_t)nfft * 12, sizeof(float));
	power   = malloc((size_t)bins * sizeof(float));
	if (!hann || !frame || !spec || !scratch || !power)
		goto out;

	/*
	 * ⚠ PERIODIC, NOT SYMMETRIC. torch.hann_window's default divides by N
	 * and numpy.hanning divides by N - 1. On a 400 point window the two
	 * differ by a quarter of a percent at the edges, which is nothing to
	 * look at and moves every mel bin.
	 */
	for (i = 0; i < nfft; i++)
		hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI *
					      (float)i / (float)nfft));

	for (f = 0; f < WHISPER_N_FRAMES; f++) {
		/*
		 * ⚠⚠ REFLECTED AT THE FRONT AND ZEROS AT THE BACK, WHICH IS NOT
		 * SYMMETRIC. Frame f is centred on sample f * hop, so the first
		 * frames read backwards off the start of the clip and whisper
		 * mirrors them about sample 0. Past the END of the audio it
		 * pads with SILENCE -- thirty seconds of it -- because the
		 * model is always fed a thirty second window whatever it was
		 * given.
		 *
		 * Reflecting at both ends instead, which is what
		 * torch.stft(center=True) and numpy's pad(mode="reflect") do,
		 * fills the empty part of the window with a repeat of the clip.
		 * That is not silence, it is the same speech again, and the
		 * transcript comes back with it in.
		 */
		int start = f * hop - nfft / 2;

		for (i = 0; i < nfft; i++) {
			long s = start + i;

			if (n == 0 || (s >= 0 && (size_t)s >= n)) {
				frame[i] = 0.0f;
				continue;
			}
			if (s < 0)
				s = -s;
			if ((size_t)s >= n)
				frame[i] = 0.0f;
			else
				frame[i] = pcm[s] * hann[i];
		}
		fft_rec(frame, nfft, spec, scratch);
		for (i = 0; i < bins; i++)
			power[i] = spec[2 * i] * spec[2 * i] +
				   spec[2 * i + 1] * spec[2 * i + 1];
		for (m = 0; m < nmel; m++) {
			const float *filt = w->mel_filters + (size_t)m * bins;
			double sum = 0.0;

			for (i = 0; i < bins; i++)
				sum += (double)filt[i] * power[i];
			if (sum < 1e-10)
				sum = 1e-10;
			sum = log10(sum);
			if (sum > mx)
				mx = sum;
			out[(size_t)m * WHISPER_N_FRAMES + f] = (float)sum;
		}
	}

	/*
	 * ⚠ THE CLAMP IS OVER THE WHOLE CLIP, not the frame. Eight decades below
	 * the loudest bin ANYWHERE in the spectrogram, then (x + 4) / 4. Doing
	 * it per frame normalises silence up to speech and the transcript comes
	 * out as a room's worth of hallucinated words.
	 */
	for (i = 0; i < nmel * WHISPER_N_FRAMES; i++) {
		double v = out[i];

		if (v < mx - 8.0)
			v = mx - 8.0;
		out[i] = (float)((v + 4.0) / 4.0);
	}
	rc = 0;
out:
	free(hann); free(frame); free(spec); free(scratch); free(power);
	return rc;
}

/* ---- a wav file ---------------------------------------------------------- */

static uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint16_t rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }

/*
 * ⚠ 16 kHz MONO IS NOT A SUGGESTION. Whisper's positional embedding is 1500
 * long and each position is ten milliseconds of audio at that rate; feed it
 * 44.1 kHz and every word arrives at the wrong time and the model transcribes
 * something shorter and confident. This resamples linearly and says it did.
 */
float *charsiu_wav_load(const char *path, size_t *n, char *err, size_t errlen)
{
	FILE *fp = fopen(path, "rb");
	uint8_t hdr[12];
	uint16_t fmt = 0, ch = 0, bits = 0;
	uint32_t rate = 0, dlen = 0;
	long dpos = -1;
	float *raw = NULL, *out = NULL;
	size_t nraw, i;

	*n = 0;
	if (!fp) {
		snprintf(err, errlen, "%s will not open", path);
		return NULL;
	}
	if (fread(hdr, 1, 12, fp) != 12 || memcmp(hdr, "RIFF", 4) ||
	    memcmp(hdr + 8, "WAVE", 4)) {
		snprintf(err, errlen, "%s is not a RIFF/WAVE file", path);
		goto fail;
	}
	for (;;) {
		uint8_t ch4[8];
		uint32_t sz;

		if (fread(ch4, 1, 8, fp) != 8)
			break;
		sz = rd32(ch4 + 4);
		if (!memcmp(ch4, "fmt ", 4)) {
			uint8_t b[16];

			if (sz < 16 || fread(b, 1, 16, fp) != 16)
				break;
			fmt = rd16(b);
			ch = rd16(b + 2);
			rate = rd32(b + 4);
			bits = rd16(b + 14);
			if (sz > 16)
				fseek(fp, (long)(sz - 16), SEEK_CUR);
		} else if (!memcmp(ch4, "data", 4)) {
			dpos = ftell(fp);
			dlen = sz;
			break;
		} else {
			fseek(fp, (long)sz + (sz & 1), SEEK_CUR);
		}
	}
	if (dpos < 0 || !ch || !rate) {
		snprintf(err, errlen, "%s has no usable data chunk", path);
		goto fail;
	}
	if (fmt != 1 || bits != 16) {
		snprintf(err, errlen,
			 "%s is format %u at %u bits; this reads 16 bit PCM",
			 path, fmt, bits);
		goto fail;
	}
	nraw = dlen / 2 / ch;
	raw = malloc(nraw * sizeof(float));
	if (!raw)
		goto fail;
	fseek(fp, dpos, SEEK_SET);
	for (i = 0; i < nraw; i++) {
		int16_t s[8];
		int k;
		float acc = 0.0f;

		if (ch > 8 || fread(s, 2, ch, fp) != ch) {
			nraw = i;
			break;
		}
		for (k = 0; k < ch; k++)
			acc += (float)s[k] / 32768.0f;
		raw[i] = acc / (float)ch;
	}
	if (rate == WHISPER_SAMPLE_RATE) {
		fclose(fp);
		*n = nraw;
		return raw;
	}
	{
		double ratio = (double)WHISPER_SAMPLE_RATE / (double)rate;
		size_t nout = (size_t)((double)nraw * ratio);

		out = malloc((nout ? nout : 1) * sizeof(float));
		if (!out)
			goto fail;
		for (i = 0; i < nout; i++) {
			double sp = (double)i / ratio;
			size_t i0 = (size_t)sp;
			size_t i1 = i0 + 1 < nraw ? i0 + 1 : nraw - 1;
			float t = (float)(sp - (double)i0);

			out[i] = raw[i0] * (1.0f - t) + raw[i1] * t;
		}
		free(raw);
		fclose(fp);
		*n = nout;
		return out;
	}
fail:
	free(raw);
	free(out);
	if (fp)
		fclose(fp);
	return NULL;
}

/* ---- the audio encoder --------------------------------------------------- */

static void wlayernorm(float *out, const float *x, const float *w,
		       const float *b, unsigned n, float eps)
{
	float mean = 0.0f, var = 0.0f, inv;
	unsigned i;

	for (i = 0; i < n; i++)
		mean += x[i];
	mean /= (float)n;
	for (i = 0; i < n; i++) {
		float d = x[i] - mean;

		var += d * d;
	}
	var /= (float)n;
	inv = 1.0f / sqrtf(var + eps);
	for (i = 0; i < n; i++)
		out[i] = (x[i] - mean) * inv * (w ? w[i] : 1.0f) +
			 (b ? b[i] : 0.0f);
}

/* ⚠ THE TANH APPROXIMATION. Whisper's reference is torch's exact erf GELU, and
 * ggml -- which is what every measured whisper output in the world comes from --
 * uses the tanh one. Following ggml keeps this comparable to the thing people
 * actually run. */
static void wgelu(float *x, size_t n)
{
	size_t i;

	for (i = 0; i < n; i++) {
		float v = x[i];

		x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f *
						(v + 0.044715f * v * v * v)));
	}
}

static void wsoftmax(float *x, unsigned n)
{
	float mx = x[0], sum = 0.0f;
	unsigned i;

	for (i = 1; i < n; i++)
		if (x[i] > mx)
			mx = x[i];
	for (i = 0; i < n; i++) {
		x[i] = expf(x[i] - mx);
		sum += x[i];
	}
	for (i = 0; i < n; i++)
		x[i] /= sum;
}

static struct gguf_tensor wflat(const struct gguf_tensor *t, uint64_t in,
				uint64_t out)
{
	struct gguf_tensor g = *t;

	g.n_dims = 2;
	g.ne[0] = in;
	g.ne[1] = out;
	g.ne[2] = g.ne[3] = 1;
	return g;
}

static struct charsiu_npu_pool *rows_pool;

static void wrows(const struct gguf_tensor *t, const float *bias,
		  const float *X, unsigned m, unsigned k, float *Y,
		  unsigned nout, struct charsiu_act *a)
{
	struct gguf_tensor g = wflat(t, k, nout);
	unsigned r, i;

	/*
	 * ⚠ m > 1 IS THE WHOLE GATE, and it is what keeps the decoder out
	 * without a second condition: it feeds one token at a time and the
	 * encoder feeds 1500 positions.
	 *
	 * ⚠ AND ONLY 2D WEIGHTS. The conv kernels are [tap][in][out] and this
	 * file already gathers them into a temporary, whose address changes
	 * every call -- the pool keys on the pointer, so a temporary would
	 * stage a new tensor each time until the slots ran out.
	 */
	if (rows_pool && m > 1 && t->n_dims <= 2 &&
	    !charsiu_pool_rows(rows_pool, t, X, m, Y)) {
		if (bias)
			for (r = 0; r < m; r++)
				for (i = 0; i < nout; i++)
					Y[(size_t)r * nout + i] += bias[i];
		return;
	}

	for (r = 0; r < m; r++) {
		float *y = Y + (size_t)r * nout;

		charsiu_act_set(a, X + (size_t)r * k, (int)k);
		gguf_matvec(&g, a, y, 0, nout);
		if (bias)
			for (i = 0; i < nout; i++)
				y[i] += bias[i];
	}
}

/*
 * A whole 1D tensor, whatever shape it claims to be.
 *
 * ⚠⚠ THE CONVOLUTION BIASES ARE [1][384], NOT [384]. gguf_row_f32 reads ne[0]
 * elements, so asking for row 0 of one of those returns ONE value and leaves
 * the rest of the caller's buffer as whatever was on the heap -- which came out
 * of the encoder as 3e24 and would, in a quieter model, have come out as a
 * transcript. Flatten every axis into the row length and read once.
 */
static const float *wrow1(const struct gguf_tensor *t, float *buf)
{
	struct gguf_tensor g;
	uint64_t n = 1;
	unsigned d;

	if (!t)
		return NULL;
	g = *t;
	for (d = 0; d < (t->n_dims ? t->n_dims : 1); d++)
		n *= t->ne[d];
	g.n_dims = 2;
	g.ne[0] = n;
	g.ne[1] = 1;
	g.ne[2] = g.ne[3] = 1;
	gguf_row_f32(&g, 0, buf);
	return buf;
}

/*
 * conv1d, kernel 3, padding 1, stride `stride`.
 *
 * ⚠ THE WEIGHT IS [tap][in][out] WITH tap FASTEST, so tap `p`'s slice is not
 * contiguous: it is every third element. Rather than gather it, this walks the
 * three taps and treats each as a matmul over `in` with a stride of 3 between
 * consecutive input channels -- which is what a flattened view of ne[0]=3 makes
 * gguf_matvec do if it is handed the wrong shape, and is exactly the bug this
 * comment exists to prevent. The gather is explicit below and costs one pass.
 */
static int conv1d3(const struct gguf_tensor *wt, const float *bias,
		   const float *in, unsigned n_in, unsigned len,
		   float *out, unsigned n_out, unsigned stride,
		   struct charsiu_act *a)
{
	unsigned olen = (len + stride - 1) / stride;
	unsigned tap, t, c, i;
	float *tapw = malloc((size_t)3 * n_out * n_in * sizeof(float));
	float *col = malloc((size_t)olen * n_in * sizeof(float));
	float *acc = malloc((size_t)olen * n_out * sizeof(float));
	struct gguf_tensor g;

	if (!tapw || !col || !acc) {
		free(tapw); free(col); free(acc);
		return -1;
	}
	memset(out, 0, (size_t)olen * n_out * sizeof(float));
	memset(&g, 0, sizeof(g));

	/*
	 * ⚠ READ THE WEIGHT THROUGH gguf_row_f32 RATHER THAN CASTING IT. The
	 * flattened view is [3 * in][out], so ROW i is output channel i's whole
	 * kernel, contiguous and already laid out [in][tap] with tap fastest.
	 * One call an output channel dequantises f16 or f32 without this file
	 * knowing which, and the three taps fall out of one pass.
	 */
	{
		struct gguf_tensor rv = wflat(wt, 3ull * n_in, n_out);
		float *row = malloc((size_t)3 * n_in * sizeof(float));

		if (!row) {
			free(tapw); free(col); free(acc);
			return -1;
		}
		for (i = 0; i < n_out; i++) {
			gguf_row_f32(&rv, i, row);
			for (c = 0; c < n_in; c++)
				for (tap = 0; tap < 3; tap++)
					tapw[((size_t)tap * n_out + i) * n_in + c] =
						row[c * 3 + tap];
		}
		free(row);
	}

	for (tap = 0; tap < 3; tap++) {
		g.n_dims = 2;
		g.ne[0] = n_in;
		g.ne[1] = n_out;
		g.ne[2] = g.ne[3] = 1;
		g.type = GGML_F32;
		g.data = tapw + (size_t)tap * n_out * n_in;

		/* the input column this tap sees, zero outside the clip */
		for (t = 0; t < olen; t++) {
			long src = (long)t * stride + (long)tap - 1;

			if (src < 0 || src >= (long)len)
				memset(col + (size_t)t * n_in, 0,
				       (size_t)n_in * sizeof(float));
			else
				for (c = 0; c < n_in; c++)
					col[(size_t)t * n_in + c] =
						in[(size_t)c * len + (size_t)src];
		}
		wrows(&g, NULL, col, olen, n_in, acc, n_out, a);
		for (i = 0; i < olen * n_out; i++)
			out[i] += acc[i];
	}
	if (bias)
		for (t = 0; t < olen; t++)
			for (i = 0; i < n_out; i++)
				out[(size_t)t * n_out + i] += bias[i];
	free(tapw); free(col); free(acc);
	return 0;
}

int charsiu_whisper_encode(const struct charsiu_whisper *w, const float *mel,
			   float *out)
{
	unsigned W = (unsigned)w->n_audio_state, H = (unsigned)w->n_audio_head;
	unsigned L = (unsigned)w->n_audio_layer, T = (unsigned)w->n_audio_ctx;
	unsigned F = 4 * W, hd = W / H, l, i, j, h, e;
	float *c1 = NULL, *x = NULL, *xb = NULL, *q = NULL, *k = NULL;
	float *v = NULL, *ff = NULL, *att = NULL, *g1 = NULL, *b1 = NULL;
	float *tmp = NULL, scale;
	struct charsiu_act a;
	int rc = -1, stop = 0;

	if (!hd || hd * H != W || w->n_missing)
		return -1;
	scale = 1.0f / sqrtf((float)hd);
	if (charsiu_act_alloc(&a, (int)(F > W ? F : W)))
		return -1;
	/* set for the length of one encode; see the note in vision.c */
	rows_pool = w->npu ? (struct charsiu_npu_pool *)&w->pool : NULL;

	c1  = malloc((size_t)WHISPER_N_FRAMES * W * sizeof(float));
	x   = malloc((size_t)T * W * sizeof(float));
	xb  = malloc((size_t)T * W * sizeof(float));
	q   = malloc((size_t)T * W * sizeof(float));
	k   = malloc((size_t)T * W * sizeof(float));
	v   = malloc((size_t)T * W * sizeof(float));
	ff  = malloc((size_t)T * F * sizeof(float));
	att = malloc((size_t)T * sizeof(float));
	g1  = malloc((size_t)F * sizeof(float));
	b1  = malloc((size_t)F * sizeof(float));
	tmp = malloc((size_t)WHISPER_N_FRAMES * W * sizeof(float));
	if (!c1 || !x || !xb || !q || !k || !v || !ff || !att || !g1 || !b1 ||
	    !tmp)
		goto out;

	/*
	 * ⚠ A STOP AFTER EACH STAGE, because "every one of 576000 values is
	 * wrong" says nothing about which stage did it. The reference can stop
	 * at the same places and the first one that disagrees is the one to
	 * read.
	 */
	{
		const char *e = getenv("CHARSIU_WHISPER_STOP");

		stop = e ? atoi(e) : 0;
	}

	if (conv1d3(w->conv1_w, wrow1(w->conv1_b, b1), mel,
		    (unsigned)w->n_mels, WHISPER_N_FRAMES, c1, W, 1, &a))
		goto out;
	wgelu(c1, (size_t)WHISPER_N_FRAMES * W);
	if (stop == 1) {
		/* [3000][W], which does not fit `out`: give the first T rows */
		memcpy(out, c1, (size_t)T * W * sizeof(float));
		rc = 0;
		goto out;
	}

	/*
	 * ⚠ conv2 READS ITS INPUT AS [channel][time] AND conv1 WROTE
	 * [time][channel]. Feeding one straight into the other transposes the
	 * whole spectrogram, which stays finite and produces a transcript.
	 */
	for (i = 0; i < W; i++)
		for (j = 0; j < WHISPER_N_FRAMES; j++)
			tmp[(size_t)i * WHISPER_N_FRAMES + j] =
				c1[(size_t)j * W + i];
	if (conv1d3(w->conv2_w, wrow1(w->conv2_b, b1), tmp, W,
		    WHISPER_N_FRAMES, x, W, 2, &a))
		goto out;
	wgelu(x, (size_t)T * W);

	for (i = 0; i < T; i++) {
		gguf_row_f32(w->e_pos, i, g1);
		for (j = 0; j < W; j++)
			x[(size_t)i * W + j] += g1[j];
	}
	if (stop == 2) {
		memcpy(out, x, (size_t)T * W * sizeof(float));
		rc = 0;
		goto out;
	}

	for (l = 0; l < L; l++) {
		struct whisper_block *B = &w->enc[l];

		for (i = 0; i < T; i++)
			wlayernorm(xb + (size_t)i * W, x + (size_t)i * W,
				   wrow1(B->attn_ln_w, g1),
				   wrow1(B->attn_ln_b, b1), W, 1e-5f);
		wrows(B->q_w, wrow1(B->q_b, b1), xb, T, W, q, W, &a);
		wrows(B->k_w, NULL, xb, T, W, k, W, &a);
		wrows(B->v_w, wrow1(B->v_b, b1), xb, T, W, v, W, &a);

		for (h = 0; h < H; h++) {
			unsigned off = h * hd;

			for (i = 0; i < T; i++) {
				const float *qi = q + (size_t)i * W + off;
				float *o = xb + (size_t)i * W + off;

				for (j = 0; j < T; j++) {
					const float *kj = k + (size_t)j * W + off;
					float d = 0.0f;

					for (e = 0; e < hd; e++)
						d += qi[e] * kj[e];
					att[j] = d * scale;
				}
				wsoftmax(att, T);
				for (e = 0; e < hd; e++)
					o[e] = 0.0f;
				for (j = 0; j < T; j++) {
					const float *vj = v + (size_t)j * W + off;
					float wg = att[j];

					for (e = 0; e < hd; e++)
						o[e] += wg * vj[e];
				}
			}
		}
		wrows(B->o_w, wrow1(B->o_b, b1), xb, T, W, q, W, &a);
		for (i = 0; i < T * W; i++)
			x[i] += q[i];

		for (i = 0; i < T; i++)
			wlayernorm(xb + (size_t)i * W, x + (size_t)i * W,
				   wrow1(B->mlp_ln_w, g1),
				   wrow1(B->mlp_ln_b, b1), W, 1e-5f);
		wrows(B->fc1_w, wrow1(B->fc1_b, b1), xb, T, W, ff, F, &a);
		wgelu(ff, (size_t)T * F);
		wrows(B->fc2_w, wrow1(B->fc2_b, b1), ff, T, F, q, W, &a);
		for (i = 0; i < T * W; i++)
			x[i] += q[i];
	}

	for (i = 0; i < T; i++)
		wlayernorm(out + (size_t)i * W, x + (size_t)i * W,
			   wrow1(w->e_ln_w, g1), wrow1(w->e_ln_b, b1), W, 1e-5f);
	rc = 0;
out:
	rows_pool = NULL;
	free(c1); free(x); free(xb); free(q); free(k); free(v);
	free(ff); free(att); free(g1); free(b1); free(tmp);
	charsiu_act_free(&a);
	return rc;
}

/* ---- the text decoder ---------------------------------------------------- */

/*
 * ⚠ THE CROSS ATTENTION KEYS AND VALUES ARE PER CLIP, NOT PER TOKEN. They come
 * from the encoder's 1500 positions, which do not change while a transcript is
 * being generated, so computing them inside the token loop would repeat
 * 1500 x 384 x 384 x 2 multiplies per layer per token -- more work than the
 * whole rest of the decoder. They are built once, here.
 */
struct whisper_decoder {
	const struct charsiu_whisper *w;
	unsigned W, H, hd, L, F, T;
	float **xk, **xv;        /* [layer][1500 * W] */
	float **sk, **sv;        /* [layer][n_text_ctx * W], the self attention cache */
	float *logits;
	struct charsiu_act a;
	float *x, *xb, *q, *att, *ff, *g1, *b1;
	int ok;
};

void charsiu_whisper_decoder_free(struct whisper_decoder *d)
{
	unsigned l;

	if (!d)
		return;
	for (l = 0; l < d->L; l++) {
		if (d->xk) free(d->xk[l]);
		if (d->xv) free(d->xv[l]);
		if (d->sk) free(d->sk[l]);
		if (d->sv) free(d->sv[l]);
	}
	free(d->xk); free(d->xv); free(d->sk); free(d->sv);
	free(d->logits); free(d->x); free(d->xb); free(d->q);
	free(d->att); free(d->ff); free(d->g1); free(d->b1);
	charsiu_act_free(&d->a);
	free(d);
}

struct whisper_decoder *charsiu_whisper_decoder_new(const struct charsiu_whisper *w,
						    const float *encoded)
{
	struct whisper_decoder *d = calloc(1, sizeof(*d));
	unsigned l;

	if (!d)
		return NULL;
	d->w = w;
	d->W = (unsigned)w->n_text_state;
	d->H = (unsigned)w->n_text_head;
	d->L = (unsigned)w->n_text_layer;
	d->T = (unsigned)w->n_audio_ctx;
	d->F = 4 * d->W;
	d->hd = d->W / d->H;
	if (!d->hd || d->hd * d->H != d->W)
		goto fail;

	if (charsiu_act_alloc(&d->a, (int)(d->F > d->W ? d->F : d->W)))
		goto fail;
	d->xk = calloc(d->L, sizeof(*d->xk));
	d->xv = calloc(d->L, sizeof(*d->xv));
	d->sk = calloc(d->L, sizeof(*d->sk));
	d->sv = calloc(d->L, sizeof(*d->sv));
	d->logits = malloc((size_t)w->n_vocab * sizeof(float));
	d->x  = malloc((size_t)d->W * sizeof(float));
	d->xb = malloc((size_t)d->W * sizeof(float));
	d->q  = malloc((size_t)d->W * sizeof(float));
	d->att = malloc((size_t)(d->T > (unsigned)w->n_text_ctx
				 ? d->T : (unsigned)w->n_text_ctx) *
			sizeof(float));
	d->ff = malloc((size_t)d->F * sizeof(float));
	d->g1 = malloc((size_t)d->F * sizeof(float));
	d->b1 = malloc((size_t)d->F * sizeof(float));
	if (!d->xk || !d->xv || !d->sk || !d->sv || !d->logits || !d->x ||
	    !d->xb || !d->q || !d->att || !d->ff || !d->g1 || !d->b1)
		goto fail;

	for (l = 0; l < d->L; l++) {
		struct whisper_block *B = &w->dec[l];

		d->xk[l] = malloc((size_t)d->T * d->W * sizeof(float));
		d->xv[l] = malloc((size_t)d->T * d->W * sizeof(float));
		d->sk[l] = malloc((size_t)w->n_text_ctx * d->W * sizeof(float));
		d->sv[l] = malloc((size_t)w->n_text_ctx * d->W * sizeof(float));
		if (!d->xk[l] || !d->xv[l] || !d->sk[l] || !d->sv[l])
			goto fail;
		/* ⚠ no bias on the key projection, here as everywhere */
		wrows(B->xk_w, NULL, encoded, d->T, d->W, d->xk[l], d->W, &d->a);
		wrows(B->xv_w, wrow1(B->xv_b, d->b1), encoded, d->T, d->W,
		      d->xv[l], d->W, &d->a);
	}
	d->ok = 1;
	return d;
fail:
	charsiu_whisper_decoder_free(d);
	return NULL;
}

const float *charsiu_whisper_step(struct whisper_decoder *d, int32_t token,
				  int pos)
{
	const struct charsiu_whisper *w = d->w;
	unsigned W = d->W, H = d->H, hd = d->hd, F = d->F, l, i, j, h, e;
	float scale = 1.0f / sqrtf((float)hd);
	struct gguf_tensor emb;

	if (!d->ok || pos < 0 || pos >= w->n_text_ctx)
		return NULL;

	gguf_row_f32(w->d_tok, (uint64_t)token, d->x);
	gguf_row_f32(w->d_pos, (uint64_t)pos, d->g1);
	for (i = 0; i < W; i++)
		d->x[i] += d->g1[i];

	for (l = 0; l < d->L; l++) {
		struct whisper_block *B = &w->dec[l];
		float *krow = d->sk[l] + (size_t)pos * W;
		float *vrow = d->sv[l] + (size_t)pos * W;

		/* --- self attention, causal by construction ---------------- */
		wlayernorm(d->xb, d->x, wrow1(B->attn_ln_w, d->g1),
			   wrow1(B->attn_ln_b, d->b1), W, 1e-5f);
		wrows(B->q_w, wrow1(B->q_b, d->b1), d->xb, 1, W, d->q, W, &d->a);
		wrows(B->k_w, NULL, d->xb, 1, W, krow, W, &d->a);
		wrows(B->v_w, wrow1(B->v_b, d->b1), d->xb, 1, W, vrow, W, &d->a);

		for (h = 0; h < H; h++) {
			unsigned off = h * hd;
			const float *qi = d->q + off;
			float *o = d->xb + off;

			for (j = 0; j <= (unsigned)pos; j++) {
				const float *kj = d->sk[l] + (size_t)j * W + off;
				float dp = 0.0f;

				for (e = 0; e < hd; e++)
					dp += qi[e] * kj[e];
				d->att[j] = dp * scale;
			}
			wsoftmax(d->att, (unsigned)pos + 1);
			for (e = 0; e < hd; e++)
				o[e] = 0.0f;
			for (j = 0; j <= (unsigned)pos; j++) {
				const float *vj = d->sv[l] + (size_t)j * W + off;
				float wg = d->att[j];

				for (e = 0; e < hd; e++)
					o[e] += wg * vj[e];
			}
		}
		wrows(B->o_w, wrow1(B->o_b, d->b1), d->xb, 1, W, d->q, W, &d->a);
		for (i = 0; i < W; i++)
			d->x[i] += d->q[i];

		/* --- cross attention, against the clip --------------------- */
		wlayernorm(d->xb, d->x, wrow1(B->x_ln_w, d->g1),
			   wrow1(B->x_ln_b, d->b1), W, 1e-5f);
		wrows(B->xq_w, wrow1(B->xq_b, d->b1), d->xb, 1, W, d->q, W,
		      &d->a);
		for (h = 0; h < H; h++) {
			unsigned off = h * hd;
			const float *qi = d->q + off;
			float *o = d->xb + off;

			for (j = 0; j < d->T; j++) {
				const float *kj = d->xk[l] + (size_t)j * W + off;
				float dp = 0.0f;

				for (e = 0; e < hd; e++)
					dp += qi[e] * kj[e];
				d->att[j] = dp * scale;
			}
			wsoftmax(d->att, d->T);
			for (e = 0; e < hd; e++)
				o[e] = 0.0f;
			for (j = 0; j < d->T; j++) {
				const float *vj = d->xv[l] + (size_t)j * W + off;
				float wg = d->att[j];

				for (e = 0; e < hd; e++)
					o[e] += wg * vj[e];
			}
		}
		wrows(B->xo_w, wrow1(B->xo_b, d->b1), d->xb, 1, W, d->q, W,
		      &d->a);
		for (i = 0; i < W; i++)
			d->x[i] += d->q[i];

		/* --- feed forward ------------------------------------------ */
		wlayernorm(d->xb, d->x, wrow1(B->mlp_ln_w, d->g1),
			   wrow1(B->mlp_ln_b, d->b1), W, 1e-5f);
		wrows(B->fc1_w, wrow1(B->fc1_b, d->b1), d->xb, 1, W, d->ff, F,
		      &d->a);
		wgelu(d->ff, F);
		wrows(B->fc2_w, wrow1(B->fc2_b, d->b1), d->ff, 1, F, d->q, W,
		      &d->a);
		for (i = 0; i < W; i++)
			d->x[i] += d->q[i];
	}

	wlayernorm(d->xb, d->x, wrow1(w->d_ln_w, d->g1),
		   wrow1(w->d_ln_b, d->b1), W, 1e-5f);

	/*
	 * ⚠ THE OUTPUT HEAD IS THE EMBEDDING TABLE, transposed -- whisper ties
	 * them. token_embedding.weight is [n_vocab][n_text_state] and that is
	 * exactly a matmul against it, one row a vocabulary entry.
	 */
	emb = wflat(w->d_tok, (uint64_t)W, (uint64_t)w->n_vocab);
	charsiu_act_set(&d->a, d->xb, (int)W);
	gguf_matvec(&emb, &d->a, d->logits, 0, (uint64_t)w->n_vocab);
	return d->logits;
}

const char *charsiu_whisper_token(const struct charsiu_whisper *w, int32_t id)
{
	if (id < 0 || id >= w->n_vocab || !w->vocab || !w->vocab[id])
		return "";
	return w->vocab[id];
}

int charsiu_whisper_transcribe(const struct charsiu_whisper *w,
			       const float *encoded, int32_t *ids, int max)
{
	struct whisper_decoder *d;
	const float *lg = NULL;
	int32_t prompt[2];
	int n = 0, pos, i, np = 2;

	/*
	 * ⚠ REFUSED RATHER THAN GUESSED. A multilingual model wants a language
	 * token and a task token between the two markers, and picking them
	 * needs a language detection pass this does not do. Transcribing a
	 * French clip with an English prompt does not fail: it answers in
	 * fluent English about nothing that was said.
	 */
	if (w->multilingual)
		return -1;

	d = charsiu_whisper_decoder_new(w, encoded);
	if (!d)
		return -1;

	prompt[0] = w->tok_sot;
	prompt[1] = w->tok_not;

	/*
	 * ⚠ THE LOGITS AFTER THE LAST PROMPT TOKEN ARE THE FIRST PREDICTION.
	 * There is no extra step for it, and adding one -- which the first
	 * version of this did -- feeds the marker twice and shifts the whole
	 * transcript by a token.
	 */
	for (i = 0; i < np; i++)
		lg = charsiu_whisper_step(d, prompt[i], i);
	pos = np;

	while (lg && n < max && pos < w->n_text_ctx) {
		int32_t best = 0;
		float bv = -1e30f;

		for (i = 0; i < w->n_vocab; i++)
			if (lg[i] > bv) {
				bv = lg[i];
				best = (int32_t)i;
			}
		if (best == w->tok_eot)
			break;
		ids[n++] = best;
		lg = charsiu_whisper_step(d, best, pos++);
	}
	charsiu_whisper_decoder_free(d);
	return n;
}
