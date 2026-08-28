// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * What is in an mmproj file, and is it a tower this can run?
 *
 * ⚠ THIS EXISTS BECAUSE THE NAMES ARE A GUESS. Every tensor name the loader
 * reaches for is llama.cpp's clip naming as this tree understands it, checked
 * against nothing. Pointed at a real mmproj this prints either "every tensor
 * this needs is here" or the exact names it wanted and did not find, which is
 * the difference between a five minute fix and the gemma4 failure: a guessed
 * name found nothing, the path was skipped in silence, and the model answered
 * anyway while missing the half of itself its name is about.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu_llm.h"
#include "charsiu_vision.h"

/*
 * ⚠ THE PIXELS ARE GENERATED, NOT READ. A cross check needs both sides to hold
 * the same image, and passing one costs a file format and a decoder before
 * either side has been shown to be right. An integer formula both sides can
 * compute EXACTLY -- no sin, no float parsing -- removes the image from the
 * list of things that can be wrong.
 */
static void fake_image(float *px, unsigned n)
{
	unsigned t;

	for (t = 0; t < n; t++)
		px[t] = ((float)(t % 251u) / 251.0f) * 2.0f - 1.0f;
}

static int encode(struct charsiu_vision *v, const char *image)
{
	unsigned n = 3u * v->image_size * v->image_size;
	unsigned tok = charsiu_vision_tokens(v), w = charsiu_vision_width(v);
	char err[256] = "";
	float *px = image ? charsiu_image_load(image, v->image_size, err,
					       sizeof(err))
			  : malloc((size_t)n * sizeof(float));
	float *out = malloc((size_t)tok * w * sizeof(float));
	unsigned i;

	if (!px) {
		fprintf(stderr, "charsiu_vision: %s\n", err);
		free(out);
		return 1;
	}
	if (!out) {
		free(px);
		return 1;
	}
	if (!image)
		fake_image(px, n);
	charsiu_vision_normalise(v, px);
	if (charsiu_vision_encode(v, px, out)) {
		fprintf(stderr, "charsiu_vision: the tower would not run\n");
		free(px); free(out);
		return 1;
	}
	printf("embeddings %u %u\n", tok, w);
	for (i = 0; i < tok * w; i++)
		printf("%.7g\n", (double)out[i]);
	free(px); free(out);
	return 0;
}

/*
 * ⚠ THE RESIZE IS TESTABLE ON ITS OWN, and it has to be: a half pixel shift is
 * invisible in a caption and fatal to a comparison. This mode needs no model.
 */
static int resize_mode(const char *path, unsigned side)
{
	char err[256] = "";
	float *px = charsiu_image_load(path, side, err, sizeof(err));
	unsigned i;

	if (!px) {
		fprintf(stderr, "charsiu_vision: %s\n", err);
		return 1;
	}
	printf("pixels 3 %u %u\n", side, side);
	for (i = 0; i < 3u * side * side; i++)
		printf("%.7g\n", (double)px[i]);
	free(px);
	return 0;
}

int main(int argc, char **argv)
{
	struct charsiu_vision v;
	const char *image = NULL;
	int rc, want_encode = 0, i;

	if (argc >= 4 && !strcmp(argv[1], "--resize"))
		return resize_mode(argv[2], (unsigned)atoi(argv[3]));

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--encode"))
			want_encode = 1;
		else if (!strcmp(argv[i], "--image") && i + 1 < argc)
			image = argv[++i];
	}
	if (image)
		want_encode = 1;

	if (argc < 2) {
		fprintf(stderr,
			"usage: charsiu_vision MMPROJ.gguf [--encode] [--image FILE]\n"
			"       charsiu_vision --resize FILE SIDE\n"
			"\n"
			"  Reads a vision tower's hparams and tensors and says\n"
			"  whether this build can run it. An mmproj is a\n"
			"  SEPARATE file from the model it belongs to.\n");
		return 2;
	}

	rc = charsiu_vision_open(&v, argv[1]);
	if (!want_encode)
		charsiu_vision_describe(&v, stdout);
	if (rc) {
		printf("\nnot usable: %s\n", charsiu_vision_why_not(&v));
		charsiu_vision_close(&v);
		return 1;
	}
	if (want_encode)
		rc = encode(&v, image);
	charsiu_vision_close(&v);
	return rc ? 1 : 0;
}
