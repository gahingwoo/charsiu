// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * Ask a CLIP model which of several sentences a picture is.
 *
 *   charsiu_clip MODEL.gguf --image cat.jpg --text "a cat" "a dog" "a plane"
 *
 * ⚠ THE SCORE IS A COSINE, not a probability. CLIP's own softmax over these
 * uses a learned temperature that is not in the gguf, so this prints the
 * similarities themselves. The ORDER is the answer; the gaps are not
 * calibrated to anything.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu.h"
#include "charsiu_llm.h"
#include "charsiu_vision.h"
#include "charsiu_clip.h"

int main(int argc, char **argv)
{
	struct charsiu_vision v;
	struct charsiu_clip_text t;
	const char *model = NULL, *image = NULL;
	int i, ntext = 0, first_text = -1, tokens_only = 0, rc = 1;
	int have_vision = 0, ids_mode = 0;
	float *ie = NULL;

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--image") && i + 1 < argc)
			image = argv[++i];
		else if (!strcmp(argv[i], "--tokens"))
			tokens_only = 1;
		/*
		 * ⚠ IDS IN DIRECTLY, so the text tower is checkable against a
		 * reference without the tokenizer in the way. Two things that
		 * can each be wrong should not be tested only together.
		 */
		else if (!strcmp(argv[i], "--text-ids")) {
			ids_mode = 1;
			first_text = i + 1;
			ntext = argc - first_text;
			break;
		}
		else if (!strcmp(argv[i], "--text")) {
			first_text = i + 1;
			ntext = argc - first_text;
			break;
		} else if (argv[i][0] != '-' && !model)
			model = argv[i];
	}
	if (!model) {
		fprintf(stderr,
			"usage: charsiu_clip MODEL.gguf [--image FILE] "
			"--text \"...\" [\"...\" ...]\n"
			"       charsiu_clip MODEL.gguf --tokens --text \"...\"\n"
			"\n"
			"  A two tower CLIP gguf: the picture and the sentences\n"
			"  land in one space and the cosine says which fits.\n");
		return 2;
	}

	if (charsiu_clip_text_open(&t, model)) {
		fprintf(stderr, "charsiu_clip: %s\n",
			charsiu_clip_text_why_not(&t));
		charsiu_clip_text_describe(&t, stderr);
		return 1;
	}

	/* ⚠ the tokens on their own, so the BPE is checkable without a tower */
	if (tokens_only) {
		for (i = first_text; i < argc; i++) {
			int32_t ids[128];
			int n, k, na = 0;

			n = charsiu_clip_tokenize(&t, argv[i], ids,
						  (int)(sizeof(ids) /
							sizeof(*ids)), &na);
			if (n < 0) {
				fprintf(stderr, "charsiu_clip: %s did not "
					"tokenize\n", argv[i]);
				goto done;
			}
			if (na)
				fprintf(stderr, "charsiu_clip: \"%s\" is not "
					"ASCII; the word split is\n"
					"  approximate there and the ids may "
					"differ from the reference\n", argv[i]);
			printf("%d", n);
			for (k = 0; k < n; k++)
				printf(" %d", ids[k]);
			printf("\n");
		}
		rc = 0;
		goto done;
	}

	if (ids_mode) {
		int32_t ids[256];
		float *te = malloc((size_t)t.proj_dim * sizeof(float));
		int n = 0;

		for (i = first_text; i < argc && n < 256; i++)
			ids[n++] = (int32_t)atoi(argv[i]);
		if (!te || charsiu_clip_encode_text(&t, ids, n, te)) {
			fprintf(stderr, "charsiu_clip: those ids would not "
				"encode\n");
			free(te);
			goto done;
		}
		printf("embedding %u\n", t.proj_dim);
		for (i = 0; i < (int)t.proj_dim; i++)
			printf("%.7g\n", (double)te[i]);
		free(te);
		rc = 0;
		goto done;
	}

	if (image) {
		char err[256] = "";
		float *px;

		if (charsiu_vision_open(&v, model)) {
			fprintf(stderr, "charsiu_clip: %s\n",
				charsiu_vision_why_not(&v));
			goto done;
		}
		have_vision = 1;
		px = charsiu_image_load(image, v.image_size, err, sizeof(err));
		if (!px) {
			fprintf(stderr, "charsiu_clip: %s\n", err);
			goto done;
		}
		charsiu_vision_normalise(&v, px);
		ie = malloc((size_t)charsiu_vision_width(&v) * sizeof(float));
		if (!ie || charsiu_vision_encode(&v, px, ie)) {
			fprintf(stderr, "charsiu_clip: the tower would not run\n");
			free(px);
			goto done;
		}
		free(px);
		if (charsiu_vision_width(&v) != t.proj_dim) {
			fprintf(stderr, "charsiu_clip: the two towers land in "
				"%u and %u wide spaces\n",
				charsiu_vision_width(&v), t.proj_dim);
			goto done;
		}
	}

	for (i = first_text; i < argc && ntext > 0; i++) {
		int32_t ids[128];
		float *te = malloc((size_t)t.proj_dim * sizeof(float));
		int n = charsiu_clip_tokenize(&t, argv[i], ids,
					      (int)(sizeof(ids) / sizeof(*ids)),
					      NULL);

		if (!te || n < 0 || charsiu_clip_encode_text(&t, ids, n, te)) {
			fprintf(stderr, "charsiu_clip: \"%s\" would not "
				"encode\n", argv[i]);
			free(te);
			goto done;
		}
		if (ie)
			printf("%8.4f  %s\n",
			       (double)charsiu_cosine(ie, te, t.proj_dim),
			       argv[i]);
		else
			printf("%d tokens  %s\n", n, argv[i]);
		free(te);
	}
	rc = 0;
done:
	free(ie);
	if (have_vision)
		charsiu_vision_close(&v);
	charsiu_clip_text_close(&t);
	return rc;
}
