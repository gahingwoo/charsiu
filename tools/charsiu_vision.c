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
#include <string.h>

#include "charsiu_llm.h"
#include "charsiu_vision.h"

int main(int argc, char **argv)
{
	struct charsiu_vision v;
	int rc;

	if (argc < 2) {
		fprintf(stderr,
			"usage: charsiu_vision MMPROJ.gguf\n"
			"\n"
			"  Reads a vision tower's hparams and tensors and says\n"
			"  whether this build can run it. An mmproj is a\n"
			"  SEPARATE file from the model it belongs to.\n");
		return 2;
	}

	rc = charsiu_vision_open(&v, argv[1]);
	charsiu_vision_describe(&v, stdout);
	if (rc) {
		printf("\nnot usable: %s\n", charsiu_vision_why_not(&v));
	}
	charsiu_vision_close(&v);
	return rc ? 1 : 0;
}
