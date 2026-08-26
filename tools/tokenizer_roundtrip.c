// SPDX-License-Identifier: GPL-2.0
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * encode -> decode must give the text back, for every tokenizer family.
 *
 * charsiu has two now: BPE with a merge table, and SentencePiece with a score
 * per piece. They segment by different algorithms and spell a space
 * differently, and a round trip is the one check that does not care which.
 *
 *   tokenizer_roundtrip MODEL.gguf
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charsiu_llm.h"
/* encode -> decode must give the text back, or the tokenizer is lying */
int main(int argc, char **argv)
{
	(void)argc;
	struct llama_model m;
	static const char *cases[] = {
		"The capital of France is Paris.",
		"hello  world   with   runs of spaces",
		"unicode: 你好世界 ünïcödé ✓ emoji 🙂",
		"punctuation!?;:'\"[]{}()<>@#$%^&*",
		"digits 0123456789 mixed42with7text",
		"",
		" leading and trailing ",
		"tabs\tand\nnewlines\r\n",
	};
	int32_t ids[4096];
	int bad = 0;

	if (llama_load(&m, argv[1]) < 0) return 1;
	for (unsigned c = 0; c < sizeof(cases)/sizeof(*cases); c++) {
		int n = tokenizer_encode(m.tk, cases[c], 0, ids, 4096);
		char out[4096]; size_t w = 0;

		if (n < 0) { printf("  ENCODE FAILED: %s\n", cases[c]); bad++; continue; }
		for (int i = 0; i < n; i++) {
			int len; const char *s = tokenizer_decode(m.tk, ids[i], &len);
			if (w + (size_t)len < sizeof(out)) { memcpy(out + w, s, len); w += len; }
		}
		out[w] = 0;
		/* ⚠ SPM ALWAYS prepends one space marker, so the decode always
		 * has one more than the input -- including when the input
		 * already starts with a space. Stripping it only in the other
		 * case was the TEST being wrong, not the tokenizer. */
		/* ⚠ BPE does NOT prepend one, so accept either -- the only
		 * ambiguity in a round trip is that one marker, and hard-coding
		 * a family into the test is how it got this wrong twice. */
		const char *got = out;
		if (strcmp(got, cases[c]) && n > 0 && out[0] == ' ')
			got = out + 1;
		if (strcmp(got, cases[c])) {
			printf("  MISMATCH (%d tok)\n    in : %s\n    out: %s\n", n, cases[c], got);
			bad++;
		} else {
			printf("  ok %3d tok  %s\n", n, cases[c][0] ? cases[c] : "(empty)");
		}
	}
	printf("  %s\n", bad ? "ROUND TRIP BROKEN" : "every case round trips");
	return bad != 0;
}
