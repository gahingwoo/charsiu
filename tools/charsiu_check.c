/*
 * Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
 * SPDX-License-Identifier: GPL-2.0
 *
 * charsiu_check -- can charsiu actually run this gguf?
 *
 * The question sounds like it should be answerable from the filename and it is
 * not. Hugging Face is full of files whose names promise a type charsiu reads
 * and whose contents are something else:
 *
 *   Llama-3.2-1B-Instruct-Q4_0_4_4.gguf   a Q4_0 REPACKED for ARM dotprod --
 *                                         same block size, interleaved weights
 *   Llama-3.2-1B-Instruct-Q6_K_L.gguf     mostly Q6_K, some tensors elsewhere
 *   ...-UD-Q6_K_XL.gguf                   likewise
 *
 * Every one of those matches a substring test for a supported type and none of
 * them is one. So this opens the file and reads the tensors.
 *
 * It is deliberately the SAME parser the runtime uses -- gguf_open() and
 * ggml_type_name() out of src/gguf.c -- so the gate cannot drift away from what
 * charsiu will actually accept at load time. A type charsiu has no traits for
 * comes back as "unsupported" here for the same reason it would fail there.
 *
 * Exit codes, because charsiu-get branches on them:
 *   0  charsiu can run this
 *   1  it is a readable gguf that charsiu cannot run (wrong arch, or a type)
 *   2  it is not a gguf, or it cannot be opened at all
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charsiu_llm.h"

/*
 * For the REPORT only. charsiu's own table (traits_of) decides what is
 * supported; this exists so "type 12" reads as "q4_K" in the message instead of
 * leaving someone to look it up. Anything not listed prints as a bare number,
 * which is the honest answer for a slot whose meaning has changed across
 * llama.cpp versions -- the repacked Q4_0 variants in particular.
 */
static const char *ggml_name_hint(uint32_t t)
{
	static const char *n[] = {
		"f32", "f16", "q4_0", "q4_1", NULL, NULL, "q5_0", "q5_1",
		"q8_0", "q8_1", "q2_K", "q3_K", "q4_K", "q5_K", "q6_K", "q8_K",
		"iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "iq4_nl", "iq3_s",
		"iq2_s", "iq4_xs",
	};
	if (t < sizeof(n) / sizeof(n[0]) && n[t])
		return n[t];
	if (t == 30)
		return "bf16";
	return NULL;
}

struct seen {
	uint32_t type;
	unsigned long count;
	int ok;
};

int main(int argc, char **argv)
{
	const char *path = NULL;
	int quiet = 0, i;
	struct gguf g;
	struct seen seen[64];
	unsigned nseen = 0;
	char arch[64] = "", name[128] = "";
	int bad_types = 0, bad_graph = 0, bad_tok = 0;
	char tokmodel[64] = "";
	unsigned long j;

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-q"))
			quiet = 1;
		else if (!path)
			path = argv[i];
		else
			path = NULL;
	}
	if (!path) {
		fprintf(stderr,
"usage: charsiu_check [-q] MODEL.gguf\n"
"  Reads the file, not the name, and says whether charsiu can run it.\n"
"  -q  one line: OK or the first reason it is not\n"
"  exit 0 runnable, 1 readable but not runnable, 2 not a gguf\n");
		return 2;
	}

	if (gguf_open(&g, path)) {
		if (quiet)
			printf("NOT-A-GGUF %s\n", path);
		else
			fprintf(stderr, "charsiu_check: cannot read %s as a gguf\n", path);
		return 2;
	}

	gguf_get_str(&g, "general.architecture", arch, sizeof(arch));
	gguf_get_str(&g, "general.name", name, sizeof(name));

	/*
	 * ⚠ THIS LIST MUST MATCH llama_load's. It exists to save a 2 GB
	 * download, so it is wrong in both directions: refusing something that
	 * runs wastes a model, and accepting something that does not wastes the
	 * download this is here to prevent.
	 *
	 * qwen2 runs on the llama graph -- same nine weights a layer, tied
	 * output head, plus a bias on Q, K and V. gemma2 and phi3 do not, and
	 * not over tensor names: gemma2 declares attn_logit_softcapping 50.0,
	 * final_logit_softcapping 30.0 and a 4096 sliding window, and phi3
	 * fuses QKV into one tensor and gate+up into another.
	 */
	if (strcmp(arch, "llama") && strcmp(arch, "qwen2") && strcmp(arch, "phi3"))
		bad_graph = 1;

	/*
	 * ⚠ THE GRAPH IS NOT THE ONLY THING THAT HAS TO MATCH. charsiu's
	 * tokenizer is BPE and needs a merge table; a file that declares
	 * tokenizer.ggml.model = llama carries SentencePiece scores instead and
	 * will load its weights and then fail to turn text into tokens.
	 * Measured on Phi-3.5-mini, which is why this check exists: the
	 * architecture passed, the tensors split correctly, and it still could
	 * not read a prompt. Saying so here saves a 2 GB download.
	 */
	{
		char *tok = tokmodel;

		gguf_get_str(&g, "tokenizer.ggml.model", tok, 64);
		if (tok[0] && strcmp(tok, "gpt2") && !gguf_find(&g, "tokenizer.ggml.merges")) {
			if (!quiet)
				printf("tokenizer     %s   <-- charsiu's is BPE, "
				       "with a merge table\n", tok);
			bad_tok = 1;
		}
	}

	for (j = 0; j < g.n_tensors; j++) {
		uint32_t t = g.t[j].type;
		unsigned k;

		for (k = 0; k < nseen; k++)
			if (seen[k].type == t)
				break;
		if (k == nseen) {
			if (nseen == sizeof(seen) / sizeof(seen[0]))
				continue;
			seen[nseen].type = t;
			seen[nseen].count = 0;
			/* THE gate: charsiu's own table, not a name match. */
			seen[nseen].ok = strcmp(ggml_type_name(t), "unsupported") != 0;
			if (!seen[nseen].ok)
				bad_types++;
			nseen++;
		}
		seen[k].count++;   /* k is the slot, new or found */
	}

	if (quiet) {
		if (bad_graph)
			printf("NO arch=%s (charsiu builds llama, qwen2 and phi3)\n", arch);
		else if (bad_tok)
			printf("NO tokenizer=%s (charsiu's is BPE, with a merge table)\n",
			       tokmodel);
		else if (bad_types) {
			for (i = 0; (unsigned)i < nseen; i++)
				if (!seen[i].ok) {
					const char *h = ggml_name_hint(seen[i].type);

					printf("NO type %u%s%s%s in %lu tensors\n",
					       seen[i].type, h ? " (" : "", h ? h : "",
					       h ? ")" : "", seen[i].count);
					break;
				}
		} else {
			printf("OK %s %s\n", arch, name[0] ? name : "-");
		}
		gguf_close(&g);
		return bad_graph || bad_types || bad_tok ? 1 : 0;
	}

	printf("file          %s\n", path);
	printf("name          %s\n", name[0] ? name : "(unnamed)");
	printf("architecture  %s%s\n", arch[0] ? arch : "(missing)",
	       bad_graph ? "   <-- charsiu builds llama, qwen2 and phi3" : "");
	printf("gguf          v%u, %llu tensors\n", g.version,
	       (unsigned long long)g.n_tensors);
	printf("tensor types\n");
	for (i = 0; (unsigned)i < nseen; i++) {
		const char *h = ggml_name_hint(seen[i].type);
		const char *cs = ggml_type_name(seen[i].type);

		printf("  %-10s type %-3u %6lu tensors   %s\n",
		       seen[i].ok ? cs : (h ? h : "?"), seen[i].type,
		       seen[i].count,
		       seen[i].ok ? "charsiu reads this"
				  : "NOT one charsiu reads");
	}
	if (bad_graph || bad_types || bad_tok)
		printf("\nVERDICT  charsiu CANNOT run this file.\n");
	else
		printf("\nVERDICT  charsiu can run this file.\n");

	gguf_close(&g);
	return bad_graph || bad_types || bad_tok ? 1 : 0;
}
