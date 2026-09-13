// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The coefficient buffer's per-output-channel scale table.
 *
 * The hardware has held one fp16 scale per output channel since this emitter
 * was written, and charsiu filled every entry with the same scalar. An int8 KV
 * surface needs the table used as intended: attention's scores matmul has one
 * output channel per POSITION, so a per-position quantisation scale goes here.
 *
 * ⚠ THE FAILURE THIS GUARDS AGAINST IS SILENT. A scale table off by one
 * channel, or one that reads past the caller's array into the padding, gives
 * every output a plausible wrong magnitude -- and this project's own history
 * says a plausible wrong number survives a text check. So the table is read
 * back and compared entry by entry, including the padding, against the
 * definition rather than against another run of the same code.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "charsiu.h"

static int fail;

static void one(unsigned n, unsigned k, int per_channel)
{
	struct charsiu_job job;
	float *sc = malloc(n * sizeof(*sc));
	int32_t *bias = calloc(n, sizeof(*bias));
	int32_t *ws = calloc(n, sizeof(*ws));
	uint8_t *buf;
	size_t bytes, tb, sb;
	unsigned oc, bad = 0;
	const uint16_t *tab;

	memset(&job, 0, sizeof(job));
	job.mm.m = 8; job.mm.k = k; job.mm.n = n;
	job.mm.wdtype = CHARSIU_INT8;
	job.mm.adtype = CHARSIU_FP16;
	job.input_scale = 0.02f;
	job.weight_scale = 0.125f;
	job.output_scale = 1.0f;
	job.input_zero_point = 128;
	job.weight_zero_point = 0x80;
	for (oc = 0; oc < n; oc++)
		sc[oc] = 0.001f * (float)(oc + 1);
	job.weight_scales = per_channel ? sc : NULL;

	bytes = charsiu_coef_bytes(&job.mm);
	buf = malloc(bytes);
	if (!sc || !bias || !ws || !buf) { printf("  out of memory\n"); fail = 1; return; }
	charsiu_build_coefs(&job, bias, ws, buf);

	/* the table's own geometry, from the same two helpers the builder uses */
	tb = ((size_t)((n + 7) / 8)) * 64;
	sb = bytes;                       /* upper bound; the operand follows */
	tab = (const uint16_t *)(buf + tb);
	for (oc = 0; oc < n; oc++) {
		float want = per_channel ? sc[oc] : job.weight_scale;

		if (tab[oc] != charsiu_float_to_half(want))
			bad++;
	}
	/* the padding channels, which the DPU never writes: the scalar */
	for (oc = n; oc < ((n + 7) / 8) * 8; oc++)
		if (tab[oc] != charsiu_float_to_half(job.weight_scale))
			bad++;
	printf("  %-11s n=%-5u k=%-5u  %u wrong of %u\n",
	       per_channel ? "per channel" : "scalar", n, k, bad,
	       ((n + 7) / 8) * 8);
	if (bad)
		fail = 1;
	(void)sb;
	free(sc); free(bias); free(ws); free(buf);
}

int main(void)
{
	static const unsigned shape[][2] = {
		{ 64, 64 }, { 128, 64 }, { 864, 64 }, { 1024, 64 },
		{ 7, 64 }, { 8, 64 }, { 9, 64 }, { 33, 128 },
	};
	unsigned i;

	printf("== the coefficient scale table, entry by entry\n");
	for (i = 0; i < sizeof(shape) / sizeof(shape[0]); i++) {
		one(shape[i][0], shape[i][1], 0);
		one(shape[i][0], shape[i][1], 1);
	}
	printf("coef_scales: %s\n", fail ? "FAILED" : "ok");
	return fail;
}
