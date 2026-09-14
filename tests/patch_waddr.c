// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * Patching the weight address into an emitted stream against emitting it that
 * way in the first place, byte for byte.
 *
 * ⚠⚠ WHY THIS IS A TEST AND NOT A READING. charsiu_patch_weight_addr exists so
 * a layer of attention can reuse one register stream over sixteen different KV
 * surfaces, and the whole safety of that is the claim that ONE word depends on
 * the weight address. That claim is true of the emitter as it is written today
 * and nothing holds it there: an emitter that folded the address into a size
 * register, or wrote it twice, would make the patch silently produce a
 * dispatch nobody asked for.
 *
 * So this emits with address A, patches to B, emits fresh with B, and requires
 * the two to be identical over every word -- across the shapes attention
 * actually runs and both weight dtypes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charsiu.h"

static int fail;

static void check(unsigned m, unsigned k, unsigned n, enum charsiu_dtype wd)
{
	struct charsiu_job a = { 0 }, b = { 0 };
	uint64_t sa[512], sb[512];
	size_t na, nb, i;
	unsigned hits;

	a.mm.m = m; a.mm.k = k; a.mm.n = n;
	a.mm.wdtype = wd; a.mm.adtype = CHARSIU_FP16;
	a.input_scale = 1.0f; a.weight_scale = 1.0f; a.output_scale = 1.0f;
	a.acc_out = 1;
	a.cbuf_window = (unsigned)charsiu_cbuf_window();
	a.input_addr = 0x10000000u; a.output_addr = 0x20000000u;
	a.coef_addr = 0x30000000u;
	b = a;
	a.weight_addr = 0x40000000u;
	b.weight_addr = 0x51234500u;

	na = charsiu_emit_job(&a, sa, 512);
	nb = charsiu_emit_job(&b, sb, 512);
	if (!na || na != nb) {
		printf("  m=%u k=%u n=%u: %zu words against %zu\n",
		       m, k, n, na, nb);
		fail++;
		return;
	}
	hits = charsiu_patch_weight_addr(sa, na, b.weight_addr);
	if (hits != 1) {
		printf("  m=%u k=%u n=%u w%d: the address is in %u words, not "
		       "one -- the patch is not safe at this shape\n",
		       m, k, n, wd == CHARSIU_INT4 ? 4 : 16, hits);
		fail++;
		return;
	}
	for (i = 0; i < na; i++)
		if (sa[i] != sb[i]) {
			printf("  m=%u k=%u n=%u: word %zu is %016llx patched "
			       "and %016llx emitted\n", m, k, n, i,
			       (unsigned long long)sa[i],
			       (unsigned long long)sb[i]);
			fail++;
			return;
		}
}

int main(void)
{
	/* the two attention shapes, the projections, and the edges */
	unsigned shapes[][3] = {
		{ 80, 64, 864 }, { 80, 1024, 64 }, { 78, 64, 112 },
		{ 80, 2048, 8192 }, { 1, 2048, 2048 }, { 32, 256, 64 },
		{ 80, 128, 448 }, { 2, 64, 64 },
	};
	unsigned i;

	for (i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
		check(shapes[i][0], shapes[i][1], shapes[i][2], CHARSIU_FP16);
		check(shapes[i][0], shapes[i][1], shapes[i][2], CHARSIU_INT4);
		check(shapes[i][0], shapes[i][1], shapes[i][2], CHARSIU_INT8);
	}
	printf("%s: the weight address is one word, %d failures\n",
	       fail ? "FAIL" : "ok", fail);
	return fail ? 1 : 0;
}
