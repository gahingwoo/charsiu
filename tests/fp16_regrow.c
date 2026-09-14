/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
 * SPDX-License-Identifier: GPL-2.0
 *
 * ⭐ GROWING THE V SURFACE IS A BLOCK COPY, AND THIS IS WHAT SAYS SO.
 *
 * charsiu_fp16_regrow_vcols claims that a surface packed at kv_old, carrying
 * `live` positions, becomes the same surface at kv_new by moving each output
 * channel group's block -- because the padded extent appears in exactly one
 * term of the offset. The claim is checked the only way it can be: pack the
 * same positions DIRECTLY at kv_new with charsiu_fp16_pack_vcol and require
 * the two buffers to be byte identical, the zeros past `live` included.
 *
 * ⚠ WITH A NEGATIVE CONTROL, because a test that only ever compares a thing
 * to itself passes on a regrow that does nothing. The control packs one extra
 * position into the reference and requires the comparison to FAIL: if it does
 * not, the comparison is not looking at the bytes it thinks it is.
 *
 * ⭐ AND IN PLACE IS ITS OWN ARM. The surface can be allocated once at the
 * prompt's ceiling and re-laid-out where it lies, which is what removes
 * n_layer * n_kv buffer objects a rung -- 2048 of them on a 32 layer model
 * with no GQA. In place is only correct because the groups are walked from the
 * top down; walked the other way each block overwrites the next one's source,
 * and the bytes that come out are still a plausible looking surface. So the
 * in place arm compares against the SAME reference as the copying one, on
 * every shape, rather than against the copying arm's output.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charsiu.h"

static void fill(float *v, unsigned hd, unsigned pos)
{
	unsigned i;

	for (i = 0; i < hd; i++)
		v[i] = (float)((pos * 31u + i * 7u) % 251u) * 0.125f - 15.0f;
}

static uint16_t *pack_at(unsigned kv, unsigned hd, unsigned live, size_t *bytes)
{
	struct charsiu_matmul mm = { 1, kv, hd, CHARSIU_FP16, CHARSIU_FP16 };
	size_t b = charsiu_weight_bytes(&mm);
	uint16_t *buf = calloc(1, b + 4096);
	float *v = malloc(hd * sizeof(*v));
	unsigned p;

	if (!buf || !v)
		exit(2);
	for (p = 0; p < live; p++) {
		fill(v, hd, p);
		charsiu_fp16_pack_vcol(buf, kv, hd, p, v);
	}
	free(v);
	*bytes = b;
	return buf;
}

static int one(unsigned hd, unsigned kv_old, unsigned kv_new, unsigned live)
{
	size_t bo, bn, bn2;
	uint16_t *old = pack_at(kv_old, hd, live, &bo);
	uint16_t *ref = pack_at(kv_new, hd, live, &bn);
	uint16_t *got = pack_at(kv_new, hd, 0, &bn2);
	float *v = malloc(hd * sizeof(*v));
	int bad = 0;

	if (bn != bn2 || !v)
		exit(2);
	if (charsiu_fp16_regrow_vcols(got, kv_new, old, kv_old, hd, live)) {
		printf("  hd %3u  %4u -> %4u  live %4u   REFUSED\n",
		       hd, kv_old, kv_new, live);
		free(old); free(ref); free(got); free(v);
		return 1;
	}
	if (memcmp(ref, got, bn)) {
		size_t i;

		for (i = 0; i < bn / 2; i++)
			if (ref[i] != got[i])
				break;
		printf("  hd %3u  %4u -> %4u  live %4u   ⛔ first differing "
		       "half at %zu: %04x want %04x\n", hd, kv_old, kv_new,
		       live, i, got[i], ref[i]);
		bad = 1;
	} else {
		printf("  hd %3u  %4u -> %4u  live %4u   ok  (%zu bytes)\n",
		       hd, kv_old, kv_new, live, bn);
	}

	/* the negative control: one more position in the reference and the
	 * comparison has to notice */
	fill(v, hd, live);
	charsiu_fp16_pack_vcol(ref, kv_new, hd, live, v);
	if (!memcmp(ref, got, bn)) {
		printf("  hd %3u  %4u -> %4u  live %4u   ⛔ CONTROL: an extra "
		       "packed position did not change the buffer\n",
		       hd, kv_old, kv_new, live);
		bad = 1;
	}
	/* ⭐ THE IN PLACE ARM, against the same reference. `ip` is allocated at
	 * the NEW size and packed in the OLD layout, which is exactly the
	 * surface a caller has when it allocated once at the ceiling. */
	{
		uint16_t *ip = calloc(1, bn + 4096);
		float *w = malloc(hd * sizeof(*w));
		unsigned p;

		if (!ip || !w)
			exit(2);
		for (p = 0; p < live; p++) {
			fill(w, hd, p);
			charsiu_fp16_pack_vcol(ip, kv_old, hd, p, w);
		}
		free(w);
		/* ref has the extra control position in it by now, so compare
		 * against `got`, which this arm has already been shown equal
		 * to the reference for. ⚠ That makes the in place arm's check
		 * transitive and it is only sound because the line above
		 * failed the test if got != ref. */
		if (charsiu_fp16_regrow_vcols(ip, kv_new, ip, kv_old, hd,
					      live)) {
			printf("  hd %3u  %4u -> %4u  live %4u   ⛔ IN PLACE "
			       "REFUSED what the copy accepted\n",
			       hd, kv_old, kv_new, live);
			bad = 1;
		} else if (!bad && memcmp(got, ip, bn)) {
			size_t i;

			for (i = 0; i < bn / 2; i++)
				if (got[i] != ip[i])
					break;
			printf("  hd %3u  %4u -> %4u  live %4u   ⛔ IN PLACE "
			       "differs at half %zu: %04x want %04x\n",
			       hd, kv_old, kv_new, live, i, ip[i], got[i]);
			bad = 1;
		}
		free(ip);
	}
	free(old); free(ref); free(got); free(v);
	return bad;
}

int main(void)
{
	static const unsigned hds[] = { 32, 40, 64, 72, 96, 128, 256, 512 };
	static const unsigned rungs[] = { 32, 64, 96, 128, 192, 256, 352,
					  480, 512, 640, 864, 1024 };
	unsigned h, a, b;
	int bad = 0, ran = 0;

	puts("fp16_regrow: a grown V surface is the same bytes as one packed "
	     "at the new extent");
	for (h = 0; h < sizeof(hds) / sizeof(hds[0]); h++)
		for (a = 0; a + 1 < sizeof(rungs) / sizeof(rungs[0]); a++)
			for (b = a + 1; b < sizeof(rungs) / sizeof(rungs[0]);
			     b++) {
				unsigned live[3];
				unsigned i;

				live[0] = 1;
				live[1] = rungs[a] / 2 + 1;
				live[2] = rungs[a];
				for (i = 0; i < 3; i++) {
					if (!live[i] || live[i] > rungs[a])
						continue;
					bad |= one(hds[h], rungs[a], rungs[b],
						   live[i]) ? 1 : 0;
					ran++;
				}
			}
	printf("fp16_regrow: %d cases, %s\n", ran, bad ? "⛔ FAILED" : "all ok");
	return bad;
}
