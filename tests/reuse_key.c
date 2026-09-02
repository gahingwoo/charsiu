// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The input reuse key, replayed on the host in the shape phase 22 found on
 * the board.
 *
 * Two cores, one buffer. llama hands q, k and v the same normed buffer, then
 * hands gate and up the same buffer again with new contents, and the deal
 * sends each tensor where the load is lightest, so a follower can land on a
 * core its leader never packed. The key on that core must not be believed
 * after a leader, whatever it says.
 *
 * This is the half a desk can verify: which calls hit. That a hit reads the
 * right bytes is the board's (phase 22: 15 of 15 identical with the drop,
 * 2026-09-02). Before the leader drop, the fourth case here hit.
 */
#include <stdio.h>
#include "../src/reusekey.h"

static float bufA[8], bufB[8];
static int bad, n;

static void expect(int got, int want, const char *what)
{
	n++;
	if (got != want) {
		printf("  reuse key: %s: expected %s, got %s\n", what,
		       want ? "hit" : "miss", got ? "hit" : "miss");
		bad++;
	}
}

/* one leader call: drop, then every device it is dealt sets its key */
static void leader(struct reuse_key *ks, const float *x, unsigned devs,
		   unsigned kslices)
{
	reuse_keys_drop(ks, 2);
	for (unsigned d = 0; d < 2; d++)
		if (devs & (1u << d))
			reuse_key_set(&ks[d], x, 4, 3072, 0x80, kslices);
}

/* one follower call on device d: hit, or pack and set */
static int follower(struct reuse_key *ks, const float *x, unsigned d,
		    unsigned need)
{
	int hit = reuse_key_hit(&ks[d], x, 4, 3072, 0x80, need);

	if (!hit)
		reuse_key_set(&ks[d], x, 4, 3072, 0x80, need);
	return hit;
}

int main(void)
{
	struct reuse_key ks[2] = { { 0 } };

	/* q packs core 0 alone; k dealt to core 0 hits, v dealt to core 1 misses */
	leader(ks, bufA, 1u << 0, 3u);
	expect(follower(ks, bufA, 0, 3u), 1, "k after q, the core q packed");
	expect(follower(ks, bufA, 1, 3u), 0, "v after q, the core q did not pack");
	/* v's own pack landed on core 1, so a third follower there hits */
	expect(follower(ks, bufA, 1, 3u), 1, "a third follower, core 1 packed by v");

	/*
	 * THE PHASE 22 SHAPE. gate packs the SAME buffer -- new contents -- on
	 * core 1 only, and up is dealt to core 0, whose key still names this
	 * buffer from q. That key describes the attention input now.
	 */
	leader(ks, bufA, 1u << 1, 3u);
	expect(follower(ks, bufA, 0, 3u), 0, "up after gate, the core gate did not pack");
	expect(follower(ks, bufA, 1, 3u), 1, "up after gate, the core gate packed");

	/* a different buffer never hits */
	leader(ks, bufA, 3u, 3u);
	expect(follower(ks, bufB, 0, 3u), 0, "another buffer, core 0");
	expect(follower(ks, bufB, 1, 3u), 0, "another buffer, core 1");

	/* the K slices packed must cover the K slices read */
	leader(ks, bufA, 3u, 1u);
	expect(follower(ks, bufA, 0, 3u), 0, "leader packed one K slice, follower reads two");
	expect(follower(ks, bufA, 1, 1u), 1, "leader packed one K slice, follower reads it");

	/* the width, K and zero point are part of the key */
	leader(ks, bufA, 3u, 3u);
	expect(reuse_key_hit(&ks[0], bufA, 6, 3072, 0x80, 3u), 0, "wider");
	expect(reuse_key_hit(&ks[0], bufA, 4, 2048, 0x80, 3u), 0, "narrower K");
	expect(reuse_key_hit(&ks[0], bufA, 4, 3072, 0x7f, 3u), 0, "another zero point");
	expect(reuse_key_hit(&ks[0], bufA, 4, 3072, 0x80, 3u), 1, "the same");

	printf("  reuse key: %d of %d cases wrong\n", bad, n);
	return bad != 0;
}
