/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * What a device's input BO holds right now, so a caller that declares its
 * input unchanged (charsiu_npu_matmul_same) can skip the pack.
 *
 * ⚠⚠ THE RULE, AND THE ROUND THAT WROTE IT. The key is the pointer, the
 * width, K, the zero point and which K slices landed; the CONTENTS are the
 * caller's word, and the caller's word covers exactly one thing: "X is what
 * it was at the call before this one". So a key is good only from the pack
 * that wrote it until the next call that does NOT declare its input
 * unchanged -- a leader -- and a leader drops every device's key before it
 * packs, whether or not it packs that device.
 *
 * Without the drop, phase 22 on the board: llama keeps ONE buffer for the
 * normed input of q, k and v and for that of gate and up, so q packs it on
 * core 0, gate packs it -- new contents, same pointer, same m, same K -- on
 * core 1 only, and up, dealt to core 0, meets a key that still says "this
 * buffer" and multiplies the attention input. Phi-3.5 broke on k, gemma4 on
 * up, Qwen3 on neither: which site breaks is which core the deal happens to
 * send the follower to, nothing about the tensor. "Views of a fused tensor"
 * and "per-layer embeddings" were both guesses at that pattern, both wrong,
 * and the per-device key before this was the fix for a different fault (a
 * follower dealt to a core the leader never packed) that happened to look
 * the same from the text.
 *
 * ⚠⚠ AND THE BYTES ARE NOT X WHEN AWQ IS ON. npuquant scales a tensor's
 * weights by kscale[k] and leaves the inverse for the caller to put on the
 * ACTIVATION, so what a device's BO holds is X times THAT TENSOR'S factor,
 * and the next tensor's factor is a different vector. The pointer is the
 * same, the width is the same, K is the same and the zero point is the
 * same: every field this key had said hit, and a hit would have multiplied
 * gate's scaled input by up's weights.
 *
 * That is the same fault the grouped decode path refuses by comparing
 * kshash, one call site over, and it is why the factor's hash is a field
 * here rather than a rule in npudev.c. With AWQ off every kshash is 0 and
 * this changes nothing.
 *
 * In a header of its own so the bookkeeping can be replayed on a host with
 * no NPU (tests/reuse_key.c, under make test), which is the half of this a
 * desk can verify. That the bytes a hit reads are the right bytes is the
 * board's: phase 22 read identical on all 15 cells with the drop
 * (2026-09-02), and phase 2 runs with reuse on by default.
 */
#ifndef CHARSIU_REUSEKEY_H
#define CHARSIU_REUSEKEY_H
#include <stddef.h>
#include <stdint.h>

struct reuse_key {
	const float *x;
	unsigned m;
	uint64_t k;
	uint8_t zp;
	unsigned kslices;   /* which K slices of X this BO holds, a bit each */
	uint64_t ksh;       /* the AWQ factor these bytes were scaled by, 0 for
			     * none -- see the note below */
	int valid;
};

/* a leader is about to pack: nothing any device holds is X any more */
static inline void reuse_keys_drop(struct reuse_key *ks, unsigned n)
{
	for (unsigned i = 0; i < n; i++)
		ks[i].valid = 0;
}

/*
 * Does this device's BO hold X at this width, K and zero point, in every K
 * slice the caller is about to read? A tensor dealt fewer K slices here
 * than the caller needs is a miss, not a partial hit.
 */
static inline int reuse_key_hit(const struct reuse_key *k, const float *x,
				unsigned m, uint64_t kk, uint8_t zp,
				unsigned need, uint64_t ksh)
{
	return k->valid && k->x == x && k->m == m && k->k == kk &&
	       k->zp == zp && k->ksh == ksh &&
	       (need & k->kslices) == need;
}

/* this device's pack of X landed, in these K slices */
static inline void reuse_key_set(struct reuse_key *k, const float *x,
				 unsigned m, uint64_t kk, uint8_t zp,
				 unsigned kslices, uint64_t ksh)
{
	k->x = x;
	k->m = m;
	k->k = kk;
	k->zp = zp;
	k->kslices = kslices;
	k->ksh = ksh;
	k->valid = 1;
}
#endif
