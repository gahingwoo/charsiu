/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
 * SPDX-License-Identifier: GPL-2.0
 *
 * ⛔ AN ATTACK ON charsiu_fp16_regrow_vcols, NOT A CONFIRMATION OF IT.
 *
 * tests/fp16_regrow.c sweeps a FIXED table of eight head dims and twelve
 * rungs, one step at a time, into destinations that are always freshly
 * zeroed. That table cannot reach four things the function now depends on:
 *
 *   1. a head dim whose padded n is NOT a multiple of the 16 wide output
 *      channel group, and an ODD head dim, where n_pad != hd. The descending
 *      walk starts at ((n_pad-1)/ng)*ng, which is n_pad-ng only when ng
 *      divides n_pad -- head dim 100 pads to 100 and its last group starts
 *      at 96 with four channels in it. The old table's only short final
 *      groups are 40 and 72, both ngsz 8, both even.
 *   2. a reduction extent that is not a whole number of positions per k
 *      group. Every rung in the old table is a multiple of 32, so k_eff
 *      equals kv and the two are never distinguishable. kv 57 has k_eff 64.
 *   3. a CHAIN. In place growth is the default path now and real use climbs
 *      32 -> 64 -> 128 -> 256 -> 864 in the SAME buffer with positions
 *      appended between rungs. A single step arm cannot see an error that
 *      only appears after two moves.
 *   4. an unzeroed destination. The old in place arm calloc's its buffer at
 *      the NEW size, so every byte the function fails to clear reads back as
 *      exactly the zero the reference wanted -- which is the shape of the bug
 *      that shipped this morning, hidden by the harness.
 *
 * So: randomised shapes, chains compared at every rung against a surface
 * packed directly there, guard halves and PROT_NONE guard PAGES at both ends,
 * a poisoned destination tail, and a refusal arm that requires every -1 to be
 * a refusal before any byte moved.
 *
 * ⚠ AND TWO CONTROLS THAT HAVE TO FAIL. A comparison that cannot see the bug
 * that shipped is not evidence. regrow_variant() below is the same block walk
 * with the two properties the function gained today turned OFF -- ascending,
 * and no clear -- and the run FAILS if either of them ever reaches the
 * reference on every case it was applicable to.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include "charsiu.h"

/* ------------------------------------------------------------------ */

static uint32_t rs = 0x9E3779B9u;

static uint32_t rnd(void)
{
	rs ^= rs << 13;
	rs ^= rs >> 17;
	rs ^= rs << 5;
	return rs;
}

static unsigned rrange(unsigned lo, unsigned hi)
{
	return lo + rnd() % (hi - lo + 1u);
}

/* Every channel of every live position is NON ZERO, so "a zero where data
 * should be" -- which is what a dropped block looks like, and what the missing
 * in place clear looked like in reverse -- is always visible. 12 bits of
 * signal per half, all of them exactly representable in fp16. */
static void fill(float *v, unsigned hd, unsigned pos)
{
	unsigned i;

	for (i = 0; i < hd; i++) {
		uint32_t h = pos * 2654435761u + i * 2246822519u;
		int x;

		h ^= h >> 15;
		h *= 2654435761u;
		h ^= h >> 13;
		x = (int)(h % 4095u) - 2047;
		if (x >= 0)
			x++;
		v[i] = (float)x;
	}
}

static size_t wbytes(unsigned kv, unsigned hd)
{
	struct charsiu_matmul mm = { 1, kv, hd, CHARSIU_FP16, CHARSIU_FP16 };

	return charsiu_weight_bytes(&mm);
}

static unsigned keff(unsigned kv, unsigned hd)
{
	struct charsiu_matmul mm = { 1, kv, hd, CHARSIU_FP16, CHARSIU_FP16 };

	return charsiu_k_eff(&mm);
}

/* the documented acceptance set, restated here so the run can say when the
 * function refuses something it promised to take, or takes something it
 * promised to refuse */
static int should_accept(unsigned hd, unsigned kv_old, unsigned kv_new,
			 unsigned live)
{
	unsigned kg = charsiu_weight_kgroup(CHARSIU_FP16);

	if (!hd || !live || !kv_old || !kv_new)
		return 0;
	if (kv_new < kv_old || live > kv_old)
		return 0;
	if (keff(kv_old, hd) % kg || keff(kv_new, hd) % kg)
		return 0;
	return 1;
}

static void pack_into(uint16_t *buf, unsigned kv, unsigned hd, unsigned lo,
		      unsigned hi)
{
	float v[1024];
	unsigned p;

	for (p = lo; p < hi; p++) {
		fill(v, hd, p);
		charsiu_fp16_pack_vcol(buf, kv, hd, p, v);
	}
}

/*
 * ⚠ DECODE LEAVES GAPS. attn_npu_fit passes live = pos -- every position in
 * the FLOAT cache -- and the mirror below it is not full: decode stopped
 * appending and attn_npu_catchup fills the holes later. So the surface handed
 * to the regrow routinely has zeros at positions below `live`, and the
 * reference has to have them in the same places.
 */
static int is_packed(unsigned p, unsigned gap)
{
	return !gap || (p % gap) != gap - 1u;
}

static void pack_sel(uint16_t *buf, unsigned kv, unsigned hd, unsigned lo,
		     unsigned hi, unsigned gap)
{
	float v[1024];
	unsigned p;

	for (p = lo; p < hi; p++) {
		if (!is_packed(p, gap))
			continue;
		fill(v, hd, p);
		charsiu_fp16_pack_vcol(buf, kv, hd, p, v);
	}
}

/* ------------------------------------------------------------------ */
/* guarded buffers: GUARD halves of 0xA5A5 on each side of the surface */

#define GUARD 1024u
#define GB    0xA5A5u

struct buf {
	uint16_t *mem;
	uint16_t *b;
	size_t half;        /* the surface, at the CEILING extent */
};

static void buf_new(struct buf *x, size_t bytes)
{
	size_t h = bytes / 2, i;

	x->half = h;
	x->mem = malloc((h + 2 * GUARD) * 2);
	if (!x->mem)
		exit(2);
	for (i = 0; i < h + 2 * GUARD; i++)
		x->mem[i] = GB;
	x->b = x->mem + GUARD;
	memset(x->b, 0, h * 2);
}

static void buf_free(struct buf *x)
{
	free(x->mem);
	x->mem = x->b = NULL;
}

static void poison(uint16_t *b, size_t from_half, size_t to_half)
{
	size_t i;

	for (i = from_half; i < to_half; i++)
		b[i] = GB;
}

static int guards_ok(const struct buf *x, size_t *where, int *before)
{
	size_t i;

	for (i = 0; i < GUARD; i++)
		if (x->mem[i] != GB) {
			*where = GUARD - i;
			*before = 1;
			return 0;
		}
	for (i = 0; i < GUARD; i++)
		if (x->b[x->half + i] != GB) {
			*where = i + 1;
			*before = 0;
			return 0;
		}
	return 1;
}

static int all_equal(const uint16_t *a, const uint16_t *b, size_t halves,
		     size_t *where)
{
	size_t i;

	if (!memcmp(a, b, halves * 2))
		return 1;
	for (i = 0; i < halves; i++)
		if (a[i] != b[i]) {
			*where = i;
			return 0;
		}
	*where = 0;
	return 0;
}

/* ------------------------------------------------------------------ */
/*
 * ⚠⚠ THE MUTANTS, BECAUSE A COMPARISON THAT CANNOT SEE A BUG IS NOT EVIDENCE.
 *
 * The same block walk as src/regcmd.c, with one property broken at a time.
 * Every one of these is a plausible way to write this function and two of them
 * are versions it actually WAS this morning. The run FAILS if any mutant is
 * never caught by the same comparison the real function is held to -- that is
 * the only thing that makes "all ok" mean anything.
 *
 * ⚠ MUT_NONE is the faithfulness check: the unmutated copy has to agree with
 * the real function byte for byte on every shape, or the mutants below are
 * mutations of something else.
 */
enum {
	MUT_NONE = 0,
	MUT_ASC,        /* walk the groups bottom up -- in place cannot survive it */
	MUT_NOCLEAR,    /* the first cut of the in place arm: never clear */
	MUT_TOPBASE,    /* start at n_pad - ng instead of the largest multiple
			 * of ng below n_pad -- the same thing only when ng
			 * divides n_pad */
	MUT_LEN_NG,     /* len uses ng where it needs ngsz (short final group) */
	MUT_LEN_RAW,    /* len uses live instead of rounding it up to a k group */
	MUT_NPAD_HD,    /* n_pad = hd, forgetting that N is rounded to a pair */
	MUT_BLK_OLD,    /* clear ngsz*keo instead of ngsz*ken */
	MUT_SKIPLAST,   /* break before the n0 == 0 group */
	MUT_N
};

static const char *mut_name[MUT_N] = {
	"(faithful copy)", "ascending walk", "never clears",
	"start at n_pad-ng", "len uses ng not ngsz", "len not rounded to kg",
	"n_pad = hd", "clear uses the OLD extent", "skips the last group"
};

static int regrow_variant(uint16_t *d, unsigned kv_new, const uint16_t *s,
			  unsigned kv_old, unsigned hd, unsigned live, int mut)
{
	struct charsiu_matmul mo = { 1, kv_old, hd, CHARSIU_FP16,
				     CHARSIU_FP16 };
	struct charsiu_matmul mn = { 1, kv_new, hd, CHARSIU_FP16,
				     CHARSIU_FP16 };
	unsigned ng = charsiu_weight_ngroup(CHARSIU_FP16);
	unsigned kg = charsiu_weight_kgroup(CHARSIU_FP16);
	unsigned n_pad = mut == MUT_NPAD_HD ? hd : ((hd + 1u) & ~1u);
	unsigned keo = charsiu_k_eff(&mo), ken = charsiu_k_eff(&mn);
	int asc = mut == MUT_ASC;
	unsigned top, n0;

	if (!should_accept(hd, kv_old, kv_new, live))
		return -1;
	if (mut == MUT_TOPBASE)
		top = n_pad > ng ? n_pad - ng : 0u;
	else
		top = ((n_pad - 1u) / ng) * ng;
	for (n0 = asc ? 0u : top; ; n0 = asc ? n0 + ng
					     : (n0 < ng ? 0u : n0 - ng)) {
		unsigned ngsz = n_pad - n0 < ng ? n_pad - n0 : ng;
		unsigned lf = mut == MUT_LEN_NG ? ng : ngsz;
		size_t oo = charsiu_w16_offset(&mo, n0, 0, CHARSIU_W16_GROUP);
		size_t on = charsiu_w16_offset(&mn, n0, 0, CHARSIU_W16_GROUP);
		size_t len = mut == MUT_LEN_RAW
			   ? (size_t)live * lf
			   : (size_t)((live + kg - 1) / kg) * kg * lf;

		if (oo == (size_t)-1 || on == (size_t)-1)
			return -1;
		if (mut == MUT_SKIPLAST && !asc && !n0)
			break;
		memmove(d + on / 2, s + oo / 2, len * 2);
		if (mut != MUT_NOCLEAR && (const uint16_t *)d == s) {
			size_t blk = (size_t)ngsz *
				     (mut == MUT_BLK_OLD ? keo : ken);

			if (blk > len)
				memset(d + on / 2 + len, 0, (blk - len) * 2);
		}
		if (asc ? n0 >= top : n0 == 0)
			break;
	}
	return 0;
}

/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */

static int fails;
static long cases;
static long refused;
static long accepted;
static long mut_n[MUT_N], mut_same[MUT_N];

static void bad(const char *arm, unsigned hd, unsigned kv_old,
		unsigned kv_new, unsigned live, const char *what, size_t at)
{
	if (fails < 40)
		printf("  ⛔ %-4s hd=%u kv_old=%u kv_new=%u live=%u : %s"
		       " (half %zu)\n", arm, hd, kv_old, kv_new, live, what,
		       at);
	fails++;
}

/*
 * One shape, five arms:
 *   SRC    the source buffer must come back byte identical
 *   COPY   dst != src, dst zeroed -- the contract the old test had
 *   INPL   dst == src, the room ABOVE kv_new poisoned: nothing may be written
 *          past the new extent
 *   TAIL   dst == src, the room BETWEEN the extents poisoned: the in place
 *          arm must not be leaning on that room having been zero
 *   refusal: -1 must leave every byte of both buffers alone
 */
static void one(unsigned hd, unsigned kv_old, unsigned kv_new, unsigned live)
{
	size_t bo = wbytes(kv_old, hd), bn = wbytes(kv_new, hd);
	size_t cb;                                   /* the ceiling, above bn */
	struct buf src, dst, ip, tail, ref;
	size_t at = 0, gw = 0;
	int gb4 = 0;
	int want = should_accept(hd, kv_old, kv_new, live);
	int r;

	cases++;
	cb = wbytes(kv_new + 64u, hd);
	if (cb < bn)
		cb = bn;
	buf_new(&src, bo);
	buf_new(&dst, bn);
	buf_new(&ref, bn);
	buf_new(&ip, cb);
	buf_new(&tail, cb);

	if (want) {
		pack_into(src.b, kv_old, hd, 0, live);
		pack_into(ref.b, kv_new, hd, 0, live);
		pack_into(ip.b, kv_old, hd, 0, live);
		pack_into(tail.b, kv_old, hd, 0, live);
	}
	poison(ip.b, bn / 2, cb / 2);
	poison(tail.b, bo / 2, cb / 2);

	{
		size_t whole = bo + 2 * GUARD * 2;
		uint16_t *snap = malloc(whole);

		memcpy(snap, src.mem, whole);
		r = charsiu_fp16_regrow_vcols(dst.b, kv_new, src.b, kv_old,
					      hd, live);
		if (memcmp(snap, src.mem, whole))
			bad("SRC", hd, kv_old, kv_new, live,
			    "the source buffer was modified", 0);
		free(snap);
	}
	if (r) {
		size_t i;

		refused++;
		if (want)
			bad("COPY", hd, kv_old, kv_new, live,
			    "REFUSED a shape the documented conditions accept",
			    0);
		/* ⚠ a refusal must be a refusal BEFORE any byte moved: the
		 * caller treats -1 as "nothing happened" and then hands this
		 * very buffer to the packer */
		for (i = 0; i < bn / 2; i++)
			if (dst.b[i] != 0) {
				bad("COPY", hd, kv_old, kv_new, live,
				    "refused AFTER writing to dst", i);
				break;
			}
		if (!guards_ok(&dst, &gw, &gb4))
			bad("COPY", hd, kv_old, kv_new, live,
			    gb4 ? "a refusal wrote before dst"
				: "a refusal wrote past dst", gw);
		goto out;
	}
	accepted++;
	if (!want)
		bad("COPY", hd, kv_old, kv_new, live,
		    "ACCEPTED a shape the documented conditions refuse", 0);
	if (!all_equal(ref.b, dst.b, bn / 2, &at))
		bad("COPY", hd, kv_old, kv_new, live,
		    "differs from a surface packed at the new extent", at);
	if (!guards_ok(&dst, &gw, &gb4))
		bad("COPY", hd, kv_old, kv_new, live,
		    gb4 ? "wrote before dst" : "wrote past dst", gw);

	/* the in place arm, with the room above kv_new poisoned */
	{
		uint16_t *snap = malloc(cb);

		memcpy(snap, ip.b, cb);
		if (charsiu_fp16_regrow_vcols(ip.b, kv_new, ip.b, kv_old, hd,
					      live)) {
			bad("INPL", hd, kv_old, kv_new, live,
			    "IN PLACE refused what the copy accepted", 0);
		} else {
			if (!all_equal(ref.b, ip.b, bn / 2, &at))
				bad("INPL", hd, kv_old, kv_new, live,
				    "differs from a surface packed at the new "
				    "extent", at);
			if (memcmp(snap + bn / 2, ip.b + bn / 2, cb - bn)) {
				size_t i;

				for (i = bn / 2; i < cb / 2; i++)
					if (snap[i] != ip.b[i])
						break;
				bad("INPL", hd, kv_old, kv_new, live,
				    "wrote ABOVE the new extent", i);
			}
			if (!guards_ok(&ip, &gw, &gb4))
				bad("INPL", hd, kv_old, kv_new, live,
				    gb4 ? "wrote before the surface"
					: "wrote past the surface", gw);
		}
		free(snap);
	}
	/* the same, with the room BETWEEN the extents poisoned too */
	if (!charsiu_fp16_regrow_vcols(tail.b, kv_new, tail.b, kv_old, hd,
				       live)) {
		if (!all_equal(ref.b, tail.b, bn / 2, &at))
			bad("TAIL", hd, kv_old, kv_new, live,
			    "in place leans on a pre zeroed destination tail",
			    at);
	}
	/* ⚠ THE MUTANTS, on this shape, held to the SAME three checks the real
	 * function is held to: the compared region, the poisoned room above
	 * the new extent, and the guard halves. Counted rather than asserted
	 * per case -- an ascending walk is the same walk when there is only
	 * one group -- but each one has to be caught SOMEWHERE, and the
	 * faithful copy has to be caught NOWHERE. */
	{
		struct buf c;
		/* room for the mutant that overruns its own block */
		size_t vb = cb + (size_t)2 * charsiu_weight_ngroup(
					CHARSIU_FP16) * keff(kv_new, hd) * 2;
		int m;

		for (m = 0; m < MUT_N; m++) {
			uint16_t *snap;
			int caught;
			size_t gw = 0;
			int gb4 = 0;

			buf_new(&c, vb);
			pack_into(c.b, kv_old, hd, 0, live);
			poison(c.b, bn / 2, vb / 2);
			snap = malloc(vb);
			memcpy(snap, c.b, vb);
			if (regrow_variant(c.b, kv_new, c.b, kv_old, hd, live,
					   m)) {
				free(snap);
				buf_free(&c);
				continue;
			}
			mut_n[m]++;
			caught = !all_equal(ref.b, c.b, bn / 2, &at)
			       || memcmp(snap + bn / 2, c.b + bn / 2, vb - bn)
			       || !guards_ok(&c, &gw, &gb4);
			if (!caught)
				mut_same[m]++;
			else if (m == MUT_NONE)
				bad("MUT", hd, kv_old, kv_new, live,
				    "the faithful copy of the walk disagrees "
				    "with the real function", at);
			free(snap);
			buf_free(&c);
		}
	}
out:
	buf_free(&src); buf_free(&dst); buf_free(&ref);
	buf_free(&ip); buf_free(&tail);
}

/* ------------------------------------------------------------------ */
/*
 * ⭐ THE LADDER, IN ONE BUFFER, WITH POSITIONS APPENDED BETWEEN RUNGS.
 *
 * attn_npu_append calls attn_npu_fit when the position that has arrived does
 * not fit, with live = pos -- every position already in the float cache, this
 * one not being in it yet -- and then packs pos itself at the NEW kv. The
 * surface was allocated once at the prompt's ceiling and is re laid out where
 * it lies. This is that, with the caller's own fallback: a -1 means a fresh
 * surface and every live position back through the packer.
 */
static void chain(unsigned hd, const unsigned *rungs, unsigned nr,
		  unsigned total, unsigned gap, int mode)
{
	unsigned ceiling = rungs[nr - 1];
	size_t cb = wbytes(ceiling, hd);
	struct buf s, ref;
	unsigned kv = rungs[0], ri = 0, p;
	size_t at = 0;
	unsigned nrung = 0;

	if (total > ceiling)
		total = ceiling;
	if (!total)
		return;
	cases++;
	buf_new(&s, cb);
	buf_new(&ref, cb);
	for (p = 0; p < total; p++) {
		if (p >= kv) {
			unsigned live = p, nk = kv;

			while (nk <= p) {
				if (ri + 1 < nr) {
					ri++;
					nk = rungs[ri];
				} else {
					nk = ceiling;
					break;
				}
			}
			if (nk <= p)
				break;         /* the ladder cannot reach */
			/* ⚠ mode 1 is the allocating rung the caller still
			 * takes when the surface has no room, and mode 2 is
			 * what attn_npu_fit does when `copied` drops in the
			 * middle of the layer loop: some rungs in place, some
			 * into a buffer that was just allocated and zeroed */
			if (mode == 1 || (mode == 2 && (nrung & 1))) {
				struct buf fresh;

				buf_new(&fresh, cb);
				if (charsiu_fp16_regrow_vcols(fresh.b, nk, s.b,
							      kv, hd, live)) {
					memset(fresh.b, 0, wbytes(nk, hd));
					pack_sel(fresh.b, nk, hd, 0, live, gap);
				}
				memcpy(s.b, fresh.b, cb);
				buf_free(&fresh);
			} else if (charsiu_fp16_regrow_vcols(s.b, nk, s.b, kv,
							     hd, live)) {
				memset(s.b, 0, wbytes(nk, hd));
				pack_sel(s.b, nk, hd, 0, live, gap);
			}
			kv = nk;
			nrung++;
			/* ⚠ AT EVERY RUNG. An error repaired by the next move
			 * is still an error: the surface is uploaded to the
			 * hardware at each one. */
			{
				struct buf r2;
				size_t h = wbytes(kv, hd) / 2;

				buf_new(&r2, h * 2);
				pack_sel(r2.b, kv, hd, 0, live, gap);
				if (!all_equal(r2.b, s.b, h, &at))
					bad("RUNG", hd, kv, kv, live,
					    "the climbing surface differs from "
					    "one packed at this rung", at);
				/* ⭐ AND GROWING TO THE EXTENT IT IS ALREADY
				 * AT MUST BE A NO OP. kv_new == kv_old is
				 * accepted, memmoves every block onto itself
				 * and then clears past `live` -- on a live
				 * surface, in place. */
				if (charsiu_fp16_regrow_vcols(s.b, kv, s.b, kv,
							      hd, live)) {
					if (should_accept(hd, kv, kv, live))
						bad("IDEM", hd, kv, kv, live,
						    "refused a same extent "
						    "regrow", 0);
				} else if (!all_equal(r2.b, s.b, h, &at)) {
					bad("IDEM", hd, kv, kv, live,
					    "a same extent regrow changed the "
					    "surface", at);
				}
				buf_free(&r2);
			}
			{
				size_t i, h0 = wbytes(kv, hd) / 2, gw = 0;
				int gb4 = 0;

				for (i = h0; i < cb / 2; i++)
					if (s.b[i] != 0) {
						bad("RUNG", hd, kv, kv, live,
						    "wrote above the current "
						    "extent", i);
						break;
					}
				if (!guards_ok(&s, &gw, &gb4))
					bad("RUNG", hd, kv, kv, live,
					    gb4 ? "wrote before the surface"
						: "wrote past the surface",
					    gw);
			}
		}
		pack_sel(s.b, kv, hd, p, p + 1, gap);
	}
	pack_sel(ref.b, kv, hd, 0, total, gap);
	if (!all_equal(ref.b, s.b, wbytes(kv, hd) / 2, &at))
		bad("CHN", hd, rungs[0], kv, total,
		    "the climbed surface differs from one packed directly at "
		    "the final extent", at);
	if (total < kv) {
		pack_into(ref.b, kv, hd, total, total + 1);
		if (all_equal(ref.b, s.b, wbytes(kv, hd) / 2, &at))
			bad("CHN", hd, rungs[0], kv, total,
			    "CONTROL: an extra packed position did not change "
			    "the buffer", 0);
	}
	buf_free(&s);
	buf_free(&ref);
}

/*
 * ⭐ THE SAME LADDER WITH NOTHING APPENDED BETWEEN THE RUNGS, which is the
 * worst case for the clear: `live` stays where it started while the extent
 * climbs, so the vacated area grows at every rung and nothing ever writes
 * over it again. live 1 against a 32 -> 864 ladder leaves 863 positions of
 * every group that have to be zero and were not, the first time.
 */
static void grow_only(unsigned hd, const unsigned *rungs, unsigned nr,
		      unsigned live, unsigned gap)
{
	size_t cb = wbytes(rungs[nr - 1], hd);
	struct buf s;
	unsigned q, kv = rungs[0];
	size_t at = 0;

	if (live > rungs[0])
		live = rungs[0];
	if (!live)
		return;
	cases++;
	buf_new(&s, cb);
	pack_sel(s.b, kv, hd, 0, live, gap);
	for (q = 1; q < nr; q++) {
		struct buf r2;
		size_t h;

		if (rungs[q] < kv)
			continue;
		if (charsiu_fp16_regrow_vcols(s.b, rungs[q], s.b, kv, hd,
					      live)) {
			memset(s.b, 0, wbytes(rungs[q], hd));
			pack_sel(s.b, rungs[q], hd, 0, live, gap);
		}
		kv = rungs[q];
		h = wbytes(kv, hd) / 2;
		buf_new(&r2, h * 2);
		pack_sel(r2.b, kv, hd, 0, live, gap);
		if (!all_equal(r2.b, s.b, h, &at))
			bad("GROW", hd, kv, kv, live,
			    "repeated growth with nothing appended differs "
			    "from a direct pack", at);
		buf_free(&r2);
		{
			size_t i, gw = 0;
			int gb4 = 0;

			for (i = h; i < cb / 2; i++)
				if (s.b[i] != 0) {
					bad("GROW", hd, kv, kv, live,
					    "wrote above the current extent",
					    i);
					break;
				}
			if (!guards_ok(&s, &gw, &gb4))
				bad("GROW", hd, kv, kv, live,
				    gb4 ? "wrote before the surface"
					: "wrote past the surface", gw);
		}
	}
	buf_free(&s);
}

/*
 * ⭐ A REAL CONTEXT LENGTH. kvmax is n_ctx rounded to 32, so a 32k context
 * gives a top rung of 32768 and an offset of ngi*ng*ke that has to be done in
 * size_t: at head dim 256 that is 8.4 million halves. Light arms only -- the
 * copy and the in place move against a direct pack -- because the buffers are
 * 16 MB each.
 */
static void huge(unsigned hd, unsigned kv_old, unsigned kv_new, unsigned live)
{
	size_t bn = wbytes(kv_new, hd);
	struct buf src, ip, ref;
	size_t at = 0, gw = 0;
	int gb4 = 0;

	cases++;
	buf_new(&src, wbytes(kv_old, hd));
	buf_new(&ip, bn);
	buf_new(&ref, bn);
	pack_into(src.b, kv_old, hd, 0, live);
	pack_into(ip.b, kv_old, hd, 0, live);
	pack_into(ref.b, kv_new, hd, 0, live);
	if (charsiu_fp16_regrow_vcols(ip.b, kv_new, ip.b, kv_old, hd, live))
		bad("HUGE", hd, kv_old, kv_new, live, "refused", 0);
	else if (!all_equal(ref.b, ip.b, bn / 2, &at))
		bad("HUGE", hd, kv_old, kv_new, live,
		    "in place differs from a direct pack", at);
	if (!guards_ok(&ip, &gw, &gb4))
		bad("HUGE", hd, kv_old, kv_new, live, "guard halves", gw);
	buf_free(&src);
	buf_free(&ip);
	buf_free(&ref);
}

/* ------------------------------------------------------------------ */
/*
 * ⭐ PROT_NONE PAGES, because guard halves only catch a write that lands in
 * the slack. A read OR a write one byte past the surface faults here instead.
 * Two sub arms: the surface flush against a dead page at its END catches an
 * overrun, flush against one at its START catches an underrun.
 */
static uint16_t *page_buf(size_t bytes, int at_end, void **base, size_t *len)
{
	size_t ps = (size_t)sysconf(_SC_PAGESIZE);
	size_t need = ((bytes + ps - 1) / ps) * ps;
	size_t tot = need + 2 * ps;
	char *m = mmap(NULL, tot, PROT_READ | PROT_WRITE,
		       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

	if (m == MAP_FAILED)
		exit(2);
	mprotect(m, ps, PROT_NONE);
	mprotect(m + ps + need, ps, PROT_NONE);
	memset(m + ps, 0, need);
	*base = m;
	*len = tot;
	return (uint16_t *)(void *)(at_end ? m + ps + need - bytes : m + ps);
}

static void paged(unsigned hd, unsigned kv_old, unsigned kv_new, unsigned live,
		  int at_end)
{
	void *sb, *db, *ib;
	size_t sl, dl, il;
	size_t bo = wbytes(kv_old, hd), bn = wbytes(kv_new, hd);
	uint16_t *s, *d, *ip;
	struct buf ref;
	size_t at = 0;

	if (!should_accept(hd, kv_old, kv_new, live))
		return;
	cases++;
	s = page_buf(bo, at_end, &sb, &sl);
	d = page_buf(bn, at_end, &db, &dl);
	ip = page_buf(bn, at_end, &ib, &il);
	pack_into(s, kv_old, hd, 0, live);
	pack_into(ip, kv_old, hd, 0, live);
	buf_new(&ref, bn);
	pack_into(ref.b, kv_new, hd, 0, live);
	if (charsiu_fp16_regrow_vcols(d, kv_new, s, kv_old, hd, live))
		bad("PAGE", hd, kv_old, kv_new, live, "refused", 0);
	else if (!all_equal(ref.b, d, bn / 2, &at))
		bad("PAGE", hd, kv_old, kv_new, live,
		    "differs under guard pages", at);
	if (!charsiu_fp16_regrow_vcols(ip, kv_new, ip, kv_old, hd, live))
		if (!all_equal(ref.b, ip, bn / 2, &at))
			bad("PAGE", hd, kv_old, kv_new, live,
			    "in place differs under guard pages", at);
	buf_free(&ref);
	munmap(sb, sl);
	munmap(db, dl);
	munmap(ib, il);
}

/* ------------------------------------------------------------------ */

/* an extent the function will take: k_eff must be a whole number of k groups,
 * which for fp16 (feature atom 8, k group 32) means kv in [32m-7, 32m] */
static unsigned good_extent(unsigned m, unsigned off)
{
	unsigned kv;

	if (!m)
		m = 1;
	kv = m * 32u - (off % 8u);
	return kv ? kv : 32u;
}

static const unsigned nasty_hd[] = {
	1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 18, 30, 31, 32, 33, 34,
	40, 47, 48, 49, 63, 64, 65, 66, 71, 72, 80, 95, 96, 97,
	100, 112, 126, 127, 128, 129, 160, 191, 192, 200, 255, 256,
	257, 288, 300, 320, 384, 400, 511, 512
};
#define NHD (sizeof(nasty_hd) / sizeof(nasty_hd[0]))

/* live values that sit on the k group edges as often as they sit anywhere:
 * len is ceil(live/kg)*kg*ngsz, so live == kg, kg+1 and kg-1 are the three
 * places the copied span changes size */
static unsigned pick_live(unsigned kv)
{
	unsigned kg = charsiu_weight_kgroup(CHARSIU_FP16);
	unsigned m;

	switch (rnd() % 8) {
	case 0: return 0;
	case 1: return 1;
	case 2: return kv;
	case 3: return kv ? kv - 1 : 0;
	case 4:
	case 5:
		m = rrange(0, kv / kg + 1u) * kg;
		m += rnd() % 3;                  /* kg*i, +1, +2 */
		if (m)
			m--;                     /* and kg*i - 1 */
		return m > kv ? kv : m;
	default:
		return rrange(0, kv);
	}
}

static unsigned pick_hd(unsigned cap)
{
	unsigned hd;

	if (rnd() & 1)
		hd = nasty_hd[rnd() % NHD];
	else
		hd = rrange(1, cap);
	return hd > cap ? rrange(1, cap) : hd;
}

int main(void)
{
	static const unsigned ladder[] = { 32, 64, 128, 256, 512, 864 };
	static const unsigned ext1[] = { 25, 28, 32, 33, 40, 48, 57, 60,
					 64, 89, 96 };
	unsigned i, j, k;
	long phase;

	{
		const char *e = getenv("CHARSIU_FUZZ_SEED");

		if (e && *e)
			rs = (uint32_t)strtoul(e, NULL, 0) | 1u;
	}
	puts("fp16_regrow_fuzz: randomised shapes, chains, a poisoned tail and "
	     "guard pages");
	printf("  seed 0x%08x  (CHARSIU_FUZZ_SEED to change it)\n", rs);
	printf("  ng=%u kg=%u  k_eff(32)=%u k_eff(57)=%u  (fp16)\n",
	       charsiu_weight_ngroup(CHARSIU_FP16),
	       charsiu_weight_kgroup(CHARSIU_FP16),
	       keff(32, 64), keff(57, 64));

	/* ---- phase 1: exhaustive over small head dims -------------------- */
	phase = cases;
	for (i = 1; i <= 48; i++)
		for (j = 0; j < sizeof(ext1) / sizeof(ext1[0]); j++)
			for (k = 0; k < sizeof(ext1) / sizeof(ext1[0]); k++) {
				unsigned a = ext1[j], b = ext1[k];
				unsigned lv[8], n = 0, q;

				lv[n++] = 0;
				lv[n++] = 1;
				lv[n++] = 31;
				lv[n++] = 32;
				lv[n++] = 33;
				lv[n++] = a / 2;
				lv[n++] = a - 1;
				lv[n++] = a;
				for (q = 0; q < n; q++)
					if (lv[q] <= a)
						one(i, a, b, lv[q]);
			}
	printf("  phase 1  head dim 1..48 (odd, and short final groups) "
	       "against 11 extents: %ld cases\n", cases - phase);

	/* ---- phase 2: randomised, one step -------------------------------- */
	phase = cases;
	for (i = 0; i < 40000; i++) {
		unsigned hd = pick_hd(160);
		unsigned a = good_extent(rrange(1, 8), rnd());
		unsigned b = good_extent(rrange(1, 8), rnd());

		if (rnd() % 8 == 0)
			b = rrange(1, 300);
		if (rnd() % 8 == 0)
			a = rrange(1, 300);
		one(hd, a, b, pick_live(a));
	}
	for (i = 0; i < 6000; i++) {
		unsigned hd = pick_hd(512);
		unsigned a = good_extent(rrange(1, 27), rnd());
		unsigned b = good_extent(rrange(1, 27), rnd());

		one(hd, a, b, pick_live(a));
	}
	/* the extents that are NOT a whole number of positions per k group,
	 * on their own: kv in [32m-7, 32m] all share one k_eff */
	for (i = 0; i < 6000; i++) {
		unsigned hd = pick_hd(200);
		unsigned m = rrange(1, 12);
		unsigned a = m * 32u - rrange(0, 7);
		unsigned b = rrange(m, m + 4) * 32u - rrange(0, 7);

		if (b < a)
			b = a;
		one(hd, a, b, pick_live(a));
	}
	printf("  phase 2  %ld randomised one step shapes, head dim up to "
	       "512, extent up to 864\n", cases - phase);

	/* ---- phase 3: the refusal path, byte for byte --------------------- */
	phase = cases;
	one(64, 64, 128, 0);           /* live 0 */
	one(64, 64, 128, 65);          /* live > kv_old */
	one(64, 128, 64, 32);          /* a shrink */
	one(64, 64, 64, 64);           /* kv_new == kv_old, live == kv_old */
	one(64, 64, 64, 1);
	one(64, 33, 128, 16);          /* k_eff 40: not a whole k group */
	one(64, 64, 129, 16);          /* k_eff 136 */
	one(64, 1, 32, 1);
	one(0, 64, 128, 32);           /* hd 0 */
	one(1, 32, 864, 32);           /* hd 1: n_pad 2, one short group */
	one(100, 32, 864, 32);         /* the head dim in the comment */
	one(100, 57, 64, 57);
	for (i = 0; i < 500; i++) {
		unsigned hd = rrange(1, 200);
		unsigned a = rrange(1, 300), b = rrange(1, 300);

		one(hd, a, b, rrange(0, a + 8));
	}
	{
		struct buf x;
		size_t whole;
		uint16_t *snap;

		buf_new(&x, wbytes(128, 64));
		pack_into(x.b, 64, 64, 0, 40);
		whole = x.half * 2 + 2 * GUARD * 2;
		snap = malloc(whole);
		memcpy(snap, x.mem, whole);
		if (!charsiu_fp16_regrow_vcols(NULL, 128, x.b, 64, 64, 40))
			bad("NULL", 64, 64, 128, 40, "accepted a NULL dst", 0);
		if (!charsiu_fp16_regrow_vcols(x.b, 128, NULL, 64, 64, 40))
			bad("NULL", 64, 64, 128, 40, "accepted a NULL src", 0);
		if (memcmp(snap, x.mem, whole))
			bad("NULL", 64, 64, 128, 40,
			    "a NULL refusal moved bytes", 0);
		free(snap);
		buf_free(&x);
	}
	printf("  phase 3  %ld refusal and edge shapes, each required to "
	       "leave both buffers byte identical\n", cases - phase);

	/* ---- phase 4: chains ---------------------------------------------- */
	phase = cases;
	for (i = 0; i < NHD; i++) {
		unsigned g;

		for (g = 0; g < 4; g++) {
			static const unsigned gaps[] = { 0, 2, 3, 7 };

			int md;

			for (md = 0; md < 3; md++) {
				chain(nasty_hd[i], ladder, 6, 864, gaps[g],
				      md);
				chain(nasty_hd[i], ladder, 6, 300, gaps[g],
				      md);
				chain(nasty_hd[i], ladder, 6, 33, gaps[g], md);
				chain(nasty_hd[i], ladder, 6, 65, gaps[g], md);
				chain(nasty_hd[i], ladder, 6, 1, gaps[g], md);
			}
			grow_only(nasty_hd[i], ladder, 6, 1, gaps[g]);
			grow_only(nasty_hd[i], ladder, 6, 31, gaps[g]);
			grow_only(nasty_hd[i], ladder, 6, 32, gaps[g]);
		}
	}
	for (i = 0; i < 4000; i++) {
		unsigned r[6], hd = pick_hd(160), n = rrange(2, 6), q;
		unsigned gap = (rnd() % 3) ? 0 : rrange(2, 9);

		r[0] = good_extent(1, rnd());
		for (q = 1; q < n; q++) {
			unsigned m = (r[q - 1] + 31u) / 32u + rrange(0, 3);

			r[q] = good_extent(m, rnd());
			if (r[q] < r[q - 1])
				r[q] = r[q - 1];
		}
		chain(hd, r, n, rrange(1, r[n - 1]), gap, (int)(rnd() % 3));
		grow_only(hd, r, n, rrange(1, r[0]), gap);
	}
	/* the ladders that are NOT multiples of 32, so k_eff and kv differ at
	 * every rung and a refusal in the middle makes the caller repack */
	for (i = 0; i < 2000; i++) {
		unsigned r[6], hd = pick_hd(128), n = rrange(2, 6), q;

		r[0] = rrange(1, 64);
		for (q = 1; q < n; q++)
			r[q] = r[q - 1] + rrange(0, 200);
		chain(hd, r, n, rrange(1, r[n - 1]), 0, (int)(rnd() % 3));
		grow_only(hd, r, n, rrange(1, r[0]), 0);
	}
	{
		static const unsigned deep[] = { 32, 64, 128, 256, 512, 1024,
						 2048, 4096 };
		static const unsigned dhd[] = { 64, 96, 100, 128, 256 };
		unsigned q;

		for (q = 0; q < sizeof(dhd) / sizeof(dhd[0]); q++) {
			chain(dhd[q], deep, 8, 4096, 0, 0);
			chain(dhd[q], deep, 8, 4096, 3, 2);
			grow_only(dhd[q], deep, 8, 1, 0);
			grow_only(dhd[q], deep, 8, 32, 0);
		}
	}
	printf("  phase 4  %ld ladders climbed in ONE buffer, with gaps, with "
	       "nothing appended, checked at every rung and at the end\n",
	       cases - phase);

	/* ---- phase 4b: a real context length ------------------------------ */
	phase = cases;
	{
		static const unsigned hh[] = { 1, 2, 17, 64, 100, 128, 256 };
		static const unsigned kk[] = { 32, 2048, 8192, 32768 };
		unsigned q, w2;

		for (q = 0; q < sizeof(hh) / sizeof(hh[0]); q++)
			for (w2 = 1; w2 < sizeof(kk) / sizeof(kk[0]); w2++) {
				huge(hh[q], kk[w2 - 1], kk[w2], 1);
				huge(hh[q], kk[w2 - 1], kk[w2], kk[w2 - 1]);
				huge(hh[q], kk[w2 - 1], kk[w2],
				     kk[w2 - 1] / 2 + 1);
				huge(hh[q], 32, kk[w2], 32);
			}
	}
	printf("  phase 4b %ld shapes at a real context length, up to 32768 "
	       "positions and 16 MB a surface\n", cases - phase);

	/* ---- phase 5: guard pages ----------------------------------------- */
	phase = cases;
	for (i = 0; i < NHD; i++) {
		unsigned hd = nasty_hd[i];

		paged(hd, 32, 64, 17, 1);
		paged(hd, 32, 64, 17, 0);
		paged(hd, 64, 864, 64, 1);
		paged(hd, 64, 864, 64, 0);
		paged(hd, 57, 64, 57, 1);
		paged(hd, 57, 64, 57, 0);
		paged(hd, 25, 32, 25, 1);
		paged(hd, 25, 32, 25, 0);
		paged(hd, 512, 512, 400, 1);
		paged(hd, 512, 512, 400, 0);
	}
	/* ⚠ AND RANDOMISED, because this is the ONLY arm that can see a read
	 * or a write one half past the surface: the guard halves live inside
	 * the same allocation and a read into them is legal to the allocator.
	 * The source here ends flush against a page with no permissions. */
	for (i = 0; i < 6000; i++) {
		unsigned hd = pick_hd(200);
		unsigned a = good_extent(rrange(1, 12), rnd());
		unsigned b = good_extent(rrange(1, 14), rnd());

		paged(hd, a, b, pick_live(a), (int)(rnd() & 1));
	}
	printf("  phase 5  %ld shapes run flush against a PROT_NONE page, at "
	       "each end, source and destination and in place\n",
	       cases - phase);

	/* ---- every mutant has to have been caught ------------------------- */
	{
		int m;

		for (m = 1; m < MUT_N; m++) {
			printf("  mutant %-28s applicable %6ld, reached the "
			       "reference anyway %6ld\n", mut_name[m],
			       mut_n[m], mut_same[m]);
			if (!mut_n[m] || mut_same[m] == mut_n[m]) {
				printf("  ⛔ CONTROL: \"%s\" was NEVER caught "
				       "-- this comparison cannot see that "
				       "class of bug\n", mut_name[m]);
				fails++;
			}
		}
		printf("  mutant %-28s applicable %6ld, disagreed %6ld "
		       "(must be 0)\n", mut_name[MUT_NONE], mut_n[MUT_NONE],
		       mut_n[MUT_NONE] - mut_same[MUT_NONE]);
	}

	printf("fp16_regrow_fuzz: %ld cases (%ld moved, %ld refused), %s\n",
	       cases, accepted, refused, fails ? "⛔ FAILED" : "all ok");
	if (fails > 40)
		printf("  (%d failures, only the first 40 printed)\n", fails);
	return fails ? 1 : 0;
}
