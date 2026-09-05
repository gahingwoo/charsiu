// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The complete register stream, and the coefficient buffer under it.
 *
 * PROVENANCE. Every constant in this file is ported from the Mesa Teflon RK3576
 * path in github.com/gahingwoo/linux-rk3576-npu, which is verified byte exact on
 * this silicon: MobileNet V1 runs end to end through it and single convolutions
 * are exact per output channel across the shape space. Those files carry
 * Tomeu Vizoso's MIT notice and this project's additions to them; MIT combines
 * into GPL-2.0-or-later, and the origin is recorded rather than the values being
 * quietly re-invented.
 *
 * The geometry, separately, is checked against the vendor's own LLM dispatches
 * with tools/cmp_vendor.py, which agrees register for register on five shape
 * classes.
 *
 * A matmul is that path with one column: inw = 1, oh = M, ow = 1, oc = N,
 * ic = K, and never depthwise.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu.h"
#include "envq.h"

#define CNA   0x0201u
#define CORE  0x0801u
#define DPU   0x1001u
#define RDMA  0x2001u
#define U28   0x0401u   /* the CBUF block the vendor writes at 0x2810 */

#define DIV_ROUND_UP(n, d)  (((n) + (d) - 1) / (d))
#define ALIGN_UP(n, a)      (DIV_ROUND_UP((n), (a)) * (a))

struct emitter {
	uint64_t *out;
	size_t max;
	size_t n;
};

/*
 * CHARSIU_OVERRIDE lets one register be changed from the environment without a
 * rebuild, so a single field can be swept from a board script.
 *
 * The format is a comma separated list of reg=value, both accepted in any base
 * strtoul takes, and the register is matched on its address alone rather than
 * on unit, because no two units in this stream share an address:
 *
 *   CHARSIU_OVERRIDE="0x1030=0x00800000,0x402c=63"
 *
 * The last matching entry wins, which makes it safe to append. This exists
 * because rounds 265 through 268 established two shortfalls that both point at
 * a size register, and answering which one needs a sweep of one field at a
 * time, which is the method that worked on 0x4050 in round 260.
 */
/* the int4 paired recipe, on when CHARSIU_W4_PAIRED is set and the weights are
 * int4. One place, so the emitter and every probe agree. */
/*
 * WHICH PART OF THE SHARED CBUF THIS STREAM USES.
 *
 * Round 362, with a control that could have failed and did: two concurrent jobs
 * on the two NPU cores corrupt each other three times in four when they carry
 * the SAME CBUF configuration, and are clean four times in four when they carry
 * DIFFERENT ones. Giving BOTH of them core 1's values is just as bad as giving
 * both core 0's, which is what rules out "core 1's numbers are simply kinder".
 *
 * So the CBUF is shared between the cores and each concurrent job needs its own
 * window. Note what this does NOT need: it does not matter which core gets
 * which window, only that two jobs in flight together do not collide -- so no
 * core selection in the uapi is required.
 *
 * Window 1's values are the vendor's own, read off task 17 of the capture
 * against task 16, the same op prepared for the other core. 0x1c00 is 7168 and
 * appears in three of them as what reads like a base offset.
 *
 * ⚠ 0x1040 is taken LITERALLY rather than derived. Its two halves both change,
 * 0x1000 to 0x2c00 and 0 to 0x1c00, and one capture is not enough to say what
 * either half means. The others are a clean substitution on charsiu's own value.
 */
int charsiu_cbuf_window(void)
{
	const char *e = envq("CHARSIU_CBUF_WINDOW");

	return e ? atoi(e) : 0;
}

/*
 * WINDOW 1'S START BANK, in banks of 0x400 CBUF entries; 7 is the vendor's
 * 0x1c00 and every window 1 value above is a function of it. Read off the
 * 2026-09-04 census of the vendor's Llama-3.2-1B .rkllm, where every op is
 * present twice and the two copies differ in EXACTLY six registers:
 *
 *   register   window 0     window 1     as a function of the start bank S
 *   0x1018     40000404     4000040b     low byte 4 + S
 *   0x1038     00000007     0000010e     end bank (7, then 14) | 0x100 for S > 0
 *   0x103c     xxxx0000     xxxx1c00     low half S * 0x400
 *   0x1040     10000000     2c001c00     (S*0x400 + 0x1000) << 16 | S*0x400
 *   0x2818     00000000     1c000000     S*0x400 << 16
 *   0x2820     00000000     00000038     S * 8
 *
 * so the CBUF is two mirror-image windows of 7 banks (0x1000 of data and
 * 0xc00 of weights each; the split form takes 0x1400 and 0x800) in 14 banks,
 * 0x3800 entries. CHARSIU_CBUF_W1_BANK=N moves window 1 to start at bank N:
 * 8 leaves a gap between the windows (a smaller weight ring), 6 makes them
 * overlap by a bank (a control that should fail worse). The wrong word of
 * the overlap fault sits in window 1's core; this asks whether it sits at
 * window 1's ADDRESS.
 */
static unsigned w1_bank(void)
{
	const char *e = envq("CHARSIU_CBUF_W1_BANK");
	int v = e ? atoi(e) : 7;

	return v < 1 || v > 13 ? 7u : (unsigned)v;
}

int charsiu_w4_paired(const struct charsiu_matmul *mm)
{
	return mm->wdtype == CHARSIU_INT4 && envq("CHARSIU_W4_PAIRED") != NULL;
}

static int override_for(unsigned reg, uint32_t *out)
{
	const char *p = envq("CHARSIU_OVERRIDE");
	int found = 0;

	while (p && *p) {
		unsigned long r, v;
		char *end;

		r = strtoul(p, &end, 0);
		if (end == p || *end != '=')
			break;
		p = end + 1;
		v = strtoul(p, &end, 0);
		if (end == p)
			break;
		p = end;
		if (r == reg) {
			*out = (uint32_t)v;
			found = 1;
		}
		while (*p == ',' || *p == ' ')
			p++;
	}
	return found;
}

static void emit(struct emitter *e, unsigned target, unsigned reg, uint32_t val)
{
	uint32_t ov;

	if (e->n >= e->max) {
		e->n = e->max + 1;
		return;
	}
	if (override_for(reg, &ov))
		val = ov;
	e->out[e->n++] = ((uint64_t)target << 48) | ((uint64_t)val << 16) | reg;
}

/* ---- requant --------------------------------------------------------------
 *
 * The DPU turns the accumulator into an output byte with one multiply and one
 * shift. Both come out of the float scale ratio's own bits, which is how Mesa
 * derives them and how the board was calibrated.
 */
struct requant {
	uint32_t scale;
	uint32_t shift;
	int32_t offset;
};

static struct requant requant_of(const struct charsiu_job *job)
{
	float conv_scale = (job->input_scale * job->weight_scale) / job->output_scale;
	union { float f; uint32_t u; } bits = { .f = conv_scale };
	struct requant r;
	unsigned shift = 127 + 31 - 32 - (bits.u >> 23) + 16;
	unsigned scale = (bits.u >> 9) & 0x7fff;

	/* Round to nearest on the bit below the field, then renormalise if the
	 * carry runs out of it: half the multiplier, one less shift. Writing
	 * 0x8000 into a 15 bit field would be silent. */
	if (bits.u & (1u << 8))
		scale++;
	if (scale > 0x7fff) {
		scale = 1u << 14;
		shift--;
	}
	if (scale < (1u << 14))
		scale |= 1u << 14;

	r.scale = scale;
	r.shift = shift - 1;
	/*
	 * THE OUTPUT STAGE, solved. Rounds 160 to 162 put three entries on it
	 * that move the accumulator, the bias and this register independently,
	 * and ONE model accounts for all 192 bytes with nothing off by a single
	 * count:
	 *
	 *     out = clamp(requant + offset, -128, 127), stored as int8
	 *
	 * Two things fall out of it, and both correct this file.
	 *
	 * ROUND 163 CORRECTED ROUND 163. Its first entry was written to be the
	 * milestone and instead it falsified the model that entry was based on:
	 * with the offset at 0, the negative half came back at 0 rather than
	 * negative, while the positive half was exact to the byte. The three
	 * entries the earlier model was fitted to all had an offset that made a
	 * floor at zero and a rail at -128 predict the same output, so they
	 * could not tell them apart. This one could. The floor is real.
	 *
	 * out = clamp(max(requant, 0) + offset, -128, 127), stored as int8
	 *
	 * 64 of 64 on all three of round 163's entries; without the floor, 32
	 * and 34 of 64 on the two that distinguish. So there IS a fused ReLU,
	 * the thing this file said twice there was not.
	 *
	 * It does not matter, and that is the useful part. Lift the accumulator
	 * by 128 in the requant domain before the floor and take the same 128
	 * back here, and the floor never reaches anything while the offset
	 * undoes the lift exactly:
	 *
	 *     max(rq + 128, 0) + (out_zp - 128) = rq + out_zp for rq >= -128
	 *
	 * which is the whole signed range. See the lift in charsiu_build_coefs.
	 * The rule that falls out, offset = out_zp - 0x80, is the one Mesa has
	 * used all along.
	 */
	{
		const char *o = envq("CHARSIU_OUT_OFF");

		r.offset = o ? (int32_t)strtol(o, NULL, 0)
			     : (int32_t)job->output_zero_point - 0x80;
	}
	return r;
}

/* ---- the coefficient buffer ------------------------------------------------
 *
 * Read in whole groups of 8 output channels, 64 bytes each: 8 int32 A, then 8
 * int16 B, then 8 int16 C. After the table comes one fp16 per output channel,
 * PADDED TO A WHOLE GROUP OF EIGHT, and the second operand word after that.
 *
 * The padding is not cosmetic. The table in front is always a multiple of 64,
 * so an unpadded scale table puts the second operand at a 16 byte boundary only
 * when the channel count is a multiple of 8, and every layer that missed came
 * back an EMPTY convolution. That cost rounds 136 to 138 to find.
 */
/* and back, for reading an fp16 output */
float charsiu_half_to_float(uint16_t h)
{
	union { float f; uint32_t u; } v;
	uint32_t sign = (uint32_t)(h & 0x8000) << 16;
	int32_t exp = (h >> 10) & 0x1f;
	uint32_t man = h & 0x3ff;

	if (!exp) {
		if (!man) { v.u = sign; return v.f; }
		/* subnormal: normalise it the slow, obvious way */
		exp = 1;
		while (!(man & 0x400)) { man <<= 1; exp--; }
		man &= 0x3ff;
	} else if (exp == 0x1f) {
		v.u = sign | 0x7f800000 | (man << 13);
		return v.f;
	}
	v.u = sign | ((uint32_t)(exp - 15 + 127) << 23) | (man << 13);
	return v.f;
}

/* the definition is charsiu_f2h in charsiu.h; this is the same bytes for
 * callers that want a symbol rather than an inline */
uint16_t charsiu_float_to_half(float f)
{
	return charsiu_f2h(f);
}

static size_t table_bytes(const struct charsiu_matmul *mm)
{
	return (size_t)DIV_ROUND_UP(mm->n, 8) * 64;
}

static size_t scale_table_bytes(const struct charsiu_matmul *mm)
{
	return (size_t)ALIGN_UP(mm->n, 8) * 2;
}

/*
 * ⚠⚠ THE ACCUMULATOR'S READ ORDER, SOLVED.
 *
 * With 0x40b8 following the row count every wanted value is in the buffer at
 * every shape and every m measured, so the arithmetic is right and this is all
 * that was left. locate() printed the map at m = 2 and again at m = 4, and one
 * expression reproduces both, all 128 positions each, with nothing left over.
 *
 * Channels go in super groups of 32. Inside a super group the rows pair up P at
 * a time, and inside a pair the four word runs alternate between the rows and
 * between the two sixteen channel halves:
 *
 *   P = m/2
 *   G = ni/32,  c = ni%32,  a = c/16,  t = c%16
 *   j     = (t/4)*8P  +  (mi%P)*8  +  a*4  +  t%4
 *   index = G*(m*32)  +  (mi/P)*(32P)  +  j
 *
 * At m = 2 that is P = 1: each row is its own block of 32 and the halves
 * interleave, channels 0..3, 16..19, 4..7, 20..23. At m = 4 it is P = 2: rows
 * pair into blocks of 64 and the two rows of a pair alternate every eight
 * words. Both are what the board printed.
 *
 * ⚠ m = 1 IS FLAT AND IS NOT THIS. P would be zero, and the expression does not
 * collapse to the identity at P = 1 either. Decode has read m = 1 flat for
 * hundreds of rounds and the sweep scores it 64 of 64 flat, so it is a separate
 * case rather than a limit of this one.
 *
 * ⚠ P = m/2 IS FITTED ON m = 2 AND m = 4. Those are the only two widths whose
 * map has been printed. m = 8 is scored by the sweep and has not been read.
 *
 * ⚠ AND THE SAME EXPRESSION SAYS HOW BIG AN OUTPUT SLOT HAS TO BE, which is
 * worth writing down here because the place that sizes one is nowhere near it.
 *
 * Its range is exactly n * m words. G tops out at (n - 1) / 32, j at 32P - 1
 * and mi / P at 1, so index_max = n * m - 1: the surface is n * m * 4 bytes
 * with nothing padded and nothing skipped. At m = 1 the flat case says the
 * same thing more plainly -- a decode slice occupies n * 4 bytes and not one
 * byte more. The batched path is already sized from this: g->bout_stride is
 * wide * m * 4, per tensor, and it is the same number.
 *
 * The DECODE path is not. npudev.c gives every output slot g->out_stride =
 * nmax * 4 = 32 KB whatever the slice is worth, so a 512 wide k projection
 * writes 2 KB into 32 KB, and the slack is not free: rocket's PREP_BO and
 * FINI_BO take a HANDLE and nothing else -- no offset, no length, and a
 * `reserved` word the kernel rejects unless it is zero -- so the cache
 * maintenance they do is over the whole allocation and not over what was
 * written. Counted off the gguf shapes at the shipped kslice = 1024, that is
 * 39.7 MB of prep and fini a token on Llama-3.2-1B where the slices write
 * 9.7 MB, and a per-entry stride would make it 18.2. Counted, not measured:
 * no host in this tree has an NPU to measure it on.
 *
 * ⚠ A SLOT BASE STILL HAS TO CLEAR 16 BYTES, which is the one thing a tighter
 * stride could get wrong. The vendor's own streams step 0x4018 by 16 for each
 * output position skipped -- see the note on 0x40b8, which is where that read
 * comes from -- so 16 is the granularity the output base has evidence for.
 * n * 4 clears it for any n that is a multiple of four, and a slice width is
 * a multiple of sixteen in every model in models/ -- npudev.c rounds a CPU
 * split DOWN to sixteen on purpose, for the int4 feature atom. A tighter
 * stride should still round up rather than trust that, because a tower is
 * free to hand this a width nobody has checked.
 */
/*
 * The running commentary switch. See charsiu.h: on for a one-shot run, which is
 * what a board log is, and off inside a conversation.
 */
static int diag_on = 1;

void charsiu_diag_quiet(int on)
{
	diag_on = !on;
}

int charsiu_diag(void)
{
	return diag_on;
}

/*
 * ⚠ THE a TERM IS SWEEPABLE, because the board says w4a16 needs a different
 * one and nothing says what.
 *
 * The in place scan on the real path: row 0 agrees on EXACTLY HALF its
 * channels and the first it misses is 16. a is (c % 32) / 16, so a = 0 is
 * exactly half the channels and 16 is the first of the other half. The int8
 * expression is right about w4a16 except for where the second sixteen channel
 * half goes, and `a * 4` is the only term that places it.
 *
 * CHARSIU_ACC_A takes a coefficient, or "swap" for the two halves exchanged.
 * A variant that is not a permutation will collide and score badly, which is
 * the honest outcome rather than a guard here.
 *
 * ⚠ READ ONCE PER PROCESS AND CACHED, which is safe only because the sweep
 * runs one variant per process. Sweeping it inside one process would need this
 * cleared and g->bmap_m invalidated -- the read order reaches the hardware path
 * through a TABLE built once per m, not through this function.
 */
static unsigned acc_a_coeff(int *swap)
{
	static int done, sw;
	static unsigned A = 4;

	if (!done) {
		const char *e = envq("CHARSIU_ACC_A");

		done = 1;
		if (e && !strcmp(e, "swap"))
			sw = 1;
		else if (e && !strcmp(e, "roleswap"))
			sw = 2;
		else if (e && !strcmp(e, "roleswap2"))
			sw = 3;
		else if (e)
			A = (unsigned)strtoul(e, NULL, 0);
	}
	*swap = sw;
	return A;
}

/*
 * ⚠⚠ THE READ ORDER DEPENDS ON THE FORMAT, NOT ONLY THE AXIS, and the board
 * said both halves of that.
 *
 *   int8, height axis   a * 4        exact to m = 80, the tower sweep
 *   int8, width axis    a * 4        its RAW surface is identical to the
 *                                    height one, mapped cell for cell
 *   w4a16, width axis   roleswap2    2048 of 2048 on every row at m = 2, 4,
 *                                    16 and 32, worst relative 5.1e-05
 *   w4a16, height axis  neither      row 1 is not written at all
 *
 * So `w4wide` is the one case that reads differently, and every other caller
 * gets exactly the expression it had. CHARSIU_ACC_A still overrides, which is
 * how the two readings were told apart in the first place.
 */
size_t charsiu_acc_index(unsigned mi, unsigned ni, unsigned m, int w4wide)
{
	unsigned P, G, c, a, t, j, A;
	int swap;

	if (m < 2)
		return ni;                      /* flat, and measured so */
	A = acc_a_coeff(&swap);
	if (!swap && w4wide)
		swap = 3;                       /* roleswap2, the solved one */
	P = m / 2;
	G = ni / 32u;
	c = ni % 32u;
	a = c / 16u;
	t = c % 16u;
	/*
	 * ⚠⚠ roleswap: a AND mi/P TRADE PLACES, and the board's own map is
	 * where it comes from rather than a guess at what might work.
	 *
	 * The in place scan says the default is right on exactly two quadrants
	 * of (row, half): (0, a=0) and (1, a=1) are correct and the two off
	 * diagonal ones are not, which is what "1024 of 2048 on each row" is.
	 * Then the swapped arm, which is wrong everywhere, printed WHERE its
	 * values went: row 0's a=1 channels at row 1's slot and row 1's a=0
	 * channels at row 0's, same channel both ways.
	 *
	 * Those two facts have one expression between them. Putting a where
	 * mi/P was and mi/P where a was agrees with the default on the two
	 * quadrants the board calls correct, differs on the two it calls wrong,
	 * and reproduces all 36 of the swapped arm's printed landings with none
	 * missed. It is a permutation at m = 2, 4, 8, 32 and 80.
	 *
	 * ⚠ EVERY ONE OF THOSE 36 IS AT m = 2, where P is 1 and (mi % P) * 8 is
	 * inert. Which of the row's two parts takes the 4 and which keeps the 8
	 * is therefore a choice at wider m, not a reading. The probe sweeps m
	 * to 32 and its per m row count is what would catch it.
	 */
	/*
	 * ⚠⚠ AND WHICH PART OF THE ROW TAKES WHICH SLOT, which m = 2 cannot
	 * see and the board has now said.
	 *
	 * Once a takes the 32P block the row has two slots left: one of stride
	 * 8 with P values and one of stride 4 with 2. At m = 2, P is 1, the
	 * stride 8 slot is a singleton and the two readings below are the SAME
	 * FUNCTION -- which is why roleswap scored 2048 of 2048 on both rows
	 * there and 226 rows of 452 at m = 4.
	 *
	 * 226 is 113 tensors times TWO ROWS. If roleswap2 is the truth then
	 * roleswap is right exactly where they coincide, which is rows 0 and
	 * m-1 and nothing else:
	 *
	 *   m      rows they share    predicts    board
	 *   2      0, 1               226 of 226  226 of 226
	 *   4      0, 3               226 of 452  226 of 452
	 *   16     0, 15              226 of 1808 226 of 1808
	 *   32     0, 31              226 of 3616 226 of 3616
	 *   8      0, 7               226 of 904  194 of 904   <-- the one miss
	 *
	 * ⚠ m = 8 IS NOT THIS AND HAS NEVER BEEN. Its worst relative error is
	 * four to six orders out in EVERY arm of every round -- 1.3e4, 3.2e4,
	 * 2.9e5, 9.7e3, 7.5e4, 1.7e5, 4.7e4 -- where its neighbours sit at 1e3.
	 * Something else is wrong at that one width and this does not explain
	 * it or claim to.
	 *
	 * ⚠⚠ AND THIS FUNCTION IS NOW EXCLUDED FROM IT, on the desktop, with a
	 * check that could have failed.
	 *
	 * Two facts, and between them there is no room for the read order to be
	 * the m = 8 fault:
	 *
	 *  1. IT IS A BIJECTION AT EVERY WIDTH AND EVERY WIDTH OF OUTPUT the
	 *     probe reaches. m of 2, 4, 8, 16, 32, 48, 64 and 80 crossed with n
	 *     of 512, 2048, 5376 and 8192: 32 shapes, every one covering
	 *     [0, m*n) exactly once, no collision, no hole, nothing out of
	 *     range, and the four-in-a-row property the gather relies on
	 *     unbroken at every j.
	 *
	 *  2. IT DOES NOT TAKE n. The only way a channel enters is G = ni/32
	 *     and (a, t) = the position inside a group of 32, so the map of an
	 *     8192 wide slice restricted to its first 2048 channels IS the map
	 *     of a 2048 wide one, group for group.
	 *
	 * The board says m = 8 is exact at n = 512 and n = 2048 and wrong at
	 * n = 8192. By (2) this function cannot tell those apart, so whatever
	 * is wrong is not in here. And by (1), at m = 8 the seven rows that ARE
	 * exact consume seven eighths of the slots, so row 0's values -- if the
	 * block wrote them into the surface at all -- can only be in the eighth
	 * this expression already reads.
	 */
	if (swap == 2 || swap == 3) {
		unsigned hi = swap == 3 ? mi / 2u : mi % P;
		unsigned lo = swap == 3 ? mi % 2u : mi / P;

		j = (t / 4u) * (8u * P) + hi * 8u + lo * 4u + (t % 4u);
		return (size_t)G * m * 32u + (size_t)a * (32u * P) + j;
	}
	if (swap)
		a = 1u - a;
	j = (t / 4u) * (8u * P) + (mi % P) * 8u + a * A + (t % 4u);
	return (size_t)G * m * 32u + (size_t)(mi / P) * (32u * P) + j;
}

size_t charsiu_coef_bytes(const struct charsiu_matmul *mm)
{
	/*
	 * THE BUFFER IS FAR BIGGER THAN WHAT IS WRITTEN INTO IT, and that is not
	 * slack, it is the difference between computing and hanging.
	 *
	 * The DPU_RDMA reads a per channel float surface from 0x5024 onwards for
	 * all output channels, and it reads it whether or not anything put data
	 * there. Under allocate and the read runs off the end of the buffer
	 * object, the IOMMU faults, the RDMA stalls, and the job times out with
	 * every register correct. The driver work hit exactly this and wrote it
	 * down: a coefficient buffer of 1280 bytes against a read of 20800 was
	 * the whole of one wall.
	 *
	 * The size is the record table plus a float region bounded by the weight
	 * count, floored at 8192 floats because a small matmul's own weight
	 * count under-allocates it, plus a page of margin. Round 147 gave this
	 * buffer 4.7 KB where the hardware reads 33 KB, and the symptom was a
	 * job that timed out with a register stream identical to Mesa's.
	 */
	/*
	 * ⚠ THE k*n BOUND IS A GUESS, AND IT DOES NOT SCALE. It makes the
	 * coefficient buffer FOUR TIMES the weight buffer -- 8.4 MB for a
	 * K=2048 N=1024 slice, 67 MB at N=8192 -- so a whole 1B model's
	 * projections would want 3.9 GB of it on top of 973 MB of weights.
	 * That is fine for one probe and impossible for a runtime.
	 *
	 * The reads this was sized against are tens of kilobytes, not megabytes:
	 * round 147's wall was 4.7 KB allocated against 33 KB read, and the
	 * driver's was 1280 bytes against 20800. Nothing has ever measured the
	 * read growing with k*n; the bound was chosen because under allocating
	 * hangs the block and nobody wanted to find the edge.
	 *
	 * CHARSIU_COEF_ELEMS overrides it so a board round can find that edge
	 * on purpose. The default is unchanged until one does.
	 */
	/*
	 * ⚠⚠ THE DEFAULT IS THE VALUE WITH EVIDENCE, AND IT USED TO BE THE
	 * GUESS. This read `mm->k * mm->n` when the variable was unset, and
	 * the paragraph above already says that bound is a guess that does not
	 * scale. What it did not say is who was actually exposed to it.
	 *
	 * Only scripts/charsiu-runner and scripts/charsiu-serve set the
	 * variable. Every board test sets it too. So the k*n bound was never
	 * measured by anything -- it was what a bare ./charsiu_run,
	 * charsiu_vision, charsiu_whisper or charsiu_clip got, and it asks for
	 * 2.39 GB of coefficient buffers on Qwen3-0.6B and 14.92 GB on
	 * Phi-3.5. The allocation fails, whine() fires, and the tensor stays on
	 * the CPU: quiet, slow, and exactly the symptom recorded at
	 * npudev.c's "the head worth 40% of a gemma token was on the CPU".
	 *
	 * 65536 is what every board round this project has run has used, on
	 * every model, without one coefficient buffer coming up short. That is
	 * evidence and k*n is not, so 65536 is the default and k*n is what
	 * CHARSIU_COEF_ELEMS=0 asks for when a round wants to find the edge on
	 * purpose.
	 */
	const char *e = envq("CHARSIU_COEF_ELEMS");
	size_t want = e ? strtoul(e, NULL, 0) : 65536;
	size_t elems = want ? want : (size_t)mm->k * mm->n;

	if (elems < 8192)
		elems = 8192;
	return table_bytes(mm) + elems * sizeof(float) + 0x100;
}

void charsiu_build_coefs(const struct charsiu_job *job, const int32_t *bias,
			 const int32_t *weight_sums, uint8_t *dst)
{
	const struct charsiu_matmul *mm = &job->mm;
	/* ⚠ once, not once an output channel. The same shape of mistake cost
	 * round 354 two minutes a run in the quantiser. */
	const int16_t coef_c = (int16_t)(envq("CHARSIU_COEF_C")
					 ? atoi(envq("CHARSIU_COEF_C")) : 16);
	size_t tb = table_bytes(mm), sb = scale_table_bytes(mm);
	uint16_t *scales;
	unsigned oc;

	memset(dst, 0, charsiu_coef_bytes(mm));

	for (oc = 0; oc < mm->n; oc++) {
		unsigned g = oc / 8, i = oc % 8;
		int32_t *a = (int32_t *)(dst + g * 64 + i * 4);
		int16_t *b = (int16_t *)(dst + g * 64 + 32 + i * 2);
		int16_t *c = (int16_t *)(dst + g * 64 + 48 + i * 2);

		/*
		 * A is the bias with the input zero point folded in. The MAC
		 * computes sum((in - 0x80)(w - wt_zp)); the true numerator is
		 * sum((in - in_zp)(w - wt_zp)), and the difference is a per
		 * channel constant, (in_zp - 0x80) times the weight sum.
		 */
		/*
		 * A carries the bias, the input zero point correction, AND the
		 * OUTPUT zero point.
		 *
		 * The last of those is now derived rather than guessed. Rounds
		 * 160 and 161 measured the output stage end to end, on two
		 * probes that between them move the accumulator and the bias
		 * independently, and it is
		 *
		 *     out = (clamp(mult * (acc + A), 0, 255) - offset) mod 256
		 *
		 * with the clamp BEFORE the offset. The floor at zero is that
		 * clamp's lower end, and it is why a projection loses its
		 * negative half: nothing about the offset can put back a number
		 * that was clamped away before the offset was applied.
		 *
		 * THE LIFT, which is what makes a signed result survive the
		 * hardware's floor at zero.
		 *
		 * The output stage is
		 *
		 *     out = clamp(max(requant, 0) + offset, -128, 127)
		 *
		 * so a negative requant is lost before the offset can do
		 * anything about it. Adding 128/mult to A raises the whole
		 * surface by exactly 128 in the requant domain, which puts every
		 * value a signed byte can hold above the floor, and the offset
		 * takes the same 128 back. Nothing is approximated: the lift is
		 * added to an int32 accumulator and removed in the output
		 * domain, so the only cost is the one rounding the requant
		 * already had.
		 *
		 * CHARSIU_NO_LIFT is the control. Without it the negative half
		 * of any two sided result comes back flat at the output zero
		 * point, which is what rounds 152 to 163 were looking at and
		 * calling a blocker.
		 */
		float mult = job->input_scale * job->weight_scale /
			     job->output_scale;

		*a = bias[oc] - (job->input_zero_point - 0x80) * weight_sums[oc]
		   + (envq("CHARSIU_NO_LIFT")
		      ? 0 : (int32_t)(128.0f / mult + 0.5f));
		/*
		 * B carries the weight zero point correction, in the same
		 * biased domain the weight is stored in. For int8 the weight
		 * goes in as w - 0x80, so the correction is 0x80 - wt_zp.
		 *
		 * A NIBBLE IS NOT BIASED, so int4's is 0 and not 0x80. Round
		 * 167 measured what getting this wrong looks like: with every
		 * weight nibble dead and the input held 32 above its zero
		 * point, the output came back 105 rather than the 0 the output
		 * stage says, and 0x80 * 2048 * mult is 210, which is twice it.
		 * The input was contributing through a correction that should
		 * not have been there.
		 */
		*b = (int16_t)((mm->wdtype == CHARSIU_INT4 ? 0 : 0x80)
			       - job->weight_zero_point);
		/*
		 * 16 is Q4 for a relative scale of 1, which is the int8
		 * convention. Whether it means the same thing on the w4a16 path
		 * is unknown, and round 181 came back with a slope 20.3 times
		 * what it should be, so CHARSIU_COEF_C makes it a knob: if the
		 * slope moves with this, C is the gain.
		 */
		*c = coef_c;
	}
	/* Carry the last real record across the rest of its group, so the group
	 * holds no zero requant multiplier beside live channels. */
	for (oc = mm->n; oc < ALIGN_UP(mm->n, 8); oc++) {
		unsigned g = oc / 8, i = oc % 8, lg = (mm->n - 1) / 8,
			 li = (mm->n - 1) % 8;

		*(int32_t *)(dst + g * 64 + i * 4) =
			*(int32_t *)(dst + lg * 64 + li * 4);
		*(int16_t *)(dst + g * 64 + 32 + i * 2) =
			*(int16_t *)(dst + lg * 64 + 32 + li * 2);
		*(int16_t *)(dst + g * 64 + 48 + i * 2) =
			*(int16_t *)(dst + lg * 64 + 48 + li * 2);
	}

	/* The scale table and the operand live between the record table and the
	 * float region, which the size above covers with room to spare. */
	scales = (uint16_t *)(dst + tb);
	for (oc = 0; oc < sb / 2; oc++)
		scales[oc] = charsiu_float_to_half(job->weight_scale);

	/* The second operand. 0x1004 is fp16 0.00049, a value this hardware's
	 * own configuration accepts; the vendor's 0x0E0E does not, under ours. */
	*(uint32_t *)(dst + tb + sb) = 0x1004;
}

/* ---- the stream ----------------------------------------------------------- */

/*
 * REPLAY THE VENDOR'S OWN STREAM, byte for byte, with only the three addresses
 * changed.
 *
 * Round 342. The capture from the vendor's LLM runtime (vendor-capture/llmcap in
 * the driver repository) is 123 registers for one int4 projection, K = 2048,
 * N = 1024, weights 1 MiB. Diffed against what this file emits for the same
 * shape, 15 registers differ and three whole groups do not exist on one side or
 * the other: the vendor writes five registers at 0x2810 that nothing here has
 * ever written, and does NOT write the DPU_RDMA block or the 0x1d enable
 * trailer that this file ends with.
 *
 * Every previous int4 round changed one of those at a time and measured the same
 * bit pattern product. This changes all of them at once by not deciding any of
 * them: the stream is the vendor's, and the only thing substituted is where the
 * input, the weights and the output live.
 *
 *   0x1088  CNA   input
 *   0x1110  CNA   weights
 *   0x4018  DPU   output
 *
 * The shape is then whatever the captured stream says, NOT what mm asks for, so
 * the caller has to size its buffers to the stream. charsiu_vendor_stream_shape()
 * reads it back out for exactly that.
 */
static const char *vendor_stream_path(void)
{
	return envq("CHARSIU_VENDOR_STREAM");
}

static size_t vendor_stream_load(uint64_t *out, size_t max)
{
	const char *path = vendor_stream_path();
	FILE *f;
	size_t n;

	if (!path)
		return 0;
	f = fopen(path, "rb");
	if (!f) {
		/*
		 * LOUD, because a silent fall back to the emitter's own stream
		 * would make the round read as a vendor replay while running
		 * something else, and a check that cannot fail has already cost
		 * this project a round.
		 */
		fprintf(stderr, "CHARSIU_VENDOR_STREAM: cannot open %s\n", path);
		exit(2);
	}
	n = fread(out, sizeof(*out), max, f);
	fclose(f);
	if (!n) {
		fprintf(stderr, "CHARSIU_VENDOR_STREAM: %s is empty\n", path);
		exit(2);
	}
	return n;
}

/*
 * The geometry the captured stream describes, so the probe can size its buffers
 * from the stream rather than from its own idea of the shape. Returns 0 if there
 * is no stream to read.
 */
int charsiu_vendor_stream_shape(unsigned *k, unsigned *n, size_t *wbytes)
{
	uint64_t buf[512];
	size_t got = vendor_stream_load(buf, sizeof(buf) / sizeof(buf[0]));
	uint32_t v101c = 0, v1024 = 0, v1028 = 0;
	size_t i;

	if (!got)
		return 0;
	for (i = 0; i < got; i++) {
		unsigned reg = (unsigned)(buf[i] & 0xffff);
		uint32_t val = (uint32_t)((buf[i] >> 16) & 0xffffffff);

		if (reg == 0x101c) v101c = val;
		else if (reg == 0x1024) v1024 = val;
		else if (reg == 0x1028) v1028 = val;
	}
	if (k) *k = (v1028 & 0xffff) + 1;
	if (n) *n = (v1024 & 0xffff) + 1;
	if (wbytes) *wbytes = v101c;
	return 1;
}

/* the three addresses the replay substitutes, and nothing else */
static void patch_addr(const struct charsiu_job *job, uint64_t *e)
{
	unsigned reg = (unsigned)(*e & 0xffff);
	uint32_t val;

	switch (reg) {
	case 0x1088: val = job->input_addr;  break;
	case 0x1110: val = job->weight_addr; break;
	case 0x4018: val = job->output_addr; break;
	default: return;
	}
	*e = (*e & ~0x0000ffffffff0000ULL) | ((uint64_t)val << 16);
}

/*
 * MERGE: this tree's stream, carrying the vendor's VALUES.
 *
 * Round 343 replayed the vendor's 123 registers verbatim and the job timed out
 * at every byte, 0 live words, four arms out of four. The reason is almost
 * certainly that nothing enabled the units: mainline rocket's drm_rocket_task
 * has only regcmd and regcmd_count, so there is no enable_mask for the vendor's
 * 0xd to go in, and the vendor stream carries no 0x1d trailer of its own.
 *
 * So come at it from the side that is known to run. Emit charsiu's stream,
 * overwrite every register the vendor also writes with the vendor's value, and
 * insert the registers only the vendor writes -- the five at 0x2810 -- BEFORE
 * the enable trailer, because they are configuration and the trailer is the go.
 * What is left of charsiu's own is exactly the two groups the vendor does not
 * write at all: the DPU_RDMA block and the trailer itself.
 */
static size_t merge_vendor_values(uint64_t *out, size_t n, size_t max)
{
	const char *path = envq("CHARSIU_VENDOR_MERGE");
	uint64_t v[512];
	size_t got, i, j, at, added = 0, overwrote = 0;
	FILE *f;

	if (!path)
		return n;
	f = fopen(path, "rb");
	if (!f) {
		fprintf(stderr, "CHARSIU_VENDOR_MERGE: cannot open %s\n", path);
		exit(2);
	}
	got = fread(v, sizeof(*v), sizeof(v) / sizeof(v[0]), f);
	fclose(f);
	if (!got) {
		fprintf(stderr, "CHARSIU_VENDOR_MERGE: %s is empty\n", path);
		exit(2);
	}
	/* where the enable trailer starts, so the inserts land before it */
	at = n;
	for (i = 0; i < n; i++)
		if ((unsigned)(out[i] & 0xffff) == 0x1008) {
			at = i;
			break;
		}
	/*
	 * CHARSIU_MERGE_UNITS narrows the merge to one register range, so the
	 * twenty registers that turn the multiply from a bit pattern product
	 * into a real weighted sum can be bisected: "1" is CNA 0x1xxx, "2" the
	 * five at 0x2810, "3" CORE 0x3xxx, "4" DPU 0x4xxx. Default is all.
	 */
	{
		const char *only = envq("CHARSIU_MERGE_UNITS");

		if (only)
			printf("MERGE: units %s only\n", only);
	}
	for (i = 0; i < got; i++) {
		unsigned reg = (unsigned)(v[i] & 0xffff);
		const char *only = envq("CHARSIU_MERGE_UNITS");
		char unit = (char)('0' + ((reg >> 12) & 0xf));
		int found = 0;

		/* the addresses stay OURS; only values are taken */
		if (reg == 0x1088 || reg == 0x1110 || reg == 0x4018)
			continue;
		if (only && !strchr(only, unit))
			continue;
		for (j = 0; j < n; j++) {
			if ((unsigned)(out[j] & 0xffff) != reg)
				continue;
			if (out[j] != v[i])
				overwrote++;
			out[j] = v[i];
			found = 1;
			break;
		}
		if (found || n >= max)
			continue;
		memmove(&out[at + 1], &out[at], (n - at) * sizeof(*out));
		out[at++] = v[i];
		n++;
		added++;
	}
	printf("MERGE: %zu vendor values applied, %zu registers inserted, "
	       "%zu entries out\n", overwrote, added, n);
	return n;
}

static size_t emit_vendor_stream(const struct charsiu_job *job, uint64_t *out,
				 size_t max)
{
	size_t got = vendor_stream_load(out, max);
	size_t i;

	if (!got)
		return 0;
	printf("REGISTER STREAM: the vendor's own, %zu entries from %s\n",
	       got, vendor_stream_path());
	for (i = 0; i < got; i++)
		patch_addr(job, &out[i]);
	/*
	 * CHARSIU_VENDOR_TRAILER appends the per unit op enable this tree ends
	 * its own stream with. The vendor does not write it because their driver
	 * puts 0xd in the task's enable_mask, and mainline rocket has no such
	 * field -- so a verbatim replay has nothing that turns the units on, and
	 * round 343 timed out on all four arms with zero live words.
	 */
	if (envq("CHARSIU_VENDOR_TRAILER") && got + 3 <= max) {
		out[got++] = ((uint64_t)CNA  << 48) | ((uint64_t)0x1du << 16) | 0x1008;
		out[got++] = ((uint64_t)CORE << 48) | ((uint64_t)0x1du << 16) | 0x3008;
		out[got++] = ((uint64_t)DPU  << 48) | ((uint64_t)0x1du << 16) | 0x4008;
		printf("  + the 0x1d enable trailer, three units\n");
	}
	return got;
}

/*
 * WHICH AXIS CARRIES M, and it is three states rather than two now.
 *
 *   CHARSIU_M_AXIS=w   the width, whatever the format
 *   CHARSIU_M_AXIS=h   the height, whatever the format -- which is how the
 *                      control arm of a board round reaches the arrangement
 *                      that is known wrong
 *   unset              the format decides: w4a16 on the width, int8 on the
 *                      height, because that is what each of them is right on
 *
 * The board, on 113 tensors of Llama-3.2-1B: w4a16 on the width is exact at
 * m = 2, 4, 16, 32, 48, 64 and 80, worst relative 5.10e-05 at every one, and
 * int8 on the height is exact to m = 80 in the tower sweep. Neither is right
 * on the other's arrangement.
 */
static int charsiu_m_axis(void)
{
	const char *e = envq("CHARSIU_M_AXIS");

	if (e && (*e == 'h' || *e == 'H'))
		return 0;
	if (e && (*e == 'w' || *e == 'W'))
		return 1;
	return -1;                              /* unset: ask the format */
}

/* the same question from outside job.c, for a given weight format: the read
 * order has to agree with the stream about which axis carries M, and npudev
 * builds that table */
int charsiu_m_axis_wide_for(int w4)
{
	int a = charsiu_m_axis();

	return a < 0 ? w4 : a;
}

size_t charsiu_emit_job(const struct charsiu_job *job, uint64_t *out, size_t max)
{
	size_t vend = emit_vendor_stream(job, out, max);

	const struct charsiu_matmul *mm = &job->mm;
	struct emitter e = { out, max, 0 };
	struct requant rq = requant_of(job);
	unsigned n_pad = ALIGN_UP(mm->n, 2);
	/*
	 * ⚠⚠ WHICH AXIS THE M ROWS GO ON, and the board says the current one
	 * is not producing rows at all.
	 *
	 * This tree puts M on the HEIGHT: one column, M rows. Round 384 read
	 * the whole output buffer back at m=2 and fitted the address function
	 * to it, 35 of 35 unique cells with no misses:
	 *
	 *     flat = (v/4)*8 + (u%2)*4 + (v%4) + (u/2)*40,  c = 16u + v
	 *
	 * The (u%2)*4 term is the slot a [n/4][m][4] surface reserves for ROW
	 * 1, and what lands in it is CHANNEL c + 16 of row 0. Sixteen is the
	 * int8 feature atom. So the hardware is spending the second position
	 * on the next atom of OUTPUT CHANNELS, and row 1 is absent everywhere
	 * -- eight sampled channels of it, all missing, against a reference
	 * with 121 distinct values in 128.
	 *
	 * The register that says so is 0x4020, the DPU's output width, which
	 * this tree writes as ow - 1 = 0 at every M. The DPU is told its
	 * output is one position wide, so one position is what it writes.
	 *
	 * CHARSIU_M_AXIS=w moves M to the width, which is Mesa's own generic
	 * encoder with inw = ow = M and the heights at 1. tools/mesa_mdiff.py
	 * --axis w prints the eleven words that changes and they are exactly
	 * the ones that carry an axis. M = 1 is IDENTICAL either way, so decode
	 * cannot move: the default stays the height until a board round says
	 * otherwise.
	 */
	unsigned wide = mm->m > 1 &&
			charsiu_m_axis_wide_for(mm->wdtype == CHARSIU_INT4);
	unsigned inw = wide ? mm->m : 1;
	unsigned irows = wide ? 1 : mm->m;
	unsigned ow = inw;
	struct charsiu_matmul surfmm = *mm;
	unsigned surf;
	unsigned rows = irows;
	unsigned lines = (wide ? ow : rows) - 1;   /* the DPU's line count */

	/*
	 * ⚠ surf IS PER SLICE OF THE INPUT SURFACE, so it counts inw columns.
	 * charsiu_entries_per_row() hard codes one column, which is right for
	 * the height axis and undercounts by exactly inw for the width one.
	 */
	surf = charsiu_entries_per_row(&surfmm) * inw;
	/*
	 * ⚠ THE SPLIT WINDOW, AND WHY IT IS SCOPED TO THE WIDTH AXIS.
	 *
	 * surf * rows is the whole input surface either way round: inw * M on
	 * the width axis and 1 * M on the height. Above 4096 of them the
	 * vendor's stream carries the split CBUF pair.
	 *
	 * Read symmetrically across its whole file rather than off the int4
	 * streams it was noticed in, "more than 4096 implies split" holds on
	 * all 8692 streams with no exception -- but the CONVERSE fails: 240 of
	 * its 4940 fp16 streams split below the threshold too. So it is a
	 * sufficient condition and not a definition, and there is no evidence
	 * at all for int8, whose 40 streams here never reach 4096.
	 *
	 * int8 above 4096 is exactly the batched path that already works on
	 * this board -- k = 8192 at m = 64 is 8192 atoms and it has been right
	 * since the 2.94x round -- so a rule read off int4 must not reach it.
	 * Scoped to `wide`, every stream that runs today is bit identical to
	 * before this line existed.
	 */
	unsigned split = wide && surf * rows > 4096;
	size_t wbytes = charsiu_weight_bytes(mm);
	int w4a16 = 0, w4_dpu = 0;
	unsigned wide8 = 0;
	unsigned rdma_mask;
	unsigned r;

	if (vend)
		return vend;

	/* The S_POINTER each unit latches its geometry against. Every unit gets
	 * it, not just the CNA: mesa writes all four and the vendor's streams
	 * carry 0x0e in each. */
	emit(&e, CNA, 0x1004, 0x0000000e);
	emit(&e, CORE, 0x3004, 0x0000000e);
	{
		unsigned S = w1_bank(), E = S + 7 > 14 ? 14 : S + 7;

		emit(&e, CNA, 0x1038,
		     job->cbuf_window == 1 ? (0x100u | E) : 0x00000007u);
	}
	emit(&e, DPU, 0x4004, 0x0000000e);
	emit(&e, RDMA, 0x5004, 0x0000000e);
	/*
	 * CONV_CON1. Zero for a plain int8 convolution, and the fp16 processing
	 * precision when the activations are 16 bit.
	 *
	 * This used to carry the vendor's own value, 0x00600120 on its int4
	 * projections and 0x20200120 on its fp16 attention, copied because it
	 * was not decoded. On an int8 job Mesa's proven path writes ZERO here,
	 * and round 144's stream diff caught it: charsiu was asking for a
	 * precision mode its operands were not in. Copying a constant from a
	 * different regime is not the same as not guessing.
	 */
	/*
	 * THE PRECISION REGISTER, read out of the vendor's own file rather than
	 * guessed, and this line was wrong for int4 until now.
	 *
	 * Diffing the .rkllm's int4 streams against its int8 ones
	 * (vendor-capture/int4_regs.py in the driver repository):
	 *
	 *   int8 weights                    0x00000000
	 *   int4 weights                    0x00600120
	 *   int4 weights, 16 bit activations 0x20600120
	 *   fp16 weights, 16 bit activations 0x20200120
	 *
	 * The int8 value is 0 and is confirmed twice, by the .rkllm's own int8
	 * LM head and by a vendor .rknn compiled at charsiu's exact shape.
	 *
	 * This wrote 0 for int4 as well, so an int4 job asked the hardware for
	 * an int8 weight fetch and would have read a layout out of nonsense.
	 * regcmd.c, the geometry only emitter, had the right constant and this
	 * one did not, which is the first time having two emitters has actually
	 * cost anything.
	 */
	emit(&e, CNA, 0x100c,
	     (mm->wdtype == CHARSIU_INT4 ? 0x00600120u :
	      mm->wdtype == CHARSIU_FP16 ? 0x00200120u : 0x00000000u) |
	     (charsiu_effective_adtype(mm) == CHARSIU_FP16 ? 0x20000000u : 0u));
	emit(&e, CNA, 0x1010, 0x00000fff);
	emit(&e, CNA, 0x1014, (1u << 3) | 1u);
	/*
	 * ⚠ THE SPLIT CBUF PAIR IS A FUNCTION OF surf * M, and this wrote the
	 * unsplit one at every M.
	 *
	 * The rule is exact on the vendor's own file: of its 3328 int4 streams,
	 * every one with surf * M > 4096 carries the split pair and every one
	 * at or below carries the unsplit, 0 disagreements. Both of its cbuf
	 * window variants take it -- 0x0404 becomes 0x0505 and 0x040b becomes
	 * 0x050c -- so it adds 0x0101 rather than replacing the word.
	 *
	 * It is a no-op below the threshold, which is every shape this tree has
	 * ever run: at M = 1 an ic of 131072 would be needed to reach it.
	 */
	emit(&e, CNA, 0x1018,
	     (job->cbuf_window == 1 ? 0x40000404u + w1_bank() : 0x40000404u) +
	     (split ? 0x0101u : 0u));
	emit(&e, CNA, 0x101c, (uint32_t)wbytes);
	emit(&e, CNA, 0x1020, (uint32_t)(wbytes / n_pad));
	emit(&e, CNA, 0x1024, n_pad - 1);
	emit(&e, CNA, 0x1028, ((surf * rows) << 16) | (charsiu_k_eff(mm) - 1));
	emit(&e, CNA, 0x102c, ((inw - 1) << 16) | (rows - 1));
	/*
	 * Bytes per kernel, DOUBLED for int8 and not for int4.
	 *
	 * The vendor's int4 projections carry it undoubled, which is where the
	 * "never doubled" reading came from, and Mesa's int8 stream for the very
	 * same shape carries twice the kernel size: 0x80 against our 0x40 at
	 * K=64. Both are right for their own precision.
	 */
	/*
	 * ⚠ AND THE DOUBLING MUST NOT REACH FP16, whose bytes are already two.
	 * The vendor writes (ic * 2) << 16 here in all 4940 of its fp16 streams,
	 * exactly, and wbytes / n_pad IS ic * 2 for fp16 -- so doubling it again
	 * asks for ic * 4. The comment above is about int8, where the kernel
	 * really is carried at twice its byte size.
	 */
	emit(&e, CNA, 0x1030,
	     ((uint32_t)(wbytes / n_pad
			 * (mm->wdtype == CHARSIU_INT4 ||
			    mm->wdtype == CHARSIU_FP16 ? 1u : 2u))
	      << 16) | (ow - 1));
	emit(&e, CNA, 0x1034, ow * rows - 1);
	/* Mesa writes 0x1038 a SECOND time here, after the geometry and before
	 * the surface stride, and that is the only ordering difference the
	 * stream diff had left. Matched rather than reasoned about: the value is
	 * the same both times, so the position is the only thing it can be
	 * carrying. */
	{
		unsigned S = w1_bank(), E = S + 7 > 14 ? 14 : S + 7;
		unsigned D = job->cbuf_window == 1 ? S * 0x400u : 0u;

		emit(&e, CNA, 0x1038,
		     job->cbuf_window == 1 ? (0x100u | E) : 0x00000007u);
		emit(&e, CNA, 0x103c, (surf << 16) | D);
		emit(&e, CNA, 0x1040,
		     (((D + 0x1000u) << 16) | D) + (split ? 0x04000000u : 0u));
	}
	emit(&e, CNA, 0x1044, (inw << 16) | surf);
	emit(&e, CNA, 0x1048, 0x0000000b);
	emit(&e, CNA, 0x104c, 0x00010001);
	emit(&e, CNA, 0x1050, 0x00010001);
	for (r = 0x1054; r <= 0x1074; r += 4)
		emit(&e, CNA, r, 0x00000000);
	emit(&e, CNA, 0x1078, ((inw - 1) << 16) | (rows - 1));
	emit(&e, CNA, 0x107c, charsiu_k_eff(mm) - 1);
	emit(&e, CNA, 0x1080, 0x00000000);      /* a matmul has no padding */
	emit(&e, CNA, 0x1084, 0x00000000);
	emit(&e, CNA, 0x1088, job->input_addr);
	emit(&e, CNA, 0x108c, 0x000f000f);
	/*
	 * ⚠⚠ DO NOT "FIX" THESE FROM THE VENDOR'S .rkllm. Round 380 did, and
	 * the board said no.
	 *
	 * The reasoning was: the vendor dispatches thousands of ops at M = 2
	 * to 128, tools/rkllm_mdiff.py shows which registers track M across
	 * them, and this tree disagrees on three -- 0x1094 (vendor holds it at
	 * 1), 0x1098 (vendor writes M exactly) and 0x118c (vendor holds it at
	 * 0). Changing them made no difference at all: m = 2, 4, 8 and 32 were
	 * as wrong after as before, at both K=256 N=64 and K=2048 N=1024.
	 *
	 * ⚠ WHAT THE INFERENCE MISSED. Every vendor stream at M > 1 is fp16,
	 * against the KV cache; its int4 and int8 weight matmuls are M = 1
	 * without exception, all 3368 of them. Those fp16 ops were never
	 * identified -- ic=1312 matches no dimension of the model -- so three
	 * registers were changed on the strength of an op nobody had named.
	 *
	 * 🏁 THEY HAVE A NAME NOW, 2026-09-05: THEY ARE ATTENTION. 2908 of the
	 * 4940 fp16 dispatches carry oc = 64, which is Llama-3.2-1B's head_dim,
	 * and their ic walks in steps of 32 with M chosen so the input surface
	 * lands just under 4096 every time (ic 2688 M 48, ic 2848 M 46, ic 4032
	 * M 32). ic = 1312 matches no dimension of the model because it is not
	 * one: it is a CONTEXT LENGTH rounded up to the feature atom, 41 * 32.
	 * The other oc values -- 32, 96, 128, 160, 192, 224, 256, each sixteen
	 * times, which is the layer count -- are the q.K^T half, where oc is the
	 * growing context.
	 *
	 * That makes the round's conclusion firmer rather than weaker, and it
	 * closes the question rather than leaving it open. Counted across the
	 * whole file, 0x1094 and 0x118c are CONSTANTS PER REGIME: fp16 holds
	 * them at 1 and 0 in all 4940, int4 carries 0x60 or 0x80 and 0x4f004f
	 * or 0x1f001f, and DPU 0x401c and 0x4020 move with them.
	 *
	 * ⚠ AND THOSE FOUR ARE THE WINDOW, NOT THE WEIGHTS. The lines below
	 * emit them as inw * rows, ow * rows, ((inw - 1) << 16) | (inh - 1) and
	 * ow - 1, so fp16 holding them at 1, 1, 0, 0 says that regime describes
	 * its window as 1 x 1 and carries the count elsewhere -- 0x1098, which
	 * is the register this note already says the vendor writes M into. An
	 * earlier reading of mine called them a weight GROUP count collapsing
	 * because fp16 carries its own exponent; that was a story fitted to four
	 * numbers without looking at what writes them, and this emitter refutes
	 * it. Round 380 took fp16's window values and put them on an int4 op:
	 * the same mistake 0x100c above describes, a constant carried across
	 * regimes.
	 *
	 * ⚠ AND THE VALUES BELOW ARE NOT A GUESS. They are Mesa's generic
	 * RK3576 encoder, rkt_regcmd.c, with inw = 1 and full_inh = M:
	 *
	 *     R_CNA(0x1090, inw * 4);
	 *     R_CNA(0x1094, inw * full_inh);
	 *     R_CNA(0x1098, (inw * stg_irows + 3) & ~3u);
	 *     R_CNA(0x118c, ((inw - 1) << 16) | (full_inh - 1));
	 *
	 * That encoder is the one this board ran M = 1, 2, 3, 4 and 8 through
	 * EXACTLY, at 512 to 1024, on 2026-08-14. So the two emitters already
	 * agree on the input surface, one of them computes M > 1 correctly and
	 * this one does not, and the difference is therefore NOT here. That is
	 * a better place to be than before the round: the geometry is excluded
	 * with data rather than assumed.
	 *
	 * CHARSIU_CNA_1098 still reaches the one register whose value Mesa
	 * rounds and regcmd.c does not.
	 */
	/*
	 * ⚠ FP16 COUNTS THE CONTRACTION AXIS HERE, NOT THE WINDOW. Exact over
	 * all 4940 vendor fp16 streams: 0x1090 = ic / 8, the 2 byte feature
	 * atom. inw * 4 is the int8/int4 form and gives 4 for a 1 wide window,
	 * which is what this emitted for an fp16 job before.
	 */
	emit(&e, CNA, 0x1090,
	     mm->wdtype == CHARSIU_FP16 ? mm->k / 8 : inw * 4);
	/*
	 * ⚠ FP16 DESCRIBES ITS WINDOW AS 1 x 1, and these four registers are
	 * where it says so. Over all 4940 of the vendor's fp16 streams,
	 * 0x1094 and DPU 0x401c are 1 and 0x118c and DPU 0x4020 are 0, without
	 * exception, while its int4 streams carry the geometry these lines
	 * compute. Nothing sets wdtype to FP16 yet, so every stream that runs
	 * today is bit identical to before these branches existed.
	 */
	/*
	 * ⚠⚠ THE FOUR fp16 WINDOW REGISTERS AND WHAT THEY COST. These carry
	 * the vendor's constants -- 1, 0, 1, 0 -- and every one of them is a
	 * quantity WITH rows IN IT: inw * rows, the window corners, ow * rows,
	 * ow - 1. Telling the block the window is 1 x 1 and then handing it m
	 * rows leaves it nowhere to put them, and the board says exactly that:
	 * output row r comes back holding k in [r*K/m, (r+1)*K/m), so it splits
	 * the CONTRACTION axis across the output rows instead of running m rows
	 * over the whole of K (npu_fp16_test --inmap, m=2 and m=4, exact).
	 *
	 * CHARSIU_FP16_WINDOW=geom puts the computed geometry back so the two
	 * can be compared on the board. The vendor's own fp16 dispatches run
	 * M up to 128 with these constants, so the constants cannot be the
	 * whole story -- but they are the only thing in the stream that carries
	 * the row count, and a knob is how that gets settled rather than
	 * argued.
	 */
	{
		const char *fw = envq("CHARSIU_FP16_WINDOW");
		int f16win = mm->wdtype == CHARSIU_FP16 &&
			     !(fw && !strcmp(fw, "geom"));

		emit(&e, CNA, 0x1094, f16win ? 1u : inw * rows);
	emit(&e, CNA, 0x1098, envq("CHARSIU_CNA_1098")
	     ? (uint32_t)strtoul(envq("CHARSIU_CNA_1098"), NULL, 0)
	     : ((inw * rows + 3) & ~3u));
	emit(&e, CNA, 0x109c, 0x00000000);
	emit(&e, CNA, 0x1100, 0x00000000);
	emit(&e, CNA, 0x1104, 0x00000000);
	emit(&e, CNA, 0x1110, job->weight_addr);
	emit(&e, CNA, 0x1140, 0x00000000);
	emit(&e, CNA, 0x1144, 0x00000000);
	/*
	 * ⚠ BOTH HALVES ARE M - 1 ON THE WIDTH AXIS, not the width and the
	 * height. The vendor's int4 streams carry 0x004f004f at M = 80 on an
	 * image ONE ROW HIGH, so the low half is not the row count there; and
	 * ((M-1) << 16) | (M-1) is exact on all 3328 of them.
	 *
	 * Round 380 set this register from the vendor's file and the board said
	 * it made no difference. That round ran on the HEIGHT axis, where M
	 * moves nothing at all, so it did not test this and does not excuse it.
	 */
	emit(&e, CNA, 0x118c,
	     f16win ? 0u
	     : wide ? (((inw - 1) << 16) | (inw - 1))
		    : (((inw - 1) << 16) | (rows - 1)));
	}

	/*
	 * ⚠⚠ THE THREE REGISTERS THAT MAKE int4 A WEIGHTED SUM. Rounds 344 to
	 * 347.
	 *
	 * Every int4 result in this tree from round 265 to round 343 was exact
	 * against a NONLINEAR reference: the effective weight was
	 * (int16) fp16bits(s), a band from 1.00 to 1.18 plus zero, because the
	 * DPU was multiplying the two fp16 bit patterns as integers. Bisecting
	 * the vendor's captured stream by register range found the switch is
	 * CORE, not the DPU where I had guessed it:
	 *
	 *     0x3018   this tree 10000001   vendor 10000200
	 *     0x301c   this tree 001f0000   vendor 0000001f   (M-1, LOW half)
	 *     0x3020   this tree 000007ef   vendor 000003ff   (N-1, plainly)
	 *
	 * 0x3018 is the one that changes the arithmetic -- with it alone the
	 * output's first four words are already the correct ones -- and the
	 * other two are needed for the job to run: alone, 0x301c is VOID and
	 * 0x3020 fails its repeat control.
	 *
	 * ⚠ 0x3018 was recorded as "hangs" in rounds 339 and 341. It was tried
	 * ALONE, at a different shape, without the other two. A register judged
	 * dead in one configuration is not dead.
	 *
	 * With these three and nothing else, at M = 1, the board computes
	 *
	 *     out[n] = sum_k s(n,k) * a(k)     in float32
	 *
	 * and round 347 checked all 1024 output words against a CPU reference:
	 * 1024 of 1024 within 1e-4 relative, 1017 within 1e-5, worst 5.46e-05.
	 *
	 * ⚠ M > 1 IS NOT THIS. At M = 8 the same three give 8 of 4096, because
	 * the vendor puts M on the WIDTH axis and this file puts it on the
	 * height axis, and at M = 1 alone the two collapse to the same thing.
	 * LLM decode is M = 1, so this is enough to be useful and is NOT enough
	 * to be called general.
	 *
	 * CHARSIU_W4_BITPAT restores the old behaviour, so the round that
	 * claims this can show the fault coming back.
	 */
	/*
	 * THE FIVE AT 0x2810, WHICH THE VENDOR WRITES AND THIS TREE NEVER DID.
	 *
	 * They cost nothing -- round 357 measured 270.1 us against a 275.9
	 * baseline with them inserted -- and on core 0 the vendor's values are
	 * all zero, so emitting them changes nothing today. They are here so
	 * that they CAN be changed, because they are the only registers whose
	 * value differs between the vendor's core 0 stream and its core 1
	 * stream, and CHARSIU_OVERRIDE can only reach a register that is
	 * actually emitted.
	 *
	 * The vendor's two captured streams for the same op differ in exactly
	 * seven entries and every one of them is CBUF:
	 *
	 *          task 16 (core 0)   task 17 (core 1)
	 *   0x1038   00000007           0000010e
	 *   0x1018   40000404           4000040b
	 *   0x103c   08000000           08001c00
	 *   0x1040   10000000           2c001c00
	 *   0x2818   00000000           1c000000
	 *   0x2820   00000000           00000038
	 *
	 * 0x1c00 is 7168 and it reads like a CBUF base offset: each core is
	 * told where ITS part of the buffer starts. rocket's own uapi says a
	 * job's tasks stay on one core "to benefit from memory residency in
	 * SRAM", and round 361 showed two processes on ONE core are clean six
	 * times out of six while two on TWO cores corrupt three times out of
	 * four. Two cores given the SAME CBUF configuration is the shape of
	 * that.
	 */
	/*
	 * ⚠⚠ AND FP16 NEEDS IT TOO, which is the only thing left in the stream.
	 * The vendor writes all five of these in its fp16 dispatches exactly as
	 * it does in its int4 ones; this emitted them for int4 alone, so an
	 * fp16 job left the block unset. After DPU 0x40b8's closed form went in,
	 * a diff of our fp16 stream against the vendor's at ic=64 oc=1024 M=32
	 * -- with acc_out matched, so the arms line up -- differed in these five
	 * registers and nothing else.
	 *
	 * It is worth trying because fp16 is EXACT at m = 1 and wrong at every
	 * m above it (K=256 N=64: m=1 exact, m=2 14.12, m=4 17.62, m=8 14,
	 * m=32 14, m=80 16.25), and this block is the CBUF configuration, which
	 * is what m changes the demands on.
	 */
	if (mm->wdtype == CHARSIU_INT4 || mm->wdtype == CHARSIU_FP16) {
		unsigned r2;
		(void)0;

		int w1 = job->cbuf_window == 1;

		for (r2 = 0x2810; r2 <= 0x2820; r2 += 4)
			emit(&e, U28, r2,
			     w1 && r2 == 0x2818 ? (w1_bank() * 0x400u) << 16 :
			     w1 && r2 == 0x2820 ? w1_bank() * 8u : 0x00000000u);
	}

	{
		int w4v = mm->wdtype == CHARSIU_INT4 &&
			  !envq("CHARSIU_W4_BITPAT");

		/*
		 * ⚠⚠ AND THE w4v FORM OF 0x301c WAS CHOSEN WHERE IT CANNOT
		 * SHOW, EXACTLY LIKE 0x40b8's LITERAL 3.
		 *
		 * lines is rows - 1, so at M = 1 it is ZERO and the two forms
		 * -- lines, and lines << 16 -- are the SAME WORD. Round 347
		 * picked the low half from a vendor capture, and that capture
		 * is M = 32 on the WIDTH axis, where the low half is where M
		 * belongs. This file's geometry is the height axis everywhere
		 * else, so the low half makes the job disagree with itself
		 * above one row, and at one row nothing could tell.
		 *
		 * On the board at m = 2 the batched path returns row 0 whole
		 * and row 1 not at all, under every reading, every row step,
		 * every input packing and every 0x40b8. A row that is misplaced
		 * turns up somewhere; a row the block was never told to produce
		 * does not.
		 *
		 * CHARSIU_W4_301C=high puts M back in the half the rest of this
		 * stream uses. The default does not move until a board round
		 * says which.
		 */
		const char *e31 = envq("CHARSIU_W4_301C");
		int high = e31 && !strcmp(e31, "high");

		/*
		 * ⚠⚠ AND FP16 TAKES THE 0x200 FORM TOO. The board's own words:
		 * with 0x10000001 here, a single fp16 weight of 1.0 against
		 * A[0] = 1.0 came back 3600, and 3600 is 0x3c * 0x3c -- the
		 * HIGH BYTES of the two fp16 patterns multiplied as int8. A[8]
		 * = 9.0 gave 4320, which is 0x48 * 0x3c. So the output stage
		 * was fp16 by then and the MULTIPLY was still int8.
		 *
		 * 0x10000200 is what the vendor carries in all 4940 of its fp16
		 * streams and in its int4 ones alike. 0x301c stays on w4v: its
		 * fp16 form is the high half, which is what the non-w4v branch
		 * already emits, and the vendor agrees there.
		 */
		emit(&e, CORE, 0x3018,
		     (w4v || mm->wdtype == CHARSIU_FP16) ? 0x10000200u
							 : 0x10000001u);
		emit(&e, CORE, 0x301c,
		     wide ? ((uint32_t)(rows - 1) << 16) | (ow - 1)
			  : ((w4v && !high) ? lines : ((uint32_t)lines << 16)));
	}
	/*
	 * ⚠ int4 NEEDS A DIFFERENT CHANNEL COUNT HERE, and this is the register
	 * that carries it. Rounds 274 to 276 read it directly: the DPU writes
	 * ceil((v+1)/2) + extra(SIZE_E_2) channels, six for six across v of 31,
	 * 47, 79, 95, 111 and 127, with SIZE_E_2 at its shipped 3 contributing
	 * 8. int8 wants n-1 and gets all n; int4 asking for n-1 gets n/2 + 8,
	 * which is 40 of 64 and is why every int4 result before round 280 was
	 * short. Inverting it, v = 2*(n - 8) - 1, and at n = 64 that is 111,
	 * which is the value the whole layout above was measured at.
	 *
	 * ⚠ AND THE WRITE QUANTISES TO 16 CHANNELS. Round 282 ran seven values
	 * of n and the number of words the hardware wrote was floor(n/16) * 16
	 * every time:
	 *
	 *   n     16  24  32  40  48  56  64
	 *   wrote 16  16  32  32  48  48  64
	 *
	 * so n of 24, 40 and 56 lost their top eight channels to the write and
	 * not to the packing: their weights were placed correctly and the
	 * channels were never written at all. Asking for the next multiple of
	 * 16 covers the real count, and the extra channels are computed and
	 * ignored.
	 */
	/*
	 * ⚠ CHARSIU_W4_PAIRED IS THE int4 RECIPE, and it replaces the round-280
	 * workaround above rather than adding to it.
	 *
	 * Rounds 332 to 334 read the layout off the board instead of guessing
	 * it. --kpair says an EVEN word takes k 0..15 and 32..47 and an ODD word
	 * takes 16..31 and 48..63, so the two together are the whole of K and
	 * **a real channel is the pair (2r, 2r+1)**. Declare 2N channels and
	 * there are 2N words, which is N real channels each with all of K, and
	 * N*K nibbles is exactly the N*K/2 byte buffer: nothing padded.
	 *
	 * Three registers carry the count and ALL THREE have to say 2N. Round
	 * 334 found that the hard way: 0x3020 alone and 0x3020 with 0x402c both
	 * wrote N words and got half the channels, and **0x4030's high half is
	 * the one that was capping the write**. With all of them:
	 *
	 *   N=256  K=2048   512 words   256 EXACT of 256
	 *   N=512  K=1024  1024 words   512 EXACT of 512
	 *   N=1024 K=2048  2048 words  1024 EXACT of 1024
	 *
	 * The last of those is a projection's shape at 1 MiB of weights, which
	 * is where int4 used to collapse.
	 */
	emit(&e, CORE, 0x3020,
	     (mm->wdtype == CHARSIU_INT4 && !envq("CHARSIU_W4_BITPAT"))
		     ? mm->n - 1
		     : (charsiu_w4_paired(mm)
			? (uint32_t)(2 * mm->n - 1)
			: (mm->wdtype == CHARSIU_INT4 && mm->n > 8
			   ? (uint32_t)(2 * (ALIGN_UP(mm->n, 16) - 8) - 1)
			   : mm->n - 1)));
	emit(&e, CORE, 0x3024, 0x00000000);

	/*
	 * THE w4a16 OUTPUT STAGE, ported from the vendor's own configuration.
	 *
	 * The .rkllm's int4 convolution streams carry NO DPU registers at all,
	 * 3328 of them, because the vendor sets the DPU up in a separate stream
	 * and lets the state stand for every projection after it. Those
	 * configuration streams are in the same file: 2908 copies of one 71
	 * register block, and it is not the int8 stage with a bit changed.
	 *
	 *   0x4010  a0000002   int8 writes 0.  PROC_PRECISION 2, which is fp16
	 *   0x4030  ..0310     int8's low half is 0710
	 *   0x4038  00000053   int8 writes 00120080
	 *   0x4044  00000002   int8 writes 1
	 *   0x4050  00023333   int8's is oc dependent, 80011111 here
	 *   0x40ac  00000000  |
	 *   0x40b0  00000001  |  an IDENTITY requant. w4a16 does not requantise
	 *   0x40b4  00000000  |  at all, the output is a float
	 *
	 * These are not independent knobs and are not swept one at a time: an
	 * identity requant behind an integer output stage means nothing. The
	 * unit is the whole stage, and the control is that int8 must be
	 * untouched, which it is, since every one of these is inside the int4
	 * branch.
	 *
	 * The geometry registers are still computed rather than copied. Only the
	 * precision dependent ones come from the vendor.
	 */
	if (mm->wdtype == CHARSIU_INT4)
		w4a16 = 1;
	/*
	 * ROUND 186 SPLITS THE PORT IN TWO, so the board can say WHICH HALF
	 * leaves the NPU unable to start the next job.
	 *
	 * Round 185 measured that cleanly. The same int8 binary, the same
	 * register stream and the same boot: FIRST it is 1023 of 1024 byte
	 * exact, LAST, after five w4a16 jobs, it times out with the output
	 * still holding the sentinel. The port went in as two groups from two
	 * separate vendor streams, the DPU output stage and the RDMA
	 * coefficient fetch, so each one can be taken back out on its own.
	 * 0x100c is NOT in either group: it is the int4 weight format and
	 * without it the run is not w4a16 at all.
	 */
	/*
	 * ⚠⚠ AND AN FP16 WEIGHT JOB WANTS THE SAME OUTPUT STAGE, because both
	 * produce a 16 bit result and the int8 stage produces a byte. The board
	 * said so first: an fp16 job with this false came back 0x80808080 in
	 * every written word, which is the int8 zero point in all four bytes --
	 * the DPU had been asked for an int8 output and gave one.
	 *
	 * It is not a guess. Every register the WIDE branches switch lands on
	 * the value the vendor's own fp16 streams carry, all four of them:
	 *
	 *   0x4010  0xa0000002    (PROC_PRECISION 2, which is fp16)
	 *   0x4044  2
	 *   0x4050  0x00023333
	 *   0x40ac / 0x40b0 / 0x40b4   0 / 1 / 0, which is no requant at all
	 *
	 * ⚠ AND THE FILTER THAT NEARLY HID IT. I diffed our fp16 stream against
	 * the vendor's and dropped every register where the vendor's int4 value
	 * equals its fp16 one, reasoning that those cannot be fp16 specific
	 * since our int4 path differs there too and works. 0x4010 is exactly
	 * such a register, and it was the one that mattered: our int4 path is
	 * w4a16, which takes this branch, so it never differed. A filter over
	 * what the VENDOR does cannot answer a question about what WE do.
	 */
	w4_dpu = (w4a16 || mm->wdtype == CHARSIU_FP16) &&
		 !envq("CHARSIU_W4_NO_DPU");

	/*
	 * CHARSIU_WIDE8 forces parts of the w4a16 OUTPUT STAGE onto an int8
	 * weight job, one bit per register, so a board round can find which one
	 * makes the output four bytes wide instead of one.
	 *
	 * It is needed because a projection cannot leave the NPU as a byte. The
	 * coefficient buffer's scale is fixed at build time, and the output
	 * magnitude of ffn_down varies by up to 2971x between tokens, so a
	 * scale sized for the largest vector quantises a typical one to nothing.
	 * Measured on the CPU model of this format, not assumed:
	 * CHARSIU_NPU_OUT8 with a frozen scale produces "a country" where the
	 * float output produces the right sentence.
	 *
	 *   bit 0  0x4010   bit 1  0x4030 low   bit 2  0x4038
	 *   bit 3  0x4044   bit 4  0x4050       bit 5  0x40ac/b0/b4 identity
	 *
	 * CHARSIU_WIDE8=0x3f is the whole bundle. The bits exist separately
	 * because "the whole bundle changed something" does not say which
	 * register the width lives in, and a single field sweep is the method
	 * that worked on 0x4050 in round 260.
	 */
	{
		const char *e8 = envq("CHARSIU_WIDE8");

		wide8 = e8 ? (unsigned)strtoul(e8, NULL, 0) : 0;
		if (job->acc_out)
			wide8 = 0x3f;       /* the whole stage, which is the only
					     * combination that works: round 311
					     * wedged the block on 0x4010 alone
					     * and on 0x4050 alone. */
		if (mm->wdtype != CHARSIU_INT8)
			wide8 = 0;
	}
#define WIDE(bit) (w4_dpu || (wide8 & (1u << (bit))))
	/*
	 * THE RDMA COEFFICIENT FETCH GROUP IS OFF BY DEFAULT SINCE ROUND 192,
	 * because two of its four registers each leave the NPU unable to start
	 * the next job and the group changes nothing about the output.
	 *
	 * Measured rather than argued. One w4a16 job then one int8 job in its
	 * own process, six masks, every row with its baseline recovered and its
	 * w4a16 job confirmed to have written:
	 *
	 *   0xf   all four        int8 timed out
	 *   0     none            int8 byte exact
	 *   1     0x501c only     int8 byte exact
	 *   2     0x5034 only     int8 TIMED OUT
	 *   4     0x5040 only     int8 byte exact
	 *   8     0x5044 only     int8 TIMED OUT
	 *
	 * And it buys nothing: rounds 182 and 183 put two knobs on the
	 * coefficient buffer and got four byte identical runs with this group
	 * on, and round 186's run without it produced element 0 = -3692, the
	 * same value as with it. A group that alters no result and wedges the
	 * next job does not go in the default stream.
	 *
	 * It stays reachable, one bit per register, because when the
	 * coefficient surface does eventually have to be fetched this is where
	 * that starts:
	 *
	 *   bit 0  0x501c    bit 1  0x5034    bit 2  0x5040    bit 3  0x5044
	 *
	 * A clear bit writes int8's value for that register. 0x5034 and 0x5044
	 * are the two that wedge, and both of them differ from int8 in bit 30
	 * and in opposite directions, which is a lead and not an explanation:
	 * mask 0xf sets both of them the way the vendor's own stream does and
	 * still wedges.
	 */
	rdma_mask = 0;
	if (w4a16 && envq("CHARSIU_W4_RDMA_MASK"))
		rdma_mask = (unsigned)strtoul(envq("CHARSIU_W4_RDMA_MASK"),
					      NULL, 0);

	emit(&e, DPU, 0x400c, 0x40000004);
	emit(&e, DPU, 0x4010, WIDE(0) ? 0xa0000002u : 0x00000000u);
	emit(&e, DPU, 0x4014, 0x00000000);
	emit(&e, DPU, 0x4018, job->output_addr);
	{
		/* the other half of the fp16 window group; see 0x1094 */
		const char *fw2 = envq("CHARSIU_FP16_WINDOW");
		int f16w2 = mm->wdtype == CHARSIU_FP16 &&
			    !(fw2 && !strcmp(fw2, "geom"));

		emit(&e, DPU, 0x401c, f16w2 ? 1u : ow * rows);
		emit(&e, DPU, 0x4020, f16w2 ? 0u : ow - 1);
	}
	emit(&e, DPU, 0x4024, wide ? rows - 1 : lines);
	/* fp16: oc / 4 - 1, exact over all 4940 vendor fp16 streams */
	emit(&e, DPU, 0x4028,
	     mm->wdtype == CHARSIU_FP16 ? mm->n / 4 - 1 : 0x00000000u);
	emit(&e, DPU, 0x402c, mm->n - 1);   /* ⚠ NOT doubled: round 334 tried
					     and it changed nothing */
	/*
	 * 0x4030's low half. Mesa uses 0x0710 for a regular convolution and
	 * 0x0310 for a depthwise one, and 0x0710 is right here.
	 *
	 * THE COMMENT THIS REPLACES WAS WRONG, and it cost round 158. It said
	 * the vendor's own matmuls use 0x0310, read off the .rkllm streams that
	 * carry 0x03ff0310 and 0x003f0310. Those streams have ONE output
	 * channel and zero weight bytes: they are not matmuls at all, and the
	 * 8308 streams that ARE the projections do not write this register.
	 * Compiling the vendor's own int8 convolution at this exact shape
	 * (vendor-capture/gen_act.py, geom/a_lin_m1) settles it: 0x003f0710,
	 * which is what this line already emitted.
	 */
	emit(&e, DPU, 0x4030,
	     ((uint32_t)(charsiu_w4_paired(mm) ? 2 * mm->n - 1 : mm->n - 1) << 16) |
	     (uint32_t)(envq("CHARSIU_DPU_4030")
			? strtoul(envq("CHARSIU_DPU_4030"), NULL, 0)
			/* fp16 reaches 0x310 through w4_dpu, like w4a16 */
			: (WIDE(1) ? 0x0310u : 0x0710u)));
	emit(&e, DPU, 0x4034, wide ? (((uint32_t)(rows - 1) << 16) | (ow - 1))
				   : ((lines << 16) | 0));
	/* Mesa's regular conv value. The vendor's DPU only streams carry 0x53
	 * here, but those are elementwise ops rather than convolutions, so it
	 * is a candidate to sweep and not a value to copy. */
	emit(&e, DPU, 0x4038, (uint32_t)(envq("CHARSIU_DPU_4038")
					 ? strtoul(envq("CHARSIU_DPU_4038"), NULL, 0)
					 : (WIDE(2) ? 0x00000053u : 0x00120080u)));
	emit(&e, DPU, 0x403c, 0x00000000);
	emit(&e, DPU, 0x4044, WIDE(3) ? 0x00000002u : 0x00000001u);
	emit(&e, DPU, 0x4048, 0x80000000);
	emit(&e, DPU, 0x404c, 0x7fffffff);
	emit(&e, DPU, 0x4050, WIDE(4) ? 0x00023333u : 0x80011111u);
	emit(&e, DPU, 0x4058, 0x80000000);
	emit(&e, DPU, 0x405c, 0x7fffffff);
	/*
	 * THE ACTIVATION, and it is already off.
	 *
	 * Compiling one int8 convolution twice, with and without a ReLU, and
	 * diffing the two register streams (vendor-capture/gen_act.py) names the
	 * control exactly: 0x4060 is 0x0902 with the ReLU and 0x0903 without,
	 * and 0x406c and 0x4074 are the two stage minimums, 0 with the ReLU and
	 * INT32_MIN without. The maximums at 0x4070 and 0x4078 do not move,
	 * which is what a ReLU is: a floor, not a ceiling. A third model, linear
	 * again with different weights, is the control that separates these from
	 * the quantisation registers, and it does: 0x40ac and 0x40b0 move
	 * between the two linears and these three do not.
	 *
	 * charsiu already emitted all three at their linear values, so whatever
	 * confines its output is NOT the fused ReLU. CHARSIU_DPU_406C forces the
	 * minimum so a board round can check that this register reaches the
	 * hardware at all: setting it to 0 must cost the output its low half, and
	 * if it changes nothing the identification above is wrong.
	 */
	emit(&e, DPU, 0x4060, 0x00000903);
	emit(&e, DPU, 0x406c, (uint32_t)(envq("CHARSIU_DPU_406C")
					 ? strtoul(envq("CHARSIU_DPU_406C"), NULL, 0)
					 : 0x80000000u));
	/*
	 * CHARSIU_DPU_4070, the other half of the liveness test. Round 159's
	 * control was UNFALSIFIABLE and that is worth writing down: it set the
	 * minimum to 0, which can only floor the channels whose accumulator is
	 * already negative, and those are exactly the channels the symptom
	 * already floors. Both branches of the rule predicted the same bytes,
	 * and the run duly produced them.
	 *
	 * A control has to bite where the output is currently RIGHT. A maximum
	 * BELOW the whole range, or a minimum ABOVE it, flattens a surface that
	 * is otherwise correct, so a stage that is connected cannot hide.
	 */
	emit(&e, DPU, 0x4070, (uint32_t)(envq("CHARSIU_DPU_4070")
					 ? strtoul(envq("CHARSIU_DPU_4070"), NULL, 0)
					 : 0x7fffffffu));
	emit(&e, DPU, 0x4074, 0x80000000);
	emit(&e, DPU, 0x4078, 0x7fffffff);
	emit(&e, DPU, 0x407c, 0x010041c1);
	emit(&e, DPU, 0x4080, 0x00000000);
	emit(&e, DPU, 0x4084, 0x00000001);
	emit(&e, DPU, 0x4088, 0x80000000);
	emit(&e, DPU, 0x408c, 0x7fffffff);
	emit(&e, DPU, 0x4090, 0x00000000);
	emit(&e, DPU, 0x4094, 0x00000000);
	emit(&e, DPU, 0x409c, 0x00000000);
	emit(&e, DPU, 0x40a4, 0x80000000);
	emit(&e, DPU, 0x40a8, 0x7fffffff);
	/* w4a16 does not requantise: the vendor writes 0, 1, 0 here and the
	 * output leaves as a float. */
	emit(&e, DPU, 0x40ac, WIDE(5) ? 0u : (uint32_t)rq.offset);
	emit(&e, DPU, 0x40b0, WIDE(5) ? 1u : rq.scale);
	emit(&e, DPU, 0x40b4, WIDE(5) ? 0u : rq.shift);
	/*
	 * 0x40b8. int8 computes ow * (2*oh - window), which is 1 at M = 1, and
	 * the vendor's w4a16 output stage writes 3 -- the ONLY register left
	 * differing after the whole bundle is applied, found by diffing
	 * emit_job's stream against the .rkllm's 67 register DPU block offline.
	 *
	 * Round 312 swept it and 3 is a peak on both columns rather than a
	 * trend: bytes written 112, 160, 208, 256, 208, 128, 64 and elements
	 * exact 4, 8, 12, 64, 16, 16, 16 at 0, 1, 2, 3, 4, 7, 15.
	 */
	/*
	 * ⚠⚠ AND ROUND 312 SWEPT IT AT M = 1, WHERE IT CANNOT VARY.
	 *
	 * The int8 arm computes ow * rows, which is the row count on the height
	 * axis and does scale with M. The acc_out arm is a literal 3 at every
	 * M, fitted on the one width where the two arms cannot be told apart.
	 *
	 * That is the shape of the m > 1 defect: charsiu_matmul, which takes
	 * the int8 arm, is 128 of 128 bytes exact at M = 2 on the board, and
	 * npu_gemm_test, which takes the acc_out arm, has never been right
	 * above one row under any reading.
	 *
	 * ⚠⚠ AND THE BOARD SAID IT IS 3 * rows.
	 *
	 * Swept 0 to 16 at three widths, scoring every candidate under every
	 * reading. The peak walks with M and nothing else comes near it:
	 *
	 *   m = 1   value  3   64 of 64     every other value 4, 8, 12 or 16
	 *   m = 2   value  6   64 of 128    every other value 8 to 21
	 *   m = 4   value 12   68 of 256    every other value 9 to 40
	 *
	 * 3, 6, 12 at m = 1, 2, 4. The literal was the m = 1 case of a value
	 * that follows the row count, exactly as the int8 arm's does, and
	 * round 312 could not have seen that: it swept at M = 1.
	 *
	 * ⚠ IT IS NOT THE WHOLE FIX. Each peak is worth about ONE ROW -- 64
	 * values whatever m is -- so the other rows are still wrong. What this
	 * buys is a reproducible signal well clear of the 8 to 20 noise floor,
	 * and a default that cannot change decode, since 3 * rows is 3 at
	 * m = 1.
	 *
	 * CHARSIU_DPU_40B8 replaces the value so the next sweep can hold it.
	 */
	/*
	 * ⚠⚠ AND `rows` IS NOT THE BATCH COUNT ON THE WIDTH AXIS, WHERE IT IS
	 * 1. 3 * rows is 3 there at every M, and 3 is exactly what the board
	 * wrote when it produced 92 of 128 words at m = 2: the baseline of the
	 * sweep and the `0x40b8 = 3` row of it are the same number twice.
	 *
	 * Swept on the board on the WIDTH axis at m = 2, K = 64 N = 64:
	 *
	 *   value    1    2    3    4    6    8   16
	 *   words   68   80   92  104  128  104   64
	 *   absent  54   41   31   20    0   21   60
	 *
	 * 6 is 3 * M and it writes the FULL SURFACE with nothing absent. It is
	 * not a trend either -- 8 is worse than 4 -- so the fill point is a
	 * point and not a floor.
	 *
	 * ⚠ THE SAME VALUE BUYS SOMETHING DIFFERENT ON EACH AXIS. The height
	 * sweep above found 3, 6, 12 at m = 1, 2, 4 and each peak was worth
	 * about ONE ROW, 64 values whatever m was. On the width axis 6 at m = 2
	 * is all 128. Same expression, and the axis is what decides whether it
	 * completes the surface or one row of it.
	 *
	 * `wide ? ow : rows` is M under both arrangements and is bit identical
	 * on the height axis, where ow is 1 and rows is M.
	 *
	 * 🏁 AND THE VENDOR'S OWN FILE CONFIRMS IT, all 3328 int4 streams, with
	 * no exception -- so 3 * M is not a fit to two board points any more.
	 *
	 * Their output block carries a CHUNK and a TOTAL, which is what makes
	 * the rule visible. Reading the four registers that move across those
	 * 3328 streams, with T = 0x401c and W = this stream's own width:
	 *
	 *   0x4018   output base + 16 * (this chunk's first position)   3328/3328
	 *   0x401c   T, the TOTAL width, not this chunk's               3328/3328
	 *   0x4028   T - W, the positions this chunk does not cover     3328/3328
	 *   0x40b8   4 * T - W                                          3328/3328
	 *   0x4020   W - 1, and 0x4034 the same
	 *
	 * They split a prefill and we do not: their (W, T) pairs are (32,32),
	 * (64,64), (80,96), (16,96), (80,128), (48,128), (40,64), (24,64),
	 * (40,96) and (40,128), so a 96 token prompt goes 80 then 16. charsiu
	 * dispatches the whole width at once, which is T = W, and every one of
	 * those registers collapses to what this file already emits: 0x4018 at
	 * the base, 0x401c = M, 0x4028 = 0, and 0x40b8 = 4M - M = 3M.
	 *
	 * ⚠ THE int8 HEAD TAKES A DIFFERENT CONSTANT -- 7 * W, read off their
	 * 8160 wide output head at W of 1, 32 and 64. The rule above is the
	 * int4 one and the 3328 it holds on are exactly the int4 streams.
	 *
	 * ⚠ AND 16 BYTES A POSITION IS THE READ ORDER'S OWN CLAIM. 0x4018
	 * stepping by 16 for each position skipped says consecutive rows sit
	 * four floats apart in the output surface, which is what
	 * charsiu_acc_index puts at (mi/2)*8 + (mi%2)*4. That is the first
	 * confirmation of the row placement from something other than this
	 * board.
	 */
	/* ow * (2 * full_oh - win_orows); on the width axis full_oh is 1 */
	{
		const char *e8b = envq("CHARSIU_DPU_40B8");
		unsigned batch = wide ? ow : rows;
		uint32_t v = (job->acc_out || charsiu_w4_paired(&job->mm))
			   ? 3u * batch : (uint32_t)(ow * (2 * rows - rows));

		/*
		 * ⚠⚠ FP16 HAS A CLOSED FORM AND IT IS THE LAST REGISTER THAT
		 * DIFFERED. With acc_out set -- which is what the fp16 probe
		 * submits -- our stream matched the vendor's on every register
		 * but this one, at BOTH attention shapes:
		 *
		 *   ic 512  oc 64   M 32   ours 0x60   vendor 0xfffffe13
		 *   ic 64   oc 1024 M 32   ours 0x60   vendor 0xffffe103
		 *
		 * and (oc / 4 + 3) - (M * oc) / 4 reproduces the vendor's value
		 * on ALL 4940 of its fp16 streams, exactly. It is fp16's: the
		 * same form matches only 512 of its 3328 int4 streams, so the
		 * int4 and acc_out arms above keep the values this board
		 * measured for them.
		 *
		 * ⚠ AND THE DIFF THAT FOUND IT HAD BEEN COMPARING THE WRONG ARM
		 * ALL EVENING. emit_job leaves acc_out OFF unless
		 * CHARSIU_ACC_OUT is set, while npu_fp16_test sets it, so every
		 * stream comparison before this one was of a stream nobody
		 * submitted. Six registers of difference became one when the
		 * dump was told to match the submit.
		 */
		if (mm->wdtype == CHARSIU_FP16)
			v = (uint32_t)((int32_t)(mm->n / 4 + 3)
				       - (int32_t)((mm->m * mm->n) / 4));
		if (e8b)
			v = (uint32_t)strtoul(e8b, NULL, 0);
		emit(&e, DPU, 0x40b8, v);
	}
	emit(&e, DPU, 0x40bc, 0x00000000);
	emit(&e, DPU, 0x40c0, 0x04440100);
	emit(&e, DPU, 0x40c8, 0x00000000);
	emit(&e, DPU, 0x40cc, 0x00000000);
	emit(&e, DPU, 0x40d0, 0x0040ffff);
#undef WIDE
	for (r = 0x4100; r <= 0x4120; r += 4)
		emit(&e, DPU, r, 0x00000000);
	emit(&e, DPU, 0x4130, 0x00000000);
	for (r = 0x4140; r <= 0x4154; r += 4)
		emit(&e, DPU, r, 0x00000000);
	emit(&e, DPU, 0x4160, 0x00000000);
	emit(&e, DPU, 0x4170, 0x00000000);
	emit(&e, DPU, 0x4174, 0x00000000);
	for (r = 0x4184; r <= 0x4194; r += 4)
		emit(&e, DPU, r, 0x00000000);

	emit(&e, RDMA, 0x500c, 0x00000000);     /* ow - 1 */
	emit(&e, RDMA, 0x5010, lines);
	emit(&e, RDMA, 0x5014, mm->n - 1);
	emit(&e, RDMA, 0x5018, 0x00000000);
	/*
	 * THE COEFFICIENT SURFACE, and w4a16 configures it differently too.
	 *
	 * Round 182 proved charsiu's coefficient buffer is not being read at all
	 * on this path: CHARSIU_NO_LIFT and CHARSIU_COEF_C both changed nothing,
	 * four runs byte identical, so neither A nor C reaches the output. The
	 * DPU stage had already been ported; the DPU_RDMA that FETCHES the
	 * surface had not.
	 *
	 * The vendor keeps that configuration in its own stream as well, 21 RDMA
	 * registers, and against what this emitted exactly four differ, none of
	 * them geometry:
	 *
	 *   0x501c   0 where int8 writes 0710
	 *   0x5034   4000004c where int8 writes 41
	 *   0x5040   the ROW COUNT. Another vendor stream carries 0x20 beside a
	 *            0x1f in 0x500c, so it is 0x500c + 1, and this wrote 0
	 *   0x5044   000280a1 where int8 writes 40000010
	 *
	 * Which makes round 182's two dead knobs the control for this round: if
	 * the surface is now fetched, CHARSIU_NO_LIFT and CHARSIU_COEF_C have to
	 * stop being inert.
	 */
	emit(&e, RDMA, 0x501c, (rdma_mask & 1) ? 0x00000000u : 0x00000710u);
	emit(&e, RDMA, 0x5020, job->coef_addr);
	emit(&e, RDMA, 0x5024, job->coef_addr +
	     (uint32_t)(table_bytes(mm) + scale_table_bytes(mm)));
	/*
	 * The rest of the DPU_RDMA. Round 142 emitted the first seven of these
	 * and stopped, and the job timed out: the unit that fetches the per
	 * channel records was half configured, so the DPU waited for data that
	 * never arrived. A stream can be geometrically perfect, engage every
	 * unit, and still hang on the one unit nobody finished programming.
	 */
	emit(&e, RDMA, 0x5028, 0x00000000);
	emit(&e, RDMA, 0x502c, 0x00000000);
	emit(&e, RDMA, 0x5030, 0x00000000);
	emit(&e, RDMA, 0x5034, (rdma_mask & 2) ? 0x4000004cu : 0x00000041u);
	emit(&e, RDMA, 0x5038, 0x00000000);
	emit(&e, RDMA, 0x5040, (rdma_mask & 4) ? rows : 0x00000000u);
	emit(&e, RDMA, 0x5044, (rdma_mask & 8) ? 0x000280a1u : 0x40000010u);
	emit(&e, RDMA, 0x5048, 0x00000000);
	emit(&e, RDMA, 0x504c, 0x00000000);
	emit(&e, RDMA, 0x5064, 0x00000000);
	emit(&e, RDMA, 0x506c, 0x00000000);
	emit(&e, RDMA, 0x5078, 0x00000000);
	emit(&e, RDMA, 0x507c, 0x00000000);

	/*
	 * ENGAGE THE UNITS. Without this nothing runs: round 141 submitted a
	 * complete and correct looking stream with no op enable in it, the
	 * kernel accepted it, and the job timed out with the IOMMU failing its
	 * stall on the way down.
	 *
	 * Each unit is enabled at its OWN op enable with the 0x1d mask, in
	 * FORWARD order: CNA, CORE, DPU, RDMA. The broadcast form, target 0x81
	 * on the PC's own enable, restarts the program counter mid stream and
	 * engages the units before the geometry is committed, which runs them on
	 * an empty window.
	 *
	 * This was written in REVERSE order first, from a comment in the driver
	 * describing what the NVDLA programming guide asks for. That comment
	 * belongs to a KNOB. The default, and the order in the stream that
	 * computes on this hardware, is forward, and the stream diff is what
	 * said so. Reasoning from a comment about an option is not the same as
	 * matching what runs.
	 */
	emit(&e, CNA, 0x1008, 0x0000001d);
	emit(&e, CORE, 0x3008, 0x0000001d);
	emit(&e, DPU, 0x4008, 0x0000001d);
	emit(&e, RDMA, 0x5008, 0x0000001d);

	return e.n > max ? 0 : merge_vendor_values(out, e.n, max);
}
