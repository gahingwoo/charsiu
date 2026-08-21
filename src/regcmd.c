// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The register command stream for a matmul, as a 1x1 convolution.
 *
 * Every value here is either read off the vendor's own dispatches in a .rkllm
 * (tools/rkllm_regcmd.py, docs/vendor-dispatch.md) or established on hardware in
 * the driver work at github.com/gahingwoo/linux-rk3576-npu. Where the two agree
 * the comment says so, because agreement between a vendor LLM dispatch and a
 * rule derived from vendor CONVOLUTION compiles is the strongest evidence this
 * project has for a field's meaning.
 *
 * What is NOT here yet: the requant coefficients, the LUT, and anything to do
 * with addresses. This emits geometry, so that it can be diffed against the
 * vendor's streams on a desktop before any of it runs.
 */
#include <stdlib.h>
#include <string.h>

#include "charsiu.h"

#define CNA   0x0201u
#define CORE  0x0801u
#define DPU   0x1001u
#define RDMA  0x2001u

#define DIV_ROUND_UP(n, d)  (((n) + (d) - 1) / (d))
#define ALIGN_UP(n, a)      (DIV_ROUND_UP((n), (a)) * (a))
#define MIN2(a, b)          ((a) < (b) ? (a) : (b))

struct emitter {
	uint64_t *out;
	size_t max;
	size_t n;
};

static void emit(struct emitter *e, unsigned target, unsigned reg, uint32_t val)
{
	if (e->n >= e->max) {
		e->n = e->max + 1;      /* overflow marker */
		return;
	}
	e->out[e->n++] = ((uint64_t)target << 48) | ((uint64_t)val << 16) | reg;
}

/*
 * The effective activation element size. int4 weights consume the activation as
 * 16 bits whatever the stream asks for, measured in round 178 by holding one
 * live nibble and sweeping a one hot input: every nibble paired with k = 2 *
 * k_ours + 1, which is a 16 bit element read out of an 8 bit buffer. Everything
 * that derives from the atom has to agree with that or the surface stride and
 * the packing describe different buffers.
 */
enum charsiu_dtype charsiu_effective_adtype(const struct charsiu_matmul *mm)
{
	if (mm->wdtype == CHARSIU_INT4 && !getenv("CHARSIU_A8_STRIDE1"))
		return CHARSIU_FP16;    /* any 2 byte type: the atom is 8 */
	return mm->adtype;
}

unsigned charsiu_entries_per_row(const struct charsiu_matmul *mm)
{
	unsigned atom = charsiu_feature_atom(charsiu_effective_adtype(mm));
	unsigned total = DIV_ROUND_UP(mm->k, atom);
	unsigned last = total % 4;
	unsigned width = 1;             /* one column, always */

	return (total / 4) * width +
	       (last == 3 ? width : DIV_ROUND_UP(last * width, 4));
}

unsigned charsiu_k_eff(const struct charsiu_matmul *mm)
{
	if (getenv("CHARSIU_NO_KALIGN"))
		return mm->k;
	return charsiu_k_padded(mm->k, charsiu_effective_adtype(mm));
}

size_t charsiu_weight_bytes(const struct charsiu_matmul *mm)
{
	/* The CNA counts output channels in PAIRS and reads the weights for the
	 * whole pair, so an odd N is rounded up here and nowhere else. Board,
	 * round 139: doing this took every odd output channel count from wrong
	 * across its second tile to exact. */
	unsigned n = ALIGN_UP(mm->n, 2);
	unsigned k = charsiu_k_eff(mm);

	switch (mm->wdtype) {
	case CHARSIU_INT4: return (size_t)k * n / 2;
	case CHARSIU_INT8: return (size_t)k * n;
	default:           return (size_t)k * n * 2;
	}
}

size_t charsiu_emit_matmul(const struct charsiu_matmul *mm,
			   uint64_t *out, size_t max)
{
	struct emitter e = { out, max, 0 };
	unsigned n_pad = ALIGN_UP(mm->n, 2);
	unsigned surf = charsiu_entries_per_row(mm);
	unsigned rows = mm->m;
	size_t wbytes = charsiu_weight_bytes(mm);

	/*
	 * 0x1004 is 0x0e in every vendor dispatch, convolution and LLM alike.
	 * CORE 0x3004 mirrors it.
	 */
	emit(&e, CNA, 0x1004, 0x0000000e);
	emit(&e, CORE, 0x3004, 0x0000000e);

	/*
	 * NOT DECODED. The vendor carries 0x00600120 on its int4 projections
	 * and 0x20200120 on its fp16 attention, so it is at least partly a
	 * precision field, and the low half is the same in both. Copied rather
	 * than derived, and marked as such: emitting 0 here is known wrong, and
	 * a constant we cannot explain is the next thing to explain.
	 */
	emit(&e, CNA, 0x100c,
	     mm->wdtype == CHARSIU_INT4 ? 0x00600120u : 0x20200120u);
	emit(&e, CNA, 0x1010, 0x00000fff);
	emit(&e, CNA, 0x1014, (1u << 3) | 1u);  /* stride 1 both axes */

	/*
	 * The CBUF pair. 0x0505 and 0x14000000 mean "more than one row window",
	 * 0x0404 and 0x10000000 mean one. Established over 87 compiled .rknn
	 * (86 of 87) and agreed with by the vendor's own LLM dispatches, whose
	 * single row matmuls all carry the non split pair.
	 *
	 * A matmul emitted here is one window by construction: it is not tiled
	 * yet. Tiling is what makes this false, and it is not written.
	 */
	emit(&e, CNA, 0x1018, 0x40000404);
	emit(&e, CNA, 0x1020, (uint32_t)(wbytes / n_pad));   /* bytes per kernel */
	emit(&e, CNA, 0x101c, (uint32_t)wbytes);
	emit(&e, CNA, 0x1024, n_pad - 1);
	emit(&e, CNA, 0x1028, ((surf * rows) << 16) | (mm->k - 1));
	/*
	 * (width - 1) << 16 | (rows - 1). The high half is the WIDTH, not the
	 * row count: the vendor carries 0x0000007f at M = 128 on a one column
	 * image, so the high half is 0 there. Every convolution this was read
	 * from before was square, which made the two readings the same number.
	 */
	emit(&e, CNA, 0x102c, ((1u - 1) << 16) | (rows - 1));
	/* Bytes per kernel again, NOT doubled. The convolution path doubles it
	 * because its int8 weights are stored two bytes each; a matmul's are
	 * not. Vendor: 1024 for int4 at K=2048, 128 for fp16 at K=64. */
	emit(&e, CNA, 0x1030, ((uint32_t)(wbytes / n_pad) << 16) | 0);
	emit(&e, CNA, 0x1034, rows - 1);        /* ow * oh - 1, one column */
	emit(&e, CNA, 0x103c, surf << 16);
	emit(&e, CNA, 0x1040, 0x10000000);
	/* (width << 16) | surf, and the width of a matmul is one column. Pinned
	 * by shapes where the two halves differ: the vendor carries 0x00010002
	 * at M = 128 with K = 64, so the high half is the width and not M. */
	emit(&e, CNA, 0x1044, (1u << 16) | surf);
	emit(&e, CNA, 0x1048, 0x0000000b);
	emit(&e, CNA, 0x1078, ((1u - 1) << 16) | (rows - 1));
	emit(&e, CNA, 0x107c, mm->k - 1);
	emit(&e, CNA, 0x1080, 0x00000000);      /* no padding: a matmul has none */
	emit(&e, CNA, 0x1084, 0x00000000);
	/* The row count, plainly. The convolution path rounds inw * rows up to a
	 * multiple of four here; the vendor's matmuls carry 1 at M = 1 and 128
	 * at M = 128, so no rounding is applied to a one column image. */
	emit(&e, CNA, 0x1098, rows);

	emit(&e, CORE, 0x3020, mm->n - 1);      /* the REAL count, not the pair */

	emit(&e, DPU, 0x400c, 0x40000004);
	emit(&e, DPU, 0x4020, 0);               /* ow - 1 */
	emit(&e, DPU, 0x4024, rows - 1);        /* oh - 1 */
	emit(&e, DPU, 0x402c, mm->n - 1);
	emit(&e, DPU, 0x4034, ((rows - 1) << 16) | 0);

	emit(&e, RDMA, 0x5014, mm->n - 1);

	return e.n > max ? 0 : e.n;
}

/*
 * Pack A[M][K] into [K/atom][M][atom], BIASED BY -0x80 like the weights.
 *
 * ROUND 161 MEASURED THIS, and it corrects what the weight packer below still
 * says about the input. The MAC does not take the input raw and subtract 0x80
 * itself: it reads the byte as a SIGNED value, exactly as it reads the weight.
 * Writing the raw uint8 makes an input of 168, which means +40 against a zero
 * point of 128, arrive as -88.
 *
 * The board says so with no room left in it. A probe with a zero bias and a MAC
 * walking through zero (CHARSIU_NEG) came back matching
 *
 *     out = (clamp(int8(in) * K * d * mult, 0, 255) + 128) mod 256
 *
 * on 64 of 64 channels with no value off by even one, where the raw reading
 * misses 63 of the 64. The same model, at the much smaller amplitude the bias
 * ramp probe uses, also accounts for the residual that had been written off as
 * three counts of rounding: an input held at 128 arrives as -128 rather than 0,
 * so the MAC is -128 * weight_sum instead of exactly zero, which is +-3 at that
 * scale. It was never rounding.
 *
 * The CNA's own pad register agrees, and did all along. It carries
 * in_zp - 0x80, not in_zp.
 */
void charsiu_pack_input(const struct charsiu_matmul *mm, const uint8_t *src,
			uint8_t *dst, size_t dst_size, uint8_t input_zero_point)
{
	/*
	 * INT4 WEIGHTS CONSUME THE ACTIVATION AS SIXTEEN BITS, whatever this
	 * asks for, and that is measured rather than assumed.
	 *
	 * Round 178 held ONE live nibble in the whole weight buffer and swept a
	 * ONE HOT input across k. Every nibble paired with exactly one k, so the
	 * layout is a permutation and nothing is broadcast, and the pairing is
	 *
	 *   byte b nibble h  ->  k = 4b + 2h + 1
	 *
	 * on all twelve offsets probed, with no exception. Against what this
	 * packer intends, k = 2b + h, that is exactly
	 *
	 *   k_hardware = 2 * k_ours + 1
	 *
	 * which is what a 16 bit element does to an 8 bit buffer: the hardware's
	 * element k is our bytes 2k and 2k+1, and it pairs with the high one. It
	 * also explains the rest at a stroke. A row of 8 bytes is 16 nibbles and
	 * therefore 16 of the hardware's elements, not 16 of ours, so channel 0
	 * takes our k = 1..31 odd and channel 1 our k = 33..63 odd, which is
	 * what byte 8 measured. The vendor only ever runs int4 as w4a16.
	 *
	 * So for int4 the atom is the 16 bit one and each value goes in the high
	 * byte of its pair. CHARSIU_A8_STRIDE1 restores the old packing as the
	 * control: with it the fault must come back.
	 */
	enum charsiu_dtype eff = charsiu_effective_adtype(mm);
	unsigned atom = charsiu_feature_atom(eff);
	int wide = eff != mm->adtype;
	unsigned esz = wide ? 2 : 1;
	unsigned i, kk;

	/*
	 * ROUND 179 SETTLED THE PAIRING and pointed past it. With each value in
	 * the high byte of a 16 bit slot, --kpair reads byte b nibble h paired
	 * with k = 2b + h, exactly what the packer intends, where the 8 bit
	 * packing read 2k + 1. So the element is 16 bits wide and this much is
	 * measured twice.
	 *
	 * What it also showed is that the LOW byte contributes nothing at all: a
	 * one hot of 100 placed there never lit a channel, and a 16 bit INTEGER
	 * cannot do that, since 100 would have been about 100 output counts. A
	 * float can and must, because a value with a zero exponent byte is a
	 * subnormal near zero. So the activation is an fp16 in this mode, which
	 * is also the only way the vendor runs int4, as w4a16.
	 *
	 * That is why the values are still wrong: writing an int8 byte into the
	 * high half of an fp16 makes a float with that byte as its sign and
	 * exponent, which is garbage of a wildly varying magnitude, and the
	 * impulse duly comes back at the rails. A real fp16 activation is the
	 * next step and is what charsiu_pack_input_f16 below is for.
	 */
	memset(dst, wide ? 0 : (uint8_t)(input_zero_point - 0x80), dst_size);
	for (i = 0; i < mm->m; i++)
		for (kk = 0; kk < mm->k; kk++) {
			size_t off = ((size_t)(kk / atom) * mm->m * atom
				      + (size_t)i * atom + kk % atom) * esz;

			dst[off + esz - 1] =
				(uint8_t)(src[i * mm->k + kk] - 0x80);
		}
}

/*
 * Pack A[M][K] as real fp16, which is what int4 weights consume.
 *
 * Same tiling as the integer packer, an element being two bytes rather than one,
 * and the value being an actual half rather than a byte dropped into half of
 * one. There is no zero point: a float carries its own sign, so the caller
 * passes the dequantised values it means.
 */
void charsiu_pack_input_f16(const struct charsiu_matmul *mm, const float *src,
			    uint8_t *dst, size_t dst_size)
{
	unsigned atom = charsiu_feature_atom(CHARSIU_FP16);
	unsigned i, kk;

	memset(dst, 0, dst_size);
	for (i = 0; i < mm->m; i++)
		for (kk = 0; kk < mm->k; kk++) {
			size_t off = ((size_t)(kk / atom) * mm->m * atom
				      + (size_t)i * atom + kk % atom) * 2;
			uint16_t h = charsiu_float_to_half(src[i * mm->k + kk]);

			dst[off] = (uint8_t)(h & 0xff);
			dst[off + 1] = (uint8_t)(h >> 8);
		}
}

void charsiu_pack_weights(const struct charsiu_matmul *mm,
			  const uint8_t *src, uint8_t *dst)
{
	unsigned ng = charsiu_weight_ngroup(mm->wdtype);
	unsigned kg = charsiu_weight_kgroup(mm->wdtype);
	unsigned n_pad = ALIGN_UP(mm->n, 2);
	unsigned ke = charsiu_k_eff(mm);
	unsigned n, k;

	/*
	 * [n/ng][k/kg][n%ng][k%kg], edge tiles cut short rather than padded.
	 * Read off the vendor's compiler at 64 by 64, 64 by 34, 64 by 56 and
	 * 48 by 40 with two position encoded models, one whose weight depends
	 * only on the output channel and one only on the input channel.
	 *
	 * The padding channel that ALIGN_UP(n, 2) adds has no weights of its
	 * own and stays at zero: the CNA computes it and the DPU, which is told
	 * the real count, never writes it.
	 */
	memset(dst, 0, charsiu_weight_bytes(mm));

	/*
	 * INT4, PLACED WHERE THE HARDWARE WAS MEASURED TO LOOK, and no further.
	 *
	 * The sparse map (tools/charsiu_int4.c --map) reads the fetch directly,
	 * one live nibble at a time. At K = 64 and N = 64 it says:
	 *
	 *   channel n is read from byte (n / 32) * 512 + (n % 32) * 8,
	 *   eight bytes, and NOTHING ELSE in the buffer is fetched at all
	 *
	 * with the same 8 byte row at K = 32, 64 and 128 and at N = 32 and 64, so
	 * the row does not grow with K. The same probe on int8 lights 512 of 512
	 * with n = byte / 32 and k = byte % 32, which is what the int8 path below
	 * writes, so the probe is known good on a case whose answer is known.
	 *
	 * ⚠ WHAT WAS NOT KNOWN IS NOW MEASURED. Rounds 265 to 277 fixed a byte
	 * width defect in the probe itself, which had been reading a four byte
	 * output as bytes, and re-read the whole fetch densely. Two things came
	 * out that this branch was written without:
	 *
	 *   the ADDRESS. With CORE 0x3020 = 111 every one of 64 channels is
	 *   written and reachable, and each is fed by TWO eight byte groups, one
	 *   at its own offset and one 256 bytes later. The group bases came off
	 *   a stride 8 sweep of the whole buffer, 128 live groups of 256.
	 *
	 *   the k, and it is per channel PARITY. byte 0 pairs with k 0 on
	 *   channel 0, byte 8 with k 16 on channel 1, byte 16 with k 0 on
	 *   channel 2. So an EVEN channel is fed k 0..15 and 32..47 and an ODD
	 *   channel is fed k 16..31 and 48..63. Confirmed independently at
	 *   K = 32, where channel 0 takes k 0..15 and channel 1 takes k 16..31.
	 *
	 * Half the reduction reaches any given channel, and WHICH half depends
	 * on the channel. That is why a global mask over k is wrong: rounds 278
	 * and 279 masked (k mod 32) < 16 for every channel, which is right for
	 * the even ones and exactly backwards for the odd.
	 *
	 * The group bases are a TABLE and not a formula on purpose. Seven of the
	 * eight steps are 128 or 384 and the last is 64, so any closed form for
	 * it would be fitted to one point. This is what was measured:
	 *
	 *   c 0..7   ->    0     c 16..23 ->  512     c 32..39 -> 1024
	 *   c 8..15  ->  128     c 24..31 ->  640     c 40..47 -> 1152
	 *   c 48..55 -> 1536     c 56..63 -> 1600
	 *
	 * and it is refused outside K = 64 or 32 with N up to 64, because that
	 * is where it was read. Rounds 167, 168, 171 and 173 all went wrong by
	 * filling a gap with a guess and this branch is not going to be the
	 * fifth.
	 *
	 * ⚠ THE ADDRESS MAP WAS READ AT 0x3020 = 111. charsiu emits n - 1
	 * there, which gives 40 channels, so this layout only describes the
	 * hardware when that register is overridden. Until the driver sets it,
	 * a job packed this way and run without the override reaches only its
	 * first 40 channels.
	 */
	if (mm->wdtype == CHARSIU_INT4) {
		unsigned order = getenv("CHARSIU_INT4_ORDER")
			? (unsigned)atoi(getenv("CHARSIU_INT4_ORDER")) : 0;

		if ((mm->k != 64 && mm->k != 32) || mm->n > 64) {
			/* refuse rather than invent, and say so */
			return;
		}

		for (n = 0; n < mm->n; n++) {
			/*
			 * ONE EXPRESSION FOR BOTH MEASURED TABLES, 16 points.
			 *
			 *   K=64  0 128 512 640 1024 1152 1536 1600
			 *   K=32  0 128 256 384  512  640  768  832
			 *
			 * Groups of eight channels pair up, the pair stride is
			 * 8*K, and the odd member of a pair sits 128 bytes into
			 * it. The LAST pair sits 64 bytes in rather than 128,
			 * in both tables independently, so that is read rather
			 * than fitted and it is not a cap: 3*512 + 128 + 56 is
			 * well inside a 2048 byte buffer.
			 *
			 * ⚠ "LAST" IS THE HIGHEST GROUP IN USE, NOT g == 7.
			 * Round 281 wrote 7 because the map was read at N = 64
			 * where the two coincide, and N = 32 came back 24 of 32
			 * and N = 16 came back 8 of 16, which is one group of
			 * eight wrong in each: g 3 at N = 32 and g 1 at N = 16,
			 * the highest group in both. So the 64 belongs to
			 * whichever odd group is last.
			 *
			 * That is fitted to two failures and one success and it
			 * is not read: nothing has run at an N whose highest
			 * group is EVEN, where this predicts no 64 anywhere.
			 */
			/*
			 * glast comes off the count the HARDWARE is told, which
			 * is n rounded up to 16, not the caller's n. On every
			 * case measured so far the two give the same base, since
			 * they can only differ on a group the rounding adds, but
			 * the register and the table should be reading the same
			 * number.
			 */
			/*
			 * ⚠ A MEASURED TABLE. Two closed forms have now been
			 * fitted to this and both were refuted by the next
			 * round: wbytes/4 in 284, and "the last block takes the
			 * odd slots" in 286, which put g4 of N = 56 at 1024
			 * where --map found it at 704.
			 *
			 * The skeleton IS regular and it is K independent.
			 * Groups of eight channels sit in BLOCKS of 8*K bytes at
			 * SLOTS of 64, with a channel's k+ half four slots on,
			 * and the K = 32 table decomposes exactly like the
			 * K = 64 one, so only the block stride scales.
			 *
			 * WHICH slots is not regular. Written as (block, slot):
			 *
			 *   G=2  (0,0)(0,1)
			 *   G=3  (0,0)(0,2)(0,3)
			 *   G=4  (0,0)(0,2)(1,0)(1,1)
			 *   G=5  (0,0)(0,2)(1,0)(1,1)(1,3)
			 *   G=6  (0,0)(0,2)(1,0)(1,2)(2,0)(2,1)
			 *   G=7  (0,0)(0,2)(1,0)(1,2)(1,3)(2,1)(2,3)
			 *   G=8  (0,0)(0,2)(1,0)(1,2)(2,0)(2,2)(3,0)(3,1)
			 *
			 * G=5's three group block is {0,1,3} and G=7's is
			 * {0,2,3}; G=7's last block is {1,3} where every other
			 * last block of two is {0,1}. Any rule covering both is
			 * a rule fitted to one point each, which is exactly what
			 * 284 and 286 were.
			 *
			 * So this is the measurement, from --map sweeps of the
			 * whole buffer at every N that is a multiple of 8 up to
			 * 64, and it refuses anything else. G = 9 and above have
			 * never been swept.
			 */
			static const unsigned char tab[9][8][2] = {
				{{0}}, {{0}},
				{ {0,0},{0,1} },
				{ {0,0},{0,2},{0,3} },
				{ {0,0},{0,2},{1,0},{1,1} },
				{ {0,0},{0,2},{1,0},{1,1},{1,3} },
				{ {0,0},{0,2},{1,0},{1,2},{2,0},{2,1} },
				{ {0,0},{0,2},{1,0},{1,2},{1,3},{2,1},{2,3} },
				{ {0,0},{0,2},{1,0},{1,2},{2,0},{2,2},{3,0},{3,1} },
			};
			unsigned g = n / 8;
			unsigned ngrp = DIV_ROUND_UP(mm->n, 8);
			size_t row;

			if (ngrp < 2 || ngrp > 8)
				return;                 /* never swept */

			row = (size_t)tab[ngrp][g][0] * 8 * mm->k
			      + (size_t)tab[ngrp][g][1] * 64
			      + (size_t)(n % 8) * 8;
			unsigned half;

			/*
			 * half 0 is the group at row, half 1 the one 256 bytes
			 * on, which exists only when K is 64. Each carries 16
			 * nibbles, and the k they carry is 16*(n&1) + 32*half.
			 */
			for (half = 0; half < (mm->k > 32 ? 2u : 1u); half++) {
				/* four slots on, which is half a block */
				size_t gb = row + (size_t)half * 4 * mm->k;
				unsigned j;

				for (j = 0; j < 16; j++) {
					unsigned kk = 16 * (n & 1) + 32 * half
						      + j;
					unsigned nib;
					size_t b;
					unsigned high;

					if (kk >= mm->k)
						continue;
					nib = src[(size_t)n * mm->k + kk] & 0xf;
					if (order) {
						b = gb + (j % 8);
						high = j >= 8;
					} else {
						b = gb + j / 2;
						high = j & 1;
					}
					if (high)
						dst[b] |= (uint8_t)(nib << 4);
					else
						dst[b] |= (uint8_t)nib;
				}
			}
		}
		return;
	}

	for (n = 0; n < mm->n; n++) {
		unsigned ngi = n / ng, ngsz = MIN2(n_pad - ngi * ng, ng);

		/*
		 * THE GROUPS ARE LAID OUT AGAINST THE PADDED K. The input
		 * surface is already padded to the feature atom by the memset
		 * in charsiu_pack_input, and round 195 measured that every K
		 * which is a multiple of 16 is byte exact and every K which is
		 * not is wrong, 40 included, so it is not 8 either. The weights
		 * for the padding stay at the weight zero point that the memset
		 * above wrote, so they meet the padded inputs' zero point and
		 * contribute nothing.
		 */
		for (k = 0; k < mm->k; k++) {
			unsigned kgi = k / kg, kgsz = MIN2(ke - kgi * kg, kg);
			size_t off = (size_t)ngi * ng * ke
				   + (size_t)kgi * kg * ngsz
				   + (size_t)(n % ng) * kgsz
				   + (k % kg);

			/*
			 * STORED BIASED BY -0x80, as a signed byte.
			 *
			 * The MAC computes sum(in_stored * w_stored) with BOTH
			 * operands taken as signed bytes, so each has to carry
			 * its own subtraction. Round 150 established that for
			 * the weight; round 161 established the same for the
			 * input, which this comment used to say was taken raw
			 * and was wrong about for eleven rounds. Storing
			 * the raw uint8 instead makes a weight at the zero
			 * point, 128, read as -128, and every dead tap in an
			 * impulse then contributes the largest negative value
			 * a byte has.
			 *
			 * Board, round 150: with the raw form, an impulse whose
			 * every output channel has exactly one live weight came
			 * back all zeros, because the live taps were swamped
			 * and clamped. Mesa's line, one file over, is
			 * weights_out[n] = weights_in[...] - 0x80.
			 */
			dst[off] = (uint8_t)(src[(size_t)n * mm->k + k] - 0x80);
		}
	}
}
