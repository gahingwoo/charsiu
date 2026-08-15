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

unsigned charsiu_entries_per_row(const struct charsiu_matmul *mm)
{
	unsigned atom = charsiu_feature_atom(mm->adtype);
	unsigned total = DIV_ROUND_UP(mm->k, atom);
	unsigned last = total % 4;
	unsigned width = 1;             /* one column, always */

	return (total / 4) * width +
	       (last == 3 ? width : DIV_ROUND_UP(last * width, 4));
}

size_t charsiu_weight_bytes(const struct charsiu_matmul *mm)
{
	/* The CNA counts output channels in PAIRS and reads the weights for the
	 * whole pair, so an odd N is rounded up here and nowhere else. Board,
	 * round 139: doing this took every odd output channel count from wrong
	 * across its second tile to exact. */
	unsigned n = ALIGN_UP(mm->n, 2);

	switch (mm->wdtype) {
	case CHARSIU_INT4: return (size_t)mm->k * n / 2;
	case CHARSIU_INT8: return (size_t)mm->k * n;
	default:           return (size_t)mm->k * n * 2;
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
	unsigned atom = charsiu_feature_atom(mm->adtype);
	unsigned i, kk;

	memset(dst, (uint8_t)(input_zero_point - 0x80), dst_size);
	for (i = 0; i < mm->m; i++)
		for (kk = 0; kk < mm->k; kk++)
			dst[(kk / atom) * mm->m * atom + i * atom + kk % atom] =
				(uint8_t)(src[i * mm->k + kk] - 0x80);
}

void charsiu_pack_weights(const struct charsiu_matmul *mm,
			  const uint8_t *src, uint8_t *dst)
{
	unsigned ng = charsiu_weight_ngroup(mm->wdtype);
	unsigned kg = charsiu_weight_kgroup(mm->wdtype);
	unsigned n_pad = ALIGN_UP(mm->n, 2);
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
	 * INT4, READ OFF THE HARDWARE IN ROUNDS 167 AND 168.
	 *
	 * The tile is the int8 one with its k group doubled and the two halves
	 * of that group folded into the two nibbles of the same byte:
	 *
	 *   [n/32][k/64][n%32][k%64],  byte = k % 32,  nibble = (k % 64) / 32
	 *
	 * so a row is 32 bytes either way and the tile shape does not change
	 * with precision. Both numbers were measured rather than assumed.
	 *
	 * Round 167: low nibbles live and high nibbles live gave the IDENTICAL
	 * output across all 64 channels, which rules out every N pairing. Both
	 * nibbles of a byte are one output channel. So charsiu_weight_ngroup's
	 * 64 for int4, taken from the RK3588 notes, is wrong here and it is 32.
	 *
	 * Round 168: with the low nibbles live, an input impulse walking k moved
	 * the output for k = 0 to 31 and did nothing for k = 32 to 63. So the
	 * low nibble is the first half of the group and the high nibble the
	 * second, and NOT k with k+1 interleaved.
	 *
	 * EDGE TILES ARE EXTRAPOLATED, not measured. A k that is not a multiple
	 * of 64 gives a short row here, by the same rule the int8 path uses for
	 * a short k group. Nothing has been run at such a shape.
	 */
	if (mm->wdtype == CHARSIU_INT4) {
		for (n = 0; n < mm->n; n++) {
			unsigned ngi = n / ng, ngsz = MIN2(n_pad - ngi * ng, ng);

			for (k = 0; k < mm->k; k++) {
				unsigned kgi = k / kg;
				unsigned kgsz = MIN2(mm->k - kgi * kg, kg);
				unsigned kbytes = (kgsz + 1) / 2;
				unsigned kin = k % kg;
				size_t off = (size_t)ngi * ng * mm->k / 2
					   + (size_t)kgi * kbytes * ngsz
					   + (size_t)(n % ng) * kbytes
					   + kin % (kg / 2);
				unsigned nib = src[(size_t)n * mm->k + k] & 0xf;

				if (kin >= kg / 2)
					dst[off] |= (uint8_t)(nib << 4);
				else
					dst[off] |= (uint8_t)nib;
			}
		}
		return;
	}

	for (n = 0; n < mm->n; n++) {
		unsigned ngi = n / ng, ngsz = MIN2(n_pad - ngi * ng, ng);

		for (k = 0; k < mm->k; k++) {
			unsigned kgi = k / kg, kgsz = MIN2(mm->k - kgi * kg, kg);
			size_t off = (size_t)ngi * ng * mm->k
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
