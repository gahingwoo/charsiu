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
#include "envq.h"

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
#include <arm_neon.h>
/*
 * float to half FOUR AT A TIME, and it TRUNCATES, exactly as
 * charsiu_float_to_half does.
 *
 * ⚠ NOT vcvt_f16_f32. The hardware instruction rounds to nearest even and this
 * project's converter drops the low thirteen mantissa bits, which is a
 * different number by up to half a last place -- about 5e-4 relative in half
 * precision, four thousand times the gap that moved a token in the SiLU
 * experiment. Rounding would very likely be the BETTER choice and it is not
 * this change's business to make it: this is the same arithmetic, faster.
 *
 * Checked against the scalar converter over 464 million float bit patterns
 * spanning the whole 32 bit space: zero differ.
 */
static inline uint16x4_t charsiu_vhalf(float32x4_t x)
{
	uint32x4_t u = vreinterpretq_u32_f32(x);
	uint32x4_t sign = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(0x8000));
	int32x4_t exp = vsubq_s32(vreinterpretq_s32_u32(
					  vandq_u32(vshrq_n_u32(u, 23),
						    vdupq_n_u32(0xff))),
				  vdupq_n_s32(112));
	uint32x4_t man = vandq_u32(u, vdupq_n_u32(0x7fffff));
	uint32x4_t h = vorrq_u32(sign,
			 vorrq_u32(vshlq_n_u32(vreinterpretq_u32_s32(exp), 10),
				   vshrq_n_u32(man, 13)));

	h = vbslq_u32(vcleq_s32(exp, vdupq_n_s32(0)), sign, h);
	h = vbslq_u32(vcgeq_s32(exp, vdupq_n_s32(0x1f)),
		      vorrq_u32(sign, vdupq_n_u32(0x7c00)), h);
	return vmovn_u32(h);
}

/*
 * The int8 activation bias, SIXTEEN BYTES AT A TIME, and it is one XOR.
 *
 * charsiu_pack_input writes (uint8_t)(v - 0x80) for every element. On eight bit
 * two's complement that is exactly v ^ 0x80 -- subtracting 128 modulo 256 only
 * ever flips the top bit -- for all 256 values, which is what lets the whole
 * bias be a single veorq_u8 rather than a widen, subtract and narrow. The
 * benchmark checks it the honest way anyway, against the untouched scalar path
 * over every value the sweep generates.
 *
 * `n` is the granularity of the layout, 16 for int8 by default, so the common
 * call is one load, one xor and one store.
 */
static inline void charsiu_bias_copy(uint8_t *d, const uint8_t *s, unsigned n)
{
	unsigned j = 0;

	for (; j + 16 <= n; j += 16)
		vst1q_u8(d + j, veorq_u8(vld1q_u8(s + j), vdupq_n_u8(0x80)));
	if (j + 8 <= n) {
		vst1_u8(d + j, veor_u8(vld1_u8(s + j), vdup_n_u8(0x80)));
		j += 8;
	}
	for (; j < n; j++)
		d[j] = (uint8_t)(s[j] - 0x80);
}

/*
 * The same, into the HIGH BYTE of a 16 bit slot, which is what int4 weights
 * make the activation surface look like (CHARSIU_A8_STRIDE1 is the control that
 * turns this case back into the plain one).
 *
 * The scalar path leaves the low byte at whatever the whole buffer memset put
 * there, which for this case is zero; st2 with a zero vector in lane 0 writes
 * the same bytes and lets the memset below `base` be dropped entirely.
 */
static inline void charsiu_bias_copy_hi(uint8_t *d, const uint8_t *s, unsigned n)
{
	unsigned j = 0;

	for (; j + 16 <= n; j += 16) {
		uint8x16x2_t p;

		p.val[0] = vdupq_n_u8(0);
		p.val[1] = veorq_u8(vld1q_u8(s + j), vdupq_n_u8(0x80));
		vst2q_u8(d + (size_t)j * 2, p);
	}
	if (j + 8 <= n) {
		uint8x8x2_t p;

		p.val[0] = vdup_n_u8(0);
		p.val[1] = veor_u8(vld1_u8(s + j), vdup_n_u8(0x80));
		vst2_u8(d + (size_t)j * 2, p);
		j += 8;
	}
	for (; j < n; j++) {
		d[(size_t)j * 2] = 0;
		d[(size_t)j * 2 + 1] = (uint8_t)(s[j] - 0x80);
	}
}
#endif

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
	if (mm->wdtype == CHARSIU_INT4 && !envq("CHARSIU_A8_STRIDE1"))
		return CHARSIU_FP16;    /* any 2 byte type: the atom is 8 */
	return mm->adtype;
}

/*
 * How many channel atoms fit in one CBUF entry.
 *
 * ⚠ MESA SAYS EIGHT AND THIS TREE HAS ALWAYS SAID FOUR. Same function, same
 * shape, one constant apart: rkt_task.c's calc_entries_per_slice() divides by
 * CBUF_ENTRY_SIZE / FEATURE_ATOMIC_SIZE = 128 / 16 = 8, and this divides by 4,
 * which is a 64 byte entry.
 *
 * At K=256 int8 that is 16 atoms, so Mesa gets surf = 2 and this gets 4 --
 * exactly double, at every K that is a multiple of eight atoms. surf is not a
 * decorative number: it goes into 0x1028 as `surf * rows`, so at M = 1 the
 * error is a factor of two on one word the hardware evidently tolerates, and
 * at M = 32 it is a factor of two on the stride between thirty two staged
 * rows. Every row after the first would then be read from the wrong place,
 * which produces WRONG VALUES rather than misplaced ones -- and the board's
 * m>1 output is wrong values: only 27% of the wanted numbers appear anywhere
 * in it.
 *
 * Mesa's generic RK3576 encoder is the one this board ran M = 1, 2, 3, 4 and 8
 * through exactly. The default stays 4 because that is what every correct
 * result in this tree was measured with; CHARSIU_ENTRY_ATOMICS=8 is the
 * question, and npu_gemm_test asks it.
 */
/*
 * ⚠ NOT CACHED IN A STATIC. npu_gemm_test sweeps this between phases in one
 * process, and a value latched on the first matmul would make the second phase
 * a silent copy of the first -- which is exactly how round 380's control ran
 * twice with the same geometry and nobody noticed until the log was read.
 */
static unsigned entry_atomics(void)
{
	const char *e = envq("CHARSIU_ENTRY_ATOMICS");
	int v = e ? atoi(e) : 4;

	return (v == 8) ? 8u : 4u;
}

unsigned charsiu_entries_per_row(const struct charsiu_matmul *mm)
{
	unsigned atom = charsiu_feature_atom(charsiu_effective_adtype(mm));
	unsigned ape = entry_atomics();
	unsigned total = DIV_ROUND_UP(mm->k, atom);
	unsigned last = total % ape;
	unsigned width = 1;             /* one column, always */

	return (total / ape) * width +
	       (last == 3 ? width : DIV_ROUND_UP(last * width, ape));
}

unsigned charsiu_k_eff(const struct charsiu_matmul *mm)
{
	if (envq("CHARSIU_NO_KALIGN"))
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
	 * THE PRECISION REGISTER, and this was two constants copied out of the
	 * vendor's int4 and fp16 streams with "emitting 0 here is known wrong"
	 * written beside them. Both halves of that are refuted by the vendor's
	 * own file, counted over every convolution in
	 * Llama-3.2-1B-rk3576-w4a16 (2026-09-05):
	 *
	 *   int8 weights                       0x00000000    40
	 *   int4 weights                       0x00600120  1408
	 *   int4 weights, 16 bit activations   0x20600120  1920
	 *   fp16 weights, 16 bit activations   0x20200120  4940
	 *
	 * So zero is what int8 carries, in all 40 of them, and job.c has had
	 * that decoded for a while -- this emitter kept sending an int8 dump
	 * the FP16 constant, which makes an int8 stream diff against the vendor
	 * show a difference that is this line's and not the stream's. It also
	 * sent plain 0x00600120 for int4 at 16 bit activations, which is
	 * charsiu's own w4a16 shape and 1920 of the vendor's dispatches.
	 *
	 * ⚠ CHECKED AGAINST THE FILE, NOT AGAINST emit_dump. The note further
	 * down this function is about exactly that mistake: emit_dump IS this
	 * function, so it cannot fail a check on it. The four rows above are
	 * read out of the .rkllm with tools/rkllm_regcmd.py, and the four cases
	 * below reproduce all four.
	 */
	{
		uint32_t prec = mm->wdtype == CHARSIU_INT4 ? 0x00600120u
			      : mm->wdtype == CHARSIU_FP16 ? 0x00200120u
			      : 0x00000000u;

		if (charsiu_effective_adtype(mm) == CHARSIU_FP16)
			prec |= 0x20000000u;
		emit(&e, CNA, 0x100c, prec);
	}
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
	 *
	 * ⚠⚠ AND THE RUNTIME DOES NOT COME THROUGH HERE. charsiu_emit_matmul
	 * has exactly one caller in the tree, tools/emit_dump.c. npudev submits
	 * through charsiu_emit_job in job.c, which HAS the split rule --
	 * `split = wide && surf * rows > 4096`, derived from the same vendor
	 * file and exact on all 3328 of its int4 streams.
	 *
	 * This note is here because I added the rule to this function instead,
	 * called it the fix for a wide-K fault, and verified it with emit_dump
	 * -- which is this same function, so the check could not have failed.
	 * If you are about to change a register here, first check whether the
	 * thing you are fixing runs through job.c.
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
	/*
	 * ⚠ THE ROW TERM IN THIS LAYOUT HAS NEVER BEEN TESTED. Every matmul in
	 * this project ran M = 1 until round 289, where m drops out of the
	 * expression entirely and any arrangement of the rows is correct.
	 *
	 * Rounds 293 to 295 narrowed M > 1 to one thing: both output rows are
	 * computed from ROW 0's activation slot and row 1's bytes are never
	 * fetched. Six registers were swept for it, three of which light row 1
	 * on --map, and NONE of them makes it read its own data. So the place
	 * left is the arrangement itself.
	 *
	 * ⚠ ROUND 296 RAN THREE ARRANGEMENTS AND THEY ARE THREE POINTS OF ONE
	 * FAMILY. What separates them is the GRANULARITY at which rows
	 * interleave: the shipped layout switches rows every atom, 8 elements
	 * and 16 bytes, and the "rows outermost" one switches every k, the whole
	 * row. Both are ends of the same axis and nothing between them has ever
	 * been tried.
	 *
	 * 32 elements is the one with a reason behind it. charsiu_entries_per_row
	 * returns 2 at K = 64 and the input bo is allocated surf * 64 * m, so an
	 * "entry" is 64 bytes, which at 2 bytes an element is 32 of them. Rows
	 * interleaving at the entry rather than the atom is the natural reading
	 * of that allocation and it has never been written.
	 *
	 * CHARSIU_A_GRAN is that axis. g elements of row 0, then g of row 1, and
	 * so on:  e = (kk/g)*m*g + i*g + kk%g
	 *
	 *   g = atom (default)   the shipped layout, 296's layout 0
	 *   g = k                rows outermost, 296's layout 1
	 *   g = 32               rows per 64 byte entry, never tried
	 *
	 * CHARSIU_A_LAYOUT still picks the two non family arrangements:
	 *   0 (default)  the granularity family above
	 *   2            [k/atom][atom][m], rows innermost, element interleaved
	 */
	{
		unsigned lay = envq("CHARSIU_A_LAYOUT")
			? (unsigned)atoi(envq("CHARSIU_A_LAYOUT")) : 0;
		unsigned gran = envq("CHARSIU_A_GRAN")
			? (unsigned)atoi(envq("CHARSIU_A_GRAN")) : atom;

		if (!gran)
			gran = atom;

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
		{
		/*
		 * ⚠⚠ THE INT8 PACKER'S COST IS NOT A CALL. IT IS A DIVIDE, AND
		 * A RELOAD OF THE STRUCT, PER ELEMENT.
		 *
		 * The fp16 packer above was fixed by removing a cross unit call
		 * to charsiu_float_to_half, and the obvious assumption is that
		 * this one has the same fault. It does not. `nm -u` on regcmd.o
		 * names charsiu_float_to_half, getenv, memset and strtol and
		 * nothing else, and objdump of the scalar loop at the bottom of
		 * this function has no `bl` in it at all. What it does have, per
		 * element, is
		 *
		 *   udiv + msub         kk / gran and kk % gran. gran comes
		 *                       from getenv, so unlike the fp16
		 *                       packer's literal 8 it cannot be turned
		 *                       into shifts, and a real divide is what
		 *                       is emitted
		 *   madd x3, umaddl     the rest of the index
		 *   ldp w3, w5, [x19]   mm->m and mm->k RELOADED FROM THE
		 *                       STRUCT after every single store,
		 *                       because dst is a uint8_t * and the
		 *                       compiler has to assume it can alias *mm
		 *   cmp w27, #2 / b.ne  the CHARSIU_A_LAYOUT test, sitting
		 *                       inside the innermost loop
		 *   ldrb / sturb        one byte in, one byte out
		 *
		 * which is why the untouched path sits at a flat 1.2 to 1.3
		 * GB/s at every shape: it is not touching memory hard enough to
		 * be near any memory limit, it is executing that.
		 *
		 * The bias is one instruction. The scalar path writes
		 * (uint8_t)(v - 0x80), and on eight bits subtracting 128 modulo
		 * 256 only ever flips the top bit, so the whole thing is
		 * veorq_u8 with 0x80 -- see charsiu_bias_copy at the top of
		 * this file.
		 *
		 * Measured on this aarch64 host, cold destination, best of ten
		 * processes with the two paths interleaved so the machine's
		 * other work hits both equally, GB/s of destination bytes:
		 *
		 *   k=2048 m=2     1.23 -> 20.70 GB/s   16.9x
		 *   k=2048 m=32    1.29 -> 37.55 GB/s   29.2x
		 *   k=2048 m=64    1.29 -> 37.06 GB/s   28.8x
		 *   k=1024 m=32    1.28 -> 26.35 GB/s   20.6x
		 *   k=1024 m=64    1.29 -> 38.08 GB/s   29.6x
		 *   k=768  m=80    1.28 -> 39.21 GB/s   30.6x   the tower shape
		 *   k=3072 m=16    1.29 -> 40.20 GB/s   31.1x
		 *   k=2049 m=64    1.30 -> 29.24 GB/s   22.5x   k % 16 != 0
		 *   k=2048 m=1     1.21 -> 10.99 GB/s    9.1x   decode
		 *   k=4096 m=1     1.23 -> 16.08 GB/s   13.1x   decode
		 *
		 * ⚠ A BIGGER RATIO THAN THE FP16 ROUND'S 4.3x DOES NOT MEAN A
		 * FASTER FUNCTION. int8 writes one byte an element where fp16
		 * writes two, so 37 GB/s here is 37 G elements a second against
		 * fp16's 24.5 GB/s = 12.3 G. The ratio is large because the
		 * baseline was much worse, and it was worse because of the
		 * divide.
		 *
		 * ⚠ GROUPS OUTERMOST, for the reason the fp16 packer's comment
		 * gives at length: the destination is a cached write back DRM
		 * mapping that PREP_BO has just synced for the CPU, so it is
		 * cold every time, and walking rows outermost touches every
		 * line of it before filling any of it. Benchmarked on a 96 MB
		 * ring of destinations so each timed call really does meet a
		 * cold one.
		 *
		 * ⚠ AND THE RING HAS TO BE WALKED ALL THE WAY ROUND. A fixed
		 * iteration count over a 4 KB destination touches 1.6 MB of the
		 * ring, which fits in cache and reports a warm number as if it
		 * were cold. The iteration count is the slot count.
		 *
		 * ⚠ THE WHOLE BUFFER MEMSET WAS THE OTHER SUSPECT AND IT IS
		 * WORTH NOTHING. Putting it back in front of the fast path,
		 * best of ten, gives 36.9 against 36.2 at k=2048 m=32, 30.7
		 * against 36.3 at m=64, 31.5 against 26.5 at k=1024 m=32, 17.7
		 * against 21.8 at k=2048 m=2: no consistent sign, +-20%, noise.
		 * A memset runs at 60 to 80 GB/s here and it WARMS the
		 * destination the pack is about to write, which very nearly
		 * pays for itself. Only the tail is cleared below anyway,
		 * because that costs nothing and matches the fp16 packer, but
		 * the speed in the table above is the vectorisation and the
		 * blocking, not the memset.
		 *
		 * ⚠ m = 1 IS ON THIS PATH, and m = 1 is int8 DECODE. At m = 1
		 * the offset collapses to kk for every layout in this function,
		 * so the destination is one flat biased copy; those are the two
		 * decode rows above. Decode is this runtime's headline number,
		 * so it is in the 189168 shape sweep rather than assumed.
		 *
		 * ⚠ AND THE PROOF NEEDS TWO PROCESSES. `plain` is cached in the
		 * static below on first use, so setting CHARSIU_NPU_PLAIN
		 * between two calls in one program compares this path against
		 * ITSELF and passes vacuously. Dump every packed buffer twice,
		 * once with the variable and once without, and diff the files:
		 * 94080 shapes, 1.98 GB of packed bytes, identical. That check
		 * was itself checked by flipping the 0x80 in charsiu_bias_copy
		 * to 0x81 -- with which the two dumps DIFFER, so the variable
		 * really does switch paths and the diff really can fail.
		 *
		 * NOT CHANGED, and measured: the three getenv calls this
		 * function makes per invocation. At k=2048 m=2 they are about
		 * half the remaining time (30.2 GB/s without them against 11.3
		 * with, on the same loop), and they matter to nothing at m=32
		 * and above. Latching them in statics would buy that back and
		 * would also make CHARSIU_A_LAYOUT and CHARSIU_A_GRAN dead to
		 * any probe that sweeps them mid process -- which is exactly
		 * the trap entry_atomics() above has a comment about. Left
		 * live.
		 *
		 * Left scalar on purpose: CHARSIU_A_LAYOUT=2. It puts rows
		 * innermost, [k/atom][atom][m], so consecutive k land m bytes
		 * apart and there is no contiguous run to vectorise -- it would
		 * be a scatter, which is the scalar store it already is. It is
		 * an env selected control no product path takes, so it keeps
		 * the old loop and goes on being the control.
		 */
		static int plain = -1;

		if (plain < 0)
			plain = envq("CHARSIU_NPU_PLAIN") != NULL;

		if (!plain && lay != 2) {
			unsigned m = mm->m, k = mm->k;
			unsigned ng = k / gran, tail = k % gran;
			uint8_t fill = wide
				     ? 0 : (uint8_t)(input_zero_point - 0x80);
			size_t base;
			unsigned g;

			/*
			 * ⚠ ONLY THE TAIL IS CLEARED. Every byte below `base`
			 * is written by the loops underneath -- for the 16 bit
			 * slot case the low bytes too, which is why
			 * charsiu_bias_copy_hi stores an explicit zero lane
			 * rather than relying on a memset that is no longer
			 * there. From `base` up is the padding the CBUF reads
			 * past the end of the data, plus the holes a k that is
			 * not a multiple of the granularity leaves in the last
			 * group, so `base` starts at that group.
			 */
			if (m == 1) {
				base = (size_t)k * esz;
				if (dst_size > base)
					memset(dst + base, fill,
					       dst_size - base);
				if (esz == 1)
					charsiu_bias_copy(dst, src, k);
				else
					charsiu_bias_copy_hi(dst, src, k);
				return;
			}

			base = (size_t)ng * m * gran * esz;
			if (dst_size > base)
				memset(dst + base, fill, dst_size - base);
			/*
			 * ⚠⚠ TWO GROUPS AT A TIME, AND THIS SECOND STEP IS
			 * WORTH MORE THAN THE VECTOR STORE WAS.
			 *
			 * A group is 16 elements, so one pass over the rows
			 * reads 16 bytes out of each row and then skips k. That
			 * is 16 bytes out of a 64 byte line, coming back for
			 * the rest of the line on the next three groups, and it
			 * is m cache line lookups per group, m * k / 16 over
			 * the whole call. Taking two groups per pass halves
			 * that and turns the row body into straight line code
			 * instead of a loop. One group at a time was 21.6 GB/s
			 * at k = 2048 m = 64; two is 37.1.
			 *
			 * Blocks of 1, 2, 4 and 8 groups, best of five
			 * processes each, interleaved so the host's other work
			 * hits all of them equally, GB/s of destination:
			 *
			 *          b1     b2     b4     b8
			 *   2048x2   30.2   44.0   40.2   39.7
			 *   2048x32  31.2   39.6   34.8   26.8
			 *   2048x64  35.9   38.2   32.3   25.7
			 *   1024x32  31.2   39.2   38.7   27.9
			 *   1024x64  35.9   40.6   30.2   26.8
			 *   768x80   37.2   40.0   38.4   24.3
			 *   3072x16  40.8   34.6   34.9   36.3
			 *
			 * Two is best or tied at six of the seven and is what
			 * is here. Four and eight fall away because the
			 * destination becomes four and eight streams m * 16
			 * apart, which at m = 64 is 1 KB and starts colliding
			 * in the L1 sets.
			 *
			 * ⚠ AND THE LENGTH HAS TO BE A CONSTANT, which is why
			 * the int8 default is written out here rather than left
			 * to the generic path underneath. charsiu_bias_copy
			 * takes `gran` as a runtime argument, so gcc emits the
			 * 16 byte loop, the 8 byte remainder test and the
			 * scalar tail test around every single row. Same work,
			 * same one group at a time, same three getenv calls,
			 * runtime length against a compile time 16: 21.6
			 * against 33.7 GB/s at k = 2048 m = 64.
			 */
			if (esz == 1 && gran == 16) {
				size_t step = (size_t)m * 16;

				for (g = 0; g + 1 < ng; g += 2) {
					const uint8_t *s = src + (size_t)g * 16;
					uint8_t *d = dst + (size_t)g * step;

					for (i = 0; i < m; i++, d += 16) {
						const uint8_t *r = s
							+ (size_t)i * k;

						vst1q_u8(d,
							veorq_u8(vld1q_u8(r),
							  vdupq_n_u8(0x80)));
						vst1q_u8(d + step,
							veorq_u8(vld1q_u8(r + 16),
							  vdupq_n_u8(0x80)));
					}
				}
				for (; g < ng; g++) {
					const uint8_t *s = src + (size_t)g * 16;
					uint8_t *d = dst + (size_t)g * step;

					for (i = 0; i < m; i++, d += 16)
						vst1q_u8(d, veorq_u8(
						  vld1q_u8(s + (size_t)i * k),
						  vdupq_n_u8(0x80)));
				}
			} else {
				for (g = 0; g < ng; g++) {
					const uint8_t *s = src
							 + (size_t)g * gran;
					uint8_t *d = dst
						   + (size_t)g * m * gran * esz;

					if (esz == 1)
						for (i = 0; i < m;
						     i++, d += gran)
							charsiu_bias_copy(d,
							  s + (size_t)i * k,
							  gran);
					else
						for (i = 0; i < m;
						     i++, d += gran * 2)
							charsiu_bias_copy_hi(d,
							  s + (size_t)i * k,
							  gran);
				}
			}
			for (i = 0; i < m; i++)
				for (kk = 0; kk < tail; kk++) {
					uint8_t *pd = dst + base
						+ ((size_t)i * gran + kk) * esz;

					pd[esz - 1] = (uint8_t)(
						src[(size_t)i * k
						    + ng * gran + kk] - 0x80);
				}
			return;
		}
		}
#endif

		memset(dst, wide ? 0 : (uint8_t)(input_zero_point - 0x80),
		       dst_size);
		for (i = 0; i < mm->m; i++)
			for (kk = 0; kk < mm->k; kk++) {
				size_t e;

				switch (lay) {
				case 2:
					e = (size_t)(kk / atom) * atom * mm->m
					    + (size_t)(kk % atom) * mm->m + i;
					break;
				default:
					e = (size_t)(kk / gran) * mm->m * gran
					    + (size_t)i * gran + kk % gran;
					break;
				}
				dst[e * esz + esz - 1] =
					(uint8_t)(src[i * mm->k + kk] - 0x80);
			}
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
	charsiu_pack_input_f16_stride(mm, src, mm->k, dst, dst_size);
}

/*
 * ⚠⚠ THE SOURCE ROW STRIDE, WHICH IS WHERE A THIRD OF PREFILL'S PACKING WENT.
 *
 * A tensor is cut into K slices and each slice packs columns [k0, k0 + sk) of
 * every row. With no stride the caller has to gather those columns into a
 * contiguous scratch first, so an element is read from X, written to the
 * scratch, read back and written as a half: fourteen bytes moved to pack six.
 * The board was doing 730 MB of that per prompt on Qwen2.5-1.5B and packing
 * measured 2628 ms of an 8941 ms matmul entry.
 *
 * With the stride the loop reads X where it lies. The access pattern is
 * unchanged -- groups outermost, m strided reads a group, for the cold
 * destination reason above -- only the stride differs, so nothing about the
 * measurements in that comment moves.
 */
void charsiu_pack_input_f16_stride(const struct charsiu_matmul *mm,
				   const float *src, size_t src_stride,
				   uint8_t *dst, size_t dst_size)
{
	unsigned atom = charsiu_feature_atom(CHARSIU_FP16);
	unsigned i, kk;

	/*
	 * ONE ROW IS THE WHOLE OF DECODE, AND ONE ROW IS CONTIGUOUS.
	 *
	 * At m = 1 the offset collapses: (kk/atom)*atom + kk%atom is kk, so the
	 * destination is just k halves in order. A token packs about 463
	 * thousand of them -- every K slice of every projection, into both
	 * devices -- and round 368 left 10.6 ms a token unaccounted between
	 * what the stage table charges to a projection and what npudev
	 * measures inside it. A scalar call per element is the obvious
	 * suspect.
	 *
	 * Only the tail is cleared: the region past k is padding the hardware
	 * still reads, and everything before it is about to be overwritten.
	 */
	/*
	 * ⚠⚠ m = 1 IS ON THIS PATH TOO, WHICH IS NOT WHAT IT LOOKS LIKE. The
	 * diff that added this block deletes nothing, so it reads as an
	 * insertion that leaves decode alone -- and it is not: the branch
	 * below takes m == 1 through the vector converter as well. Decode is
	 * this runtime's headline number, so that had to be proved rather
	 * than assumed.
	 *
	 * ⚠ AND THE PROOF NEEDS TWO PROCESSES. `plain` is cached in a static
	 * on first use, so setting CHARSIU_NPU_PLAIN between two calls in one
	 * program compares this path against ITSELF and passes vacuously. A
	 * 315 shape check written that way came back "0 mismatched" while
	 * measuring nothing at all. Run the dump twice, once with the variable
	 * and once without, and diff the files: 315 shapes including m = 1,
	 * k from 1 to 3072 and every k % 8, byte for byte identical.
	 */
#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
	{
	static int plain = -1;

	if (plain < 0)
		plain = envq("CHARSIU_NPU_PLAIN") != NULL;
	if (mm->m == 1 && !plain) {
		size_t used = (size_t)mm->k * 2;

		if (dst_size > used)
			memset(dst + used, 0, dst_size - used);
		for (kk = 0; kk + 4 <= mm->k; kk += 4)
			vst1_u16((uint16_t *)(dst + (size_t)kk * 2),
				 charsiu_vhalf(vld1q_f32(src + kk)));
		for (; kk < mm->k; kk++) {
			uint16_t h = charsiu_float_to_half(src[kk]);

			dst[kk * 2] = (uint8_t)(h & 0xff);
			dst[kk * 2 + 1] = (uint8_t)(h >> 8);
		}
		return;
	}
	/*
	 * ⚠⚠ AND m > 1 IS THE BATCHED PATH, WHICH WAS LEFT SCALAR.
	 *
	 * The case above was vectorised because it is the whole of decode. The
	 * rows case underneath it is the whole of PREFILL, and it still went one
	 * element at a time. On the board a batched matmul at m = 32 over
	 * Llama-3.2-1B's 225 tensors spends 350 ms of 1811 ms packing -- 19% --
	 * against 169 ms actually waiting on the NPU.
	 *
	 * The disassembly named two costs and neither is arithmetic. `objdump
	 * -d` of the loop below shows a `bl charsiu_float_to_half` PER ELEMENT:
	 * the converter lives in job.c, the Makefile carries no -flto, so gcc
	 * cannot inline across the two units and every single half costs a call
	 * plus about ten integer instructions of index arithmetic and two
	 * `strb`. The memset ahead of it is the other: it zeroes the whole
	 * buffer, and the loop then overwrites almost all of what it zeroed.
	 *
	 * ⚠ THE FIRST BENCHMARK OF THIS PUT THE CONVERTER IN THE SAME UNIT and
	 * so measured a build that does not exist. With it inlined gcc
	 * auto-vectorises even a plain scalar loop, and a scalar rewrite that
	 * only hoists the index arithmetic came out fastest of everything tried:
	 * 27.3 GB/s at k = 2048, m = 32. Rebuilt the way the Makefile builds --
	 * separate objects, no -flto -- that same rewrite is 7.6 GB/s at that
	 * same shape, 1.3x, and nothing like a fix. The CALL is the cost, so
	 * removing the call is the fix, and that means the vector converter.
	 *
	 * ⚠⚠ THE LOOP ORDER IS NOT FREE, AND ONLY A COLD DESTINATION SHOWS IT.
	 *
	 * A group of 8 consecutive k is 16 contiguous destination bytes, so
	 * either loop can be the outer one. Walking ROWS outermost keeps the
	 * source sequential and writes 16 bytes every m * 16; walking GROUPS
	 * outermost writes the destination straight through and reads m strided
	 * sources. Warm they are indistinguishable, 26.4 against 26.5 GB/s.
	 *
	 * Cold they are not, because the destination is ordinary cached memory
	 * and a 16 byte store to a cold line still fetches the line. rocket
	 * builds these buffers with drm_gem_shmem_create() and never sets
	 * map_wc, so the mmap is write-back, and PREP_BO invalidates the input
	 * BO immediately before this loop runs -- which is to say the board
	 * always meets a cold one. Cycling a 96 MB ring of destinations so every
	 * timed call gets a fresh buffer, at k = 2048 m = 64, rows-outermost
	 * falls to 6.87 GB/s while groups-outermost holds 24.57. Groups
	 * outermost is best or tied in every shape measured, warm and cold, so
	 * it is the one here. Unrolling it four rows deep was worth nothing
	 * (24.53 against 24.57) and is not carried.
	 *
	 * Measured on this aarch64 host, cold destination, before -> after:
	 *
	 *   k=2048 m=2    0.0043 -> 0.0010 ms    5.70 ->  24.80 GB/s   4.35x
	 *   k=2048 m=32   0.0723 -> 0.0160 ms    5.44 ->  24.55 GB/s   4.51x
	 *   k=2048 m=64   0.1488 -> 0.0320 ms    5.29 ->  24.57 GB/s   4.65x
	 *   k=1024 m=32   0.0338 -> 0.0079 ms    5.81 ->  24.86 GB/s   4.28x
	 *   k=1024 m=64   0.0687 -> 0.0160 ms    5.73 ->  24.59 GB/s   4.30x
	 *   k=2048 m=1    0.0005 ms, 24.64 GB/s, untouched
	 *
	 * ⚠ THE VALUES ARE THE SAME VALUES. charsiu_vhalf truncates exactly as
	 * charsiu_float_to_half does, which was checked again here over all
	 * 4294967296 float bit patterns, eight threads, ZERO differing --
	 * denormals, both infinities, quiet and signalling NaN, 65504 and the
	 * pattern one ulp past it, and the truncation boundary at 0x3f801fff /
	 * 0x3f802000. Nothing in this rounds where the old path truncated.
	 *
	 * ⚠ ONLY THE TAIL IS CLEARED, for the reason the m = 1 case gives: with
	 * k a multiple of the atom every byte below `base` is written by the
	 * loop, and everything from `base` up is padding the CBUF reads past the
	 * end of the data. A k that is NOT a multiple of the atom leaves holes
	 * in the last group, so `base` starts at that group and it is zeroed
	 * too. CHARSIU_NPU_PLAIN drops through to the scalar loop below, which
	 * is unchanged and remains the control for all of this.
	 */
	if (mm->m > 1 && !plain && atom == 8) {
		unsigned m = mm->m, ng = mm->k / atom, tail = mm->k % atom;
		size_t base = (size_t)ng * m * atom * 2;
		unsigned g;

		if (dst_size > base)
			memset(dst + base, 0, dst_size - base);
		for (g = 0; g < ng; g++) {
			const float *s = src + (size_t)g * atom;
			uint8_t *d = dst + (size_t)g * m * atom * 2;

			for (i = 0; i < m; i++, d += 16) {
				const float *r = s + (size_t)i * src_stride;

				vst1q_u16((uint16_t *)d,
					  vcombine_u16(charsiu_vhalf(vld1q_f32(r)),
						       charsiu_vhalf(vld1q_f32(r + 4))));
			}
		}
		for (i = 0; i < m; i++)
			for (kk = 0; kk < tail; kk++) {
				uint16_t h = charsiu_float_to_half(
					src[(size_t)i * src_stride + ng * atom + kk]);
				uint8_t *p = dst + base + (size_t)i * 16
					   + (size_t)kk * 2;

				p[0] = (uint8_t)(h & 0xff);
				p[1] = (uint8_t)(h >> 8);
			}
		return;
	}
	}
#endif
	memset(dst, 0, dst_size);
	for (i = 0; i < mm->m; i++)
		for (kk = 0; kk < mm->k; kk++) {
			size_t off = ((size_t)(kk / atom) * mm->m * atom
				      + (size_t)i * atom + kk % atom) * 2;
			uint16_t h = charsiu_float_to_half(src[(size_t)i * src_stride + kk]);

			dst[off] = (uint8_t)(h & 0xff);
			dst[off + 1] = (uint8_t)(h >> 8);
		}
}

/*
 * ROWS [n0, n0 + nrows) ONLY, AND NO MEMSET.
 *
 * The two live layouts write disjoint bytes per output channel -- int4 puts
 * channel nn at (nn/16)*512*(kb/32) + (nn%16)*32 and writes whole bytes from
 * there, int8 indexes dst by n as well -- so a range of channels can be packed
 * without touching any other range's bytes. That is what lets staging split
 * over the pool: about 620 MB of nibbles for this model, 4.4 seconds of a cold
 * start on the board, on one core.
 *
 * ⚠ THE CALLER OWNS THE ZEROING, once, before any range runs. And the legacy
 * bit pattern layout is NOT in here: it accumulates with |=, so two channels
 * can share a byte and ranges would race. charsiu_pack_weights below keeps it.
 */
void charsiu_pack_weights_rows(const struct charsiu_matmul *mm,
			       const uint8_t *src, uint8_t *dst,
			       unsigned n0, unsigned nrows)
{
	unsigned ng = charsiu_weight_ngroup(mm->wdtype);
	unsigned kg = charsiu_weight_kgroup(mm->wdtype);
	unsigned n_pad = ALIGN_UP(mm->n, 2);
	unsigned ke = charsiu_k_eff(mm);
	unsigned n, k;

	if (mm->wdtype == CHARSIU_INT4) {
		size_t cap = charsiu_weight_bytes(mm);
		unsigned kb = ALIGN_UP(ke, 32);
		unsigned kmax = mm->k < ke ? mm->k : ke;
		unsigned nn, kk;

		for (nn = n0; nn < n0 + nrows; nn++) {
			size_t nbase = (size_t)(nn / 16) * 512 * (kb / 32)
				     + (size_t)(nn % 16) * 32;
			const uint8_t *row = src + (size_t)nn * mm->k;

			for (kk = 0; kk < kmax; kk += 32) {
				size_t i = nbase + (size_t)(kk / 32) * 512;
				unsigned run = kmax - kk < 32 ? kmax - kk : 32;
				uint8_t *d;
				unsigned c;

				if ((i + run - 1) >> 1 >= cap)
					break;
				d = dst + (i >> 1);
				for (c = 0; c + 1 < run; c += 2)
					*d++ = (uint8_t)((row[kk + c] & 0xf) |
							 ((row[kk + c + 1] & 0xf)
							  << 4));
				if (c < run)
					*d = (uint8_t)((*d & 0xf0) |
						       (row[kk + c] & 0xf));
			}
		}
		return;
	}

	for (n = n0; n < n0 + nrows; n++) {
		unsigned ngi = n / ng, ngsz = MIN2(n_pad - ngi * ng, ng);

		for (k = 0; k < mm->k; k++) {
			unsigned kgi = k / kg, kgsz = MIN2(ke - kgi * kg, kg);
			size_t off = (size_t)ngi * ng * ke
				   + (size_t)kgi * kg * ngsz
				   + (size_t)(n % ng) * kgsz
				   + (k % kg);

			dst[off] = (uint8_t)(src[(size_t)n * mm->k + k] - 0x80);
		}
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
	/*
	 * ⚠⚠ THE int4 LAYOUT, AS THE HARDWARE ACTUALLY FETCHES IT. Round 345.
	 *
	 * Everything in the block below this one describes how the hardware
	 * fetched while CORE 0x3018 was wrong and the DPU was multiplying fp16
	 * bit patterns as integers. It was measured honestly and it is not the
	 * layout of a machine that computes a weighted sum.
	 *
	 * This one was read off a SLOPE MAP: change one nibble, submit, and the
	 * difference in the output is that nibble's contribution, which is
	 * exactly its activation times its int4 value. Divide by that output
	 * word's float32 ulp -- per word, and sign flipped for a negative word,
	 * because a negative float's integer image runs backwards -- and the
	 * activation index falls out. 122 of 128 nibbles landed on
	 *
	 *     k = i mod 32,  channel = i / 32
	 *
	 * the six misses being small activations lost to float32 rounding. In
	 * closed form, with 16 channels and 32 k to a block:
	 *
	 *     nibble(n,k) = (n/16)*512*(K/32) + (k/32)*512 + (n%16)*32 + k%32
	 *
	 * which is exactly N*K/2 nibbles with nothing left over. Round 346 had
	 * the board check every live word of a real projection against a CPU
	 * reference: 1280 of 1280 at M = 32 under the vendor's own registers,
	 * and round 347 got 1024 of 1024 at M = 1, which is the decode case.
	 *
	 * Sixteen is the feature atom this project has met before, in the Mesa
	 * channel work and in 0x4050's parity.
	 *
	 * CHARSIU_W4_BITPAT selects the old layout, and it is the same switch
	 * that restores the old CORE registers, so the two cannot disagree.
	 */
	if (mm->wdtype == CHARSIU_INT4 && !envq("CHARSIU_W4_BITPAT")) {
		/*
		 * ⚠ THE BOUND AND THE DIVISIONS ARE HOISTED. The first version
		 * called charsiu_weight_bytes() inside the inner loop, which is
		 * one function call per nibble -- about 1.4 billion of them for
		 * this model -- and round 352's int4 arm never finished loading.
		 * The int8 branch below does one add per element, so the two
		 * were not remotely comparable.
		 *
		 * Blocks of 32 k for each of 16 channels, so walk the blocks and
		 * let the inner loop be a straight run of 32.
		 */
		size_t cap = charsiu_weight_bytes(mm);
		unsigned kb = ALIGN_UP(ke, 32);
		unsigned kmax = mm->k < ke ? mm->k : ke;
		unsigned nn, kk;

		for (nn = 0; nn < mm->n; nn++) {
			size_t nbase = (size_t)(nn / 16) * 512 * (kb / 32)
				     + (size_t)(nn % 16) * 32;
			const uint8_t *row = src + (size_t)nn * mm->k;

			for (kk = 0; kk < kmax; kk += 32) {
				size_t i = nbase + (size_t)(kk / 32) * 512;
				unsigned run = kmax - kk < 32 ? kmax - kk : 32;
				uint8_t *d;
				unsigned c;

				if ((i + run - 1) >> 1 >= cap)
					break;
				/*
				 * i is always EVEN here -- nbase and the block
				 * step are both multiples of 32 -- so a run
				 * writes whole bytes and never has to read one
				 * back. That matters: this runs over every
				 * weight in the model at load, about 1.2
				 * billion nibbles, and the read modify write
				 * version made round 352's int4 arm take
				 * minutes where int8 took 303 ms.
				 */
				d = dst + (i >> 1);
				for (c = 0; c + 1 < run; c += 2)
					*d++ = (uint8_t)((row[kk + c] & 0xf) |
							 ((row[kk + c + 1] & 0xf)
							  << 4));
				if (c < run)
					*d = (uint8_t)((*d & 0xf0) |
						       (row[kk + c] & 0xf));
			}
		}
		return;
	}

	if (mm->wdtype == CHARSIU_INT4) {
		unsigned order = envq("CHARSIU_INT4_ORDER")
			? (unsigned)atoi(envq("CHARSIU_INT4_ORDER")) : 0;

		/*
		 * K is refused outside 64 and 32 because those are the only two
		 * the fetch has been swept at. N is not capped at 64 any more:
		 * round 287 swept N = 72 and the group count is what the layout
		 * is indexed by, so the guard belongs on the group count and it
		 * is below.
		 */
		/*
		 * ⚠ K = 32 AND 64 ARE THE ONLY TWO EVER SWEPT, and an LLM wants
		 * K in the thousands. The skeleton is K independent where it has
		 * been checked: the slot table by group count reproduces both,
		 * and only the block stride 8*K and the k+ offset 4*K scale. So
		 * 128 and 256 are ALLOWED here as a prediction, not a reading,
		 * and refused above that until swept.
		 *
		 * At K = 128, N = 64 the prediction is bases 0, 128, 1024, 1152,
		 * 2048, 2176, 3072, 3136 with the k+ group 512 bytes on.
		 */
		/*
		 * ⚠ K MUST BE A MULTIPLE OF 32, which is the run count K/32
		 * being a whole number. Round 302 ran every multiple of 16
		 * between 128 and 192:
		 *
		 *   K = 144, K/32 = 4.5    8 of 64 exact
		 *   K = 160, K/32 = 5     64 of 64
		 *   K = 176, K/32 = 5.5    8 of 64 exact
		 *   K = 192, K/32 = 6     64 of 64
		 *
		 * ⚠ AND K = 192 WAS NEVER A LAYOUT FAULT. Rounds 300 and 301
		 * recorded it as "writes every channel and computes none, the
		 * first non power of two K", and it was this guard: 192 was not
		 * in the whitelist, so the packer returned without writing a
		 * byte. --map lit anyway because it writes raw bytes and does
		 * not go through here.
		 */
		/*
		 * ⚠ THE UPPER BOUND HERE IS MINE, NOT THE HARDWARE'S. 256 was
		 * the largest K anything had been run at, and K = 256 fails on
		 * the CHANNEL COUNT rather than the layout, so nothing has ever
		 * said the layout stops. Round 305 lifts it to 2048, which is
		 * the K a real projection wants, because round 304 measured what
		 * the small envelope costs: int4 is 7 to 19 percent faster than
		 * int8 inside it and a K = 2048 by N = 1024 projection needs
		 * about 133 jobs against int8's one, which is twelve times
		 * slower overall.
		 */
		if (mm->k < 32 || mm->k > 2048 || (mm->k & 31))
			return;

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
			 * ⚠ THE THIRD CLOSED FORM, and this one has eight
			 * measured tables behind it rather than one.
			 *
			 * Two died in the round after they were written:
			 * wbytes/4 in 284, which also broke three working
			 * geometries, and "the last block takes the odd slots"
			 * in 286, which put g4 of N = 56 at 1024 where --map
			 * found it at 704. Both were fitted to a single point.
			 *
			 * What changed is that --map has now swept the whole
			 * buffer at every N that is a multiple of 8 from 16 to
			 * 72, so there are eight tables, and G = 9 is what made
			 * the odd case fall out. Written as flat slot indices,
			 * block times 8 plus slot:
			 *
			 *   G=2  0 1              G=3  0 2 3
			 *   G=4  0 2 8 9          G=5  0 2 8 9 11
			 *   G=6  0 2 8 10 16 17   G=7  0 2 8 10 11 17 19
			 *   G=8  0 2 8 10 16 18 24 25
			 *   G=9  0 2 8 10 16 17 19 25 27
			 *
			 * EVEN G is plain: pairs at slots 0 and 2 in every block
			 * but the last, which takes 0 and 1.
			 *
			 * ODD G has one block of three at b3 = (G-1)/4, with
			 * slots 0 and 2 before it, slots 1 and 3 after it, and
			 * the three itself being {0,2,3} when G mod 4 is 3 and
			 * {0,1,3} when it is 1. Four points on b3, two on each
			 * arm of the mod 4, and two on the "after" blocks.
			 *
			 * A channel's k+ half is four slots on, and the whole
			 * skeleton is K independent: the K = 32 table decomposes
			 * the same way, only the block stride 8*K scales.
			 *
			 * It predicts G = 10 and G = 11 outright:
			 *   G=10  0 2 8 10 16 18 24 26 32 33
			 *   G=11  0 2 8 10 16 18 19 25 27 33 35
			 * and refuses above 16 blocks, where nothing is swept.
			 */
			unsigned g = n / 8;
			unsigned ngrp = DIV_ROUND_UP(mm->n, 8);
			unsigned blk, slot;
			size_t row;

			/*
			 * ⚠ THIS GUARD WAS READ AS A HARDWARE RESULT ONCE
			 * ALREADY. Round 306 ran K = 2048 at N = 512 and 1024
			 * and got exact 0, which is this line refusing: G is 64
			 * and 128 there. The same mistake as the K whitelist in
			 * rounds 300 and 301, where K = 192 was recorded as a
			 * layout fault for two rounds and was my own refusal.
			 *
			 * So it goes to 256 groups, which is N = 2048. The slot
			 * form was read at G up to 11 and has since held at 20
			 * and at 32, so it is extrapolation above that and says
			 * so here rather than by returning silently.
			 */
			if (ngrp < 2 || ngrp > 256)
				return;

			if (!(ngrp & 1)) {
				blk = g / 2;
				slot = (blk == ngrp / 2 - 1) ? (g & 1)
							     : (g & 1) * 2;
			} else {
				unsigned b3 = (ngrp - 1) / 4;

				if (g < 2 * b3) {
					blk = g / 2;
					slot = (g & 1) * 2;
				} else if (g < 2 * b3 + 3) {
					static const unsigned s3a[3] = {0,2,3};
					static const unsigned s3b[3] = {0,1,3};

					blk = b3;
					slot = (ngrp % 4 == 3)
						? s3a[g - 2 * b3]
						: s3b[g - 2 * b3];
				} else {
					unsigned r = g - (2 * b3 + 3);

					blk = b3 + 1 + r / 2;
					slot = (r & 1) ? 3 : 1;
				}
			}

			row = (size_t)blk * 8 * mm->k + (size_t)slot * 64
			      + (size_t)(n % 8) * 8;
			unsigned half;

			/*
			 * half 0 is the group at row, half 1 the one 256 bytes
			 * on, which exists only when K is 64. Each carries 16
			 * nibbles, and the k they carry is 16*(n&1) + 32*half.
			 */
			/*
			 * ⚠ THE GROUP COUNT IS K/32 AND THE SPACING IS A
			 * CONSTANT 256, and both were K = 64 coincidences.
			 *
			 * Round 298 swept K = 128 and the group bases landed
			 * exactly where 8*K predicted, so the block stride does
			 * scale. What did not was the rest: channel 0 is fed by
			 * FOUR eight byte runs there, at 0, 256, 512 and 768,
			 * where K = 64 has two at 0 and 256.
			 *
			 * The old code wrote half * 4 * K for the second run's
			 * offset, and 4*K is 256 exactly when K is 64. That is
			 * the same trap as 8*K against wbytes/4 in round 284: an
			 * expression fitted at the one K everything was measured
			 * at. The spacing is 256 at both K and the COUNT is what
			 * scales, K/32, which is 1, 2 and 4 at K of 32, 64 and
			 * 128, and matches the K/2 k per channel already known.
			 */
			for (half = 0; half < (mm->k / 32 ? mm->k / 32 : 1u);
			     half++) {
				size_t gb = row + (size_t)half * 256;
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
