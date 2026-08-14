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
#include <string.h>

#include "charsiu.h"

#define CNA   0x0201u
#define CORE  0x0801u
#define DPU   0x1001u
#define RDMA  0x2001u

#define DIV_ROUND_UP(n, d)  (((n) + (d) - 1) / (d))
#define ALIGN_UP(n, a)      (DIV_ROUND_UP((n), (a)) * (a))

struct emitter {
	uint64_t *out;
	size_t max;
	size_t n;
};

static void emit(struct emitter *e, unsigned target, unsigned reg, uint32_t val)
{
	if (e->n >= e->max) {
		e->n = e->max + 1;
		return;
	}
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
	r.offset = job->output_zero_point - 0x80;
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
static uint16_t float_to_half(float f)
{
	union { float f; uint32_t u; } v = { .f = f };
	uint32_t sign = (v.u >> 16) & 0x8000;
	int32_t exp = (int32_t)((v.u >> 23) & 0xff) - 127 + 15;
	uint32_t man = v.u & 0x7fffff;

	if (exp <= 0)
		return (uint16_t)sign;
	if (exp >= 0x1f)
		return (uint16_t)(sign | 0x7c00);
	return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

static size_t table_bytes(const struct charsiu_matmul *mm)
{
	return (size_t)DIV_ROUND_UP(mm->n, 8) * 64;
}

static size_t scale_table_bytes(const struct charsiu_matmul *mm)
{
	return (size_t)ALIGN_UP(mm->n, 8) * 2;
}

size_t charsiu_coef_bytes(const struct charsiu_matmul *mm)
{
	return table_bytes(mm) + scale_table_bytes(mm) + 4;
}

void charsiu_build_coefs(const struct charsiu_job *job, const int32_t *bias,
			 const int32_t *weight_sums, uint8_t *dst)
{
	const struct charsiu_matmul *mm = &job->mm;
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
		*a = bias[oc] - (job->input_zero_point - 0x80) * weight_sums[oc];
		*b = (int16_t)(0x80 - job->weight_zero_point);
		*c = 16;                /* per tensor: every channel at the max */
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

	scales = (uint16_t *)(dst + tb);
	for (oc = 0; oc < sb / 2; oc++)
		scales[oc] = float_to_half(job->weight_scale);

	/* The second operand. 0x1004 is fp16 0.00049, a value this hardware's
	 * own configuration accepts; the vendor's 0x0E0E does not, under ours. */
	*(uint32_t *)(dst + tb + sb) = 0x1004;
}

/* ---- the stream ----------------------------------------------------------- */

size_t charsiu_emit_job(const struct charsiu_job *job, uint64_t *out, size_t max)
{
	const struct charsiu_matmul *mm = &job->mm;
	struct emitter e = { out, max, 0 };
	struct requant rq = requant_of(job);
	unsigned n_pad = ALIGN_UP(mm->n, 2);
	unsigned surf = charsiu_entries_per_row(mm);
	unsigned rows = mm->m;
	unsigned lines = rows - 1;              /* the DPU's line count */
	size_t wbytes = charsiu_weight_bytes(mm);
	unsigned r;

	/* The S_POINTER each unit latches its geometry against. Every unit gets
	 * it, not just the CNA: mesa writes all four and the vendor's streams
	 * carry 0x0e in each. */
	emit(&e, CNA, 0x1004, 0x0000000e);
	emit(&e, CORE, 0x3004, 0x0000000e);
	emit(&e, DPU, 0x4004, 0x0000000e);
	emit(&e, RDMA, 0x5004, 0x0000000e);
	emit(&e, CNA, 0x1038, 0x00000007);
	emit(&e, CNA, 0x100c,
	     mm->wdtype == CHARSIU_INT4 ? 0x00600120u : 0x20200120u);
	emit(&e, CNA, 0x1010, 0x00000fff);
	emit(&e, CNA, 0x1014, (1u << 3) | 1u);
	emit(&e, CNA, 0x1018, 0x40000404);
	emit(&e, CNA, 0x101c, (uint32_t)wbytes);
	emit(&e, CNA, 0x1020, (uint32_t)(wbytes / n_pad));
	emit(&e, CNA, 0x1024, n_pad - 1);
	emit(&e, CNA, 0x1028, ((surf * rows) << 16) | (mm->k - 1));
	emit(&e, CNA, 0x102c, rows - 1);        /* (width - 1) << 16 is zero */
	emit(&e, CNA, 0x1030, ((uint32_t)(wbytes / n_pad) << 16) | 0);
	emit(&e, CNA, 0x1034, rows - 1);
	emit(&e, CNA, 0x103c, surf << 16);
	emit(&e, CNA, 0x1040, 0x10000000);
	emit(&e, CNA, 0x1044, (1u << 16) | surf);
	emit(&e, CNA, 0x1048, 0x0000000b);
	emit(&e, CNA, 0x104c, 0x00010001);
	emit(&e, CNA, 0x1050, 0x00010001);
	for (r = 0x1054; r <= 0x1074; r += 4)
		emit(&e, CNA, r, 0x00000000);
	emit(&e, CNA, 0x1078, rows - 1);
	emit(&e, CNA, 0x107c, mm->k - 1);
	emit(&e, CNA, 0x1080, 0x00000000);      /* a matmul has no padding */
	emit(&e, CNA, 0x1084, 0x00000000);
	emit(&e, CNA, 0x1088, job->input_addr);
	emit(&e, CNA, 0x108c, 0x000f000f);
	emit(&e, CNA, 0x1090, 1 * 4);           /* inw * 4 */
	emit(&e, CNA, 0x1094, rows);            /* inw * full_inh */
	emit(&e, CNA, 0x1098, rows);
	emit(&e, CNA, 0x109c, 0x00000000);
	emit(&e, CNA, 0x1100, 0x00000000);
	emit(&e, CNA, 0x1104, 0x00000000);
	emit(&e, CNA, 0x1110, job->weight_addr);
	emit(&e, CNA, 0x1140, 0x00000000);
	emit(&e, CNA, 0x1144, 0x00000000);
	emit(&e, CNA, 0x118c, rows - 1);        /* (inw - 1) << 16 is zero */

	emit(&e, CORE, 0x3018, 0x10000001);
	emit(&e, CORE, 0x301c, (lines << 16) | 0);
	emit(&e, CORE, 0x3020, mm->n - 1);
	emit(&e, CORE, 0x3024, 0x00000000);

	emit(&e, DPU, 0x400c, 0x40000004);
	emit(&e, DPU, 0x4010, 0x00000000);
	emit(&e, DPU, 0x4014, 0x00000000);
	emit(&e, DPU, 0x4018, job->output_addr);
	emit(&e, DPU, 0x401c, rows);            /* ow * full_oh */
	emit(&e, DPU, 0x4020, 0x00000000);      /* ow - 1 */
	emit(&e, DPU, 0x4024, lines);
	emit(&e, DPU, 0x4028, 0x00000000);
	emit(&e, DPU, 0x402c, mm->n - 1);
	emit(&e, DPU, 0x4030, ((mm->n - 1) << 16) | 0x0710);
	emit(&e, DPU, 0x4034, (lines << 16) | 0);
	emit(&e, DPU, 0x4038, 0x00120080);
	emit(&e, DPU, 0x403c, 0x00000000);
	emit(&e, DPU, 0x4044, 0x00000001);
	emit(&e, DPU, 0x4048, 0x80000000);
	emit(&e, DPU, 0x404c, 0x7fffffff);
	emit(&e, DPU, 0x4050, 0x80011111);
	emit(&e, DPU, 0x4058, 0x80000000);
	emit(&e, DPU, 0x405c, 0x7fffffff);
	emit(&e, DPU, 0x4060, 0x00000903);
	emit(&e, DPU, 0x406c, 0x80000000);
	emit(&e, DPU, 0x4070, 0x7fffffff);
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
	emit(&e, DPU, 0x40ac, (uint32_t)rq.offset);
	emit(&e, DPU, 0x40b0, rq.scale);
	emit(&e, DPU, 0x40b4, rq.shift);
	emit(&e, DPU, 0x40b8, 1 * (2 * rows - rows));   /* ow * (2*oh - window) */
	emit(&e, DPU, 0x40bc, 0x00000000);
	emit(&e, DPU, 0x40c0, 0x04440100);
	emit(&e, DPU, 0x40c8, 0x00000000);
	emit(&e, DPU, 0x40cc, 0x00000000);
	emit(&e, DPU, 0x40d0, 0x0040ffff);
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
	emit(&e, RDMA, 0x501c, 0x00000710);
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
	emit(&e, RDMA, 0x5034, 0x00000041);
	emit(&e, RDMA, 0x5038, 0x00000000);
	emit(&e, RDMA, 0x5040, 0x00000000);
	emit(&e, RDMA, 0x5044, 0x40000010);
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
	 * REVERSE data flow order so the downstream units are ready before the
	 * input DMA starts feeding them. The broadcast form, target 0x81 on the
	 * PC's own enable, restarts the program counter mid stream and engages
	 * the units before the geometry is committed, which runs them on an
	 * empty window. Both of those were established on hardware in the
	 * driver work; this is the form that computes.
	 */
	emit(&e, DPU, 0x4008, 0x0000001d);      /* output first */
	emit(&e, RDMA, 0x5008, 0x0000001d);
	emit(&e, CORE, 0x3008, 0x0000001d);     /* the MAC */
	emit(&e, CNA, 0x1008, 0x0000001d);      /* input LAST */

	return e.n > max ? 0 : e.n;
}
