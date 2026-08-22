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
#include <stdlib.h>
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
static int override_for(unsigned reg, uint32_t *out)
{
	const char *p = getenv("CHARSIU_OVERRIDE");
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
		const char *o = getenv("CHARSIU_OUT_OFF");

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

uint16_t charsiu_float_to_half(float f)
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
	const char *e = getenv("CHARSIU_COEF_ELEMS");
	size_t elems = e ? strtoul(e, NULL, 0) : (size_t)mm->k * mm->n;

	if (elems < 8192)
		elems = 8192;
	return table_bytes(mm) + elems * sizeof(float) + 0x100;
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
		   + (getenv("CHARSIU_NO_LIFT")
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
		*c = (int16_t)(getenv("CHARSIU_COEF_C")
			       ? atoi(getenv("CHARSIU_COEF_C")) : 16);
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
	int w4a16 = 0, w4_dpu = 0;
	unsigned wide8 = 0;
	unsigned rdma_mask;
	unsigned r;

	/* The S_POINTER each unit latches its geometry against. Every unit gets
	 * it, not just the CNA: mesa writes all four and the vendor's streams
	 * carry 0x0e in each. */
	emit(&e, CNA, 0x1004, 0x0000000e);
	emit(&e, CORE, 0x3004, 0x0000000e);
	emit(&e, CNA, 0x1038, 0x00000007);
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
	     (mm->wdtype == CHARSIU_INT4 ? 0x00600120u : 0x00000000u) |
	     (charsiu_effective_adtype(mm) == CHARSIU_FP16 ? 0x20000000u : 0u));
	emit(&e, CNA, 0x1010, 0x00000fff);
	emit(&e, CNA, 0x1014, (1u << 3) | 1u);
	emit(&e, CNA, 0x1018, 0x40000404);
	emit(&e, CNA, 0x101c, (uint32_t)wbytes);
	emit(&e, CNA, 0x1020, (uint32_t)(wbytes / n_pad));
	emit(&e, CNA, 0x1024, n_pad - 1);
	emit(&e, CNA, 0x1028, ((surf * rows) << 16) | (charsiu_k_eff(mm) - 1));
	emit(&e, CNA, 0x102c, rows - 1);        /* (width - 1) << 16 is zero */
	/*
	 * Bytes per kernel, DOUBLED for int8 and not for int4.
	 *
	 * The vendor's int4 projections carry it undoubled, which is where the
	 * "never doubled" reading came from, and Mesa's int8 stream for the very
	 * same shape carries twice the kernel size: 0x80 against our 0x40 at
	 * K=64. Both are right for their own precision.
	 */
	emit(&e, CNA, 0x1030,
	     ((uint32_t)(wbytes / n_pad * (mm->wdtype == CHARSIU_INT4 ? 1u : 2u))
	      << 16) | 0);
	emit(&e, CNA, 0x1034, rows - 1);
	/* Mesa writes 0x1038 a SECOND time here, after the geometry and before
	 * the surface stride, and that is the only ordering difference the
	 * stream diff had left. Matched rather than reasoned about: the value is
	 * the same both times, so the position is the only thing it can be
	 * carrying. */
	emit(&e, CNA, 0x1038, 0x00000007);
	emit(&e, CNA, 0x103c, surf << 16);
	emit(&e, CNA, 0x1040, 0x10000000);
	emit(&e, CNA, 0x1044, (1u << 16) | surf);
	emit(&e, CNA, 0x1048, 0x0000000b);
	emit(&e, CNA, 0x104c, 0x00010001);
	emit(&e, CNA, 0x1050, 0x00010001);
	for (r = 0x1054; r <= 0x1074; r += 4)
		emit(&e, CNA, r, 0x00000000);
	emit(&e, CNA, 0x1078, rows - 1);
	emit(&e, CNA, 0x107c, charsiu_k_eff(mm) - 1);
	emit(&e, CNA, 0x1080, 0x00000000);      /* a matmul has no padding */
	emit(&e, CNA, 0x1084, 0x00000000);
	emit(&e, CNA, 0x1088, job->input_addr);
	emit(&e, CNA, 0x108c, 0x000f000f);
	emit(&e, CNA, 0x1090, 1 * 4);           /* inw * 4 */
	emit(&e, CNA, 0x1094, rows);            /* inw * full_inh */
	/*
	 * Rounded UP to a multiple of four, which is Mesa's rule.
	 *
	 * This is the ONLY register where charsiu's whole stream differs from a
	 * vendor int8 convolution compiled at this exact shape, once addresses,
	 * this model's quantisation and the four op enables are set aside:
	 * charsiu writes 4 and the vendor writes 1. Mesa's value is the one
	 * proven on this path, so it stays the default, and CHARSIU_CNA_1098
	 * asks the board whether the difference matters.
	 */
	emit(&e, CNA, 0x1098, getenv("CHARSIU_CNA_1098")
	     ? (uint32_t)strtoul(getenv("CHARSIU_CNA_1098"), NULL, 0)
	     : ((rows * 1 + 3) & ~3u));
	emit(&e, CNA, 0x109c, 0x00000000);
	emit(&e, CNA, 0x1100, 0x00000000);
	emit(&e, CNA, 0x1104, 0x00000000);
	emit(&e, CNA, 0x1110, job->weight_addr);
	emit(&e, CNA, 0x1140, 0x00000000);
	emit(&e, CNA, 0x1144, 0x00000000);
	emit(&e, CNA, 0x118c, rows - 1);        /* (inw - 1) << 16 is zero */

	emit(&e, CORE, 0x3018, 0x10000001);
	emit(&e, CORE, 0x301c, (lines << 16) | 0);
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
	emit(&e, CORE, 0x3020,
	     mm->wdtype == CHARSIU_INT4 && mm->n > 8
		     ? (uint32_t)(2 * (ALIGN_UP(mm->n, 16) - 8) - 1)
		     : mm->n - 1);
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
	w4_dpu = w4a16 && !getenv("CHARSIU_W4_NO_DPU");

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
		const char *e8 = getenv("CHARSIU_WIDE8");

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
	if (w4a16 && getenv("CHARSIU_W4_RDMA_MASK"))
		rdma_mask = (unsigned)strtoul(getenv("CHARSIU_W4_RDMA_MASK"),
					      NULL, 0);

	emit(&e, DPU, 0x400c, 0x40000004);
	emit(&e, DPU, 0x4010, WIDE(0) ? 0xa0000002u : 0x00000000u);
	emit(&e, DPU, 0x4014, 0x00000000);
	emit(&e, DPU, 0x4018, job->output_addr);
	emit(&e, DPU, 0x401c, rows);            /* ow * full_oh */
	emit(&e, DPU, 0x4020, 0x00000000);      /* ow - 1 */
	emit(&e, DPU, 0x4024, lines);
	emit(&e, DPU, 0x4028, 0x00000000);
	emit(&e, DPU, 0x402c, mm->n - 1);
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
	emit(&e, DPU, 0x4030, ((mm->n - 1) << 16) |
	     (uint32_t)(getenv("CHARSIU_DPU_4030")
			? strtoul(getenv("CHARSIU_DPU_4030"), NULL, 0)
			: (WIDE(1) ? 0x0310u : 0x0710u)));
	emit(&e, DPU, 0x4034, (lines << 16) | 0);
	/* Mesa's regular conv value. The vendor's DPU only streams carry 0x53
	 * here, but those are elementwise ops rather than convolutions, so it
	 * is a candidate to sweep and not a value to copy. */
	emit(&e, DPU, 0x4038, (uint32_t)(getenv("CHARSIU_DPU_4038")
					 ? strtoul(getenv("CHARSIU_DPU_4038"), NULL, 0)
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
	emit(&e, DPU, 0x406c, (uint32_t)(getenv("CHARSIU_DPU_406C")
					 ? strtoul(getenv("CHARSIU_DPU_406C"), NULL, 0)
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
	emit(&e, DPU, 0x4070, (uint32_t)(getenv("CHARSIU_DPU_4070")
					 ? strtoul(getenv("CHARSIU_DPU_4070"), NULL, 0)
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
	emit(&e, DPU, 0x40b8, job->acc_out ? 3u
					   : (uint32_t)(1 * (2 * rows - rows)));
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

	return e.n > max ? 0 : e.n;
}
