// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The RK3588 matmul, as Mesa's rocket driver would program a 1x1 convolution.
 *
 * Ported from Mesa src/gallium/drivers/rocket (rkt_regcmd.c fill_first_regcmd,
 * rkt_task.c fill_task and rkt_split_tasks, rkt_coefs.c, rkt_ml.c), Copyright
 * (c) 2024 Tomeu Vizoso, MIT, read at mesa 0e9e108. MIT combines into
 * GPL-2.0-or-later. Field positions are from that tree's registers.xml.
 *
 * WHAT IS DELIBERATELY THE SAME AS MESA, EVEN WHERE IT LOOKS ODD:
 *   - the extra weight bank ("the calc above might be wrong on this HW")
 *   - FEATURE_GRAINS = 50 + stride + 1 ("seems to pass the most tests")
 *   - the surface stride formula, line_stride * (h / 4 - 1), which goes
 *     negative below h = 4. Mesa stores that float into an unsigned, which is
 *     undefined in C and is 0 on aarch64. It is written as 0 here so the
 *     behaviour is the same on every compiler, and it is a suspect for the
 *     "height below four computes wrong" finding on this chip, not a fix.
 * Changing any of them is a board question, not an edit.
 *
 * ONE TASK ONLY. rkt_split_tasks slices the input height when the input does
 * not fit the CBUF beside the weights. That path is not ported, and
 * rk3588_emit_matmul refuses a shape that would need it.
 */
#include <math.h>
#include <string.h>

#include "charsiu_rk3588.h"

#define DIV_ROUND_UP(n, d)  (((n) + (d) - 1) / (d))
#define ALIGN_UP(n, a)      (DIV_ROUND_UP((n), (a)) * (a))
#define MAX(a, b)           ((a) > (b) ? (a) : (b))
#define MIN(a, b)           ((a) < (b) ? (a) : (b))

/* target ids: the block's enum value in registers.xml, plus one */
#define T_PC    0x0101u
#define T_CNA   0x0201u
#define T_CORE  0x0801u
#define T_DPU   0x1001u
#define T_RDMA  0x2001u

static unsigned in_channels(const struct rk3588_mm *mm)
{
	return ALIGN_UP(MAX(mm->k, RK3588_FEATURE_ATOM), RK3588_FEATURE_ATOM);
}

static unsigned out_channels(const struct rk3588_mm *mm)
{
	return ALIGN_UP(MAX(mm->n, 32u), 32u);
}

size_t rk3588_input_bytes(const struct rk3588_mm *mm)
{
	/* calc_raw_input_size: Mesa allocates twice the channel groups */
	return (size_t)mm->w * mm->h *
	       DIV_ROUND_UP(mm->k, RK3588_FEATURE_ATOM) * 2 * RK3588_FEATURE_ATOM;
}

size_t rk3588_output_bytes(const struct rk3588_mm *mm)
{
	return (size_t)mm->w * mm->h *
	       DIV_ROUND_UP(mm->n, RK3588_FEATURE_ATOM) * 2 * RK3588_FEATURE_ATOM;
}

size_t rk3588_weight_bytes(const struct rk3588_mm *mm)
{
	return (size_t)ALIGN_UP(mm->n, 2) *
	       ALIGN_UP(in_channels(mm), RK3588_WEIGHT_ATOM) * 2;
}

size_t rk3588_bias_bytes(const struct rk3588_mm *mm)
{
	/* the DPU reads out_channels() of them, not n */
	return (size_t)out_channels(mm) * sizeof(int32_t);
}

/* NC1HWC2, C2 = 16, signed: ((c1 * h + y) * w + x) * 16 + c2 */
void rk3588_pack_input(const struct rk3588_mm *mm, const uint8_t *A, uint8_t *out)
{
	unsigned groups = DIV_ROUND_UP(mm->k, RK3588_FEATURE_ATOM);
	size_t n = 0;

	for (unsigned u = 0; u < groups; u++)
		for (unsigned y = 0; y < mm->h; y++)
			for (unsigned x = 0; x < mm->w; x++)
				for (unsigned c = 0; c < RK3588_FEATURE_ATOM; c++) {
					unsigned ch = u * RK3588_FEATURE_ATOM + c;
					size_t row = (size_t)y * mm->w + x;

					out[n++] = ch < mm->k
						? (uint8_t)(A[row * mm->k + ch] - 0x80)
						: (uint8_t)(mm->input_zp - 0x80);
				}
}

/* rkt_fill_weights with a 1x1 kernel, loop for loop */
void rk3588_pack_weights(const struct rk3588_mm *mm, const uint8_t *B, uint8_t *out)
{
	unsigned ic = in_channels(mm);
	unsigned oc = ALIGN_UP(mm->n, 2);
	unsigned ic1 = DIV_ROUND_UP(ic, RK3588_WEIGHT_ATOM);
	unsigned ic2n = MIN(ic, (unsigned)RK3588_WEIGHT_ATOM);
	size_t n = 0;

	for (unsigned o1 = 0; o1 < DIV_ROUND_UP(oc, RK3588_WEIGHT_ATOM); o1++)
		for (unsigned i1 = 0; i1 < ic1; i1++)
			for (unsigned o2 = 0; o2 < MIN(oc, (unsigned)RK3588_WEIGHT_ATOM); o2++)
				for (unsigned i2 = 0; i2 < ic2n; i2++) {
					unsigned o = o1 * RK3588_WEIGHT_ATOM + o2;
					unsigned i = i1 * RK3588_WEIGHT_ATOM + i2;

					if (mm->n > 2 && o >= ALIGN_UP(mm->n, 2))
						continue;
					if (o >= mm->n)
						out[n++] = 0;
					else if (i >= mm->k) {
						if (i2 < 16 || (mm->k % 32) > 16)
							out[n++] = (uint8_t)(mm->weight_zp - 0x80);
					} else
						out[n++] = (uint8_t)(B[(size_t)o * mm->k + i] - 0x80);
				}
}

/* rkt_fill_biases with truncate_bits = 0 */
void rk3588_fill_bias(const struct rk3588_mm *mm, const uint8_t *B,
		      const int32_t *bias, int32_t *out)
{
	memset(out, 0, rk3588_bias_bytes(mm));
	for (unsigned o = 0; o < mm->n; o++) {
		int32_t corr = 0;

		for (unsigned i = 0; i < mm->k; i++)
			corr += ((int32_t)B[(size_t)o * mm->k + i] - (int32_t)mm->weight_zp) *
				((int32_t)mm->input_zp - 0x80);
		out[o] = (bias ? bias[o] : 0) - corr;
	}
}

/* output_in[g][y][x][c] + 0x80 */
void rk3588_unpack_output(const struct rk3588_mm *mm, const uint8_t *raw, uint8_t *C)
{
	for (unsigned y = 0; y < mm->h; y++)
		for (unsigned x = 0; x < mm->w; x++)
			for (unsigned o = 0; o < mm->n; o++) {
				unsigned g = o / RK3588_FEATURE_ATOM;
				unsigned c = o % RK3588_FEATURE_ATOM;
				size_t at = (((size_t)g * mm->h + y) * mm->w + x) *
					    RK3588_FEATURE_ATOM + c;
				size_t row = (size_t)y * mm->w + x;

				C[row * mm->n + o] = (uint8_t)(raw[at] + 0x80);
			}
}

static uint32_t fui(float f)
{
	uint32_t u;

	memcpy(&u, &f, sizeof(u));
	return u;
}

void rk3588_out_cvt(const struct rk3588_mm *mm, unsigned *scale, unsigned *shift)
{
	float conv = (mm->input_scale * mm->weight_scale) / mm->output_scale;
	uint32_t bits = fui(conv);

	*shift = 127 + 31 - 32 - (bits >> 23) + 16;
	*scale = ((bits >> 9) & 0x7fff) + 1;
	if (*scale < (1u << 14))
		*scale |= 1u << 14;
}

struct em {
	uint64_t *out;
	size_t max, n;
};

static void raw(struct em *e, unsigned target, unsigned reg, uint32_t val)
{
	if (e->n < e->max)
		e->out[e->n] = ((uint64_t)target << 48) | ((uint64_t)val << 16) | reg;
	e->n++;
}

static void word(struct em *e, uint64_t v)
{
	if (e->n < e->max)
		e->out[e->n] = v;
	e->n++;
}

static unsigned target_of(unsigned reg)
{
	switch (reg & 0xf000) {
	case 0x0000: return T_PC;
	case 0x1000: return T_CNA;
	case 0x3000: return T_CORE;
	case 0x4000: return T_DPU;
	case 0x5000: return T_RDMA;
	}
	return 0;
}

#define EMIT(reg, val)  raw(&e, target_of(reg), (reg), (uint32_t)(val))

/* calc_input_banks, int8 */
static unsigned input_banks(const struct rk3588_mm *mm)
{
	unsigned ape = RK3588_CBUF_ENTRY_SIZE / RK3588_FEATURE_ATOM;
	unsigned c_atoms = DIV_ROUND_UP(mm->k, RK3588_FEATURE_ATOM);
	unsigned last = c_atoms % ape;
	unsigned eps = (c_atoms / ape) * mm->w +
		       (last == 3 ? mm->w : DIV_ROUND_UP(last * mm->w, ape));

	return DIV_ROUND_UP(eps * mm->h, RK3588_CBUF_ENTRIES_PER_BANK);
}

/* calc_weights_banks, extra bank and all */
static unsigned weight_banks(const struct rk3588_mm *mm, unsigned n)
{
	return DIV_ROUND_UP(DIV_ROUND_UP(in_channels(mm) * n, RK3588_CBUF_ENTRY_SIZE),
			    RK3588_CBUF_ENTRIES_PER_BANK) + 1;
}

/* rkt_split_tasks' one-task case */
static int fits_one_task(const struct rk3588_mm *mm, unsigned n)
{
	unsigned wb = weight_banks(mm, n);

	return wb + 1 < RK3588_CBUF_BANKS &&
	       input_banks(mm) <= RK3588_CBUF_BANKS - wb;
}

unsigned rk3588_max_tile_n(const struct rk3588_mm *mm)
{
	unsigned nt = 0;

	for (unsigned n = 32; n <= ALIGN_UP(mm->n, 32); n += 32) {
		if (!fits_one_task(mm, n))
			break;
		nt = n;
	}
	return nt;
}

/* weight tiles start 256 byte aligned; nothing measured asks for more */
#define WTILE_ALIGN 256

unsigned rk3588_plan_tiles(const struct rk3588_mm *mm, unsigned nt,
			   struct rk3588_tile *t, unsigned max, size_t *wbytes)
{
	unsigned count = 0;
	size_t off = 0;

	if (!nt)
		nt = rk3588_max_tile_n(mm);
	if (!nt || nt % 32 || !fits_one_task(mm, MIN(nt, mm->n)))
		return 0;
	for (unsigned n0 = 0; n0 < mm->n; n0 += nt, count++) {
		struct rk3588_mm sub = *mm;

		if (count == max)
			return 0;
		sub.n = MIN(nt, mm->n - n0);
		t[count].n0 = n0;
		t[count].n = sub.n;
		t[count].woff = off;
		off += ALIGN_UP(rk3588_weight_bytes(&sub), WTILE_ALIGN);
	}
	if (wbytes)
		*wbytes = off;
	return count;
}

void rk3588_tile_mm(const struct rk3588_mm *mm, const struct rk3588_tile *t,
		    struct rk3588_mm *sub)
{
	*sub = *mm;
	sub->n = t->n;
	sub->weight_addr = mm->weight_addr + (uint32_t)t->woff;
	sub->bias_addr = mm->bias_addr + t->n0 * (uint32_t)sizeof(int32_t);
	sub->output_addr = mm->output_addr +
			   (t->n0 / RK3588_FEATURE_ATOM) * mm->h * mm->w * RK3588_FEATURE_ATOM;
}

void rk3588_pack_weights_tiled(const struct rk3588_mm *mm,
			       const struct rk3588_tile *t, unsigned tiles,
			       const uint8_t *B, uint8_t *out)
{
	for (unsigned i = 0; i < tiles; i++) {
		struct rk3588_mm sub = *mm;

		sub.n = t[i].n;
		rk3588_pack_weights(&sub, B + (size_t)t[i].n0 * mm->k, out + t[i].woff);
	}
}

size_t rk3588_emit_matmul(const struct rk3588_mm *mm, uint64_t *out, size_t max)
{
	struct em e = { out, max, 0 };
	unsigned w = mm->w, h = mm->h;
	unsigned ic = in_channels(mm), oc = out_channels(mm);
	unsigned kernels = ALIGN_UP(mm->n, 2);
	unsigned in_banks = input_banks(mm);

	/* fill_task, 1x1, stride 1, no padding, input_channels_real > 1 */
	unsigned line_stride = w * RK3588_FEATURE_ATOM / 4;
	float surf_f = (float)line_stride * ((float)h / 4 - 1);
	unsigned surf_stride = surf_f > 0 ? (unsigned)surf_f : 0;
	unsigned out_surf_stride = w * RK3588_FEATURE_ATOM * h / RK3588_FEATURE_ATOM;
	unsigned data_entries = DIV_ROUND_UP(w * 2 * DIV_ROUND_UP(mm->k, RK3588_FEATURE_ATOM), 8);
	unsigned surfaces_per_row = w * h * 2;
	unsigned scale, shift;
	unsigned con0;

	if (mm->k < 2 || !w || !h || !mm->n)
		return 0;
	/* rkt_split_tasks: one task when the whole input fits beside the weights */
	if (!fits_one_task(mm, mm->n))
		return 0;
	con0 = ((RK3588_CBUF_BANKS - in_banks) << 4) | in_banks;

	rk3588_out_cvt(mm, &scale, &shift);

	EMIT(0x1040, con0);                             /* CNA_CBUF_CON0 */
	EMIT(0x1104, 0);                                /* CNA_DCOMP_REGNUM */
	EMIT(0x1100, 0);                                /* CNA_DCOMP_CTRL */
	EMIT(0x100c, 0);                                /* CNA_CONV_CON1 */
	EMIT(0x4004, 0xe);                              /* DPU_S_POINTER: pp mode, both pp enables */
	EMIT(0x5004, 0xe);                              /* DPU_RDMA_RDMA_S_POINTER */
	EMIT(0x100c, 0);                                /* CNA_CONV_CON1, again, as Mesa */
	EMIT(0x1010, (50 + 1 + 1) << 4);                /* CNA_CONV_CON2 FEATURE_GRAINS */
	EMIT(0x1014, (1 << 3) | 1);                     /* CNA_CONV_CON3 strides */
	EMIT(0x1020, (w << 16) | h);                    /* CNA_DATA_SIZE0 */
	EMIT(0x1024, ((mm->k - 1) << 16) | ic);         /* CNA_DATA_SIZE1 */
	EMIT(0x1028, w);                                /* CNA_DATA_SIZE2 DATAOUT_WIDTH */
	EMIT(0x102c, w * h);                            /* CNA_DATA_SIZE3 DATAOUT_ATOMICS */
	EMIT(0x1030, ic * kernels);                     /* CNA_WEIGHT_SIZE0 */
	EMIT(0x1034, ic);                               /* CNA_WEIGHT_SIZE1 */
	EMIT(0x1038, (1 << 24) | (1 << 16) | kernels);  /* CNA_WEIGHT_SIZE2 */
	EMIT(0x1040, con0);                             /* CNA_CBUF_CON0 */
	EMIT(0x1044, data_entries);                     /* CNA_CBUF_CON1 */
	EMIT(0x104c, (1 << 3) | (1 << 1) | 1);          /* CNA_CVT_CON0 sign, type, bypass */
	EMIT(0x1050, 1 << 16);                          /* CNA_CVT_CON1..4 scale 1 */
	EMIT(0x1054, 1 << 16);
	EMIT(0x1058, 1 << 16);
	EMIT(0x105c, 1 << 16);
	EMIT(0x1060, 0);                                /* CNA_FC_CON0 */
	EMIT(0x1064, 0);                                /* CNA_FC_CON1 */
	EMIT(0x1068, 0);                                /* CNA_PAD_CON0 */
	EMIT(0x1070, mm->input_addr);                   /* CNA_FEATURE_DATA_ADDR */
	EMIT(0x1074, 0);                                /* CNA_FC_CON2 */
	EMIT(0x1078, (15 << 16) | 15);                  /* CNA_DMA_CON0 burst lengths */
	EMIT(0x107c, line_stride);                      /* CNA_DMA_CON1 */
	EMIT(0x1080, surf_stride);                      /* CNA_DMA_CON2 */
	EMIT(0x1084, (w << 16) | h);                    /* CNA_FC_DATA_SIZE0 */
	EMIT(0x1088, ic);                               /* CNA_FC_DATA_SIZE1 */
	EMIT(0x1100, 0);                                /* CNA_DCOMP_CTRL */
	EMIT(0x1104, 0);                                /* CNA_DCOMP_REGNUM */
	EMIT(0x1110, mm->weight_addr);                  /* CNA_DCOMP_ADDR0 */
	for (unsigned r = 0x1140; r <= 0x117c; r += 4)  /* CNA_DCOMP_AMOUNT0..15 */
		EMIT(r, 0);
	EMIT(0x1180, 0);                                /* CNA_CVT_CON5 */
	EMIT(0x1184, (uint32_t)((int32_t)mm->input_zp - 0x80)); /* CNA_PAD_CON1 */

	EMIT(0x3010, 1);                                /* CORE_MISC_CFG QD_EN */
	EMIT(0x3014, ((h - 1) << 16) | (w - 1));        /* CORE_DATAOUT_SIZE_0 */
	EMIT(0x3018, oc - 1);                           /* CORE_DATAOUT_SIZE_1 */
	EMIT(0x301c, 0);                                /* CORE_CLIP_TRUNCATE */
	raw(&e, T_CORE, 0x3030, 0);

	EMIT(0x400c, (15 << 5) | (2 << 1));             /* DPU_FEATURE_MODE_CFG */
	EMIT(0x4010, 0);                                /* DPU_DATA_FORMAT */
	EMIT(0x4014, 0);                                /* DPU_OFFSET_PEND */
	EMIT(0x4020, mm->output_addr);                  /* DPU_DST_BASE_ADDR */
	EMIT(0x4024, out_surf_stride << 4);             /* DPU_DST_SURF_STRIDE */
	EMIT(0x4030, w - 1);                            /* DPU_DATA_CUBE_WIDTH */
	EMIT(0x4034, h - 1);                            /* DPU_DATA_CUBE_HEIGHT */
	EMIT(0x4038, 0);                                /* DPU_DATA_CUBE_NOTCH_ADDR */
	EMIT(0x403c, ((mm->n - 1) << 16) | (oc - 1));   /* DPU_DATA_CUBE_CHANNEL */
	EMIT(0x4040, (2 << 16) | (1 << 8) | (1 << 6) | (1 << 4)); /* DPU_BS_CFG */
	EMIT(0x4044, 0);                                /* DPU_BS_ALU_CFG */
	EMIT(0x4048, 0);                                /* DPU_BS_MUL_CFG */
	EMIT(0x404c, 0);                                /* DPU_BS_RELUX_CMP_VALUE */
	EMIT(0x4050, (1 << 8) | (1 << 5) | (1 << 2));   /* DPU_BS_OW_CFG */
	EMIT(0x4054, (uint32_t)(0x80 - (int32_t)mm->weight_zp) & 0xffff); /* DPU_BS_OW_OP */
	EMIT(0x4058, oc - 1);                           /* DPU_WDMA_SIZE_0 */
	EMIT(0x405c, ((h - 1) << 16) | (w - 1));        /* DPU_WDMA_SIZE_1 */
	EMIT(0x4060, (1 << 6) | (1 << 4) | (1 << 1) | 1); /* DPU_BN_CFG, all bypassed */
	EMIT(0x4064, 0);
	EMIT(0x4068, 0);
	EMIT(0x406c, 0);
	EMIT(0x4070, (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1); /* DPU_EW_CFG, bypassed */
	EMIT(0x4074, 0);                                /* DPU_EW_CVT_OFFSET_VALUE */
	EMIT(0x4078, 1);                                /* DPU_EW_CVT_SCALE_VALUE */
	EMIT(0x407c, 0);                                /* DPU_EW_RELUX_CMP_VALUE */
	EMIT(0x4080, (uint32_t)((int32_t)mm->output_zp - 0x80)); /* DPU_OUT_CVT_OFFSET */
	EMIT(0x4084, scale);                            /* DPU_OUT_CVT_SCALE */
	EMIT(0x4088, shift - 1);                        /* DPU_OUT_CVT_SHIFT */
	for (unsigned r = 0x4090; r <= 0x40ac; r += 4)  /* DPU_EW_OP_VALUE_0..7 */
		EMIT(r, 0);
	EMIT(0x40c0, surfaces_per_row << 4);            /* DPU_SURFACE_ADD */
	raw(&e, T_DPU, 0x40c4, 0);
	for (unsigned r = 0x4100; r <= 0x412c; r += 4)  /* DPU_LUT_*, all zero */
		EMIT(r, 0);

	EMIT(0x500c, w - 1);                            /* RDMA_DATA_CUBE_WIDTH */
	EMIT(0x5010, h - 1);                            /* RDMA_DATA_CUBE_HEIGHT */
	EMIT(0x5014, oc - 1);                           /* RDMA_DATA_CUBE_CHANNEL */
	EMIT(0x5018, 0);                                /* RDMA_SRC_BASE_ADDR */
	EMIT(0x501c, 1 << 1);                           /* RDMA_BRDMA_CFG DATA_USE 1 */
	EMIT(0x5020, mm->bias_addr);                    /* RDMA_BS_BASE_ADDR */
	EMIT(0x5028, 0);                                /* RDMA_NRDMA_CFG */
	EMIT(0x502c, 0);                                /* RDMA_BN_BASE_ADDR */
	EMIT(0x5034, 1);                                /* RDMA_ERDMA_CFG DISABLE */
	EMIT(0x5038, 0);                                /* RDMA_EW_BASE_ADDR */
	EMIT(0x5040, 0);                                /* RDMA_EW_SURF_STRIDE */
	EMIT(0x5044, (15 << 11) | (1 << 4));            /* RDMA_FEATURE_MODE_CFG */
	EMIT(0x5048, 0);                                /* RDMA_SRC_DMA_CFG */
	EMIT(0x504c, 0);                                /* RDMA_SURF_NOTCH */
	EMIT(0x5064, 0);                                /* RDMA_PAD_CFG */
	EMIT(0x5068, 0x01010101);                       /* RDMA_WEIGHT */
	EMIT(0x506c, 0);                                /* RDMA_EW_SURF_NOTCH */

	/*
	 * THE TAIL. One task, so no next-task address: Mesa writes a bare zero
	 * word where PC_BASE_ADDRESS would go, then PC_REGISTER_AMOUNTS, then
	 * the 0x0041 word the TRM asks for before op_en, then the broadcast
	 * enable on target 0x81. This is the form job.c records as WRONG on the
	 * RK3576 (job.c, "ENGAGE THE UNITS"). It is what Mesa runs here.
	 */
	word(&e, 0);
	EMIT(0x0014, 0);                                /* PC_REGISTER_AMOUNTS */
	word(&e, 0x0041000000000000ull);
	raw(&e, 0x81, 0x0008, (14 << 1) | 1);           /* PC_OPERATION_ENABLE */

	return e.n > max ? 0 : e.n;
}
