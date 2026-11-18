// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The RK3588 register stream, kept apart from the RK3576 one on purpose.
 *
 * WHY A SECOND EMITTER AND NOT A TABLE OF CONSTANTS. The two chips do not share
 * a register map. The RK3576 stream job.c builds was decoded against Mesa's
 * RK3588 registers.xml on a ROCK 5T (2026-11-18): 51 of its 143 writes land on
 * addresses the RK3588 does not have, and most of the rest land on a register
 * that means something else here -- 0x4044..0x40a8 carry clip bounds on the
 * RK3576 and BS_MUL_CFG, WDMA_SIZE and EW_CFG on this chip. Every job timed
 * out. So nothing in job.c is a starting point for this file, and nothing here
 * may be assumed to hold on the RK3576.
 *
 * PROVENANCE. Every value is Mesa's, from src/gallium/drivers/rocket
 * (rkt_regcmd.c, rkt_task.c, rkt_coefs.c and rkt_ml.c, Tomeu Vizoso, MIT), read
 * at mesa 0e9e108 (2026-11-18). That path runs MobileNet end to end on the
 * RK3588. It is int8 in and int8 out, through the DPU's requantiser, and that
 * is all this file claims until a board round says otherwise.
 *
 * A matmul goes in as a 1x1 convolution: an input cube of width w, height h
 * and K channels, N output channels, and the M = w * h rows of A are its
 * pixels, row r at (y, x) = (r / w, r % w).
 */
#ifndef CHARSIU_RK3588_H
#define CHARSIU_RK3588_H

#include <stddef.h>
#include <stdint.h>

#define RK3588_CBUF_BANKS            12
#define RK3588_CBUF_BANK_SIZE        32768
#define RK3588_CBUF_ENTRIES_PER_BANK 256
#define RK3588_CBUF_ENTRY_SIZE       (RK3588_CBUF_BANK_SIZE / RK3588_CBUF_ENTRIES_PER_BANK)
#define RK3588_FEATURE_ATOM          16
#define RK3588_WEIGHT_ATOM           32

struct rk3588_mm {
	unsigned w, h;          /* the input cube; M = w * h */
	unsigned k, n;

	/* uint8 zero points, the TFLite convention Mesa's path is written in */
	unsigned input_zp, weight_zp, output_zp;
	float input_scale, weight_scale, output_scale;

	uint32_t input_addr, weight_addr, bias_addr, output_addr;
};

size_t rk3588_input_bytes(const struct rk3588_mm *mm);
size_t rk3588_weight_bytes(const struct rk3588_mm *mm);
size_t rk3588_bias_bytes(const struct rk3588_mm *mm);
size_t rk3588_output_bytes(const struct rk3588_mm *mm);

/* A is M x K uint8, row major. B is N x K uint8, one row per output channel. */
void rk3588_pack_input(const struct rk3588_mm *mm, const uint8_t *A, uint8_t *out);
void rk3588_pack_weights(const struct rk3588_mm *mm, const uint8_t *B, uint8_t *out);
/* bias is N int32 in the accumulator's units, or NULL for none */
void rk3588_fill_bias(const struct rk3588_mm *mm, const uint8_t *B,
		      const int32_t *bias, int32_t *out);
/* C is M x N uint8 */
void rk3588_unpack_output(const struct rk3588_mm *mm, const uint8_t *raw, uint8_t *C);

/* the requantiser's two fields, exposed so a test can model it */
void rk3588_out_cvt(const struct rk3588_mm *mm, unsigned *scale, unsigned *shift);

/* 0 if the shape does not fit one task; rk3588_plan_tiles splits it over N */
size_t rk3588_emit_matmul(const struct rk3588_mm *mm, uint64_t *out, size_t max);

/*
 * N TILING. A projection at an LLM's width does not fit the CBUF in one task
 * (K = 2048 fits about 144 output channels beside its input), so it is cut
 * into tiles of output channels, each its own complete stream.
 *
 * A tile's width is a multiple of 32, which is the unit the DPU writes in, so
 * tile t's output is exactly channel groups n0/16 .. and every tile writes
 * into the ONE output buffer at its own offset with nothing to stitch. Its
 * bias is the full bias array at n0. Its weights are packed on their own, at
 * woff in one weight buffer.
 */
struct rk3588_tile {
	unsigned n0, n;
	size_t woff;
};

/* the widest tile that fits one task, a multiple of 32; 0 if none does */
unsigned rk3588_max_tile_n(const struct rk3588_mm *mm);

/*
 * Cut mm into tiles of at most nt channels (nt = 0 means rk3588_max_tile_n).
 * Returns the tile count, 0 if nt does not fit or more than max are needed.
 * *wbytes is the weight buffer the tiles need.
 */
unsigned rk3588_plan_tiles(const struct rk3588_mm *mm, unsigned nt,
			   struct rk3588_tile *t, unsigned max, size_t *wbytes);

/* the one-task problem tile t is, addresses already offset */
void rk3588_tile_mm(const struct rk3588_mm *mm, const struct rk3588_tile *t,
		    struct rk3588_mm *sub);

/* every tile's weights, each at its woff */
void rk3588_pack_weights_tiled(const struct rk3588_mm *mm,
			       const struct rk3588_tile *t, unsigned tiles,
			       const uint8_t *B, uint8_t *out);

#endif
