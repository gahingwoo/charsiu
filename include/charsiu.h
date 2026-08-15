/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
#ifndef CHARSIU_H
#define CHARSIU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A matmul on this NPU is a 1x1 convolution. C[M,N] = A[M,K] . B[N,K]^T with
 *
 *   K   the contraction axis      -> the convolution's INPUT channels
 *   N   the output axis           -> its OUTPUT channels, one 1x1 filter each
 *   M   the rows                  -> the spatial HEIGHT of a 1 column image
 *
 * M is the height rather than the width on purpose. The RK3576 measures a row's
 * CBUF cost per column, so a 1 column image of M rows is the arrangement whose
 * cost does not grow with M beyond the rows themselves, and it is the shape the
 * vendor's own LLM dispatches use.
 */
enum charsiu_dtype {
	CHARSIU_INT8,
	CHARSIU_INT4,
	CHARSIU_FP16,
};

/* The feature atom: how many channels share one 16 byte unit of a column. */
static inline unsigned charsiu_feature_atom(enum charsiu_dtype dt)
{
	switch (dt) {
	case CHARSIU_INT4: return 32;
	case CHARSIU_INT8: return 16;
	default:           return 8;   /* every 2 byte type */
	}
}

/* The weight tile: the buffer is [N/ng][K/kg][N%ng][K%kg], cut short at the
 * edges rather than padded. Established against the vendor's own compiler at
 * 64 by 64, 64 by 34, 64 by 56 and 48 by 40 for the int8 case; the int4 N group
 * is the RK3588 table's and is NOT yet confirmed on this silicon. */
static inline unsigned charsiu_weight_ngroup(enum charsiu_dtype dt)
{
	switch (dt) {
	case CHARSIU_INT4: return 64;
	case CHARSIU_INT8: return 32;
	default:           return 16;
	}
}

static inline unsigned charsiu_weight_kgroup(enum charsiu_dtype dt)
{
	(void)dt;
	return 32;
}

/*
 * The weight and the activation dtype are SEPARATE. The vendor's own model is
 * named w4a16 and means it: 4 bit weights against 16 bit activations. Reading
 * that as one dtype puts the feature atom at 32 instead of 8 and the input
 * surface stride out by a factor of four, which is how it was caught: the
 * vendor's 0x1028 carries 128 for a K of 4096 and only an atom of 8 gives that.
 */
struct charsiu_matmul {
	unsigned m;                  /* rows of A, and of C */
	unsigned k;                  /* the contraction axis */
	unsigned n;                  /* output channels */
	enum charsiu_dtype wdtype;   /* the weights */
	enum charsiu_dtype adtype;   /* the activations, and the input surface */
};

/*
 * Emit the register command stream for one matmul.
 *
 * Writes at most `max` little-endian u64 entries, each
 *
 *     [63:48] target   [47:16] value   [15:0] register
 *
 * and returns how many it wrote, or 0 if `max` is too small. Addresses are left
 * at zero: they are patched in at submit time, and a stream with none in it is
 * directly comparable against one read out of a vendor model file.
 */
size_t charsiu_emit_matmul(const struct charsiu_matmul *mm,
			   uint64_t *out, size_t max);

/*
 * Pack B[N][K] row major into the NPU's weight tile order, for `dtype`.
 * `dst` must hold charsiu_weight_bytes(mm) bytes.
 */
/*
 * Pack A[M][K] row major into [K/atom][M][atom], biased by -0x80. `dst_size` is
 * the whole buffer, which is filled with the biased zero point first because
 * the CBUF reads past what a matmul writes.
 */
void charsiu_pack_input(const struct charsiu_matmul *mm, const uint8_t *src,
			uint8_t *dst, size_t dst_size, uint8_t input_zero_point);

void charsiu_pack_weights(const struct charsiu_matmul *mm,
			  const uint8_t *src, uint8_t *dst);

size_t charsiu_weight_bytes(const struct charsiu_matmul *mm);

/* The 64 byte units one row of A occupies in the CBUF. A column costs
 * DIV_ROUND_UP(K, 16) atoms of 16 bytes, four to a unit, except that a last
 * unit holding exactly three atoms does not pack and costs a whole unit. */
unsigned charsiu_entries_per_row(const struct charsiu_matmul *mm);



/* ---- a complete, submittable stream --------------------------------------- */

/*
 * Everything a real submit needs beyond the geometry: where the operands are,
 * and how the accumulator becomes an output byte.
 *
 * The register values behind this come from the Mesa Teflon driver's RK3576
 * path in linux-rk3576-npu, which is verified byte exact on this silicon over
 * a long series of board rounds. They are ported rather than re-derived: a
 * value nobody can point at the origin of is exactly what this project refuses
 * to write.
 */
struct charsiu_job {
	struct charsiu_matmul mm;

	uint32_t input_addr;         /* A, in the NPU address space */
	uint32_t weight_addr;        /* B, packed */
	uint32_t output_addr;        /* C */
	uint32_t coef_addr;          /* the per channel record table */

	float input_scale;
	float weight_scale;
	float output_scale;
	int input_zero_point;
	int weight_zero_point;
	int output_zero_point;
};

/* The whole stream, addresses and requant included. */
size_t charsiu_emit_job(const struct charsiu_job *job, uint64_t *out, size_t max);

/* Build the coefficient buffer the DPU reads: the per channel A/B/C records,
 * the fp16 scale table, and the second operand word after it. Without it the
 * convolution comes back empty. */
size_t charsiu_coef_bytes(const struct charsiu_matmul *mm);
void charsiu_build_coefs(const struct charsiu_job *job, const int32_t *bias,
			 const int32_t *weight_sums, uint8_t *dst);

/* ---- the device ---------------------------------------------------------- */

struct charsiu_device;

struct charsiu_bo {
	uint32_t handle;
	uint64_t dma_address;   /* what a register in the stream points at */
	size_t size;
	void *map;              /* the CPU mapping; valid between prep and fini */
};

struct charsiu_device *charsiu_open(const char *path);
void charsiu_close(struct charsiu_device *dev);

int charsiu_bo_alloc(struct charsiu_device *dev, size_t size, struct charsiu_bo *bo);
void charsiu_bo_free(struct charsiu_device *dev, struct charsiu_bo *bo);

/* Take CPU ownership before touching bo->map, and give it back before submitting.
 * These are the cache maintenance; skipping them returns stale data rather than
 * failing. */
int charsiu_bo_prep(struct charsiu_device *dev, struct charsiu_bo *bo,
		    int64_t timeout_ns);
int charsiu_bo_fini(struct charsiu_device *dev, struct charsiu_bo *bo);

int charsiu_submit(struct charsiu_device *dev, const struct charsiu_bo *regcmd,
		   unsigned regcmd_count, const uint32_t *in_handles,
		   unsigned in_count, const uint32_t *out_handles,
		   unsigned out_count);

#ifdef __cplusplus
}
#endif

#endif /* CHARSIU_H */
