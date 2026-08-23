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

/*
 * THE K THE HARDWARE ACTUALLY READS, which is K rounded UP to the feature atom.
 *
 * Round 195 swept K across the boundary at N = 64, int8, and the line is at 16
 * and not at 32:
 *
 *   16, 32, 48, 64, 96   byte exact
 *   31, 33, 40, 63, 65   not, and 40 is a multiple of 8 so it is not 8 either
 *
 * 16 is the int8 feature atom. The input packer already pads to it: it memsets
 * the whole surface to the zero point first, so the tail of the last atom
 * contributes nothing. The WEIGHT packer did not, it cut the last k group short
 * at the real K, so the two sides disagreed about where every group after the
 * first begins.
 *
 * Padding costs nothing arithmetically: the padded inputs sit at the zero point
 * and the padded weights at theirs, so their products are zero. It costs a
 * little buffer.
 *
 * CHARSIU_NO_KALIGN restores the old behaviour, which is the control.
 */
static inline unsigned charsiu_k_padded(unsigned k, enum charsiu_dtype adt)
{
	unsigned atom = charsiu_feature_atom(adt);

	return (k + atom - 1) / atom * atom;
}

/* The weight tile: the buffer is [N/ng][K/kg][N%ng][K%kg], cut short at the
 * edges rather than padded. Established against the vendor's own compiler at
 * 64 by 64, 64 by 34, 64 by 56 and 48 by 40 for the int8 case, and CONFIRMED
 * directly on the hardware by a sparse map: one live byte at a time gives
 * n = byte / 32 and k = byte % 32 on all 512 probes of a 64 by 64 buffer.
 * The int4 numbers below are not confirmed and are known to be wrong. */
static inline unsigned charsiu_weight_ngroup(enum charsiu_dtype dt)
{
	switch (dt) {
	/*
	 * 32 here is NOT confirmed. Round 167's argument for it, that low
	 * nibbles live and high nibbles live give the identical output, was
	 * withdrawn in round 171: a full buffer fill cannot distinguish one
	 * permutation from another. A sparse map does show 32 channels in the
	 * first block, at 8 bytes each, but their group stride is 512 where
	 * everything else about the tile predicts 1024, so the grouping is not
	 * understood.
	 */
	case CHARSIU_INT4: return 32;
	case CHARSIU_INT8: return 32;
	default:           return 16;
	}
}

static inline unsigned charsiu_weight_kgroup(enum charsiu_dtype dt)
{
	/*
	 * 64 for int4 is WRONG and known to be. Round 168's argument for it was
	 * withdrawn with round 167's, and a sparse map since measures the int4
	 * row at 8 bytes against int8's 32 at the same K, so the hardware's k
	 * group is 16 weights and not 64. It is left as it is because the rest
	 * of the tile is not understood either and a half corrected layout would
	 * be harder to reason about than a wholly wrong one.
	 */
	return dt == CHARSIU_INT4 ? 64 : 32;
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

unsigned charsiu_k_eff(const struct charsiu_matmul *mm);
int charsiu_w4_paired(const struct charsiu_matmul *mm);
int charsiu_cbuf_window(void);
size_t charsiu_weight_bytes(const struct charsiu_matmul *mm);

/* The 64 byte units one row of A occupies in the CBUF. A column costs
 * DIV_ROUND_UP(K, 16) atoms of 16 bytes, four to a unit, except that a last
 * unit holding exactly three atoms does not pack and costs a whole unit. */
/*
 * What the hardware actually treats the activation as. int4 weights consume it
 * as 16 bits whatever the stream asks for; round 178 measured the pairing as
 * k_hardware = 2 * k_ours + 1, which is a 16 bit element read out of an 8 bit
 * buffer. The atom, the surface stride and the packing all have to come from
 * this one place or they describe different buffers.
 */
enum charsiu_dtype charsiu_effective_adtype(const struct charsiu_matmul *mm);

/* An IEEE half, for building fp16 activations. The DPU already needed one for
 * the coefficient buffer's scale table; this is the same conversion exposed,
 * because int4 weights consume the activation as a 16 bit float and a runtime
 * has to be able to produce one. */
uint16_t charsiu_float_to_half(float f);
float charsiu_half_to_float(uint16_t h);

/* Pack A[M][K] as real fp16, which is what int4 weights consume. No zero point:
 * a float carries its own sign, so pass the dequantised values. */
void charsiu_pack_input_f16(const struct charsiu_matmul *mm, const float *src,
			    uint8_t *dst, size_t dst_size);

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

	/*
	 * The output is the RAW SIGNED 32 BIT ACCUMULATOR, four bytes an
	 * element, not a byte requantised through the coefficient buffer.
	 *
	 * A projection has to leave the NPU this way. A coefficient buffer's
	 * output scale is frozen when the buffer is built, and ffn_down's
	 * output magnitude moves by 2971x between tokens, so a scale sized for
	 * the largest vector quantises a typical one to nothing -- measured on
	 * the CPU model of this format, where a frozen scale turns the right
	 * sentence into "a country".
	 *
	 * It is the vendor's own w4a16 output stage, applied to int8 weights.
	 * Board round 312, M=1 K=2048 N=1024: 4096 of 4096 bytes written and
	 * 1024 of 1024 elements byte exact against the accumulator.
	 */
	int acc_out;
};

/* The whole stream, addresses and requant included. */
size_t charsiu_emit_job(const struct charsiu_job *job, uint64_t *out, size_t max);
int charsiu_vendor_stream_shape(unsigned *k, unsigned *n, size_t *wbytes);

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

/*
 * One task is one register stream. Tasks inside a job are chained on a single
 * core; jobs are what the scheduler can spread across cores. At M = 1 the
 * arithmetic is under a microsecond, so which of those two a runtime uses is the
 * whole performance question.
 */
struct charsiu_task {
	uint32_t regcmd;          /* IOVA of the stream, must be under 4 GiB */
	uint32_t regcmd_count;
};

struct charsiu_joblist {
	const struct charsiu_task *tasks;
	unsigned task_count;
	const uint32_t *in_handles;
	unsigned in_count;
	const uint32_t *out_handles;
	unsigned out_count;
};

int charsiu_submit_jobs(struct charsiu_device *dev,
			const struct charsiu_joblist *jobs, unsigned job_count);

int charsiu_submit(struct charsiu_device *dev, const struct charsiu_bo *regcmd,
		   unsigned regcmd_count, const uint32_t *in_handles,
		   unsigned in_count, const uint32_t *out_handles,
		   unsigned out_count);

#ifdef __cplusplus
}
#endif

#endif /* CHARSIU_H */
