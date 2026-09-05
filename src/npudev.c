// SPDX-License-Identifier: GPL-2.0-or-later
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */

/*
 * A projection, on the NPU, inside the decode loop.
 *
 * This is the substitution the whole project has been building towards, and it
 * is a substitution rather than a design because the CPU already computes the
 * same arithmetic:
 *
 *     acc[n] = sum_k a_q[k] * w_q[n][k]        <- npu_matvec, and now the DPU
 *     y[n]   = acc[n] * a_scale * w_scale[n]   <- always the CPU
 *
 * So the acceptance test is not "coherent text". It is BIT IDENTICAL TOKENS
 * against CHARSIU_NPU_QUANT=1 on the CPU, because both sides compute the same
 * integer sum and only the machine differs. Anything else is a defect with a
 * known reference to bisect against.
 *
 * The output comes back as the raw signed 32 bit accumulator (job.acc_out),
 * which board round 312 measured at a projection's shape: M=1 K=2048 N=1024,
 * 1024 of 1024 elements byte exact.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <dirent.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "reusekey.h"
#include "kslice.h"
#include "overlap.h"
#include "charsiu.h"
#include "charsiu_llm.h"

/*
 * ⚠⚠ t->q IS NIBBLE PACKED FOR int4 AND THIS FILE IS ITS OTHER READER.
 *
 * The quantiser holds an int4 weight in HALF a byte -- 298.0 MB of q on
 * Qwen3-0.6B rather than 596.0, and 1861.2 rather than 3722.4 on Phi-3.5-mini
 * -- laid out row major, ((k + 1) / 2) bytes a row, low nibble first. The long
 * note over npu_q_packed in src/npuquant.c has the layout and, more
 * importantly, why "is it packed" is ONE process wide bool rather than anything
 * per tensor: this file only holds t->name, and two files disagreeing by a
 * nibble about the same buffer is a wrong answer that reads as a right one.
 *
 * Declared here rather than in charsiu_llm.h because the width q is held at is
 * the quantiser's business and nothing outside these two files reads it.
 *
 * ⚠ AND IT IS NOT THE SAME QUESTION AS g->w4. g->w4 is what the DEVICE was
 * opened as and decides which byte the packer is handed; npu_q_packed is how
 * the weights are stored on the way in. They come apart in a real case: a
 * vision tower forces charsiu_npu_open_mode(want_w4 = 0) so its batch is not
 * turned into one dispatch a row, while the quantiser still reads
 * CHARSIU_NPU_W4V from the runner's config and produces int4 codes. So every
 * reader below tests them separately.
 */
int npu_q_packed(void);
size_t npu_q_stride(uint64_t k);

/* one int4 code out of a packed row of t->q, by column */
static inline int q_code(const int8_t *row, uint64_t i)
{
	unsigned v = (unsigned)(uint8_t)row[i >> 1];

	v = (i & 1) ? (v >> 4) : (v & 0xfu);
	return v >= 8 ? (int)v - 16 : (int)v;
}

/*
 * One SLICE of a projection.
 *
 * Round 313 put every attention projection on the hardware and got tokens
 * identical to the CPU, then wedged the board on the feed forward: ffn_gate is
 * N = 8192 and ffn_down is K = 8192, and the largest shape anything had been
 * measured at was N = 1024, K = 2048. So a projection is cut into slices of a
 * shape that HAS been measured. acc_out is what makes that exact: a raw int32
 * sum means partial sums over a split K add with no rounding at all.
 *
 * ⚠ AND THE SLICES OF ONE PROJECTION GO IN ONE SUBMIT. Round 316 ran all 112
 * projections correctly at 480 UNCHAINED submits a token, which is about 91 ms
 * of fixed cost and left the NPU slower than the CPU. Chaining is what the
 * measurement has been pointing at since round 165: break even is 2.2 MB of
 * weights a submit, and a whole ffn_gate is 16 MB.
 *
 * That needs every slice to write somewhere different, so each one gets its own
 * region of the output buffer and its own region of the input buffer, baked
 * into its register stream when it is built.
 */
struct npu_slot {
	struct charsiu_bo wt, coef, regcmd;
	struct charsiu_job job;
	unsigned nreg;
	unsigned n0, k0;           /* where this slice starts in the tensor */
	unsigned out_slot;         /* which output region it writes */
	unsigned di;               /* which device its buffers live on */
	/*
	 * THIS SLICE'S GROUP SCALES, ONE PER CHANNEL, LAID OUT AS THE SUM READS
	 * THEM.
	 *
	 * A grouped tensor keeps its scales as scale[channel * ngroups + group],
	 * so the read back walked them with a stride of ngroups: eight cache
	 * lines apart on the down projection, and unvectorisable everywhere.
	 * The values are the same values, gathered once when the tensor is
	 * staged rather than once a token for the life of the process.
	 *
	 * It costs what the scale array itself costs, 4.8 MB for this model,
	 * against 620 MB of weights.
	 */
	float *sc;
};

struct npu_entry {
	const struct npu_tensor *t;
	struct charsiu_bo out[2];  /* ITS OWN, one per device */
	unsigned first, count;     /* slots, n fastest */
	unsigned n_slices, k_slices;
	double weight_mb;
	/*
	 * THE ROWS THE CPU KEEPS, IF ANY.
	 *
	 * Round 370 measured what the memory controller has spare: the NPU
	 * alone pulls 10.46 GB/s, and with four threads reading DRAM beside it
	 * the pair reach 15.46, so about 5 GB/s is going unused. Decode is a
	 * dependency chain, so the only way to spend that is to give a second
	 * engine part of the SAME tensor -- and the cheapest second engine is
	 * the CPU, which is already here and is already BLOCKED in prep_bo for
	 * most of the fence.
	 *
	 * cq holds those rows' weights packed two to a byte, row major. It has
	 * to be packed, because the whole idea is bandwidth: reading rows at
	 * one byte a code would cost twice what the NPU pays for the same
	 * weights.
	 *
	 * ⚠ t->q IS PACKED THE SAME WAY NOW, and identically -- same row major
	 * order, same ((k + 1) / 2) stride, same low-nibble-first. So on the
	 * int4 path this is a memcpy of the row rather than a gather and a
	 * shift, and cpu_rows below could read t->q directly. It is still kept
	 * as its own array: the CPU's rows want to be contiguous and warm
	 * rather than strided through a tensor whose other rows the NPU is
	 * streaming at the same time, which is the whole point of the split.
	 */
	unsigned n_npu;            /* rows [0, n_npu) go to the hardware */
	uint8_t *cq;               /* rows [n_npu, n), nibble packed */
	/*
	 * WHAT EACH DEVICE WAS GIVEN, WORKED OUT ONCE WHERE IT IS DECIDED.
	 *
	 * The slice to device assignment is fixed at staging and never moves,
	 * so the bytes and the task count a call will cost are known before the
	 * call happens. Summing them here rather than in the matvec keeps the
	 * per call accounting to a handful of adds over at most eight entries,
	 * which matters because the thing being measured is a 133 us fixed cost
	 * and an instrument that costs a microsecond of it is not an instrument.
	 *
	 * The two cores are submitted together and waited on together, so the
	 * call's wall clock follows whichever device got MORE -- see the fit on
	 * struct charsiu_npu. That is why both are kept per device rather than
	 * summed: max(d) is the quantity, not the total.
	 */
	double mb_dev[2];          /* weight megabytes, per device */
	unsigned nt_dev[2];        /* chained tasks, per device */
};

/*
 * THE BATCHED OUTPUT, SHARED BY GEOMETRY RATHER THAN OWNED BY A TENSOR.
 *
 * ⚠ NOT ONE BUFFER FOR EVERYTHING, and that is the lesson decode already paid
 * for two hundred rounds ago: bo_prep and bo_fini are cache maintenance over a
 * WHOLE buffer object, so one buffer shared across every tensor has to be sized
 * for the widest and every call pays for all of it. Sized for nmax * m it
 * reached 35 MB a device with the head staged, and the chain that was supposed
 * to remove the fence spent more than it saved: 3.63x at m = 32 became 3.07x.
 *
 * ⚠⚠ BUT ONE BUFFER PER TENSOR WAS PAYING FOR THAT LESSON A HUNDRED TIMES
 * OVER. A buffer object is an ioctl, an mmap and an IOVA reservation, and the
 * board timed a pair of them at 2.9 ms: 225 of them at one m came to 652 ms.
 * That is 36% of an 1811 ms batched matmul, more than the gather, the packing
 * and the fence, and the largest single line item in a time to first token that
 * is four to five times the vendor's on the same silicon. The NPU is idle for
 * 91% of a batched matmul and this was the biggest single reason why.
 *
 * ⚠⚠ AND THIS FIX MADE BOTH OF THOSE NUMBERS STALE. The pool worked: measured
 * on the board across eight models, `prep` is 0.1% to 0.5% of a batched matmul
 * now, not 36%. The idle figure went with it -- the split reads about 44%
 * fence, 26% read, 26% pack, 2% submit, 0.2% prep, so the hardware is BUSY for
 * roughly 44% of the call and idle for 56%, not 91%. The paragraph above is
 * kept because it is the record of what this fix was worth. Do not quote the
 * 36% or the 91% as current; charsiu_pool_report_batch prints the live split.
 *
 * What the size depends on is not WHICH tensor it is. It is the widest slice,
 * how many slots the busier device gets, and m -- and a transformer has a
 * handful of those, not one per tensor. Llama-3.2-1B's 113 matmul tensors, at
 * nmax = 8192 and kmax = 4096, want exactly four buffers between them:
 *
 *   wide  slots   n  the tensors that want it       bytes a device at m = 32
 *    512     1   32  attn_k, attn_v                            69632
 *   2048     1   48  attn_q, attn_output, ffn_down            266240
 *   8192     1   32  ffn_gate, ffn_up                        1052672
 *   8192     8    1  the head, 128256 wide in 16 slices      8392704
 *
 * So 113 allocations become 4 and 57.1 MB a device becomes 9.8; Qwen3-0.6B is
 * 198 tensors, also 4, and 65.8 MB a device becomes 11.3. That second number
 * matters on its own and not only as time: this runtime peaks at 1068 MB on
 * Qwen3 where Rockchip's peaks at 513, and 109 MB of that peak was output
 * buffers, all but four of them the same size as one already sitting there.
 *
 * ⚠ THE 225 IS PHI-3.5, NOT LLAMA, AND IT IS EXACTLY ONE PER TENSOR. The pass
 * that printed `alloc 652 x225` was board_w4_axis on Phi-3.5-mini, whose
 * tensors column reads 225 on the same line -- 32 layers of seven projections
 * plus the output head. gemma4's run says 277 and llama's says 113, and every
 * one of them matches its own tensor count. So the old cost was one allocation
 * per tensor with no second widening anywhere, and the new cost is one per
 * shape: 225 becomes four.
 *
 * ⚠ AND EVERY TENSOR STILL PREPS EXACTLY THE BYTES IT NEEDS, which is what
 * keeps the lesson at the top of this comment paid rather than re-learned:
 * attn_output gets a 266 KB buffer and the head gets an 8 MB one, the same two
 * sizes they had when each owned its own.
 */
struct npu_outbuf {
	unsigned wide;             /* the widest slice, in output channels */
	unsigned slots;            /* what the busier of the two devices gets */
	unsigned m;                /* the rows it is sized for, 0 if unbuilt */
	unsigned busy;             /* a bit per device with a submit outstanding */
	/*
	 * ⚠ SEPARATE, ONE PER DEVICE. A buffer object belongs to the file
	 * that created it, so the two cores -- which are two open files -- can
	 * never be handed the same one.
	 */
	struct charsiu_bo bo[2];
};

struct charsiu_npu {
	/*
	 * TWO DEVICES, BECAUSE TWO OPEN FILES ARE WHAT REACH TWO CORES.
	 *
	 * rocket gives every open DRM file its own drm_sched_entity, built over
	 * the list of every core, and an entity runs one job at a time. One file
	 * therefore never uses more than one core no matter how deep the queue.
	 *
	 * The two cores share the CBUF, so the two devices also carry different
	 * CBUF windows -- round 363: six concurrent processes on split windows,
	 * every one 1024 of 1024 exact, against three of four corrupt when they
	 * share a window.
	 *
	 * CHARSIU_NPU_ONEDEV puts it back to a single device, which is the
	 * control for every number this buys.
	 */
	struct charsiu_device *dev[2];
	unsigned ndev;
	/*
	 * /dev/cpu_dma_latency, held open for as long as the device is. See
	 * the note where it is opened: it is worth 24% of decode on this
	 * board and it is not a kernel patch. -1 when not held.
	 */
	int qos_fd;
	struct charsiu_bo in[2];      /* one per device: they cannot be shared */
	unsigned in_stride, out_stride, max_slices, maxtask;
	uint8_t *scratch;
	int32_t *acc;
	/*
	 * CHARSIU_NPU_W4V, the int4 decode path. Rounds 344 to 351 settled that
	 * this hardware computes a real int4 by fp16 dot product into float32
	 * once CORE 0x3018, 0x301c and 0x3020 carry the vendor's values, and
	 * that it is 1.42 to 1.78 times faster than int8 at every shape swept.
	 *
	 * It is OPT IN. The int8 path here produces tokens identical to the CPU
	 * at 6.55 tok/s and is not going to be replaced by a path that has never
	 * decoded a sentence.
	 *
	 * The activation goes in as the REAL float, not the int8 quantised one,
	 * so this mode has no input zero point, no wsum correction and no d1 in
	 * the dequantisation -- and it skips charsiu_act's quantisation
	 * entirely, which is 10.1 ms of a 153 ms token on the CPU side.
	 */
	int w4;
	/*
	 * CHARSIU_NPU_W4_MIDRISE: the vendor's grid, w = (s + 0.5) * d, with no
	 * code for zero. The hardware still computes sum(s * a), so the half
	 * step is 0.5 * d * sum(a) -- one number per K slice per token, shared
	 * by every output channel, which is why it is nearly free here.
	 */
	int midrise;
	double *asum;      /* per K slice, the sum of the activation */
	float *fscr;
	float *accf;
	uint8_t *wpack;
	double add_us, t_first;
	struct charsiu_task *tasks;
	uint32_t *handles;
	unsigned nmax, kmax, max_n;
	/*
	 * ⚠⚠ WHICH CORE THE NEXT SLICE GOES TO, CARRIED ACROSS TENSORS.
	 *
	 * A slice's device is fixed when the tensor is STAGED -- its weights,
	 * its coefficients and its register stream are three buffer objects on
	 * g->dev[di], and the input and output addresses baked into that stream
	 * are that device's -- so this is the only place the decision can be
	 * made. Nothing about the assignment reaches the arithmetic: the sum
	 * over slices reads e->out[s->di] by s->out_slot and is identical
	 * whichever way they are dealt.
	 *
	 * It used to be `di = (ki * ns + ni) & 1`, which restarts at 0 for
	 * every tensor. That balances the slices of ONE tensor and can do
	 * nothing for a call that carries several -- matvec_pair in src/llama.c
	 * sends q, k and v as one call and gate and up as another -- and it
	 * balances them by COUNT rather than by bytes. Both failures are real
	 * on models this tree ships against, and both are invisible on
	 * Llama-3.2-1B, which is why they survived: every one of its dimensions
	 * is a power of two, so every tensor cuts into an even number of equal
	 * slices and the index deal is already optimal.
	 *
	 *   Qwen3-0.6B    n_embd 1024, so q, k, v, gate and up are ONE K slice
	 *                 each at KMAX 1024. q, k and v all land on device 0
	 *                 and device 1 sits out the call; so do gate and up.
	 *   gemma-3-1b    n_embd 1152 cuts into 1024 + 128, so ki = 0 is always
	 *                 device 0 and ki = 1 always device 1. gate+up comes to
	 *                 2 tasks and 7.078 MB against 2 tasks and 0.885 MB --
	 *                 the counts are even and the bytes are eight to one.
	 *
	 * So the deal is least-loaded instead, over a running cost per device,
	 * and the cost is the board's own fitted line (see the fit on `calls`
	 * above). What each device is CHARGED for a slice is the two variable
	 * terms of that line; the per call intercept is paid by both and
	 * cancels.
	 *
	 * ⚠⚠ AND THE COUNTERS RESET WHEN K CHANGES, WHICH IS NOT A GUESS ABOUT
	 * LAYER STRUCTURE. charsiu_npu_matvec_group REFUSES a group whose
	 * entries do not share one K -- `if (g->ent[ids[i]].t->k != e0->t->k)
	 * return -1`, because a group shares one packed activation -- so two
	 * tensors of different K can never be in the same call. A change of K
	 * is therefore a guaranteed call boundary, and it is the only one
	 * staging can see: charsiu_npu_add is called one tensor at a time by
	 * charsiu_pool_get on first use, and nothing tells it that the tensor
	 * it is staging will be dispatched with the next two.
	 *
	 * Carrying a residue ACROSS that boundary is what a purely running
	 * counter gets wrong, and the offline geometry check priced it: without
	 * the reset the synthetic TinyLLAMA token comes out 8 us WORSE than the
	 * index deal and gemma4 leaves 415 us on the table, because a call that
	 * ended uneven pushes the next one off centre. With it, the deal lands
	 * on the per call optimum -- an exhaustive minimum over every
	 * two-colouring of the call's slices -- on all six models measured.
	 */
	double deal_load[2];
	uint64_t deal_k;
	int deal_index;            /* CHARSIU_NPU_DEAL_INDEX, the old deal */
	struct npu_slot *slot;
	unsigned n_slot, slot_cap;
	struct npu_entry *ent;
	unsigned n_ent, ent_cap;
	/*
	 * ⚠ THE BATCHED PATH'S OWN BUFFERS, allocated on first use and never by
	 * a decode. A slice's weights and coefficients do not depend on m and
	 * are reused as staged; the register stream, the activation and the
	 * output all do, so the batched path brings its own rather than
	 * disturbing the ones decode has been reading for hundreds of rounds.
	 */
	struct charsiu_bo bin[2], breg[2];
	unsigned bm;               /* the m those are sized for, 0 if unbuilt */
	unsigned bnks, bnslots;    /* and how many K slices and slots */
	/*
	 * What EACH DEVICE'S input BO holds right now, so a caller that
	 * declares its input unchanged (charsiu_npu_matmul_same) can skip the
	 * pack. The key is the pointer, the width, K and the zero point; the
	 * CONTENTS are the caller's word. Cleared whenever the BOs are rebuilt.
	 *
	 * ⚠⚠ PER DEVICE, BECAUSE THE FIRST VERSION WAS NOT AND THE BOARD SAID
	 * SO IN ONE ROUND: 6 of 9 models' batched prompts stopped matching
	 * their token loop. A tensor is packed only into the BOs of the
	 * devices its slots were dealt to, and a small projection -- Qwen2.5's
	 * k with two KV heads, gemma3's with one -- goes whole to ONE core.
	 * So q packed core 0, k was dealt to core 1, and k "reused" a BO on
	 * core 1 that held whatever the last tensor there had left. The three
	 * models that stayed right were the three whose every projection is
	 * wide enough to be split across both cores. One key for two buffers
	 * described neither.
	 */
	struct reuse_key bin_key[2];   /* the rule is in reusekey.h */
	int reuse_ask;
	unsigned long reuse_hits, reuse_misses;
	/*
	 * ⚠ WHY A MISS MISSED. Phase 9 on 2026-09-05 reported Phi-3.5 as
	 * "reused 0 times, packed anyway 2304 times when declared the same"
	 * while every other model reused, and that one number has four
	 * different fixes behind it. The four are counted apart so the next
	 * round does not have to guess which: dropped, a different X, a
	 * different shape, or this device not holding the K slices asked for.
	 */
	unsigned long reuse_why[4];   /* dropped, other X, other shape, slices */
	size_t bin_stride, bout_stride;
	/* the output buffers, one per geometry rather than one per tensor:
	 * see the comment on struct npu_outbuf */
	struct npu_outbuf *obuf;
	unsigned n_obuf, obuf_cap;
	/* the last batched call's buffer and tensor, for charsiu_npu_slot_word */
	struct npu_outbuf *last_ob;
	int last_id;
	float *bscr;               /* m rows of one slice's K, gathered */
	uint8_t *bq;               /* and quantised, for the int8 path */
	/*
	 * ⚠ THE READ ORDER AS A TABLE, because charsiu_acc_index costs four
	 * integer divisions and the read back runs it once per output element:
	 * 23 million of them for one m = 32 pass over a 1B model, which on an
	 * A72 is most of a second. The mapping depends only on (m, n), so it is
	 * built once and looked up.
	 */
	/*
	 * ⚠⚠ BUILT ONCE, AT THE WIDEST SLICE THERE CAN BE.
	 *
	 * charsiu_acc_index costs four integer divisions and there is one
	 * output element per call of it, so it is a table. What made the table
	 * cost more than the loop it saved is that it was keyed on the tensor's
	 * width and so REBUILT for every tensor: 113 of them twice over, each
	 * m * n entries, which at m = 32 is most of the 283 ms the split
	 * charged to reading.
	 *
	 * charsiu_acc_index does not depend on n AT ALL -- it is
	 * (ni/32)*(m*32) + (mi/P)*(32P) + j, and n appears nowhere. So one
	 * table at nmax serves every tensor, and a narrower slice just uses
	 * fewer of each row's entries. It is rebuilt only when m changes.
	 *
	 * ⚠ INDEX IT AT THE STRIDE IT WAS BUILT WITH, never at the slice's own
	 * width. Doing that cost exactly one tensor -- the head, whose last n
	 * slice is 5376 against 8192 -- and 3585 rows of 3616.
	 */
	uint32_t *bmap;
	unsigned bmap_m;
	unsigned bmap_n4;	/* the table is one entry per FOUR channels */
	/*
	 * Whether rows 4h..4h+3 of every channel quad sit at index, +4, +8,
	 * +12 in this table -- one 64-byte line -- which is what lets the read
	 * back take four rows off one line (read_rows4). Checked on the table
	 * itself whenever it is rebuilt, never assumed from the layout.
	 */
	int bmap4;
	int bmap2;   /* rows 2h, 2h+1 at index, +4: the two-row premise */
	/*
	 * ⚠ WHAT THE BATCHED TIME IS MADE OF. It costs 135 ms at m = 2, which
	 * is 9.14 GB/s and the DRAM roof, and 754 at m = 32, which is 1.64. The
	 * extra is linear in the rows, about 20 ms a row, and a speedup against
	 * a one row loop cannot say whether that is the hardware computing more
	 * or this file preparing more, because both sides pay the CPU part.
	 */
	double bpack_us, bsub_us, bfence_us, bread_us;
	/*
	 * ⚠ PACK HAD NO PARTS. Phase 9 on the board, 2026-09-04, Qwen3 at chunk
	 * 80: "pack" was 2.0 ms a row, 0.8 ms a call, and the fp16 packer moves
	 * the 160 KB a call takes in about 7 us. Whatever the other 790 us are
	 * -- the register streams emitted per slot, the two FINI ioctls a
	 * device, the copies -- this splits them, so the next round can say.
	 */
	double bpack_emit_us, bpack_fini_us;
	/*
	 * ⚠⚠ THE DENOMINATOR, AND IT HAS TO LIVE HERE. The obvious one is
	 * charsiu_npu_pool::hw_ms, but that is only incremented by
	 * charsiu_pool_rows, which VISION AND WHISPER call and LLAMA DOES
	 * NOT -- llama calls charsiu_npu_matmul directly. Using it would
	 * have divided the five shares by zero on exactly the workload the
	 * split was added to explain. Timed at this entry instead, so every
	 * caller is covered and none has to remember.
	 */
	double bwall_us;
	double bprep_us;	/* buffers and the output zero, before any of it */
	unsigned char *bseen;	/* which n slices of Y have been written */
	unsigned bseen_n;
	double balloc_us;	/* the output BO allocation, inside prep */
	unsigned balloc_n;
	float *bd1;                /* each row's own quantisation scale */
	unsigned long submits;
	double weight_mb;          /* summed over submits, for the report */
	/*
	 * Wall clock actually spent in the hardware path, submit and read back
	 * together. Three tokens per second predictions in a row were wrong
	 * because a cost was assumed rather than measured, so the split between
	 * the NPU and the CPU stops being an inference here.
	 */
	double busy_us;
	/*
	 * And what that time is MADE of. bo_prep is not a read: it WAITS for the
	 * job, so the fence and the copy have to be told apart or the 23 ms this
	 * leaves over stays a residual rather than a measurement.
	 *
	 * ⚠ AND THE FENCE NUMBER IS NOT PURE WAITING. rocket_ioctl_prep_bo is a
	 * dma_resv_wait_timeout FOLLOWED BY dma_sync_sgtable_for_cpu, so the
	 * invalidate over the whole output buffer is charged to the fence, not
	 * to the read back. fini_bo is the other half of that pair, a
	 * dma_sync_sgtable_for_device, and it IS separable -- so it is separate
	 * here, because "13 ms reading back" and "13 ms cleaning a cache the CPU
	 * only read" ask for different fixes.
	 */
	double submit_us, fence_us, copy_us, fini_us;
	/*
	 * ⚠ ON TOP OF busy_us, NOT INSIDE IT. Packing the activation happens
	 * before the timer that covers a submit, so round 368 left 10.6 ms a
	 * token between what the stage table charges to a projection and what
	 * this file measures inside one. This is the missing piece, measured
	 * rather than derived.
	 */
	double pack_us;
	/*
	 * And the CPU's share of the projections, which runs INSIDE the fence
	 * window rather than beside it: it is time the calling thread used to
	 * spend blocked. It is charged here so a round can see what it cost as
	 * well as what it bought.
	 */
	double cpu_us;
	/*
	 * ⚠ THE WHOLE CALL, entry to return, so the residue stops being a
	 * SUBTRACTION.
	 *
	 * The stage table's five NPU rows minus the hardware path minus the
	 * packing left 1.95 ms a token at 64 tokens and 7.22 at 384 -- and the
	 * hardware path itself was 3.16 ms a token FASTER at the longer
	 * context, which is time moving from inside these timers to outside
	 * them rather than any work being done. Two derived quantities have
	 * already been read wrong today. This one is measured.
	 *
	 * What is left over after THIS is only what llama.c does around the
	 * call: finding the tensor and quantising the output.
	 */
	double call_us;

	/*
	 * ⚠⚠ THE CALL IS THE UNIT OF WALL CLOCK AND `submits` IS NOT, WHICH
	 * MAKES THE LINE ABOVE IT IN THE REPORT HALF OF WHAT IT LOOKS LIKE.
	 *
	 * One call issues one submit PER DEVICE and then waits on both, so
	 * busy_us covers a window the two submits SHARED while g->submits
	 * counted two of them. Every "us a submit" this project has printed is
	 * therefore a call's wall clock divided by the core count, and the fit
	 * recorded in PLAN.md as `us a submit = 102.7 * MB + 112` is describing
	 * a call whose fixed cost is 224 us, not 112.
	 *
	 * ⚠ ITS PRODUCT IS RIGHT, WHICH IS WORSE THAN BEING WRONG OUTRIGHT:
	 * 12632 submits x 112 us and 6316 calls x 224 us are the same 1418 ms,
	 * so the total looks checked while the per unit number it was read off
	 * is out by a factor of two. Anyone reaching for "112 us a submit" as
	 * the thing to attack is attacking half a cost.
	 *
	 * SO THE MODEL IS FITTED PER CALL, IN THREE TERMS, ON THE BOARD. Read
	 * off TinyLLAMA's five decode stages against the geometry this file
	 * cuts -- n_embd 2048, n_ff 5632, 22 layers, KMAX 1024, NMAX 8192, both
	 * cores, int4 -- the stages and the line through them are
	 *
	 *     q k v     1.311 MB  3 tasks   measured  392.7   fit  383.5
	 *     o         1.049     1         measured  276.4   fit  280.9
	 *     gate+up   5.767     2         measured  845.0   fit  836.9
	 *     down      3.146     3         measured  573.6   fit  585.4
	 *     head     16.777     4         measured 2100.0   fit 2122.1
	 *
	 *   us a call = 128.7 + 36.8 * tasks + 110.0 * MB   (busier core)
	 *
	 * inside 2.4% at all five. The stages are weighted as a token presents
	 * them -- twenty two of the first four and one head -- because that is
	 * what the accumulators below will see, and it moves the line by about
	 * 3% against fitting the five rows evenly.
	 *
	 * ⚠ AND IT AGREES WITH A ROUND THAT NEVER SAW A MODEL. The 2026-08-15
	 * shape sweep fitted synthetic matmuls at 32 chained tasks and got 26.3
	 * us a task plus 172 us a submit plus 84.3 us a megabyte. Same three
	 * terms, same order, from different shapes on a different day.
	 *
	 * ⚠⚠ WHAT THAT SPLIT SAYS ABOUT THE ROOF. Per token it is 11.5 ms of
	 * per call cost, 7.4 ms of per task cost and 29.1 ms of weights. Those
	 * three add to the 48.0 ms stage total exactly, and that is arithmetic
	 * rather than agreement -- a least squares fit with an intercept always
	 * splits its own input exactly. What says the split is real is that a
	 * typical call sits 9 us off the line, 1.7% of a 540 us mean call. So
	 * 39% of what decode spends on the hardware is DISPATCH. The 550 MB a
	 * token over 58.4 ms that reads as "9.4 GB/s, the bandwidth roof" is an
	 * average over stages that run from 6.67 GB/s (q k v) to 15.60 (the
	 * head): a roof does not have a 2.3x spread across shapes, a fixed cost
	 * does. gate+up alone moves 253.8 MB a token in 18.59 ms, which is
	 * 13.65 GB/s across the two cores WITH its own dispatch still in it, so
	 * the streaming rate is strictly above that and decode is nowhere near
	 * it.
	 *
	 * ⚠ WHICH IS ALSO THE ANSWER TO "THE VENDOR DOES 19.71 AND WE DO
	 * 17.39". 19.71 tok/s is 50.7 ms a token; take off the 10.4 ms this
	 * token spends outside the projections and the weights would have to
	 * move at 12.8 GB/s, which is BELOW the 13.65 our own gate+up stage
	 * already demonstrates. They do not need bandwidth we have not got.
	 * They need fewer of the 89 calls and 202 tasks a token costs.
	 *
	 * ⚠ THIS IS AN INSTRUMENT, NOT A FIX. It is here because the numbers
	 * above had to be fitted by hand from a five row stage table and a
	 * spreadsheet of assumed shapes, which is not something the next round
	 * should have to repeat: the board has thousands of calls a run and can
	 * fit its own line, on its own shapes, for nine adds a call.
	 *
	 * ⚠ AND THE OBVIOUS FIX IS STILL NOT FREE, FOR A SECOND REASON NOW.
	 * PLAN.md already records that raising KMAX to cut the task count
	 * coarsens int4, because the K slice must BE the quantisation group.
	 * The geometry said it also COSTS A CORE: slices were dealt as
	 * `di = (ki * ns + ni) & 1`, which restarts at 0 for every tensor, so a
	 * tensor that ends up with ONE slice landed entirely on device 0 and
	 * device 1 sat out the call. At KMAX = 2048 every one of TinyLLAMA's
	 * k = 2048 projections becomes a single slice: q, k and v all queued on
	 * core 0 and the group's busier core carried 2.621 MB in 3 tasks
	 * instead of 1.311 in 3, which by the line above is 527 us against 385.
	 *
	 * 🏁 THAT HALF IS FIXED -- the deal is least-loaded across a call now,
	 * see g->deal_load -- AND RAISING KMAX IS STILL A LOSS. An offline walk
	 * of the real .gguf geometry, scored with the line above, priced both
	 * halves separately. It reproduces the five rows of the stage table
	 * from the shapes alone (383 / 281 / 837 / 585 / 2121 us against the
	 * measured 392.7 / 276.4 / 845.0 / 573.6 / 2100.0) and it reproduces
	 * the 527 above, so it is describing this hardware and not a model of
	 * it. A whole TinyLLAMA token, in microseconds of hardware path:
	 *
	 *   KMAX 1024   index deal 48012    least loaded 47969
	 *   KMAX 2048   index deal 68064    least loaded 50410
	 *
	 * The index deal's 68064 is the +42% that was measured as "37% slower"
	 * on the board, from geometry alone. Fixing the deal takes almost all
	 * of it back -- and 50410 is still 5.1% WORSE than staying at 1024.
	 *
	 * ⚠⚠ WHICH IS ARITHMETIC RATHER THAN A SURPRISE, AND IT CLOSES THE
	 * IDEA. Merging two K slices into one removes ONE task, worth 36.8 us,
	 * and hands the surviving slice both halves' bytes -- which on a
	 * 1.05 MB slice is 115 us. The task term is only 7.4 ms of a token
	 * against the per call term's 11.5, and the per call term is untouched
	 * by KMAX: the number of CALLS is set by how many times llama.c comes
	 * in, 65 or 89 or 113 a token, and no slicing changes it. So there was
	 * never 11.5 ms in reach here, only some fraction of 7.4 ms, and the
	 * bytes it moves onto one core cost more than it saves.
	 *
	 * Across the five real models the same walk says KMAX 2048 is a win on
	 * two (gemma3 -8.9%, phi3 -5.1%, both because their K is not a multiple
	 * of 1024 and a coarser cut spends fewer runt slices) and a loss or a
	 * wash on the other three. There is no consistent win, and every one of
	 * them costs the quantiser: the K slice IS the int4 group, and one
	 * scale for a 2048 long row measured 0.1067 relative error against
	 * group 32's 0.0666. The coupling stays.
	 *
	 * ⚠⚠ ONE CLAUSE OF THAT IS TRUE ONLY OF THIS SLICER, AND IT IS THE ONE
	 * THAT SOUNDS LIKE A LAW. "The bytes it moves onto one core cost more
	 * than it saves" describes a cut with ONE GLOBAL KMAX and ONE GLOBAL
	 * NMAX, where a tensor that stops needing a K cut becomes a SINGLE
	 * slice and takes a whole core's share of the bytes with it. That is a
	 * property of the cut this file makes, not of a wider K. Cut N at the
	 * same time and the bytes do not move at all: a call whose slices come
	 * out even splits its weight bytes in half whatever the cut, so the
	 * megabyte term on the busier core is the SAME and only the task term
	 * falls.
	 *
	 * Scored with the line above over the real .gguf geometry, with the cut
	 * chosen freely PER TENSOR and capped at K <= 4096 and N <= 8192 (the
	 * widest of each that has ever run on this board -- see round 322 for
	 * why K = 8192 is not on the list), a decode token's hardware path:
	 *
	 *   Llama-3.2-1B    48825 -> 45587 us   -6.6%   tasks_hi 176 -> 88
	 *   Qwen3-0.6B      38417 -> 36678      -4.5%
	 *   gemma-3-1b      53556 -> 45958     -14.2%
	 *   gemma-4-E2B     76583 -> 70311      -8.2%
	 *   Phi-3.5-mini   142567 -> 123752    -13.2%
	 *
	 * Llama's MB_hi is 308.9 on both sides of that, which is the whole
	 * point: 88 tasks came off and not one byte moved.
	 *
	 * ⚠⚠ AND IT IS THE CEILING OF SOMETHING THAT CANNOT BE BUILT. Every one
	 * of those cuts wants a dispatch of K = 2048 or wider under a group of
	 * 1024, and one dispatch cannot cover K wider than one group -- see the
	 * long note above tensor_grouped(), which is now a measurement rather
	 * than an assertion. Pin K at 1024, keep the same free choice of N, and
	 * the whole of the win is 0.0% on Llama, +0.3% on Qwen3, -1.9% on
	 * gemma3 and -0.7% on gemma4. Only Phi-3.5 finds anything, -5.7%, and
	 * that is the deal balancing its very large tensors rather than fewer
	 * tasks: its task count goes UP while its MB_hi falls 1014.7 to 930.6.
	 *
	 * So the reachable part of the 7.4 ms task term, without touching the
	 * quantiser, is a couple of percent on four models and a deal fix on
	 * the fifth. The rest of it is behind the group.
	 */
	unsigned long calls;       /* matvec and matvec_group entries */
	unsigned long tasks_hi;    /* tasks on whichever device got more */
	double mb_hi;              /* and megabytes on that device */
	/*
	 * ⚠ AND THE SAME CALLS' TOTAL, WHICH IS WHAT MAKES mb_hi READABLE.
	 *
	 * mb_hi on its own cannot say whether a call was balanced: 16 MB on the
	 * busier core is perfect if the other core also carried 16 and a wasted
	 * core if it carried none. mb_hi / (mb_all / ndev) is 1.0 when the deal
	 * is even and 2.0 when one core did all of it.
	 *
	 * ⚠ NOT g->weight_mb, WHICH LOOKS LIKE THE SAME NUMBER AND IS NOT.
	 * charsiu_npu_matmul adds to weight_mb and never calls account_call, so
	 * a run with any prefill in it has weight_mb counting bytes mb_hi never
	 * saw. This is summed in account_call, over exactly the calls the ratio
	 * is about.
	 */
	double mb_all;
	/*
	 * The normal equations for y = A + B * tasks + C * MB, and f_yy so the
	 * fit can say how well it fits.
	 *
	 * ⚠ THE THREE PARTS ADDING TO THE TOTAL PROVES NOTHING. The first
	 * normal equation IS `A * n + B * sum(tasks) + C * sum(MB) = sum(us)`,
	 * so a least squares fit with an intercept splits the hardware path
	 * exactly, always, however badly the line describes the calls. What
	 * says it describes them is the residual, which needs sum(us * us).
	 */
	double f_n, f_t, f_m, f_tt, f_tm, f_mm, f_y, f_ty, f_my, f_yy;

	/*
	 * A wedged block answers every submit with a driver side timeout and
	 * the ioctl still returns success, so the only reliable detector is the
	 * clock. Three slow submits and this path retires itself.
	 */
	double slow_us, min_gbs;
	/*
	 * ⚠ AND HOW OFTEN, because the one-shot message on its own is a
	 * MISLEADING INSTRUMENT and it cost two rounds of wrong hypothesis.
	 *
	 * Rounds 373, 374 and 374's repeat each printed one notice, always at
	 * K=2048 N=2048, and only ever in a long context run. That looked like
	 * the NPU idling across attention's 16 to 38 ms gaps and paying to wake
	 * up. It is not, and arithmetic settles it without a board round:
	 *
	 *   - the floor is a RATE, so the stall needed to trip it scales with
	 *     the tensor. o_proj is 2.1 MB and trips on 1.0 ms; down is 8.4 MB
	 *     and needs 4.2; the output head is 131 MB and needs 65.7. o_proj
	 *     is simply the most sensitive thing being watched.
	 *   - the GROUPED path has no check at all, so q, k, v, gate and up can
	 *     never trip it. Only three tensors are watched and o_proj is the
	 *     smallest of them.
	 *   - and it cannot be per token: o_proj at 0.84 GB/s would take 98 ms
	 *     rather than 8, and the token would be 180 ms instead of 90.
	 *
	 * So it is a rare stall, and a long run trips it because it has six
	 * times as many submits, not because of the gaps. Counting them says
	 * that outright instead of leaving it to be re-guessed.
	 */
	unsigned long slow_n;
	double slow_worst;
	unsigned slow_worst_k, slow_worst_n;
	int strikes, dead, nochain, slowed, nofini, inprep, plain;
	int kfit;
	int even_ks;      /* K slices of equal width, see slice_k() */
	/*
	 * ⚠⚠ WHETHER KFIT COULD FIRE AT ALL, which is not the same question
	 * as whether it helped. The `ks--` below needs a REMAINDER: a model
	 * whose every K is a multiple of kmax has none, so KFIT leaves the
	 * dispatch plan byte for byte identical and any measured difference
	 * belongs to the measurement. A board round scored two such models as
	 * losses and would have kept the switch off for it.
	 */
	unsigned kfit_hits, kfit_seen;
	/*
	 * ⚠⚠ THE BUFFERS WITHOUT THE SLICING -- the control that separates
	 * what KFIT COSTS from what it BUYS. Turning KFIT on does two
	 * independent things: it widens five buffers to 2 * kmax, and it
	 * makes the last slice absorb the remainder. The first happens
	 * UNCONDITIONALLY, on every model, including the ones where no
	 * tensor can fire -- and those models measured a small consistent
	 * LOSS across two board rounds while their dispatch plan was
	 * provably identical in both arms. This flag does the widening and
	 * not the slicing, so the two can be priced apart instead of
	 * argued about.
	 */
	int kwide_only;
	/* one message per REASON; the pointer identifies it, see whine() */
	const char *whined[8];
	unsigned n_whined;
	int serialpack;
	/*
	 * ⚠⚠ POOLING THE READ BACK IS MEASURED SLOWER AND IS OFF. It is a
	 * parallelisation over disjoint rows and nothing else -- the text is
	 * identical on all eight models, so it is CORRECT -- and it lost on
	 * every one of them, 5% to 18% of the whole prefill, with the read
	 * itself exactly DOUBLING. See the note above read_rows.
	 */
	int poolread;              /* 0 never, 1 always, 2 when m * n >= poolread_min */
	unsigned poolread_min;
	unsigned long bread_pooled, bread_serial;   /* slots read each way */
	/*
	 * ⚠ HOW MANY TIMES Y IS WALKED, AND HOW FEW IT COULD BE. Every slot's
	 * read is one pass over its own range of the caller's Y: the first
	 * assigns and the K slices after it add. Fusing the slices a device
	 * holds for one output range would make that one pass instead of
	 * several, and whether that is worth writing depends entirely on how
	 * the deal spread the slices -- one slice per device per range and
	 * there is nothing to fuse. So count the passes and count the ranges
	 * before touching the loop.
	 */
	unsigned long bread_passes, bread_ranges;
	uint64_t bseen_dev;        /* (device, output range) pairs seen this call */
	int read4;      /* CHARSIU_NPU_READ4: four rows off one line, default on */
	/*
	 * What fraction of every projection's OUTPUT CHANNELS the CPU keeps.
	 * 0 is the hardware doing all of it, which is every round before 371.
	 */
	double cpu_frac;
	float *afscr;              /* the activation, rounded through fp16 */
	unsigned long slices;
};

static double now_us(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static unsigned env_u(const char *name, unsigned dflt)
{
	const char *e = getenv(name);

	return e ? (unsigned)strtoul(e, NULL, 0) : dflt;
}

/*
 * Say why, out loud, the first time.
 *
 * Round 315 ran a whole ladder in which the hardware never engaged, and the
 * text was right every time because the CPU quietly did the work. A silent
 * fallback is worse than a loud failure: it produces a result that looks like
 * evidence.
 */
/*
 * GROUPED SCALES FOR FREE, when the K slice IS the quantisation group.
 *
 * charsiu already cuts K into slices and sums their accumulators, so if each
 * slice covers exactly one group of the quantiser, the group's scale can be
 * applied to that slice's contribution on the way in and nothing extra has to
 * run on the hardware. Round 352's int4 sentence was English, on topic and
 * repetitive, which is what ONE absmax scale for a whole 2048 long row does to
 * four bits: measured offline, per channel RTN is 0.1067 relative error against
 * group 32's 0.0666.
 *
 * Set CHARSIU_NPU_KMAX and CHARSIU_NPU_W4_GROUP to the same value. The
 * condition is deliberately strict -- the slice must BE the group -- because a
 * slice covering part of a group would need a scale per part and there is
 * nowhere to put one.
 */
/*
 * ⚠⚠ AND "NOWHERE TO PUT ONE" IS A PROPERTY OF THE BLOCK, NOT A CHOICE THIS
 * FILE MADE. ONE DISPATCH CANNOT COVER K WIDER THAN ONE QUANTISATION GROUP.
 *
 * The sentence above was an assertion for a long time, and the obvious idea it
 * blocks -- decouple KMAX from W4_GROUP, dispatch K = 4096 with four groups of
 * 1024 scales in the weights, halve the task count without coarsening the
 * quantiser -- is worth a re-read of the register map every time somebody
 * notices the coupling. So here is why it does not work, in the form that
 * closes it.
 *
 * A dispatch produces ONE number per output channel. The CNA and the CORE
 * reduce over every input channel the dispatch was handed, and everything that
 * can touch the result afterwards lives in the DPU: BS, BN, EW and the output
 * convert, in that order, every one of them indexed by OUTPUT channel and every
 * one of them running AFTER the reduction. Their operand surfaces come from the
 * DPU_RDMA, whose DATA_CUBE_CHANNEL at 0x5014 is the output channel count.
 * Nothing in the map segments the K reduction or writes a partial sum per range
 * of K -- DPU_SURFACE_ADD at 0x40c0 comes closest and it ADDS surfaces, with no
 * per-surface operand. So the limit is not the coefficient FORMAT, which is
 * only a table of ceil(n/8) 64 byte records plus one fp16 a channel and could
 * be widened in an afternoon. It is the accumulator, which cannot.
 *
 * On the int4 path charsiu does not even use the DPU's multiplier: acc_out
 * forces the whole vendor output stage on, the requant reads identity
 * (0x40ac/b0/b4 = 0, 1, 0), the raw accumulator comes back and the group scale
 * is applied on the CPU by scaled_add. That is why the gather in add_slice can
 * take ONE scale a channel for the slice and no more.
 *
 * Both ways round it close. A factor folded into the ACTIVATION has to be the
 * same for every output channel -- which is exactly why the per k AWQ factor in
 * npuquant.c is free, and exactly why a per (channel, group) scale is not.
 * Folding it into the WEIGHTS means multiplying a four bit code by a ratio and
 * rounding it back into four bits, which is per channel quantisation with extra
 * steps.
 *
 * ⚠⚠ AND THE VENDOR'S K = 4096 DISPATCH IS NOT DOING WHAT OURS WOULD HAVE TO.
 * That is the part worth having, because their compiled streams are the only
 * evidence available for what this block will accept, and the shape of their
 * cut -- 2 x K=2048 N=1024 for q and o, 2 x K=2048 N=4096 for gate and up,
 * 4 x K=4096 N=1024 for down, against our 1024 wide K pieces -- reads like a
 * demonstration that a wide K under a fine group runs. It is not one.
 *
 * Read out of Llama-3.2-1B-Instruct-rk3576-w4a16.rkllm with
 * tools/rkllm_regcmd.py, three things say so and they agree:
 *
 *   - the weight byte count in CNA 0x101c is EXACTLY ic * oc / 2. 0x100000 at
 *     K=2048 N=1024, 0x200000 at K=4096 N=1024. Nibbles and nothing else;
 *     there is no room beside them for a scale table;
 *   - the int4 convolution stream carries no DPU_RDMA registers at all, so no
 *     coefficient surface is fetched for it -- 0x5020 and 0x5024 are never
 *     written; and
 *   - its requant is the identity, 0x40ac/b0/b4 = 0, 1, 0, the same three
 *     values this file copies.
 *
 * So no per group scale enters their dispatch either. And the file says what
 * their group actually is, in the open: the scales are stored as contiguous
 * fp32 runs and the length of a run is the tensor's OUTPUT CHANNEL COUNT.
 * Layer 0 begins at byte 0x1fc77da0 and goes 2048, 512, 512, 2048, 8192, 8192,
 * 2048 -- q, k, v, o, gate, up, down -- each run followed by an equally long
 * run of integer valued floats, which is the zero point. That block repeats for
 * 16 layers, and the embedding and head share one run of 128256. 505088 scales
 * for 1235746816 weights.
 *
 * ⚠ AND THE RUNS WERE IDENTIFIED RATHER THAN GUESSED, because a length on its
 * own could be a norm. Correlating each run ELEMENT BY ELEMENT against the per
 * row dynamic range of the matching tensor in the q8_0 copy of the same model
 * gives +0.93 for attn_q, +0.97 for attn_k, +0.91 for attn_v, +0.97 for
 * attn_output, +0.99 for ffn_gate, +0.84 for ffn_up and +0.98 for ffn_down.
 * Row r of the run is the scale of row r of that tensor. (The ratio of range to
 * scale is not the same constant across the seven -- 4.4 on attn_q against 15.2
 * on attn_output -- which is the other half of the point: their quantiser is
 * doing something to the weights before it rounds them, and getting quality out
 * of one scale a row is a quantiser result, not a dispatch one.)
 *
 * ONE SCALE AND ONE ZERO POINT PER ROW. Their group is the whole of K, 8192
 * long for down. A per row scale factors straight out of a K sum, so their K
 * cut is free the way an UNGROUPED tensor's K cut is free here, and they choose
 * 4096 because 8192 does not run (round 322 hit the same wall from the other
 * side: 0.65 GB/s and 131 job timeouts) and choose N to hand the second core an
 * equal share. Their wide K is not a finer group surviving a wide dispatch. It
 * is no group at all.
 *
 * ⚠ WHICH ALSO PRICES THE ONLY DOOR LEFT, and it is a quantiser question rather
 * than a dispatch one: go where the vendor is, one scale a row, and pay for it
 * with a calibrated quantiser instead of RTN. Measured offline on the real
 * weights with this file's own rule (d = vmax / -8), relative Frobenius error
 * of the reconstruction, group 1024 against one scale a row: attn_q 0.1427 ->
 * 0.1518, attn_output 0.1622 -> 0.1783, ffn_down (K = 8192) 0.1402 -> 0.1707.
 * The per k AWQ factor takes most of that back on the K = 2048 tensors (attn_q
 * 0.1409, attn_output 0.1596, both BELOW the grouped number) and does not on
 * ffn_down, which is the one that would gain the most tasks. Weight error is
 * not the objective -- npuquant.c says why -- so that is a starting price, not
 * a verdict.
 */
/*
 * acc[j] += fo[j] * sc[j], AND BIT FOR BIT WHAT THE SCALAR LOOP DID.
 *
 * Two things make that claim checkable rather than hopeful. The elements are
 * independent -- each j lands in its own accumulator, so there is no reduction
 * whose order could change. And the product of two floats has at most 48
 * significant bits, which is exact in a double, so the scalar loop's
 * (float)((double)a * (double)b) is the correctly rounded float product and
 * nothing else.
 *
 * ⚠ THE BARRIER IS LOAD BEARING. Without it the compiler fuses the multiply
 * and the add into fmla, which rounds ONCE where the source rounds twice, and
 * 460190 of 20.5 million accumulations came out different in the host check.
 * The cast in the scalar tail is the same barrier written in C.
 */
#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
#include <arm_neon.h>
static void scaled_add(float *acc, const float *fo, const float *sc, unsigned n)
{
	unsigned j = 0;

	for (; j + 4 <= n; j += 4) {
		float32x4_t p = vmulq_f32(vld1q_f32(fo + j), vld1q_f32(sc + j));

		__asm__("" : "+w"(p));
		vst1q_f32(acc + j, vaddq_f32(vld1q_f32(acc + j), p));
	}
	for (; j < n; j++)
		acc[j] += (float)((double)fo[j] * (double)sc[j]);
}
#else
static void scaled_add(float *acc, const float *fo, const float *sc, unsigned n)
{
	for (unsigned j = 0; j < n; j++)
		acc[j] += (float)((double)fo[j] * (double)sc[j]);
}
#endif

/*
 * THE CPU'S SHARE OF A PROJECTION, run in the window the calling thread
 * currently spends BLOCKED in prep_bo.
 *
 * y[r] = sum over groups of scale[r][g] * sum over the group of code * a,
 * which is the same arithmetic the hardware does: the same int4 codes, the
 * same per group scale, and an activation rounded through fp16 first so both
 * halves of the split see the same numbers.
 *
 * ⚠ ONE THREAD, DELIBERATELY. Round 370 measured the CPU reading memory at
 * 7.13 GB/s on one thread, 6.65 on two and 6.32 on four: a single thread
 * already saturates the controller, so a fan out here would cost a
 * synchronisation and buy nothing. It is also why this is worth trying at all
 * -- the same round found the NPU and a CPU reader reaching 15.46 GB/s
 * together against 10.46 for the NPU alone.
 */
static void cpu_rows(const struct npu_entry *e, const float *af, float *y)
{
	const struct npu_tensor *t = e->t;
	uint64_t k = t->k;
	uint64_t grp = t->kgroup ? t->kgroup : k;
	uint64_t ngrp = (k + grp - 1) / grp;
	size_t per = ((size_t)k + 1) / 2;
	unsigned nc = (unsigned)t->n - e->n_npu;

	for (unsigned r = 0; r < nc; r++) {
		const uint8_t *row = e->cq + (size_t)r * per;
		const float *sc = t->scale + (size_t)(e->n_npu + r) * ngrp;
		double acc = 0.0;

		for (uint64_t gi = 0; gi < ngrp; gi++) {
			uint64_t lo = gi * grp;
			uint64_t hi = lo + grp < k ? lo + grp : k;
			uint64_t i = lo;
			float part = 0.0f;

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
			{
			float32x4_t a0 = vdupq_n_f32(0.0f);
			float32x4_t a1 = vdupq_n_f32(0.0f);
			float32x4_t a2 = vdupq_n_f32(0.0f);
			float32x4_t a3 = vdupq_n_f32(0.0f);

			/*
			 * ⚠ THE VECTOR PATH READS BYTE i/2 AND TAKES ITS LOW
			 * NIBBLE AS WEIGHT i, so it only means that when i is
			 * EVEN. Every group in this runtime starts on a
			 * multiple of kmax and is even, but a group that
			 * started odd would silently read every weight off by
			 * one: at k = 34 with groups of 17 the test measured
			 * 2.26 relative against 1e-6 elsewhere. One scalar
			 * step fixes the alignment.
			 */
			if ((i & 1) && i < hi) {
				part += (float)(((row[i >> 1] >> 4) >= 8)
					? (row[i >> 1] >> 4) - 16
					: (row[i >> 1] >> 4)) * af[i];
				i++;
			}
			for (; i + 16 <= hi; i += 16) {
				uint8x8_t b = vld1_u8(row + (i >> 1));
				int8x8_t l = vshr_n_s8(vshl_n_s8(
					vreinterpret_s8_u8(vand_u8(b,
						vdup_n_u8(0x0f))), 4), 4);
				int8x8_t h = vshr_n_s8(vreinterpret_s8_u8(b), 4);
				int8x8x2_t z = vzip_s8(l, h);
				int16x8_t w;

				w = vmovl_s8(z.val[0]);
				a0 = vfmaq_f32(a0,
					vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))),
					vld1q_f32(af + i));
				a1 = vfmaq_f32(a1,
					vcvtq_f32_s32(vmovl_s16(vget_high_s16(w))),
					vld1q_f32(af + i + 4));
				w = vmovl_s8(z.val[1]);
				a2 = vfmaq_f32(a2,
					vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))),
					vld1q_f32(af + i + 8));
				a3 = vfmaq_f32(a3,
					vcvtq_f32_s32(vmovl_s16(vget_high_s16(w))),
					vld1q_f32(af + i + 12));
			}
			/* four accumulators, not two: it halves the
			 * dependency chain and the summation error with it */
			part += vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1),
						     vaddq_f32(a2, a3)));
			}
#endif
			for (; i < hi; i++) {
				uint8_t byte = row[i >> 1];
				int v = (i & 1) ? (byte >> 4) : (byte & 0xf);

				part += (float)(v >= 8 ? v - 16 : v) * af[i];
			}
			acc += (double)part * (double)sc[gi];
		}
		y[e->n_npu + r] = (float)acc;
	}
}

static int tensor_grouped(const struct charsiu_npu *g, const struct npu_tensor *t)
{
	return g->w4 && t->kgroup && t->kgroup < t->k &&
	       (t->k % t->kgroup) == 0 && t->kgroup == (uint64_t)g->kmax;
}

/*
 * ⚠⚠ ONCE PER REASON, NOT ONCE PER RUN.
 *
 * This printed the first refusal and then went silent for the rest of the
 * process, which is exactly backwards: the first refusal is usually a tensor
 * nobody cares about and the interesting one comes later. A gemma3 run with
 * CHARSIU_NPU_MAXN=262144 staged 182 tensors of 183 -- the 128256 wide output
 * head, forty percent of its token, silently stayed on the CPU -- and the log
 * carried no reason at all because something earlier had already used up the
 * one message.
 *
 * A reason is a string literal here, so comparing the POINTER is enough to
 * tell two apart, and eight of them is more than this file has.
 */
static void whine(struct charsiu_npu *g, const char *what, unsigned k, unsigned n)
{
	unsigned i;

	/*
	 * ⚠ NOT INTO A CONVERSATION. "NOT on the NPU -- int4 computes one row
	 * (K=2048 N=8192)" is exactly the line a board round needs and exactly
	 * the line somebody who typed a question should never see. It is not an
	 * error: the tensor took the CPU and the answer is correct.
	 */
	if (!charsiu_diag())
		return;
	for (i = 0; i < g->n_whined; i++)
		if (g->whined[i] == what)
			return;
	if (g->n_whined < sizeof(g->whined) / sizeof(*g->whined))
		g->whined[g->n_whined++] = what;
	fprintf(stderr, "charsiu: NOT on the NPU -- %s (K=%u N=%u)\n", what, k, n);
}

/*
 * ⚠⚠ THE WIDEST K A SINGLE SLICE CAN CARRY. Every buffer that holds one must
 * be sized by this and not by kmax.
 *
 * CHARSIU_NPU_KFIT makes the last K slice ABSORB the remainder instead of
 * being it -- the `ks--` in charsiu_npu_add -- so that slice runs from a kmax
 * boundary to the end of the tensor and is up to 2 * kmax - 1 wide.
 *
 * When KFIT was written, scratch and wpack were widened for it and the
 * BATCHED buffers were not: bin_stride, bscr and bq were all still kmax * m.
 * The batched packer gathers one slice's whole K into bscr, quantises it into
 * bq and packs it into bin at bin_stride, so all three overran by up to 2x.
 * bscr and bq are plain mallocs, so the board answered with
 *
 *     malloc(): corrupted top size
 *     Aborted
 *
 * on the first staged model, every time, on gemma-3-1b and gemma-4-E2B alike.
 *
 * ⚠ AND NO HOST COULD SEE IT. With no /dev/accel the NPU never opens, nothing
 * is staged and no batched buffer is allocated, so KFIT measured "text
 * identical and slightly faster" on the desk while it aborted on the card.
 * The switch had sat in this tree described as written, legal and default
 * off, and the round that priced it repeated "already legal". Nothing had
 * ever run it.
 *
 * One function, so the fifth place that needs this bound cannot be the one
 * that gets forgotten.
 */
static size_t kmax_wide(const struct charsiu_npu *g)
{
	return (size_t)g->kmax * ((g->kfit || g->kwide_only) ? 2 : 1);
}

struct charsiu_npu *charsiu_npu_open(unsigned max_k, unsigned max_n,
				     unsigned max_tensors)
{
	return charsiu_npu_open_mode(max_k, max_n, max_tensors, -1);
}

/*
 * ⚠ want_w4 = -1 ASKS THE ENVIRONMENT, 0 AND 1 DECIDE.
 *
 * A caller that batches cannot let the environment choose. w4a16 computes
 * exactly one row whatever it is asked for -- five rounds established that and
 * no register changes it -- so a vision tower whose device opened in int4
 * because the runner's config sets CHARSIU_NPU_W4V=1 would dispatch its 1024
 * patches ONE AT A TIME. It would still be correct, and it would be slower than
 * the CPU it was moved off.
 */
/* defined after charsiu_npu_close, where the hold is dropped too */
static void qos_hold(struct charsiu_npu *g, int say);

struct charsiu_npu *charsiu_npu_open_mode(unsigned max_k, unsigned max_n,
					  unsigned max_tensors, int want_w4)
{
	struct charsiu_npu *g = calloc(1, sizeof(*g));
	unsigned ns, ks;

	if (!g)
		return NULL;
	g->qos_fd = -1;
	g->dev[0] = charsiu_open(NULL);
	g->ndev = 1;
	if (g->dev[0] && !getenv("CHARSIU_NPU_ONEDEV")) {
		g->dev[1] = charsiu_open(NULL);
		if (g->dev[1])
			g->ndev = 2;
		else
			fprintf(stderr, "charsiu: only one NPU file could be "
				"opened; the second core stays idle\n");
	}
	if (!g->dev[0]) {
		free(g);
		return NULL;
	}
	/*
	 * The defaults are the widest slice MEASURED to give identical tokens,
	 * not the narrowest that was ever verified.
	 *
	 * Round 321 swept them and the cost turned out to be per TASK rather
	 * than per submit -- round 319 had varied tasks per submit at a fixed
	 * 606 slices and got a flat line, so the two sweeps together say the
	 * submit is nearly free and the task is about 35 us:
	 *
	 *   slices  606    542    319    287
	 *   tok/s   5.55   5.62   5.83   5.90
	 *
	 * Round 322 pushed it further and found the limit is on K and not on N:
	 *
	 *   slices  287    192    144    128
	 *   tok/s   5.90   6.08   6.09   0.55 <- K=8192, 131 job timeouts
	 *
	 * ⚠ N = 8192 DOES NOT WEDGE. That is the shape round 313 needed a power
	 * cycle to escape, so 313's hang was the requantised byte output or the
	 * 67 MB coefficient buffer, NOT the width. K = 8192 is the one that
	 * collapses, to 0.65 GB/s, and it collapses rather than hanging.
	 */
	g->nmax = env_u("CHARSIU_NPU_NMAX", 8192);
	g->kmax = env_u("CHARSIU_NPU_KMAX", 4096);
	g->slow_us = (double)env_u("CHARSIU_NPU_SLOW_US", 100000);
	/*
	 * ⚠ IT DOES NOT SPLIT THE SUBMIT. Whatever its name and its older
	 * comment suggested, the only thing that reads `nochain` scales the
	 * slow-job threshold by the chain length. Setting it changes what
	 * gets WARNED about, never what gets submitted.
	 */
	g->nochain = getenv("CHARSIU_NPU_NOCHAIN") != NULL;
	/*
	 * 0 is unlimited. A cap exists because the output head is 126 chained
	 * tasks and 253 buffer handles in one submit, and it reached only
	 * 4.2 GB/s where an eight task submit reaches 10.
	 */
	/*
	 * ⚠⚠ AND IT HAS NEVER CAPPED ANYTHING. `maxtask` is assigned here and
	 * read NOWHERE in this tree. The paragraph above describes the
	 * measurement that motivated it -- the head's 126 chained tasks at
	 * 4.2 GB/s against an eight task submit's 10 -- and that hypothesis
	 * has therefore never had a working control: every round that set
	 * CHARSIU_NPU_MAXTASK to test it measured its own baseline twice.
	 *
	 * CHARSIU_NPU_NOCHAIN is the same story with a smaller blast radius:
	 * it is read once, and only to scale the slow-job threshold, not to
	 * split the submit its own comment claims it splits.
	 *
	 * A variable that is read and then ignored is worse than one that does
	 * not exist, because a round can be built on it. It says so now, and
	 * it will keep saying so until something reads it.
	 */
	g->maxtask = env_u("CHARSIU_NPU_MAXTASK", 0);
	if (g->maxtask) {
		static int said;

		if (!said++)
			fprintf(stderr, "charsiu: CHARSIU_NPU_MAXTASK=%u is "
				"IGNORED -- nothing in this tree reads it, so "
				"this run is the same as one without it\n",
				g->maxtask);
	}
	/*
	 * ⚠ THE RETIREMENT GUARD WAS BLIND TO A THIRTEEN FOLD SLOWDOWN.
	 *
	 * Round 322's K = 8192 rung ran at 0.65 GB/s with 131 driver timeouts
	 * and 3718 IOMMU errors, and nothing printed: the budget is 100 ms plus
	 * a millisecond a megabyte, which is 1 GB/s, and 0.65 sat just under it
	 * while every submit stayed inside the flat allowance.
	 *
	 * Retiring the path on that would be wrong -- the answers were still
	 * correct, it was only slow -- so this is a separate, non fatal notice.
	 * "It stopped answering" and "it is thirteen times slower than it has
	 * ever been" are different things and a run should say which.
	 */
	g->min_gbs = (double)env_u("CHARSIU_NPU_MIN_MBPS", 2000) / 1000.0;
	g->max_n = max_n;
	g->ent_cap = max_tensors;

	/*
	 * ⚠ AN EMPTY VALUE MEANS OFF, and it did not. `!= NULL` makes
	 * CHARSIU_NPU_W4V= turn int4 ON, which is the opposite of what anybody
	 * types it for, and there is no other way to get int8 past a runner
	 * that sets the variable itself. A board round meant to measure the
	 * int8 batched path ran int4 and said so in its own report, which is
	 * the only reason it was caught.
	 *
	 * `*e != '0'` is what npu_mode() and act_set() in this tree already do.
	 */
	if (want_w4 >= 0) {
		g->w4 = want_w4 ? 1 : 0;
	} else {
		const char *e4 = getenv("CHARSIU_NPU_W4V");

		g->w4 = e4 && *e4 && *e4 != '0';
	}
	/*
	 * SKIP THE FLUSH ON A BUFFER THE CPU ONLY READ.
	 *
	 * fini_bo is dma_sync_sgtable_for_device, which on arm64 CLEANS every
	 * line of the buffer to the point of coherency. That is what a buffer
	 * the CPU WROTE needs -- the activation buffer does need it -- but an
	 * output buffer is only ever read here, so its lines are clean already
	 * and the walk writes nothing back. What makes skipping it safe is that
	 * the next read is preceded by prep_bo, whose sync_for_cpu invalidates
	 * those stale clean lines before the CPU can see them.
	 *
	 * ROUND 367 RAN IT: 3.4 ms a token of flush went to 0.02, the sentence
	 * came back word for word identical to the arm that kept the flush and
	 * to the one device control, and the run went 9.86 to 10.26 tok/s. So it
	 * is the default now, and CHARSIU_NPU_FINI puts the clean back.
	 */
	g->nofini = getenv("CHARSIU_NPU_FINI") == NULL;
	/*
	 * AND THE SAME ARGUMENT ON THE OTHER SIDE, which round 367 did NOT run.
	 *
	 * prep_bo on the ACTIVATION buffer waits for a write fence and then
	 * invalidates. Neither is needed: the device only ever READS that
	 * buffer, so there is no write fence to wait for, and the CPU is about
	 * to overwrite every byte the next job will read, so there is nothing
	 * stale worth dropping first. Its fini stays -- the CPU wrote it, and
	 * that clean is what makes the bytes visible to the hardware.
	 *
	 * What makes it safe to write at all is the ORDER: the previous job on
	 * this buffer finished before this call, because its output prep waited
	 * on the fence that covers the whole job, reads included.
	 *
	 * It is 65 entries a token times two devices: 130 ioctls and 130 cache
	 * walks. CHARSIU_NPU_INPREP puts them back.
	 */
	g->inprep = getenv("CHARSIU_NPU_INPREP") != NULL;
	/*
	 * ONE SWITCH FOR THE THREE THINGS ROUND 369 CHANGED that have no
	 * behaviour of their own to show: the vectorised half conversion, the
	 * activation reaching the packer without a copy, and the sum landing in
	 * the caller's buffer instead of a staging one. All three are provably
	 * neutral -- the packer was compared byte for byte against the old loop
	 * at thirteen shapes -- so the control is not about whether they are
	 * right. It is so a round that comes out SLOWER can say which of them
	 * did it, which is the lesson round 368's attention arm taught.
	 */
	g->plain = getenv("CHARSIU_NPU_PLAIN") != NULL;
	/*
	 * ⚠ THE LEGACY BIT PATTERN LAYOUT CANNOT BE SPLIT. It accumulates with
	 * |=, so two channels can share a byte and two threads would race for
	 * it. charsiu_pack_weights_rows does not implement that layout at all,
	 * which would silently produce the CURRENT one instead, so the check
	 * belongs here rather than in a comment.
	 */
	g->serialpack = g->plain || getenv("CHARSIU_W4_BITPAT") != NULL;
	{
		const char *e = getenv("CHARSIU_NPU_CPU_FRAC");

		g->cpu_frac = e ? atof(e) : 0.0;
		if (g->cpu_frac < 0.0)
			g->cpu_frac = 0.0;
		if (g->cpu_frac > 0.9)
			g->cpu_frac = 0.9;
	}
	g->midrise = g->w4 && getenv("CHARSIU_NPU_W4_MIDRISE") != NULL;
	/*
	 * ⚠ THE RUNT K SLICE, AND WHAT IT COSTS. ceil(k / KMAX) leaves the
	 * remainder in a slice of its own, and a slice costs about a task --
	 * round 321 measured 35 us of it -- whatever its width.
	 *
	 * Every llama dimension is a power of two and divides the 1024 slice
	 * exactly, so this never arose. gemma3 is n_embd 1152: q, k, v, gate
	 * and up all split 1024 + 128, which is five slices a layer that carry
	 * an ninth of a slice of work. Its board log reads 468 slices where 338
	 * would do, and 6.16 GB/s where llama reaches 10.3.
	 *
	 * KFIT lets the LAST slice absorb the remainder instead, so 1152 is one
	 * slice rather than two. A slice can then be up to 2 * KMAX - 1 wide,
	 * which is what the sizes below have to allow for.
	 *
	 * ⚠ UNGROUPED TENSORS ONLY. A grouped tensor carries one scale per
	 * (channel, K group) and the gather in add_slice reads the group at
	 * k0 / kgroup, so a slice that spans two groups would apply the first
	 * group's scale to both. Ungrouped ones are scaled once at the end and
	 * their K split is free: acc_out sums int32 across the slices, so any
	 * split of the same K gives the same accumulator.
	 *
	 * Off until a board round says it is both correct and faster.
	 *
	 * ⚠ AND THE "UNGROUPED TENSORS ONLY" RESTRICTION COSTS IT NOTHING ON
	 * THE MODELS IT IS FOR, which is not obvious and is why it is written
	 * down. npuquant.c falls back to one scale a row whenever k % grp is
	 * nonzero -- `if (k % grp) grp = k;` -- so a tensor whose K does not
	 * divide the slice is ALREADY ungrouped, and a tensor whose K does
	 * divide it has no remainder for KFIT to absorb. The two conditions are
	 * complementary. gemma3 (K 1152 and 6912) and gemma4's q, k, v, gate
	 * and up (K 1536) are ungrouped today, so KFIT applies to every tensor
	 * it would help and is blocked on nothing but a board round.
	 *
	 * The offline walk over the fitted line prices it: gemma3 53556 ->
	 * 49347 us a token, -7.9%, and gemma4 76583 -> 73870, -3.5%. Llama,
	 * Qwen3 and Phi-3.5 are unchanged, all their K being multiples of 1024.
	 */
	g->kfit = getenv("CHARSIU_NPU_KFIT") != NULL;
	/*
	 * ⚠ EQUAL K SLICES, AND WHAT MADE IT WORTH ASKING. slice_k() gives
	 * every slice KMAX and lets the last one take the remainder, so
	 * Phi-3.5's K = 3072 at KMAX 2048 is 2048 + 1024 and Qwen2.5's
	 * K = 8960 is 2048 x4 + 768. Two things follow from the unequal
	 * widths, and phase 9 on 2026-09-05 measured the second:
	 *
	 *   - the two cores get unequal work on every K sliced tensor, and
	 *     the fence waits for the wide one;
	 *   - deal_pick balances by accumulated load, so a wide slice and a
	 *     narrow one leave the loads uneven and the NEXT tensor gets the
	 *     opposite assignment. The map flips per tensor, so a follower's
	 *     slice is always on the device that does not hold it. Phi-3.5
	 *     reused its packed input 0 times out of 2304 asks and gemma4 0
	 *     of 528, and every one of those misses was counted as "the K
	 *     slices are on the other device".
	 *
	 * Equal slices cost deal_pick the same load twice, so the assignment
	 * is stable by construction and the two cores get equal work -- one
	 * change for both, without touching the dealer or the reuse key.
	 *
	 * ⚠ OFF, AND THE BOARD IS WHY. Phase 2 is clean on nine models with it
	 * on, so it is correct; it is just not worth anything. gemma4's 528
	 * misses went to 0 and its prompt moved 30110 -> 29884 ms, which is
	 * inside the spread, and Phi-3.5 did not move at all because at the
	 * KMAX llama.c picks for it (1024) its slices were already even.
	 * kslice.h has the whole result.
	 */
	g->even_ks = getenv("CHARSIU_NPU_EVEN_KS") != NULL;
	/*
	 * 🏁 2026-09-05: THE POOLED READ IS ON, ABOVE A SIZE. The note below
	 * priced it when the read was 241 ms and the barrier 190 of that; at
	 * today's shapes the read is the largest share of a batched matmul
	 * (27 to 51%) and the work a dispatch is tens of times the barrier.
	 * Phase 9 with both arms, governor pinned, 916 tokens, text identical
	 * on all nine models: Phi-3.5 35618 -> 33620 ms, SmolLM2-1.7B 19066 ->
	 * 17370, gemma-3 14951 -> 13487, gemma-4 31360 -> 28919, tinyllama
	 * 13045 -> 12212, SmolLM2-135M 5894 -> 5391 -- and Qwen2.5 16441 ->
	 * 17219, Qwen3 12145 -> 13216. The two that lost have the narrowest
	 * tensors, which is the old note still being right for small work.
	 * So: pool a slot's rows when m * n reaches CHARSIU_NPU_POOL_READ_MIN
	 * elements (default 262144, one megabyte of floats), never below.
	 * CHARSIU_NPU_POOL_READ=1 pools always, =0 never; both are arms.
	 */
	{
		const char *e = getenv("CHARSIU_NPU_POOL_READ");

		g->poolread = !e || !*e ? 2 : *e == '0' ? 0 : 1;   /* 2 = by size */
		g->poolread_min = env_u("CHARSIU_NPU_POOL_READ_MIN", 262144);
	}
	/*
	 * ⚠⚠ OFF. The host said 1.4 to 2.2x faster and THE BOARD SAID 2.3x
	 * SLOWER, on every one of eight models, same night (phase 9,
	 * 2026-09-03: Qwen3 read 3234 ms with it against 1339 without, Phi-3.5
	 * 27770 against 12408, gemma4 18030 against 7071; pack and fence did
	 * not move, so the column isolates it). Four rows off one line means
	 * one read stream and FOUR write streams whose rows sit n floats
	 * apart -- 4 KB on Qwen3, 32 KB on the wide slices -- and the A72's
	 * L1 is 32 KB two way with a store buffer that does not merge four
	 * interleaved partial lines; the host's core does. The bytes argument
	 * that carried vision's attention across did not carry this: it
	 * changed which SIDE of the copy is scattered, and the board's side
	 * cost more. CHARSIU_NPU_READ4=1 is the probe; tools/bench_gather is
	 * the host number that was wrong about this board.
	 */
	g->read4 = getenv("CHARSIU_NPU_READ4") ? atoi(getenv("CHARSIU_NPU_READ4")) : 0;
	if (g->read4 == 1)
		g->read4 = 4;
	/*
	 * =2 was the half step: two rows off one line, one read stream and TWO
	 * write streams. The board priced it 2026-09-04, phase 9: read slower
	 * on all eight models, +15% to +40% (Qwen3 1219 to 1558 ms, gemma4
	 * 6747 to 8382). One write stream is what this core wants; the row
	 * loop stays.
	 */
	g->kwide_only = !g->kfit && getenv("CHARSIU_NPU_KFIT_WIDE") != NULL;
	/*
	 * ⚠ THE CONTROL FOR THE DEAL. `di = (ki * ns + ni) & 1` was the
	 * assignment every number in this file before round 391 was measured
	 * with, so it has to stay reachable in one boot beside its replacement
	 * -- see the long note on g->deal_load. It is exactly neutral on
	 * Llama-3.2-1B, whose dimensions are all powers of two, which is the
	 * cheapest way for a board round to check the switch itself works.
	 */
	g->deal_index = getenv("CHARSIU_NPU_DEAL_INDEX") != NULL;
	ns = (max_n + g->nmax - 1) / g->nmax;
	ks = (max_k + g->kmax - 1) / g->kmax;
	g->max_slices = ns * ks;
	g->slot_cap = max_tensors * g->max_slices;

	{
		unsigned kwide = (unsigned)kmax_wide(g);
		struct charsiu_matmul widest = { 1, kwide, g->nmax,
						 CHARSIU_INT8, CHARSIU_INT8 };

		g->in_stride = charsiu_entries_per_row(&widest) * 64;
		g->out_stride = g->nmax * 4;
		/* an fp16 activation is two bytes where an int8 one is one */
		if (g->w4)
			g->in_stride *= 2;
	}

	g->ent = calloc(g->ent_cap, sizeof(*g->ent));
	g->slot = calloc(g->slot_cap, sizeof(*g->slot));
	/* ⚠ the widest a slice can be, which KFIT doubles -- see above */
	g->scratch = malloc((size_t)g->nmax * kmax_wide(g) + max_k);
	g->acc = calloc(max_n, sizeof(*g->acc));
	g->accf = calloc(max_n, sizeof(*g->accf));
	g->fscr = calloc(max_k ? max_k : 1, sizeof(*g->fscr));
	g->afscr = calloc(max_k ? max_k : 1, sizeof(*g->afscr));
	g->wpack = malloc((size_t)g->nmax * kmax_wide(g) + 4096);
	g->asum = calloc(ks ? ks : 1, sizeof(*g->asum));
	/* a GROUP can carry several tensors' slices, so four times over */
	g->tasks = calloc(4 * g->max_slices, sizeof(*g->tasks));
	g->handles = calloc(1 + 8 * g->max_slices, sizeof(*g->handles));
	if (!g->ent || !g->slot || !g->scratch || !g->acc || !g->accf ||
	    !g->fscr || !g->afscr || !g->wpack || !g->asum || !g->tasks || !g->handles)
		goto fail;

	/* one activation buffer per device: a buffer object belongs to the file
	 * that made it, so the two cannot share one */
	for (unsigned d = 0; d < g->ndev; d++)
		if (charsiu_bo_alloc(g->dev[d],
				     (size_t)g->in_stride * ks + 4096, &g->in[d]))
			goto fail;

	/*
	 * ⚠⚠ HOLD THE CPUs OUT OF DEEP IDLE WHILE THE NPU IS OPEN.
	 *
	 * rk3576.dtsi gives CPU_SLEEP an exit latency of 250 us. A decode step
	 * is about 150 calls, and each call is several wakeups -- the irq
	 * thread, the scheduler thread, the waiter, the CPU thread pool's
	 * workers -- on CPUs that had nothing to do for the 300 us the fence
	 * took and went to sleep. Phase 21 on the board, same prompt, same
	 * kernel, ondemand governor:
	 *
	 *   deep idle allowed         7.64 tok/s
	 *   CPU_SLEEP disabled        9.46 tok/s     +24%
	 *   spin on the fence only    8.06 tok/s     +5%   (the waiter's share)
	 *
	 * so most of it is paid by the kernel threads, not by the waiter, and
	 * a spinning waiter cannot buy it back. What can, from userspace, is
	 * the PM QoS interface: a process that writes a latency bound to
	 * /dev/cpu_dma_latency and keeps the file open forbids every idle
	 * state whose exit latency exceeds it, on every CPU, until it closes
	 * the file. Audio and network stacks do exactly this. 100 us allows
	 * WFI and forbids CPU_SLEEP; the fd goes away with the device, and
	 * with the process if it dies.
	 *
	 * CHARSIU_NPU_IDLE=1 leaves the CPUs alone (the control), and
	 * CHARSIU_NPU_DMA_LATENCY_US moves the bound. Needs root, which the
	 * board has; without it this says so once and carries on.
	 */
	qos_hold(g, 1);
	return g;

fail:
	charsiu_npu_close(g);
	return NULL;
}

/*
 * ⚠ WILL A BATCH BE TAKEN, ASKED BEFORE ONE IS TRIED. A caller that has two
 * strategies has to choose before it acts: trying the batch and falling back
 * has already done the work of one of them.
 */
int charsiu_npu_batches(const struct charsiu_npu *g)
{
	return g && !g->w4;
}

/*
 * Take the PM QoS hold described above charsiu_npu_open_mode's call, if it
 * is not held already and the control has not been asked for. `say` prints
 * the one line about it: the open says it, a server taking the hold back
 * before every request does not.
 */
static void qos_hold(struct charsiu_npu *g, int say)
{
	const char *e;
	int32_t us;
	int fd;

	if (g->qos_fd >= 0 || getenv("CHARSIU_NPU_IDLE"))
		return;
	e = getenv("CHARSIU_NPU_DMA_LATENCY_US");
	us = e ? (int32_t)atoi(e) : 100;
	fd = open("/dev/cpu_dma_latency", O_RDWR | O_CLOEXEC);
	if (fd >= 0 && write(fd, &us, sizeof(us)) == (ssize_t)sizeof(us)) {
		g->qos_fd = fd;
		if (say)
			fprintf(stderr, "charsiu NPU: holding the CPUs out of idle "
				"states deeper than %d us while the NPU is open "
				"(CHARSIU_NPU_IDLE=1 to allow them)\n", (int)us);
	} else {
		int err = errno;

		if (fd >= 0)
			close(fd);
		if (say)
			fprintf(stderr, "charsiu NPU: could not hold "
				"/dev/cpu_dma_latency (%s); deep idle stays "
				"allowed, which costs about a quarter of decode\n",
				strerror(err));
	}
}

/*
 * A server holds the device for days and sits at accept() between requests;
 * the hold is for a request, not for the process. See charsiu_llm.h.
 */
void charsiu_npu_idle(struct charsiu_npu *g, int idle)
{
	if (!g)
		return;
	if (idle) {
		if (g->qos_fd >= 0) {
			close(g->qos_fd);      /* the CPUs may sleep deeply again */
			g->qos_fd = -1;
		}
	} else {
		qos_hold(g, 0);
	}
}

void charsiu_npu_close(struct charsiu_npu *g)
{
	if (!g)
		return;
	charsiu_npu_idle(g, 1);
	if (g->dev[0]) {
		for (unsigned i = 0; i < g->n_slot; i++) {
			unsigned d = g->slot[i].di;

			charsiu_bo_free(g->dev[d], &g->slot[i].wt);
			charsiu_bo_free(g->dev[d], &g->slot[i].coef);
			charsiu_bo_free(g->dev[d], &g->slot[i].regcmd);
		}
		for (unsigned i = 0; i < g->n_slot; i++)
			free(g->slot[i].sc);
		for (unsigned i = 0; i < g->n_ent; i++)
			free(g->ent[i].cq);
		for (unsigned i = 0; i < g->n_ent; i++)
			for (unsigned d = 0; d < g->ndev; d++)
			charsiu_bo_free(g->dev[d], &g->ent[i].out[d]);
		for (unsigned i = 0; i < g->n_obuf; i++)
			for (unsigned d = 0; d < g->ndev; d++)
			charsiu_bo_free(g->dev[d], &g->obuf[i].bo[d]);
		for (unsigned d = 0; d < g->ndev; d++) {
			charsiu_bo_free(g->dev[d], &g->in[d]);
			/* the batched path's, which a decode never allocates.
			 * Its output is in the geometry pool freed just above. */
			charsiu_bo_free(g->dev[d], &g->bin[d]);
			charsiu_bo_free(g->dev[d], &g->breg[d]);
			charsiu_close(g->dev[d]);
		}
	}
	free(g->slot);
	free(g->ent);
	free(g->scratch);
	free(g->acc);
	free(g->accf);
	free(g->fscr);
	free(g->afscr);
	free(g->wpack);
	free(g->bscr);
	free(g->bq);
	free(g->bd1);
	free(g->bmap);
	free(g->obuf);
	free(g->bseen);
	free(g->asum);
	free(g->tasks);
	free(g->handles);
	free(g);
}

unsigned long charsiu_npu_submits(const struct charsiu_npu *g)
{
	return g ? g->submits : 0;
}

/*
 * What the hardware actually did, printed whether it went well or not. A run
 * that cannot say how many jobs it submitted cannot be read as evidence about
 * the hardware, and round 315 was exactly that run.
 *
 * The megabytes per submit are here because that is the number the whole
 * chaining question turns on: break even against the fixed submit cost is 2.2.
 */
/*
 * Does the hardware path need the int8 activation? int4 takes the float one
 * and never looks at q1, so llama.c can skip realising it -- but only npudev
 * knows which mode it opened in, and duplicating the getenv in the caller is
 * how two switches drift apart.
 */
int charsiu_npu_needs_q1(const struct charsiu_npu *g)
{
	return !g || !g->w4;
}

/*
 * THE THREE TERM SOLVE, AND WHY IT REFUSES RATHER THAN GUESSES.
 *
 * Gaussian elimination with partial pivoting on a 3x3, which is about as much
 * numerical work as this deserves. What matters is the refusal: a run that
 * presented only one shape -- a single tensor benchmark, a model whose every
 * projection happens to be the same size, or a run that made two calls -- has a
 * singular or nearly singular system, and the fit it produces would be three
 * numbers with no information in them that a reader would nonetheless quote.
 *
 * ⚠ THE THRESHOLD IS RELATIVE. The entries span the call count, the megabytes
 * and their squares, so an absolute epsilon is meaningless: 1e-9 is small next
 * to a sum of squares over ten thousand calls and enormous next to one over
 * three. The pivot is compared against the largest entry the matrix started
 * with, which is scale free.
 */
static int solve3(double m[3][3], double *v, double *x)
{
	double big = 0.0;
	int i, j, r;

	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
			if (fabs(m[i][j]) > big)
				big = fabs(m[i][j]);
	if (big <= 0.0)
		return -1;
	for (i = 0; i < 3; i++) {
		int piv = i;

		for (r = i + 1; r < 3; r++)
			if (fabs(m[r][i]) > fabs(m[piv][i]))
				piv = r;
		if (fabs(m[piv][i]) < 1e-12 * big)
			return -1;
		if (piv != i) {
			for (j = 0; j < 3; j++) {
				double t = m[i][j];

				m[i][j] = m[piv][j];
				m[piv][j] = t;
			}
			{
				double t = v[i];

				v[i] = v[piv];
				v[piv] = t;
			}
		}
		for (r = 0; r < 3; r++) {
			double f;

			if (r == i)
				continue;
			f = m[r][i] / m[i][i];
			for (j = i; j < 3; j++)
				m[r][j] -= f * m[i][j];
			v[r] -= f * v[i];
		}
	}
	for (i = 0; i < 3; i++)
		x[i] = v[i] / m[i][i];
	return 0;
}

void charsiu_npu_report(const struct charsiu_npu *g)
{
	if (!g)
		return;
	/*
	 * ⚠ SAY WHICH WEIGHT WIDTH THIS WAS. A tokens-per-second number is not
	 * comparable without it -- int4 moves half the bytes of int8 and this
	 * report is where a board log gets read from months later. Round 389's
	 * 16.39 tok/s could not be placed against the README's 14.70 because
	 * neither line said.
	 */
	fprintf(stderr, "charsiu NPU: weights are %s, %u devices\n",
		g->w4 ? "int4" : "int8", g->ndev);
	fprintf(stderr,
		"charsiu NPU: %u tensors, %lu slices, %lu submits, %.2f MB per "
		"submit%s\n",
		g->n_ent, g->slices, g->submits,
		g->submits ? g->weight_mb / (double)g->submits : 0.0,
		g->dead ? "  (RETIRED: it stopped answering)" : "");
	if (g->submits)
		fprintf(stderr,
			"charsiu NPU: %.0f ms in the hardware path, %.2f GB/s "
			"of weights, %.0f us a submit -- a call issues one per "
			"core and waits on both, so that is a CALL's wall "
			"clock over %u\n"
			"charsiu NPU: of that, %.0f ms submitting, %.0f ms "
			"waiting for the fence (the invalidate is in there), "
			"%.0f ms summing the slices, %.0f ms in the flush\n"
			"charsiu NPU: and %.0f ms packing the activation, "
			"which is ON TOP of the hardware path above; "
			"%.0f ms was the CPU's own share of the projections, "
			"inside the fence window\n"
			"charsiu NPU: %.0f ms in these calls end to end, so "
			"%.0f ms of them is neither hardware nor packing\n",
			g->busy_us / 1e3, g->weight_mb / g->busy_us * 1e3,
			g->busy_us / (double)g->submits, g->ndev,
			g->submit_us / 1e3, g->fence_us / 1e3,
			g->copy_us / 1e3, g->fini_us / 1e3, g->pack_us / 1e3,
			g->cpu_us / 1e3, g->call_us / 1e3,
			(g->call_us - g->busy_us - g->pack_us) / 1e3);
	/*
	 * ⚠⚠ WHERE THE TIME GOES, SPLIT THREE WAYS INSTEAD OF DIVIDED BY A
	 * SUBMIT COUNT THAT DOUBLE COUNTS THE CORES.
	 *
	 * The line above prints megabytes and microseconds "a submit", which is
	 * a call's wall clock over the number of cores it used -- see the
	 * comment on g->calls. This is the same run in the units the clock
	 * actually measured, and it separates the part that scales with the
	 * bytes from the part that does not.
	 *
	 * ⚠ THE FIXED SHARE IS THE WHOLE POINT. If it is small then this
	 * hardware path is bandwidth bound and the only thing left is to move
	 * fewer bytes. If it is large -- and on TinyLLAMA decode the offline
	 * fit puts it at 40% of the hardware path, 11.9 ms per call plus 7.3 ms
	 * per task against 28.8 ms of weights -- then the tokens per second are
	 * being spent on dispatch, and the bytes per second figure above is an
	 * average across shapes rather than a roof anything is pressed against.
	 *
	 * The GB/s here is the aggregate across the cores and it does NOT
	 * assume they were given equal shares: it is the whole run's weight
	 * megabytes over the time the fit attributes to weights, so an uneven
	 * split shows up as a lower rate rather than as an invisible one.
	 */
	if (g->calls > 3) {
		double m[3][3], v[3], x[3];

		m[0][0] = g->f_n;  m[0][1] = g->f_t;  m[0][2] = g->f_m;
		m[1][0] = g->f_t;  m[1][1] = g->f_tt; m[1][2] = g->f_tm;
		m[2][0] = g->f_m;  m[2][1] = g->f_tm; m[2][2] = g->f_mm;
		v[0] = g->f_y; v[1] = g->f_ty; v[2] = g->f_my;
		fprintf(stderr,
			"charsiu NPU: %lu calls, %lu tasks and %.0f MB on the "
			"busier core, %.0f us a call\n",
			g->calls, g->tasks_hi, g->mb_hi,
			g->busy_us / (double)g->calls);
		/*
		 * ⚠ HOW LOPSIDED THE CALLS WERE, WHICH mb_hi ALONE CANNOT SAY.
		 *
		 * 1.00 is the two cores carrying the same bytes; 2.00 is one
		 * core doing all of it while the other waits on a fence for
		 * nothing. Before the least-loaded deal, Qwen3-0.6B ran its
		 * q/k/v call and its gate/up call at a flat 2.00 -- n_embd 1024
		 * is one K slice at KMAX 1024, so every one of those five
		 * tensors was a single slice and every single slice went to
		 * device 0. It is printed rather than derived because the ratio
		 * is the whole claim the deal makes, and a run that regressed
		 * should be able to say in one line whether the deal is why.
		 */
		if (g->ndev > 1 && g->mb_all > 0.0)
			fprintf(stderr,
				"charsiu NPU: the busier core carried %.2fx an "
				"even share of the weights (1.00 is balanced, "
				"2.00 is one core idle)%s\n",
				g->mb_hi / (g->mb_all / (double)g->ndev),
				g->deal_index
				? " -- CHARSIU_NPU_DEAL_INDEX is set, so this "
				  "is the old per tensor deal" : "");
		/*
		 * ⚠ SAY WHEN THE FIT DECLINES. Phase 21's second arm printed
		 * the stage table and no cost-model line, and the phase could
		 * only report "no line": the fit had been refused silently.
		 * The normal matrix goes singular when every call has the same
		 * task count or the same weight size -- a run too short or too
		 * uniform to separate the three terms -- and saying so is the
		 * difference between a mystery and a shorter prompt.
		 */
		/*
		 * ⚠ OUTSIDE THE COST MODEL'S BRANCH. This sat inside
		 * `if (!solve3(...))`, so the line that says whether the two
		 * cores ran together vanished on any run short or uniform
		 * enough that the three-term fit went singular -- which is
		 * every quick check somebody would make while asking exactly
		 * that question. The board printed it for a real prompt and
		 * printed nothing for `-p hi -n 4`.
		 */
		if (g->ndev > 1 && g->bwall_us > 0.0)
			fprintf(stderr, "charsiu NPU: batched calls, %s\n",
				charsiu_npu_overlap_note());
		if (g->bread_passes)
			fprintf(stderr, "charsiu NPU: the read walked Y %lu times"
				" over %lu (device, output range) pairs, so fusing"
				" the K slices a device holds would save %lu of"
				" them\n", g->bread_passes, g->bread_ranges,
				g->bread_passes - g->bread_ranges);
		if (g->bread_pooled + g->bread_serial)
			fprintf(stderr, "charsiu NPU: read back %lu slots on the pool"
				" and %lu one thread (%s)\n",
				g->bread_pooled, g->bread_serial,
				g->poolread == 1 ? "CHARSIU_NPU_POOL_READ=1, always" :
				g->poolread == 0 ? "CHARSIU_NPU_POOL_READ=0, never" :
				"pooled from CHARSIU_NPU_POOL_READ_MIN elements of output");
		if (solve3(m, v, x))
			fprintf(stderr, "charsiu NPU: the cost model did not fit "
				"(the calls do not separate 'a task' from 'a "
				"MB': %lu calls, tasks summed %.0f, MB summed "
				"%.1f)\n", g->calls, g->f_t, g->f_m);
		if (!solve3(m, v, x)) {
			double fix = x[0] * (double)g->calls / 1e3;
			double tsk = x[1] * (double)g->tasks_hi / 1e3;
			double byt = x[2] * g->mb_hi / 1e3;

			fprintf(stderr,
				"charsiu NPU: us a call = %.0f + %.1f a task "
				"+ %.1f a MB (both on the busier core), so of "
				"%.0f ms in the hardware path %.0f ms is per "
				"call, %.0f ms is per task and %.0f ms is the "
				"weights at %.2f GB/s across %u core%s\n",
				x[0], x[1], x[2], g->busy_us / 1e3, fix, tsk,
				byt, byt > 0.0 ? g->weight_mb / byt : 0.0,
				g->ndev, g->ndev == 1 ? "" : "s");
			/*
			 * ⚠ THE RESIDUAL, NOT THE SUM. fix + tsk + byt is the
			 * hardware path to the last decimal by construction --
			 * see the comment on f_yy -- so the number that says
			 * whether to believe the split is how far a typical
			 * call sits off the line. On TinyLLAMA's five decode
			 * shapes that is 9 us against a 540 us mean call,
			 * 1.7%; anything much larger means the calls are not
			 * three terms and the megabyte figure above should not
			 * be quoted.
			 */
			if (g->busy_us > 0.0) {
				double ss = g->f_yy - x[0] * g->f_y
					  - x[1] * g->f_ty - x[2] * g->f_my;
				double rms = ss > 0.0
					   ? sqrt(ss / (double)g->calls) : 0.0;

				fprintf(stderr,
					"charsiu NPU: %.0f%% of the hardware "
					"path is dispatch rather than bytes, "
					"and a typical call sits %.0f us off "
					"that line, %.1f%% of the %.0f us it "
					"takes\n",
					100.0 * (fix + tsk) / (g->busy_us / 1e3),
					rms,
					100.0 * rms * (double)g->calls / g->f_y,
					g->f_y / (double)g->calls);
			}
		} else {
			fprintf(stderr,
				"charsiu NPU: too few distinct shapes to split "
				"that into a fixed and a per byte part\n");
		}
	}
	if (g->slow_n)
		fprintf(stderr,
			"charsiu NPU: %lu of %lu submits came in under %.1f "
			"GB/s, worst %.2f at K=%u N=%u\n"
			"charsiu NPU: ⚠ the floor is a RATE, so the stall that "
			"trips it scales with the tensor -- and only the three "
			"UNGROUPED ones are watched at all, of which K=2048 "
			"N=2048 is the smallest and trips on a 1 ms hiccup\n",
			g->slow_n, g->submits, g->min_gbs, g->slow_worst,
			g->slow_worst_k, g->slow_worst_n);
	if (g->kfit)
		fprintf(stderr,
			"charsiu NPU: KFIT narrowed %u of %u staged tensors\n",
			g->kfit_hits, g->kfit_seen);
	if (g->kwide_only)
		fprintf(stderr,
			"charsiu NPU: KFIT narrowed 0 of 0 staged tensors "
			"(wide buffers only, no slicing)\n");
	if (!g->submits)
		fprintf(stderr,
			"charsiu NPU: NOTHING RAN ON THE HARDWARE. Every number "
			"in this run came from the CPU.\n");
}

/*
 * One range of a slice's output channels: gather the quantised bytes into the
 * shape the packer wants, then pack them. Both are indexed by channel and both
 * write disjoint bytes, so the ranges do not meet.
 */
struct wrows {
	struct charsiu_npu *g;
	const struct npu_tensor *t;
	const struct charsiu_matmul *mm;
	unsigned n0, k0, k;
};

static void pack_rows(void *vw, uint64_t r0, uint64_t nr)
{
	const struct wrows *w = vw;
	struct charsiu_npu *g = w->g;
	/*
	 * ⚠ TWO SEPARATE QUESTIONS, and they used to be one. pk is how the
	 * weight is STORED and g->w4 is what the device wants HANDED to it: an
	 * unsigned byte around a zero point of 128 for int8, or the signed code
	 * in the low nibble for int4, which is what two's complement already
	 * puts there for a value in [-8, 7]. Unpack first, then decide the
	 * byte, and the two are free to differ -- which they do whenever a
	 * caller forces int8 on the device while the quantiser is at four bits.
	 */
	const int pk = npu_q_packed();
	const size_t stride = npu_q_stride(w->t->k);

	for (uint64_t r = r0; r < r0 + nr; r++) {
		const int8_t *src = w->t->q + (size_t)(w->n0 + r) * stride;
		uint8_t *dst = g->scratch + (size_t)r * w->k;

		for (unsigned c = 0; c < w->k; c++) {
			uint64_t i = (uint64_t)w->k0 + c;
			int v = pk ? q_code(src, i) : src[i];

			dst[c] = g->w4 ? (uint8_t)((unsigned)v & 0xfu)
				       : (uint8_t)(v + 128);
		}
	}
	charsiu_pack_weights_rows(w->mm, g->scratch, g->wpack,
				  (unsigned)r0, (unsigned)nr);
}

/*
 * THE ZERO POINT CORRECTION FOR ONE SLICE: the sum of this slice's codes,
 * channel by channel, which charsiu_build_coefs folds into the coefficient
 * buffer so the hardware's unsigned operand comes back to the signed one.
 *
 * ⚠ ITS CALLER'S !g->w4 GATE IS NOT THE SAME QUESTION AS "q IS ONE BYTE A
 * CODE". This runs when the DEVICE is int8, and an int8 device does not make
 * the WEIGHTS int8: a tower that forced want_w4 = 0 -- which every batching
 * caller does, because w4a16 makes exactly one row -- under a config that sets
 * CHARSIU_NPU_W4V has int4 codes in a packed q and an int8 device reading them.
 * Summing the bytes there would add two codes at a time and put a wrong
 * correction into every coefficient, which the hardware applies without
 * complaint and which comes back as text.
 *
 * ⚠ IT IS ALSO WHY THIS IS ITS OWN FUNCTION. Nothing here can be reached from a
 * host without an NPU -- add_slice needs three buffer objects before it gets
 * this far -- so the only way to check a reader of q against the layout it
 * reads is to be able to call it.
 */
static void slice_wsum(const struct npu_tensor *t, unsigned n0, unsigned n,
		       unsigned k0, unsigned k, int32_t *wsum)
{
	const int pk = npu_q_packed();
	const size_t stride = npu_q_stride(t->k);

	for (unsigned r = 0; r < n; r++) {
		const int8_t *src = t->q + (size_t)(n0 + r) * stride;
		int32_t a = 0;

		for (unsigned c = 0; c < k; c++) {
			uint64_t i = (uint64_t)k0 + c;

			a += pk ? q_code(src, i) : src[i];
		}
		wsum[r] = a;
	}
}

/*
 * THE CPU'S ROWS, PACKED TWO WEIGHTS TO A BYTE: rows [n0, n) of t into cq,
 * ((k + 1) / 2) bytes each, low nibble first.
 *
 * ⚠ WHICH IS BYTE FOR BYTE WHAT t->q ALREADY HOLDS on the int4 path, so that
 * case is a memcpy of the row and the loop underneath is what is left for a q
 * still held one byte a code. That is not dead code and the reason is worth
 * keeping: an int8 DEVICE never reaches here at all, because the whole split is
 * gated on g->w4 -- but an int4 device whose quantiser was narrowed by
 * CHARSIU_NPU_W4_ONLY does, and its q is not packed.
 */
static void cq_fill(const struct npu_tensor *t, unsigned n0, uint8_t *cq)
{
	const int pk = npu_q_packed();
	const size_t stride = npu_q_stride(t->k);
	size_t per = ((size_t)t->k + 1) / 2;
	unsigned nc = (unsigned)t->n - n0;

	for (unsigned r = 0; r < nc; r++) {
		const int8_t *src = t->q + (size_t)(n0 + r) * stride;
		uint8_t *dst = cq + (size_t)r * per;
		uint64_t i;

		if (pk) {
			memcpy(dst, src, per);
			continue;
		}
		for (i = 0; i + 1 < t->k; i += 2)
			dst[i >> 1] = (uint8_t)((src[i] & 0xf) |
						((src[i + 1] & 0xf) << 4));
		if (i < t->k)
			dst[i >> 1] = (uint8_t)(src[i] & 0xf);
	}
}

/*
 * THE EXTENT OF ONE SLICE, IN ONE PLACE.
 *
 * charsiu_npu_add walks the (ki, ni) grid TWICE -- once to find out how many
 * slices each device is about to be given, so the output buffers can be sized
 * for what they will actually hold, and once to stage them. The two walks have
 * to agree exactly or a slice writes past the end of a buffer sized for fewer,
 * so the widths are computed here rather than written out twice.
 */
static unsigned slice_k0(const struct charsiu_npu *g, uint64_t k, unsigned ks,
			 unsigned ki)
{
	return charsiu_slice_k0(k, ks, ki, g->kmax, g->even_ks, g->kfit);
}

static unsigned slice_k(const struct charsiu_npu *g, uint64_t k, unsigned ks,
			unsigned ki)
{
	return charsiu_slice_kw(k, ks, ki, g->kmax, g->even_ks, g->kfit);
}

static unsigned slice_n(const struct charsiu_npu *g, unsigned n_npu, unsigned ni)
{
	unsigned n0 = ni * g->nmax;

	return (n_npu - n0) < g->nmax ? (n_npu - n0) : g->nmax;
}

/* the weight megabytes a slice costs the device it lands on, at its own width */
static double slice_mb(const struct charsiu_npu *g, unsigned k, unsigned n)
{
	return (double)k * (double)n / (g->w4 ? 2.0 : 1.0) / 1e6;
}

/*
 * WHICH CORE THIS SLICE GOES TO -- the one decision point, control included.
 *
 * The two terms are the board's fitted line for a call, minus its intercept:
 *
 *     us a call = 128.7 + 36.8 * tasks + 110.0 * MB      (busier core)
 *
 * Both devices pay the 128.7 whatever this returns, so charging it here would
 * only add a constant to both sides of every comparison. What is left says a
 * task is worth a third of a megabyte, which is why the deal cannot be by bytes
 * alone: a run of tiny slices piled on one core costs real time.
 *
 * ⚠ CHARSIU_NPU_DEAL_INDEX PUTS THE OLD DEAL BACK, and it has to be here
 * rather than at the call site because the sizing pass and the staging pass
 * both ask this question and a switch either of them missed would size a buffer
 * for one deal and fill it with another.
 *
 * ⚠ THE COUNTERS COME IN AS A PARAMETER AND THAT IS DELIBERATE, WHICH IS ALSO
 * WHY g IS const HERE WHILE THIS FUNCTION STILL CHANGES STATE. The staging pass
 * hands it g->deal_load and moves the run along; the sizing pass hands it a
 * copy on the stack and asks the same question without disturbing anything. One
 * function, two callers, and no way for them to answer differently.
 */
#define DEAL_US_TASK   36.8
#define DEAL_US_MB    110.0

static unsigned deal_pick(const struct charsiu_npu *g, double load[2],
			  unsigned ki, unsigned ni, unsigned ns,
			  unsigned k, unsigned n)
{
	unsigned d;

	if (g->ndev < 2)
		return 0;
	if (g->deal_index)
		return (ki * ns + ni) & 1;
	d = load[0] <= load[1] ? 0 : 1;
	load[d] += DEAL_US_TASK + DEAL_US_MB * slice_mb(g, k, n);
	return d;
}

/* One slice: rows [n0, n0+n) and columns [k0, k0+k) of t, writing region si. */
static int add_slice(struct charsiu_npu *g, unsigned di,
		     const struct npu_tensor *t,
		     unsigned n0, unsigned n, unsigned k0, unsigned k,
		     unsigned ki, unsigned si, uint32_t out_base)
{
	struct npu_slot *s = &g->slot[g->n_slot];

	charsiu_note("staging a slice", (unsigned long)n, (unsigned long)k);
	int32_t *bias = NULL, *wsum = NULL;
	int rc = -1;

	memset(s, 0, sizeof(*s));
	s->n0 = n0;
	s->k0 = k0;
	s->out_slot = si;
	s->di = di;
	/* the two cores share the CBUF, so the two devices take different
	 * windows -- see charsiu_job.cbuf_window */
	/*
	 * ⚠ CHARSIU_CBUF_SWAP=1 GIVES DEVICE 0 WINDOW 1 AND DEVICE 1 WINDOW 0.
	 * The overlap fault's wrong word (row 16, channel 3, both cores in
	 * flight, 2026-09-04) sat in DEVICE 1's K slice in 48 of 48 element
	 * reads and never in device 0's. Device 1 is two things at once: the
	 * second fd, whose job rocket puts on whichever core is free, and
	 * CBUF window 1 (data at 0x1c00, weights at 0x2c00, the vendor's
	 * values off one capture). Swapping the windows keeps the fd and moves
	 * the window: if the wrong word moves to device 0, it is the window's;
	 * if it stays on device 1, it is the core's or the ordering's.
	 */
	s->job.cbuf_window = getenv("CHARSIU_CBUF_SWAP") ? di ^ 1u : di;
	s->job.mm.m = 1;
	s->job.mm.k = k;
	s->job.mm.n = n;
	s->job.mm.wdtype = g->w4 ? CHARSIU_INT4 : CHARSIU_INT8;
	s->job.mm.adtype = g->w4 ? CHARSIU_FP16 : CHARSIU_INT8;
	s->job.input_zero_point = 128;
	s->job.weight_zero_point = 128;
	s->job.output_zero_point = 0;
	s->job.input_scale = 1.0f;
	s->job.weight_scale = 1.0f;
	s->job.output_scale = 1.0f;
	s->job.acc_out = 1;

	if (charsiu_bo_alloc(g->dev[di], charsiu_weight_bytes(&s->job.mm) + 4096, &s->wt) ||
	    charsiu_bo_alloc(g->dev[di], charsiu_coef_bytes(&s->job.mm) + 4096, &s->coef) ||
	    charsiu_bo_alloc(g->dev[di], 4096, &s->regcmd)) {
		/*
		 * ⚠ SAY WHICH BUFFER AND HOW BIG, because one of the three is
		 * enormous and it is not the weights.
		 *
		 * charsiu_coef_bytes bounds the coefficient surface by k*n,
		 * which makes it FOUR TIMES the weight buffer: 67 MB for an
		 * N=8192 slice. A 262144 wide output head is 32 such slices,
		 * so routing it asks for two gigabytes of coefficients on top
		 * of 150 MB of weights, and the allocation that fails is the
		 * reason a head worth forty percent of a gemma token stays on
		 * the CPU. The k*n bound is a guess nobody has measured -- see
		 * the comment on charsiu_coef_bytes -- and CHARSIU_COEF_ELEMS
		 * is how a board round finds the real one.
		 */
		fprintf(stderr, "charsiu: this slice wanted %.1f MB of weights "
			"and %.1f MB of coefficients\n",
			charsiu_weight_bytes(&s->job.mm) / 1e6,
			charsiu_coef_bytes(&s->job.mm) / 1e6);
		whine(g, "a buffer would not allocate", k, n);
		goto out;
	}

	/*
	 * Gather this slice's scales into the order the sum wants. Only a
	 * grouped tensor has a scale per (channel, group); an ungrouped one is
	 * scaled once at the end, where the stride does not arise.
	 */
	if (tensor_grouped(g, t)) {
		uint64_t ng = t->k / t->kgroup;
		uint64_t gi = k0 / t->kgroup;

		s->sc = malloc((size_t)n * sizeof(float));
		if (!s->sc) {
			whine(g, "the scale gather would not allocate", k, n);
			goto out;
		}
		for (unsigned j = 0; j < n; j++)
			s->sc[j] = t->scale[(uint64_t)(n0 + j) * ng + gi];
	}

	/* its own slot in each shared buffer, baked into the stream */
	s->job.input_addr = (uint32_t)g->in[di].dma_address + ki * g->in_stride;
	s->job.output_addr = out_base + si * g->out_stride;
	s->job.weight_addr = (uint32_t)s->wt.dma_address;
	s->job.coef_addr = (uint32_t)s->coef.dma_address;

	/*
	 * int8 wants unsigned bytes around a zero point of 128; int4 wants the
	 * signed code in the low nibble, which is what two's complement already
	 * puts there for a value in [-8, 7].
	 *
	 * ⚠ AND THIS IS THE OTHER HALF OF A COLD START. Round 369's board log
	 * reads 4.4 seconds "adding" against 7.9 "quantising", and where the
	 * quantiser now splits over the pool this did not: 620 MB of nibbles
	 * gathered and packed on one core. Both steps index by output channel
	 * and write disjoint bytes, so a range of channels is a whole unit of
	 * work -- checked byte for byte against the whole matrix call at nine
	 * shapes and five split counts.
	 */
	{
		struct wrows wr = { g, t, &s->job.mm, n0, k0, k };

		memset(g->wpack, 0, charsiu_weight_bytes(&s->job.mm));
		if (g->serialpack)
			pack_rows(&wr, 0, n);
		else
			charsiu_parallel_for(pack_rows, &wr, n);
	}
	/*
	 * ⚠ PACK INTO ORDINARY MEMORY AND THEN COPY, because the int4 layout
	 * writes STRIDED into the buffer -- sixteen consecutive bytes, then a
	 * jump of 256 -- and a buffer object's mapping does not absorb that the
	 * way a sequential write is absorbed. Round 352 spent 101 SECONDS
	 * staging 113 tensors this way against int8's 303 ms, at a steady
	 * second a tensor, and int8 only escapes it because its layout is
	 * nearly sequential. The copy afterwards is one sequential pass.
	 */
	charsiu_bo_prep(g->dev[di], &s->wt, 1000000000);
	memcpy(s->wt.map, g->wpack, charsiu_weight_bytes(&s->job.mm));
	charsiu_bo_fini(g->dev[di], &s->wt);

	bias = calloc(n, sizeof(*bias));
	wsum = calloc(n, sizeof(*wsum));
	if (!bias || !wsum)
		goto out;
	/* the weight sums this slice's K range accounts for, not the tensor's.
	 * int4 has no input zero point, so there is nothing for them to
	 * correct and they stay at zero. */
	if (!g->w4)
		slice_wsum(t, n0, n, k0, k, wsum);

	/*
	 * Zero bias and no lift, so the accumulator arrives unmodified. The
	 * lift clears a fused ReLU in the REQUANT domain and acc_out bypasses
	 * that domain, so adding it here would only corrupt the sum.
	 */
	setenv("CHARSIU_NO_LIFT", "1", 1);
	charsiu_bo_prep(g->dev[di], &s->coef, 1000000000);
	charsiu_build_coefs(&s->job, bias, wsum, s->coef.map);
	charsiu_bo_fini(g->dev[di], &s->coef);
	unsetenv("CHARSIU_NO_LIFT");

	charsiu_bo_prep(g->dev[di], &s->regcmd, 1000000000);
	s->nreg = (unsigned)charsiu_emit_job(&s->job, s->regcmd.map, 4096 / 8);
	charsiu_bo_fini(g->dev[di], &s->regcmd);
	if (!s->nreg) {
		whine(g, "the register stream came back empty", k, n);
		goto out;
	}

	g->n_slot++;
	rc = 0;
out:
	free(bias);
	free(wsum);
	return rc;
}

int charsiu_npu_add(struct charsiu_npu *g, const struct npu_tensor *t)
{
	double t_add = now_us();
	struct npu_entry *e;
	unsigned e_n_npu;

	/* ⚠ start the clock on the FIRST tensor, not the first heartbeat, or
	 * the first sixteen are free and round 353's log said "0 ms". */
	if (g->t_first == 0.0)
		g->t_first = t_add;
	/*
	 * ⚠ t->name IS A FIXED ARRAY INSIDE npu_tensor, not a stack buffer, so
	 * it is still readable from a signal handler after this frame is gone.
	 */
	charsiu_note(t->name, (unsigned long)t->n, (unsigned long)t->k);
	unsigned ns, ks, first = g->n_slot, si = 0;
	unsigned nslot[2];         /* what the deal gives each device */

	if (g->dead) {
		whine(g, "the hardware path is already retired", (unsigned)t->k,
		      (unsigned)t->n);
		return -1;
	}
	if (g->n_ent == g->ent_cap) {
		whine(g, "no tensor slots left", (unsigned)t->k, (unsigned)t->n);
		return -1;
	}
	if (t->n > g->max_n) {
		whine(g, "wider than the device was opened for", (unsigned)t->k,
		      (unsigned)t->n);
		return -1;
	}
	/*
	 * ⚠ THE TWO SIDES MUST AGREE ABOUT GROUPING, and when they did not the
	 * answer was wrong rather than absent. The quantiser rounded a partial
	 * last group up and wrote scales as scale[row * ngrp + group];
	 * tensor_grouped() below refuses a remainder, so the consumer read the
	 * same array as scale[row] and every row took some other row's scale.
	 * Qwen2.5-1.5B (k 1536 and 8960 against a 1024 slice) decoded fluent
	 * nonsense on the board while the same file was correct on the CPU.
	 *
	 * The quantiser no longer emits that state. This is here so that if it
	 * ever does again, the tensor falls back to the CPU and says why,
	 * instead of returning numbers nobody can tell are wrong.
	 */
	if (t->kgroup && t->kgroup < t->k && (t->k % t->kgroup)) {
		whine(g, "a partial weight group would be read as one scale a row",
		      (unsigned)t->k, (unsigned)t->n);
		return -1;
	}

	/*
	 * DECIDE THE SPLIT FIRST, because everything below is geometry over the
	 * hardware's share.
	 *
	 * Rounded DOWN to sixteen: sixteen output channels is the feature atom
	 * the int4 weight layout blocks by, and giving the hardware a ragged
	 * count to save the CPU a handful of rows is a bad trade. A tensor too
	 * narrow to split keeps all of it.
	 */
	e_n_npu = (unsigned)t->n;
	/*
	 * ⚠ ONLY WHERE THE SUM ALREADY LANDS IN THE CALLER'S BUFFER. The split
	 * writes the CPU's rows into y before the fence, and the read back then
	 * fills the rest; that only works on the path where the hardware's rows
	 * go straight into y and the conversion at the end is skipped, which is
	 * grouped int4 with the vector paths on.
	 */
	if (g->cpu_frac > 0.0 && g->w4 && !g->plain && tensor_grouped(g, t) &&
	    t->n >= 64) {
		unsigned keep = (unsigned)((double)t->n * (1.0 - g->cpu_frac));

		keep &= ~15u;
		if (keep < 16)
			keep = 16;
		if (keep < (unsigned)t->n)
			e_n_npu = keep;
	}

	ns = (unsigned)((e_n_npu + g->nmax - 1) / g->nmax);
	ks = (unsigned)((t->k + g->kmax - 1) / g->kmax);
	/* the last slice absorbs the remainder rather than being it */
	if (g->kfit) {
		g->kfit_seen++;
		if (ks > 1 && (t->k % g->kmax) && !tensor_grouped(g, t)) {
			ks--;
			g->kfit_hits++;
		}
	}
	if (ns * ks > g->max_slices || first + ns * ks > g->slot_cap) {
		whine(g, "no slice slots left", (unsigned)t->k, (unsigned)t->n);
		return -1;
	}

	/*
	 * ⚠⚠ A CHANGE OF K IS A CALL BOUNDARY, and it is the only one staging
	 * can see. charsiu_npu_matvec_group refuses a group whose entries do
	 * not share one K, so nothing that follows a K change can be in the
	 * same call as anything before it, and the running deal starts level.
	 * See the long note on g->deal_load for why carrying the residue across
	 * that line measures WORSE than not balancing at all.
	 */
	if (g->deal_k != t->k) {
		g->deal_load[0] = g->deal_load[1] = 0.0;
		g->deal_k = t->k;
	}

	/*
	 * ⚠⚠ DEAL FIRST, THEN SIZE THE BUFFERS FOR WHAT WAS DEALT.
	 *
	 * The old assignment alternated, so each device held ceil(count / 2)
	 * slices and the buffers could be sized from the count alone. A
	 * least-loaded deal does not: gemma4's q, k and v come out 2 slices on
	 * one core and 4 on the other, which is the point of it. So the walk
	 * below is the same walk that stages, run against a COPY of the
	 * counters, purely to learn how many output regions each device is
	 * about to be handed.
	 *
	 * The two walks share slice_k, slice_n and deal_pick, so they cannot
	 * disagree about geometry or about the deal. If they ever did, the
	 * bound check in the group read back -- `(s->out_slot + 1) *
	 * out_stride > e->out[s->di].size` -- is the net: it prints the slot
	 * and the buffer size and retires the path rather than writing past it.
	 */
	{
		double probe[2] = { g->deal_load[0], g->deal_load[1] };

		nslot[0] = nslot[1] = 0;
		for (unsigned ki = 0; ki < ks; ki++) {
			unsigned kw = slice_k(g, t->k, ks, ki);

			for (unsigned ni = 0; ni < ns; ni++)
				nslot[deal_pick(g, probe, ki, ni, ns, kw,
						slice_n(g, e_n_npu, ni))]++;
		}
	}

	e = &g->ent[g->n_ent];
	memset(e, 0, sizeof(*e));
	/*
	 * ⚠ ITS OWN OUTPUT BUFFER, AND THIS IS NOT TIDINESS.
	 *
	 * One shared buffer had to be sized for the WIDEST tensor, and round
	 * 318 added the 128256 wide output head, which took it from 128 KB to
	 * 2.48 MB. charsiu_bo_prep and _fini are cache maintenance over a WHOLE
	 * buffer object, and every one of the 113 matvecs a token paid it: 280
	 * MB of cache operations a token where there had been 14. That is most
	 * of why routing the head made the model 18% SLOWER.
	 */
	/*
	 * ⚠ SIZED FOR THE SLICES THIS DEVICE ACTUALLY GETS, not for all of them.
	 *
	 * Allocating both buffers at the full size doubled the cache
	 * maintenance a matvec pays: charsiu_bo_prep and _fini work over a WHOLE
	 * buffer object, and round 366 measured 13.4 ms a token in the readback
	 * against 11.6 on the one device build. The output head alone is 512 KB
	 * a buffer, so this is half a megabyte of cache operations a token
	 * bought back for nothing.
	 *
	 * ⚠ AND IT IS THE DEAL'S OWN COUNT NOW, not ceil(ns * ks / 2). That
	 * expression was only ever true because the slices alternated; a
	 * least-loaded deal can give one device more than half of a tensor, and
	 * a buffer sized on the old assumption would be written past. It is
	 * also SMALLER wherever the deal is uneven, which is the same cache
	 * maintenance argument running the other way.
	 */
	for (unsigned d = 0; d < g->ndev; d++) {
		size_t slots = (size_t)nslot[d] + 1;

		if (charsiu_bo_alloc(g->dev[d],
				     slots * g->out_stride + 4096,
				     &e->out[d])) {
			whine(g, "an output buffer would not allocate",
			      (unsigned)t->k, (unsigned)t->n);
			return -1;
		}
	}

	/*
	 * ⚠ THE SLICES OF ONE TENSOR SPLIT TOO, not just the members of a
	 * group. Round 364 put the second core in and got only 7%, because the
	 * o_proj, the down_proj and the 128256 wide output head all go through
	 * the single projection path -- more than 40% of the weight traffic in
	 * tensors that were never grouped with anything. Splitting the SLICES
	 * gives those two cores as well.
	 *
	 * ⚠ AND ACROSS TENSORS AS WELL AS WITHIN ONE, which is what g->deal_load
	 * carries and what an index that restarted per tensor could not do.
	 */
	{
	unsigned sid[2] = { 0, 0 };

	for (unsigned ki = 0; ki < ks; ki++) {
		unsigned k0 = slice_k0(g, t->k, ks, ki);
		unsigned k = slice_k(g, t->k, ks, ki);

		for (unsigned ni = 0; ni < ns; ni++, si++) {
			unsigned n0 = ni * g->nmax;
			unsigned n = slice_n(g, e_n_npu, ni);
			unsigned d = deal_pick(g, g->deal_load, ki, ni, ns,
					       k, n);

			if (add_slice(g, d, t, n0, n, k0, k, ki, sid[d]++,
				      (uint32_t)e->out[d].dma_address) < 0) {
				g->n_slot = first;
				return -1;
			}
			g->slices++;
		}
	}
	}

	e->t = t;
	e->first = first;
	e->count = ns * ks;
	e->n_slices = ns;
	e->k_slices = ks;
	e->n_npu = e_n_npu;
	/*
	 * WHAT EACH CORE ENDED UP WITH, counted where the assignment is made.
	 *
	 * ⚠ THIS IS WHAT THE DEAL IS SCORED ON, so it is worth saying what it
	 * can and cannot fix. TinyLLAMA's down_proj cuts k = 5632 into five
	 * 1024 slices and one of 512, and no deal divides that evenly: the best
	 * two-colouring is 3.146 MB against 2.621, which is what the index deal
	 * already gave. What the least-loaded deal fixes is the case the index
	 * one could not see at all -- a call carrying several tensors, and a
	 * tensor whose slices are of different widths.
	 */
	for (unsigned i = 0; i < e->count; i++) {
		const struct npu_slot *sl = &g->slot[e->first + i];
		unsigned d = sl->di < 2 ? sl->di : 0;

		e->nt_dev[d]++;
		e->mb_dev[d] += (double)sl->job.mm.k * (double)sl->job.mm.n
			      / (g->w4 ? 2.0 : 1.0) / 1e6;
	}
	/*
	 * THE CPU'S ROWS, PACKED TWO WEIGHTS TO A BYTE.
	 *
	 * Reading the CPU's share at one byte a code would cost twice the bytes
	 * the hardware pays for the same weights, and bandwidth is the entire
	 * point of the split, so those rows get their own packed copy: low
	 * nibble first, row major, nothing scrambled. It is f * n * k / 2 bytes,
	 * 136 MB of this model at a quarter of the rows.
	 *
	 * ⚠ ON THE int4 PATH IT IS A COPY NOW, because t->q is held in exactly
	 * this layout -- see cq_fill, which is where the packing went so that a
	 * host with no NPU can still drive it.
	 */
	if (e_n_npu < (unsigned)t->n) {
		unsigned nc = (unsigned)t->n - e_n_npu;
		size_t per = ((size_t)t->k + 1) / 2;

		e->cq = malloc((size_t)nc * per);
		if (!e->cq) {
			whine(g, "the CPU's share would not allocate",
			      (unsigned)t->k, (unsigned)t->n);
			e->n_npu = (unsigned)t->n;   /* fall back to all NPU */
		} else {
			cq_fill(t, e_n_npu, e->cq);
		}
	}
	/*
	 * ⚠ BYTES, NOT ELEMENTS. This counted n*k for both precisions, so every
	 * "GB/s of weights" this project has printed for int4 was DOUBLE the
	 * real figure -- 13.4 GB/s in round 356's log is 6.7. int8 was right by
	 * accident, one byte an element. The honest comparison is int4 at 6.7
	 * GB/s against int8's 9.46, which is what it looked like from the
	 * outside and what the shape sweep said.
	 */
	e->weight_mb = (double)e_n_npu * (double)t->k
		     / (g->w4 ? 2.0 : 1.0) / 1e6;
	/*
	 * A HEARTBEAT WHILE THE WEIGHTS ARE STAGED. Round 352's int4 arm printed
	 * nothing for minutes and there was no way to tell a slow load from a
	 * wedge: charsiu_run's own output does not appear until the generation
	 * is done. Four lines for a 113 tensor model is not noise.
	 */
	/*
	 * ⚠ THE HEARTBEAT SPLITS THE TIME NOW. Round 353 showed int4 staging at
	 * 102 s against int8's 16 s -- SIX times, not the three hundred I first
	 * read, because int8's own staging is 16 s and its "load 345 ms" line is
	 * only the gguf mmap. Packing into ordinary memory and copying did NOT
	 * move it, so the strided write to the buffer object was not the cause
	 * and I have no second guess. This measures instead: g->add_us is time
	 * inside charsiu_npu_add, and whatever is left of the wall clock between
	 * heartbeats belongs to npu_tensor_build, which is the quantiser.
	 */
	g->add_us += now_us() - t_add;
	if ((g->n_ent % 16) == 15) {
		fprintf(stderr,
			"charsiu NPU: %u tensors staged, %.0f ms of which "
			"%.0f ms adding and %.0f ms quantising\n",
			g->n_ent + 1, (now_us() - g->t_first) / 1000.0,
			g->add_us / 1000.0,
			((now_us() - g->t_first) - g->add_us) / 1000.0);
	}
	return (int)g->n_ent++;
}

/*
 * ONE CALL'S GEOMETRY AND ONE CALL'S WALL CLOCK, INTO THE FIT.
 *
 * The three terms the report solves for -- a cost per call, a cost per chained
 * task and a cost per megabyte -- are not separable from any single run of the
 * decode loop, because a real model only ever presents five or six shapes and
 * each of them varies all three at once. They ARE separable across those five,
 * which is what least squares is for, so every call drops its (tasks, MB, us)
 * into the normal equations and the report solves them at the end.
 *
 * ⚠ THE MAX, NOT THE SUM. The devices are submitted before either is waited
 * on, so a call ends when the SLOWER core finishes; charging it the total would
 * fit a line to a quantity the clock never measured.
 *
 * ⚠ AND THE MEGABYTES ARE THE HARDWARE'S OWN. mb_dev is summed from the slices'
 * mm.k * mm.n at the device's own weight width, so the CPU's rows under
 * CHARSIU_NPU_CPU_FRAC are already out of it and int4 is already halved -- the
 * mistake that made every int4 GB/s in this project double the real figure for
 * three rounds.
 */
static void account_call(struct charsiu_npu *g, const int *ids, unsigned n,
			 double us)
{
	double mb[2] = { 0.0, 0.0 }, hm;
	double nt[2] = { 0.0, 0.0 }, ht;
	unsigned i;

	for (i = 0; i < n; i++) {
		const struct npu_entry *e = &g->ent[ids[i]];

		mb[0] += e->mb_dev[0];
		mb[1] += e->mb_dev[1];
		nt[0] += e->nt_dev[0];
		nt[1] += e->nt_dev[1];
	}
	hm = mb[0] > mb[1] ? mb[0] : mb[1];
	ht = nt[0] > nt[1] ? nt[0] : nt[1];

	g->calls++;
	g->tasks_hi += (unsigned long)ht;
	g->mb_hi += hm;
	g->mb_all += mb[0] + mb[1];

	g->f_n  += 1.0;
	g->f_t  += ht;
	g->f_m  += hm;
	g->f_tt += ht * ht;
	g->f_tm += ht * hm;
	g->f_mm += hm * hm;
	g->f_y  += us;
	g->f_ty += ht * us;
	g->f_my += hm * us;
	g->f_yy += us * us;
}

int charsiu_npu_matvec(struct charsiu_npu *g, int id,
		       const struct charsiu_act *a, float *y)
{
	struct npu_entry *e;
	struct charsiu_joblist jl;
	const int32_t *out;
	unsigned nh = 0, i;
	double t0, tpack, tcall = now_us();
	float *af;

	if (g->dead || id < 0 || (unsigned)id >= g->n_ent)
		return -1;
	charsiu_note("a matvec on one tensor", (unsigned long)id,
		     (unsigned long)a->n);
	e = &g->ent[id];
	if ((unsigned)a->n != e->t->k) {
		whine(g, "the activation is not this tensor's K", (unsigned)a->n,
		      (unsigned)e->t->n);
		return -1;
	}

	/* every K slice's activation, each in its own region */
	/*
	 * ⚠ THE ACTIVATION GOES INTO EVERY DEVICE, and round 365 shipped without
	 * it. The slices of one tensor are spread across both devices now, so
	 * the ones on the other device read an input buffer nobody wrote --
	 * and the decode came back as word salad on BOTH int8 and int4 while
	 * the one device control was perfect. The grouped path had already been
	 * fixed for exactly this and the single path was not.
	 *
	 * A buffer object belongs to the file that created it, so there is no
	 * sharing one: it has to be packed twice. That is a few kilobytes
	 * against the megabytes of weights a submit fetches.
	 */
	tpack = now_us();
	for (unsigned d = 0; d < g->ndev; d++) {
	if (g->inprep)
		charsiu_bo_prep(g->dev[d], &g->in[d], 1000000000);
	for (unsigned ki = 0; ki < e->k_slices; ki++) {
		const struct npu_slot *s = &g->slot[e->first + ki * e->n_slices];

		if (g->w4) {
			const float *src = a->f + s->k0;

			/*
			 * STRAIGHT FROM THE ACTIVATION, NO COPY. The packer
			 * reads src[kk] at m = 1, so the scratch buffer was
			 * 463 thousand float copies a token to hand it bytes
			 * it could already see. The running total is the
			 * midrise grid's half step, and midrise is off by
			 * default and measured worse when it was on, so it no
			 * longer costs a double add per element either.
			 */
			if (g->midrise || g->plain) {
				double as = 0.0;

				for (i = 0; i < s->job.mm.k; i++) {
					g->fscr[i] = src[i];
					as += (double)g->fscr[i];
				}
				g->asum[ki] = as;
				src = g->fscr;
			}
			charsiu_pack_input_f16(&s->job.mm, src,
					       (uint8_t *)g->in[d].map
					       + ki * g->in_stride,
					       g->in_stride);
		} else {
			for (i = 0; i < s->job.mm.k; i++)
				g->scratch[i] = (uint8_t)((int)a->q1[s->k0 + i]
							  + 128);
			charsiu_pack_input(&s->job.mm, g->scratch,
					   (uint8_t *)g->in[d].map
					   + ki * g->in_stride,
					   g->in_stride,
					   s->job.input_zero_point);
		}
	}
	charsiu_bo_fini(g->dev[d], &g->in[d]);
	}
	g->pack_us += now_us() - tpack;
	/*
	 * ⚠⚠ WHICH DEVICES WERE ACTUALLY GIVEN WORK, because the sync below
	 * used to ask both regardless.
	 *
	 * The loop below skips a device with no slices of this entry -- `if
	 * (!nt) continue` -- but the prep and the fini that follow it ran over
	 * `g->ndev` unconditionally. So a core that was provably never written
	 * had its whole output buffer invalidated on the way in and cleaned on
	 * the way out, once per entry, once per token, for nothing.
	 *
	 * It is not a small nothing. rknn_core_0 and rknn_core_1 carry no
	 * dma-coherent in rk3576.dtsi, so these are real cache maintenance over
	 * the whole buffer object: rocket_ioctl_prep_bo takes a handle and
	 * nothing else -- no offset, no length, and its one spare word is
	 * checked to be zero -- and it does dma_sync_sgtable_for_cpu over the
	 * entire sgtable. Counted on the real gguf shapes that is 10.3 MB a
	 * token on Qwen3-0.6B alone, and Qwen3 is the model whose single-slice
	 * projections leave a device idle most often.
	 *
	 * ⚠ AND IT GOT WORSE WITH THE LEAST LOADED DEAL, not better. The old
	 * index deal spread a tensor's slices across both cores by construction,
	 * so `nt` was rarely zero; a deal that puts a small tensor entirely on
	 * one core makes the other core's empty buffer the common case.
	 */
	unsigned sent = 0;

	/*
	 * ONE submit for the whole projection, unless a cap says otherwise.
	 *
	 * CHARSIU_NPU_NOCHAIN puts it back to a submit per slice, so a round can
	 * carry its own before and after in one boot. CHARSIU_NPU_MAXTASK caps
	 * the tasks per submit, which exists to test a specific suspicion: the
	 * output head is 126 chained tasks and 253 buffer handles in one submit
	 * and reached 4.2 GB/s, where an eight task submit reaches 10, so the
	 * driver's per handle work is a candidate for the difference.
	 */
	/*
	 * ONE JOBLIST PER DEVICE, BOTH ISSUED BEFORE EITHER IS WAITED ON. The
	 * slices of this tensor alternate between the devices, so a projection
	 * that is not part of a group still uses both cores.
	 */
	t0 = now_us();
	for (unsigned d = 0; d < g->ndev; d++) {
		unsigned nt = 0;

		nh = 0;
		g->handles[nh++] = g->in[d].handle;
		for (i = 0; i < e->count; i++) {
			const struct npu_slot *s = &g->slot[e->first + i];

			if (s->di != d)
				continue;
			g->tasks[nt].regcmd = (uint32_t)s->regcmd.dma_address;
			g->tasks[nt].regcmd_count = s->nreg;
			nt++;
			g->handles[nh++] = s->wt.handle;
			g->handles[nh++] = s->coef.handle;
		}
		if (!nt)
			continue;
		sent |= 1u << d;
		jl.tasks = g->tasks;
		jl.task_count = nt;
		jl.in_handles = g->handles;
		jl.in_count = nh;
		jl.out_handles = &e->out[d].handle;
		jl.out_count = 1;
		if (charsiu_submit_jobs(g->dev[d], &jl, 1)) {
			g->strikes = 3;
			break;
		}
		g->submits++;
	}
	g->submit_us += now_us() - t0;

	/*
	 * THE CPU'S ROWS, WHILE THE HARDWARE HAS THE REST. The submit above is
	 * asynchronous, so from here until prep_bo the calling thread has
	 * nothing to do but block. Nothing is scheduled and nothing is waited
	 * on: the work simply fills a window that was already being spent.
	 */
	if (e->cq && e->n_npu < (unsigned)e->t->n) {
		double tc = now_us();

		for (uint64_t i = 0; i < e->t->k; i++)
			g->afscr[i] = charsiu_half_to_float(
					charsiu_float_to_half(a->f[i]));
		cpu_rows(e, g->afscr, y);
		g->cpu_us += now_us() - tc;
	}

	if (g->strikes < 3) {
		double t1 = now_us();

		for (unsigned d = 0; d < g->ndev; d++)
			if (sent & (1u << d))
				charsiu_bo_prep(g->dev[d], &e->out[d],
						2000000000);
		g->fence_us += now_us() - t1;
		t1 = now_us();
		int grp = tensor_grouped(g, e->t);

		/*
		 * ONE OF THESE, NOT BOTH. int4 sums into accf and int8 into
		 * acc, and clearing the other one is 2 MB a token of writes
		 * for an array nothing will read: the 128256 wide head alone
		 * is half a megabyte of it.
		 *
		 * AND WHEN THE LAST STEP WOULD BE A COPY, SUM WHERE THE ANSWER
		 * GOES. A grouped int4 tensor applies its scale per slice on
		 * the way in, so the conversion at the end of this function is
		 * y[i] = accf[i] and nothing else: a staging buffer read and
		 * written for no reason.
		 */
		af = (g->w4 && grp && !g->plain) ? y : g->accf;
		/* ⚠ the hardware's rows only: the CPU's are already written */
		if (g->w4)
			memset(af, 0, (size_t)e->n_npu * sizeof(*af));
		else
			memset(g->acc, 0, (size_t)e->t->n * sizeof(*g->acc));
		for (i = 0; i < e->count; i++) {
			const struct npu_slot *s = &g->slot[e->first + i];
			const uint8_t *base = (const uint8_t *)e->out[s->di].map +
					      s->out_slot * g->out_stride;

			/* int4 writes float32, int8 the raw int32 accumulator */
			if (g->w4) {
				const float *fo = (const float *)base;

				if (grp) {
					double hs = g->midrise
						  ? 0.5 * g->asum[s->k0 / g->kmax]
						  : 0.0;
					const float *sc = s->sc;

					/* non null whenever grp is: the two
					 * ask tensor_grouped the same
					 * question, and a gather that will not
					 * allocate fails the staging */
					if (hs == 0.0 && !g->plain) {
						scaled_add(af + s->n0, fo, sc,
							   s->job.mm.n);
						continue;
					}
					for (unsigned j = 0; j < s->job.mm.n; j++)
						af[s->n0 + j] +=
						  (float)((fo[j] + hs) * sc[j]);
				} else {
					for (unsigned j = 0; j < s->job.mm.n; j++)
						af[s->n0 + j] += fo[j];
				}
				continue;
			}
			out = (const int32_t *)base;
			for (unsigned j = 0; j < s->job.mm.n; j++)
				g->acc[s->n0 + j] += out[j];
		}
		g->copy_us += now_us() - t1;
		t1 = now_us();
		/* ⚠ the same mask: a buffer nobody prepped must not be finied,
		 * or the CPU hands back ownership of something it never took. */
		if (!g->nofini)
			for (unsigned d = 0; d < g->ndev; d++)
				if (sent & (1u << d))
					charsiu_bo_fini(g->dev[d],
							&e->out[d]);
		g->fini_us += now_us() - t1;
		g->weight_mb += e->weight_mb;

		/*
		 * The limit scales with what the submit fetches, or a legitimate
		 * big one is mistaken for a wedge. A millisecond a megabyte is
		 * ten times slower than this hardware has ever been.
		 */
		{
			double took = now_us() - t0;
			double gbs = e->weight_mb / took * 1e3;

			/*
			 * ⚠ ONE CLOCK READ FEEDS BOTH. busy_us used to take its
			 * own a few hundred nanoseconds before this block took
			 * this one, which is nothing against a 400 us call and
			 * is still two different numbers for one quantity. The
			 * fit and the total have to be the SAME measurement or
			 * a residual between them means nothing.
			 */
			g->busy_us += took;
			account_call(g, &id, 1, took);

			if (took > g->slow_us * (g->nochain ? e->count : 1)
				  + e->weight_mb * 1000.0)
				g->strikes++;
			else
				g->strikes = 0;

			/*
			 * ⚠ NOT ON A WARM UP. Round 323's first call to a
			 * tensor came in at 1.25 GB/s while the run averaged
			 * 9.35, so the notice fired on a cold buffer and said
			 * nothing true about the run. A warning that cries wolf
			 * on every boot is worse than none.
			 */
			if (gbs < g->min_gbs && g->submits > g->n_ent * 2) {
				g->slow_n++;
				if (g->slow_worst == 0.0 || gbs < g->slow_worst) {
					g->slow_worst = gbs;
					g->slow_worst_k = (unsigned)e->t->k;
					g->slow_worst_n = (unsigned)e->t->n;
				}
			}
			if (!g->slowed && gbs < g->min_gbs &&
			    g->submits > g->n_ent * 2) {
				g->slowed = 1;
				fprintf(stderr,
					"charsiu: the NPU is SLOW, %.2f GB/s at "
					"K=%u N=%u -- correct but degraded, and "
					"nothing is being retired\n",
					gbs, (unsigned)e->t->k,
					(unsigned)e->t->n);
			}
		}
	}

	if (g->strikes >= 3) {
		g->dead = 1;
		fprintf(stderr,
			"charsiu: the NPU stopped answering on a %u task submit "
			"(K=%u N=%u); everything from here runs on the CPU\n",
			e->count, (unsigned)e->t->k, (unsigned)e->t->n);
		return -1;
	}

	/*
	 * int4 took the REAL activation, so there is no d1 to undo: the block
	 * returns sum_k code(n,k) * a(k) in float and only the weight scale is
	 * left. int8 took a->q1 and needs both.
	 */
	{
		int grp = tensor_grouped(g, e->t);
		double hs = 0.0;

		if (g->w4 && grp && !g->plain) {
			g->call_us += now_us() - tcall;
			charsiu_note("something outside the NPU code", 0, 0);
	return 0;              /* it was summed into y */
		}
		if (g->midrise && !grp)
			for (unsigned ki = 0; ki < e->k_slices; ki++)
				hs += 0.5 * g->asum[ki];
		for (i = 0; i < (unsigned)e->t->n; i++)
			y[i] = g->w4
			     ? (grp ? g->accf[i]
				    : (float)(((double)g->accf[i] + hs)
					      * e->t->scale[i]))
			     : (float)g->acc[i] * a->d1 * e->t->scale[i];
	}
	g->call_us += now_us() - tcall;
	charsiu_note("something outside the NPU code", 0, 0);
	return 0;
}

/*
 * M ROWS THROUGH ONE SET OF WEIGHTS, which is the whole of prefill.
 *
 * The vendor dispatches every one of its 3328 int4 projections at M = 1, so it
 * re-streams all 487 MB of weights for every prompt token. At M = 32 the same
 * bytes serve thirty two rows.
 *
 * ⚠ THIS DOES NOT TOUCH THE DECODE PATH. charsiu_npu_matvec is unchanged, and
 * the control for every board round here is that the sentence and the tok/s do
 * not move. A slice's weights and coefficients do not depend on m and are read
 * exactly as staged; what does depend on m is the register stream, the packed
 * activation and the output, so this brings its own buffers.
 *
 * ⚠ ONE SUBMIT PER SLICE, deliberately. Decode chains a projection's slices
 * into one submit and that is worth having, but the first version of a path
 * that has never run has no business also being the first version of a chained
 * one. The weight bytes dominate either way.
 *
 * ⚠ int4 ONLY for now. That is what CHARSIU_NPU_W4V selects and what the
 * runtime uses; int8 batched would need its own d1 handling and has no caller.
 */
/*
 * ⚠⚠ SIZED FOR A CHAIN, NOT FOR ONE SLICE.
 *
 * The first version submitted a slice and waited a full fence on it, which the
 * report priced: 5072 ms of 7448 was fence. Decode has chained a projection's
 * slices into one submit per device since round 321 for exactly this reason,
 * and the fence is what a chain removes.
 *
 * So every slice needs its own region rather than sharing one: its own packed
 * activation (by K slice, since slices of the same K share it), its own output,
 * and its own register stream. The three grow with m and with how many slices
 * land on a device, so they are reallocated when either does and never by a
 * decode, which uses none of them.
 */
/*
 * ⚠⚠ AN ODD BATCH WIDTH HAS NO EXPRESSION ON THIS SURFACE, and that is a
 * property of the layout rather than a pattern in the measurements.
 *
 * charsiu_acc_index -- the read order, in src/job.c -- was linked into a
 * standalone exhaustive checker and swept over m = 2..96 crossed with n = 512,
 * 2048 and 8192, asking four things of every (m, n): that every index lands in
 * range, that no two slots collide, that no slot is left unwritten, and that
 * the four consecutive slots the gather relies on stay consecutive. With no
 * exceptions at all:
 *
 *   m EVEN   a clean bijection, four in a row intact
 *   m ODD    a COLLISION, at every n
 *
 * And it cannot be repaired by fitting a constant. In the roleswap2 branch the
 * map covers 64 * P slots per group where the group needs 32 * m, so it fits
 * only when 64P == 32m, i.e. only when P == m/2 -- and no integer P exists for
 * an odd m. The surface is organised in PAIRS OF ROWS. An odd width is not a
 * width this arrangement can name.
 *
 * ⚠⚠ WHICH SEPARATES TWO FAULTS THAT WERE BEING READ AS ONE, and conflating
 * them is what made this take four rounds:
 *
 *   m = 31, odd          0 of 6975 rows on phi3, 0 of 8587 on gemma4. THE READ
 *                        ORDER, and now proven offline rather than inferred
 *                        from wrong text.
 *   m = 2, 4, 16, 32,    exact on the board, worst relative 1.6e-04 over 225
 *   48, 64, 80           to 277 real tensors. Even, as the proof requires.
 *   m = 8, even          871 rows of 904 with two cores and 904 of 904 under
 *                        CHARSIU_NPU_ONEDEV. NOT the read order -- the read
 *                        order is a bijection there. The core pair, which is
 *                        its own fault with its own refusal below.
 *
 * This also explains four models at once. Llama-3.2-1B's 65 token prompt used
 * to chunk to 32, 32 and a tail of 1, which is below the batching minimum and
 * went to the token loop, and its text has always been right. Phi-3.5's 87
 * tokens chunked to 32, 32, 23 and Gemma-4-E2B's 88 to 32, 32, 24 -- and only
 * the odd one produces wrong text. The fault was never the model, it was the
 * last chunk.
 *
 * ⚠⚠ AND THE REFUSAL IS THE SAFETY NET, NOT THE OPTIMISATION. A width this
 * says no to falls back to a row at a time, which is what int4 did before any
 * of the batched path existed and is correct. So the worst case of this
 * predicate being too narrow is SLOW, never wrong. What turns the law into
 * speed is the chunker in tools/charsiu_run.c, which only ever asks for widths
 * this accepts; if the two ever fall out of step the result is a refused chunk
 * run a row at a time -- a slower prefill and the same text.
 */
static int w4_width_expressible(unsigned m)
{
	return (m % 2) == 0;
}

/*
 * Both switches or neither. An int4 batch on the height axis is the wrong
 * answer at a very good speed, which is the one failure mode this tree has
 * already shipped once.
 *
 * ⚠ AND "height" IS A THIRD VALUE, because the first board round's control was
 * VACUOUS. tests/board_w4_axis.sh ran the height axis as the arm that must
 * fail, and it did -- by hitting this refusal, which is a decision in software
 * and says nothing about the hardware. A control that cannot reach the thing
 * it is controlling for is not a control.
 *
 * CHARSIU_NPU_W4_BATCH=height is "yes, on the axis that is known wrong, I am
 * running the arm that must fail". Nothing else should ever set it.
 */
static const char *w4_batch_why_not(unsigned m)
{
	const char *b = getenv("CHARSIU_NPU_W4_BATCH");

	/* "height" is the board control deliberately reaching the arrangement
	 * that is known wrong; nothing else should ever set it */
	if (b && !strcmp(b, "height"))
		return NULL;
	if (b && *b == '0')
		return "int4 batching is switched off";
	if (!charsiu_m_axis_wide_for(1))
		return "int4 batches on the width axis and this asked for height";
	/*
	 * ⚠⚠ THE PROBE HAS TO BE ABLE TO ASK ABOUT THE WIDTHS THAT ARE
	 * REFUSED, because asking is how every line of the table above was
	 * measured and is the only way it will be re-measured. 23 and 31 are
	 * widths the runtime will never choose again, and they are precisely
	 * the widths the next round has to hand the hardware.
	 *
	 * CHARSIU_NPU_W4_ANYM=1 lifts the width rule entirely, m = 8 included,
	 * and nothing but a probe should ever set it. It does NOT lift the axis
	 * check above: the height axis is the arrangement five rounds proved
	 * writes one row, and reaching that deliberately has its own switch,
	 * for the same reason this one has its own name -- a round that sets a
	 * switch for one reason must not quietly get a second meaning with it.
	 */
	if (getenv("CHARSIU_NPU_W4_ANYM"))
		return NULL;
	/*
	 * ⚠⚠ TWO REFUSALS, TWO REASONS, AND THEY ARE NOT THE SAME FAULT. The
	 * odd widths are the accumulator read order and are proven wrong
	 * offline; m = 8 is the core pair and the read order is a clean
	 * bijection there. Giving them one shared string is exactly the
	 * conflation that cost four rounds -- a round reading "the width is
	 * refused" cannot tell which of the two it just hit.
	 */
	if (!w4_width_expressible(m))
		return "an odd batch width, which the accumulator read order "
		       "cannot express: the surface is organised in pairs of "
		       "rows, and 64*P slots per group can only equal the 32*m "
		       "it needs when m is even";
	/*
	 * ⚠⚠ m = 8 IS THE ONE WIDTH THAT IS STILL WRONG, and the board named
	 * it rather than leaving it as a count.
	 *
	 * Every other width the probe reaches is exact -- 2, 4, 16, 32, 48, 64
	 * and 80, worst relative 5.10e-05 across 113 tensors. m = 8 returns
	 * 871 rows of 904, and the 33 that miss are not scattered: they are
	 * ROW 0 of the n = 8192 tensors, every ffn_gate and ffn_up in the
	 * model, at that width and no other.
	 *
	 * One shape and one row is a small enough target to find. Until it is
	 * found this refuses the width, and the caller falls back to a row at
	 * a time for that chunk, which is correct and merely slower.
	 *
	 * ⚠⚠ IT NEEDS BOTH NUMBERS, AND THAT IS THE WHOLE SHAPE OF IT. m = 8
	 * is exact at n = 512 and n = 2048; n = 8192 is exact at m = 2, 4, 16,
	 * 32, 48, 64 and 80. Neither number is wrong on its own, so nothing
	 * that is a function of only one of them can be the cause -- which is
	 * most of this path, and a desktop round retired it:
	 *
	 *   the read order      A BIJECTION at all 32 (m, n) the probe runs,
	 *                       and it does not take n at all, so it cannot be
	 *                       n-selective. See charsiu_acc_index.
	 *   the input packing   a function of m and k. gate/up share k = 2048
	 *                       with attn_q, attn_o and the head, which are
	 *                       exact at m = 8.
	 *   the register stream SEPARABLE: emitted over m of 2..80 crossed with
	 *                       n of 512, 2048, 5376 and 8192, every one of its
	 *                       148 words moves with m or with n and none with
	 *                       both -- 0 joint entries. Stronger, NOT ONE WORD
	 *                       of the m=8 n=8192 stream is a word the board
	 *                       has not already run correctly at some other
	 *                       shape.
	 *   0x40b8              4*T - W on 3328 of 3328 vendor int4 streams,
	 *                       which is 3*M for a single chunk. Confirmed, not
	 *                       fitted.
	 *
	 * What is left is the OUTPUT SURFACE, the one object that is a function
	 * of m and n together. And even there the size is not it: (m=8,
	 * n=8192) and (m=32, n=2048) are both 262144 bytes and the second is
	 * exact, so the fault is not m*n -- it wants the two separately.
	 *
	 * ⚠ TWO CONTROLS, EACH ONE ENVIRONMENT VARIABLE, EACH ABLE TO FAIL:
	 *
	 *   CHARSIU_NPU_ONEDEV=1   both K slices of a tensor go to one core
	 *                          instead of running concurrently on two.
	 *                          Round 362 measured two cores corrupting each
	 *                          other through the shared CBUF, and the
	 *                          batched path submits both before waiting on
	 *                          either -- concurrency is its design. If m=8
	 *                          comes back exact on one core the fault is
	 *                          the pair, not the shape.
	 *   CHARSIU_NPU_NMAX=4096  slice n = 8192 in two. The vendor's widest
	 *                          int4 dispatch in the whole .rkllm is 4096
	 *                          output channels -- ours is the only shape
	 *                          that asks for twice that. If m=8 comes back
	 *                          exact the fault is the width.
	 *
	 * Both need this refusal lifted to say anything, and that is what
	 * CHARSIU_NPU_W4_M8=1 is for: its own name rather than a second meaning
	 * for CHARSIU_NPU_W4_BATCH, so a round that sets the batch switch for
	 * some other reason cannot quietly also let the broken width through.
	 * Nothing but a control should ever set it.
	 *
	 * ⚠⚠ AND THE READ ORDER LINE ABOVE IS NOW SETTLED RATHER THAN
	 * ARGUED. The exhaustive sweep of charsiu_acc_index over m = 2..96 and
	 * n = 512, 2048 and 8192 makes it a bijection at EVERY even width, m =
	 * 8 included, with the four-in-a-row property intact. So m = 8 is not
	 * the read order, and it is not the same fault as the odd widths
	 * refused above -- which is why it keeps its own reason string and its
	 * own switch.
	 */
	/*
	 * ⚠⚠ AND IT IS NOT ONLY m = 8. THE DENSE SWEEP FOUND m = 10 TOO, with
	 * the same signature, and this refusal was one width wide when it
	 * shipped.
	 *
	 * Llama-3.2-1B, first 8 staged tensors, both cores, widths 2..64:
	 *
	 *   2  16/16    4  32/32    6  48/48   12  96/96   14 112/112  ok
	 *   8  62/64   10  79/80                                       NOT
	 *   every odd width  0 of N                                    NOT
	 *
	 * and the misses at both 8 and 10 are ROW 0 of the n = 8192 tensors --
	 * blk.0.ffn_gate at m = 8, blk.0.ffn_up at m = 10. One shape, one row,
	 * two widths. So the second fault is not "m = 8"; it is something
	 * about row 0 of a wide output that fires at some small even widths,
	 * and 8 was simply the first one anybody asked about.
	 *
	 * ⚠ WHICH MEANS THIS LIST IS A RECORD OF WHAT HAS BEEN MEASURED, NOT A
	 * RULE. A width missing from it has been measured exact; a width the
	 * board has never seen is trusted on the layout proof alone, and m =
	 * 10 is the standing evidence that the layout proof is not enough by
	 * itself. Widen it the moment a sweep names another.
	 *
	 * 🏁 AND IT IS THE CORE PAIR, WITH THE DENSE SWEEP'S SECOND ARM AS THE
	 * PROOF RATHER THAN A GUESS.
	 *
	 *   m       two cores          one core (CHARSIU_NPU_ONEDEV=1)
	 *   8       62 of 64           64 of 64, worst 0.00e+00
	 *   10      79 of 80           80 of 80, worst 0.00e+00
	 *   12..62  exact              exact
	 *   odd     0 of N             0 of N   <- unchanged, so it is NOT this
	 *
	 * and the uncapped m = 8 pass over all 113 tensors: 33 MISS on two
	 * cores, ZERO on one. Every one of the 33 is k=2048 n=8192 row 0.
	 *
	 * So this is two cores stepping on row 0 of a wide output, at some
	 * small even widths, and nothing about the width itself. One core
	 * makes it bit identical. That is not the fix -- the pool opens one
	 * device for the whole run or two, so "one core for m = 8 only" is not
	 * a switch that exists -- but the chunker never emits 8 or 10 anyway,
	 * so this refusal is the net under a width that should never arrive.
	 */
	if ((m == 8 || m == 10) && !getenv("CHARSIU_NPU_W4_M8"))
		return "int4 at m=8 and m=10 misses row 0 of the n=8192 tensors";
	return NULL;
}

static int batch_bufs(struct charsiu_npu *g, unsigned m, unsigned nks,
		      unsigned nslots)
{
	size_t ins, outs, regs;

	if (g->bm >= m && g->bnks >= nks && g->bnslots >= nslots)
		return 0;
	reuse_keys_drop(g->bin_key, 2);   /* BOs replaced */
	if (m > g->bm)
		g->bm = m;
	if (nks > g->bnks)
		g->bnks = nks;
	if (nslots > g->bnslots)
		g->bnslots = nslots;
	m = g->bm; nks = g->bnks; nslots = g->bnslots;

	/* the f16 packer writes k * 2 bytes a row; int8 writes one */
	g->bin_stride = kmax_wide(g) * m * (g->w4 ? 2 : 1);
	ins = g->bin_stride * nks + 4096;
	regs = (size_t)nslots * 4096;
	outs = 0;

	for (unsigned d = 0; d < g->ndev; d++) {
		charsiu_bo_free(g->dev[d], &g->bin[d]);
		charsiu_bo_free(g->dev[d], &g->breg[d]);
		if (charsiu_bo_alloc(g->dev[d], ins, &g->bin[d]) ||
		    charsiu_bo_alloc(g->dev[d], regs, &g->breg[d])) {
			fprintf(stderr, "charsiu: the batch buffers wanted "
				"%.1f MB and would not allocate\n",
				(double)(ins + outs + regs) / 1e6);
			whine(g, "the batch buffers would not allocate",
			      g->kmax, g->nmax * m);
			g->bm = 0;
			return -1;
		}
	}
	free(g->bscr);
	free(g->bq);
	free(g->bd1);
	g->bscr = malloc(kmax_wide(g) * m * sizeof(*g->bscr));
	g->bq = malloc(kmax_wide(g) * m);
	/*
	 * ⚠ ONE d1 PER SLOT PER ROW, not one per row. int8's activation scale
	 * is per row AND is recomputed over each K slice's own range, and the
	 * read back happens after every slice has been packed -- so a single
	 * array would hand every slice the last one's scales.
	 */
	g->bd1 = malloc((size_t)nks * m * sizeof(*g->bd1));
	if (!g->bscr || !g->bq || !g->bd1) {
		whine(g, "the batch scratch would not allocate", g->kmax, m);
		g->bm = 0;
		return -1;
	}
	return 0;
}

/*
 * ⚠⚠ THE SEGMENTS HAVE TO ADD UP, and for a while they did not.
 *
 * At m = 32 the four of them came to 451 ms of a 606 ms batched matmul and the
 * other 155 was unnamed -- 26%, four times the fence. Optimising a 44% share
 * while a 26% one has no name is how this tree has been caught before, so
 * `prep` is the fifth: everything from entry to the first packed byte, which
 * is batch_bufs, the output allocation and the memset of Y.
 *
 * ⚠ AND IT WAS NOT MOSTLY THE PROBE, WHICH IS WHY THE COUNTER EXISTS. The
 * output buffer used to be allocated per tensor on `bout_m < m`, so a sweep
 * that walks m reallocated all 113 of them at every width while a real
 * prefill, whose chunk is one fixed 32, paid it once. It was not the sweep:
 * the counter said 225 allocations and 652 ms at ONE width, 36% of an 1811 ms
 * batched matmul, and that is what sent the buffers into a pool keyed on
 * geometry. It still counts, because a pool that quietly reallocates is the
 * same bug wearing a different name.
 */
void charsiu_npu_batch_split(struct charsiu_npu *g, double *pack, double *sub,
			     double *fence, double *read, int reset)
{
	*pack = g->bpack_us / 1e3;
	*sub = g->bsub_us / 1e3;
	*fence = g->bfence_us / 1e3;
	*read = g->bread_us / 1e3;
	if (reset)
		g->bpack_us = g->bsub_us = g->bfence_us = g->bread_us = 0.0;
}

/*
 * Wall clock across every charsiu_npu_matmul call. This is the denominator the
 * five shares are read against; see bwall_us for why the pool's hw_ms is not.
 */
void charsiu_npu_batch_pack_split(struct charsiu_npu *g, double *emit,
				  double *fini, int reset)
{
	*emit = g->bpack_emit_us / 1e3;
	*fini = g->bpack_fini_us / 1e3;
	if (reset)
		g->bpack_emit_us = g->bpack_fini_us = 0.0;
}

double charsiu_npu_batch_wall(struct charsiu_npu *g, int reset)
{
	double v = g->bwall_us / 1e3;

	if (reset)
		g->bwall_us = 0.0;
	return v;
}

double charsiu_npu_batch_prep(struct charsiu_npu *g, int reset)
{
	double v = g->bprep_us / 1e3;

	if (reset)
		g->bprep_us = 0.0;
	return v;
}

double charsiu_npu_batch_alloc(struct charsiu_npu *g, unsigned *n, int reset)
{
	double v = g->balloc_us / 1e3;

	if (n)
		*n = g->balloc_n;
	if (reset) {
		g->balloc_us = 0.0;
		g->balloc_n = 0;
	}
	return v;
}

/*
 * ⚠ HISTORY: serialised by default from 2026-08-30 to 2026-09-04, because the
 * overlapped default was wrong 13 runs in 16 -- and the reason turned out to
 * be the board's NPU voltage, see overlap_safe() below, which now decides.
 *
 * phi3 at width 24, sixteen repeats an arm, on a freshly booted board with no
 * NPU timeout anywhere in the round:
 *
 *   default (two cores, overlapped)     13 of 16 WRONG
 *   onedev  (one core)                   0 of 16
 *   serial  (two cores, never at once)   0 of 16
 *
 * The overlap is the fault. Serialising costs a batched projection's
 * parallelism and keeps decode's second core, which CHARSIU_NPU_ONEDEV does
 * not -- decode measured 14.70 tok/s on two cores against 9.71 on one, so
 * giving the core up for the whole process was never an acceptable answer.
 *
 * CHARSIU_NPU_BATCH_PARALLEL=1 puts the overlap back. It exists to price this
 * and to reproduce the fault, NOT as a configuration to run: it is the arm
 * that returns wrong text.
 */

static int batch_serial(void)
{
	static int z = -1;

	if (z < 0) {
		const char *e = getenv("CHARSIU_NPU_BATCH_PARALLEL");

		if (e && *e)
			z = *e == '0';
		else
			z = !overlap_safe(overlap_why, sizeof(overlap_why));
	}
	return z;
}

/* what batch_serial decided and why, for the report */
const char *charsiu_npu_overlap_note(void)
{
	static char note[160];
	const char *e = getenv("CHARSIU_NPU_BATCH_PARALLEL");

	if (e && *e)
		snprintf(note, sizeof(note), "the two cores %s by CHARSIU_NPU_BATCH_PARALLEL=%s",
			 batch_serial() ? "serialised" : "overlapped", e);
	else
		snprintf(note, sizeof(note), "the two cores %s: %s", batch_serial() ?
			 "serialised" : "overlapped", overlap_why);
	return note;
}

/*
 * ⚠ THE FAULT HAS A WIDTH, and the map so far (phi3, 16 runs a cell,
 * attach-once kernel, 2026-09-03/04, CHARSIU_NPU_BATCH_PARALLEL=1):
 *
 *   full chunks of 24            3 to 15 of 16 WRONG (13 of 16 at KMAX 1024)
 *   full chunks of 22            1 of 16
 *   a TAIL of 24 (62 + 24)       1 of 16
 *   full chunks of 12, 14, 16, 18, 20, 26                    0 of 16 each
 *   full chunks of 28, 32, 42, 48, 56, 58, 62, 64, 72, 74, 80   0 of 16 each
 *   tails of 2, 4, 6, 12, 14, 16, 22, 28, 30                0 of 16 each
 *   m = 8 and m = 10, the dense sweep of 08-30              wrong
 *   24 at KMAX 4096 (one K slice for K = 3072)              0 of 16
 *
 * so the kernel was never the fix (the map is on the attach-once kernel),
 * "m mod 16 in {8, 10}" was a guess the 42/56/58/72/74 row killed, "small
 * full chunks" was the next guess and 12..20 and 26 killed that. There is
 * no width law: bad {8, 10, 22, 24}, clean either side of them, and the
 * fault gets WORSE with more K slices (KMAX 1024) and goes away with fewer
 * (KMAX 4096), which is the one lead the map left. CHARSIU_NPU_PARALLEL_MIN_M=N
 * overlaps the two cores for a call of m >= N rows and serialises below
 * it; 0, the default, never overlaps.
 *
 * ⚠ 28 IS PRICED AND NOT SHIPPED. With N = 28 phase 2 was 9 of 9 identical
 * and phase 7 read TTFT 833/1218/3948/2905 ms against the serial default's
 * 1037/1565/5073/3604 (Qwen3/TinyLLAMA/Phi3/Gemma4): a fifth off the prompt.
 * What holds it back is the evidence, not the number: every width of 28 or
 * more is clean at 16 runs, on ONE model, and 16 clean runs bound a fault
 * rate at about one in six -- width 22 fails one in sixteen. A default that
 * returns wrong text one prompt in fifty is not a default. The element
 * probe (board_overlap_slots.sh) is what turns this into a mechanism, and a
 * mechanism is what makes 28 (or any N) safe rather than unobserved.
 * Speculative passes at m = 4 or 6 stay serial under any N above 6.
 */
static unsigned parallel_min_m(void)
{
	static int v = -1;

	if (v < 0) {
		const char *e = getenv("CHARSIU_NPU_PARALLEL_MIN_M");

		v = e ? atoi(e) : 0;
		if (v < 0)
			v = 0;
	}
	return (unsigned)v;
}

/*
 * CHARSIU_NPU_SUBMIT_FIRST=1 submits device 1 before device 0 in a batched
 * call. rocket puts a job on the least loaded core, so the device submitted
 * first lands on core 0 and the second on core 1 whenever both are idle; the
 * probe that found the wrong word always in DEVICE 1's slice cannot tell the
 * fd from the physical core without this. With CHARSIU_CBUF_SWAP it separates
 * fd, core and CBUF window in three runs.
 */
static int submit_first(void)
{
	static int v = -1;

	if (v < 0) {
		const char *e = getenv("CHARSIU_NPU_SUBMIT_FIRST");

		v = e ? atoi(e) == 1 : 0;
	}
	return v;
}

/* serialise the two cores for a call of m rows? */
static int batch_serial_for(unsigned m)
{
	if (!batch_serial())
		return 0;               /* CHARSIU_NPU_BATCH_PARALLEL=1: never */
	return !(parallel_min_m() && m >= parallel_min_m());
}

/* CHARSIU_NPU_PACK_GATHER=1: gather every K slice into scratch before packing,
 * which is what this did before the stride, and the control for measuring it */
static int pack_gather(void)
{
	static int v = -1;

	if (v < 0) {
		const char *e = getenv("CHARSIU_NPU_PACK_GATHER");

		v = e && *e != '0';
	}
	return v;
}

static int batch_zero(void)
{
	static int z = -1;

	if (z < 0) {
		const char *e = getenv("CHARSIU_NPU_BATCH_ZERO");

		z = e && *e != '0';
	}
	return z;
}

/*
 * THE OUTPUT BUFFER FOR ONE TENSOR, OUT OF THE POOL.
 *
 * The key is (widest slice, slots on the busier device), which is everything
 * the size depends on except m -- see the comment on struct npu_outbuf for what
 * that collapses to on a real model, and for why one buffer for everything is
 * the wrong answer to the same question.
 *
 * ⚠ IT NEVER SHRINKS, and it did not before either: the per tensor version grew
 * on `bout_m < m` and kept whatever it had, so a prefill that runs 32, 32, 32,
 * 14 reallocates nothing on the short last chunk. What is new is that the 14
 * row chunk of ffn_gate and the 14 row chunk of ffn_up are now the SAME buffer,
 * because they are the same geometry, so neither of them allocates at all.
 *
 * ⚠⚠ GIVE THE OLD ONE BACK FIRST. charsiu_bo_alloc overwrites the handle and
 * the mapping in place, so widening leaked both -- an mmap, a GEM handle and
 * its IOVA, per tensor per device, every time m grew. The probe sweeps 2, 4, 8,
 * 16, 32, 48, 64, 80 over 113 tensors on two devices, and that was about 840 MB
 * of IOVA thrown away in one run, on top of 620 MB of weights, against a 32 bit
 * window the batched path narrows addresses into without checking. A pool of
 * four buffers makes the same sweep cost four widenings a step instead of 113,
 * but the free is still what keeps it honest.
 *
 * ⚠ A BUFFER THAT WAS NEVER ALLOCATED HAS TO BE SAFE TO FREE, which is why the
 * pool is zeroed as it grows: charsiu_bo_free returns immediately on a NULL
 * map, and every path into the allocation below frees before it allocates.
 */
static struct npu_outbuf *batch_outbuf(struct charsiu_npu *g, unsigned wide,
				       unsigned slots, unsigned m)
{
	struct npu_outbuf *ob = NULL;
	double ta;
	size_t want;

	for (unsigned i = 0; i < g->n_obuf; i++)
		if (g->obuf[i].wide == wide && g->obuf[i].slots == slots) {
			ob = &g->obuf[i];
			break;
		}
	if (!ob) {
		if (g->n_obuf == g->obuf_cap) {
			unsigned cap = g->obuf_cap ? g->obuf_cap * 2 : 8;
			struct npu_outbuf *t2 = realloc(g->obuf,
						(size_t)cap * sizeof(*t2));

			if (!t2) {
				whine(g, "the batched output pool would not grow",
				      wide, m);
				return NULL;
			}
			memset(t2 + g->obuf_cap, 0,
			       (size_t)(cap - g->obuf_cap) * sizeof(*t2));
			g->obuf = t2;
			g->obuf_cap = cap;
		}
		ob = &g->obuf[g->n_obuf++];
		ob->wide = wide;
		ob->slots = slots;
	}
	/*
	 * ⚠⚠ WAIT FOR ANYTHING STILL WRITING IT BEFORE HANDING IT ON.
	 *
	 * The normal path submits, fences and reads inside one call, so the
	 * hardware has finished with the buffer long before this function can be
	 * asked for it again -- there is no thread here and no second tensor in
	 * flight. The paths that are NOT normal are the ones that matter: an
	 * empty register stream on device 1 returns after device 0 has already
	 * been submitted, and the read order table failing to allocate returns
	 * between the two fences. Per tensor those left a job writing a buffer
	 * that only the same tensor could reach, and its next call fenced it.
	 * Shared, the next tensor of the same shape is a different caller, and
	 * it would submit over the top of a job that is still running.
	 *
	 * One prep is the whole of the fix -- it is a dma_resv wait -- and the
	 * fini hands ownership back so the next submit is reading a coherent
	 * buffer. It costs nothing in the normal path, where `busy` is cleared
	 * by the fence that was going to happen anyway.
	 */
	if (ob->busy) {
		for (unsigned d = 0; d < g->ndev; d++)
			if (ob->busy & (1u << d)) {
				charsiu_bo_prep(g->dev[d], &ob->bo[d], 2000000000);
				charsiu_bo_fini(g->dev[d], &ob->bo[d]);
			}
		ob->busy = 0;
	}
	if (ob->m >= m)
		return ob;

	ta = now_us();
	want = (size_t)wide * m * 4 * slots + 4096;
	for (unsigned d = 0; d < g->ndev; d++) {
		charsiu_bo_free(g->dev[d], &ob->bo[d]);
		if (charsiu_bo_alloc(g->dev[d], want, &ob->bo[d])) {
			whine(g, "the batched output would not allocate",
			      wide, m);
			ob->m = 0;
			return NULL;
		}
	}
	ob->m = m;
	g->balloc_us += now_us() - ta;
	g->balloc_n++;
	return ob;
}

/*
 * ⚠⚠ ONE ROW IS A UNIT OF WORK, AND THE READ BACK NEVER USED THAT. Reading the
 * accumulators is 26% of a batched matmul measured across eight models on the
 * board -- 11.0 s of Phi-3.5's 33.8 s -- and it ran on one core while the pool
 * that llama_state_new starts sat idle. Before this, npudev.c contained exactly
 * one charsiu_parallel_for, on the weight pack at staging, and none on either
 * of the two hot batched loops.
 *
 * Rows are safe to split and slices are not: for a fixed slot every row writes
 * Y + r * n + s->n0, which is disjoint, while two K slices of the SAME slot
 * accumulate into the same elements and must stay ordered. So the slice loop
 * stays serial and the rows inside it go wide.
 *
 * ⚠ firstw IS INVARIANT ACROSS THE ROWS and has to be, which is why it is
 * passed in rather than recomputed here. g->bseen is only set after the whole
 * row loop, so every row of a slot sees the same flag; reading it per row from
 * threads would add a race on top of a correctness bug.
 *
 * ⚠ THE BODY BELOW IS THE ORIGINAL LOOP BODY VERBATIM, indentation included.
 * It is macro heavy, and the #undef block at its end is why: re-indenting it
 * broke the directives, and so did closing the function on the same line as
 * the last one.
 *
 * ⚠⚠ AND THE POOL LOST. Measured on the board over eight models, the whole
 * prefill got 5% to 18% SLOWER and the read share went from about 26% to about
 * 46% -- the read itself exactly doubled, on every model. The text is identical
 * in both arms, so the split is correct; it is the granularity that is wrong.
 *
 *     model            serial      pooled
 *     Phi-3.5-mini     97973 ms   109269 ms   +11.5%
 *     Qwen2.5-1.5B     44997 ms    53276 ms   +18.4%
 *     Qwen3-0.6B       40233 ms    42273 ms    +5.1%
 *     SmolLM2-1.7B     52831 ms    57827 ms    +9.5%
 *     SmolLM2-135M     15138 ms    16497 ms    +9.0%
 *     gemma-3-1b       26033 ms    30498 ms   +17.2%
 *     gemma-4-E2B      72244 ms    85228 ms   +18.0%
 *     tinyllama        41346 ms    47722 ms   +15.4%
 *
 * Two reasons, and both are about the size of the unit rather than the code:
 *
 *   - ONE DISPATCH PER SLICE. A dispatch is a broadcast, eight wakeups and two
 *     rounds of mutex, and the work it fans out is m = 32 rows of one slice --
 *     a few hundred microseconds. The pool costs more than that.
 *   - AN EVEN SPLIT OVER UNEVEN CORES. RK3576 is four A72 and four A53,
 *     sysconf gives 8, cpus_pin only acts if CHARSIU_CPUS is set, and the
 *     barrier waits for the A53s.
 *
 * ⚠ Hoisting the dispatch out of the slice loop was the obvious next move and
 * the arithmetic does not support it: it multiplies the work per dispatch by
 * the slice count, about three, against an overhead that is already larger
 * than the work. Whisper and the vision tower got 3.3x from this same pool
 * because their ranges are whole tensors, not 32 rows.
 *
 * The read is m * n * ks, and ks is the number of K slices -- so the way to
 * make it smaller is fewer, wider slices, not more cores. That is KMAX, which
 * was measured 37% slower once and attributed to the index deal that has since
 * been fixed, and has not been re-measured against the deal that replaced it.
 */
struct read_rows {
	struct charsiu_npu *g;
	const struct npu_entry *e;
	const struct npu_slot *s;
	const float *fo;
	const int32_t *io;
	float *Y;
	unsigned m, sn, ki;
	int grp, firstw;
};

/*
 * TWO ROWS OFF ONE LINE: rows 2h and 2h+1 of a channel quad sit at index and
 * index+4, 32 bytes of the same line. One read stream, two write streams.
 * The four-row form below lost 2.3x on the board to its four write streams;
 * whether two is on the right side of the A72's store buffer is a board
 * question, and CHARSIU_NPU_READ4=2 asks it. Same arithmetic per element.
 */
static int read_rows2(struct read_rows *c, uint64_t r0, uint64_t nr)
{
	struct charsiu_npu *g = c->g;
	const struct npu_entry *e = c->e;
	const struct npu_slot *s = c->s;
	const float *fo = c->fo;
	float *Y = c->Y;
	unsigned sn = c->sn, n4 = sn / 4, j;
	int firstw = c->firstw;
	const float *sc = g->w4 && c->grp ? s->sc : NULL;

	if (g->read4 != 2 || !g->w4 || !g->bmap2 || r0 % 2 || nr % 2)
		return 0;
	for (unsigned r = (unsigned)r0; r < (unsigned)(r0 + nr); r += 2) {
		const uint32_t *mp = g->bmap + (size_t)r * g->bmap_n4;
		float *y0 = Y + (size_t)r * e->t->n + s->n0;
		float *y1 = y0 + e->t->n;

#define ROW2(OP, S0, S1, S2, S3)                                             \
		for (j = 0; j < n4; j++) {                                   \
			const float *fp = fo + mp[j];                        \
			unsigned q = j * 4;                                  \
			y0[q + 0] OP fp[0] * S0; y0[q + 1] OP fp[1] * S1;    \
			y0[q + 2] OP fp[2] * S2; y0[q + 3] OP fp[3] * S3;    \
			y1[q + 0] OP fp[4] * S0; y1[q + 1] OP fp[5] * S1;    \
			y1[q + 2] OP fp[6] * S2; y1[q + 3] OP fp[7] * S3;    \
		}
		if (sc) {
			if (firstw) { ROW2(=,  sc[j * 4], sc[j * 4 + 1], sc[j * 4 + 2], sc[j * 4 + 3]) }
			else        { ROW2(+=, sc[j * 4], sc[j * 4 + 1], sc[j * 4 + 2], sc[j * 4 + 3]) }
		} else {
			if (firstw) { ROW2(=,  1.0f, 1.0f, 1.0f, 1.0f) }
			else        { ROW2(+=, 1.0f, 1.0f, 1.0f, 1.0f) }
		}
#undef ROW2
		for (j = n4 * 4; j < sn; j++) {
			unsigned base = mp[j / 4] + j % 4;
			float s0 = sc ? sc[j] : 1.0f;
			float v0 = fo[base] * s0, v1 = fo[base + 4] * s0;

			if (firstw) { y0[j] = v0; y1[j] = v1; }
			else        { y0[j] += v0; y1[j] += v1; }
		}
	}
	return 1;
}

/*
 * ⚠ FOUR ROWS OFF ONE LINE, and why the last attempt at this loop was wrong
 * about where the time went.
 *
 * The comment inside read_rows below reasons that a 16-byte run out of a
 * 64-byte line costs the DRAM four times the surface. It does not: the other
 * three quarters of that line are rows r+1, r+2 and r+3 of the SAME channel
 * quad -- index, +4, +8, +12 in the w4a16 layout -- and the row-at-a-time
 * loop comes back for them on the next three row passes, a working set of
 * n/4 lines apart, which the L2 keeps. What it pays is one L2 round trip per
 * row per line instead of one per line, four times the load instructions and
 * four times the table reads.
 *
 * Measured on the host with tools/bench_gather, byte for byte the same Y:
 * walking the surface in order and scattering into Y, the fix that comment
 * proposes, is 4 to 10 times SLOWER (the scatter side gets the partial lines
 * and the TLB); taking the four rows off each line once is 1.4 to 2.2 times
 * faster at every shape from m = 32 by 1024 to m = 80 by 8192. The board is
 * the one that prices it -- the board is where a NEON form of the row loop
 * moved nothing -- and CHARSIU_NPU_READ4=0 is the row-at-a-time control.
 *
 * The premise is checked on the table when it is built (g->bmap4), the
 * arithmetic per element is the row loop's exactly (multiply, then assign
 * or add), and anything this cannot take -- int8, a width that is not a
 * multiple of four, a pooled range that is not four aligned -- goes through
 * the row loop unchanged.
 */
static int read_rows4(struct read_rows *c, uint64_t r0, uint64_t nr)
{
	struct charsiu_npu *g = c->g;
	const struct npu_entry *e = c->e;
	const struct npu_slot *s = c->s;
	const float *fo = c->fo;
	float *Y = c->Y;
	unsigned sn = c->sn, n4 = sn / 4, j;
	int firstw = c->firstw;
	const float *sc = g->w4 && c->grp ? s->sc : NULL;

	if (g->read4 != 4 || !g->w4 || !g->bmap4 || r0 % 4 || nr % 4)
		return 0;
	for (unsigned r = (unsigned)r0; r < (unsigned)(r0 + nr); r += 4) {
		const uint32_t *mp = g->bmap + (size_t)r * g->bmap_n4;
		float *y0 = Y + (size_t)r * e->t->n + s->n0;
		float *y1 = y0 + e->t->n, *y2 = y1 + e->t->n, *y3 = y2 + e->t->n;

#define ROW4(OP, S0, S1, S2, S3)                                             \
		for (j = 0; j < n4; j++) {                                   \
			const float *fp = fo + mp[j];                        \
			unsigned q = j * 4;                                  \
			y0[q + 0] OP fp[0]  * S0; y0[q + 1] OP fp[1]  * S1;  \
			y0[q + 2] OP fp[2]  * S2; y0[q + 3] OP fp[3]  * S3;  \
			y1[q + 0] OP fp[4]  * S0; y1[q + 1] OP fp[5]  * S1;  \
			y1[q + 2] OP fp[6]  * S2; y1[q + 3] OP fp[7]  * S3;  \
			y2[q + 0] OP fp[8]  * S0; y2[q + 1] OP fp[9]  * S1;  \
			y2[q + 2] OP fp[10] * S2; y2[q + 3] OP fp[11] * S3;  \
			y3[q + 0] OP fp[12] * S0; y3[q + 1] OP fp[13] * S1;  \
			y3[q + 2] OP fp[14] * S2; y3[q + 3] OP fp[15] * S3;  \
		}
		if (sc) {
			if (firstw) { ROW4(=,  sc[j * 4], sc[j * 4 + 1], sc[j * 4 + 2], sc[j * 4 + 3]) }
			else        { ROW4(+=, sc[j * 4], sc[j * 4 + 1], sc[j * 4 + 2], sc[j * 4 + 3]) }
		} else {
			if (firstw) { ROW4(=,  1.0f, 1.0f, 1.0f, 1.0f) }
			else        { ROW4(+=, 1.0f, 1.0f, 1.0f, 1.0f) }
		}
#undef ROW4
		/* the tail past the last whole quad, per row, as the row loop does */
		for (j = n4 * 4; j < sn; j++) {
			unsigned base = mp[j / 4] + j % 4;
			float s0 = sc ? sc[j] : 1.0f;
			float v0 = fo[base] * s0, v1 = fo[base + 4] * s0,
			      v2 = fo[base + 8] * s0, v3 = fo[base + 12] * s0;

			if (firstw) { y0[j] = v0; y1[j] = v1; y2[j] = v2; y3[j] = v3; }
			else        { y0[j] += v0; y1[j] += v1; y2[j] += v2; y3[j] += v3; }
		}
	}
	return 1;
}

static void read_rows(void *ctx, uint64_t r0, uint64_t nr)
{
	struct read_rows *c = ctx;
	struct charsiu_npu *g = c->g;
	const struct npu_entry *e = c->e;
	const struct npu_slot *s = c->s;

	if (read_rows2(c, r0, nr) || read_rows4(c, r0, nr))
		return;
	const float *fo = c->fo;
	const int32_t *io = c->io;
	float *Y = c->Y;
	unsigned m = c->m, sn = c->sn, ki = c->ki;
	int grp = c->grp, firstw = c->firstw;

	(void)m; (void)ki; (void)grp; (void)io; (void)fo;
	for (unsigned r = (unsigned)r0; r < (unsigned)(r0 + nr); r++) {
					const uint32_t *mp = g->bmap
							  + (size_t)r * g->bmap_n4;
					float *yr = Y + (size_t)r * e->t->n
						  + s->n0;
					unsigned n4 = sn / 4, j;

					/*
					 * ⚠ FOUR AT A TIME OFF ONE INDEX. The
					 * tail is whatever a slice's width
					 * leaves over, and it recomputes its
					 * own base rather than reading a table
					 * entry that may not exist.
					 */
/*
					 * ⚠ THE BRANCH IS OUTSIDE THE LOOP, and
					 * it cannot be a multiply by zero: Y is
					 * the caller's buffer and a bit pattern
					 * in untouched memory can be a NaN,
					 * which times zero is a NaN and not a
					 * zero.
					 *
					 * ⚠⚠ AND A NEON FORM OF THIS WAS
					 * WRITTEN, MEASURED AND TAKEN OUT.
					 *
					 * The four are one vector, so a run is
					 * vld1q from one index, vmulq, vst1q,
					 * and it was bit identical to this --
					 * checked, including the rounding,
					 * because the C here is a multiply then
					 * an add and an fmla rounds once where
					 * it rounds twice. On this toolchain
					 * mul-then-add differs from the C in 0
					 * of a million and fmla in 227529, so
					 * the C is not contracted and the
					 * vector form matched.
					 *
					 * The board moved by NOTHING: read was
					 * 223, 320, 555 ms at m of 32, 48 and
					 * 80 before it and 229, 337, 580 after.
					 *
					 * ⚠ AND THE ARITHMETIC SAYS WHY, which
					 * is the part worth keeping. The gather
					 * moves about 403 MB at m = 32 and 1007
					 * at m = 80 -- Y once per K slice, read
					 * and written -- in 229 and 580 ms,
					 * which is 1.76 and 1.74 GB/s, the same
					 * rate at both. A run is 16 BYTES and a
					 * cache line is 64, and the runs are
					 * scattered, so the DRAM sees four
					 * times that: about 7 GB/s against this
					 * board's 9.4 roof, 75% of it.
					 *
					 * It was never instruction bound.
					 * Fewer instructions cannot help and
					 * the only lever left is fewer BYTES:
					 * walking the source sequentially and
					 * scattering into Y would use whole
					 * lines instead of a quarter of each.
					 */
#define GATHER4(OP, VAL)                                                     \
					for (j = 0; j < n4; j++) {           \
						float *yp = yr + j * 4;      \
						VAL;                         \
						yp[0] SC_##OP v0;            \
						yp[1] SC_##OP v1;            \
						yp[2] SC_##OP v2;            \
						yp[3] SC_##OP v3;            \
					}
#define SC_ASSIGN  =
#define SC_ADD     +=
#define W4G  const float *fp = fo + mp[j], *cp = s->sc + j * 4;              \
	     float v0 = fp[0]*cp[0], v1 = fp[1]*cp[1],                       \
		   v2 = fp[2]*cp[2], v3 = fp[3]*cp[3]
#define W4   const float *fp = fo + mp[j];                                   \
	     float v0 = fp[0], v1 = fp[1], v2 = fp[2], v3 = fp[3]
#define I8   const int32_t *ip = io + mp[j];                                 \
	     float v0 = (float)ip[0]*d1, v1 = (float)ip[1]*d1,               \
		   v2 = (float)ip[2]*d1, v3 = (float)ip[3]*d1
					if (g->w4 && grp) {
						const float *sc = s->sc;

						if (firstw) { GATHER4(ASSIGN, W4G) }
						else        { GATHER4(ADD, W4G) }
						for (j = n4 * 4; j < sn; j++) {
							float v = fo[mp[j / 4] + j % 4] * sc[j];

							if (firstw) yr[j] = v;
							else        yr[j] += v;
						}
					} else if (g->w4) {
						if (firstw) { GATHER4(ASSIGN, W4) }
						else        { GATHER4(ADD, W4) }
						for (j = n4 * 4; j < sn; j++) {
							float v = fo[mp[j / 4] + j % 4];

							if (firstw) yr[j] = v;
							else        yr[j] += v;
						}
					} else {
						float d1 = g->bd1[(size_t)ki * m + r];

						if (firstw) { GATHER4(ASSIGN, I8) }
						else        { GATHER4(ADD, I8) }
						for (j = n4 * 4; j < sn; j++) {
							float v = (float)io[mp[j / 4] + j % 4] * d1;

							if (firstw) yr[j] = v;
							else        yr[j] += v;
						}
					}
#undef GATHER4
#undef SC_ASSIGN
#undef SC_ADD
#undef W4G
#undef W4
#undef I8
	}
}

/*
 * ⚠ THE BODY IS A STATIC INNER SO THE WALL CLOCK CANNOT BE FORGOTTEN. This
 * function has eleven return points and wrapping each of them is a bug waiting
 * for the twelfth. See bwall_us.
 */
static int npu_matmul_inner(struct charsiu_npu *g, int id, const float *X,
			    unsigned m, float *Y)
{
	struct npu_entry *e;
	struct npu_outbuf *ob;
	struct charsiu_joblist jl;
	double t0, tprep = now_us();

	if (g->dead || id < 0 || (unsigned)id >= g->n_ent || m < 2)
		return -1;
	/*
	 * ⚠⚠ int8 IS THE PATH THAT DOES MORE THAN ONE ROW, and that is not a
	 * preference, it is the only thing on this board with evidence.
	 *
	 * npu_gemm_test is EXACT on the int8 accumulator at m = 1, 2, 4 and 8
	 * and at every N from 32 to 2048. w4a16 produces ONE row and no
	 * register makes it produce two: fed the same activation twice, row 1
	 * comes back matching row 0 in 1 of 2048, and every word that differs
	 * between the two streams has been put back one at a time -- the DPU
	 * and RDMA blocks are identical to begin with, the CNA differences are
	 * datatype scaling, CORE 0x301c is inert and 0x3018 is the arithmetic
	 * switch.
	 *
	 * ⚠⚠ THE LAST SENTENCE OF THIS USED TO BE "the vendor never batches a
	 * weight matmul at all, so there is no M > 1 int4 stream anywhere to
	 * copy", AND IT IS FALSE. It came from reading M off the row count.
	 * The vendor's Llama-3.2-1B .rkllm holds 3328 int4 streams and 2816 of
	 * them -- 85% -- are batched, at M of 16, 24, 32, 40, 48, 64 and 80.
	 * They read as one row because the int4 path carries M on the WIDTH,
	 * so its row count is 1 whatever M is; its fp16 streams put M on the
	 * height, where rows and pixels agree, which is why the two axes were
	 * never told apart. tools/cmp_vendor.py compares on the pixel count now.
	 *
	 * ⚠ The largest int4 M they emit is 80, which is also where this board's
	 * batched prefill stops being exact -- but that is NOT the same fact
	 * twice: ours is int8 on the height axis and theirs is int4 on the
	 * width. Their int8 head runs 128 rows, one row high like the rest, so
	 * they put M on the width for both weight formats and use the height
	 * for nothing but fp16 attention. Our 80 is therefore a property of an
	 * arrangement they never use, and board_rows_sweep.sh now asks both.
	 *
	 * So every word of the paragraph above is about the HEIGHT axis, which
	 * is the one that produces a single row. What has never run on this
	 * board is the width axis, and that is now the only form of this the
	 * vendor is known to use.
	 *
	 * Batching amortises the weight bytes over m rows, which is exactly
	 * what makes int8's extra byte cheap: at m = 32 the same weights serve
	 * thirty two rows either way.
	 */
	/*
	 * ⚠⚠ int4 PRODUCES ONE ROW AND MUST REFUSE, and this guard was removed
	 * by the commit that taught this function int8.
	 *
	 * w4a16 computes exactly one row whatever it is asked for. Five rounds
	 * established that: fed the SAME activation twice, row 1 comes back
	 * matching row 0 in 1 of 2048, the DPU and RDMA blocks are identical to
	 * a stream that does two rows, and every CNA word that differs was put
	 * back one at a time. Adding int8 replaced the refusal with a comment
	 * about int8 being the path with evidence, and left nothing stopping
	 * int4 from taking it.
	 *
	 * The board said so immediately and in the only way that counts. An
	 * int4 prompt came back "ITES  (un- a- -  ( -  ' \ l'" at a very
	 * respectable 37.46 tok/s, which is the speed of a wrong answer, and
	 * the default config passes CHARSIU_NPU_W4V=1 -- so this was every
	 * int4 model with a prompt of two tokens or more.
	 *
	 * The caller falls back to a row at a time, which is correct and is
	 * what int4 did before any of this existed.
	 *
	 * ⚠ CHARSIU_NPU_W4_BATCH=1 LIFTS IT, and only together with
	 * CHARSIU_M_AXIS=w. The height axis is the form five rounds proved
	 * writes one row, so letting it batch would just reproduce the wrong
	 * answer at 37 tok/s again. The width axis is what the vendor's own
	 * stream does and what charsiu now matches it on, and it has never been
	 * on this board -- so it is an experiment with a switch, not a default.
	 * llama_batch_probe checks every row before it times anything.
	 */
	if (g->w4) {
		const char *why = w4_batch_why_not(m);

		if (why) {
			whine(g, why, (unsigned)g->ent[id].t->k,
			      (unsigned)g->ent[id].t->n);
			return -1;
		}
	}
	e = &g->ent[id];
	/*
	 * ⚠⚠ THE INPUT SURFACE HAS A CEILING AND WE FOUND IT BY GOING OVER IT.
	 *
	 * On the width axis the surface is (k_slice / 32) * m CBUF entries.
	 * charsiu_emit_job splits the CBUF window above 4096 of them, a rule
	 * read off the vendor's own file -- and its LARGEST split sample is
	 * 5120, which is 2048 at m = 80. Above that nothing in that file says
	 * what the stream should look like, and the board says our guess is
	 * wrong. Nine cells, all of them:
	 *
	 *   surf 2560 3840 4096 5120   right    (1024x80, 1536x80, 4096x32, 2048x80)
	 *   surf 7680 10240            WRONG    (3072x80, 4096x80)
	 *   surf 1024 2048 3072        right    (1024x32, 2048x32, 3072x32)
	 *
	 * so the bound is in (5120, 7680] and 5120 is exactly where the
	 * evidence stops. A wider surface than that is extrapolation, and this
	 * refuses it rather than computing a wrong answer quietly.
	 *
	 * ⚠ IT IS NOT A SIZE LIMIT ON K, which is what three earlier readings
	 * of this thought it was. K = 4096 is fine at m = 32 and wrong at
	 * m = 80; a single dispatch at K = 4096, N = 1536, m = 80 is EXACT
	 * (phase 15). Only the product moves it.
	 *
	 * ⚠ AND THE FIX, IF SOMEBODY WANTS THESE WIDTHS, IS NOT A BIGGER
	 * NUMBER HERE. It is whatever the vendor emits above 5120, which is a
	 * third window state nothing on disk has ever shown -- so it has to be
	 * searched for, not derived.
	 *
	 * 🏁 AND THERE IS NOTHING TO SEARCH FOR, at least not in this file.
	 * The census above quoted 5120 as "its LARGEST SPLIT SAMPLE", which is
	 * a sample and reads like one. The whole file, all 8808 convolutions
	 * of Llama-3.2-1B-rk3576-w4a16 (tools/rkllm_regcmd.py, 2026-09-05):
	 *
	 *   max input surface   5120
	 *   above 5120          0 dispatches
	 *
	 * and it holds that ceiling by LOWERING M AS K RISES, which is the
	 * same trade this guard forces on us and not a workaround for it:
	 *
	 *   K 2048  ->  m 80   surface 5120
	 *   K 4096  ->  m 40   surface 5120
	 *   K 3968  ->  m 33   surface 4092
	 *   K 3744  ->  m 35   surface 4095
	 *
	 * So 5120 is not a conservative reading of the vendor, it IS the
	 * vendor, and KMAX 2048 at a chunk of 80 is its widest configuration
	 * exactly. Whatever is above the line, the closed stack does not go
	 * there either.
	 *
	 * ⚠⚠ 5120 IS MEASURED, AND ITS CAUSE IS NOT KNOWN. Read it as a fence
	 * post, never as an explanation.
	 *
	 * Walking the surface directly, one slice, K held at 4096 so that it is
	 * 128 * m:
	 *
	 *     4096   exact        6144   WRONG
	 *     5120   exact        7168   WRONG
	 *                         8192   WRONG
	 *
	 * so the line is in (5120, 6144] and 5120 is the last value measured
	 * good. It is also, independently, the largest surface in the vendor's
	 * own file. Two lines of evidence landing on one number is why the
	 * guard sits there.
	 *
	 * ⚠ FIVE EXPLANATIONS HAVE FITTED THIS AND DIED, in order: the CBUF
	 * split pair (already in job.c and already right), K on its own
	 * (K = 4096 is fine at m = 32), K * N at 2 MiB (killed by a single
	 * dispatch at 3072 KiB), the core pair (identical on one core), and
	 * window 1's base at 0x1c00 = 7168, which sat in the gap the data left
	 * and was killed by 6144 failing. Each fitted everything known when it
	 * was proposed. The bound has held and every story about it has not.
	 *
	 * ⚠ AND TIGHTENING IT BUYS NOTHING. At 6144, the far end of the
	 * bracket, KMAX 3072 still needs a chunk of 53 and K at m = 80 still
	 * stops at 2457 -- no width becomes reachable that is not reachable
	 * now, and phase 11 measured a chunk of 96 tied with 80 anyway.
	 */
	{
		unsigned kw = 0, i;

		for (i = 0; i < e->count; i++) {
			unsigned sk = (unsigned)g->slot[e->first + i].job.mm.k;

			if (sk > kw)
				kw = sk;
		}
		/*
		 * ⚠⚠ AND IT ONLY MEANS ANYTHING ON THE WIDTH AXIS. (kw / 32) * m
		 * IS THE WIDTH AXIS'S SURFACE AND NOBODY ELSE'S: charsiu_emit_job
		 * sets inw = m only when wide, so on the height axis the input is
		 * one column of m rows and the surface is 1 * m -- three orders
		 * smaller and nowhere near any ceiling. job.c had already written
		 * down that a rule read off int4 must not reach int8, and this
		 * guard shipped without the gate and reached it anyway.
		 *
		 * ⚠ IT COST THE VISION TOWER, which is the one caller that opens
		 * want_w4 = 0 with a wide K. SmolVLM-256M's ffn_down is K = 3072
		 * and charsiu_pool_rows batches 64 rows, so (3072 / 32) * 64 is
		 * 6144 -- the first cell in the int4 bracket above -- and all
		 * twelve of them plus the idefics3 projector were refused. The
		 * scoreboard's encoder went 5.57 s to 31.0 s, the pool reported
		 * "73 asked and 13 fell back", and 87.6% of the run was ffn
		 * matmuls with only 1287 ms of it on the hardware. Gated, the
		 * same board reads 5.37 s, 0 fell back, ffn 962 ms.
		 *
		 * ⚠ 15.5 s IS THE WRONG BASELINE and I quoted it first: that is
		 * board_modalities' vision number from a different round, not
		 * this scoreboard's encoder.
		 *
		 * whisper opens want_w4 = 0 too and escaped only by being small:
		 * its widest K is 4 * n_audio_state, which is 2048 at base and
		 * 4096 entries at 64 rows, just under.
		 */
		/*
		 * ⚠ AND THE HEIGHT AXIS HAS ITS OWN LINE. Phase 19 walked it on
		 * int8: (K / 32) * rows of 4096, 6144, 7680 and 8192 exact,
		 * 8960 and 10240 wrong on every row -- with K alone (128 at 32
		 * rows) and the output (122880 floats) both exact, so it is the
		 * input surface again, in (8192, 8960]. The same hatch lifts it
		 * for the probe that walks it.
		 */
		if (!charsiu_m_axis_wide_for(g->w4)) {
			if (!getenv("CHARSIU_NPU_ANY_SURFACE") &&
			    (size_t)(kw / 32) * m > 8192) {
				whine(g, "the input surface on the height axis is "
				      "past 8192, where the board says every row "
				      "comes back wrong", kw, m);
				return -1;
			}
			kw = 0;
		}
		/*
		 * ⚠⚠ AND A PROBE HAS TO BE ABLE TO ASK ABOUT WHAT THIS
		 * REFUSES. w4_batch_why_not learned this already and says so
		 * above itself: asking is how every line of its table was
		 * measured and the only way any of it gets re-measured. This
		 * guard shipped without the hatch and the very next round --
		 * phase 16, one tensor sliced, the rung this fault has been
		 * missing all week -- came back with its two interesting cells
		 * REFUSED BY IT. Nothing but a probe should set this.
		 */
		if (!getenv("CHARSIU_NPU_ANY_SURFACE") &&
		    (size_t)(kw / 32) * m > 5120) {
			/*
			 * ⚠ CHARSIU_NPU_KFIT IS THE LIKELY WAY TO GET HERE, and
			 * the two are in direct conflict at the shipped width.
			 * KFIT widens the last slice to kmax + K % kmax, which
			 * at KMAX 2048 is 2816 on Qwen2.5 and gemma-3-1b and
			 * 3584 on tinyllama -- 7040 and 8960 entries at a chunk
			 * of 80, both past this. So turning KFIT on there does
			 * not make the batch faster, it makes there be no batch:
			 * this returns -1 and the caller falls back a row at a
			 * time. It was measured +7.3% on gemma-3-1b back when
			 * KMAX was 1024 and the widest it produced was 1792.
			 */
			whine(g, g->kfit
			      ? "KFIT widened a slice past the input surface "
				"ceiling, so this tensor is not batched at all"
			      : "the input surface is past the widest the "
				"vendor's own file ever splits", kw, m);
			return -1;
		}
	}
	if (e->n_npu != (unsigned)e->t->n) {
		/* a CPU share would have to be batched too, and is not */
		whine(g, "batched cannot split rows with the CPU",
		      (unsigned)e->t->k, (unsigned)e->t->n);
		return -1;
	}
	{
		unsigned nslots[2] = { 0, 0 }, wide = 0, most;

		for (unsigned i = 0; i < e->count; i++) {
			const struct npu_slot *s = &g->slot[e->first + i];

			nslots[s->di]++;
			if (s->job.mm.n > wide)
				wide = (unsigned)s->job.mm.n;
		}
		most = nslots[0] > nslots[1] ? nslots[0] : nslots[1];
		if (batch_bufs(g, m, e->k_slices, most))
			return -1;
		/*
		 * ⚠ SIZED FOR THIS TENSOR'S OWN WIDEST SLICE, not for nmax.
		 * attn_q is 2048 wide and the head is 8192; one buffer for both
		 * makes attn_q pay the head's cache maintenance on every call.
		 */
		g->bout_stride = (size_t)wide * m * 4;
		/*
		 * ⚠⚠ AND SHARED WITH EVERY TENSOR OF THE SAME SHAPE, which is
		 * where 652 ms of a 1811 ms batched matmul went. The buffer this
		 * hands back is sized (wide, most, m) and nothing else, so the
		 * two lines above still decide its bytes -- what changed is that
		 * ffn_gate and ffn_up, and all 32 of them across the layers, now
		 * ask for one buffer between them instead of 32.
		 *
		 * ⚠ AND THAT CLOSES `prep`. It was 26% of a batched matmul;
		 * removing the zero of Y, which was the whole of the hypothesis
		 * at the time, moved it 12%. The allocation was the other half,
		 * and it was the half nobody had counted.
		 *
		 * ⚠ ONE TENSOR IS IN FLIGHT AT A TIME, which is what makes that
		 * safe. There is no thread in this path: llama.c's prefill calls
		 * matmul_rows for one projection at a time and charsiu_pool_rows
		 * walks its chunks in a loop, and this function does not return
		 * until it has fenced and read every device it submitted to.
		 * batch_outbuf holds the net under the paths where it returns
		 * early anyway.
		 */
		ob = batch_outbuf(g, wide, most, m);
		if (!ob)
			return -1;
		g->last_ob = ob;
		g->last_id = id;
	}

	/*
	 * ⚠⚠ THE ZERO OF Y WAS 26% OF A BATCHED MATMUL, and it was a whole
	 * extra pass over the output for nothing.
	 *
	 * The gather accumulates -- `yr[j] += ...` -- because a tensor's K
	 * slices each contribute a partial sum, so Y had to start at zero. At
	 * m = 32 on Llama-3.2-1B that is 64.6 MB zeroed and then 64.6 MB
	 * written, and the board measured the zero at 155 ms of a 600 ms
	 * matmul: 417 MB/s, the same rate at every width, which is what says
	 * it is the memset and not the allocation beside it.
	 *
	 * So the FIRST contribution to an output range assigns and the rest
	 * accumulate, and nothing is zeroed but a byte per n slice.
	 *
	 * ⚠ THE FLAG IS PER OUTPUT RANGE, NOT PER K SLICE, and that is not a
	 * detail. The same output range's ki = 0 and ki = 1 can land on
	 * DIFFERENT devices -- deal_pick makes no promise at all about which,
	 * and the index deal it replaced did not either once ns was odd -- and
	 * the read loop walks devices outermost, so ki = 1 can be read first.
	 * "ki == 0 assigns" would have clobbered it.
	 */
	g->bseen_dev = 0;
	if (g->bseen_n < e->n_slices) {
		unsigned char *t2 = realloc(g->bseen, e->n_slices);

		if (!t2) {
			whine(g, "the first-write flags would not allocate",
			      e->n_slices, 0);
			return -1;
		}
		g->bseen = t2;
		g->bseen_n = e->n_slices;
	}
	memset(g->bseen, 0, e->n_slices);
	/*
	 * ⚠⚠ THE CONTROL FOR ALL OF THE ABOVE, because assign-on-first-write
	 * is the one thing here that can hand back a caller's stale buffer.
	 *
	 * If an output range never gets a first write -- a slice skipped, a
	 * count off by one, a flag set for a range nothing covers -- then with
	 * the memset gone Y keeps whatever was in it, and what was in it is
	 * the PREVIOUS token's answer. That is invisible on a fresh buffer, it
	 * is invisible whenever the stale value happens to be close, and it
	 * looks exactly like the intermittent wrong text gemma4 and phi3 are
	 * producing: right ten runs, then a sentence about practicing.
	 *
	 * CHARSIU_NPU_BATCH_ZERO=1 puts the whole-buffer zero back. It costs
	 * the 26% this optimisation bought and it decides the question: if the
	 * text goes right and STAYS right, the first-write bookkeeping is
	 * wrong; if it stays wrong, this whole family is excluded and nobody
	 * has to think about it again.
	 */
	if (batch_zero())
		memset(Y, 0, (size_t)m * e->t->n * sizeof(*Y));
	g->bprep_us += now_us() - tprep;
	t0 = now_us();

	/*
	 * ⚠ ONE SUBMIT PER DEVICE FOR THE WHOLE PROJECTION, and both issued
	 * before either is waited on, which is what decode does. The fence was
	 * 5072 ms of a 7448 ms report when this waited on every slice.
	 */
	/*
	 * ⚠ REUSE IS DECIDED PER DEVICE. Each device's key says whether ITS
	 * input BO already holds this X at this width for this K; the caller's
	 * declaration says whether that is allowed to matter. Either alone
	 * packs that device. A device this tensor has no slot on is neither
	 * packed nor read, and its key is left describing what is there.
	 */
	uint8_t zp = g->slot[e->first].job.input_zero_point;

	/*
	 * ⚠⚠ A LEADER DROPS EVERY KEY FIRST, on the devices this tensor will
	 * not pack as much as on the ones it will: the caller not declaring
	 * its input unchanged means X is new everywhere. reusekey.h has the
	 * round that found out (phase 22: q on core 0, gate on core 1, up
	 * back on core 0 reading the attention input).
	 */
	if (!g->reuse_ask)
		reuse_keys_drop(g->bin_key, sizeof(g->bin_key) / sizeof(g->bin_key[0]));

	for (unsigned dd = 0; dd < g->ndev; dd++) {
		unsigned d = g->ndev == 2 && submit_first() ? dd ^ 1u : dd;
		unsigned nt = 0, nh = 0, done_ki = 0, need = 0;
		double tp = now_us();

		for (unsigned i = 0; i < e->count; i++)
			if (g->slot[e->first + i].di == d)
				need |= 1u << (i / e->n_slices);
		int reuse = g->reuse_ask &&
			    reuse_key_hit(&g->bin_key[d], X, m, e->t->k, zp, need);

		if (g->reuse_ask) {
			const struct reuse_key *bk = &g->bin_key[d];

			if (reuse) {
				g->reuse_hits++;
			} else {
				g->reuse_misses++;
				if (!bk->valid)
					g->reuse_why[0]++;
				else if (bk->x != X)
					g->reuse_why[1]++;
				else if (bk->m != m || bk->k != e->t->k ||
					 bk->zp != zp)
					g->reuse_why[2]++;
				else
					g->reuse_why[3]++;
			}
		}
		if (!reuse)
			g->bin_key[d].valid = 0;   /* until this device's pack lands */
		/*
		 * ⚠ NO PREP ON A BUFFER ONLY THE DEVICE READS, the way the row
		 * path already does with `in` (CHARSIU_NPU_INPREP puts it
		 * back). PREP_BO is a fence wait on WRITERS plus
		 * dma_sync_sgtable_for_cpu over the whole object, and neither
		 * does anything for a buffer the hardware never writes and the
		 * CPU is about to overwrite; the FINI that cleans the CPU's
		 * writes out to the device stays. Phase 20 priced the batched
		 * call's "pack" at 22 ms a pass for FOUR rows -- 150 us a call,
		 * which is not 16 KB of packing, it is the ioctls: two preps
		 * and two finis a device a call.
		 */
		if (g->inprep) {
			if (!reuse)
				charsiu_bo_prep(g->dev[d], &g->bin[d], 1000000000);
			charsiu_bo_prep(g->dev[d], &g->breg[d], 1000000000);
		}
		g->handles[nh++] = g->bin[d].handle;

		/*
		 * ⚠⚠ ONCE PER K SLICE, NOT ONCE PER SLOT.
		 *
		 * A tensor's slots are n_slices wide by k_slices deep, and every
		 * slot in a K column reads the SAME activation. Packing inside
		 * the slot loop packed it n_slices times over -- sixteen times
		 * for the output head -- and the time split says packing is 259
		 * ms of a 758 ms pass at m = 32. The regions were already
		 * indexed by K slice; this stops writing each one repeatedly.
		 */
		for (unsigned i = 0; i < e->count; i++) {
			const struct npu_slot *s = &g->slot[e->first + i];
			unsigned sk = s->job.mm.k, ki = i / e->n_slices;
			struct charsiu_matmul mm = s->job.mm;

			if (reuse || s->di != d || ((done_ki >> ki) & 1u))
				continue;
			done_ki |= 1u << ki;
			mm.m = m;
			/*
			 * ⚠⚠ PACK OUT OF X ONLY WHEN THE SLICE IS THE WHOLE
			 * ROW, AND THE BOARD IS WHY.
			 *
			 * A K slice is columns [k0, k0 + sk) of every row, and
			 * this used to copy them into g->bscr so the packer
			 * could read contiguous rows: fourteen bytes moved per
			 * element packed. Handing the packer X and a row
			 * stride removes that round trip, and on a tensor whose
			 * K fits one slice it is free money -- Qwen2.5-1.5B,
			 * whose K is 1536 against a 2048 slice, went from 2628
			 * to 2215 ms of packing.
			 *
			 * On a tensor that is CUT it is a loss, and a large
			 * one. Phi-3.5's K is 3072, so a 2048 slice and a 1024
			 * slice, and the packer walks groups outermost with m
			 * strided reads a group. Reading X the stride is the
			 * whole row, so every line fetched carries columns the
			 * slice does not want and the working set is the whole
			 * activation; reading bscr the stride is the slice, the
			 * lines are all wanted, and the buffer was written
			 * sequentially a moment earlier. Measured on the board:
			 * packing 4384 ms with the gather, 6583 without it, and
			 * the whole matmul entry 21312 against 23724.
			 *
			 * So the gather stays wherever the tensor is sliced,
			 * and int8 keeps it always: it takes a per row maximum
			 * over the slice and quantises into g->bq, which is a
			 * pass over the slice either way.
			 */
			/*
			 * ⚠ THE ARM, BECAUSE THE FIRST TWO ROUNDS OF THIS WERE
			 * NOISE. Three board runs of phase 9 disagreed by more
			 * than the change: Phi-3.5 packed 4384, 6583 and 3158
			 * ms on paths that should have been the same twice,
			 * and Qwen2.5 2628, 2215 and 3497. Phase 9 runs under
			 * the ondemand governor and packing is CPU work, so a
			 * build against a build measures the governor.
			 * CHARSIU_NPU_PACK_GATHER=1 forces the gather, so one
			 * binary can run both arms in one session.
			 */
			if (!g->w4 || sk != e->t->k || pack_gather())
				for (unsigned r = 0; r < m; r++)
					memcpy(g->bscr + (size_t)r * sk,
					       X + (size_t)r * e->t->k + s->k0,
					       sk * sizeof(*g->bscr));
			if (!g->w4) {
				/*
				 * int8's scale is per row and taken over THIS K
				 * slice's own range, so it is kept per K slice
				 * for the read back.
				 *
				 * ⚠⚠ AND IT MULTIPLIES BY THE RECIPROCAL BECAUSE
				 * charsiu_act_q1 DOES. x * (1/d) and x / d are
				 * not the same float: the reciprocal rounds
				 * once and the product rounds again, the divide
				 * rounds once, and a value sitting halfway
				 * between two codes comes out one code apart.
				 * This path divided and the row loop multiplied,
				 * so a batch and its own row-by-row reference
				 * disagreed on a handful of near-zero channels
				 * -- 127 of 49152 on the vision tower's own
				 * shape, one row of 64, mean got/want 1.0007.
				 * Small enough to look like noise and stable
				 * enough to look like structure, which is how it
				 * cost a board round to tell apart.
				 *
				 * ⚠ THIS ONLY MAKES THE TWO IDENTICAL ON A SINGLE
				 * SLICE. q1's amax is over the whole row and this
				 * one is over sk, so a multi-slice int8 tensor is
				 * quantised FINER here on purpose and cannot
				 * match the row loop to 0.1% -- and should not
				 * be asked to.
				 */
				for (unsigned r = 0; r < m; r++) {
					float mx = 0.0f, d1, id1;

					for (unsigned kk = 0; kk < sk; kk++) {
						float v = fabsf(g->bscr[(size_t)r * sk + kk]);

						if (v > mx)
							mx = v;
					}
					d1 = mx > 0.0f ? mx / 127.0f : 1.0f;
					id1 = d1 != 0.0f ? 1.0f / d1 : 0.0f;
					g->bd1[(size_t)ki * m + r] = d1;
					for (unsigned kk = 0; kk < sk; kk++) {
						int q = (int)lrintf(g->bscr[(size_t)r * sk + kk] * id1);

						if (q > 127) q = 127;
						if (q < -127) q = -127;
						g->bq[(size_t)r * sk + kk] = (uint8_t)(q + 128);
					}
				}
				charsiu_pack_input(&mm, g->bq,
						   (uint8_t *)g->bin[d].map + ki * g->bin_stride,
						   g->bin_stride, s->job.input_zero_point);
			} else if (sk == e->t->k && !pack_gather()) {
				/* the slice is the whole row, so k0 is 0 */
				charsiu_pack_input_f16_stride(&mm, X, e->t->k,
						(uint8_t *)g->bin[d].map + ki * g->bin_stride,
						g->bin_stride);
			} else {
				charsiu_pack_input_f16(&mm, g->bscr,
						(uint8_t *)g->bin[d].map + ki * g->bin_stride,
						g->bin_stride);
			}
		}

		double tpe = now_us();

		for (unsigned i = 0; i < e->count; i++) {
			const struct npu_slot *s = &g->slot[e->first + i];
			unsigned ki = i / e->n_slices;
			struct charsiu_job job = s->job;
			size_t nreg;

			if (s->di != d)
				continue;
			job.mm.m = m;
			job.input_addr = (uint32_t)g->bin[d].dma_address
				       + (uint32_t)(ki * g->bin_stride);
			job.output_addr = (uint32_t)ob->bo[d].dma_address
					+ (uint32_t)(nt * g->bout_stride);
			nreg = charsiu_emit_job(&job,
					(uint64_t *)((uint8_t *)g->breg[d].map + (size_t)nt * 4096),
					4096 / 8);
			if (!nreg) {
				whine(g, "the batched register stream came back empty",
				      (unsigned)job.mm.k, (unsigned)job.mm.n);
				charsiu_bo_fini(g->dev[d], &g->bin[d]);
				charsiu_bo_fini(g->dev[d], &g->breg[d]);
				return -1;
			}
			g->tasks[nt].regcmd = (uint32_t)g->breg[d].dma_address
					    + (uint32_t)(nt * 4096);
			g->tasks[nt].regcmd_count = (unsigned)nreg;
			g->handles[nh++] = s->wt.handle;
			g->handles[nh++] = s->coef.handle;
			nt++;
			g->weight_mb += (double)charsiu_weight_bytes(&s->job.mm) / 1e6;
		}
		g->bpack_emit_us += now_us() - tpe;
		tpe = now_us();
		if (!reuse)
			charsiu_bo_fini(g->dev[d], &g->bin[d]);
		charsiu_bo_fini(g->dev[d], &g->breg[d]);
		g->bpack_fini_us += now_us() - tpe;
		g->bpack_us += now_us() - tp;
		/*
		 * THIS device's BO now holds X -- but only if a slice was
		 * actually packed into it. A tensor with no slot here wrote
		 * nothing, and the key must not claim otherwise.
		 */
		if (!reuse && done_ki)
			reuse_key_set(&g->bin_key[d], X, m, e->t->k, zp, done_ki);
		if (!nt)
			continue;
		tp = now_us();
		jl.tasks = g->tasks;
		jl.task_count = nt;
		jl.in_handles = g->handles;
		jl.in_count = nh;
		jl.out_handles = &ob->bo[d].handle;
		jl.out_count = 1;
		if (charsiu_submit_jobs(g->dev[d], &jl, 1)) {
			g->strikes = 3;
			g->dead = 1;
			return -1;
		}
		g->submits++;
		/* ⚠ from here the hardware owns this buffer, and batch_outbuf is
		 * the one that has to know it if this call returns before the
		 * fence below clears it again */
		ob->busy |= 1u << d;
		g->bsub_us += now_us() - tp;
		/*
		 * ⚠⚠ THE CORE PAIR, AND THE ONE KNOB THAT REMOVES IT.
		 *
		 * Batched w4a16 is bit exact on ONE core and wrong on two, and
		 * the board has now said so three ways: m = 8 is 62 of 64 on two
		 * cores and 64 of 64 at worst 0.00e+00 on one; m = 10 is 79 of
		 * 80 against 80 of 80; and phi3 at width 24 is wrong 26 runs of
		 * 32 across two two-core arms and 0 of 16 on one core. The 33
		 * misses of the uncapped m = 8 pass are all ROW 0 of the n =
		 * 8192 tensors, and one core has none of them.
		 *
		 * The two cores share the CBUF and take different windows
		 * (charsiu_job.cbuf_window = di), so the collision is not the
		 * window index. What it IS has not been found, and the shape of
		 * the evidence says it does not have to be: the loop above puts
		 * both devices in flight at once, which is right for decode --
		 * m = 1 has never shown this -- and is the only thing a batched
		 * submit does that a single one does not.
		 *
		 * Waiting on each device before submitting the next means the
		 * two never run together. That is ON by default now: the board
		 * round that priced it found the overlapped default wrong 13 of
		 * 16 and the serialised one 0 of 16, on the same freshly booted
		 * card, in the same minute. CHARSIU_NPU_BATCH_PARALLEL=1 puts
		 * the overlap back, and returns wrong text when it does.
		 */
		if (batch_serial_for(m) && g->ndev > 1) {
			double tf = now_us();

			charsiu_bo_prep(g->dev[d], &ob->bo[d], 2000000000);
			ob->busy &= ~(1u << d);
			g->bfence_us += now_us() - tf;
		}
	}

	/* ⚠ both submitted, then both waited on: that is the point */
	for (unsigned d = 0; d < g->ndev; d++) {
		double tf = now_us();
		unsigned nt;

		charsiu_bo_prep(g->dev[d], &ob->bo[d], 2000000000);
		ob->busy &= ~(1u << d);
		g->bfence_us += now_us() - tf;
		tf = now_us();
		{
			/* ⚠ THE KEY IS m ALONE and that is still right: the
			 * format and the axis are fixed for the life of a
			 * pool, so only the width can change under it. */
			/*
			 * ⚠⚠ ONE ENTRY PER FOUR CHANNELS, because the read
			 * order is FOUR CONSECUTIVE SLOTS and always has been.
			 *
			 * charsiu_acc_index(r, j+q) == charsiu_acc_index(r, j)
			 * + q for q of 1, 2 and 3 at every j that is a
			 * multiple of four: checked at 1503680 groups over
			 * both formats, m of 2 to 80 and n of 64 to 8192, with
			 * none broken. The `t % 4` term is the only one that
			 * moves inside a group of four and it moves by one.
			 *
			 * The table was a uint32 per output channel, so the
			 * gather below read FOUR BYTES OF INDEX FOR EVERY FOUR
			 * BYTES OF DATA -- half its memory traffic was the
			 * table. At m = 32 that gather is 368 ms of a 702 ms
			 * batched matmul and at m = 80 it is 933 of 1746: the
			 * dominant cost of a batched projection now that the
			 * hardware part is right.
			 */
			if (g->bmap_m != m) {
				unsigned n4 = (g->nmax + 3) / 4;
				uint32_t *t2 = realloc(g->bmap,
					(size_t)m * n4 * sizeof(*t2));

				if (!t2) {
					whine(g, "the read order table would not allocate",
					      m, g->nmax);
					return -1;
				}
				g->bmap = t2;
				g->bmap_n4 = n4;
				for (unsigned r = 0; r < m; r++)
					for (unsigned j = 0; j < n4; j++)
						g->bmap[(size_t)r * n4 + j] =
						  (uint32_t)charsiu_acc_index(r, j * 4, m,
							g->w4 && charsiu_m_axis_wide_for(1));
				g->bmap_m = m;
				/* read_rows2's premise: rows 2h, 2h+1 at index, +4 */
				g->bmap2 = m % 2 == 0;
				for (unsigned r = 0; g->bmap2 && r < m; r += 2)
					for (unsigned j = 0; g->bmap2 && j < n4; j++)
						if (g->bmap[(size_t)(r + 1) * n4 + j] !=
						    g->bmap[(size_t)r * n4 + j] + 4) {
							g->bmap2 = 0;
							break;
						}
				/* read_rows4's premise, read off the table just built */
				g->bmap4 = m % 4 == 0;
				for (unsigned r = 0; g->bmap4 && r < m; r += 4)
					for (unsigned j = 0; g->bmap4 && j < n4; j++)
						for (unsigned lo = 1; lo < 4; lo++)
							if (g->bmap[(size_t)(r + lo) * n4 + j] !=
							    g->bmap[(size_t)r * n4 + j] + lo * 4) {
								g->bmap4 = 0;
								break;
							}
			}
			/*
			 * ⚠⚠ NOT ON THE POOL. THIS HAS BEEN TRIED TWICE AND
			 * LOST TWICE.
			 *
			 * Round one: 283 ms became 463, and I blamed the table
			 * being rebuilt inside every dispatch. Round two, with
			 * the table built once and nothing in the worker but
			 * the gather: 241 ms became 430. Same 190 ms either
			 * time, which is what says it is the pool and not the
			 * work -- 226 dispatches a pass, one per tensor per
			 * device, at about 0.84 ms of barrier each.
			 *
			 * The work per dispatch is one tensor's rows, and there
			 * is not enough of it to pay for a wakeup. Parallelism
			 * here would have to be at a coarser grain than a
			 * tensor, which means the caller's loop rather than
			 * this one.
			 */
			nt = 0;
			for (unsigned i = 0; i < e->count; i++) {
				const struct npu_slot *s = &g->slot[e->first + i];
				unsigned sn = s->job.mm.n, ki = i / e->n_slices;
				const float *fo;
				const int32_t *io;
				int grp = tensor_grouped(g, e->t);

				if (s->di != d)
					continue;
				fo = (const float *)((uint8_t *)ob->bo[d].map
						     + (size_t)nt * g->bout_stride);
				io = (const int32_t *)fo;
				/*
				 * ⚠ THE TABLE IS BUILT AT `wide` AND A SLICE
				 * CAN BE NARROWER -- the head's last one is
				 * 5376 against 8192. charsiu_acc_index does not
				 * depend on n at all, so the table is valid for
				 * any narrower slice, but only if the entries
				 * beyond its width are skipped rather than the
				 * stride being changed. Indexing it at sn
				 * instead cost exactly one tensor: 3585 rows of
				 * 3616.
				 */
				{
					unsigned ni = s->n0 / g->nmax;
					struct read_rows rr = {
						g, e, s, fo, io, Y, m, sn, ki,
						grp, !g->bseen[ni]
					};

					g->bread_passes++;
					if (!((g->bseen_dev >> (d * 16)) & (1u << ni)))
						g->bread_ranges++;
					g->bseen_dev |= (uint64_t)1 << (d * 16 + ni);
					if (g->poolread == 1 ||
					    (g->poolread == 2 &&
					     (size_t)m * sn >= g->poolread_min)) {
						charsiu_parallel_for(read_rows, &rr, m);
						g->bread_pooled++;
					} else {
						read_rows(&rr, 0, m);
						g->bread_serial++;
					}
				}
				/* ⚠ AFTER the row loop: every row of this slot
				 * shares the flag, and setting it inside would
				 * make row 0 assign and rows 1.. accumulate onto
				 * whatever was in the caller's buffer. */
				g->bseen[s->n0 / g->nmax] = 1;
				nt++;
			}
		}
		g->bread_us += now_us() - tf;
		if (!g->nofini)
			charsiu_bo_fini(g->dev[d], &ob->bo[d]);
	}

	g->busy_us += now_us() - t0;

	/* an ungrouped tensor is scaled once, per channel, at the end; int8
	 * always is, because its d1 went in above */
	if (!g->w4 || !tensor_grouped(g, e->t))
		for (unsigned r = 0; r < m; r++)
			for (unsigned j = 0; j < (unsigned)e->t->n; j++)
				Y[(size_t)r * e->t->n + j] *= e->t->scale[j];
	return 0;
}

int charsiu_npu_matmul(struct charsiu_npu *g, int id, const float *X,
		       unsigned m, float *Y)
{
	double t0 = now_us();
	int rc;

	g->reuse_ask = 0;
	rc = npu_matmul_inner(g, id, X, m, Y);
	g->bwall_us += now_us() - t0;
	return rc;
}

/*
 * ⚠⚠ ON, AND IT WAS OFF FOR A DAY BECAUSE OF THE BOARD. Input reuse shipped
 * twice in one day and phase 2 stopped the round both times: first 6 of 9
 * models, with one key for two devices; then, with a key per device, still
 * Phi-3.5 and gemma4-E2B. The host cannot see either fault (no NPU: the
 * batched path falls back to the row loop), and two rounds of guessing was
 * the budget before this went back to being a probe.
 *
 * Phase 22 then bisected it by site: Phi-3.5 broke on k after q, gemma4 on
 * up after gate, Qwen3 on nothing. The second fault was a key with no
 * expiry -- a leader packing one core left the other core's key naming the
 * same buffer with the old contents (reusekey.h). "Views of a fused tensor"
 * and "per-layer embeddings", the two guesses, were wrong. With the leader
 * drop phase 22 read identical on all 15 cells (2026-09-02), and the drop
 * can only turn a hit into a miss, so nothing that was right before it can
 * be wrong after it. CHARSIU_NPU_REUSE=0 turns it off; phase 22 is the
 * bisect if phase 2 ever stops on it again.
 */
static int reuse_enabled(void)
{
	static int v = -1;

	if (v < 0) {
		const char *e = getenv("CHARSIU_NPU_REUSE");

		v = !(e && *e == '0');
	}
	return v;
}

int charsiu_npu_matmul_same(struct charsiu_npu *g, int id, const float *X,
			    unsigned m, float *Y)
{
	double t0 = now_us();
	int rc;

	g->reuse_ask = reuse_enabled();
	rc = npu_matmul_inner(g, id, X, m, Y);
	g->reuse_ask = 0;
	g->bwall_us += now_us() - t0;
	return rc;
}

int charsiu_npu_reuse_on(void)
{
	return reuse_enabled();
}

/* the widest K one dispatch carries: a tensor wider than this is sliced */
unsigned charsiu_npu_kmax(const struct charsiu_npu *g)
{
	return g ? g->kmax : 4096;
}

/*
 * Slot i of tensor id, as the deal left it: the device it ran on, its K slice
 * index, and the output channels [n0, n1) it wrote. Slots are n fastest, so
 * the first n_slices of them are K slice 0. -1 past the last slot.
 *
 * ⚠ THIS IS WHAT TURNS A WRONG ROW INTO A CORE. The overlap fault (both
 * cores in flight, width 24, phi3) was mapped for four days by TEXT -- right
 * or wrong, 16 runs a width -- which can say a width is bad and nothing about
 * where. A miss in the batch probe knows its channels; with this it knows
 * which slot covered them and which core that slot was dealt to.
 */
int charsiu_npu_slot_deal(const struct charsiu_npu *g, int id, unsigned i,
			  unsigned *di, unsigned *ki, unsigned *n0, unsigned *n1)
{
	const struct npu_entry *e;
	const struct npu_slot *s;

	if (!g || id < 0 || (unsigned)id >= g->n_ent)
		return -1;
	e = &g->ent[id];
	if (i >= e->count)
		return -1;
	s = &g->slot[e->first + i];
	*di = s->di;
	*ki = i / e->n_slices;
	*n0 = s->n0;
	*n1 = s->n0 + s->job.mm.n;
	return 0;
}

/*
 * The word slot i of tensor id wrote for (r, c) in the LAST batched call, and
 * what the gather added for it: raw is the bit pattern (a float on the int4
 * path, an int32 on int8), contrib is that word scaled the way read_rows
 * scales it, and final is the per channel scale applied after the sum (1.0
 * when the group scale already did it). Sum contrib over the slots covering
 * c, times final, and that is Y[r][c] as the gather computed it.
 *
 * ⚠ fresh = 1 INVALIDATES THE CPU'S CACHE OF THE BUFFER FIRST -- a PREP on a
 * job that has already finished is just the dma_sync -- so a word that changes
 * between a stale read and a fresh one was a line the CPU held, not a number
 * the hardware wrote. That is the one question the overlap fault has left:
 * (row 16, channel 3) is wrong one time in fifty at width 24 with both cores
 * in flight, and text, rows and even the slot cannot say which side of the
 * bus the wrong word came from. -1 when the last call was not this tensor,
 * or (r, c) is not in slot i.
 */
int charsiu_npu_slot_word(struct charsiu_npu *g, int id, unsigned i, unsigned r,
			  unsigned c, int fresh, uint32_t *raw, float *contrib,
			  float *final)
{
	const struct npu_entry *e;
	const struct npu_slot *s;
	const uint8_t *base;
	unsigned nt = 0, d, cc, idx, ki;

	if (!g || id < 0 || (unsigned)id >= g->n_ent || !g->last_ob ||
	    g->last_id != id || !g->bmap)
		return -1;
	e = &g->ent[id];
	if (i >= e->count || r >= g->bmap_m)
		return -1;
	s = &g->slot[e->first + i];
	if (c < s->n0 || c >= s->n0 + s->job.mm.n)
		return -1;
	d = s->di;
	ki = i / e->n_slices;
	/* the region index is this slot's rank among the device's slots */
	for (unsigned j = 0; j < i; j++)
		if (g->slot[e->first + j].di == d)
			nt++;
	if (fresh)
		charsiu_bo_prep(g->dev[d], &g->last_ob->bo[d], 2000000000);
	base = (const uint8_t *)g->last_ob->bo[d].map + (size_t)nt * g->bout_stride;
	cc = c - s->n0;
	idx = g->bmap[(size_t)r * g->bmap_n4 + cc / 4] + cc % 4;
	*raw = ((const uint32_t *)base)[idx];
	if (g->w4) {
		float v = ((const float *)base)[idx];
		int grp = tensor_grouped(g, e->t);

		*contrib = grp ? v * s->sc[cc] : v;
		*final = grp ? 1.0f : e->t->scale[c];
	} else {
		*contrib = (float)((const int32_t *)base)[idx]
			 * g->bd1[(size_t)ki * g->bmap_m + r];
		*final = e->t->scale[c];
	}
	return 0;
}

void charsiu_npu_reuse_stats(const struct charsiu_npu *g, unsigned long *hits,
			     unsigned long *misses, unsigned long why[4])
{
	*hits = g ? g->reuse_hits : 0;
	*misses = g ? g->reuse_misses : 0;
	for (unsigned i = 0; i < 4; i++)
		why[i] = g ? g->reuse_why[i] : 0;
}

/*
 * SEVERAL PROJECTIONS, ONE SUBMIT AND ONE FENCE.
 *
 * q, k and v all multiply the SAME RMSNorm output, and so do gate and up. They
 * are independent of each other, so there is no reason to wait for one before
 * starting the next -- and round 321 measured the fence at 94% of the hardware
 * path, so a fence removed is worth more than a submit removed.
 *
 * 113 fences a token becomes 65. Whether that is worth anything is what the
 * round measures; the arithmetic is unchanged either way, so the tokens must
 * stay identical.
 */
int charsiu_npu_matvec_group(struct charsiu_npu *g, const int *ids, unsigned n,
			     const struct charsiu_act *a, float **ys)
{
	struct npu_entry *e0;
	struct charsiu_joblist jl;
	uint32_t outh[8];
	unsigned nh = 0, ntask = 0, i, j;
	double t0, t1, tpack, fspent = 0.0, tcall = now_us();

	if (g->dead || !n || n > 8)
		return -1;
	for (i = 0; i < n; i++)
		if (ids[i] < 0 || (unsigned)ids[i] >= g->n_ent)
			return -1;
	charsiu_note("a group: checking the entries", (unsigned long)n,
		     (unsigned long)a->n);
	e0 = &g->ent[ids[0]];
	for (i = 1; i < n; i++)
		if (g->ent[ids[i]].t->k != e0->t->k)
			return -1;       /* a group shares one activation */
	if ((unsigned)a->n != e0->t->k)
		return -1;

	/*
	 * The activation, once for every K slice, INTO EVERY DEVICE THE GROUP
	 * USES. A group shares one input vector but its entries may sit on
	 * different devices, and a buffer object belongs to the file that
	 * created it. Packing it twice is a few kilobytes against the megabytes
	 * of weights each submit fetches.
	 */
	{
	unsigned nd = 0;

	tpack = now_us();
	for (unsigned d = 0; d < g->ndev; d++) {
		nd++;
	charsiu_note("a group: packing the activation", (unsigned long)d,
		     (unsigned long)e0->k_slices);
	if (g->inprep)
		charsiu_bo_prep(g->dev[d], &g->in[d], 1000000000);
	for (unsigned ki = 0; ki < e0->k_slices; ki++) {
		const struct npu_slot *s = &g->slot[e0->first + ki * e0->n_slices];

		if (g->w4) {
			const float *src = a->f + s->k0;

			/*
			 * STRAIGHT FROM THE ACTIVATION, NO COPY. The packer
			 * reads src[kk] at m = 1, so the scratch buffer was
			 * 463 thousand float copies a token to hand it bytes
			 * it could already see. The running total is the
			 * midrise grid's half step, and midrise is off by
			 * default and measured worse when it was on, so it no
			 * longer costs a double add per element either.
			 */
			if (g->midrise || g->plain) {
				double as = 0.0;

				for (i = 0; i < s->job.mm.k; i++) {
					g->fscr[i] = src[i];
					as += (double)g->fscr[i];
				}
				g->asum[ki] = as;
				src = g->fscr;
			}
			charsiu_pack_input_f16(&s->job.mm, src,
					       (uint8_t *)g->in[d].map
					       + ki * g->in_stride,
					       g->in_stride);
		} else {
			for (i = 0; i < s->job.mm.k; i++)
				g->scratch[i] = (uint8_t)((int)a->q1[s->k0 + i]
							  + 128);
			charsiu_pack_input(&s->job.mm, g->scratch,
					   (uint8_t *)g->in[d].map
					   + ki * g->in_stride,
					   g->in_stride,
					   s->job.input_zero_point);
		}
	}
	charsiu_bo_fini(g->dev[d], &g->in[d]);
	}
	g->pack_us += now_us() - tpack;
	(void)nd;
	}

	/*
	 * ONE JOBLIST PER DEVICE, BOTH SUBMITTED BEFORE EITHER IS WAITED ON.
	 *
	 * This is the whole point of the second file. Submitting is a queueing
	 * ioctl and the fence is waited separately, so issuing device 0's work
	 * and then device 1's leaves both cores running at once without a
	 * thread anywhere.
	 *
	 * Round 356 put 89 ms of a 117 ms token inside the fence, so this is
	 * the only place in the decode where a second core can be worth
	 * anything.
	 */
	t0 = now_us();
	for (unsigned d = 0; d < g->ndev; d++) {
		unsigned no = 0;

		charsiu_note("a group: building the joblist", (unsigned long)d,
			     (unsigned long)n);
		nh = 0;
		ntask = 0;
		g->handles[nh++] = g->in[d].handle;
		for (i = 0; i < n; i++) {
			struct npu_entry *e = &g->ent[ids[i]];
			unsigned any = 0;

			for (j = 0; j < e->count; j++) {
				const struct npu_slot *s =
					&g->slot[e->first + j];

				/* ⚠ filter by the SLICE's device, not the
				 * entry's: since round 365 a tensor's slices
				 * are spread across both. */
				if (s->di != d)
					continue;
				if (ntask >= 4 * g->max_slices)
					return -1;
				g->tasks[ntask].regcmd =
					(uint32_t)s->regcmd.dma_address;
				g->tasks[ntask].regcmd_count = s->nreg;
				ntask++;
				g->handles[nh++] = s->wt.handle;
				g->handles[nh++] = s->coef.handle;
				any = 1;
			}
			if (any)
				outh[no++] = e->out[d].handle;
		}
		if (!no)
			continue;
		jl.tasks = g->tasks;
		jl.task_count = ntask;
		jl.in_handles = g->handles;
		jl.in_count = nh;
		jl.out_handles = outh;
		jl.out_count = no;
		charsiu_note("a group: submitting", (unsigned long)ntask,
			     (unsigned long)nh);
		if (charsiu_submit_jobs(g->dev[d], &jl, 1)) {
			g->strikes = 3;
			g->dead = 1;
			fprintf(stderr, "charsiu: a %u task group submit "
				"failed on device %u\n", ntask, d);
			return -1;
		}
		g->submits++;
	}
	g->submit_us += now_us() - t0;

	/*
	 * The group's CPU rows, in the same window. A group shares one
	 * activation, so it is rounded through fp16 once for all of them.
	 */
	{
		unsigned any = 0;

		for (i = 0; i < n; i++)
			if (g->ent[ids[i]].cq)
				any = 1;
		if (any) {
			double tc = now_us();

			charsiu_note("a group: the CPU's own rows",
				     (unsigned long)n, (unsigned long)e0->t->k);
			for (uint64_t q = 0; q < e0->t->k; q++)
				g->afscr[q] = charsiu_half_to_float(
					charsiu_float_to_half(a->f[q]));
			for (i = 0; i < n; i++) {
				struct npu_entry *e = &g->ent[ids[i]];

				if (e->cq && e->n_npu < (unsigned)e->t->n)
					cpu_rows(e, g->afscr, ys[i]);
			}
			g->cpu_us += now_us() - tc;
		}
	}

	t1 = now_us();
	for (i = 0; i < n; i++)
		for (unsigned d = 0; d < g->ndev; d++) {
			charsiu_note("a group: waiting on the fence",
				     (unsigned long)i, (unsigned long)d);
			charsiu_bo_prep(g->dev[d], &g->ent[ids[i]].out[d],
					2000000000);
		}
	g->fence_us += now_us() - t1;

	t1 = now_us();
	for (i = 0; i < n; i++) {
		struct npu_entry *e = &g->ent[ids[i]];
		const int32_t *out;

		int grp = tensor_grouped(g, e->t);
		float *af = (g->w4 && grp && !g->plain) ? ys[i] : g->accf;

		/*
		 * ONE OF THESE, NOT BOTH. int4 sums into accf and int8 into
		 * acc, and clearing the other one is 2 MB a token of writes
		 * for an array nothing will read: the 128256 wide head alone
		 * is half a megabyte of it.
		 *
		 * AND WHEN THE LAST STEP WOULD BE A COPY, SUM WHERE THE ANSWER
		 * GOES. A grouped int4 tensor applies its scale per slice on
		 * the way in, so the conversion below is ys[i][q] = accf[q]
		 * and nothing else.
		 */
		/* ⚠ the hardware's rows only: the CPU's are already written */
		charsiu_note("a group: clearing the accumulator",
			     (unsigned long)e->t->n, (unsigned long)e->n_npu);
		if (g->w4)
			memset(af, 0, (size_t)e->n_npu * sizeof(*af));
		else
			memset(g->acc, 0, (size_t)e->t->n * sizeof(*g->acc));
		for (j = 0; j < e->count; j++) {
			charsiu_note("a group: reading a slice back",
				     (unsigned long)i, (unsigned long)e->count);
			const struct npu_slot *s = &g->slot[e->first + j];
			const uint8_t *base;

			/*
			 * ⚠ THE TWO NUMBERS THAT CAN MAKE THE NEXT LINE A NULL
			 * DEREFERENCE. e->out is an array of ndev buffers, so
			 * a slot whose di is not a device index reads past it
			 * and takes whatever .map happens to be there. Naming
			 * them here costs one store and turns a segfault into
			 * a sentence.
			 */
			charsiu_note("a group: a slice's device and out slot",
				     (unsigned long)s->di,
				     (unsigned long)s->out_slot);
			if (s->di >= g->ndev || !e->out[s->di].map) {
				fprintf(stderr, "charsiu: slice %u of tensor "
					"%u has device %u of %u and map %p\n",
					j, i, s->di, g->ndev,
					s->di < g->ndev ? e->out[s->di].map
							: NULL);
				g->dead = 1;
				return -1;
			}
			base = (const uint8_t *)e->out[s->di].map +
			       s->out_slot * g->out_stride;

			/*
			 * ⚠ THE SLICE'S OWN WIDTH AND OFFSET, which decide how
			 * far the two loops below walk. The guard above proved
			 * the base pointer is a real mapping; a garbage n or n0
			 * walks off the end of it, or off the accumulator, and
			 * looks exactly the same from outside.
			 */
			charsiu_note("a group: a slice's width and offset",
				     (unsigned long)s->job.mm.n,
				     (unsigned long)s->n0);
			if (s->n0 + s->job.mm.n > g->max_n ||
			    s->job.mm.n > g->nmax ||
			    (s->out_slot + 1) * (size_t)g->out_stride
				    > e->out[s->di].size) {
				fprintf(stderr, "charsiu: slice %u of tensor "
					"%u wants n0 %u + n %u of %u, slot %u "
					"of a %zu byte buffer\n",
					j, i, s->n0, s->job.mm.n, g->max_n,
					s->out_slot, (size_t)e->out[s->di].size);
				g->dead = 1;
				return -1;
			}

			if (g->w4) {
				const float *fo = (const float *)base;

				if (grp) {
					double hs = g->midrise
						  ? 0.5 * g->asum[s->k0 / g->kmax]
						  : 0.0;
					const float *sc = s->sc;

					/* non null whenever grp is: the two
					 * ask tensor_grouped the same
					 * question, and a gather that will not
					 * allocate fails the staging */
					if (hs == 0.0 && !g->plain) {
						scaled_add(af + s->n0, fo, sc,
							   s->job.mm.n);
						continue;
					}
					for (unsigned q = 0; q < s->job.mm.n; q++)
						af[s->n0 + q] +=
						  (float)((fo[q] + hs) * sc[q]);
				} else {
					for (unsigned q = 0; q < s->job.mm.n; q++)
						af[s->n0 + q] += fo[q];
				}
				continue;
			}
			charsiu_note("a group: summing an int8 slice",
				     (unsigned long)s->job.mm.n,
				     (unsigned long)s->n0);
			out = (const int32_t *)base;
			for (unsigned q = 0; q < s->job.mm.n; q++)
				g->acc[s->n0 + q] += out[q];
		}
		{
			double tf = now_us();

			if (!g->nofini)
				for (unsigned d = 0; d < g->ndev; d++)
					charsiu_bo_fini(g->dev[d], &e->out[d]);
			fspent += now_us() - tf;
		}
		if (af != ys[i]) {
			double hsu = 0.0;

			charsiu_note("a group: converting to the caller's rows",
				     (unsigned long)e->t->n,
				     (unsigned long)(uintptr_t)e->t->scale);

			if (g->midrise && !grp)
				for (unsigned ki = 0; ki < e->k_slices; ki++)
					hsu += 0.5 * g->asum[ki];
			/*
			 * ⚠ grp STILL HAS TO BE ASKED HERE. This branch is
			 * reached with CHARSIU_NPU_PLAIN, and a grouped tensor
			 * has already had a scale applied per slice on the way
			 * in: multiplying by the row scale as well would give
			 * the control a wrong answer, which is worse than
			 * having no control.
			 */
			for (unsigned q = 0; q < (unsigned)e->t->n; q++)
				ys[i][q] = g->w4
					 ? (grp ? g->accf[q]
					        : (float)(((double)g->accf[q]
						   + hsu) * e->t->scale[q]))
					 : (float)g->acc[q] * a->d1
					   * e->t->scale[q];
		}
		g->weight_mb += e->weight_mb;
	}
	g->copy_us += now_us() - t1 - fspent;
	g->fini_us += fspent;
	{
		/*
		 * ⚠ THE GROUP IS ONE CALL, and that is the whole reason it
		 * exists: q, k and v share a submit and a fence, so charging
		 * the fit three calls' fixed cost for one fence would say
		 * grouping bought nothing. account_call sums the three entries
		 * per device and takes the busier one, which is exactly what
		 * the clock below measured.
		 */
		double took = now_us() - t0;

		g->busy_us += took;
		account_call(g, ids, n, took);
	}
	g->call_us += now_us() - tcall;
	/*
	 * ⚠⚠ CLEAR THE BREADCRUMB ON THE WAY OUT, or it outlives the function
	 * and every crash in the CALLER is reported against the last thing
	 * this did. Round 385 spent a board run on "a slice's width and offset
	 * (512, 0)" where both numbers were legal and every bound around them
	 * held -- which is exactly what a stale note looks like.
	 */
	charsiu_note("something outside the NPU code", 0, 0);
	return 0;
}
