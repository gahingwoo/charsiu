// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * An fp16 matmul whose WEIGHT belongs to the caller.
 *
 * ⚠⚠ WHY THIS IS NOT IN npudev.c. Everything there is built around a weight
 * tensor that is STAGED once -- charsiu_npu_add, then matmul by id -- because
 * a model's weights do not change. Attention's second operand is the KV cache:
 * a different buffer for every layer and every head, and one that grows by a
 * row per token. Threading that through the staging machinery would put a
 * growing, per-head buffer into the path that decode and prefill already run
 * on, for no benefit to either. This is its own unit and the int4 path cannot
 * see it.
 *
 * What the board established on 2026-09-05, and what this encodes:
 *
 *   - the weight layout is GROUP: ngroup 16, kgroup 32, two byte elements.
 *     charsiu_fp16_woffset() is that layout, so the caller can write the cache
 *     STRAIGHT INTO IT as tokens are appended and never pay a pack.
 *   - fp16 needs the w4a16 output stage, CORE 0x3018's 0x200 form, and
 *     DPU 0x40b8 = (oc/4 + 3) - (M*oc)/4. All of that lives in job.c and is
 *     selected by wdtype == CHARSIU_FP16; nothing here repeats it.
 *   - the job cost is nearly flat in m: 0.366 ms at m=80 and 0.480 at m=178,
 *     governor pinned, five alternating rounds, spreads 0.339-0.377 and
 *     0.459-0.512. So the caller should batch as wide as it can.
 *
 * ⚠ THE PACK IS THE WHOLE GAME. Packing a K=1024 N=64 weight costs 1.62 ms,
 * four times the job. If a caller packs per dispatch there is nothing here
 * worth having; charsiu_fp16_woffset() exists so it does not have to.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "charsiu.h"
#include "charsiu_llm.h"

struct charsiu_fp16 {
	struct charsiu_device *dev;
	struct charsiu_bo wt, in, ob, coef, reg;
	size_t wsz, insz, obsz, coefsz;
	unsigned long calls, refused;
};

/* the weight buffer a (k, n) fp16 matmul needs, in bytes */
size_t charsiu_fp16_wbytes(unsigned k, unsigned n)
{
	struct charsiu_matmul mm = { 1, k, n, CHARSIU_FP16, CHARSIU_FP16 };

	return charsiu_weight_bytes(&mm);
}

/* where element (n, k) of the weight goes, in bytes. SIZE_MAX if out of range */
size_t charsiu_fp16_woffset(unsigned k, unsigned n, unsigned ni, unsigned ki)
{
	struct charsiu_matmul mm = { 1, k, n, CHARSIU_FP16, CHARSIU_FP16 };

	return charsiu_w16_offset(&mm, ni, ki, CHARSIU_W16_GROUP);
}

static int want(struct charsiu_fp16 *f, size_t wsz, size_t insz, size_t obsz,
		size_t coefsz)
{
	if (f->wsz >= wsz && f->insz >= insz && f->obsz >= obsz &&
	    f->coefsz >= coefsz)
		return 0;
	charsiu_bo_free(f->dev, &f->reg);  charsiu_bo_free(f->dev, &f->coef);
	charsiu_bo_free(f->dev, &f->ob);   charsiu_bo_free(f->dev, &f->in);
	charsiu_bo_free(f->dev, &f->wt);
	memset(&f->wt, 0, sizeof(f->wt));   memset(&f->in, 0, sizeof(f->in));
	memset(&f->ob, 0, sizeof(f->ob));   memset(&f->coef, 0, sizeof(f->coef));
	memset(&f->reg, 0, sizeof(f->reg));
	f->wsz = wsz; f->insz = insz; f->obsz = obsz; f->coefsz = coefsz;
	if (charsiu_bo_alloc(f->dev, wsz, &f->wt) ||
	    charsiu_bo_alloc(f->dev, insz, &f->in) ||
	    charsiu_bo_alloc(f->dev, obsz, &f->ob) ||
	    charsiu_bo_alloc(f->dev, coefsz, &f->coef) ||
	    charsiu_bo_alloc(f->dev, 4096, &f->reg))
		return -1;
	return (f->wt.map && f->in.map && f->ob.map && f->coef.map &&
		f->reg.map) ? 0 : -1;
}

struct charsiu_fp16 *charsiu_fp16_open(void)
{
	struct charsiu_fp16 *f = calloc(1, sizeof(*f));

	if (!f)
		return NULL;
	f->dev = charsiu_open(NULL);
	if (!f->dev) {
		free(f);
		return NULL;
	}
	return f;
}

void charsiu_fp16_close(struct charsiu_fp16 *f)
{
	if (!f)
		return;
	charsiu_bo_free(f->dev, &f->reg);  charsiu_bo_free(f->dev, &f->coef);
	charsiu_bo_free(f->dev, &f->ob);   charsiu_bo_free(f->dev, &f->in);
	charsiu_bo_free(f->dev, &f->wt);
	charsiu_close(f->dev);
	free(f);
}

/*
 * X is m by k, row major, float. W holds charsiu_fp16_wbytes(k, n) bytes in
 * the layout charsiu_fp16_woffset describes. Y is m by n, row major, float.
 *
 * ⚠ IT REFUSES RATHER THAN COMPUTES WHAT IT HAS NOT BEEN SHOWN. K=16 N=8
 * wedged the NPU for two jobs and then timed out both cores, and the shapes
 * that survive a loop cleanly are the ones with a real n. Anything under the
 * two byte feature atom on either axis is not a shape this has ever run.
 */
int charsiu_fp16_matmul(struct charsiu_fp16 *f, const float *X, unsigned m,
			unsigned k, unsigned n, const void *W, float *Y)
{
	struct charsiu_job job = { 0 };
	size_t nreg, insz, wsz;

	if (!f || !X || !W || !Y || !m || k < 32 || n < 32) {
		if (f)
			f->refused++;
		return -1;
	}
	job.cbuf_window = (unsigned)charsiu_cbuf_window();
	job.mm.m = m; job.mm.k = k; job.mm.n = n;
	job.mm.wdtype = CHARSIU_FP16;
	job.mm.adtype = CHARSIU_FP16;
	job.input_scale = 1.0f; job.weight_scale = 1.0f; job.output_scale = 1.0f;
	job.acc_out = 1;                /* the accumulator, which for fp16 is fp32 */

	wsz = charsiu_weight_bytes(&job.mm);
	insz = (size_t)charsiu_entries_per_row(&job.mm) * 64 * m + 4096;
	if (want(f, wsz + 4096, insz, (size_t)m * n * 4 + 4096,
		 charsiu_coef_bytes(&job.mm) + 4096))
		return -1;

	charsiu_bo_prep(f->dev, &f->wt, 1000000000);
	memcpy(f->wt.map, W, wsz);
	charsiu_bo_fini(f->dev, &f->wt);

	/*
	 * ⚠⚠ ROW MAJOR, NOT charsiu_pack_input_f16's INTERLEAVE, AND THE BOARD
	 * SAID SO SLOT BY SLOT.
	 *
	 * charsiu_pack_input_f16 writes [k/8][m][8]: row r's k values are
	 * spread through the buffer at a stride of m*8. npu_fp16_test
	 * --inslots put one 1.0 into each packed input slot in turn and read
	 * which output row answered and with which k. At m=2, K=256:
	 *
	 *     slots   0..255   -> row 0, k = slot
	 *     slots 256..511   -> row 1
	 *
	 * so this path wants row r's k CONTIGUOUS at r*k. Feeding it the
	 * interleaved layout is why every output row came back holding
	 * k in [r*K/m, (r+1)*K/m) -- it reads contiguous runs and gets a
	 * mixture -- and why m=1 was exact: at m=1 the interleave is the
	 * identity.
	 *
	 * Six register-level fixes were tried against that symptom and none
	 * moved it, because the fault was never in the stream.
	 */
	charsiu_bo_prep(f->dev, &f->in, 1000000000);
	{
		uint8_t *d = f->in.map;

		memset(d, 0, insz);
		for (size_t i = 0; i < (size_t)m * k; i++) {
			uint16_t h = charsiu_float_to_half(X[i]);

			if ((i + 1) * 2 > insz)
				break;
			d[i * 2] = (uint8_t)(h & 0xff);
			d[i * 2 + 1] = (uint8_t)(h >> 8);
		}
	}
	charsiu_bo_fini(f->dev, &f->in);

	{
		int32_t *zero = calloc(n, sizeof(int32_t));

		if (!zero)
			return -1;
		charsiu_bo_prep(f->dev, &f->coef, 1000000000);
		charsiu_build_coefs(&job, zero, zero, f->coef.map);
		charsiu_bo_fini(f->dev, &f->coef);
		free(zero);
	}

	job.input_addr = (uint32_t)f->in.dma_address;
	job.output_addr = (uint32_t)f->ob.dma_address;
	job.weight_addr = (uint32_t)f->wt.dma_address;
	job.coef_addr = (uint32_t)f->coef.dma_address;

	charsiu_bo_prep(f->dev, &f->reg, 1000000000);
	nreg = charsiu_emit_job(&job, f->reg.map, 4096 / 8);
	charsiu_bo_fini(f->dev, &f->reg);
	if (!nreg)
		return -1;

	/*
	 * ⚠ A SENTINEL, NOT ZEROS. A job that never wrote and a job that
	 * computed zero are the same four bytes otherwise, and this project
	 * has read the first as the second four times in a week.
	 */
	charsiu_bo_prep(f->dev, &f->ob, 1000000000);
	for (unsigned i = 0; i < m * n; i++)
		((uint32_t *)f->ob.map)[i] = 0xdeadbeefu;
	charsiu_bo_fini(f->dev, &f->ob);
	{
		uint32_t ins[2] = { f->in.handle, f->wt.handle };
		uint32_t outs[1] = { f->ob.handle };

		if (charsiu_submit(f->dev, &f->reg, (unsigned)nreg, ins, 2,
				   outs, 1))
			return -1;
	}
	charsiu_bo_prep(f->dev, &f->ob, 1000000000);   /* the fence wait */
	{
		const uint32_t *o = f->ob.map;
		unsigned untouched = 0;

		for (unsigned i = 0; i < m * n; i++)
			untouched += o[i] == 0xdeadbeefu;
		if (untouched == m * n) {
			charsiu_bo_fini(f->dev, &f->ob);
			f->refused++;
			return -1;              /* the job wrote nothing */
		}
		/* flat, and measured: --outmap reads m by n row major */
		memcpy(Y, o, (size_t)m * n * 4);
	}
	charsiu_bo_fini(f->dev, &f->ob);
	f->calls++;
	return 0;
}

void charsiu_fp16_stats(const struct charsiu_fp16 *f, unsigned long *calls,
			unsigned long *refused)
{
	*calls = f ? f->calls : 0;
	*refused = f ? f->refused : 0;
}
