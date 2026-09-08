#define _POSIX_C_SOURCE 200809L
/*
 * npu_out_fmt -- what the NPU actually writes, read as three formats at once.
 *
 * ⚠⚠ WHY THIS IS THE QUESTION. TTFT is 15.8% behind the vendor on int8 and the
 * prefill row spends 0.94 ms of 4.59 reading accumulators back. That read is
 * `m * n * ceil(K/KMAX) * 4` -- four bytes an element, because npudev reads the
 * output as int32. If the hardware is already writing something narrower, or
 * could be told to, the read halves and the gap to the vendor is 0.67.
 *
 * The tree's own registers say the stage is already the vendor's float one:
 * with acc_out set, job.c forces CHARSIU_WIDE8 = 0x3f, which is
 * 0x4010 = a0000002 (PROC_PRECISION 2, fp16), 0x4044 = 2, 0x4050 = 0x00023333
 * and an identity requant. And npudev then reads the result as int32 and gets
 * the right answer, which those two facts cannot both explain.
 *
 * So: one tiny matmul with a KNOWN product, and the output buffer printed as
 * int32, as fp32 and as pairs of fp16. Whichever reads back as the answer is
 * what the hardware writes, and that decides whether a narrower read exists at
 * all.
 *
 *   A = all ones (int8 1), B = all ones, K wide  ->  every output = K
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charsiu.h"

int main(int argc, char **argv)
{
	unsigned k = argc > 1 ? (unsigned)atoi(argv[1]) : 64;
	unsigned n = argc > 2 ? (unsigned)atoi(argv[2]) : 32;
	struct charsiu_device *dev;
	struct charsiu_job job = { 0 };
	struct charsiu_bo wt, in, coef, reg, ob;
	uint8_t *A, *B;
	size_t nreg, insz;
	unsigned i;

	dev = charsiu_open(NULL);
	if (!dev) { fprintf(stderr, "no accel device\n"); return 1; }

	job.cbuf_window = (unsigned)charsiu_cbuf_window();
	job.mm.m = 1; job.mm.k = k; job.mm.n = n;
	job.mm.wdtype = CHARSIU_INT8;
	job.mm.adtype = CHARSIU_INT8;
	job.input_zero_point = 128;
	job.weight_zero_point = 128;
	job.input_scale = job.weight_scale = job.output_scale = 1.0f;
	job.acc_out = 1;

	insz = (size_t)charsiu_entries_per_row(&job.mm) * 64 + 4096;
	if (charsiu_bo_alloc(dev, charsiu_weight_bytes(&job.mm) + 4096, &wt) ||
	    charsiu_bo_alloc(dev, insz, &in) ||
	    charsiu_bo_alloc(dev, charsiu_coef_bytes(&job.mm) + 4096, &coef) ||
	    charsiu_bo_alloc(dev, 4096, &reg) ||
	    charsiu_bo_alloc(dev, (size_t)n * 8 + 4096, &ob)) {
		fprintf(stderr, "allocation failed\n"); return 1;
	}
	A = malloc(k); B = malloc((size_t)k * n);
	memset(A, 1, k);
	memset(B, 1, (size_t)k * n);
	memset(wt.map, 0, charsiu_weight_bytes(&job.mm));
	charsiu_pack_weights(&job.mm, B, wt.map);
	memset(coef.map, 0, charsiu_coef_bytes(&job.mm));
	charsiu_pack_input(&job.mm, A, in.map, insz, job.input_zero_point);
	memset(ob.map, 0xCD, (size_t)n * 8);
	charsiu_bo_fini(dev, &wt); charsiu_bo_fini(dev, &in);
	charsiu_bo_fini(dev, &coef); charsiu_bo_fini(dev, &ob);

	job.input_addr  = (uint32_t)in.dma_address;
	job.output_addr = (uint32_t)ob.dma_address;
	job.weight_addr = (uint32_t)wt.dma_address;
	job.coef_addr   = (uint32_t)coef.dma_address;

	charsiu_bo_prep(dev, &reg, 1000000000);
	nreg = charsiu_emit_job(&job, reg.map, 4096 / 8);
	charsiu_bo_fini(dev, &reg);
	if (!nreg) { fprintf(stderr, "empty register stream\n"); return 1; }
	{
		uint32_t ins[2] = { in.handle, wt.handle };
		uint32_t outs[1] = { ob.handle };

		if (charsiu_submit(dev, &reg, (unsigned)nreg, ins, 2, outs, 1)) {
			fprintf(stderr, "submit failed\n"); return 1;
		}
	}
	charsiu_bo_prep(dev, &ob, 2000000000);

	printf("A = 1^%u, B = 1, so every output should be %u\n\n", k, k);
	printf("  %4s %12s %14s %10s %10s   %s\n", "i", "as int32",
	       "as fp32", "fp16 lo", "fp16 hi", "raw");
	for (i = 0; i < (n < 8 ? n : 8); i++) {
		const uint8_t *p = (const uint8_t *)ob.map + (size_t)i * 4;
		int32_t i32; float f32; uint16_t h0, h1;

		memcpy(&i32, p, 4); memcpy(&f32, p, 4);
		memcpy(&h0, p, 2); memcpy(&h1, p + 2, 2);
		printf("  %4u %12d %14.4g %10.4g %10.4g   %02x%02x%02x%02x\n",
		       i, i32, (double)f32,
		       (double)charsiu_half_to_float(h0),
		       (double)charsiu_half_to_float(h1),
		       p[0], p[1], p[2], p[3]);
	}
	printf("\n⚠ Whichever column reads %u is the format the hardware writes.\n"
	       "  If it is int32, four bytes an element is what the read costs\n"
	       "  and a narrower one needs the requant, not just a cast.\n", k);
	charsiu_close(dev);
	return 0;
}
