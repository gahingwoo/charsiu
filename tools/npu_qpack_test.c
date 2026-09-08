/*
 * npu_qpack_test -- int8's batched activation quantiser, both arms, byte for
 * byte.
 *
 * npudev.c vectorised and pooled that quantiser because round 146 measured it
 * at 2.66 ms a row against int4's 0.86 -- three times wider, and the whole of
 * why int8's prefill is slower than int4's despite a SMALLER fence and read.
 * A rewrite of a quantiser has to be proved identical, and the thing that can
 * carry the fault is one row, not a model: comparing generated text would pass
 * a version that is wrong on a handful of near-zero channels, which is exactly
 * the bug the note above this code in npudev.c records losing a board round to.
 *
 * So the data here is chosen to break it rather than to look typical: codes
 * sitting exactly halfway between two integers, where round-to-nearest-even is
 * the entire question; values past the +/-127 clamp; an all-zero row, where d1
 * is the degenerate case; near-denormal magnitudes; and every width remainder
 * mod 16 and mod 4, because the vector body strides 16 and the maximum strides
 * 4.
 *
 * Exit 0 and one line if they agree.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__ARM_NEON) || defined(CHARSIU_NO_NEON)
int main(void)
{
	printf("npu_qpack_test: needs NEON, and this build has none -- "
	       "nothing compared\n");
	return 0;
}
#else
#include <arm_neon.h>
static void plain(const float *row, unsigned sk, uint8_t *out, float *d1o)
{
	float mx = 0.0f, d1, id1;
	unsigned kk;

	for (kk = 0; kk < sk; kk++) {
		float v = fabsf(row[kk]);
		if (v > mx) mx = v;
	}
	d1 = mx > 0.0f ? mx / 127.0f : 1.0f;
	id1 = d1 != 0.0f ? 1.0f / d1 : 0.0f;
	*d1o = d1;
	for (kk = 0; kk < sk; kk++) {
		int q = (int)lrintf(row[kk] * id1);
		if (q > 127) q = 127;
		if (q < -127) q = -127;
		out[kk] = (uint8_t)(q + 128);
	}
}

static void neon(const float *row, unsigned sk, uint8_t *out, float *d1o)
{
	float mx = 0.0f, d1, id1;
	unsigned kk = 0;
	{
		float32x4_t m4 = vdupq_n_f32(0.0f);
		for (; kk + 4 <= sk; kk += 4)
			m4 = vmaxq_f32(m4, vabsq_f32(vld1q_f32(row + kk)));
		mx = vmaxvq_f32(m4);
	}
	for (; kk < sk; kk++) {
		float v = fabsf(row[kk]);
		if (v > mx) mx = v;
	}
	d1 = mx > 0.0f ? mx / 127.0f : 1.0f;
	id1 = d1 != 0.0f ? 1.0f / d1 : 0.0f;
	*d1o = d1;
	kk = 0;
	{
		const float32x4_t s4 = vdupq_n_f32(id1);
		const int32x4_t hi = vdupq_n_s32(127);
		const int32x4_t lo = vdupq_n_s32(-127);
		for (; kk + 16 <= sk; kk += 16) {
			int32x4_t a = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row + kk), s4));
			int32x4_t b = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row + kk + 4), s4));
			int32x4_t c = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row + kk + 8), s4));
			int32x4_t e = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row + kk + 12), s4));
			int16x8_t p, q;
			a = vmaxq_s32(vminq_s32(a, hi), lo);
			b = vmaxq_s32(vminq_s32(b, hi), lo);
			c = vmaxq_s32(vminq_s32(c, hi), lo);
			e = vmaxq_s32(vminq_s32(e, hi), lo);
			p = vcombine_s16(vmovn_s32(a), vmovn_s32(b));
			q = vcombine_s16(vmovn_s32(c), vmovn_s32(e));
			vst1q_u8(out + kk,
				 veorq_u8(vreinterpretq_u8_s8(
						  vcombine_s8(vmovn_s16(p), vmovn_s16(q))),
					  vdupq_n_u8(0x80)));
		}
	}
	for (; kk < sk; kk++) {
		int q = (int)lrintf(row[kk] * id1);
		if (q > 127) q = 127;
		if (q < -127) q = -127;
		out[kk] = (uint8_t)(q + 128);
	}
}

int main(void)
{
	unsigned widths[] = { 1, 3, 4, 15, 16, 17, 31, 32, 33, 64, 1023, 1024, 2048, 3072 };
	unsigned nw = sizeof(widths) / sizeof(*widths), w, i, trial;
	unsigned long bad = 0, checked = 0, dbad = 0;
	float *row = malloc(4096 * sizeof(float));
	uint8_t *a = malloc(4096), *b = malloc(4096);
	float da, db;

	srand(1);
	for (w = 0; w < nw; w++) {
		unsigned sk = widths[w];
		for (trial = 0; trial < 4000; trial++) {
			for (i = 0; i < sk; i++) {
				switch (trial % 5) {
				case 0: row[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; break;
				case 1: row[i] = 0.0f; break;
				/* halfway codes: k + 0.5 quanta, where the tie
				 * rule is the whole question */
				case 2: row[i] = ((int)(rand() % 255) - 127 + 0.5f) / 127.0f; break;
				case 3: row[i] = (float)rand() / RAND_MAX * 1e-7f; break;
				default: row[i] = ((float)rand() / RAND_MAX - 0.5f) * 1e6f; break;
				}
			}
			if (trial % 7 == 0 && sk > 2)
				row[rand() % sk] = -row[rand() % sk] * 1000.0f;
			memset(a, 0xAA, 4096); memset(b, 0x55, 4096);
			plain(row, sk, a, &da);
			neon(row, sk, b, &db);
			if (memcmp(a, b, sk)) {
				if (bad < 4) {
					for (i = 0; i < sk; i++)
						if (a[i] != b[i]) {
							printf("  sk=%u trial=%u i=%u  x=%.9g  plain=%u neon=%u\n",
							       sk, trial, i, row[i], a[i], b[i]);
							break;
						}
				}
				bad++;
			}
			if (da != db) dbad++;
			checked++;
		}
	}
	printf("qpack: %lu rows compared, %lu byte mismatches, %lu d1 mismatches\n",
	       checked, bad, dbad);
	return bad || dbad ? 1 : 0;
}
#endif
