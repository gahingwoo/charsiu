/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
 *
 * WHAT THE MEMORY CONTROLLER HAS LEFT.
 *
 * Decode is bandwidth bound: a token reads all 620 MB of an int4 1B model, and
 * round 369 measured the NPU pulling 9.63 GB/s of it. The board is LPDDR5 at
 * 2736 MHz over two 16 bit channels, so the bus peak is about 21.9 GB/s and
 * the NPU alone is using under half of it.
 *
 * That number decides whether splitting work across the CPU, the GPU and the
 * NPU can pay at all. All three sit behind ONE memory controller, and the
 * dependency chain of decode leaves nothing to overlap: every projection feeds
 * the next, so a second engine has to take part of the SAME tensor and read
 * from the SAME DRAM at the same time. If the controller has nothing spare,
 * the split is zero sum however cleverly it is scheduled.
 *
 * This reads a buffer far larger than any cache, sequentially, and reports what
 * it got. Run it alone, then run it again while the NPU decodes, and the pair
 * of numbers answers the question.
 *
 *   CHARSIU_MB       buffer megabytes, default 256
 *   CHARSIU_SECONDS  how long to read for, default 5
 *   CHARSIU_THREADS  how many readers, default 1
 */
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
#include <arm_neon.h>
#endif

static double now_s(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

struct reader {
	const uint8_t *base;
	size_t bytes;
	double seconds;
	double got_gbs;
	uint64_t sink;
};

static void *read_loop(void *vr)
{
	struct reader *r = vr;
	double t0 = now_s(), t1;
	uint64_t passes = 0;
	uint64_t sink = 0;

	do {
		const uint8_t *p = r->base;
		const uint8_t *end = r->base + r->bytes;

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
		uint32x4_t a0 = vdupq_n_u32(0), a1 = vdupq_n_u32(0);
		uint32x4_t a2 = vdupq_n_u32(0), a3 = vdupq_n_u32(0);

		for (; p + 64 <= end; p += 64) {
			a0 = vaddq_u32(a0, vld1q_u32((const uint32_t *)p));
			a1 = vaddq_u32(a1, vld1q_u32((const uint32_t *)(p + 16)));
			a2 = vaddq_u32(a2, vld1q_u32((const uint32_t *)(p + 32)));
			a3 = vaddq_u32(a3, vld1q_u32((const uint32_t *)(p + 48)));
		}
		a0 = vaddq_u32(vaddq_u32(a0, a1), vaddq_u32(a2, a3));
		sink += vaddvq_u32(a0);
#else
		for (; p + 8 <= end; p += 8)
			sink += *(const uint64_t *)p;
#endif
		passes++;
		t1 = now_s();
	} while (t1 - t0 < r->seconds);

	r->sink = sink;
	r->got_gbs = (double)passes * (double)r->bytes / (t1 - t0) / 1e9;
	return NULL;
}

int main(void)
{
	const char *e;
	size_t mb = (e = getenv("CHARSIU_MB")) ? (size_t)atoi(e) : 256;
	double secs = (e = getenv("CHARSIU_SECONDS")) ? atof(e) : 5.0;
	int nth = (e = getenv("CHARSIU_THREADS")) ? atoi(e) : 1;
	uint8_t *buf;
	struct reader *r;
	pthread_t *th;
	double total = 0.0;
	uint64_t sink = 0;

	if (nth < 1)
		nth = 1;
	buf = malloc(mb << 20);
	if (!buf) {
		printf("MEMBW: %zu MB would not allocate\n", mb);
		return 1;
	}
	/* touch every page, and leave a pattern that is not all zero */
	for (size_t i = 0; i < (mb << 20); i += 4096)
		buf[i] = (uint8_t)i;
	memset(buf + 1, 0x5a, (mb << 20) - 1);

	r = calloc((size_t)nth, sizeof(*r));
	th = calloc((size_t)nth, sizeof(*th));
	if (!r || !th)
		return 1;
	for (int i = 0; i < nth; i++) {
		size_t per = (mb << 20) / (size_t)nth & ~(size_t)63;

		r[i].base = buf + (size_t)i * per;
		r[i].bytes = per;
		r[i].seconds = secs;
	}
	for (int i = 1; i < nth; i++)
		pthread_create(&th[i], NULL, read_loop, &r[i]);
	read_loop(&r[0]);
	for (int i = 1; i < nth; i++)
		pthread_join(th[i], NULL);

	for (int i = 0; i < nth; i++) {
		total += r[i].got_gbs;
		sink += r[i].sink;
	}
	printf("MEMBW: %d thread%s over %zu MB for %.0f s: %.2f GB/s"
	       "  (sink %016llx)\n",
	       nth, nth == 1 ? "" : "s", mb, secs, total,
	       (unsigned long long)sink);
	return 0;
}
