#define _POSIX_C_SOURCE 200809L
/*
 * npu_prep_cost -- what a PREP and a FINI cost, by buffer size, with no job in
 * the way.
 *
 * The decode budget came down to one unsplit number. Round 148 put the CPU
 * side of a call at 7.3% of a token; round 150's spin counters showed 99% of
 * polls win at a mean of 114 us, so the blocking wakeup is about 2%. What is
 * left inside "waiting for the fence (the invalidate is in there)" is the
 * hardware computing plus the output buffer's cache maintenance -- and the
 * report cannot separate them, because the ioctl that waits is the ioctl that
 * invalidates.
 *
 * ⚠ A BUFFER THAT WAS NEVER SUBMITTED HAS NO FENCE TO WAIT ON. So prep on a
 * freshly allocated BO is the invalidate alone, timed without a job, without
 * the NPU doing anything, and without any risk of disturbing a run. That is
 * the whole trick here.
 *
 * charsiu's decode output buffer is `slots * out_stride`, and out_stride is
 * CHARSIU_NPU_NMAX * 4 -- 32 KB at the default 8192, whatever the tensor's own
 * n is. If the invalidate is a meaningful part of the 114 us, sizing that
 * buffer to the tensor instead of to nmax is worth doing; if it is 2 us, it is
 * not, and the per-call floor is the device's own and this project is finished
 * arguing about it.
 *
 *   npu_prep_cost [reps]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "charsiu.h"

static double now_us(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

int main(int argc, char **argv)
{
	static const size_t sizes[] = {
		4096, 8192, 32768, 65536, 262144, 1048576, 4194304
	};
	struct charsiu_device *dev;
	unsigned reps = argc > 1 ? (unsigned)atoi(argv[1]) : 2000;
	unsigned i;

	dev = charsiu_open(NULL);
	if (!dev) {
		fprintf(stderr, "npu_prep_cost: no accel device\n");
		return 1;
	}
	printf("prep and fini on a buffer that was never submitted, so there is\n"
	       "no fence to wait on and the number is cache maintenance alone.\n"
	       "%u repetitions each.\n\n", reps);
	printf("  %10s  %10s  %10s  %12s  %12s\n",
	       "bytes", "prep us", "fini us", "prep GB/s", "fini GB/s");
	for (i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) {
		struct charsiu_bo bo;
		double t0, tp, tf;
		unsigned r;

		if (charsiu_bo_alloc(dev, sizes[i], &bo)) {
			printf("  %10zu  (would not allocate)\n", sizes[i]);
			continue;
		}
		/* touch every page: a fault in the timed loop is not the
		 * measurement, and the first pass would carry all of them */
		memset(bo.map, 0, sizes[i]);
		charsiu_bo_prep(dev, &bo, 1000000000);
		charsiu_bo_fini(dev, &bo);

		t0 = now_us();
		for (r = 0; r < reps; r++)
			charsiu_bo_prep(dev, &bo, 1000000000);
		tp = (now_us() - t0) / reps;

		t0 = now_us();
		for (r = 0; r < reps; r++)
			charsiu_bo_fini(dev, &bo);
		tf = (now_us() - t0) / reps;

		printf("  %10zu  %10.2f  %10.2f  %12.2f  %12.2f\n",
		       sizes[i], tp, tf,
		       sizes[i] / tp / 1e3, sizes[i] / tf / 1e3);
		charsiu_bo_free(dev, &bo);
	}
	printf("\n⚠ An ioctl with no work still costs an ioctl, so read the\n"
	       "  smallest row as the syscall floor and the slope as the cache.\n");
	charsiu_close(dev);
	return 0;
}
