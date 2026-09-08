#define _POSIX_C_SOURCE 200809L
/*
 * npu_job_cost -- what a JOB costs to start, measured instead of subtracted.
 *
 * ⚠⚠ WHY THIS EXISTS. Rounds 148 to 151 took decode's 71 us per-call floor
 * apart: ~11 us of blocking wakeup, 9.2 us of cache maintenance, ~5 us of
 * syscalls. The remaining ~46 us was then written down as "the device starting
 * a job, which no userspace work removes" -- and that number was never
 * measured. It is a RESIDUAL, and the same notebook page had just retracted a
 * different residual for exactly this reason.
 *
 * A residual attributed to hardware is the shape of the wall this project
 * already walked into once. Every explanation of that one was a property of
 * the CBUF sequencer until it turned out to be a field layout in a header
 * copied from RK3588. So: measure it.
 *
 * FOUR ARMS, and the second is the one nobody has run:
 *
 *   A  one job, one ioctl                     the floor itself
 *   B  TWO jobs in ONE ioctl, one fd          never tried
 *   C  two jobs, two ioctls, one fd           the ioctl's own share
 *   D  two jobs, two ioctls, two fds          what charsiu does today
 *
 * charsiu opens the accel node twice and submits one job per fd in a `for`
 * loop, so the second core starts a whole syscall after the first. But
 * charsiu_submit_jobs already takes a LIST, and device.c says why that matters:
 * "Jobs are what the scheduler can hand to different cores." If B beats D, the
 * second core has been starting late for the life of this runtime.
 *
 * The matmul is deliberately tiny -- the point is the fixed cost, so the
 * arithmetic should be nothing. A task sweep on one job prices chaining, which
 * is the other half: if a job is expensive and a task is cheap, packing more
 * tasks per job is the lever, and if they cost the same it is not.
 *
 *   npu_job_cost [reps]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "charsiu.h"

#define NREG 4096

struct unit {                 /* everything one job needs */
	struct charsiu_bo wt, in, coef, reg, ob;
	struct charsiu_task task;
	uint32_t ins[2], outs[1];
	size_t nreg;
};

static double now_us(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

/* the smallest matmul that still assembles: the arithmetic must not matter */
static int unit_make(struct charsiu_device *dev, struct unit *u,
		     unsigned m, unsigned k, unsigned n)
{
	struct charsiu_job job = { 0 };
	size_t insz;

	memset(u, 0, sizeof(*u));
	job.cbuf_window = (unsigned)charsiu_cbuf_window();
	job.mm.m = m; job.mm.k = k; job.mm.n = n;
	job.mm.wdtype = CHARSIU_INT8;
	job.mm.adtype = CHARSIU_INT8;
	job.input_zero_point = 128;
	job.weight_zero_point = 128;
	job.input_scale = job.weight_scale = job.output_scale = 1.0f;
	job.acc_out = 1;

	insz = (size_t)charsiu_entries_per_row(&job.mm) * 64 * m + 4096;
	if (charsiu_bo_alloc(dev, charsiu_weight_bytes(&job.mm) + 4096, &u->wt) ||
	    charsiu_bo_alloc(dev, insz, &u->in) ||
	    charsiu_bo_alloc(dev, charsiu_coef_bytes(&job.mm) + 4096, &u->coef) ||
	    charsiu_bo_alloc(dev, NREG, &u->reg) ||
	    charsiu_bo_alloc(dev, (size_t)m * n * 4 + 4096, &u->ob))
		return -1;
	memset(u->wt.map, 0, charsiu_weight_bytes(&job.mm));
	memset(u->in.map, 0, insz);
	memset(u->coef.map, 0, charsiu_coef_bytes(&job.mm));
	memset(u->ob.map, 0, (size_t)m * n * 4);
	charsiu_bo_fini(dev, &u->wt);
	charsiu_bo_fini(dev, &u->in);
	charsiu_bo_fini(dev, &u->coef);
	charsiu_bo_fini(dev, &u->ob);

	job.input_addr  = (uint32_t)u->in.dma_address;
	job.output_addr = (uint32_t)u->ob.dma_address;
	job.weight_addr = (uint32_t)u->wt.dma_address;
	job.coef_addr   = (uint32_t)u->coef.dma_address;

	charsiu_bo_prep(dev, &u->reg, 1000000000);
	u->nreg = charsiu_emit_job(&job, u->reg.map, NREG / 8);
	charsiu_bo_fini(dev, &u->reg);
	if (!u->nreg)
		return -1;
	if (u->reg.dma_address >> 32)
		return -1;
	u->task.regcmd = (uint32_t)u->reg.dma_address;
	u->task.regcmd_count = (uint32_t)u->nreg;
	u->ins[0] = u->in.handle; u->ins[1] = u->wt.handle;
	u->outs[0] = u->ob.handle;
	return 0;
}

static void unit_free(struct charsiu_device *dev, struct unit *u)
{
	charsiu_bo_free(dev, &u->wt); charsiu_bo_free(dev, &u->in);
	charsiu_bo_free(dev, &u->coef); charsiu_bo_free(dev, &u->reg);
	charsiu_bo_free(dev, &u->ob);
}

static void fill(struct charsiu_joblist *jl, struct unit *u, unsigned ntask)
{
	static struct charsiu_task chain[64];
	unsigned t;

	for (t = 0; t < ntask && t < 64; t++)
		chain[t] = u->task;          /* the same program, t times */
	jl->tasks = chain;
	jl->task_count = ntask;
	jl->in_handles = u->ins;  jl->in_count = 2;
	jl->out_handles = u->outs; jl->out_count = 1;
}

int main(int argc, char **argv)
{
	unsigned reps = argc > 1 ? (unsigned)atoi(argv[1]) : 400;
	struct charsiu_device *d0, *d1;
	struct unit a, b, b1;
	struct charsiu_joblist jl[2];
	double t0;
	unsigned r, nt;
	int have_d1 = 0;

	d0 = charsiu_open(NULL);
	if (!d0) { fprintf(stderr, "npu_job_cost: no accel device\n"); return 1; }
	d1 = charsiu_open(NULL);
	have_d1 = d1 != NULL;

	if (unit_make(d0, &a, 1, 64, 32) || unit_make(d0, &b, 1, 64, 32)) {
		fprintf(stderr, "npu_job_cost: could not build a job on fd 0\n");
		return 1;
	}
	if (have_d1 && unit_make(d1, &b1, 1, 64, 32)) {
		fprintf(stderr, "npu_job_cost: could not build a job on fd 1\n");
		have_d1 = 0;
	}
	printf("m=1 k=64 n=32, so the arithmetic is nothing and this is dispatch.\n");
	printf("%u repetitions an arm.\n\n", reps);

	/* --- the task sweep on ONE job: what does chaining cost --- */
	printf("  one job, tasks chained inside it\n");
	printf("  %8s %12s %14s\n", "tasks", "us a job", "us a task");
	for (nt = 1; nt <= 16; nt *= 2) {
		double us;

		fill(&jl[0], &a, nt);
		charsiu_submit_jobs(d0, jl, 1);
		charsiu_bo_prep(d0, &a.ob, 1000000000);
		t0 = now_us();
		for (r = 0; r < reps; r++) {
			fill(&jl[0], &a, nt);
			if (charsiu_submit_jobs(d0, jl, 1))
				break;
			charsiu_bo_prep(d0, &a.ob, 1000000000);
		}
		us = (now_us() - t0) / reps;
		printf("  %8u %12.2f %14.2f\n", nt, us, us / nt);
	}

	/* --- the MB axis: is the bandwidth term a constant at all --- */
	/*
	 * ⚠⚠ THE COEFFICIENT THE SHAPE PREDICTOR NEEDS AND CANNOT ASSUME.
	 *
	 * charsiu_shapes fits a token as calls*a + tasks*b + MB*c. Calibrated
	 * on qwen3 alone it predicts gemma4 +12.6% and Phi-3.5 +25.1%, both
	 * HIGH and monotone in size -- which is what a single average `c` does
	 * when the real one rises with the tensor. Round 147 saw exactly that
	 * inside one run: 8.29 GB/s on o_proj, 15.56 on the output head.
	 *
	 * So sweep the bytes with the task count HELD AT ONE. If the us a MB
	 * falls as MB grows, `c` is not a constant and no per-model constant
	 * will fix a predictor built on one.
	 */
	/*
	 * ⚠⚠ THE SMALL END IS NOISE AND TWO ROUNDS PROVED IT, so it is
	 * repeated more and reported with a spread rather than as a point.
	 *
	 * Rounds 155 and 161 on the same shapes: 0.0020 MB read 60.10 then
	 * 37.96 us, 0.0328 read 40.13 then 73.31, 0.2621 read 44.44 then
	 * 88.93. A factor of two, in both directions. Round 161 was also non
	 * monotone inside itself -- 0.5243 MB at 77.81 us against 0.2621 at
	 * 88.93, twice the bytes and less time.
	 *
	 * The large end is not like that at all: 4.1943 read 406.81 then
	 * 399.14, and 8.3886 read 785.00 then 769.01, agreeing to 2%.
	 *
	 * That matters because it decides what a prediction MEANS. Every call
	 * SmolLM2-135M makes is 0.13 to 0.69 MB -- entirely inside the noisy
	 * band -- and the shape predictor missed it by 36%. Phi-3.5's calls are
	 * 11 to 20 MB, inside the stable band, and it is missed by 6%. Until
	 * the small end is repeatable, the first of those two numbers cannot
	 * be attributed to the model at all.
	 */
	printf("\n  one job, one task, the weight bytes swept\n");
	printf("  %6s %6s %9s %10s %10s %9s\n",
	       "k", "n", "MB", "us", "us a MB", "spread");
	{
		/*
		 * ⚠ THE RANGE HAS TO COVER WHAT A MODEL ACTUALLY ASKS FOR.
		 * The first sweep stopped at 8.39 MB and charsiu_shapes then
		 * had to extrapolate Phi-3.5's gate+up, which is 19.6 MB in one
		 * call -- 2.3x outside the data. Its predictions came back
		 * claiming 106% of the token was matmul, which is not a fit
		 * error, it is a question asked outside where it was answered.
		 * The small end matters for the other reason: SmolLM2-135M's
		 * calls are 0.13 to 0.69 MB and its prediction missed by 36%.
		 */
		static const unsigned ks[] = { 64, 256, 512, 1024, 1024, 1024,
					       2048, 2048, 4096, 4096 };
		static const unsigned ns[] = { 32, 128, 256,  256,  512, 1024,
					       2048, 4096, 4096, 8192 };
		unsigned c;

		for (c = 0; c < sizeof(ks) / sizeof(*ks); c++) {
			struct unit u;
			double us, mb;

			/*
			 * ⚠⚠ REFUSE WHAT THE DEVICE WILL NOT TAKE, HERE, NOT BY
			 * SUBMITTING IT.
			 *
			 * The sweep was widened to 67 MB with 4096 x 16384 on
			 * the end. n = 16384 is past the 8192 the device is
			 * opened for; the job was submitted anyway, timed out,
			 * and took the IOMMU with it -- "Error during raw
			 * reset, MMU_DTE_ADDR is not functioning", the state
			 * this project has a memory note about, and the board
			 * was dead for five hours.
			 *
			 * A probe that walks an axis has to know where the axis
			 * ends. Both bounds are already written down elsewhere
			 * in this tree and neither was checked here.
			 */
			if (ns[c] > 8192 || (size_t)(ks[c] / 32) * 1 > 5120) {
				printf("  %6u %6u  (past the device's limits:"
				       " n <= 8192, (k/32)*m <= 5120)\n",
				       ks[c], ns[c]);
				continue;
			}
			if (unit_make(d0, &u, 1, ks[c], ns[c])) {
				printf("  %6u %6u  (would not build)\n",
				       ks[c], ns[c]);
				continue;
			}
			mb = (double)ks[c] * ns[c] / 1e6;   /* int8 weights */
			fill(&jl[0], &u, 1);
			charsiu_submit_jobs(d0, jl, 1);
			charsiu_bo_prep(d0, &u.ob, 1000000000);
			{
				double best = 1e18, worst = 0.0;
				unsigned pass;

				/* five passes, so the row carries its own
				 * spread and a noisy point cannot be read as
				 * a measurement */
				for (pass = 0; pass < 5; pass++) {
					t0 = now_us();
					for (r = 0; r < reps; r++) {
						fill(&jl[0], &u, 1);
						if (charsiu_submit_jobs(d0, jl, 1))
							break;
						charsiu_bo_prep(d0, &u.ob,
								1000000000);
					}
					us = (now_us() - t0) / reps;
					if (us < best) best = us;
					if (us > worst) worst = us;
				}
				us = best;
				printf("  %6u %6u %9.4f %10.2f %10.1f %8.1f%%\n",
				       ks[c], ns[c], mb, us,
				       mb > 0 ? us / mb : 0.0,
				       100.0 * (worst - best) / best);
			}
			unit_free(d0, &u);
		}
	}

	/* --- the four ways to get two jobs onto the hardware --- */
	printf("\n  two jobs, four ways\n");
	{
		double us;

		t0 = now_us();
		for (r = 0; r < reps; r++) {
			fill(&jl[0], &a, 1);
			charsiu_submit_jobs(d0, jl, 1);
			charsiu_bo_prep(d0, &a.ob, 1000000000);
		}
		us = (now_us() - t0) / reps;
		printf("  A  one job, one ioctl                 %8.2f us\n", us);

		t0 = now_us();
		for (r = 0; r < reps; r++) {
			fill(&jl[0], &a, 1);
			fill(&jl[1], &b, 1);
			/* ⚠ THE ARM NOBODY HAS RUN: one ioctl, two jobs, and
			 * the driver is free to put them on both cores. */
			if (charsiu_submit_jobs(d0, jl, 2))
				break;
			charsiu_bo_prep(d0, &a.ob, 1000000000);
			charsiu_bo_prep(d0, &b.ob, 1000000000);
		}
		us = (now_us() - t0) / reps;
		printf("  B  two jobs, ONE ioctl, one fd        %8.2f us\n", us);

		t0 = now_us();
		for (r = 0; r < reps; r++) {
			fill(&jl[0], &a, 1);
			charsiu_submit_jobs(d0, jl, 1);
			fill(&jl[0], &b, 1);
			charsiu_submit_jobs(d0, jl, 1);
			charsiu_bo_prep(d0, &a.ob, 1000000000);
			charsiu_bo_prep(d0, &b.ob, 1000000000);
		}
		us = (now_us() - t0) / reps;
		printf("  C  two jobs, two ioctls, one fd       %8.2f us\n", us);

		if (have_d1) {
			t0 = now_us();
			for (r = 0; r < reps; r++) {
				fill(&jl[0], &a, 1);
				charsiu_submit_jobs(d0, jl, 1);
				fill(&jl[0], &b1, 1);
				charsiu_submit_jobs(d1, jl, 1);
				charsiu_bo_prep(d0, &a.ob, 1000000000);
				charsiu_bo_prep(d1, &b1.ob, 1000000000);
			}
			us = (now_us() - t0) / reps;
			printf("  D  two jobs, two ioctls, two fds      %8.2f us"
			       "   <- what charsiu does\n", us);
		} else {
			printf("  D  (a second fd would not open)\n");
		}
	}
	printf("\n⚠ B against D is the question. If B is faster, charsiu's second\n"
	       "  core has been starting a syscall late since it was written.\n"
	       "⚠ And A against C says what an ioctl costs when the work is the\n"
	       "  same, which is the control for reading B at all.\n");

	unit_free(d0, &a); unit_free(d0, &b);
	if (have_d1) { unit_free(d1, &b1); charsiu_close(d1); }
	charsiu_close(d0);
	return 0;
}
