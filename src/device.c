// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The device shim: /dev/accel/accel0 through the mainline rocket uAPI.
 *
 * rocket is a generic register command submitter. It owns no op set: a job is a
 * buffer of register writes plus the BOs they address, and what the NPU computes
 * is entirely a userspace matter. That is why an open LLM runtime is possible at
 * all without touching the kernel.
 *
 * Four ioctls, and the ordering rule that goes with them:
 *
 *   CREATE_BO   allocate, and get back a handle, a DMA address for the register
 *               stream to point at, and an mmap offset for the CPU
 *   PREP_BO     take CPU ownership before reading or writing through the map
 *   FINI_BO     give it back before submitting
 *   SUBMIT      run the tasks
 *
 * PREP and FINI are the cache maintenance. Skipping them does not fail, it
 * returns stale data, which is the worst kind of bug to chase on an accelerator.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "charsiu.h"
#include "rocket_uapi.h"

struct charsiu_device {
	int fd;
	struct charsiu_bo reserve;   /* holds IOVA 0 so nothing real lands there */
};

/*
 * ⚠ A BOARD WITH NO DEBUGGER DESERVES BETTER THAN "Segmentation fault".
 *
 * Every crash this project has had on hardware arrived as that one word and
 * nothing else: no tensor, no shape, no stage. Two rounds went into narrowing
 * one of them to a four line function by rebuilding on the host under ASan,
 * which only worked because that particular bug also reproduced there. A bug
 * in the ioctl path does not.
 *
 * So leave a breadcrumb. charsiu_note() stores a pointer to a STATIC string
 * plus two numbers; the handler writes them and re-raises, so the shell still
 * reports the real signal and a core file is still produced.
 *
 * ⚠ ASYNC SIGNAL SAFE, which rules out printf and every string function that
 * allocates. write() and a hand rolled integer are all that is allowed here,
 * and the note must point at a string literal or a buffer that outlives the
 * crash -- never at something on a stack that is about to be unwound.
 */
static const char *volatile note_what = "nothing yet";
static volatile unsigned long note_a, note_b;

void charsiu_note(const char *what, unsigned long a, unsigned long b)
{
	note_what = what;
	note_a = a;
	note_b = b;
}

static void note_num(char *buf, size_t *at, unsigned long v)
{
	char t[24];
	int n = 0;

	if (!v) { buf[(*at)++] = '0'; return; }
	while (v && n < 24) { t[n++] = (char)('0' + v % 10); v /= 10; }
	while (n) buf[(*at)++] = t[--n];
}

static void note_crash(int sig)
{
	char buf[256];
	size_t at = 0;
	const char *w = note_what;
	const char *p = "\ncharsiu: died in ";

	while (*p) buf[at++] = *p++;
	while (w && *w && at < sizeof(buf) - 48) buf[at++] = *w++;
	buf[at++] = ' '; buf[at++] = '(';
	note_num(buf, &at, note_a);
	buf[at++] = ','; buf[at++] = ' ';
	note_num(buf, &at, note_b);
	buf[at++] = ')'; buf[at++] = ' '; buf[at++] = 's'; buf[at++] = 'i';
	buf[at++] = 'g'; buf[at++] = '=';
	note_num(buf, &at, (unsigned long)sig);
	buf[at++] = '\n';
	if (write(2, buf, at) < 0) { /* nothing useful to do */ }

	signal(sig, SIG_DFL);
	raise(sig);
}

void charsiu_note_install(void)
{
	static int done;

	if (done || getenv("CHARSIU_NO_CRASH_NOTE"))
		return;
	done = 1;
	signal(SIGSEGV, note_crash);
	signal(SIGBUS, note_crash);
	signal(SIGILL, note_crash);
	signal(SIGFPE, note_crash);
}

struct charsiu_device *charsiu_open(const char *path)
{
	charsiu_note_install();
	struct charsiu_device *dev;
	int fd;

	fd = open(path ? path : "/dev/accel/accel0", O_RDWR | O_CLOEXEC);
	if (fd < 0)
		return NULL;

	dev = calloc(1, sizeof(*dev));
	if (!dev) {
		close(fd);
		return NULL;
	}
	dev->fd = fd;

	/*
	 * Burn the first allocation.
	 *
	 * rocket hands out IOVA from a per fd drm_mm started at the IOMMU
	 * aperture, so the FIRST buffer object of an fd sits at address 0. That
	 * is a legal address for data, but the program counter is pointed at
	 * the register stream by writing its address into BASE_ADDRESS, and a
	 * stream at 0 is the one thing that cannot be told from an unset
	 * pointer. Mesa never meets this because it has allocated a dozen
	 * buffers before it builds a stream.
	 *
	 * Round 146 had a stream identical to Mesa's, entry for entry, and still
	 * timed out with its register stream at IOVA 0. Holding the address
	 * costs one page and removes the question.
	 */
	charsiu_bo_alloc(dev, 4096, &dev->reserve);
	return dev;
}

void charsiu_close(struct charsiu_device *dev)
{
	if (!dev)
		return;
	charsiu_bo_free(dev, &dev->reserve);
	close(dev->fd);
	free(dev);
}

int charsiu_bo_alloc(struct charsiu_device *dev, size_t size, struct charsiu_bo *bo)
{
	struct drm_rocket_create_bo req = { 0 };
	void *map;

	if (!dev || !bo || !size)
		return -EINVAL;

	req.size = (uint32_t)size;
	if (ioctl(dev->fd, DRM_IOCTL_ROCKET_CREATE_BO, &req))
		return -errno;

	map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
		   dev->fd, (off_t)req.offset);
	if (map == MAP_FAILED)
		return -errno;

	bo->handle = req.handle;
	bo->dma_address = req.dma_address;
	bo->size = size;
	bo->map = map;
	return 0;
}

void charsiu_bo_free(struct charsiu_device *dev, struct charsiu_bo *bo)
{
	struct drm_gem_close req = { 0 };

	if (!bo || !bo->map)
		return;
	munmap(bo->map, bo->size);
	bo->map = NULL;
	/*
	 * And give the HANDLE back, which this did not do. rocket has no
	 * DESTROY_BO of its own, so the generic GEM close is it, and without it
	 * anything that allocates per shape rather than once leaks every buffer
	 * until the process exits. The shape sweep in charsiu_bench is the first
	 * thing here that allocates in a loop, and it would have leaked about
	 * 300 MB of IOVA in seven iterations.
	 */
	if (dev && bo->handle) {
		req.handle = bo->handle;
		ioctl(dev->fd, DRM_IOCTL_GEM_CLOSE, &req);
		bo->handle = 0;
	}
}

int charsiu_bo_prep(struct charsiu_device *dev, struct charsiu_bo *bo,
		    int64_t timeout_ns)
{
	struct drm_rocket_prep_bo req = { 0 };
	struct timespec now;

	/*
	 * THE KERNEL WANTS AN ABSOLUTE DEADLINE, not a duration.
	 * drm_timeout_abs_to_jiffies() reads this field as CLOCK_MONOTONIC
	 * nanoseconds, so passing a duration asks it to wait until a moment
	 * that has already gone, it computes a timeout of zero, and the wait
	 * returns -EBUSY immediately without waiting for anything.
	 *
	 * Round 148: every one of charsiu's submits reported "wait FAILED -16"
	 * and that was read as the job hanging. The job may well have been
	 * running; the tool then exited, closing the fd and tearing down the
	 * buffers under an NPU that was still reading them. This function takes
	 * a duration, which is what a caller means, and converts it.
	 */
	clock_gettime(CLOCK_MONOTONIC, &now);
	timeout_ns += (int64_t)now.tv_sec * 1000000000LL + now.tv_nsec;

	req.handle = bo->handle;
	req.timeout_ns = timeout_ns;
	/*
	 * EINTR here is a signal, not a failure, and the deadline has to be
	 * carried across the retry rather than restarted or a slow job can wait
	 * forever in a loop that keeps resetting its own timeout.
	 */
	while (ioctl(dev->fd, DRM_IOCTL_ROCKET_PREP_BO, &req)) {
		if (errno != EINTR)
			return -errno;
	}
	return 0;
}

int charsiu_bo_fini(struct charsiu_device *dev, struct charsiu_bo *bo)
{
	struct drm_rocket_fini_bo req = { 0 };

	req.handle = bo->handle;
	if (ioctl(dev->fd, DRM_IOCTL_ROCKET_FINI_BO, &req))
		return -errno;
	return 0;
}

/*
 * THE TWO AXES OF BATCHING, and why this takes a list rather than one stream.
 *
 * A submit carries jobs and a job carries tasks, and they are not the same
 * lever. Tasks inside one job are CHAINED on a single core: the program counter
 * walks them without the driver being asked again, which is the thing the
 * PC_TASK_CON field layout work in the driver repository had to fix before it
 * worked at all. Jobs are what the scheduler can hand to different cores.
 *
 * This matters more here than anywhere else in charsiu. A projection at M = 1
 * with K = 2048 and N = 1024 is about 4 MOP, which at this hardware's rated 6
 * TOPS is under a microsecond of arithmetic. Anything measurable around it is
 * dispatch, so the interesting number is not how fast one submit is, it is what
 * the SECOND task in the same submit costs. The RK3588 stacks concluded a single
 * row matmul belongs on the CPU; if that is true here it will be visible as a
 * per submit floor that batching cannot get under.
 */
int charsiu_submit_jobs(struct charsiu_device *dev,
			const struct charsiu_joblist *jobs, unsigned job_count)
{
	struct drm_rocket_submit submit = { 0 };
	struct drm_rocket_job *kj;
	struct drm_rocket_task *kt;
	unsigned total = 0, i, t, at = 0;
	int ret = 0;

	if (!dev || !jobs || !job_count)
		return -EINVAL;
	for (i = 0; i < job_count; i++)
		total += jobs[i].task_count;
	if (!total)
		return -EINVAL;

	kj = calloc(job_count, sizeof(*kj));
	kt = calloc(total, sizeof(*kt));
	if (!kj || !kt) {
		free(kj); free(kt);
		return -ENOMEM;
	}

	for (i = 0; i < job_count; i++) {
		kj[i].tasks = (uint64_t)(uintptr_t)(kt + at);
		kj[i].task_count = jobs[i].task_count;
		kj[i].task_struct_size = sizeof(*kt);
		kj[i].in_bo_handles = (uint64_t)(uintptr_t)jobs[i].in_handles;
		kj[i].in_bo_handle_count = jobs[i].in_count;
		kj[i].out_bo_handles = (uint64_t)(uintptr_t)jobs[i].out_handles;
		kj[i].out_bo_handle_count = jobs[i].out_count;

		/*
		 * struct charsiu_task carries regcmd as a uint32_t already,
		 * which is the whole of the range check: the field the driver
		 * takes is 32 bits because the program counter fetches from a
		 * 32 bit IOVA window per fd. The narrowing happens where a
		 * dma_address is turned into one, in charsiu_submit below and
		 * in whatever builds a task list.
		 */
		for (t = 0; t < jobs[i].task_count; t++, at++) {
			kt[at].regcmd = jobs[i].tasks[t].regcmd;
			kt[at].regcmd_count = jobs[i].tasks[t].regcmd_count;
		}
	}

	submit.jobs = (uint64_t)(uintptr_t)kj;
	submit.job_count = job_count;
	submit.job_struct_size = sizeof(*kj);
	if (ioctl(dev->fd, DRM_IOCTL_ROCKET_SUBMIT, &submit))
		ret = -errno;
	free(kj);
	free(kt);
	return ret;
}

int charsiu_submit(struct charsiu_device *dev, const struct charsiu_bo *regcmd,
		   unsigned regcmd_count, const uint32_t *in_handles,
		   unsigned in_count, const uint32_t *out_handles,
		   unsigned out_count)
{
	struct charsiu_task task;
	struct charsiu_joblist job;

	if (regcmd->dma_address >> 32)
		return -ERANGE;

	task.regcmd = (uint32_t)regcmd->dma_address;
	task.regcmd_count = regcmd_count;

	job.tasks = &task;
	job.task_count = 1;
	job.in_handles = in_handles;
	job.in_count = in_count;
	job.out_handles = out_handles;
	job.out_count = out_count;

	return charsiu_submit_jobs(dev, &job, 1);
}
