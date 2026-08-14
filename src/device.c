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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "charsiu.h"
#include "rocket_uapi.h"

struct charsiu_device {
	int fd;
};

struct charsiu_device *charsiu_open(const char *path)
{
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
	return dev;
}

void charsiu_close(struct charsiu_device *dev)
{
	if (!dev)
		return;
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
	(void)dev;
	if (!bo || !bo->map)
		return;
	munmap(bo->map, bo->size);
	bo->map = NULL;
}

int charsiu_bo_prep(struct charsiu_device *dev, struct charsiu_bo *bo,
		    int64_t timeout_ns)
{
	struct drm_rocket_prep_bo req = { 0 };

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

int charsiu_submit(struct charsiu_device *dev, const struct charsiu_bo *regcmd,
		   unsigned regcmd_count, const uint32_t *in_handles,
		   unsigned in_count, const uint32_t *out_handles,
		   unsigned out_count)
{
	struct drm_rocket_task task = { 0 };
	struct drm_rocket_job job = { 0 };
	struct drm_rocket_submit submit = { 0 };

	/*
	 * The task's regcmd field is 32 bits: it is the DMA address the NPU's
	 * program counter fetches from, and the driver hands out a 32 bit IOVA
	 * window per fd. A BO above 4 GiB of IOVA cannot be a register stream.
	 */
	if (regcmd->dma_address >> 32)
		return -ERANGE;

	task.regcmd = (uint32_t)regcmd->dma_address;
	task.regcmd_count = regcmd_count;

	job.tasks = (uint64_t)(uintptr_t)&task;
	job.task_count = 1;
	job.task_struct_size = sizeof(task);
	job.in_bo_handles = (uint64_t)(uintptr_t)in_handles;
	job.in_bo_handle_count = in_count;
	job.out_bo_handles = (uint64_t)(uintptr_t)out_handles;
	job.out_bo_handle_count = out_count;

	submit.jobs = (uint64_t)(uintptr_t)&job;
	submit.job_count = 1;
	submit.job_struct_size = sizeof(job);

	if (ioctl(dev->fd, DRM_IOCTL_ROCKET_SUBMIT, &submit))
		return -errno;
	return 0;
}
