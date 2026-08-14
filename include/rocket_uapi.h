/* SPDX-License-Identifier: MIT */
/*
 * The mainline rocket accel uAPI, copied so charsiu builds against a toolchain
 * whose headers predate it. This must stay identical to
 * include/uapi/drm/rocket_accel.h in the kernel; it is not a place to invent
 * fields.
 */
#ifndef CHARSIU_ROCKET_UAPI_H
#define CHARSIU_ROCKET_UAPI_H

#include <stdint.h>
#include <sys/ioctl.h>

#define DRM_COMMAND_BASE 0x40

#define DRM_ROCKET_CREATE_BO 0x00
#define DRM_ROCKET_SUBMIT    0x01
#define DRM_ROCKET_PREP_BO   0x02
#define DRM_ROCKET_FINI_BO   0x03

struct drm_rocket_create_bo {
	uint32_t size;
	uint32_t handle;
	uint64_t dma_address;
	uint64_t offset;
};

struct drm_rocket_prep_bo {
	uint32_t handle;
	uint32_t reserved;
	int64_t timeout_ns;
};

struct drm_rocket_fini_bo {
	uint32_t handle;
	uint32_t reserved;
};

struct drm_rocket_task {
	uint32_t regcmd;
	uint32_t regcmd_count;
};

struct drm_rocket_job {
	uint64_t tasks;
	uint64_t in_bo_handles;
	uint64_t out_bo_handles;
	uint32_t task_count;
	uint32_t task_struct_size;
	uint32_t in_bo_handle_count;
	uint32_t out_bo_handle_count;
};

struct drm_rocket_submit {
	uint64_t jobs;
	uint32_t job_count;
	uint32_t job_struct_size;
	uint64_t reserved;
};

#define DRM_IOCTL_ROCKET_CREATE_BO \
	_IOWR('d', DRM_COMMAND_BASE + DRM_ROCKET_CREATE_BO, struct drm_rocket_create_bo)
#define DRM_IOCTL_ROCKET_SUBMIT \
	_IOW('d', DRM_COMMAND_BASE + DRM_ROCKET_SUBMIT, struct drm_rocket_submit)
#define DRM_IOCTL_ROCKET_PREP_BO \
	_IOW('d', DRM_COMMAND_BASE + DRM_ROCKET_PREP_BO, struct drm_rocket_prep_bo)
#define DRM_IOCTL_ROCKET_FINI_BO \
	_IOW('d', DRM_COMMAND_BASE + DRM_ROCKET_FINI_BO, struct drm_rocket_fini_bo)

#endif /* CHARSIU_ROCKET_UAPI_H */
