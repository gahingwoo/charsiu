// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * Open the NPU, allocate a buffer, write through it, read it back.
 *
 * The smallest thing that proves charsiu's own path to the hardware works: the
 * device opens, a BO comes back with a DMA address the register stream can
 * point at and a mapping the CPU can reach, and the cache maintenance round
 * trips. Nothing here computes anything. It fails loudly if the IOVA is above
 * the 32 bit window a register stream can address, because that is a real limit
 * and a confusing one to hit later.
 *
 * Board, round 140: the first BO comes back at IOVA 0, and that is correct. The
 * driver hands out addresses from a per fd drm_mm started at the IOMMU
 * aperture, so ZERO IS A VALID ADDRESS HERE. An address register reading 0
 * cannot be taken to mean unset on this driver, which is the opposite of how
 * the same zero reads in a vendor model file, where it is an unpatched
 * placeholder.
 */
#include <stdio.h>
#include <string.h>

#include "charsiu.h"

int main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : "/dev/accel/accel0";
	struct charsiu_device *dev;
	struct charsiu_bo bo = { 0 };
	unsigned char *p;
	int ret, i, bad = 0;

	dev = charsiu_open(path);
	if (!dev) {
		printf("open %s FAILED\n", path);
		return 1;
	}
	printf("open %s ok\n", path);

	ret = charsiu_bo_alloc(dev, 4096, &bo);
	if (ret) {
		printf("create bo FAILED: %d\n", ret);
		return 1;
	}
	printf("bo handle %u  dma 0x%llx  size %zu  %s%s\n", bo.handle,
	       (unsigned long long)bo.dma_address, bo.size,
	       (bo.dma_address >> 32) ? "ABOVE the 32 bit regcmd window"
				      : "in the regcmd window",
	       bo.dma_address ? "" : "  (zero is a valid IOVA here)");

	ret = charsiu_bo_prep(dev, &bo, 1000000000);
	if (ret) {
		printf("prep FAILED: %d\n", ret);
		return 1;
	}
	p = bo.map;
	for (i = 0; i < 4096; i++)
		p[i] = (unsigned char)(i * 7);
	ret = charsiu_bo_fini(dev, &bo);
	if (ret) {
		printf("fini FAILED: %d\n", ret);
		return 1;
	}
	ret = charsiu_bo_prep(dev, &bo, 1000000000);
	if (ret) {
		printf("second prep FAILED: %d\n", ret);
		return 1;
	}
	for (i = 0; i < 4096; i++)
		if (p[i] != (unsigned char)(i * 7))
			bad++;
	charsiu_bo_fini(dev, &bo);
	printf("write/readback through the mapping: %s (%d bytes differ)\n",
	       bad ? "MISMATCH" : "ok", bad);

	charsiu_bo_free(dev, &bo);
	charsiu_close(dev);
	return bad ? 1 : 0;
}
