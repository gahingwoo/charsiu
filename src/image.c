// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * A picture off the disk, in the shape a vision tower reads.
 *
 * ⚠ THE PREPROCESSING IS PART OF THE MODEL. A tower trained on one resize and
 * one normalisation, fed another, does not fail: it answers confidently about
 * a different picture. So the mean and the standard deviation come from the
 * mmproj's own keys, and the resize is written down here rather than left to
 * whatever the caller had lying around.
 */
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#define STBI_NO_STDIO_WRITE
#include "stb_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu_llm.h"
#include "charsiu_vision.h"

/*
 * Bilinear, on the half pixel centres.
 *
 * ⚠ HALF PIXEL CENTRES, not corner aligned. torchvision's default and PIL's
 * both map dst pixel i to src (i + 0.5) * scale - 0.5, and the corner aligned
 * form ((i * (w - 1)) / (dw - 1)) differs by half a pixel at every point. On a
 * 512 wide image that is a shift no test of ours would notice and every caption
 * would be slightly about the wrong crop.
 */
static void resize_bilinear(const unsigned char *src, int sw, int sh, int comp,
			    float *dst, int dw, int dh)
{
	int c, y, x;

	for (c = 0; c < 3; c++) {
		int sc = c < comp ? c : comp - 1;

		for (y = 0; y < dh; y++) {
			float fy = ((float)y + 0.5f) * (float)sh / (float)dh - 0.5f;
			int y0 = (int)floorf(fy), y1 = y0 + 1;
			float wy = fy - (float)y0;

			if (y0 < 0) { y0 = 0; }
			if (y1 < 0) { y1 = 0; }
			if (y0 > sh - 1) { y0 = sh - 1; }
			if (y1 > sh - 1) { y1 = sh - 1; }

			for (x = 0; x < dw; x++) {
				float fx = ((float)x + 0.5f) * (float)sw /
					   (float)dw - 0.5f;
				int x0 = (int)floorf(fx), x1 = x0 + 1;
				float wx = fx - (float)x0, a, b;

				if (x0 < 0) { x0 = 0; }
				if (x1 < 0) { x1 = 0; }
				if (x0 > sw - 1) { x0 = sw - 1; }
				if (x1 > sw - 1) { x1 = sw - 1; }

				a = (1.0f - wx) *
				    (float)src[((size_t)y0 * sw + x0) * comp + sc] +
				    wx * (float)src[((size_t)y0 * sw + x1) * comp + sc];
				b = (1.0f - wx) *
				    (float)src[((size_t)y1 * sw + x0) * comp + sc] +
				    wx * (float)src[((size_t)y1 * sw + x1) * comp + sc];
				dst[((size_t)c * dh + y) * dw + x] =
					((1.0f - wy) * a + wy * b) / 255.0f;
			}
		}
	}
}

/*
 * Load `path`, resize it to `side` x `side`, scale to 0..1 and lay it out as
 * [3][side][side]. The caller normalises with the tower's own mean and std.
 *
 * ⚠ IT SQUASHES RATHER THAN CROPS. Every tower here takes a square, and the
 * two ways to get one are to letterbox and to stretch. Stretching is what
 * transformers' image processors do for these models, so it is what this does;
 * it is written here so that when a caption is subtly wrong about a very tall
 * photograph, this is a line to read rather than a thing to guess.
 */
float *charsiu_image_load(const char *path, unsigned side, char *err,
			  size_t errlen)
{
	int w = 0, h = 0, comp = 0;
	unsigned char *px;
	float *out;

	px = stbi_load(path, &w, &h, &comp, 0);
	if (!px) {
		snprintf(err, errlen, "%s: %s", path, stbi_failure_reason());
		return NULL;
	}
	if (comp < 1 || w < 1 || h < 1) {
		stbi_image_free(px);
		snprintf(err, errlen, "%s: %dx%d with %d channels", path, w, h,
			 comp);
		return NULL;
	}
	out = malloc((size_t)3 * side * side * sizeof(float));
	if (!out) {
		stbi_image_free(px);
		snprintf(err, errlen, "%u x %u will not allocate", side, side);
		return NULL;
	}
	resize_bilinear(px, w, h, comp, out, (int)side, (int)side);
	stbi_image_free(px);
	return out;
}
