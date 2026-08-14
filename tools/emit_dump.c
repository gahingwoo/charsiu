// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/* Print the register stream charsiu would emit for one matmul, so it can be
 * diffed against the vendor's for the same shape without a board. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu.h"

int main(int argc, char **argv)
{
	struct charsiu_matmul mm = { 1, 2048, 1024, CHARSIU_INT4, CHARSIU_FP16 };
	uint64_t buf[256];
	size_t n, i;

	if (argc > 3) {
		mm.m = (unsigned)atoi(argv[1]);
		mm.k = (unsigned)atoi(argv[2]);
		mm.n = (unsigned)atoi(argv[3]);
	}
	if (argc > 4) {                 /* weight dtype */
		if (!strcmp(argv[4], "int8")) mm.wdtype = CHARSIU_INT8;
		else if (!strcmp(argv[4], "fp16")) mm.wdtype = CHARSIU_FP16;
	}
	if (argc > 5) {                 /* activation dtype */
		if (!strcmp(argv[5], "int8")) mm.adtype = CHARSIU_INT8;
		else if (!strcmp(argv[5], "int4")) mm.adtype = CHARSIU_INT4;
	}

	n = charsiu_emit_matmul(&mm, buf, sizeof(buf) / sizeof(buf[0]));
	if (!n) {
		fprintf(stderr, "emit failed\n");
		return 1;
	}
	for (i = 0; i < n; i++) {
		uint64_t e = buf[i];
		printf("%04x %04x %08x\n", (unsigned)(e >> 48),
		       (unsigned)(e & 0xffff), (unsigned)((e >> 16) & 0xffffffff));
	}
	return 0;
}
