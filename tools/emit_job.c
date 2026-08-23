// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/* Print the COMPLETE job stream on the host, so a change to it is checked
 * before it costs a board round. Round 145 lost one to a silent edit: a fix
 * that never landed in the source read as applied because the binary that
 * carried the other two fixes was newer. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "charsiu.h"

int main(int argc, char **argv)
{
	struct charsiu_job job = { 0 };
	uint64_t buf[512];
	size_t n, i;

	job.cbuf_window = (unsigned)charsiu_cbuf_window();
	job.mm.m = argc > 1 ? (unsigned)atoi(argv[1]) : 1;
	job.mm.k = argc > 2 ? (unsigned)atoi(argv[2]) : 64;
	job.mm.n = argc > 3 ? (unsigned)atoi(argv[3]) : 64;
	job.mm.wdtype = CHARSIU_INT8;
	job.mm.adtype = CHARSIU_INT8;
	if (argc > 4 && !strcmp(argv[4], "int4")) job.mm.wdtype = CHARSIU_INT4;
	if (argc > 5 && !strcmp(argv[5], "fp16")) job.mm.adtype = CHARSIU_FP16;
	job.input_scale = 0.02f; job.weight_scale = 0.01f; job.output_scale = 0.25f;
	job.input_zero_point = 128; job.weight_zero_point = 128;
	job.acc_out = getenv("CHARSIU_ACC_OUT") != NULL;
	job.output_zero_point = 128;
	job.input_addr = 0x1000; job.weight_addr = 0x3000;
	job.output_addr = 0x5000; job.coef_addr = 0x7000;

	n = charsiu_emit_job(&job, buf, sizeof(buf) / sizeof(buf[0]));
	for (i = 0; i < n; i++)
		printf("CS %02zu t=%04x r=%04x v=%08x\n", i,
		       (unsigned)(buf[i] >> 48), (unsigned)(buf[i] & 0xffff),
		       (unsigned)((buf[i] >> 16) & 0xffffffff));
	return n ? 0 : 1;
}
