// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * May the two NPU cores overlap? The table, walked on a desk.
 *
 * The board answered the physical question on 2026-09-04: at 786 MHz the
 * overlap corrupts one word in a few thousand rows at 750 mV and is exact at
 * 800, and at 594 MHz it is exact at 750. What ships is a comparison against
 * the vendor's own worst-bin voltage for each of its OPP steps, and THAT is
 * what this checks -- including the two answers that cost a board round to
 * get right: an unreadable rail is unsafe, and a clock nobody can read is
 * assumed to be mainline's 800 MHz rather than something kind.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/overlap.h"

static int bad, n;

static void expect(const char *hz, const char *uv, int want, const char *what)
{
	char why[96];
	int got;

	n++;
	if (hz) setenv("CHARSIU_NPU_CLK_HZ", hz, 1); else unsetenv("CHARSIU_NPU_CLK_HZ");
	if (uv) setenv("CHARSIU_NPU_RAIL_UV", uv, 1); else unsetenv("CHARSIU_NPU_RAIL_UV");
	got = charsiu_npu_overlap_ok(why, sizeof(why));
	if (got != want) {
		printf("  overlap guard: %s: expected %s, got %s (%s)\n", what,
		       want ? "overlap" : "serial", got ? "overlap" : "serial", why);
		bad++;
	}
}

int main(void)
{
	/* the vendor's rk3576 npu_opp_table, worst leakage bin */
	expect("600000000", "725000", 1, "600 MHz at its own 725 mV");
	expect("600000000", "724000", 0, "600 MHz a millivolt short");
	expect("700000000", "775000", 1, "700 MHz at 775 mV");
	expect("700000000", "750000", 0, "700 MHz at 750 mV");
	expect("800000000", "800000", 1, "800 MHz at 800 mV");
	expect("900000000", "850000", 1, "900 MHz at 850 mV");
	expect("900000000", "800000", 0, "900 MHz at 800 mV");

	/* the board, before and after the DTB change */
	expect("786431991", "750000", 0, "mainline as shipped: 786 MHz at 750 mV");
	expect("786431991", "800000", 1, "the 800 mV DTB");
	expect("786431991", "850000", 1, "the 850 mV DTB");
	expect("594000000", "750000", 1, "the 594 MHz DTB");

	/* what cannot be read */
	expect(NULL, "750000", 0, "no clock reading: 800 MHz assumed, 750 mV is short");
	expect(NULL, "800000", 1, "no clock reading at 800 mV");
	expect("786431991", "-1", 0, "an unreadable rail is never safe");
	expect("1000000000", "875000", 0, "above the vendor's table, at any voltage");

	printf("  overlap guard: %d of %d cases wrong\n", bad, n);
	return bad != 0;
}
