// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
#ifndef CHARSIU_OVERLAP_H
#define CHARSIU_OVERLAP_H
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * 🏁 2026-09-04: THE OVERLAP FAULT WAS THE NPU'S VOLTAGE MARGIN, NOT THE
 * OVERLAP. At width 24 with both cores in flight, core 1 wrote one word of
 * a row wrong -- the right value plus 1024, or a few bits around bit 10 of
 * the accumulator -- one row in a few thousand, on the same board, same
 * kernel, same binary. Mainline runs the NPU clock (clk_rknn_dsu0, which is
 * compute AND the shared CBUF) at 786.43 MHz off the audio PLL with no OPP,
 * and the NPU rail at whatever U-Boot left, 750 mV. The vendor's own OPP
 * table asks 800 mV of its 800 MHz step at the worst leakage bin, and 725
 * only at 600 MHz. Four DTBs, same probe, KMAX 1024, 4 passes of 5400 rows:
 *
 *   786 MHz, 750 mV (mainline as shipped)   11 to 25 wrong words a pass
 *   594 MHz, 750 mV                          0, 0, 0, 0   (10% slower)
 *   786 MHz, 800 mV                          0, 0, 0, 0   (full speed)
 *   786 MHz, 850 mV                          0, 0, 0, 0
 *
 * So the two cores may overlap when the rail and the clock sit inside the
 * vendor's envelope, and must not otherwise. This reads both from sysfs
 * (the rail from /sys/class/regulator, the clock from debugfs when it is
 * mounted; unknown clock is taken as 800 MHz, the mainline default, and an
 * unknown rail as unsafe) and decides. CHARSIU_NPU_BATCH_PARALLEL=1 forces
 * the overlap and =0 forbids it, whatever the board says.
 */
static long sysfs_long(const char *path)
{
	FILE *f = fopen(path, "r");
	long v = -1;

	if (!f)
		return -1;
	if (fscanf(f, "%ld", &v) != 1)
		v = -1;
	fclose(f);
	return v;
}

/*
 * ⚠ WALK THE DIRECTORY, DO NOT GUESS THE NUMBERS. regulator.N is numbered by
 * probe order and nothing bounds N: a loop to 64 finds the rail on this board
 * and would silently return "unreadable" -- which this code reads as unsafe,
 * so a board that is fine would serialise for a reason nobody could see.
 */
static long npu_rail_uv(void)
{
	/* ⚠ THE PROBE HATCH. A guard whose inputs cannot be set is a guard that
	 * can only be tested by rebooting the board into a different voltage.
	 * CHARSIU_NPU_RAIL_UV and CHARSIU_NPU_CLK_HZ replace the two readings,
	 * so tests/overlap_guard.c walks the vendor's whole table on a desk and
	 * the board can ask for the unsafe branch without a reboot. */
	const char *env = getenv("CHARSIU_NPU_RAIL_UV");
	DIR *d;

	if (env && *env)
		return atol(env);
	d = opendir("/sys/class/regulator");
	struct dirent *e;
	long uv = -1;

	if (!d)
		return -1;
	while (uv < 0 && (e = readdir(d))) {
		char path[320], name[64];
		FILE *f;

		if (e->d_name[0] == '.')
			continue;
		snprintf(path, sizeof(path), "/sys/class/regulator/%s/name", e->d_name);
		f = fopen(path, "r");
		if (!f)
			continue;
		if (fscanf(f, "%63s", name) == 1 && !strcmp(name, "vdd_npu_s0")) {
			snprintf(path, sizeof(path),
				 "/sys/class/regulator/%s/microvolts", e->d_name);
			uv = sysfs_long(path);
		}
		fclose(f);
	}
	closedir(d);
	return uv;
}

static long npu_clk_hz(void)
{
	const char *e = getenv("CHARSIU_NPU_CLK_HZ");
	long hz;

	if (e && *e)
		return atol(e);
	hz = sysfs_long("/sys/kernel/debug/clk/clk_rknn_dsu0/clk_rate");
	return hz > 0 ? hz : 800000000;   /* debugfs not mounted: mainline's default */
}

/* the vendor's worst-bin voltage for each NPU OPP step (rk3576 npu_opp_table, L0) */
static int overlap_safe(char *why, size_t n)
{
	static const struct { long hz; long uv; } opp[] = {
		{ 600000000, 725000 }, { 700000000, 775000 },
		{ 800000000, 800000 }, { 900000000, 850000 },
	};
	long uv = npu_rail_uv(), hz = npu_clk_hz();
	long need = -1;

	for (unsigned i = 0; i < sizeof(opp) / sizeof(opp[0]); i++)
		if (hz <= opp[i].hz) {
			need = opp[i].uv;
			break;
		}
	if (uv < 0) {
		snprintf(why, n, "the NPU rail is not readable, %ld MHz", hz / 1000000);
		return 0;
	}
	if (need < 0) {
		snprintf(why, n, "%ld MHz is above the vendor's table", hz / 1000000);
		return 0;
	}
	snprintf(why, n, "%ld MHz at %ld mV, the vendor asks %ld", hz / 1000000,
		 uv / 1000, need / 1000);
	return uv >= need;
}

static char overlap_why[96];

/* the decision alone, with the two readings taken as given: 1 = may overlap */
int charsiu_npu_overlap_ok(char *why, size_t n)
{
	char buf[96];

	if (!why) { why = buf; n = sizeof(buf); }
	return overlap_safe(why, n);
}

#endif /* CHARSIU_OVERLAP_H */
