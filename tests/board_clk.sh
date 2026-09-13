# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# ⚠⚠⚠ THE NPU CLOCK, OR A REFUSAL. Sourced by every board script.
#
# Ten scripts read the rate as `$(cat /sys/kernel/debug/clk/... 2>/dev/null)`
# and print it into their header. On 2026-09-14 the board rebooted mid-session
# and came up with debugfs NOT MOUNTED, so that read produced an empty string:
# the header said "npu clk   Hz" and the round carried on and produced a full
# TTFT ladder. A missing clock reads exactly like a clock of nothing.
#
# The rate is not decoration. It is in the DT, it cannot be set from sysfs, and
# every number this project compares across rounds assumes 594 MHz. A round at
# an unknown rate is not a slower round, it is an unusable one.
#
# ⚠⚠ AND THE REFUSAL HAS TO BE OUTSIDE THE SUBSTITUTION. The first version of
# this put `exit 1` inside npu_clk and every caller used it as `$(npu_clk)` --
# which is a SUBSHELL, so the exit ended the substitution and the script
# printed a blank clock and carried on. Exactly the hole it was written to
# close, with a refusal in it that could not fire. Callers now do
#
#     NPUCLK=$(npu_clk) || exit 1
#
# because an assignment from a substitution does carry its exit status.
#
#   npu_clk        prints the rate in Hz, or returns 1 having said why
#   CHARSIU_NPU_CLK_ANY=1  prints what it finds and carries on
#   CHARSIU_NPU_CLK_PATH=...  point it elsewhere, which is how the refusal
#                             below is exercised: a check nobody has seen fire
#                             is not a check
npu_clk() {
	_c=${CHARSIU_NPU_CLK_PATH:-/sys/kernel/debug/clk/clk_rknn_dsu0/clk_rate}
	if [ ! -r "$_c" ]; then
		mount -t debugfs none /sys/kernel/debug 2>/dev/null
	fi
	_r=$(cat "$_c" 2>/dev/null)
	if [ -z "$_r" ] && [ -z "${CHARSIU_NPU_CLK_PATH:-}" ]; then
		_r=$(cat /sys/kernel/debug/clk/aclk_rknn_root/clk_rate 2>/dev/null)
	fi
	if [ -z "$_r" ]; then
		if [ -n "${CHARSIU_NPU_CLK_ANY:-}" ]; then
			echo "unknown"
			return 0
		fi
		echo "" >&2
		echo "⛔ THE NPU CLOCK CANNOT BE READ, and this round would" >&2
		echo "   have printed a blank where the rate goes." >&2
		echo "" >&2
		echo "   /sys/kernel/debug/clk/clk_rknn_dsu0/clk_rate is not" >&2
		echo "   readable. debugfs is usually not mounted after a" >&2
		echo "   reboot; 'mount -t debugfs none /sys/kernel/debug'" >&2
		echo "   fixes it and this script already tried." >&2
		echo "" >&2
		echo "   Every cross-round number here assumes 594 MHz, which" >&2
		echo "   is set in the device tree and cannot be read back any" >&2
		echo "   other way. CHARSIU_NPU_CLK_ANY=1 runs anyway." >&2
		return 1
	fi
	echo "$_r"
}

# the boot this round ran on, so two rounds can be told apart. The board drifts
# about 3% between boots and r393 measured that on a ladder; a table that does
# not carry its boot cannot be compared with one that does.
npu_boot() { cat /proc/sys/kernel/random/boot_id 2>/dev/null; }
