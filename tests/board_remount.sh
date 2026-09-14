#!/bin/sh
#
# What a reboot takes away, and how to put it back. Run this first, on the
# board, after every boot.
#
# ⚠ THE BOARD LOSES TWO MOUNTS ON EVERY BOOT, and both failures are silent in
# the way this tree keeps meeting:
#
#   /opt/vendor        seven of the nine models and the vendor runtime. Gone,
#                      it is an EMPTY DIRECTORY -- so a round finds two models
#                      instead of nine, compares them, and prints "0 differing".
#   /sys/kernel/debug  the NPU clock. Gone, board_clk.sh refuses (it is the one
#                      that learned); anything that reads the node by hand
#                      prints a blank or UNREADABLE and carries on.
#
# ⚠ AND NOTHING WRITTEN DOWN SAID WHICH DEVICE. The note said "remount
# /opt/vendor by hand" for weeks without saying from what, so every reboot cost
# a partition hunt. It is /dev/mmcblk0p3 -- the 11.7 GB one, no label, holding
# bin/ lib/ model/ models/ and rknpu.ko. p1 is BOOT, p2 is rootfs.
#
# Power is the user's. This script mounts and reports; it never reboots, resets
# or cycles anything.
set -u
VDEV="${CHARSIU_VENDOR_DEV:-/dev/mmcblk0p3}"

if [ ! -r /sys/kernel/debug/clk ]; then
	mount -t debugfs none /sys/kernel/debug 2>/dev/null || true
fi
if [ -z "$(ls -A /opt/vendor 2>/dev/null)" ]; then
	mkdir -p /opt/vendor
	mount "$VDEV" /opt/vendor || {
		echo "⛔ could not mount $VDEV at /opt/vendor" >&2
		echo "   set CHARSIU_VENDOR_DEV if the partition moved;" >&2
		echo "   /proc/partitions lists what this board has." >&2
		exit 1
	}
fi

# Report, and make each line falsifiable rather than reassuring.
N=$(ls /opt/vendor/models/*.gguf 2>/dev/null | wc -l)
C=$(cat /sys/kernel/debug/clk/clk_rknn_dsu0/clk_rate 2>/dev/null)
echo "boot      $(cat /proc/sys/kernel/random/boot_id)"
echo "uptime    $(cut -d' ' -f1 /proc/uptime) s"
echo "vendor    $N gguf in /opt/vendor/models"
echo "npu clk   ${C:-UNREADABLE} Hz"
echo "governor  $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor 2>/dev/null)"

rc=0
# ⚠ SEVEN IS THE NUMBER, and a floor that is never checked is not a floor.
[ "$N" -ge 7 ] || { echo "⛔ expected 7 gguf, found $N" >&2; rc=1; }
[ -n "$C" ] || { echo "⛔ the NPU clock is still unreadable" >&2; rc=1; }
exit $rc
