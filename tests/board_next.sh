#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
#
# THE NEXT BOARD SESSION, AS ONE COMMAND.
#
# Five things were built on a machine with no NPU on 2026-09-02 and none of
# them has a number from the hardware. This runs the questions in the order
# that buys the most per minute and stops at the first thing that fails or
# wedges the block, so the decisions below can be read off one log.
#
#   sh /opt/charsiu/board_next.sh            run everything
#   sh /opt/charsiu/board_next.sh --from 3   resume after a reboot at step 3
#   sh /opt/charsiu/board_next.sh --plan     print the steps and do nothing
#
#   step  runs                          answers                      decides
#   1     board_verify 2 4 5 7          did input reuse, 80 rows and  keep or revert
#                                       whisper grouping change any   today's commits
#                                       text, answer or transcript
#   2     board_verify 21               how much of the 130 us call   spin on by default?
#                                       floor is wake latency         touch cpuidle?
#   3     install the test kernel       (the two rocket patches: the  -- then REBOOT and
#         by tag, keep the old one      domain kept attached, and the    --from 4
#                                       completion in the hardirq)
#   4     board_verify 2 and 21 again   what the patches take off a   50 us is decode
#                                       call, text held               parity; v12 or not
#   5     board_verify 20               what an m=4 pass costs        --spec on by default
#                                       against a decode step         on Qwen3?
#   6     board_verify 19               which quantity bounds the     an int8 guard, and
#                                       height axis                   whisper medium/large
#
# ⚠ STEP 3 REPLACES THE KERNEL. charsiu-install keeps the one it replaces as
# /boot/Image.previous and board-kernel-revert.sh puts it back; this board has
# no boot menu, so a kernel that does not boot needs a serial console or the
# SD card.
# ⚠ A WEDGED NPU NEEDS A REBOOT. board_verify says so when it happens; reboot
# and come back with --from N.
set -u
BIN=${CHARSIU_BIN_DIR:-/opt/charsiu}
OUT=${CHARSIU_BOARD_DIR:-/root/charsiu-board}
RAW=https://raw.githubusercontent.com/${CHARSIU_REPO:-gahingwoo/linux-rk3576-npu}/main/rfc-send-v12/attach-once
FROM=1; PLAN=0
while [ $# -gt 0 ]; do
	case "$1" in
	--from) FROM=$2; shift ;;
	--plan) PLAN=1 ;;
	*) echo "usage: board_next.sh [--from N] [--plan]"; exit 2 ;;
	esac
	shift
done
[ "$PLAN" = 1 ] || mkdir -p "$OUT"
SUM=$OUT/next-summary.txt
say() { printf '\n=========== %s ===========\n' "$*"; }
note() { printf '%s\n' "$*" | tee -a "$SUM"; }
step() { [ "$PLAN" = 1 ] && { echo "step $1: $2"; return 1; }; [ "$1" -ge "$FROM" ] || return 1; say "step $1: $2"; return 0; }
verify() {
	sh "$BIN/board_verify.sh" "$1" 2>&1 | tee "$OUT/next-step$2.txt"
	if grep -q "THE NPU WEDGED" "$OUT/next-step$2.txt"; then
		note "step $2: the NPU wedged. Reboot, then: sh $BIN/board_next.sh --from $2"
		exit 3
	fi
	if grep -q "FAILURE(S)" "$OUT/next-step$2.txt"; then
		note "step $2: board_verify reported failures; read $OUT/next-step$2.txt before going on"
		return 1
	fi
	return 0
}
call_line() { grep -o "us a call = [0-9.]* + [0-9.]* a task + [0-9.]* a MB" "$1" | head -1; }

[ "$PLAN" = 1 ] || [ "$FROM" -gt 1 ] || : >"$SUM"
[ "$PLAN" = 1 ] || note "board_next $(date '+%F %T') on $(uname -r), from step $FROM"

if step 1 "correctness under today's three runtime changes"; then
	verify "2 4 5 7" 1 || { note "STOP: today's runtime commits changed an answer. Revert before measuring anything."; exit 1; }
	note "step 1: text, answer and transcript unchanged"
	grep -h "input reused" "$OUT"/verify-*.txt 2>/dev/null | tail -1 | sed 's/^/  /' | tee -a "$SUM"
fi

if step 2 "the call floor: wake latency alone"; then
	verify 21 2 || true
	note "step 2: $(grep -E 'cpu-sleep|spin ' "$OUT/next-step2.txt" | sed 's/^ *//' | tr '\n' ';')"
fi

if step 3 "the call floor: the two rocket patches, as a whole kernel"; then
	# ⚠⚠ rocket IS BUILT IN on this board (kernel/npu.fragment:
	# CONFIG_DRM_ACCEL_ROCKET=y, because a module probes after the
	# late_initcall that cuts vdd_npu_s0), so there is no module to swap.
	# The two patches come as a whole Image, built with the board's own
	# config and the same release string, published as a GitHub PRE-release
	# that `releases/latest` never returns, and installed on purpose by tag.
	# charsiu-install keeps the kernel it replaces as Image.previous;
	# board-kernel-revert.sh puts it back.
	#
	# ⚠ THIS BOARD HAS NO BOOT MENU. If the new Image does not boot, the way
	# back is a serial console or the SD card, not this script.
	TAG=${KERNEL_TAG:-kernel-7.2.0-rc5-next-20260730-attach-once}
	curl -fsSL -o "$OUT/board-kernel-revert.sh" "$RAW/board-kernel-revert.sh" || true
	# ⚠ THE INSTALLER IS NOT IN $BIN. charsiu-update keeps the source tree
	# it built from in one of these and runs scripts/charsiu-install.sh out
	# of it; the first version of this step guessed $BIN/charsiu-install.sh,
	# sh could not open it, and the `if` read tee's exit status and said
	# "installed, reboot now" about a kernel that had not been touched.
	INST=""
	for d in "${XDG_CACHE_HOME:-$HOME/.cache}/charsiu/src" "$HOME/charsiu" /opt/charsiu/src "$BIN/src"; do
		[ -f "$d/scripts/charsiu-install.sh" ] && { INST=$d/scripts/charsiu-install.sh; break; }
	done
	if [ -z "$INST" ]; then
		curl -fsSL -o "$OUT/charsiu-install.sh" \
		    "https://raw.githubusercontent.com/${CHARSIU_SRC_REPO_PATH:-gahingwoo/charsiu}/dev/scripts/charsiu-install.sh" \
		    && INST=$OUT/charsiu-install.sh
	fi
	[ -n "$INST" ] || { note "step 3: no charsiu-install.sh anywhere and no network to fetch one"; exit 1; }
	grep -q CHARSIU_KERNEL_TAG "$INST" || { note "step 3: $INST predates CHARSIU_KERNEL_TAG -- run 'charsiu update dev' first"; exit 1; }
	CHARSIU_KERNEL_TAG=$TAG CTUI_ASSUME=yes sh "$INST" --kernel >"$OUT/next-step3-install.txt" 2>&1
	rc=$?
	cat "$OUT/next-step3-install.txt"
	# ⚠ SUCCESS IS THE INSTALLER'S OWN WORD, not an exit status through a
	# pipe. It prints "Kernel installed." only after Image, dtb and modules
	# are in place and the previous kernel is kept.
	if [ "$rc" = 0 ] && grep -q "Kernel installed" "$OUT/next-step3-install.txt"; then
		note "step 3: installed kernel $TAG; the previous one is /boot/Image.previous"
		note "  REBOOT NOW, then: sh $BIN/board_next.sh --from 4"
		note "  (to go back: sh $OUT/board-kernel-revert.sh, then reboot)"
		exit 0
	fi
	note "step 3: the kernel install did NOT complete (exit $rc; see next-step3-install.txt). /boot was not changed. Do not reboot for this."
	exit 1
fi

if step 4 "the call floor under the new kernel"; then
	# ⚠ uname -r IS THE SAME STRING FOR EVERY KERNEL IN THE BISECT, so say
	# which Image this is by its hash: 370c4612 both patches, 7773c43b
	# attach-only, e70a8b6a control (no patches), and the August release
	# is whatever Image.previous hashes to.
	note "step 4: running on $(uname -r), Image $(sha256sum /boot/Image 2>/dev/null | cut -c1-8), rocket $(dmesg | grep -c 'Rockchip NPU core') core lines"
	verify 2 4a || { note "step 4: TEXT CHANGED under the new kernel -- revert: sh $OUT/board-kernel-revert.sh, reboot"; exit 1; }
	verify 21 4b || true
	note "step 4: before   $(call_line "$OUT/next-step2.txt")"
	note "step 4: patched  $(call_line "$OUT/next-step4b.txt")"
	note "  (150 calls a token on Qwen3: every 10 us off 'a call' is 1.5 ms a token; 50 us is parity)"
fi

if step 5 "speculative decoding: the price of a pass"; then
	verify 20 5 || true
	grep -E "^ *(plain|--spec)" "$OUT/next-step5.txt" | sed 's/^ */step 5: /' | tee -a "$SUM"
	note "  (speed up = tok/pass over pass cost in decode steps; break-even on Qwen3 is 2.29)"
fi

if step 6 "the height axis: which quantity is the bound"; then
	verify 19 6 || true
	grep -E "exact|WRONG" "$OUT/next-step6.txt" | sed 's/^ */step 6: /' | tee -a "$SUM"
fi

[ "$PLAN" = 1 ] && exit 0
say "summary"
cat "$SUM"
echo "logs: $OUT/next-step*.txt"
