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
#   3     swap rocket.ko, then 2 and 21 what "keep the domain         50 us is decode
#         then rocket-hardirq.ko, same   attached" takes off a call,   parity; which of
#                                       and the hardirq on top of it  the two go to v12
#   4     board_verify 20               what an m=4 pass costs        --spec on by default
#                                       against a decode step         on Qwen3?
#   5     board_verify 19               which quantity bounds the     an int8 guard, and
#                                       height axis                   whisper medium/large
#
# ⚠ STEP 3 CHANGES THE RUNNING KERNEL MODULE. The original is kept beside it
# and `board-swap-rocket-ko.sh --revert` puts it back. The module is checked
# against uname -r before anything is touched.
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

if step 3 "the call floor: keep the IOMMU domain attached (rocket.ko)"; then
	cd "$OUT" || exit 1
	for f in rocket.ko rocket.ko.sha256 board-swap-rocket-ko.sh; do
		curl -fsSL -o "$f" "$RAW/$f" || { note "step 3: could not fetch $f from $RAW"; exit 1; }
	done
	sed 's#  .*/#  #' rocket.ko.sha256 | sha256sum -c - || { note "step 3: rocket.ko checksum mismatch"; exit 1; }
	pkill -f charsiu_serve 2>/dev/null || true
	sh board-swap-rocket-ko.sh rocket.ko 2>&1 | tee "$OUT/next-step3-swap.txt"
	if ! grep -q "reloaded" "$OUT/next-step3-swap.txt"; then
		note "step 3: the module did not reload (see next-step3-swap.txt). If it says next boot, reboot and --from 3."
		exit 1
	fi
	verify 2 3a || { note "step 3: TEXT CHANGED under the new module -- revert: sh $OUT/board-swap-rocket-ko.sh --revert"; exit 1; }
	verify 21 3b || true
	note "step 3: before        $(call_line "$OUT/next-step2.txt")"
	note "step 3: attach-once   $(call_line "$OUT/next-step3b.txt")"
	# the second piece: the completion handled in the hardirq, on top of the first
	for f in rocket-hardirq.ko rocket-hardirq.ko.sha256; do
		curl -fsSL -o "$f" "$RAW/$f" || { note "step 3: could not fetch $f"; exit 1; }
	done
	sed 's#  .*/#  #' rocket-hardirq.ko.sha256 | sha256sum -c - || { note "step 3: rocket-hardirq.ko checksum mismatch"; exit 1; }
	sh board-swap-rocket-ko.sh rocket-hardirq.ko 2>&1 | tee "$OUT/next-step3-swap2.txt"
	if grep -q "reloaded" "$OUT/next-step3-swap2.txt"; then
		verify 2 3c || { note "step 3: TEXT CHANGED under rocket-hardirq.ko -- revert"; exit 1; }
		verify 21 3d || true
		note "step 3: +hardirq      $(call_line "$OUT/next-step3d.txt")"
	else
		note "step 3: rocket-hardirq.ko did not reload; the attach-once module stays in"
	fi
	note "  (150 calls a token on Qwen3: every 10 us off 'a call' is 1.5 ms a token; 50 us is parity)"
	note "  (the module left installed is the last one that reloaded; --revert restores the original)"
fi

if step 4 "speculative decoding: the price of a pass"; then
	verify 20 4 || true
	grep -E "^ *(plain|--spec)" "$OUT/next-step4.txt" | sed 's/^ */step 4: /' | tee -a "$SUM"
	note "  (speed up = tok/pass over pass cost in decode steps; break-even on Qwen3 is 2.29)"
fi

if step 5 "the height axis: which quantity is the bound"; then
	verify 19 5 || true
	grep -E "exact|WRONG" "$OUT/next-step5.txt" | sed 's/^ */step 5: /' | tee -a "$SUM"
fi

[ "$PLAN" = 1 ] && exit 0
say "summary"
cat "$SUM"
echo "logs: $OUT/next-step*.txt"
