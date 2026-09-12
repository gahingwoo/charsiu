#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# TTFT against prompt length, so the fixed cost is MEASURED rather than
# extrapolated from two points.
#
# ⚠⚠ WHY THIS EXISTS. r391 separated charsiu's prefill into a per-token rate
# and a fixed cost by timing two prompt lengths and fitting a line through
# them. Two points always fit a line. What that fit assumed -- that TTFT is
# linear in the token count -- is only true inside one chunk:
# `llama_prefill_chunk_cap()` is 163840/widest_k_slice, which is 160 tokens for
# Llama-3.2-1B at KMAX 1024, and a prompt longer than that is two dispatches.
# r391's points were 16 and 79, both inside one chunk, so its fit was safe by
# luck rather than by design.
#
# ⚠⚠ AND THE REAL TRAP IS NOT THE CAP, IT IS THE ODD TOKEN. `prefill_width()`
# does `w &= ~1`, so a prompt of 79 tokens runs as a batched 78 plus ONE token
# through `llama_forward` -- and that token-loop step is inside the prompt
# timer. r391's two points were 16 (no leftover) and 79 (one leftover), so its
# slope carried a whole decode step that its intercept did not, in one arm
# only. The runtime says so out loud -- "prompt batched for 78 of 79 tokens,
# the rest a token at a time" -- and nothing was reading that line.
#
# So every point here prints its batched width and its leftover, and the
# marginal slope is computed only between points that BOTH have no leftover.
#
# This also sweeps far enough to cross the cap, so the step is visible if it is
# there and the linear region is identified rather than assumed.
#
# 🔑 THE NUMBER THIS IS FOR. r391 put charsiu's fixed cost at 109 ms at maximum
# CPU against the vendor's 82.2 ms -- the one half of prefill where their
# runtime is ahead of ours, and a part of the prompt that NOTHING in this tree
# has ever decomposed. board_prefill_stages.sh measures milliseconds a ROW and
# explicitly excludes staging; the intercept is neither of those.
#
# ⚠ The token count is READ BACK from the runtime, never assumed from the text.
# A prompt built by repeating a clause does not tokenise to a round number, and
# the whole point of the x axis is that it is exact.
#
#   sh tests/board_ttft_curve.sh
set -u

REPEAT=${TC_REPEAT:-3}
RUN=${CHARSIU_RUN:-/root/charsiu_run_maxn}
M=${CHARSIU_MODEL:-/opt/charsiu/models/Llama-3.2-1B-Instruct-Q4_0.gguf}
# ⚠ MAXN is the default since the r391 round, but every other board script in
# this tree still spells the environment out and a round that reads differently
# from its neighbours is a round nobody can compare.
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"

# ⚠ PIN THE CLOCK. r391 measured the intercept moving 143.3 -> 109.0 ms across
# the governor, which is larger than anything this sweep is looking for.
for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo userspace > "$p/scaling_governor" 2>/dev/null || true
	tr ' ' '\n' < "$p/scaling_available_frequencies" | grep -v '^$' | sort -n \
		| tail -1 > "$p/scaling_setspeed" 2>/dev/null || true
done
sleep 1

echo "== TTFT against prompt length"
echo "   boot      $(cat /proc/sys/kernel/random/boot_id)"
echo "   npu clk   $(cat /sys/kernel/debug/clk/clk_rknn_dsu0/clk_rate 2>/dev/null) Hz"
echo "   cpu       $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq)/$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq) kHz, governor $(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_governor)"
echo "   binary    $RUN  $(ls -l --full-time "$RUN" 2>/dev/null | awk '{print $6}')"
echo "   chunk cap 160 tokens for this model at KMAX 1024 -- the sweep crosses it"
echo "   repeats   $REPEAT per point, one warm-up discarded"
echo

CLAUSE="A written record can be copied, checked against other records, and read by someone who never met the person who wrote it. "

mid() { printf '%s\n' $1 | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$0} END{if(NR%2)printf "%s\n",a[(NR+1)/2]; else printf "%.1f\n",(a[NR/2]+a[NR/2+1])/2}'; }
lo()  { printf '%s\n' $1 | tr ' ' '\n' | grep -v '^$' | sort -n | head -1; }
hi()  { printf '%s\n' $1 | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1; }

printf '   %7s %8s %5s  %9s %14s   %s\n' tokens batched left 'TTFT ms' range 'marginal ms/tok'
printf '   %7s %8s %5s  %9s %14s   %s\n' ------ ------- ---- ------- ----- ---------------

PREV_N=; PREV_T=
for REPS in ${TC_REPS:-1 2 4 8 12 18 24 34}; do
	P=""; i=0
	while [ $i -lt "$REPS" ]; do P="$P$CLAUSE"; i=$((i+1)); done

	# The warm-up doubles as the width read. ⚠ There is NO env var for
	# this: charsiu_diag() is a static that is ON unless the binary calls
	# charsiu_diag_quiet(), so the widths line is already on stderr and the
	# only thing needed is to stop discarding it. That line is the only
	# place the leftover token is visible.
	W=$(env $E "$RUN" "$M" -p "$P" -n 1 --ignore-eos -c 1024 -t 4 2>&1 >/dev/null \
	    | grep -oE 'batched for [0-9]+ of [0-9]+|widths [0-9]+x[0-9]+' | head -2 | tr '\n' ' ')
	BATCHED=$(echo "$W" | grep -oE 'batched for [0-9]+' | grep -oE '[0-9]+' | head -1)

	T=; N=; n=0
	while [ $n -lt "$REPEAT" ]; do
		n=$((n+1))
		O=$(env $E "$RUN" "$M" -p "$P" -n 1 --ignore-eos -c 1024 -t 4 2>/dev/null | grep '^\[load')
		[ -n "$O" ] || continue
		N=$(echo "$O" | sed 's/.*prompt \([0-9]*\) tok in.*/\1/')
		T="$T $(echo "$O" | sed 's/.*prompt [0-9]* tok in \([0-9]*\) ms.*/\1/')"
	done
	[ -n "$T" ] || { echo "   ${REPS} clauses: NO OUTPUT"; continue; }
	MT=$(mid "$T")
	# no "batched for X of Y" line means nothing was left over
	[ -n "${BATCHED:-}" ] || BATCHED=$N
	LEFT=$((N - BATCHED))

	# ⚠ a marginal slope across a point with a leftover token is a slope
	# with a decode step hidden in it. Say so rather than printing it.
	MARG=-
	if [ -n "$PREV_N" ] && [ "$N" != "$PREV_N" ]; then
		if [ "$LEFT" = 0 ] && [ "${PREV_LEFT:-1}" = 0 ]; then
			MARG=$(awk "BEGIN{printf \"%.3f\", ($MT-$PREV_T)/($N-$PREV_N)}")
		else
			MARG="(leftover)"
		fi
	fi
	printf '   %7s %8s %5s  %9s %6s..%-6s   %s\n' \
		"$N" "$BATCHED" "$LEFT" "$MT" "$(lo "$T")" "$(hi "$T")" "$MARG"
	PREV_N=$N; PREV_T=$MT; PREV_LEFT=$LEFT
done

echo
echo "restoring schedutil"
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil > "$p/scaling_governor" 2>/dev/null || true; done
