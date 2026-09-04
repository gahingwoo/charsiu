#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# Does packing a K slice out of the caller's matrix beat gathering it first?
#
# ⚠⚠ ONE BINARY, ONE SESSION, THE PERFORMANCE GOVERNOR. Three phase 9 runs
# measured this by rebuilding between them and disagreed by more than the
# change: Phi-3.5 packed 4384, 6583 and 3158 ms on paths that were the same
# twice over, because phase 9 runs under ondemand and packing is CPU work.
# So the arms are an environment variable, they alternate, and each is run
# REPEATS times so the spread is visible next to the difference.
#
#   CHARSIU_AB_MODEL=  a gguf (default: the first Phi or Qwen found)
#   CHARSIU_AB_WIDTH=80    the batch width to probe
#   CHARSIU_AB_REPEATS=3
#   CHARSIU_AB_MAXT=40     tensors a pass, so a repeat is quick
set -u
W=${CHARSIU_AB_WIDTH:-80}
N=${CHARSIU_AB_REPEATS:-3}
RUN=${CHARSIU_RUN_BIN:-}
[ -n "$RUN" ] || for d in /opt/charsiu ./build .; do [ -x "$d/charsiu_run" ] && { RUN=$d/charsiu_run; break; }; done
[ -n "${RUN:-}" ] || { echo "no charsiu_run" >&2; exit 1; }
M=${CHARSIU_AB_MODEL:-}
if [ -z "$M" ]; then
	for p in '*Phi-3*Q4_0*.gguf' '*Qwen2.5*Q4_0*.gguf' '*Q4_0*.gguf'; do
		for d in "$HOME/.charsiu/models" /opt/charsiu/models; do
			for f in $d/$p; do [ -r "$f" ] && { M=$f; break 3; }; done
		done
	done
fi
[ -r "${M:-/nonexistent}" ] || { echo "no model" >&2; exit 1; }

# ⚠ THE GOVERNOR IS THE WHOLE POINT. Put it back on the way out.
OLD=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "")
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
	[ -w "$g" ] && echo performance > "$g" 2>/dev/null
done
trap '[ -n "$OLD" ] && for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do [ -w "$g" ] && echo "$OLD" > "$g" 2>/dev/null; done' EXIT

W4="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_KMAX=2048 \
CHARSIU_NPU_W4_GROUP=1024 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 \
CHARSIU_PROBE_WIDTHS=$W CHARSIU_PROBE_MAXT=${CHARSIU_AB_MAXT:-40}"

echo "model     $M"
echo "governor  $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) (was $OLD)"
echo "width $W, $N repeats an arm, arms alternate"
echo
echo "arm      pass  pack ms   entry ms   rows"
i=1
while [ "$i" -le "$N" ]; do
	for arm in stride gather; do
		case $arm in
		gather) E="CHARSIU_NPU_PACK_GATHER=1" ;;
		*)      E="" ;;
		esac
		# shellcheck disable=SC2086
		out=$(env $W4 $E "$RUN" "$M" --batch-probe "$W" 2>&1)
		line=$(printf '%s' "$out" | grep -E "^ *$W  " | tail -1)
		pack=$(printf '%s' "$line" | sed -n 's/.*pack *\([0-9]*\).*/\1/p')
		entry=$(printf '%s' "$line" | awk '{for(i=1;i<=NF;i++) if($i=="ms"){print $(i-1); exit}}')
		rows=$(printf '%s' "$line" | sed -n 's/.*\([0-9]* of [0-9]*\).*/\1/p')
		printf '%-8s %-5s %-9s %-10s %s\n' "$arm" "$i" "${pack:-?}" "${entry:-?}" "${rows:-?}"
	done
	i=$((i + 1))
done
echo
echo "read the spread within an arm before believing any gap between them."
