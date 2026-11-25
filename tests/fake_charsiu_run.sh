#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# A stand-in for charsiu_run, so tests/board_arms.sh can be exercised on a
# desk with no NPU. It is NOT a model: it prints a canned text and a summary
# line in charsiu_run's format, with jitter, and it can be told to fail.
#
# Its knobs are environment variables, which is exactly how board_arms.sh
# hands an arm its switches, so an arm spec like
#   --b /path/fake_charsiu_run.sh:FAKE_SPEED=1.3,FAKE_TEXT=alt
# is a faster arm that also says something different.
#
#   FAKE_SPEED=1.0     decode speed multiplier
#   FAKE_JITTER=3      +- percent noise on every timing
#   FAKE_TEXT=alt      change one line of the text (TEXT DIFFERENT path)
#   FAKE_FAIL=timeout|crash|nosummary|every3
#                      fail the way a board run can fail; every3 makes every
#                      third call of this arm print no summary
#   FAKE_STATE=FILE    a counter file for every3
#
# It also answers --version, and as charsiu_ppl (called with a corpus file
# second) it prints a ppl line moved by FAKE_PPL_SHIFT.
set -u
[ "${1:-}" = "--version" ] && { echo "fake-0000000"; exit 0; }

model=${1:-}; shift 2>/dev/null || true
prompt=""; n=64; corpus=""; pn=300
while [ $# -gt 0 ]; do
	case "$1" in
	-p) prompt=$2; shift 2 ;;
	-n) n=$2; pn=$2; shift 2 ;;
	-c|-t|--temp) shift 2 ;;
	--chat|--ignore-eos|--batch) shift ;;
	*) corpus=$1; shift ;;
	esac
done

jit() { # value, percent
	awk -v v="$1" -v j="${FAKE_JITTER:-3}" -v s="$$$(date +%N)" \
		'BEGIN { srand(s); printf "%.2f", v * (1 + (rand() * 2 - 1) * j / 100) }'
}

if [ -n "$corpus" ]; then
	# charsiu_ppl mode
	[ -r "$corpus" ] || { echo "$corpus: No such file" >&2; exit 1; }
	p=$(awk -v s="${FAKE_PPL_SHIFT:-0}" 'BEGIN { printf "%.4f", 21.3127 + s }')
	echo "  32/299  ppl 20.1" >&2
	echo "ppl $p  over $((pn - 1)) scored positions of $pn tokens  ($model)"
	exit 0
fi

case "${FAKE_FAIL:-}" in
timeout) sleep 30 ;;
crash) echo "charsiu: about to fall over" >&2; kill -SEGV $$ ;;
every3)
	st=${FAKE_STATE:-/tmp/fake_charsiu_state}
	c=$(cat "$st" 2>/dev/null || echo 0); c=$((c + 1)); echo "$c" > "$st"
	[ $((c % 3)) -eq 0 ] && { echo "partial text with no summary"; exit 0; } ;;
esac

echo "charsiu: pinned to CPUs ${CHARSIU_CPUS:-none}, 4 of them" >&2
echo "charsiu: npu 594 MHz at 750 mV, the vendor asks 750" >&2
echo "charsiu: cpu4 governor schedutil, $(jit 2016 1) MHz idle" >&2

len=$(printf '%s' "$prompt" | wc -w)
echo "This is the fake answer to a prompt of $len words."
echo "The second line is the same for every arm."
if [ "${FAKE_TEXT:-}" = alt ]; then
	echo "The third line is where arm B went its own way."
else
	echo "The third line is where both arms normally agree."
fi
echo "A last line."
ptok=$((len * 4 / 3 + 12))
pms=$(jit "$((ptok * 7 + 40))" "${FAKE_JITTER:-3}")
tps=$(jit "$(awk -v s="${FAKE_SPEED:-1.0}" 'BEGIN { print 15.0 * s }')" "${FAKE_JITTER:-3}")
gms=$(awk -v n="$n" -v t="$tps" 'BEGIN { printf "%.0f", n * 1000 / t }')
[ "${FAKE_FAIL:-}" = nosummary ] && exit 0
printf '\n\n[load 812 ms | staging 1903 ms | prompt %d tok in %.0f ms, %.2f tok/s | gen %d tok in %s ms, %s tok/s | peak 901 MB]\n' \
	"$ptok" "$pms" "$(awk -v a="$ptok" -v b="$pms" 'BEGIN{printf "%.2f", a*1000/b}')" "$n" "$gms" "$tps"
echo "[first $((n / 2)) tok $(jit "$tps" 2) tok/s, last $((n / 2)) tok $(jit "$tps" 2) tok/s, board 51 -> 55 C]"
exit 0
