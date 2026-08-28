#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# How many threads, and on which cores?
#
# ⚠ THE TOWERS PIN NOTHING. charsiu-runner writes CHARSIU_CPUS from [run] cpus
# for the language model; charsiu_whisper and charsiu_vision are their own
# commands and nobody sets it for them, so pool_start takes every core the
# machine has. On this board that is four A53s and four A72s, and a barrier
# waits for the slowest range -- so eight threads can be slower than four.
#
# The config already says cpus = 4-7 and MEASURED that pinning to the A53s costs
# 23% of a decode. That was the language model. This asks the same question of
# the towers, whose shape is different: one dispatch a layer over thousands of
# independent rows.
#
#   sh tests/board_threads.sh
set -eu

DIR=${CHARSIU_BOARD_DIR:-$HOME/charsiu-board}
WSP=""
for d in /usr/bin /opt/charsiu "$PWD/build"; do
	[ -x "$d/charsiu_whisper" ] && { WSP="$d/charsiu_whisper"; break; }
done
[ -n "$WSP" ] || { echo "charsiu_whisper not found" >&2; exit 1; }
[ -f "$DIR/jfk.wav" ] || { echo "$DIR/jfk.wav is missing; run board_modalities.sh first" >&2; exit 1; }

ms() { awk '{printf "%d", $1 * 1000}' /proc/uptime; }
run() {   # label, then the environment
	_l=$1; shift
	t0=$(ms)
	OUT=$(env "$@" CHARSIU_STAGES=1 "$WSP" "$DIR/ggml-tiny.en.bin" \
		--transcribe --audio "$DIR/jfk.wav" 2>"$DIR/.th.err")
	SEC=$(awk -v m="$(( $(ms) - t0 ))" 'BEGIN{printf "%.1f", m/1000}')
	A=$(awk '/attention/{print $2}' "$DIR/.th.err")
	M=$(awk '/mel spectrogram/{print $3}' "$DIR/.th.err")
	case "$OUT" in
	*"ask not what your country"*) V=ok ;;
	*)                             V="WRONG WORDS" ;;
	esac
	printf '%-26s %7s s   attn %6s ms   mel %6s ms   %s\n' "$_l" "$SEC" \
		"$A" "$M" "$V"
}

echo "whisper tiny.en on jfk.wav, 11 s of audio"
echo
run "all cores, all threads"      CHARSIU_NPU=1
run "8 threads"                   CHARSIU_NPU=1 CHARSIU_THREADS=8
run "4 threads, unpinned"         CHARSIU_NPU=1 CHARSIU_THREADS=4
run "4 threads on the A72s"       CHARSIU_NPU=1 CHARSIU_THREADS=4 CHARSIU_CPUS=4-7
run "4 threads on the A53s"       CHARSIU_NPU=1 CHARSIU_THREADS=4 CHARSIU_CPUS=0-3
run "2 threads on the A72s"       CHARSIU_NPU=1 CHARSIU_THREADS=2 CHARSIU_CPUS=4-7
run "1 thread"                    CHARSIU_NPU=1 CHARSIU_THREADS=1
echo
echo "⚠ A barrier waits for the slowest range, so a row on the A53s holds the"
echo "  A72s up. If 4 on the A72s beats 8 on everything, that is what it is."
