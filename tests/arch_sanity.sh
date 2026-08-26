#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
#
# Does each architecture still know a fact?
#
# ⚠ THIS IS NOT A CORRECTNESS ORACLE and must not be read as one. It is a smoke
# test for one specific failure mode that nothing else here catches: a graph
# that is wrong in a way which still produces FLUENT ENGLISH.
#
# The case it was written for is RoPE. charsiu rotates element 2i against
# 2i+1, which is right for llama and smollm3, whose weights the convert step
# permutes for it, and wrong for qwen2, qwen3 and phi3, whose weights it does
# not. Under the wrong pairing every element is still rotated by an angle from
# the right table, just partnered with the wrong neighbour, so the output stays
# inside the vocabulary and reads as a sentence. Measured on
# Qwen2.5-0.5B-Instruct:
#
#   wrong: "The capital of France is a country in the world. The capital of
#           the world. The capital of the world."
#   right: "The capital of France is Paris. It is the largest city in France"
#
# Nothing that compares charsiu to charsiu can see that -- neon_control.sh
# passes either way, because both builds are the same wrong graph. A reference
# run against llama.cpp can, and forward_cross.py is that, but it needs
# llama.cpp built and an f32 copy of every file. This needs neither, and it
# would have failed on the first qwen2 run.
#
#   tests/arch_sanity.sh MODEL_DIR
#
# Every gguf in the directory is asked the same question. A file whose answer
# does not contain the word is reported; the exit status is the number of them.

set -e
DIR="${1:?usage: arch_sanity.sh MODEL_DIR}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/build/charsiu_run"

PROMPT="The capital of France is"
WANT="Paris"

make -C "$ROOT" build/charsiu_run >/dev/null

bad=0
n=0
for m in "$DIR"/*.gguf; do
	[ -e "$m" ] || continue
	n=$((n + 1))
	# ⚠ ONLY ON "OK". charsiu_check's refusals start with NO and put a
	# reason in the second field, so an unconditional $2 labels a rejected
	# file with a fragment of the sentence explaining why.
	arch=$("$ROOT/build/charsiu_check" -q "$m" 2>/dev/null |
	       awk '$1 == "OK" { print $2 }')
	out=$("$RUN" "$m" -p "$PROMPT" -n 24 -c 512 -q 2>/dev/null | head -1)
	case "$out" in
	*"$WANT"*)
		printf 'ok   %-16s %s\n' "${arch:-?}" "$(basename "$m")"
		;;
	*)
		printf 'BAD  %-16s %s\n' "${arch:-?}" "$(basename "$m")"
		printf '       %s\n' "$out"
		bad=$((bad + 1))
		;;
	esac
done

if [ "$n" -eq 0 ]; then
	echo "no gguf in $DIR"
	exit 1
fi
echo "$((n - bad))/$n knew the capital of France"
exit "$bad"
