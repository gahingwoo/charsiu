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
#   CHARSIU_BOARD_DIR=... tests/arch_sanity.sh
#
# Every gguf in the directory is asked the same question. A file whose answer
# does not contain the word is reported; the exit status is the number of them.

set -e

# ⚠ SOURCED HERE AND NOT FURTHER DOWN: board_clk.sh is what makes
# CHARSIU_RUN and CHARSIU_RUN_BIN two names for one knob, and this
# script picks its binary below. Sourcing it after that point set the
# alias too late to be read -- which is how round 414 measured the
# INSTALLED binary for twenty minutes while believing otherwise.
. "$(dirname "$0")/board_clk.sh"
# ⚠ THE ENVIRONMENT IS THE SECOND WAY IN, AND IT HAD TO BE. The board's
# regress.sh exports CHARSIU_BOARD_DIR and then calls this with no argument,
# which this refused -- so section 1 of the r411 regression, "every
# architecture still knows a fact", has printed a usage line and NOTHING ELSE
# every time it has run. `set -e` does not stop it because the call is inside a
# pipeline, so the regression carried on and reported the rest as if the
# architectures had passed. Same shape as the prefill script that found no
# models and exited 0: a check that cannot run reads exactly like a check that
# found nothing wrong.
DIR="${1:-${CHARSIU_BOARD_DIR:-}}"
if [ -z "$DIR" ]; then
	echo "arch_sanity.sh: no model directory." >&2
	echo "  give one as \$1, or set CHARSIU_BOARD_DIR." >&2
	exit 2
fi
if [ ! -d "$DIR" ]; then
	echo "arch_sanity.sh: $DIR is not a directory" >&2
	exit 2
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# ⚠ CHARSIU_RUN SO THIS CAN RUN WHERE THE MODELS ARE. The models live on the
# board and the board has no compiler, so a script that builds before it checks
# is a check that cannot run on the only machine that has something to check.
RUN="${CHARSIU_RUN:-$ROOT/build/charsiu_run}"
# ⚠⚠ AND charsiu_check THE SAME WAY, WHICH IT WAS NOT. This resolved only
# $ROOT/build/charsiu_check, and on the board there is no tree -- the binaries
# live in /opt/charsiu. So `arch` was EMPTY on every row and the architecture
# column, the one this whole section is named after, printed `?` on the only
# machine where the check means anything. Look next to whichever charsiu_run we
# were given first, then fall back to the tree.
CHECK="${CHARSIU_CHECK:-}"
[ -n "$CHECK" ] || { _d=$(dirname "$RUN"); [ -x "$_d/charsiu_check" ] && CHECK="$_d/charsiu_check"; }
[ -n "$CHECK" ] || CHECK="$ROOT/build/charsiu_check"

PROMPT="The capital of France is"
WANT="Paris"

[ -n "${CHARSIU_RUN:-}" ] || make -C "$ROOT" build/charsiu_run >/dev/null

# ⚠ ONCE WITH CHARSIU_STAGES, because a crash that needs an environment
# variable is still a crash. The sliding window's start was shadowed by
# llama_forward's timing variable, which only the STAGE macro assigns to, so
# every run with stages on took a wild pointer into softmax -- and the board
# has stages on, so every NPU run died on it while every test here passed.
# Four board rounds went into that. One extra pass here would have caught it.
: "${CHARSIU_STAGES_PASS:=1}"

bad=0
n=0
archs=""
archs_ok=""
for m in "$DIR"/*.gguf; do
	[ -e "$m" ] || continue
	n=$((n + 1))
	# ⚠ ONLY ON "OK". charsiu_check's refusals start with NO and put a
	# reason in the second field, so an unconditional $2 labels a rejected
	# file with a fragment of the sentence explaining why.
	arch=$("$CHECK" -q "$m" 2>/dev/null |
	       awk '$1 == "OK" { print $2 }')
	archs="$archs ${arch:-?}"
	out=$("$RUN" "$m" -p "$PROMPT" -n 24 -c 512 -q 2>/dev/null | head -1)
	case "$out" in
	*"$WANT"*)
		if [ "$CHARSIU_STAGES_PASS" = 1 ]; then
			sout=$(CHARSIU_STAGES=1 "$RUN" "$m" -p "$PROMPT" -n 8 \
				-c 512 -q 2>/dev/null | head -1) || sout=""
			case "$sout" in
			*"$WANT"*) ;;
			*)
				printf 'BAD  %-16s %s  (only with CHARSIU_STAGES=1)\n' \
					"${arch:-?}" "$(basename "$m")"
				printf '       %s\n' "$sout"
				bad=$((bad + 1))
				continue
				;;
			esac
		fi
		printf 'ok   %-16s %s\n' "${arch:-?}" "$(basename "$m")"
		archs_ok="$archs_ok ${arch:-?}"
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
# ⚠⚠ FILES AND ARCHITECTURES ARE DIFFERENT COUNTS, and this printed only the
# first while being quoted as the second. The stable merge at 0c85c71 says
# "7/7 architectures"; seven is the number of GGUF FILES in /opt/vendor/models,
# and five architectures cover them. Print both, and say which is which.
na=$(printf '%s\n' $archs | grep -v '^?$' | sort -u | wc -l)
nk=$(printf '%s\n' $archs_ok | grep -v '^?$' | sort -u | wc -l)
nq=$(printf '%s\n' $archs | grep -c '^?$' || true)
echo "$((n - bad))/$n FILES knew the capital of France"
if [ "$na" -gt 0 ]; then
	echo "$nk/$na ARCHITECTURES knew it: $(printf '%s\n' $archs | grep -v '^?$' | sort -u | tr '\n' ' ')"
fi
[ "${nq:-0}" -eq 0 ] || echo "⚠ $nq file(s) have no architecture: charsiu_check ($CHECK) did not answer for them"
exit "$bad"
