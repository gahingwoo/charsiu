#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
#
# Is speculative decoding bit-identical to the plain greedy loop, and did it
# actually speculate?
#
# ⚠ THREE ARMS, AND THE THIRD ONE IS THE POINT. A verifier that accepts
# everything produces fluent text at a wonderful rate; a verifier that runs
# plain produces the right text at the old rate and a report line that reads
# as "speculation gained nothing". Neither is visible from the text alone.
#
#   plain       charsiu_run, greedy                          the reference
#   spec        --spec 3, prompt-lookup drafts               must match, AND
#                                                            must have accepted
#                                                            something
#   junk        --spec 3 with CHARSIU_SPEC_JUNK=1            must match, AND
#                                                            must have accepted
#                                                            (almost) nothing
#
# The prompt asks for repetition so that prompt lookup has something to find:
# on open prose it mostly proposes nothing and this test could pass with an
# acceptance of zero, which would say nothing about the verifier.
#
# ⚠ THIS NEEDS NO HARDWARE. On a machine with no NPU the batched forward runs a
# row at a time on the CPU, so the ORDER of the speculative pass is exercised
# and the batched MATMUL is not -- which is the same blind spot phase 2 has, and
# why the board round still has to run it. What this can prove is that the
# verification and the roll back are right; what it cannot is that the
# hardware batch at m = 4 is.
#
#   tests/spec_identity.sh MODEL_DIR [N_TOKENS]
#
# A model the batched forward refuses is reported and skipped: that is a fact
# about the model, and the runner says so on stderr.

set -u
DIR="${1:?usage: spec_identity.sh MODEL_DIR [N_TOKENS]}"
N="${2:-48}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROMPT="Repeat this sentence exactly three times, word for word: the quick brown fox jumps over the lazy dog."
# ⚠ ON THE BOARD THERE IS NO MAKEFILE AND NO build/: the binaries sit beside
# this script in /opt/charsiu. The first board run of this died on
# "No rule to make target build/charsiu_run" before measuring anything.
RUN=${CHARSIU_RUN_BIN:-}
for c in "$ROOT/build/charsiu_run" "$ROOT/charsiu_run" "$(dirname "$0")/charsiu_run"; do
	[ -n "$RUN" ] && break
	[ -x "$c" ] && RUN=$c
done
if [ -z "$RUN" ] && [ -f "$ROOT/Makefile" ]; then
	make -C "$ROOT" build/charsiu_run >/dev/null && RUN="$ROOT/build/charsiu_run"
fi
[ -x "${RUN:-/nonexistent}" ] || { echo "no charsiu_run to test with"; exit 2; }

# the generated text only: everything before the runner's own report
text() { sed '/^\[load /,$d'; }
# the [spec ...] line, and the accepted count out of it
specline() { grep '^\[spec ' | head -1; }
accepted() { sed -n 's/.*accepted \([0-9]*\).*/\1/p'; }

bad=0; n=0; skipped=0
for mdl in "$DIR"/*.gguf; do
	[ -e "$mdl" ] || continue
	n=$((n + 1))
	name=$(basename "$mdl")
	# ⚠ A MODEL THAT STOPS AT ONCE VERIFIES NOTHING. gemma-3-1b answers a
	# bare prompt with its end-of-turn token as the very first token, so
	# every arm commits one token and the identity is vacuous. Push it past
	# its own end: the text is then a continuation of a finished turn, which
	# is a poor answer and a perfectly good oracle -- the three arms still
	# have to agree on every token of it.
	# ⚠ DECIDED ON THE PASS COUNT, NOT ON THE TEXT: the runner echoes the
	# prompt, so a word count of the output is never small.
	EXTRA=""
	spec=$("$RUN" "$mdl" -p "$PROMPT" -n "$N" --spec 3 2>"$ROOT/build/spec.err")
	np=$(printf '%s' "$spec" | specline | sed -n 's/.*: \([0-9]*\) passes.*/\1/p')
	if [ "${np:-0}" -le 2 ] && ! grep -q "refuses this model\|speculation is off" "$ROOT/build/spec.err"; then
		EXTRA="--ignore-eos"
		spec=$("$RUN" "$mdl" -p "$PROMPT" -n "$N" --spec 3 $EXTRA 2>"$ROOT/build/spec.err")
	fi
	plain=$("$RUN" "$mdl" -p "$PROMPT" -n "$N" $EXTRA 2>/dev/null)
	if grep -q "refuses this model\|speculation is off" "$ROOT/build/spec.err"; then
		printf '  – %-40s refused: %s\n' "$name" \
		    "$(sed -n 's/.*(\(.*\)).*/\1/p' "$ROOT/build/spec.err" | head -1)"
		skipped=$((skipped + 1))
		continue
	fi
	junk=$(CHARSIU_SPEC_JUNK=1 "$RUN" "$mdl" -p "$PROMPT" -n "$N" --spec 3 $EXTRA 2>/dev/null)

	tp=$(printf '%s' "$plain" | text)
	ts=$(printf '%s' "$spec" | text)
	tj=$(printf '%s' "$junk" | text)
	ls=$(printf '%s' "$spec" | specline)
	lj=$(printf '%s' "$junk" | specline)
	as=$(printf '%s' "$ls" | accepted); as=${as:-0}
	aj=$(printf '%s' "$lj" | accepted); aj=${aj:-0}

	ok=1
	if [ "$tp" != "$ts" ]; then
		echo "  ⚠⚠ $name: speculative text DIFFERS from plain greedy"
		printf '     plain: %.100s\n     spec:  %.100s\n' "$tp" "$ts"
		ok=0
	fi
	if [ "$tp" != "$tj" ]; then
		echo "  ⚠⚠ $name: text moved under JUNK drafts -- the verifier is not verifying"
		ok=0
	fi
	if [ -z "$ls" ]; then
		echo "  ⚠⚠ $name: no [spec ...] line -- the arm ran plain and said nothing"
		ok=0
	elif [ "$as" -lt 1 ]; then
		echo "  ⚠⚠ $name: identical text but ZERO drafts accepted -- this proved nothing"
		echo "     $ls"
		ok=0
	fi
	if [ "$aj" -gt 1 ]; then
		echo "  ⚠⚠ $name: junk drafts were accepted $aj times"
		ok=0
	fi
	if [ $ok = 1 ]; then
		printf '  ✓ %-40s identical x3   %s%s\n' "$name" \
		    "$(printf '%s' "$ls" | sed 's/^\[spec k=3: //; s/\]$//')" \
		    "${EXTRA:+   (past its own end)}"
	else
		bad=$((bad + 1))
	fi
done
echo
echo "$n models, $skipped refused, $bad wrong"
exit "$bad"
