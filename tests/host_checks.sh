#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
#
# Is every host checker in tests/ actually run by `make test`?
#
# THREE OF THEM WERE NOT, AND EACH SAID OF ITSELF THAT IT MATTERED.
# tools/acc_index_check.c calls itself "the check that should have existed
# before the first board round" and was built by `all` and invoked by nothing.
# verify_selftest.sh guards a mistake that has already shipped a wrong answer.
# version_all.sh says in its own header that it runs on the desk against
# build/, and round 418 wrote it without wiring it anywhere. A check nobody
# runs is a sentence, which is the same thing corpus_fixed.sh, probe_list.sh
# and arch_list.sh each exist to stop in their own corner.
#
# THE RULE. A script in tests/ whose name does not start with board_ either
# appears in the Makefile's test target, or is named below with the reason it
# cannot be. Adding a host script now forces that decision instead of leaving
# it to whoever next reads the Makefile.
#
# Both directions, like probe_list.sh: a script nobody runs, and an exception
# for a script that no longer exists.
set -u

HERE=$(dirname "$0")
ROOT=$(cd "$HERE/.." && pwd)
MK="$ROOT/Makefile"
fail=0

# name:reason. These need something a bare checkout does not have.
EXCEPT="arch_sanity.sh:needs a directory of models and built binaries
host_awq.sh:takes a model file as argv[1]
neon_control.sh:takes a binary and a board directory
prefill_control.sh:needs a board directory of models
spec_identity.sh:takes a binary and a board directory
vattn_sweep.sh:a timing sweep, not a pass or fail
vendor_quality.sh:needs the vendor .rkllm and an f16 gguf
whisper_transcribe.sh:takes a tool, a model and a .wav"

# The test target's recipe: from `test:` to the next rule at column 0.
recipe=$(awk '/^test:/{f=1} f{print} f && /^[a-zA-Z0-9_.]+:/ && !/^test:/{exit}' "$MK")

for p in "$ROOT"/tests/*.sh; do
	n=$(basename "$p")
	case $n in board_*) continue ;; esac
	if printf '%s\n' "$recipe" | grep -q "tests/$n"; then
		printf '  ok   %-24s run by make test\n' "$n"
		continue
	fi
	why=$(printf '%s\n' "$EXCEPT" | sed -n "s|^$n:||p")
	if [ -n "$why" ]; then
		printf '  ok   %-24s exempt: %s\n' "$n" "$why"
	else
		printf '  !!   %-24s is a host script and nothing runs it\n' "$n"
		fail=$((fail + 1))
	fi
done

# An exception for a script that is gone is a stale rule.
printf '%s\n' "$EXCEPT" | while IFS=: read -r n why; do
	[ -n "$n" ] || continue
	[ -r "$ROOT/tests/$n" ] || printf '  !!   %-24s is exempted and does not exist\n' "$n"
done
printf '%s\n' "$EXCEPT" | while IFS=: read -r n why; do
	[ -n "$n" ] || continue
	[ -r "$ROOT/tests/$n" ] || exit 1
done || fail=$((fail + 1))

if [ "$fail" -ne 0 ]; then
	printf '\n%d host script(s) unaccounted for.\n' "$fail"
	exit 1
fi
printf '\nevery host script in tests/ is run or has a reason.\n'
