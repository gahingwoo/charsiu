#!/bin/sh
# Can every phase of board_verify.sh run ALONE?
#
# ⚠⚠ WHY THIS EXISTS. Phase 10 shipped with its prompt built inside phase 9's
# case arm, so `board_verify.sh 10` -- one phase, which is exactly how a sweep
# gets run -- died on its first model with "P9: parameter not set". The same
# mistake had already been made once and fixed once, for the short prompt, and
# the fix did not become a check, so it was free to happen again.
#
# A phase is a unit a board round can ask for on its own. Anything it reads has
# to come from the preamble, not from a phase that happened to run first.
#
# This needs no NPU and no models: it points the round at an EMPTY model
# directory, so every loop body is skipped and what is left is exactly the
# variable references, the arithmetic and the message strings. It cannot tell
# you a phase measures the right thing; it can only tell you the phase runs.
set -u

HERE=$(dirname "$0")
TMP=${TMPDIR:-/tmp}/charsiu-selftest.$$
mkdir -p "$TMP/models" "$TMP/out"
trap 'rm -rf "$TMP"' EXIT

PHASES=${*:-1 2 3 4 5 6 7 8 9 10}
fail=0
for ph in $PHASES; do
	out=$(CHARSIU_BIN_DIR=${CHARSIU_BIN_DIR:-$HERE/../build} \
	      CHARSIU_MODELS="$TMP/models" CHARSIU_BOARD_DIR="$TMP/out" \
	      timeout 120 sh "$HERE/board_verify.sh" "$ph" 2>&1)
	# ⚠ THE TWO WORDINGS ARE TWO SHELLS. dash says "parameter not set" and
	# bash says "unbound variable"; matching only one of them passes on the
	# board and fails on the desk, or the other way round.
	if printf '%s' "$out" | grep -qE "parameter not set|unbound variable"; then
		printf '  !! phase %s reads a variable it does not set\n' "$ph"
		printf '%s' "$out" | grep -E "parameter not set|unbound variable" \
		    | head -2 | sed 's/^/       /'
		fail=$((fail + 1))
	else
		printf '  ok phase %s runs alone\n' "$ph"
	fi
done

if [ "$fail" -gt 0 ]; then
	printf '\n%s phase(s) cannot be run on their own.\n' "$fail"
	exit 1
fi
printf '\nevery phase runs on its own.\n'
