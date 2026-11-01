#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
#
# Does the installer copy every board script to the board?
#
# PROBE_SCRIPTS in charsiu-install.sh is the list `charsiu update dev` puts in
# /opt/charsiu, and it is the only path a board round can type. A script that
# is in tests/ and not in that list lives under ~/.cache on the board, which
# is a board round that does not happen -- and the comment above the list has
# said so since the first time it drifted. It drifted twice more anyway: six
# scripts written between 09-11 and 09-13 were all missing.
#
# So the rule runs instead of being written down. Both directions: a script
# nothing installs, and an installed name that no longer exists.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INST="$ROOT/scripts/charsiu-install.sh"
bad=0

LIST=$(sed -n '/^PROBE_SCRIPTS="/,/"$/p' "$INST" \
       | sed 's/^PROBE_SCRIPTS="//; s/"$//; s/\\$//' | tr ' ' '\n' | grep -v '^$')

for f in "$ROOT"/tests/board_*.sh; do
	[ -e "$f" ] || continue
	b=$(basename "$f")
	printf '%s\n' "$LIST" | grep -qx "$b" || {
		echo "  NOT INSTALLED  $b"; bad=$((bad + 1)); }
done
for b in $LIST; do
	[ -r "$ROOT/tests/$b" ] || { echo "  NOT IN tests/    $b"; bad=$((bad + 1)); }
done

# WHAT AN INSTALLED SCRIPT CALLS HAS TO BE INSTALLED TOO. board_regress.sh
# was in the list and arch_sanity.sh and neon_control.sh, which it runs, were
# not; nor was charsiu_run_scalar. A fresh dev install could not regress at
# all (2026-11-01). Follow $HERE/x.sh and $B/x from every listed script.
BINS=$(sed -n 's/^\(RUNTIME\|PROBE\)_BINS="//p; /^\t *charsiu_vision/p' "$INST" \
       | sed 's/"$//; s/\\$//' | tr ' \t' '\n\n' | grep -v '^$')
for b in $LIST; do
	[ -r "$ROOT/tests/$b" ] || continue
	for r in $(grep -o '\$HERE/[A-Za-z0-9_]*\.sh' "$ROOT/tests/$b" | sed 's#\$HERE/##' | sort -u); do
		printf '%s\n' "$LIST" | grep -qx "$r" || {
			echo "  CALLED, NOT INSTALLED  $b -> $r"; bad=$((bad + 1)); }
	done
	for r in $(grep -o '\$B/[a-z_0-9]*' "$ROOT/tests/$b" | sed 's#\$B/##' | sort -u); do
		case "$r" in models|corpus) continue ;; esac
		printf '%s\n' "$BINS" | grep -qx "$r" || {
			echo "  CALLED, NOT INSTALLED  $b -> binary $r"; bad=$((bad + 1)); }
	done
done

if [ "$bad" = 0 ]; then
	echo "probe_list: every board script is in PROBE_SCRIPTS, and every"
	echo "            name in PROBE_SCRIPTS is a file ($(printf '%s\n' "$LIST" | grep -c .) scripts)"
else
	echo "probe_list: FAILED, $bad"
fi
exit $bad
