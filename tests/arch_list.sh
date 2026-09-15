#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
#
# Do charsiu_check and llama_load accept the same architectures?
#
# charsiu_check.c's list carries a comment reading "THIS LIST MUST MATCH
# llama_load's", and then explains why being wrong in either direction costs
# something: refusing a model that runs wastes a model, accepting one that does
# not wastes the 2 GB download the check exists to prevent.
#
# Nothing enforced it. The two lists agree today -- they were compared by hand
# on 2026-09-15 and both read llama qwen2 qwen3 gemma3 gemma4 phi3 smollm3 --
# but that is a fact about today, and this tree has been bitten three times by
# a list duplicated in two places with a comment asking them to stay equal:
# PROBE_SCRIPTS twice, and the Makefile's source lists when vision landed.
# probe_list.sh exists because of the first of those. This is the same rule for
# the architectures.
#
# It reads the source rather than running anything, so it works with no board
# and no model.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOAD="$ROOT/src/llama.c"
CHECK="$ROOT/tools/charsiu_check.c"
for f in "$LOAD" "$CHECK"; do
	[ -r "$f" ] || { echo "arch_list.sh: cannot read $f" >&2; exit 2; }
done

# THE ARCHITECTURE COMPARISONS ONLY. Both files compare other strings with
# strcmp -- tensor names, tokenizer models -- so this matches strcmp(arch, ...)
# and nothing else, which is the variable both lists are about.
archs() { grep -oE 'strcmp\(arch, "[a-z0-9_]+"\)' "$1" | sed 's/.*"\(.*\)".*/\1/' | sort -u; }

a=$(archs "$LOAD")
b=$(archs "$CHECK")

# clip IS NOT A GRAPH AND IS NOT IN THIS COMPARISON. charsiu_check reads
# mmproj files as well, and "clip" is how it tells one apart; llama_load never
# sees one. Excluded by name here rather than by being quietly absent, so that
# a future architecture cannot hide behind the exception.
b=$(printf '%s\n' "$b" | grep -v '^clip$')

# NO PROCESS SUBSTITUTION. The first draft used comm with <(...), which is
# a bashism: the board runs busybox sh, where that is a syntax error, and this
# is meant to be runnable anywhere the rest of tests/ is. Two plain loops.
bad=0
in_list() {
	for _y in $2; do [ "$_y" = "$1" ] && return 0; done
	return 1
}
for x in $a; do
	in_list "$x" "$b" && continue
	echo "  llama_load accepts $x and charsiu_check does not: a model that"
	echo "     runs would be refused before it is downloaded"
	bad=$((bad + 1))
done
for x in $b; do
	in_list "$x" "$a" && continue
	echo "  charsiu_check accepts $x and llama_load does not: the 2 GB"
	echo "     download this check exists to prevent would happen anyway"
	bad=$((bad + 1))
done

n=$(printf '%s\n' "$a" | grep -c .)
if [ "$bad" -eq 0 ]; then
	echo "arch_list: charsiu_check and llama_load accept the same $n"
	echo "           architectures: $(printf '%s ' $a)"
else
	echo "arch_list: $bad disagreement(s)"
fi
exit "$bad"
