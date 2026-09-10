#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
#
# The evaluation corpora have not changed.
#
# ⚠ THIS IS NOT A UNIT TEST, IT IS A LOCK. charsiu_ppl is deterministic, which
# is the only reason a number measured last week can be compared with one
# measured today -- and that holds for exactly as long as the text is the same
# bytes. An edit here does not fail anything: it silently re-bases every
# perplexity in docs/lab-notebook.md and the README against a corpus that is no
# longer the one they were measured on.
#
# So the hashes are checked rather than trusted, in `make test`, where a diff
# that touches these files cannot get past without saying so.
#
# If a corpus genuinely has to change, the honest move is a NEW file beside
# these and a new baseline measured on it. tests/corpus/README.md says why.
set -e
D="$(dirname "$0")/corpus"
bad=0
check() {
	got=$(md5sum "$D/$1" 2>/dev/null | cut -d' ' -f1)
	if [ "$got" != "$2" ]; then
		echo "  corpus: $1 is $got, was $2"
		bad=$((bad + 1))
	fi
}
check long.txt  4237c8fc3163a359fc21bde60c7b1d8b
check calib.txt 7fd405ffb641d25ddbed66f3c6414fbe
check ppl.txt   96bd8dd96214fcc776d55bbacb643dc8
echo "  corpus: $bad of 3 files changed"
exit $bad
