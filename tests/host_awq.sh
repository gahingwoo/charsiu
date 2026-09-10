#!/bin/sh
# AWQ, on the host CPU reference, where every arm is deterministic.
#
# This exists because the fault it guards was invisible for as long as it was
# there: CHARSIU_NPU_AWQ with no statistics fell back to the column means of
# the weights and said nothing, and that fallback measures 40% WORSE than not
# using AWQ at all. Nothing in the tree exercised it.
#
#   usage: tests/host_awq.sh MODEL.gguf CALIB.txt EVAL.txt
set -e
M=${1:?model} C=${2:?calibration text} E=${3:?evaluation text}
B=$(dirname "$0")/../build
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

#
# ⚠ A SIBLING WOULD BE FOUND BY THE ARM THAT IS SUPPOSED TO FIND NOTHING.
# llama_load picks up <model>.awq on its own, so the "no statistics" arm is
# only that arm when there is no such file.
#
if [ -f "$M.awq" ]; then
	echo "FAIL: $M.awq exists -- the no-statistics arm would find it"
	exit 1
fi

export CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1
ppl() { "$B"/charsiu_ppl "$M" "$E" -n 300 2>/dev/null | tail -1 |
        grep -o 'ppl [0-9.]*' | awk '{print $2}'; }

CHARSIU_CALIB="$T/stats" "$B"/charsiu_ppl "$M" "$C" -n 150 >/dev/null 2>&1
test -s "$T/stats" || { echo "FAIL: the calibration pass wrote nothing"; exit 1; }

OFF=$(ppl)
NONE=$(CHARSIU_NPU_AWQ=0.20 ppl)
WITH=$(CHARSIU_NPU_AWQ=0.20 CHARSIU_AWQ_STATS="$T/stats" ppl)

echo "  AWQ off              $OFF"
echo "  AWQ on, no stats     $NONE   (must equal off: it declines)"
echo "  AWQ on, with stats   $WITH   (must beat off)"

rc=0
[ "$NONE" = "$OFF" ] || { echo "FAIL: AWQ with no statistics did not decline"; rc=1; }
awk -v a="$WITH" -v b="$OFF" 'BEGIN{exit !(a+0 < b+0)}' ||
	{ echo "FAIL: AWQ with statistics did not beat AWQ off"; rc=1; }
[ $rc = 0 ] && echo "  host_awq: ok"
exit $rc
