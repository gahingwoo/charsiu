#!/bin/sh
# AWQ, on the host CPU reference, where every arm is deterministic.
#
# This exists because the fault it guards was invisible for as long as it was
# there: CHARSIU_NPU_AWQ with no statistics fell back to the column means of
# the weights and said nothing, and that fallback measures 40% WORSE than not
# using AWQ at all. Nothing in the tree exercised it.
#
#   usage: tests/host_awq.sh MODEL.gguf [CALIB.txt [EVAL.txt]]
#
# ⚠ THE TEXTS DEFAULT TO tests/corpus, WHICH IS THE POINT. Every perplexity
# recorded in this tree was measured on those bytes, so an arm run against
# anything else is a number that cannot be put beside them. Pass your own only
# when the question is about the corpus.
set -e
D=$(dirname "$0")
M=${1:?model} C=${2:-$D/corpus/calib.txt} E=${3:-$D/corpus/long.txt}
# ⚠ ON THE BOARD THERE IS NO build/. The binaries sit beside this script in
# /opt/charsiu, which is what `charsiu update dev` installs -- spec_identity.sh
# lost a whole board run to exactly this and says so at its own resolver.
B=$D/../build
[ -x "$B/charsiu_ppl" ] || B=$D
[ -x "$B/charsiu_ppl" ] || { echo "no charsiu_ppl beside $0 or in ../build"; exit 2; }
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

#
# ⚠⚠ W4_GROUP=1024 BECAUSE THE HOST REFERENCE DOES NOT GET IT FOR FREE.
# llama_auto_kmax() pins the group to 1024 and is called only when the NPU is
# on, so `CHARSIU_NPU=0` takes npuquant's code default of one absmax a row --
# a quantiser the board never runs. The assertions here are relational and
# hold either way, but the NUMBERS this prints are read by people, and they
# should be the board's.
#
export CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
       CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024
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
