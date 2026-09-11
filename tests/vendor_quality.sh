#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# The vendor's int4, SCORED -- the empty cell beside every speed comparison.
#
# Every speed claim this project makes is against a runtime whose output
# quality nobody has measured, so neither charsiu row could be read as better.
# tools/rkllm_rebuild.py writes the vendor's own weights into an f16 gguf, and
# this scores that file against the same weights through charsiu's quantiser
# and through llama.cpp's, on one corpus, at one length, with no quantiser
# running at inference in ANY arm.
#
# ⚠⚠ THE PROTOCOL IS PINNED HERE AND IT WAS NOT BEFORE. The first round of
# this comparison recorded `reference 19.8844` in three places and never wrote
# down which text or how many tokens, so the numbers could not be put beside
# anything else in the tree. They are on tests/corpus/long.txt at -n 300 now,
# which is the corpus tests/corpus_fixed.sh locks by md5 and every other
# quality number in the tree uses.
#
# ⚠⚠ AND A PERPLEXITY BELONGS TO A FILE. charsiu re-quantises whatever it
# loads, so the source format is inside the answer: the same Llama-3.2-1B
# reads 33.8071 from Q4_0 and 28.7072 from Q8_0 at group 1024. Every arm in
# one run descends from ONE file, rkllm_codes.REF, so what the comparison
# scores is each quantiser and not what happened before it.
#
# ⛔ THE ROUND OF RECORD IS FROM Llama-3.2-1B-Instruct-f16.gguf, NOT Q8_0,
# which was the default here until 2026-09-11. The vendor quantised the
# original weights; an arm that quantises Q8_0 instead is not being asked to
# quantise the same thing. Set CHARSIU_RKLLM_REF to the f16 original. The
# reference's md5 goes into every arm's filename, because the origin is part
# of the answer and an arm name alone does not carry it.
#
#   sh tests/vendor_quality.sh [43|L3|No1]
#
# 43   the 43 matrices whose scale still satisfies (max-min)/15, so the
#      reference IS what the vendor quantised and no calibration is involved
# L3   layers 3..15, 91 matrices, the vendor arm carrying their own norms
# No1  everything except layer 1, 105 matrices
#
# ⛔ THERE IS NO `full` ARM AND THAT IS DELIBERATE. All 112 matrices reads
# 58.76 against 32.13 for the same set without layer 1, because blk.1's row
# gauge is the most extreme in the model -- ffn_up at rho 22.3 against
# ffn_down at 0.298 -- and blk.1.ffn_down is not reconstructed at all. That
# number is a measurement of my reconstruction, not of their quality.
set -eu

SET=${1:-43}
D=$(dirname "$0")
R=$D/..
M=$R/models
#
# ⚠⚠ BOTH PASSAGES, ALWAYS, BECAUSE ONE PASSAGE CANNOT ORDER ANYTHING CLOSER
# THAN ABOUT 10%. Three conclusions read off long.txt alone on 2026-09-11 all
# reversed on long2.txt. The corpus was hardcoded here, so the second opinion
# had to be taken by hand and only the 43-matrix rung ever got one.
#
# It costs nothing. Each arm is scored while its 2.5 GB file is on the disk
# and the file is removed either way, so a second passage is a second forward
# pass and not a second build. CHARSIU_VQ_CORPUS overrides the list.
#
E=${CHARSIU_VQ_CORPUS:-"$D/corpus/long.txt $D/corpus/long2.txt"}
NTOK=${CHARSIU_VQ_NTOK:-300}
KEEP=${CHARSIU_VQ_KEEP:-0}

[ -x "$R/build/charsiu_ppl" ] || { echo "no build/charsiu_ppl -- run make"; exit 2; }

case "$SET" in
43)  ARMS="ref rho1 vendor43 chr43 q4043 noise"
     WHAT="43 matrices, rho = 1, no calibration involved" ;;
L3)  ARMS="ref vendorL3 chrL3 q40L3"
     WHAT="91 matrices, layers 3..15, vendor arm carries their norms" ;;
No1) ARMS="ref vendorNo1 chrNo1"
     WHAT="105 matrices, everything except layer 1" ;;
*)   echo "usage: $0 [43|L3|No1]"; exit 2 ;;
esac

echo "== the vendor's quantiser, scored"
echo "   set      $SET -- $WHAT"
for e in $E; do
	[ -f "$e" ] || { echo "no $e"; exit 2; }
	echo "   corpus   $e  ($(md5sum "$e" | cut -c1-32))"
done
echo "   tokens   $NTOK"
#
# ⛔ THE ORIGIN IS PART OF THE FILENAME, NOT JUST OF THE HEADER. An arm cached
# under its arm name alone is reused across a change of reference, so a round
# rebuilt from f16 would score the Q8_0-origin file still sitting in models/.
# Two such files were there on 2026-09-11, two days older than the control.
#
eval "$(python3 -P -c 'import sys, os; sys.path.insert(0,"'"$R"'/tools"); import rkllm_codes as R; print("SRC=%s\nTAG=%s" % (os.path.basename(R.REF), R.ref_tag()))')"
echo "   source   $SRC  ($TAG)"
echo

#
# ⚠ NO QUANTISER AT INFERENCE, IN ANY ARM. The quantisation has already
# happened, into f16, which is the only way the arms differ in exactly the
# thing being compared. CHARSIU_NPU_QUANT=1 here would quantise all four files
# a second time and measure charsiu's quantiser four times over.
#
ppl() {
	env CHARSIU_NPU=0 CHARSIU_NPU_QUANT=0 \
		"$R/build/charsiu_ppl" "$1" "$2" -n "$NTOK" 2>/dev/null |
		tail -1 | grep -o 'ppl [0-9.]*' | tail -1 | awk '{print $2}'
}

ncorp=0
for e in $E; do ncorp=$((ncorp + 1)); eval "REFPPL_$ncorp="; done

printf '  %-10s' arm
i=0
for e in $E; do i=$((i + 1)); printf ' %18s' "$(basename "$e")"; done
printf '\n'

for a in $ARMS; do
	F=$M/Llama-3.2-1B-$a-$TAG-F16.gguf
	built=0
	if [ ! -f "$F" ]; then
		# ⚠ EACH FILE IS ABOUT 2.5 GB. Build, score, remove, next --
		# the whole set at once does not fit on this disk.
		free=$(df -Pm "$M" | awk 'NR==2{print $4}')
		if [ "$free" -lt 3000 ]; then
			echo "  $a: SKIPPED, only ${free} MB free"
			continue
		fi
		echo "  $a: building..."
		python3 -P "$R/tools/rkllm_rebuild.py" "$a" >/dev/null || {
			echo "  $a: BUILD FAILED"; continue; }
		built=1
	fi
	printf '  %-10s' "$a"
	i=0
	for e in $E; do
		i=$((i + 1))
		P=$(ppl "$F" "$e")
		eval "B=\$REFPPL_$i"
		if [ -z "$P" ]; then
			printf ' %18s' FAILED
		elif [ -z "$B" ]; then
			eval "REFPPL_$i=\$P"
			printf ' %18s' "$P"
		else
			printf ' %10s %+7.2f%%' "$P" \
				"$(awk -v a="$P" -v b="$B" 'BEGIN{print (a/b-1)*100}')"
		fi
	done
	printf '\n'
	[ "$built" = 1 ] && [ "$KEEP" = 0 ] && rm -f "$F"
done

echo
#
# ⚠⚠ READ `noise` BEFORE READING THE VENDOR ROW. A reconstruction can be
# wrong in a way a weight norm barely charges for and a forward pass charges
# enormously: the first version of this scored 1700.98 at 18.3% median weight
# error, and the same per-tensor error as unstructured Gaussian noise scored
# 32.10. If the vendor row and the noise row are close, the vendor row is
# measuring the reconstruction rather than their quantiser.
#
echo "⚠ the vendor row is only about their quantiser if it is far from noise:"
echo "  a column scaled by the wrong factor is a systematically wrong channel,"
echo "  not a small perturbation, and Frobenius hardly charges for it."
