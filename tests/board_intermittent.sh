#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# How OFTEN is a refused model's batched text wrong, and does zeroing the
# output buffer stop it?
#
# ⚠⚠ WHY A RATE AND NOT ANOTHER TABLE.
#
# gemma4 came back clean ten times on 2026-08-30 -- five runs on each core
# count, text identical to its token loop -- and wrong twice the same day, once
# grossly ("practicing practicing practicing"). phi3's default chunk of 32 was
# wrong twice and then right. board_chunk_sweep printed a tail-versus-no-tail
# story out of a table where chunk 29 has NO tail and was wrong and chunk 32
# HAS one and was right.
#
# None of that is a pattern in the widths. It is one fault that fires
# sometimes, and every round that runs each cell once will keep drawing a
# different picture of it. So: one configuration, many runs, one number.
#
# ⚠⚠ AND THE ARM THAT MATTERS IS onedev, NOT zero. I had this backwards.
#
# CHARSIU_NPU_BATCH_ZERO=1 puts back the whole-buffer memset that
# assign-on-first-write removed. It was added here as a SEMANTIC control -- if
# a range never gets its first write, Y keeps the previous token's answer --
# and on that reading "wrong without it, right with it" would have meant the
# first-write bookkeeping.
#
# It does not read that way. gemma4 came back 0 of 16 on the shipped path and
# 1 of 16 WITH the buffer zeroed, which the semantic story cannot produce: a
# memset before the submit does not change what the hardware computes. What it
# changes is TIMING. So this arm is a timing perturbation, and a fault that
# appears only under one is a race, not a stale read.
#
# Which lines up with what the dense sweep proved about m = 8 and m = 10: two
# cores stepping on ROW 0 of a wide output, bit identical the moment
# CHARSIU_NPU_ONEDEV=1 makes it one core. And the sweep ran each width ONCE,
# so a fault firing one run in sixteen would have been missed at every width
# except the two where it is dense.
#
# So the discriminating arm is:
#
#   onedev   CHARSIU_NPU_ONEDEV=1   one core, the control for the core pair
#
# Clean on one core over enough runs and dirty on two is the same fault as
# m = 8, at a lower rate, and the whole residual collapses into it. Dirty on
# one core too is something else and keeps its own investigation.
#
# ⚠ CHARSIU_BATCH_FORCE IS A PROBE SWITCH: these models are refused.
#
# `charsiu update dev` installs this at /opt/charsiu/board_intermittent.sh.
#
#   sh board_intermittent.sh [MODEL-or-substring] [RUNS]
#
#   CHARSIU_INT_ARMS="default zero"    which arms to run
#   CHARSIU_INT_CHUNK=32               prefill chunk
#   CHARSIU_INT_NGEN=8                 tokens generated
set -u

RUN=${CHARSIU_RUN_BIN:-}
[ -n "$RUN" ] || RUN=$(command -v charsiu_run 2>/dev/null || true)
[ -n "$RUN" ] || for d in /opt/charsiu /usr/bin "$PWD/build" ./build .; do
	[ -x "$d/charsiu_run" ] && { RUN="$d/charsiu_run"; break; }
done
[ -n "${RUN:-}" ] || { echo "charsiu_run not found" >&2; exit 1; }

if [ ! -e /dev/accel/accel0 ] && [ -z "${CHARSIU_ALLOW_NO_NPU:-}" ]; then
	echo "NO /dev/accel/accel0 -- without the NPU the batched matmul never" >&2
	echo "runs, every arm agrees, and a rate of zero means nothing. Run it" >&2
	echo "on the board.  CHARSIU_ALLOW_NO_NPU=1 overrides." >&2
	exit 1
fi

DIRS="$HOME/.charsiu/models $HOME/models /opt/charsiu/models \
${CHARSIU_BOARD_DIR:-$HOME/charsiu-board}"
norm() { echo "$1" | tr 'A-Z' 'a-z' | tr -cd 'a-z0-9'; }

MODEL=${1:-}
if [ -n "$MODEL" ] && [ ! -r "$MODEL" ]; then
	w=$(norm "$MODEL"); hit=
	for d in $DIRS; do
		for f in "$d"/*.gguf; do
			[ -r "$f" ] || continue
			case $(norm "$(basename "$f")") in
			*"$w"*) hit=$f; break 2 ;;
			esac
		done
	done
	[ -n "$hit" ] || { echo "no model matching '$MODEL'" >&2; exit 1; }
	MODEL=$hit
fi
[ -n "${MODEL:-}" ] && [ -r "$MODEL" ] || {
	echo "usage: board_intermittent.sh <model.gguf or substring> [runs]" >&2
	exit 1
}

RUNS=${2:-8}
CHUNK=${CHARSIU_INT_CHUNK:-32}
NGEN=${CHARSIU_INT_NGEN:-8}
ARMS=${CHARSIU_INT_ARMS:-"default onedev serial"}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
# ⚠ ONE PROCESS NOW DOES $RUNS PROMPTS, so the clock is per ARM, not per
# run: 90s of staging and prompt each, floored at 300.
TMOS=${CHARSIU_INT_TIMEOUT:-0}
[ "$TMOS" -gt 0 ] 2>/dev/null || TMOS=$((RUNS * 90 + 300))
TMO=""
command -v timeout >/dev/null 2>&1 && TMO="timeout $TMOS"
WEDGED=0
# ⚠⚠ ONE PROMPT, DEFINED ONE WAY, AND ITS TOKEN COUNT PRINTED.
#
# board_text_all.sh spelled this literally and every other script built it with
# `seq 1 32 | tr`, which leaves a TRAILING SPACE. That is not cosmetic: it
# retokenises, and phi3 goes from 87 tokens to 88 -- a different last chunk, so
# two scripts comparing "the same prompt" were not.
#
# It cost a whole reading. gemma4 came back 10 of 10 clean under the literal
# form and 8 of 8 WRONG under the seq form, and that was written down as the
# model flipping, on a day when the only code change was inside the probe.
CHARSIU_PROMPT_END=${CHARSIU_PROMPT_END:-}
PROMPT="$(seq 1 32 | tr '\n' ' ')"
PROMPT=${PROMPT% }$CHARSIU_PROMPT_END
W4="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"

echo "model    $MODEL"
echo "binary   $RUN"
echo "config   chunk $CHUNK, gen $NGEN, $RUNS runs an arm"
echo "arms     $ARMS"
# ⚠ DESCRIBE ONLY THE ARMS THAT WILL RUN. The header listed all four while
# ARMS held three, which is the same class of lie as a computed width: the
# thing on screen has to be the thing that happened.
for A in $ARMS; do case $A in
default) echo "  default  as shipped: two NPU cores" ;;
onedev)  echo "  onedev   CHARSIU_NPU_ONEDEV=1 -- ONE core; the control for the core pair" ;;
serial)  echo "  serial   CHARSIU_NPU_BATCH_SERIAL=1 -- both cores, never at the same"
         echo "           time: the candidate FIX, and it keeps decode's second core" ;;
zero)    echo "  zero     CHARSIU_NPU_BATCH_ZERO=1 -- a TIMING perturbation, not a"
         echo "           semantic control: the memset cannot change what the"
         echo "           hardware computes, so a failure only here is a race" ;;
esac; done
echo

# --- the control, twice, because the token loop must be stable too ---------
# ⚠ IF THE TOKEN LOOP ITSELF IS NOT REPRODUCIBLE then every comparison below
# is against a moving target, and the first thing to fix is not the batching.
i=1
while [ "$i" -le 2 ]; do
	# shellcheck disable=SC2086
	env $W4 CHARSIU_NO_BATCH_PREFILL=1 "$RUN" "$MODEL" -p "$PROMPT" \
		-n "$NGEN" --ignore-eos >"$T/c$i.out" 2>/dev/null
	# ⚠ BEFORE THE STRIP: the count lives on the bracketed line the next
	# command deletes, and a prompt whose length nobody prints is how two
	# rounds compared different prompts and called it a flip.
	[ "$i" -eq 1 ] && NTOK=$(sed -n 's/.*prompt \([0-9]*\) tok in.*/\1/p' \
		"$T/c$i.out" | head -1)
	sed -i 's/^\[.*//' "$T/c$i.out"
	i=$((i + 1))
done
if ! cmp -s "$T/c1.out" "$T/c2.out"; then
	echo "⚠⚠ THE TOKEN LOOP DISAGREED WITH ITSELF on two runs of the same"
	echo "   prompt. The reference is not stable, so nothing measured against"
	echo "   it means anything. Stop here and chase that."
	diff "$T/c1.out" "$T/c2.out" | head -6 | sed 's/^/   /'
	exit 1
fi
cp "$T/c1.out" "$T/ref.out"
# the reference, normalised the same way the segments are
sed -e 's/^\[.*//' -e '/^[[:space:]]*$/d' "$T/ref.out" >"$T/refn.out"
# ⚠⚠ AN EMPTY REFERENCE MATCHES EVERYTHING. If the control produced no text
# -- a model that would not load, a normalisation that ate the whole file --
# then every segment below compares equal and the round reports a perfect
# score having measured nothing. This is the cheapest possible check against
# the failure this whole script exists to avoid.
if [ ! -s "$T/refn.out" ]; then
	echo "⚠⚠ THE CONTROL PRODUCED NO TEXT after normalisation, so every" >&2
	echo "   comparison below would trivially pass. Refusing to run." >&2
	sed -n '1,10p' "$T/ref.out" | sed 's/^/    /' >&2
	exit 1
fi
printf 'reference is %s line(s), %s bytes\n' \
	"$(wc -l <"$T/refn.out")" "$(wc -c <"$T/refn.out")"
# one batched run purely to read its width breakdown back
# shellcheck disable=SC2086
env $W4 CHARSIU_BATCH_FORCE=1 CHARSIU_PREFILL_CHUNK="$CHUNK" "$RUN" "$MODEL" \
	-p "$PROMPT" -n 1 --ignore-eos >/dev/null 2>"$T/w.err"
# ⚠⚠ ASK THE RUN WHICH WIDTHS IT USED. DO NOT COMPUTE THEM.
#
# This line was `NTOK % CHUNK` and it printed "a tail of 23" for a phi3 round
# that ran a tail of TWENTY TWO -- because the chunker had been changed to
# emit only even widths and this arithmetic had not. The round then read as a
# test of the width that was failing when it was a test of a different one.
#
# charsiu_run prints the breakdown itself, "(widths 2x32+1x22)". Read that.
# A number this script derives is a number that can disagree with the binary;
# a number the binary prints cannot.
printf 'prompt is %s tokens\n' "${NTOK:-?}"
WIDTHS=$(sed -n 's/.*(widths \([^)]*\)).*/\1/p' "$T/w.err" | head -1)
if [ -n "$WIDTHS" ]; then
	printf 'batched widths, as the binary reports them: %s\n' "$WIDTHS"
	case $WIDTHS in
	*x8+*|*x8|*x10+*|*x10)
		echo "⚠ that includes a width the gate refuses (8 or 10): those"
		echo "  chunks fall back a row at a time, so this round is not"
		echo "  measuring them on the batched path at all." ;;
	esac
else
	echo "⚠ THE BATCHED PATH PRINTED NO WIDTH BREAKDOWN. Either this binary"
	echo "  predates the even-only chunker or the prompt was not batched at"
	echo "  all -- and either way the arms below are not testing what the"
	echo "  header says."
	tail -3 "$T/w.err" | sed 's/^/    /'
fi
echo "control (token loop, reproducible over 2 runs):"
echo "  ...$(tr -d '\n' < "$T/ref.out" | tail -c 56)"
echo

printf '%-9s %-6s %s\n' arm rate 'runs in order (. right, X wrong)'
printf '%-9s %-6s %s\n' --- ---- ----
res_default=""; res_zero=""
for ARM in $ARMS; do
	case $ARM in
	default) EXTRA="" ;;
	onedev)  EXTRA="CHARSIU_NPU_ONEDEV=1" ;;
	serial)  EXTRA="CHARSIU_NPU_BATCH_SERIAL=1" ;;
	zero)    EXTRA="CHARSIU_NPU_BATCH_ZERO=1" ;;
	*) echo "unknown arm '$ARM'" >&2; continue ;;
	esac
	# ⚠⚠ ONE PROCESS AN ARM, NOT ONE A SAMPLE.
	#
	# This asked a rate question by starting charsiu_run once per sample:
	# 16 runs an arm, three arms, plus controls, is 51 loads of a 2.2 GB
	# model. Phi-3.5 stages in about eight seconds, so the great majority
	# of every round was spent re-uploading weights that had not changed,
	# to measure something that happens after they are uploaded. A round
	# took long enough that it was worth complaining about, and it was.
	#
	# --repeat runs the prompt N times in one process, resetting only the
	# cache position, and separates the answers with a marker line.
	#
	# ⚠ It is not the identical experiment: N repeats share one staging and
	# one set of device buffers where N processes did not. For a race in
	# the submit path that is the same question -- and if a fault ever
	# turns out to need a fresh process, CHARSIU_INT_PROC=1 puts the old
	# loop back.
	bad=0; marks=""; did=0
	rm -f "$T"/seg*
	# shellcheck disable=SC2086
	$TMO env $W4 CHARSIU_BATCH_FORCE=1 CHARSIU_PREFILL_CHUNK="$CHUNK" \
		$EXTRA "$RUN" "$MODEL" -p "$PROMPT" -n "$NGEN" --ignore-eos \
		--repeat "$RUNS" >"$T/b.out" 2>"$T/b.err"
	rc=$?
	if [ $rc -eq 124 ]; then
		echo
		echo "⚠⚠ THE $ARM ARM DID NOT FINISH IN ${TMOS}s. Almost certainly"
		echo "   a wedged NPU. Nothing measured."
		WEDGED=1
	fi
	if grep -qiE "job timed out|not functioning|iommu" "$T/b.err"; then
		echo
		echo "⚠⚠ THE NPU WEDGED during the $ARM arm:"
		grep -iE "job timed out|not functioning|iommu" "$T/b.err" \
			| head -3 | sed 's/^/     /'
		WEDGED=1
	fi
	# split the stream on the markers charsiu_run prints
	awk -v d="$T" '''BEGIN { n = 1; f = d "/seg1" }
		/^--- charsiu repeat [0-9]+ ---$/ { n++; f = d "/seg" n; next }
		{ print > f }''' "$T/b.out"
	# ⚠ BLANK LINES GO FROM BOTH SIDES. The marker carries a leading
	# newline, so segment one ends with a blank line the control does not
	# have. Nothing this prompt generates is a blank line, and the control
	# gets the same treatment, so this cannot hide a difference in the text.
	n=1
	while [ "$n" -le "$RUNS" ]; do
		[ -f "$T/seg$n" ] || break
		sed -i -e 's/^\[.*//' -e '/^[[:space:]]*$/d' "$T/seg$n"
		if cmp -s "$T/refn.out" "$T/seg$n"; then
			marks="$marks."
		else
			marks="${marks}X"; bad=$((bad + 1))
			cp "$T/seg$n" "$T/worst-$ARM.out"
		fi
		did=$((did + 1))
		n=$((n + 1))
	done
	if [ "$did" -lt "$RUNS" ] && [ "$WEDGED" -eq 0 ]; then
		echo
		echo "⚠ ONLY $did OF $RUNS REPEATS CAME BACK on the $ARM arm."
		echo "  Either this binary has no --repeat (charsiu update dev) or"
		echo "  the run stopped early. Its rate below is over $did."
		tail -3 "$T/b.err" | sed 's/^/    /'
	fi
	# ⚠ THE DENOMINATOR IS REPEATS THAT CAME BACK, never RUNS: an arm that
	# stopped early must not claim every sample happened.
	printf '%-9s %-6s %s\n' "$ARM" "$bad/$did" "$marks"
	[ "$WEDGED" -eq 0 ] || break
	[ "$bad" -eq 0 ] || printf '          a wrong one: ...%s\n' \
		"$(tr -d '\n' < "$T/worst-$ARM.out" | tail -c 56)"
	eval "res_$ARM=$bad"
done

echo
echo "======================================================================"
if [ "$WEDGED" -ne 0 ]; then
	echo "⚠⚠ THE ROUND STOPPED ON A WEDGED NPU. NOTHING HERE IS A RESULT."
	echo
	echo "This board's reset path does not recover the block after a job"
	echo "times out -- the IOMMU says MMU_DTE_ADDR is not functioning and"
	echo "every job after that fails or falls back to the CPU a row at a"
	echo "time, which looks exactly like a clean arm."
	echo
	echo "  REBOOT THE BOARD and run this again. Numbers taken after a"
	echo "  timeout, in this round or the one before it, do not count."
	exit 1
fi
d=${res_default:-}; o=${res_onedev:-}; z=${res_zero:-}
# ⚠ ANY arm firing counts. A fault that shows up only under the timing
# perturbation is still a fault on this hardware; the perturbation is not a
# configuration anyone ships, but it is not a fault the perturbation invented
# either.
two=0
[ -n "$d" ] && [ "$d" -gt 0 ] && two=$((two + d))
[ -n "$z" ] && [ "$z" -gt 0 ] && two=$((two + z))
# ⚠ THE SERIAL ARM IS THE CANDIDATE FIX, so it is reported on its own line
# whatever the rest says -- it is the only arm here that could ship.
sr=${res_serial:-}
if [ -n "$sr" ]; then
	if [ "$sr" -eq 0 ]; then
		echo "serial (both cores, never at once): $RUNS of $RUNS CLEAN"
	else
		echo "serial (both cores, never at once): WRONG $sr of $RUNS --"
		echo "  so overlapping the two submits is not the whole story."
	fi
fi
if [ -z "$o" ]; then
	echo "The onedev arm did not run, so the core pair -- the one story that"
	echo "already explains m = 8 and m = 10 -- was not tested. Run:"
	echo "  CHARSIU_INT_ARMS=\"default onedev zero\" sh $0 $(basename "$MODEL") $RUNS"
elif [ "$two" -gt 0 ] && [ "$o" -eq 0 ]; then
	echo "→ THE CORE PAIR, AT A LOW RATE. Wrong $two time(s) across the two"
	echo "  core arms and $RUNS of $RUNS clean on ONE core. That is the same"
	echo "  fault the dense sweep proved at m = 8 and m = 10 -- two cores on"
	echo "  row 0 of a wide output -- firing rarely at this width instead of"
	echo "  densely. The residual is not a second bug."
	echo "  ⚠ It also means the sweep's \"every even width exact\" is ONE"
	echo "    sample a width, and cannot see a one-in-sixteen fault."
elif [ "$two" -gt 0 ] && [ "$o" -gt 0 ]; then
	echo "→ NOT THE CORE PAIR. Wrong $o of $RUNS on one core as well, so"
	echo "  concurrency is excluded with a rate on both sides and this is a"
	echo "  fault of its own. Next: CHARSIU_DBG_LAYERS=1 on both paths."
elif [ "$two" -eq 0 ] && [ "$o" -eq 0 ]; then
	echo "→ IT DID NOT FIRE, $RUNS runs on each of $ARMS."
	echo "  Not a pass on its own: this fault has gone ten and sixteen runs"
	echo "  clean before. Raise the run count or change the condition"
	echo "  (chunk, gen length, taskset) until it fires, and only then"
	echo "  compare arms."
else
	echo "→ CLEAN ON TWO CORES AND DIRTY ON ONE, which nothing predicts."
	echo "  Suspect the run before the silicon and repeat it."
fi
echo "======================================================================"
