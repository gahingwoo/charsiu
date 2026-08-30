#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# Are gemma4 and phi3 wrong because of what they ARE, or because of the two
# cores?
#
# ⚠⚠ THE QUESTION THIS ANSWERS, AND WHY IT COMES FIRST.
#
# Two models are refused, and each refusal names the property that
# DISTINGUISHES the model: gemma4 has per layer embeddings, phi3's q, k and v
# are views into one attn_qkv. Both notes say in as many words that this is a
# hypothesis about the cause and only a fact about the model.
#
# There is a standing alternative that has nothing to do with either model.
# m = 8 was four to six orders out on EVERY model, in every arm of every round,
# until CHARSIU_NPU_ONEDEV=1 made it 904 of 904 bit identical -- so this
# hardware has one KNOWN way to be wrong at m > 1 and it is the two cores
# stepping on each other, not the weights. And the two tables that looked at
# gemma4 disagreed with each other on the same prompt, which is what
# intermittent looks like.
#
# If a FORCED phi3 comes back right on one core, both refusals are about the
# wrong thing and the fix is a scheduling fix that would also unrefuse m = 8.
# If it comes back wrong on one core, the core pair is excluded for this class
# with hard data and board_w4_axis.sh is next, to name the tensor.
#
# Three arms a model, and the control is the token loop:
#
#   control  CHARSIU_NO_BATCH_PREFILL=1     what the model actually says
#   forced   CHARSIU_BATCH_FORCE=1          the refused path, both cores
#   onedev   + CHARSIU_NPU_ONEDEV=1         the refused path, one core
#
# ⚠ CHARSIU_BATCH_FORCE IS A PROBE SWITCH. It batches a model this tree
# refuses, says so on stderr and on the summary line, and a number measured
# under it is a number about a model that is still refused.
#
# `charsiu update dev` installs this at /opt/charsiu/board_refused_onedev.sh.
#
#   sh board_refused_onedev.sh [MODEL-or-substring ...]     (default: every
#                                                            refused Q4_0)
set -u

NGEN=${CHARSIU_REFUSED_NGEN:-8}
RUN=${CHARSIU_RUN_BIN:-}
[ -n "$RUN" ] || RUN=$(command -v charsiu_run 2>/dev/null || true)
[ -n "$RUN" ] || for d in /opt/charsiu /usr/bin "$PWD/build" ./build .; do
	[ -x "$d/charsiu_run" ] && { RUN="$d/charsiu_run"; break; }
done
[ -n "${RUN:-}" ] || { echo "charsiu_run not found" >&2; exit 1; }

DIRS="$HOME/.charsiu/models $HOME/models /opt/charsiu/models \
${CHARSIU_BOARD_DIR:-$HOME/charsiu-board}"
# ⚠⚠ AND THIS MACHINE MUST HAVE THE NPU. With no /dev/accel, `matmul_rows`
# falls back to a matvec a row: the batched loop's ORDER runs and the batched
# MATMUL does not, so every arm agrees and the round reads as a pass. That is
# precisely the false pass that let gemma4 and phi3 ship wrong -- six
# architectures, text identical, logits compared, ASAN clean, all of it on a
# machine that could not see the bug. Refuse rather than reassure.
if [ ! -e /dev/accel/accel0 ] && [ -z "${CHARSIU_ALLOW_NO_NPU:-}" ]; then
	echo "======================================================================" >&2
	echo "NO /dev/accel/accel0 -- THIS MACHINE CANNOT ANSWER THIS QUESTION." >&2
	echo "" >&2
	echo "Without the NPU the batched matmul never runs, every arm agrees, and" >&2
	echo "the output reads as a pass. Run this on the board." >&2
	echo "" >&2
	echo "  CHARSIU_ALLOW_NO_NPU=1  runs it anyway, to test the script itself." >&2
	echo "======================================================================" >&2
	exit 1
fi
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

PROMPT="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32"
W4="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"

# ⚠ CASE AND PUNCTUATION FOLDED, because the file is `Phi-3.5-mini-...` and
# the thing anyone types is `phi3`.
norm() { echo "$1" | tr 'A-Z' 'a-z' | tr -cd 'a-z0-9'; }

# --- which models ----------------------------------------------------------
MODELS=""
if [ $# -gt 0 ]; then
	for want in "$@"; do
		if [ -r "$want" ]; then MODELS="$MODELS $want"; continue; fi
		w=$(norm "$want"); hit=
		for d in $DIRS; do
			for f in "$d"/*.gguf; do
				[ -r "$f" ] || continue
				case $(norm "$(basename "$f")") in
				*"$w"*) hit=$f; break 2 ;;
				esac
			done
		done
		[ -n "$hit" ] || { echo "no model matching '$want'" >&2; exit 1; }
		MODELS="$MODELS $hit"
	done
else
	seen=
	for d in $DIRS; do
		[ -d "$d" ] || continue
		for f in "$d"/*.gguf; do
			[ -r "$f" ] || continue
			b=$(basename "$f")
			case " $seen " in *" $b "*) continue ;; esac
			seen="$seen $b"
			case "$b" in *Q4_0*|*q4_0*) MODELS="$MODELS $f" ;; esac
		done
	done
fi
[ -n "${MODELS# }" ] || { echo "no Q4_0 model found in $DIRS" >&2; exit 1; }

echo "binary   $RUN"
echo "prompt   \"1 2 ... 32\", gen $NGEN, int4 on the NPU"
echo

run_arm() { # run_arm <outfile> <extra env...>
	o=$1; shift
	# shellcheck disable=SC2086
	env $W4 "$@" "$RUN" "$MODELS_ONE" -p "$PROMPT" -n "$NGEN" \
		--ignore-eos >"$o.out" 2>"$o.err"
	sed -i 's/^\[.*//' "$o.out"
}

nref=0; ncore=0; nmodel=0; nfine=0
for MODELS_ONE in $MODELS; do
	b=$(basename "$MODELS_ONE")

	# ⚠ ONLY THE REFUSED ONES, AND ASK BEFORE RUNNING ANYTHING ELSE. A model
	# that batches by default has nothing to force, and running it here
	# would produce three arms that agree and say nothing -- which is the
	# exact shape of the Phi-3.5 round that cost a day.
	#
	# ⚠ AND THE CONTROL CANNOT ANSWER THIS. Under
	# CHARSIU_NO_BATCH_PREFILL the diagnostic says so and never reaches the
	# "not batched: <reason>" line, so asking the control why a model is
	# refused always comes back empty. This is a short separate run on the
	# DEFAULT path, which is the only arm that prints the reason.
	# shellcheck disable=SC2086
	env $W4 "$RUN" "$MODELS_ONE" -p "1 2 3 4" -n 1 --ignore-eos \
		>/dev/null 2>"$T/d.err"
	why=$(sed -n 's/.*this model is not batched: //p' "$T/d.err" | head -1)
	if [ -z "$why" ]; then
		printf '%s\n  not refused -- it batches by default, so there is nothing to force here.\n  board_text_all.sh is the check for this one.\n\n' "$b"
		continue
	fi
	nref=$((nref + 1))
	printf '===== %s =====\n' "$b"
	printf '  refused for: %s\n' "$why"

	# ⚠ REPEATS, BECAUSE "AGREES ON THIS RUN" IS NOT A VERDICT. gemma4 has
	# now been called wrong by one table and right by two, on the same
	# prompt, which is either two different builds or an intermittent
	# fault. One number that says how many of N runs agreed settles which.
	REPS=${CHARSIU_REFUSED_REPS:-1}
	run_arm "$T/c" CHARSIU_NO_BATCH_PREFILL=1
	fagree=0; oagree=0; r=1
	while [ "$r" -le "$REPS" ]; do
		run_arm "$T/f" CHARSIU_BATCH_FORCE=1
		run_arm "$T/o" CHARSIU_BATCH_FORCE=1 CHARSIU_NPU_ONEDEV=1
		cmp -s "$T/c.out" "$T/f.out" && fagree=$((fagree + 1))
		cmp -s "$T/c.out" "$T/o.out" && oagree=$((oagree + 1))
		r=$((r + 1))
	done
	[ "$REPS" -eq 1 ] || printf '  over %s runs: two cores agreed %s times, one core %s times\n' \
		"$REPS" "$fagree" "$oagree"

	for a in f o; do
		grep -q "prompt batched" "$T/$a.err" || {
			printf '  ⚠ THE %s ARM DID NOT BATCH -- CHARSIU_BATCH_FORCE is\n' \
			       "$a"
			printf '    not in this binary. `charsiu update dev` and run again.\n'
			tail -4 "$T/$a.err" | sed 's/^/      /'
		}
	done

	# ⚠ EVERY run must agree, not the last one: one disagreement in five is
	# still a wrong answer shipped one prompt in five.
	fok=no; ook=no; sok=no
	[ "$fagree" -eq "$REPS" ] && fok=yes
	[ "$oagree" -eq "$REPS" ] && ook=yes
	# ⚠ AND THE TWO FORCED ARMS AGAINST EACH OTHER. Both wrong and
	# IDENTICAL is a deterministic fault in the model's path; both wrong and
	# DIFFERENT is intermittent, and then concurrency is not excluded no
	# matter what the one core arm said. The first round printed neither.
	cmp -s "$T/f.out" "$T/o.out" && sok=yes
	# ⚠⚠ THE TAIL, NOT THE HEAD. Round one printed `cut -c1-64` of each
	# arm, which on a "1 2 ... 32" prompt is 64 characters of prompt echo:
	# all three lines read identical while cmp said they were not. The
	# generated text -- the only part that can differ -- is at the END.
	tail_of() { tr -d '\n' < "$1" | tail -c "${CHARSIU_REFUSED_TAIL:-56}"; }
	printf '  control (token loop) : ...%s\n' "$(tail_of "$T/c.out")"
	printf '  forced,  two cores   : ...%s   [%s]\n' \
	       "$(tail_of "$T/f.out")" "$fok"
	printf '  forced,  ONE core    : ...%s   [%s]\n' \
	       "$(tail_of "$T/o.out")" "$ook"
	printf '  the two forced arms agree with each other: %s\n' "$sok"
	if [ $fok = no ] || [ $ook = no ]; then
		echo "  --- control against forced, two cores ---"
		diff "$T/c.out" "$T/f.out" | head -8 | sed 's/^/    /'
		echo "  --- control against forced, ONE core ---"
		diff "$T/c.out" "$T/o.out" | head -8 | sed 's/^/    /'
	fi

	if [ $fok = yes ] && [ $ook = yes ]; then
		echo "  → BOTH ARMS AGREE WITH THE CONTROL, $REPS run(s) each."
		if [ "$REPS" -lt 3 ]; then
			echo "    That is not yet a verdict when the fault may be"
			echo "    intermittent. Repeat it:"
			echo "      CHARSIU_REFUSED_REPS=5 sh $0 $b"
		else
			echo "    $REPS clean runs on both arms. If this model was ever"
			echo "    seen wrong, the difference was the BUILD, not the"
			echo "    silicon -- and the refusal can be lifted by a round"
			echo "    that says so."
		fi
		nfine=$((nfine + 1))
	elif [ $fok = no ] && [ $ook = yes ]; then
		echo "  → THE CORE PAIR. Wrong on two cores, right on one. Nothing"
		echo "    about this model is the cause; the refusal is aimed at the"
		echo "    wrong property, and the same scheduling fix should also"
		echo "    unrefuse m = 8."
		ncore=$((ncore + 1))
	elif [ $fok = no ] && [ $ook = no ] && [ $sok = yes ]; then
		echo "  → NOT THE CORE PAIR, AND DETERMINISTIC. Wrong on one core"
		echo "    too, and the two forced arms are byte identical to each"
		echo "    other -- so the same wrong answer comes out however the"
		echo "    work is scheduled. Concurrency is excluded with hard"
		echo "    data. Next:"
		echo "      board_w4_axis.sh $b   -- it names the tensor and the row."
		nmodel=$((nmodel + 1))
	elif [ $fok = no ] && [ $ook = no ]; then
		echo "  → WRONG BOTH WAYS AND WRONG DIFFERENTLY. One core did not"
		echo "    fix it, but the two arms do not agree with each other"
		echo "    either, so this is INTERMITTENT and concurrency is NOT"
		echo "    excluded -- one core still runs two K slices in sequence"
		echo "    through one queue. Run this again with"
		echo "    CHARSIU_REFUSED_REPS=5 before reading anything into it."
		nmodel=$((nmodel + 1))
	else
		echo "  → RIGHT ON TWO CORES AND WRONG ON ONE, which no story"
		echo "    predicts. Suspect the run, not the silicon: check that"
		echo "    both arms loaded the same file and run it again."
	fi
	echo
done

echo "======================================================================"
if [ "$nref" -eq 0 ]; then
	echo "NO REFUSED MODEL WAS FOUND, so nothing was tested. That is not a"
	echo "pass: name one explicitly, e.g."
	echo "  sh board_refused_onedev.sh phi3 gemma4"
	exit 1
fi
echo "$nref refused models forced: $ncore blamed on the core pair, $nmodel"
echo "cleared of it, $nfine agreeing on this run."
echo
echo "⚠ EVERY NUMBER HERE IS FROM A REFUSED MODEL. Nothing in this round"
echo "  unrefuses anything by itself -- it decides WHICH question the next"
echo "  round asks."
echo "======================================================================"
