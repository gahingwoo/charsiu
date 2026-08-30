#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# A refused model, batched anyway, under the conditions that disagree.
#
# ⚠⚠ WHAT THE FIRST TWO ROUNDS ESTABLISHED, because it changed the question
# twice.
#
# Round one asked whether gemma4 and phi3 are wrong because of what they ARE
# (per layer embeddings, weights that are views) or because of the two NPU
# cores. Answer for phi3: NOT the cores -- wrong on one core too. And then
# board_w4_axis.sh said phi3's MATMUL is exact at every width, 225 tensors,
# 18000 of 18000 rows at m = 80, worst relative 1.61e-04. So the views are not
# wrong either, and the fault is somewhere in the batched loop that is NOT a
# matmul: a norm, a rope table, a cache offset -- the shape of thing that
# blk.N.layer_output_scale was, which no refusal named because the loop simply
# never applied it.
#
# Round two, on gemma4: ten clean runs, five on each core count, text identical
# to the token loop. Against ONE round that said its text differs. Same board,
# same binary, same prompt, back to back. The two rounds are not two builds and
# five clean runs is not luck -- they are two CONDITIONS:
#
#   prefill_control.sh   taskset -c 4-7,  -t 4,  -c 512,  -n 16   -> DIFFERS
#   board_text_all.sh    no pinning,   default threads,   -n 8    -> identical
#   round two (this)     no pinning,   default threads,   -n 8    -> identical
#
# A wrong answer that depends on the thread count or the affinity is a race in
# the CPU side of the batched loop, and it would never be found by running the
# same condition again. So this runs BOTH, and the interesting cell is the one
# where they disagree.
#
# ⚠ CHARSIU_BATCH_FORCE IS A PROBE SWITCH. It batches a model this tree
# refuses, says so on stderr and on the summary line, and a number measured
# under it is a number about a model that is still refused.
#
# `charsiu update dev` installs this at /opt/charsiu/board_refused_onedev.sh.
#
#   sh board_refused_onedev.sh [MODEL-or-substring ...]   (default: every
#                                                          refused Q4_0)
#
#   CHARSIU_REFUSED_CONDS="plain pinned"   which conditions to run
#   CHARSIU_REFUSED_REPS=5                 forced arms, this many times each
#   CHARSIU_REFUSED_NGEN=8                 tokens generated in the plain arm
set -u

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

PROMPT=$(seq 1 32 | tr '\n' ' ')
W4="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"
CONDS=${CHARSIU_REFUSED_CONDS:-"plain pinned"}
REPS=${CHARSIU_REFUSED_REPS:-1}
NGEN=${CHARSIU_REFUSED_NGEN:-8}

# ⚠ CASE AND PUNCTUATION FOLDED, because the file is `Phi-3.5-mini-...` and
# the thing anyone types is `phi3`.
norm() { echo "$1" | tr 'A-Z' 'a-z' | tr -cd 'a-z0-9'; }

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
echo "prompt   \"1 2 ... 32\", int4 on the NPU"
echo "conds    $CONDS   (forced arms $REPS time(s) each)"
echo "  plain   no pinning, default threads, gen $NGEN"
echo "  pinned  taskset -c 4-7, -t 4, -c 512, gen 16   <- prefill_control's"
echo

# ⚠ THE TAIL, NOT THE HEAD. Round one printed the first 64 characters of each
# arm, which on a "1 2 ... 32" prompt is 64 characters of prompt echo: all
# three lines read identical while cmp said they were not. The generated text
# -- the only part that can differ -- is at the END.
tail_of() { tr -d '\n' < "$1" | tail -c "${CHARSIU_REFUSED_TAIL:-52}"; }

run_arm() { # run_arm <outfile-stem> <extra env...>
	o=$1; shift
	# shellcheck disable=SC2086
	$PIN env $W4 "$@" "$RUN" "$MODEL" -p "$PROMPT" -n "$NG" \
		--ignore-eos $EXTRA >"$o.out" 2>"$o.err"
	sed -i 's/^\[.*//' "$o.out"
}

nref=0; nbad=0; ncond=0
for MODEL in $MODELS; do
	b=$(basename "$MODEL")

	# ⚠ ONLY THE REFUSED ONES, AND ASK BEFORE RUNNING ANYTHING ELSE. A model
	# that batches by default has nothing to force here.
	#
	# ⚠ AND THE CONTROL CANNOT ANSWER THIS. Under CHARSIU_NO_BATCH_PREFILL
	# the diagnostic says so and never reaches the "not batched: <reason>"
	# line, so asking the control why a model is refused comes back empty.
	# shellcheck disable=SC2086
	env $W4 "$RUN" "$MODEL" -p "1 2 3 4" -n 1 --ignore-eos \
		>/dev/null 2>"$T/d.err"
	why=$(sed -n 's/.*this model is not batched: //p' "$T/d.err" | head -1)
	if [ -z "$why" ]; then
		printf '%s\n  not refused -- it batches by default, so there is\n  nothing to force. board_text_all.sh is its check.\n\n' "$b"
		continue
	fi
	nref=$((nref + 1))
	printf '===== %s =====\n' "$b"
	printf '  refused for: %s\n' "$why"
	modelbad=0

	for COND in $CONDS; do
		case $COND in
		plain)  PIN=""; EXTRA=""; NG=$NGEN ;;
		pinned) PIN=""; EXTRA="-c 512 -t 4"; NG=16
			command -v taskset >/dev/null 2>&1 && PIN="taskset -c 4-7"
			[ -n "$PIN" ] || echo "  (no taskset: the pinned arm is only -t 4 -c 512)" ;;
		*)      echo "  unknown condition '$COND'" >&2; continue ;;
		esac
		ncond=$((ncond + 1))

		run_arm "$T/c" CHARSIU_NO_BATCH_PREFILL=1
		fagree=0; oagree=0; r=1
		while [ "$r" -le "$REPS" ]; do
			run_arm "$T/f" CHARSIU_BATCH_FORCE=1
			run_arm "$T/o" CHARSIU_BATCH_FORCE=1 CHARSIU_NPU_ONEDEV=1
			cmp -s "$T/c.out" "$T/f.out" && fagree=$((fagree + 1))
			cmp -s "$T/c.out" "$T/o.out" && oagree=$((oagree + 1))
			r=$((r + 1))
		done
		# ⚠ EVERY run must agree, not the last one: one disagreement in
		# five is still a wrong answer shipped one prompt in five.
		fok=no; ook=no; sok=no
		[ "$fagree" -eq "$REPS" ] && fok=yes
		[ "$oagree" -eq "$REPS" ] && ook=yes
		# ⚠ AND THE TWO FORCED ARMS AGAINST EACH OTHER. Both wrong and
		# identical is deterministic; both wrong and different is a
		# race, and then one core proves nothing -- it still runs two K
		# slices in sequence through one queue.
		cmp -s "$T/f.out" "$T/o.out" && sok=yes

		printf '  -- %-6s ------------------------------------------------\n' "$COND"
		printf '     control (token loop) : ...%s\n' "$(tail_of "$T/c.out")"
		printf '     forced,  two cores   : ...%s  [%s %s/%s]\n' \
		       "$(tail_of "$T/f.out")" "$fok" "$fagree" "$REPS"
		printf '     forced,  ONE core    : ...%s  [%s %s/%s]\n' \
		       "$(tail_of "$T/o.out")" "$ook" "$oagree" "$REPS"
		printf '     the two forced arms agree with each other: %s\n' "$sok"
		if [ $fok = no ] || [ $ook = no ]; then
			modelbad=1
			echo "     --- control against forced, two cores ---"
			diff "$T/c.out" "$T/f.out" | head -6 | sed 's/^/       /'
		fi
		eval "R_$COND=$fok/$ook/$sok"
	done

	# ⚠ THE CELL WHERE THE CONDITIONS DISAGREE IS THE RESULT. Everything
	# else is one more run of a question already answered.
	echo "  ---------------------------------------------------------------"
	for COND in $CONDS; do
		eval "v=\$R_$COND"
		printf '  %-7s two cores/one core/arms agree: %s\n' "$COND" "$v"
	done
	case " $CONDS " in
	*" plain "*)
	  case " $CONDS " in
	  *" pinned "*)
	    pl=${R_plain%%/*}; pn=${R_pinned%%/*}
	    if [ "$pl" = yes ] && [ "$pn" = no ]; then
		echo "  → IT IS THE CPU SIDE, NOT THE MODEL. Right unpinned at the"
		echo "    default thread count, wrong pinned to four cores with"
		echo "    -t 4. No tensor is a function of the thread count, so"
		echo "    this is a race in the batched loop's own threaded work"
		echo "    -- and the NPU matmul is already known exact here."
		echo "    Next: sweep -t alone (1 2 4 8) with no taskset, then"
		echo "    taskset alone at the default -t. One of the two moves it."
	    elif [ "$pl" = no ] && [ "$pn" = no ]; then
		echo "  → WRONG UNDER BOTH CONDITIONS, so it is not the pinning."
		echo "    The matmul is exact at every width, so the fault is"
		echo "    non matmul work in the batched loop. Next:"
		echo "      CHARSIU_DBG_LAYERS=1 against the token loop -- it is"
		echo "      the tool that found gemma4's three."
	    elif [ "$pl" = yes ] && [ "$pn" = yes ]; then
		echo "  → RIGHT UNDER BOTH CONDITIONS, $REPS run(s) each. If this"
		echo "    model was ever seen wrong, neither condition here"
		echo "    reproduces it -- do not lift the refusal on this; find"
		echo "    the round that saw it and copy its exact command."
	    else
		echo "  → WRONG UNPINNED AND RIGHT PINNED, which is the way round"
		echo "    no story predicts. Suspect the run before the silicon."
	    fi ;;
	  esac ;;
	esac
	[ "$modelbad" -eq 0 ] || nbad=$((nbad + 1))
	echo
done

echo "======================================================================"
if [ "$nref" -eq 0 ]; then
	echo "NO REFUSED MODEL WAS FOUND, so nothing was tested. That is not a"
	echo "pass: name one explicitly, e.g."
	echo "  sh board_refused_onedev.sh phi3 gemma4"
	exit 1
fi
echo "$nref refused models, $ncond condition runs, $nbad models wrong somewhere."
echo
echo "⚠ EVERY NUMBER HERE IS FROM A REFUSED MODEL. Nothing in this round"
echo "  unrefuses anything by itself -- it decides WHICH question the next"
echo "  round asks."
echo "======================================================================"
