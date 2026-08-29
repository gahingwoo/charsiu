#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# prefill_control.sh: what is int4's prefill rate actually made of?
#
# int4 REFUSES the batched matmul -- w4a16 computes one row and five rounds
# established that no register makes it compute two -- so an int4 prompt still
# goes through the projections a token at a time. It measured 19.24 tok/s
# against a decode of 15.46 anyway, and the standing explanation is that a
# prompt needs logits for its last token only, so llama_prefill_batch skips the
# output head n - 1 times and that is the whole of the gap.
#
# That explanation has never been measured. It is written into PLAN.md and into
# a commit message as though it had been. This is the control:
# CHARSIU_NO_BATCH_PREFILL=1 sends the prompt through the token loop, head and
# all, with nothing else different.
#
#   batched -> control -> batched
#
# Two batched samples BRACKET the control, because a board that warms over a
# minute can produce a difference that is not the flag. If the two batched
# numbers disagree by more than the gap being measured, this round says
# nothing and the answer is to run it again cold.
#
# ⚠⚠ THE FIRST VERSION OF THIS MEASURED NOTHING AND LOOKED LIKE IT HAD. It
# picked whatever Q4_0 it found first -- Phi-3.5-mini, whose K and V are fused,
# which llama_prefill_batch refuses outright -- so all three runs took the same
# token loop and returned 4.96, 5.00 and 5.10 tok/s. Three numbers that agree
# are the shape of a result. So: the model is named, the architecture is
# checked out of the run's own mouth before the other two runs happen, and the
# path each run took is a column in the table.
#
# `charsiu update dev` installs this at /opt/charsiu/prefill_control.sh, next
# to the other probes, because that is how this board tests.
#
# Usage: prefill_control.sh [MODEL.gguf] [N_GEN]
set -eu

MODEL=${1:-}
NGEN=${2:-16}

# --- the binary ------------------------------------------------------------
RUN=${CHARSIU_RUN_BIN:-}
[ -n "$RUN" ] || RUN=$(command -v charsiu_run 2>/dev/null || true)
[ -n "$RUN" ] || for d in /opt/charsiu "$HOME/charsiu" .; do
	[ -x "$d/charsiu_run" ] && { RUN="$d/charsiu_run"; break; }
done
[ -n "${RUN:-}" ] || { echo "prefill_control: charsiu_run not found" >&2; exit 1; }

# --- the model -------------------------------------------------------------
# ⚠ int4, and llama, and SAID OUT LOUD. The 19.24 this exists to explain came
# off Llama-3.2-1B-Instruct-Q4_0; another model is a different number that
# cannot be compared to it, and another architecture may not batch at all.
# ~/.charsiu/models first: that is the directory the installer chowns to the
# user and the one charsiu-get fills.
DIRS="$HOME/.charsiu/models $HOME/models /opt/charsiu/models"
if [ -z "$MODEL" ]; then
	for pat in '*Llama-3.2*Q4_0*.gguf' '*llama*Q4_0*.gguf' '*Q4_0*.gguf'; do
		for d in $DIRS; do
			for f in "$d"/$pat; do
				[ -f "$f" ] && { MODEL="$f"; break 3; }
			done
		done
	done
fi
[ -n "$MODEL" ] && [ -f "$MODEL" ] \
	|| { echo "prefill_control: no int4 gguf found in $DIRS -- pass one" >&2
	     exit 1; }
case "$MODEL" in
*Q4_0*|*q4_0*) ;;
*) echo "prefill_control: $MODEL is not a Q4_0, and this control is about int4" >&2
   exit 1 ;;
esac

D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT

# The same counting prompt the 19.24 came from: the model continues the
# sequence, so a wrong answer is visible without a reference.
PROMPT=$(seq 1 32 | tr '\n' ' ')

TASK=""
command -v taskset >/dev/null 2>&1 && TASK="taskset -c 4-7"

# ⚠ THE SIX THAT DECIDE WHETHER A PROJECTION REACHES THE NPU AT ALL, and
# KMAX == W4_GROUP or the int4 path is silently not taken.
export CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
       CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
       CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536

# ⚠ AND CHARSIU_NO_BATCH_PREFILL IS NEVER EXPORTED. It is read with getenv()
# and tested against NULL, so an exported empty string means ON -- which is how
# a whole round meant to measure int8 ran int4 instead. The batched half must
# run with the name ABSENT from its environment, not present and empty, so it
# goes on the command line as a prefix and nowhere else.
unset CHARSIU_NO_BATCH_PREFILL 2>/dev/null || true

echo "model    $MODEL"
echo "prompt   \"1 2 ... 32\",  gen $NGEN,  binary $RUN"
echo

# ⚠ charsiu_run puts the summary line on STDOUT, with the generated text. The
# first version of this grepped stderr for it and printed an empty table.
one() {
	_tag=$1
	echo "== $_tag ==" >&2
	if [ "$_tag" = control ]; then
		CHARSIU_NO_BATCH_PREFILL=1 $TASK "$RUN" "$MODEL" \
			-p "$PROMPT" -n "$NGEN" --ignore-eos -c 512 -t 4 \
			>"$D/$_tag.txt" 2>"$D/$_tag.log"
	else
		$TASK "$RUN" "$MODEL" \
			-p "$PROMPT" -n "$NGEN" --ignore-eos -c 512 -t 4 \
			>"$D/$_tag.txt" 2>"$D/$_tag.log"
	fi
	grep -h '^\[' "$D/$_tag.txt" >&2 || true
	grep -h '^charsiu: prompt' "$D/$_tag.log" >&2 || true
}

# what the run says it did with the prompt, in its own words
path_of() { grep -h '^charsiu: prompt' "$D/$1.log" 2>/dev/null \
		| sed 's/^charsiu: prompt //' | head -1; }
rate_of() { grep -h 'prompt .* tok in' "$D/$1.txt" 2>/dev/null \
		| sed 's/.*|\( *prompt[^|]*\)|.*/\1/' | tr -s ' '; }
gen_of()  { grep -h 'gen .* tok in' "$D/$1.txt" 2>/dev/null \
		| sed 's/.*|\( *gen[^|]*\)|.*/\1/' | tr -s ' '; }
# the generated text alone: the summary lines are on stdout too and they are
# timings, so comparing them as text makes every pair of runs "differ".
text_of() { grep -v '^\[' "$D/$1.txt" | grep -v '^[[:space:]]*$'; }

# ⚠ ONE RUN FIRST, AND STOP IF IT NEVER BATCHED. Three runs of an architecture
# that cannot batch cost four minutes and produce three numbers that agree.
one batched1
case "$(path_of batched1)" in
"batched"*) ;;
*)  echo
    echo "prefill_control: this model never took the batched path."
    echo "  it said: $(path_of batched1)"
    echo "  There is nothing here to control against. Use a model that batches"
    echo "  -- Llama-3.2-1B-Instruct-Q4_0 is what the 19.24 came from."
    exit 1 ;;
esac

one control
one batched2

echo
echo "=============== int4 prefill: batched against the token loop ==========="
printf '%-9s %-38s %s\n' run prompt path
for t in batched1 control batched2; do
	printf '%-9s %-38s %s\n' "$t" "$(rate_of "$t")" "$(path_of "$t")"
done
echo
for t in batched1 control batched2; do
	printf '%-9s decode %s\n' "$t" "$(gen_of "$t")"
done
echo
# ⚠⚠ THIS CHECK'S EXPECTATION INVERTED ON 2026-08-29 AND THE OLD ONE WOULD
# HAVE READ AS A FAILURE.
#
# It used to say "batched must be >0, control must be 0", and it was right
# while int4 refused every batch: a refusal printed by charsiu_npu_matmul was
# proof the batched path had reached the int4 matmul at all, and a batched run
# WITHOUT one meant the shipped wrong-answer bug was back.
#
# int4 batches now, exact at m = 2, 4, 16, 32, 48, 64 and 80, so zero refusals
# is what a correct run looks like and the old sentence would have condemned
# it. What survives is the m = 8 refusal, which is real and which a prompt only
# meets if a chunk lands on that width.
#
# The proof of the batched path is the PATH column and the rate above it, and
# the proof of correctness is the text comparison below. This line is now the
# m = 8 fallback counter and says so.
echo "int4 batch refusals (m=8 falls back; every other width batches)"
for t in batched1 control batched2; do
	printf '%-9s %s\n' "$t" \
		"$(grep -c 'int4 at m=8' "$D/$t.log" || true)"
done
echo
text_of batched1 >"$D/a.txt"; text_of control >"$D/b.txt"
if cmp -s "$D/a.txt" "$D/b.txt"; then
	echo "text      IDENTICAL to the control"
else
	echo "text      ⚠ DIFFERS FROM THE CONTROL -- the rate is beside the point"
	diff "$D/b.txt" "$D/a.txt" || true
fi
echo "======================================================================="
