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
# ⚠ int4. An int8 gguf here measures the OTHER path and the run will look fine.
# ⚠ ~/.charsiu/models FIRST: that is where the installer puts the directory it
# chowns to the user, and it is the one charsiu-get fills.
if [ -z "$MODEL" ]; then
	for d in "$HOME/.charsiu/models" "$HOME/models" /opt/charsiu/models; do
		for f in "$d"/*Q4_0*.gguf; do
			[ -f "$f" ] && { MODEL="$f"; break 2; }
		done
	done
fi
[ -n "$MODEL" ] && [ -f "$MODEL" ] \
	|| { echo "prefill_control: no int4 gguf; pass one" >&2; exit 1; }

case "$MODEL" in
*Q4_0*|*q4_0*) ;;
*) echo "prefill_control: $MODEL is not a Q4_0 -- this control is about int4" >&2 ;;
esac

# --- the prompt ------------------------------------------------------------
# The same counting prompt the 19.24 came from: 64 tokens, and the model
# continues the sequence, so a wrong answer is visible without a reference.
PROMPT=$(seq 1 32 | tr '\n' ' ')

D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT

TASK=""
command -v taskset >/dev/null 2>&1 && TASK="taskset -c 4-7"

# ⚠ THE SIX THAT DECIDE WHETHER A PROJECTION REACHES THE NPU AT ALL, and
# KMAX == W4_GROUP or the int4 path is silently not taken.
export CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
       CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
       CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536

# ⚠ AND CHARSIU_NO_BATCH_PREFILL IS NEVER EXPORTED. It is read with getenv()
# and tested for NULL, so an exported empty string means ON -- which is how a
# whole round meant to measure int8 ran int4 instead. The batched half must run
# with the name absent from its environment, not present and empty, so it goes
# on the command as a prefix and nowhere else.
unset CHARSIU_NO_BATCH_PREFILL 2>/dev/null || true

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
	grep -h '^\[' "$D/$_tag.log" >&2 || true
}

one batched1
one control
one batched2

echo
echo "=============== int4 prefill: batched against the token loop ==========="
echo "model   $(basename "$MODEL")   prompt \"1 2 ... 32\"   gen $NGEN"
echo
for t in batched1 control batched2; do
	printf '%-9s %s\n' "$t" "$(grep -h 'prompt .* tok in' "$D/$t.log" \
		| sed 's/.*|\( *prompt[^|]*\)|.*/\1/' | tr -s ' ')"
done
echo
for t in batched1 control batched2; do
	printf '%-9s decode %s\n' "$t" "$(grep -h 'gen .* tok in' "$D/$t.log" \
		| sed 's/.*|\( *gen[^|]*\)|.*/\1/' | tr -s ' ')"
done
echo
# ⚠ THIS IS THE PROOF THE FLAG LANDED, and it is worth more than the rates.
# The refusal is printed by charsiu_npu_matmul, which only the batched path
# calls: the two batched runs must show it and the control must show none. A
# control that still refuses never took the token loop, and a batched run that
# does not refuse took the batched matmul on int4, which is the shipped
# wrong-answer bug coming back.
echo "refusals (batched must be >0, control must be 0)"
for t in batched1 control batched2; do
	printf '%-9s %s\n' "$t" \
		"$(grep -c 'int4 computes one row' "$D/$t.log" || true)"
done
echo
if cmp -s "$D/batched1.txt" "$D/control.txt"; then
	echo "text      IDENTICAL to the control"
else
	echo "text      ⚠ DIFFERS FROM THE CONTROL -- the rate is beside the point"
	diff "$D/control.txt" "$D/batched1.txt" || true
fi
echo "======================================================================="
