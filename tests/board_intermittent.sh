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
# ⚠ AND IT RUNS THE CONTROL THAT CAN END IT.
#
# The batched matmul does not zero Y any more. The gather assigns on the FIRST
# contribution to an output range and accumulates after that, tracked by a byte
# per range -- worth 26% of a batched matmul, and the one thing in this path
# that can hand back the caller's STALE buffer. If a range never gets its first
# write, Y keeps the previous token's answer, which is invisible when the stale
# value is close and reads as a sentence about practicing when it is not.
#
#   CHARSIU_NPU_BATCH_ZERO=1 puts the whole-buffer zero back.
#
# Wrong without it and right WITH it, over enough runs to mean something, is
# the first-write bookkeeping. Wrong both ways excludes the whole family.
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
ARMS=${CHARSIU_INT_ARMS:-"default zero"}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
PROMPT=$(seq 1 32 | tr '\n' ' ')
W4="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"

echo "model    $MODEL"
echo "binary   $RUN"
echo "config   chunk $CHUNK, gen $NGEN, $RUNS runs an arm"
echo "arms     $ARMS"
echo "  default  as shipped: the gather assigns on first write, Y is not zeroed"
echo "  zero     CHARSIU_NPU_BATCH_ZERO=1, the whole output buffer zeroed first"
echo

# --- the control, twice, because the token loop must be stable too ---------
# ⚠ IF THE TOKEN LOOP ITSELF IS NOT REPRODUCIBLE then every comparison below
# is against a moving target, and the first thing to fix is not the batching.
i=1
while [ "$i" -le 2 ]; do
	# shellcheck disable=SC2086
	env $W4 CHARSIU_NO_BATCH_PREFILL=1 "$RUN" "$MODEL" -p "$PROMPT" \
		-n "$NGEN" --ignore-eos >"$T/c$i.out" 2>/dev/null
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
echo "control (token loop, reproducible over 2 runs):"
echo "  ...$(tr -d '\n' < "$T/ref.out" | tail -c 56)"
echo

printf '%-9s %-6s %s\n' arm rate 'runs in order (. right, X wrong)'
printf '%-9s %-6s %s\n' --- ---- ----
res_default=""; res_zero=""
for ARM in $ARMS; do
	case $ARM in
	default) EXTRA="" ;;
	zero)    EXTRA="CHARSIU_NPU_BATCH_ZERO=1" ;;
	*) echo "unknown arm '$ARM'" >&2; continue ;;
	esac
	bad=0; marks=""; i=1
	while [ "$i" -le "$RUNS" ]; do
		# shellcheck disable=SC2086
		env $W4 CHARSIU_BATCH_FORCE=1 CHARSIU_PREFILL_CHUNK="$CHUNK" \
			$EXTRA "$RUN" "$MODEL" -p "$PROMPT" -n "$NGEN" \
			--ignore-eos >"$T/b.out" 2>/dev/null
		sed -i 's/^\[.*//' "$T/b.out"
		if cmp -s "$T/ref.out" "$T/b.out"; then
			marks="$marks."
		else
			marks="${marks}X"; bad=$((bad + 1))
			cp "$T/b.out" "$T/worst-$ARM.out"
		fi
		i=$((i + 1))
	done
	printf '%-9s %-6s %s\n' "$ARM" "$bad/$RUNS" "$marks"
	[ "$bad" -eq 0 ] || printf '          a wrong one: ...%s\n' \
		"$(tr -d '\n' < "$T/worst-$ARM.out" | tail -c 56)"
	eval "res_$ARM=$bad"
done

echo
echo "======================================================================"
d=${res_default:-}; z=${res_zero:-}
if [ -z "$d" ] || [ -z "$z" ]; then
	echo "Only one arm ran, so there is no control. Run both:"
	echo "  CHARSIU_INT_ARMS=\"default zero\" sh $0 $(basename "$MODEL") $RUNS"
elif [ "$d" -gt 0 ] && [ "$z" -eq 0 ]; then
	echo "→ THE FIRST WRITE BOOKKEEPING. Wrong $d of $RUNS as shipped and"
	echo "  RIGHT $RUNS of $RUNS with the output buffer zeroed. An output"
	echo "  range is not getting its first write, so Y keeps the previous"
	echo "  token's answer. The fix is in the flag, not the matmul -- the"
	echo "  matmul is already known exact at every width."
	echo "  ⚠ $RUNS clean runs is evidence, not proof, for a fault that has"
	echo "    already gone ten runs without firing. Raise RUNS before"
	echo "    changing code on the strength of it."
elif [ "$d" -gt 0 ] && [ "$z" -gt 0 ]; then
	echo "→ NOT THE ZERO. Wrong $d of $RUNS as shipped and $z of $RUNS with"
	echo "  the buffer zeroed, so stale output is excluded and the whole"
	echo "  assign-on-first-write family with it. What is left is the"
	echo "  batched loop's non matmul work. Next:"
	echo "    CHARSIU_DBG_LAYERS=1 on both paths -- it names the layer, and"
	echo "    it is the tool that found gemma4's three."
elif [ "$d" -eq 0 ] && [ "$z" -eq 0 ]; then
	echo "→ IT DID NOT FIRE. $RUNS clean runs on both arms. This fault has"
	echo "  gone ten runs clean before, so this is not a pass -- raise the"
	echo "  run count, or change the condition (chunk, gen length, taskset)"
	echo "  until it fires, and only then compare arms."
else
	echo "→ WRONG ONLY WITH THE BUFFER ZEROED, which no story predicts."
	echo "  Suspect the run before the silicon and repeat it."
fi
echo "======================================================================"
