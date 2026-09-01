#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# Does EVERY model's batched prompt say what its token loop says, ON THE BOARD?
#
# ⚠⚠ WHY THIS EXISTS. On 2026-08-30 gemma4's batched prompt came back 3.5x
# faster and WRONG on the card -- "31 32 1 2 3" where the token loop counted on
# to 35 -- after passing on a desktop: six architectures, text identical to
# their token loops, top-12 logits compared, ASAN clean.
#
# The desktop could not have caught it. With no NPU `matmul_rows` falls back to
# a matvec a row, so the batched loop's ORDER runs and the batched MATMUL does
# not. Every host check exercised the half that was already right.
#
# And prefill_control.sh, which is the check that would have caught it, has
# only ever been pointed at llama -- it prefers *Llama-3.2*Q4_0* by design,
# because the number it exists to explain is llama's. So four models have been
# batching on this board with nobody ever comparing their output.
#
# This runs that comparison for every model present. It is slow and it is the
# only thing standing between "it batches" and "it is right".
#
# `charsiu update dev` installs this at /opt/charsiu/board_text_all.sh.
#
#   sh board_text_all.sh [N_GEN]
set -u

NGEN=${1:-8}
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
# ⚠⚠ KMAX IS NOT PINNED HERE, AND THAT IS THE POINT. This file used to set
# CHARSIU_NPU_KMAX=1024 and CHARSIU_NPU_W4_GROUP=1024 under a comment claiming
# it was "the int4 environment the board actually runs". That stopped being
# true when llama_auto_kmax landed: the runtime now picks the widest K slice
# under which no tensor's grouping changes, which is 4096 on gemma-3-1b and
# 1024 on Phi-3.5, and pinning 1024 turns that off.
#
# A probe that SWEEPS an axis has to pin it. A probe that VERIFIES or SCORES
# the product must not, or it verifies a configuration nobody ships -- which is
# the guard-in-the-probe-not-the-product mistake this tree has already shipped
# a wrong answer behind. Phases 8, 9, 10 and 11 pin it because they are
# sweeping. This one does not, because it is the scoreboard.
# the int4 environment the board actually runs, same as board_vendor.sh
W4="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"

echo "binary   $RUN"
echo "prompt   \"1 2 ... 32\", gen $NGEN, int4 on the NPU"
echo
printf '%-38s %-14s %s\n' model path verdict
printf '%-38s %-14s %s\n' ----- ---- -------

seen=""; nbad=0; nrun=0
for d in $DIRS; do
	[ -d "$d" ] || continue
	for M in "$d"/*.gguf; do
		[ -f "$M" ] || continue
		b=$(basename "$M")
		case " $seen " in *" $b "*) continue ;; esac
		seen="$seen $b"
		case "$b" in *Q4_0*|*q4_0*) ;; *) continue ;; esac
		# ⚠ A SUBSTRING FILTER, for callers isolating one axis. Phase 13
		# sweeps the K slice width and can only do that on models whose
		# K divides none of the candidate widths -- on any other model
		# the weights change with the width and the comparison stops
		# being about slicing.
		if [ -n "${CHARSIU_TEXT_ONLY:-}" ]; then
			case "$b" in *${CHARSIU_TEXT_ONLY}*) ;; *) continue ;; esac
		fi

		# shellcheck disable=SC2086
		env $W4 "$RUN" "$M" -p "$PROMPT" -n "$NGEN" --ignore-eos \
			>"$T/b.out" 2>"$T/b.err"
		# shellcheck disable=SC2086
		env $W4 CHARSIU_NO_BATCH_PREFILL=1 "$RUN" "$M" -p "$PROMPT" \
			-n "$NGEN" --ignore-eos >"$T/c.out" 2>"$T/c.err"
		p=$(grep -oE "prompt batched|prompt a token" "$T/b.err" | head -1)
		nrun=$((nrun + 1))
		# ⚠ strip only the bracketed report lines; the text is the point
		# -- but read the token count off them FIRST, because a prompt
		# whose length nobody prints is how this script and its
		# neighbours came to compare different prompts silently.
		ntok=$(sed -n 's/.*prompt \([0-9]*\) tok in.*/\1/p' "$T/c.out" \
		       | head -1)
		sed -i 's/^\[.*//' "$T/b.out" "$T/c.out"
		if cmp -s "$T/b.out" "$T/c.out"; then
			v="text identical"
		else
			v="⚠ TEXT DIFFERS"
			nbad=$((nbad + 1))
		fi
		printf '%-38s %-14s %s\n' "$b" "${p:-?}" "$v  (${ntok:-?} tok)"
		[ "$v" = "text identical" ] || {
			diff "$T/c.out" "$T/b.out" | head -4 | sed 's/^/     /'
		}
	done
done

echo
if [ "$nrun" -eq 0 ]; then
	echo "⚠ NO Q4_0 MODEL FOUND in $DIRS -- nothing was checked, and that is"
	echo "  not a pass."
	exit 1
fi
echo "$nrun models compared, $nbad differing."
echo "⚠ A model that says 'prompt a token' is REFUSED, not verified: its"
echo "  batched path was never exercised, so 'text identical' means only that"
echo "  the token loop agrees with itself."
[ "$nbad" -eq 0 ] || exit 1
