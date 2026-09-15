#!/bin/sh
#
# The regression a round runs before it merges to stable: correctness first,
# then the numbers. It lived only on the board as /root/regress.sh until round
# 414, and that is why it carried two holes for as long as it did.
#
#   tests/board_regress.sh [MODEL_DIR] [CHARSIU_DIR]
#
# HOLE ONE, and it is the reason this file exists. The board copy called
#    `sh arch_sanity.sh` with no argument, having exported CHARSIU_BOARD_DIR
#    instead. arch_sanity has REQUIRED $1 since it gained one, so section 1 --
#    "every architecture still knows a fact" -- printed a usage line and
#    nothing else on every run. `set -e` did not stop it, because the call is
#    inside a pipeline to `tail`, so the rest of the regression printed its
#    green sections underneath a heading that had checked nothing. A merge to
#    stable quoted "7/7 architectures" from it.
#
# HOLE TWO, the same shape. The board copy read the clock as
#       cat clk_npu_dsu0 || cat aclk_rknn_root || echo UNREADABLE
#    and the node that exists is clk_rknn_dsu0. So it printed UNREADABLE and
#    carried on. board_clk.sh has done this properly since round 399: it
#    mounts debugfs if it has to and REFUSES if it still cannot read, because
#    every cross round number in this project assumes 594 MHz and a blank where
#    the rate goes is how a round gets compared against a different clock.
#
# Both are the same failure: a check that cannot run reads exactly like a check
# that found nothing wrong.
#
# AND HOLE ONE'S MECHANISM WAS STILL OPEN AFTER HOLE ONE WAS FIXED. r414 gave
# arch_sanity its argument and left the `| tail -16` in place, so `set -e`
# still could not see a failure: a pipeline's status is its LAST command's, and
# tail always succeeds. The argument was the bug of that day; the pipe is what
# made it invisible, and the pipe is what would make the next one invisible
# too.
#
# `run` below puts the output in a file, checks the real status, and then shows
# the slice the section wants. A section that fails now stops the regression
# with the name of what failed, which is the whole point of a harness that
# gates a merge to stable.
set -e
TMPD="${TMPDIR:-/tmp}/regress.$$"
mkdir -p "$TMPD"
trap 'rm -rf "$TMPD"' EXIT INT TERM
nrun=0
# run WHAT SHOW -- command in $1 as a single string, a filter in $2
run() {
	nrun=$((nrun + 1))
	out="$TMPD/sec$nrun"
	if ! sh -c "$1" > "$out" 2>&1; then
		echo "   !! FAILED: $1"
		sed 's/^/   | /' "$out"
		exit 1
	fi
	eval "$2" < "$out"
}
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="${1:-${CHARSIU_BOARD_DIR:-/opt/vendor/models}}"
B="${2:-${CHARSIU_DIR:-/opt/charsiu}}"
RUN="${CHARSIU_RUN:-$B/charsiu_run}"
PPL="${CHARSIU_PPL:-$B/charsiu_ppl}"
# THE SCALAR CONTROL COMES FROM THE ENVIRONMENT TOO, and it did not. Section
# 3 hardcoded "$B/charsiu_run_scalar" and overrode whatever the caller had set,
# so a round that deployed a matching control still measured the installed one.
# Same shape as the CHARSIU_RUN/CHARSIU_RUN_BIN split: setting the knob a
# script does not read is a silent wrong arm. neon_control.sh now refuses a
# control that cannot say its commit, which is how this surfaced.
SCAL="${CHARSIU_RUN_SCALAR:-$B/charsiu_run_scalar}"
M="$B/models/Llama-3.2-1B-Instruct-Q4_0.gguf"
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"

. "$HERE/board_clk.sh"
NPUCLK=$(npu_clk) || exit 1

echo "================ 0. what is being regressed"
echo "   boot    $(cat /proc/sys/kernel/random/boot_id)"
echo "   binary  $RUN  md5 $(md5sum "$RUN" | cut -c1-12)"
echo "   version $("$RUN" --version 2>&1 | head -1)"
echo "   npu clk $NPUCLK Hz"
echo "   models  $DIR"
echo

echo "================ 1. every architecture still knows a fact"
run "CHARSIU_RUN='$RUN' sh '$HERE/arch_sanity.sh' '$DIR'" "tail -16"
echo "   and the ones in $B/models"
run "CHARSIU_RUN='$RUN' sh '$HERE/arch_sanity.sh' '$B/models'" "tail -8"
echo

echo "================ 2. nine models, batched against their own token loop"
run "CHARSIU_RUN='$RUN' sh '$HERE/board_text_all.sh' 8" "grep -E 'gguf|models compared'"
echo

echo "================ 3. the vector kernels against the scalar ones"
# NOT PIPED THROUGH tail. Section 3's whole output when it finds something
# is the FAIL lines and the two sentences under each; tail -14 kept the end of
# the list and could drop the first failures. It also swallows the refusal
# above. Print it, and let the reader see the whole thing.
CHARSIU_RUN="$RUN" CHARSIU_RUN_SCALAR="$SCAL" \
	run "sh '$HERE/neon_control.sh' '$DIR'" "grep -vE '^charsiu: (this thread|cpu[0-9]|the pool)'" 
echo

echo "================ 4. the gelu identity: an exact arm must not move a token"
for m in "$M" "$DIR/gemma-3-1b-it-Q4_0.gguf"; do
	[ -f "$m" ] || { echo "   $(basename "$m")  NOT HERE -- not checked"; continue; }
	a=$(env $E "$RUN" "$m" -p "The capital of France is" -n 12 -q 2>/dev/null | grep -v '^\[')
	b=$(env $E CHARSIU_EXACT_GELU=1 "$RUN" "$m" -p "The capital of France is" -n 12 -q 2>/dev/null | grep -v '^\[')
	if [ "$a" = "$b" ]; then
		echo "   $(basename "$m")  gelu identical"
	else
		echo "   $(basename "$m")  GELU MOVED THE TEXT"
		echo "      $a"; echo "      $b"
	fi
done
echo

echo "================ 5. perplexity, the deterministic instrument"
# NAME BOTH INPUTS, BY md5. A perplexity is not a property of the runtime:
# it belongs to the model file AND the corpus file, and neither is identified
# by a round that just prints a number. The stable merge at 0c85c71 quotes
# "perplexity 32.8025 int4 on the NPU" and this section, on
# Llama-3.2-1B-Instruct-Q4_0 against corpus/long.txt, reads 41.2777 -- the same
# to the digit on two different binaries, so it is not a regression, it is a
# different input that was never written down. long2.txt on the same model and
# the same binary reads 71.2016, which is the size of the effect.
CORPUS="${CHARSIU_PPL_CORPUS:-$B/corpus/long.txt}"
echo "   model  $(basename "$M")  md5 $(md5sum "$M" | cut -c1-12)"
echo "   corpus $(basename "$CORPUS")  md5 $(md5sum "$CORPUS" | cut -c1-12)"
run "env $E '$PPL' '$M' '$CORPUS'" "tail -4"
echo

echo "================ 6. decode"
run "env $E '$RUN' '$M' -p 'The capital of France is' -n 64 --ignore-eos -q -c 1024 -t 4" "grep '^.load'" 
echo
echo "   boot $(cat /proc/sys/kernel/random/boot_id)"
