#!/bin/sh
#
# The regression a round runs before it merges to stable: correctness first,
# then the numbers. It lived only on the board as /root/regress.sh until round
# 414, and that is why it carried two holes for as long as it did.
#
#   tests/board_regress.sh [MODEL_DIR] [CHARSIU_DIR]
#
# ⛔ HOLE ONE, and it is the reason this file exists. The board copy called
#    `sh arch_sanity.sh` with no argument, having exported CHARSIU_BOARD_DIR
#    instead. arch_sanity has REQUIRED $1 since it gained one, so section 1 --
#    "every architecture still knows a fact" -- printed a usage line and
#    nothing else on every run. `set -e` did not stop it, because the call is
#    inside a pipeline to `tail`, so the rest of the regression printed its
#    green sections underneath a heading that had checked nothing. A merge to
#    stable quoted "7/7 architectures" from it.
#
# ⛔ HOLE TWO, the same shape. The board copy read the clock as
#       cat clk_npu_dsu0 || cat aclk_rknn_root || echo UNREADABLE
#    and the node that exists is clk_rknn_dsu0. So it printed UNREADABLE and
#    carried on. board_clk.sh has done this properly since round 399: it
#    mounts debugfs if it has to and REFUSES if it still cannot read, because
#    every cross round number in this project assumes 594 MHz and a blank where
#    the rate goes is how a round gets compared against a different clock.
#
# Both are the same failure: a check that cannot run reads exactly like a check
# that found nothing wrong.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="${1:-${CHARSIU_BOARD_DIR:-/opt/vendor/models}}"
B="${2:-${CHARSIU_DIR:-/opt/charsiu}}"
RUN="${CHARSIU_RUN:-$B/charsiu_run}"
PPL="${CHARSIU_PPL:-$B/charsiu_ppl}"
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
CHARSIU_RUN="$RUN" sh "$HERE/arch_sanity.sh" "$DIR" 2>&1 | tail -16
echo "   and the ones in $B/models"
CHARSIU_RUN="$RUN" sh "$HERE/arch_sanity.sh" "$B/models" 2>&1 | tail -8
echo

echo "================ 2. nine models, batched against their own token loop"
CHARSIU_RUN="$RUN" sh "$HERE/board_text_all.sh" 8 2>&1 | grep -E "gguf|models compared"
echo

echo "================ 3. the vector kernels against the scalar ones"
CHARSIU_RUN="$RUN" CHARSIU_RUN_SCALAR="$B/charsiu_run_scalar" \
	sh "$HERE/neon_control.sh" "$DIR" 2>&1 | tail -14
echo

echo "================ 4. the gelu identity: an exact arm must not move a token"
for m in "$M" "$DIR/gemma-3-1b-it-Q4_0.gguf"; do
	[ -f "$m" ] || { echo "   $(basename "$m")  NOT HERE -- not checked"; continue; }
	a=$(env $E "$RUN" "$m" -p "The capital of France is" -n 12 -q 2>/dev/null | grep -v '^\[')
	b=$(env $E CHARSIU_EXACT_GELU=1 "$RUN" "$m" -p "The capital of France is" -n 12 -q 2>/dev/null | grep -v '^\[')
	if [ "$a" = "$b" ]; then
		echo "   $(basename "$m")  gelu identical"
	else
		echo "   $(basename "$m")  ⛔ GELU MOVED THE TEXT"
		echo "      $a"; echo "      $b"
	fi
done
echo

echo "================ 5. perplexity, the deterministic instrument"
env $E "$PPL" "$M" "$B/corpus/long.txt" 2>&1 | tail -4
echo

echo "================ 6. decode"
env $E "$RUN" "$M" -p "The capital of France is" -n 64 --ignore-eos -q -c 1024 -t 4 2>&1 | grep '^.load'
echo
echo "   boot $(cat /proc/sys/kernel/random/boot_id)"
