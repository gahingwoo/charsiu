#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# One round that verifies everything unverified.
#
# ⚠⚠ WHY THIS EXISTS. Seven changes have landed since the last board round and
# not one of them has executed on the hardware -- this desk has no /dev/accel,
# so every number attached to them is construction or a host proxy. Verifying
# them one at a time is how a batch stops being attributable: if the eighth
# round finds a wrong answer, nobody can say which of the seven did it.
#
# So: correctness first, for everything, before a single speed number is read.
# A faster wrong answer is the failure this tree has already shipped once.
#
# What is being verified, and what each phase would catch:
#
#   1  acc_index_check          the width law, asserted on the board itself
#   2  board_text_all           9 models, batched against their own token loop
#   3  the gelu identity        CHARSIU_EXACT_GELU must not change any text
#   4  whisper end to end       the known clip has to come out as the known words
#   5  vision + clip            the towers, against their own cross-checks
#   6  the deal                 least-loaded against CHARSIU_NPU_DEAL_INDEX=1
#   7  the scoreboard           against Rockchip, REPEAT=3 because one reading
#                               of this table has a 25% spread
#   8  CHARSIU_NPU_KFIT         written, legal, default off, NEVER RUN
#
# Phases 1 to 5 are correctness and any failure stops the round. 6 and 7 are
# numbers and are allowed to disappoint.
#
# `charsiu update dev` installs this at /opt/charsiu/board_verify.sh.
#
#   sh board_verify.sh [PHASES]     default: all of them
#   PHASES is a list like "1 2 3", or "fast" for 1 2 3, or "slow" for 6 7.
set -u

PHASES=${*:-1 2 3 4 5 6 7 8}
case "$PHASES" in
fast) PHASES="1 2 3" ;;
slow) PHASES="6 7 8" ;;
esac

BIN=${CHARSIU_BIN_DIR:-/opt/charsiu}
MODELS=${CHARSIU_MODELS:-$HOME/.charsiu/models}
[ -d "$MODELS" ] || MODELS=/opt/charsiu/models
OUT=${CHARSIU_BOARD_DIR:-$HOME/charsiu-board}
mkdir -p "$OUT"
FAIL=0; SKIP=""; RAN=""

say() { printf '\n=========== %s ===========\n' "$*"; }
bad() { printf '  ⚠⚠ %s\n' "$*"; FAIL=$((FAIL + 1)); }
ok()  { printf '  ✓ %s\n' "$*"; }

# ⚠⚠ THE NPU DOES NOT COME BACK FROM A TIMEOUT, and after it wedges every run
# falls back to the CPU a row at a time -- which is CORRECT, so every later
# phase reads CLEAN. A round has already died printing a header and no rows.
# Check dmesg between phases and stop at the first sign.
DMESG0=$(dmesg 2>/dev/null | grep -c "NPU job timed out" || echo 0)
wedged() {
	n=$(dmesg 2>/dev/null | grep -c "NPU job timed out" || echo 0)
	[ "$n" = "$DMESG0" ] && return 1
	bad "THE NPU WEDGED during $1. Everything after this would fall back to"
	printf '     the CPU and read as a pass. Reboot and start again; nothing\n'
	printf '     measured after a timeout counts.\n'
	dmesg 2>/dev/null | tail -4 | sed 's/^/     /'
	return 0
}

have() { [ -x "$BIN/$1" ] || [ -f "$BIN/$1" ]; }
run()  { RAN="$RAN $1"; }
skip() { SKIP="$SKIP $1($2)"; printf '  – skipped: %s\n' "$2"; }

echo "charsiu board verification"
echo "  binaries  $BIN"
echo "  logs      $OUT"
echo "  models    $MODELS"
echo "  phases    $PHASES"
echo "  governor  $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
echo "  ⚠ phase 7 sets the performance governor itself, as their protocol does."

for p in $PHASES; do
case $p in

1) say "1. the width law, asserted on this hardware"
   if have acc_index_check; then
	run 1
	"$BIN/acc_index_check" >"$OUT/verify-acc.txt" 2>&1
	rc=$?
	tail -1 "$OUT/verify-acc.txt" | sed 's/^/  /'
	[ $rc = 0 ] && ok "every width 2..96 agrees with m % 2 == 0" \
	           || bad "the width law does not hold on this board"
   else
	skip 1 "acc_index_check not installed -- charsiu update dev"
   fi ;;

2) say "2. nine models: batched prompt against their own token loop"
   run 2
   sh "$BIN/board_text_all.sh" >"$OUT/verify-text.txt" 2>&1
   rc=$?
   sed -n '/^model /,$p' "$OUT/verify-text.txt"
   if wedged "phase 2"; then break; fi
   if [ $rc = 0 ] && grep -q "0 differing" "$OUT/verify-text.txt"; then
	ok "every model's batched prompt matches its token loop"
   else
	bad "a model's batched prompt does NOT match its token loop"
	printf '     this stops the round: nothing below is worth reading.\n'
	break
   fi ;;

3) say "3. the gelu identity: the exact control must not change any text"
   # ⚠ CHARSIU_EXACT_GELU puts back the tanhf every one of these paths used
   # to run. If it changes a single token, the identity is not an identity
   # here and four commits are wrong. It is the cheapest check in the round.
   run 3
   P=$(seq 1 32 | tr '\n' ' '); P=${P% }
   W4="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"
   nb=0
   nm=0
   for M in "$MODELS"/*Q4_0*.gguf; do
	[ -r "$M" ] || continue
	b=$(basename "$M")
	nm=$((nm + 1))
	# shellcheck disable=SC2086
	env $W4 "$BIN/charsiu_run" "$M" -p "$P" -n 8 --ignore-eos \
	    >"$OUT/.g_fast" 2>/dev/null
	# shellcheck disable=SC2086
	env $W4 CHARSIU_EXACT_GELU=1 "$BIN/charsiu_run" "$M" -p "$P" -n 8 \
	    --ignore-eos >"$OUT/.g_exact" 2>/dev/null
	sed -i 's/^\[.*//' "$OUT/.g_fast" "$OUT/.g_exact"
	if cmp -s "$OUT/.g_fast" "$OUT/.g_exact"; then
		printf '  %-40s same\n' "$b"
	else
		printf '  %-40s ⚠ DIFFERS\n' "$b"
		diff "$OUT/.g_exact" "$OUT/.g_fast" | head -4 | sed 's/^/       /'
		nb=$((nb + 1))
	fi
   done
   if wedged "phase 3"; then break; fi
   # ⚠⚠ ZERO MODELS IS NOT ZERO DIFFERENCES. With no *Q4_0*.gguf under
   # $MODELS the loop above runs no iterations and nb stays 0, which reads
   # as a clean phase having compared nothing at all. That is the exact
   # shape of false pass this round exists to avoid.
   if [ "$nm" = 0 ]; then
	bad "NO Q4_0 MODEL under $MODELS -- this phase compared nothing"
	printf '     pass CHARSIU_MODELS=/path if they live somewhere else.\n'
   elif [ "$nb" = 0 ]; then
	ok "the exponential changes no text, over $nm model(s)"
   else
	bad "$nb of $nm model(s) changed text under CHARSIU_EXACT_GELU"
   fi
   ;;

4) say "4. whisper end to end, and the gelu row that was invisible"
   A=""; MW=""
   for f in "$OUT/jfk.wav" "$HOME/jfk.wav"; do [ -r "$f" ] && A=$f && break; done
   for f in "$OUT"/ggml-tiny*.bin "$HOME"/ggml-tiny*.bin; do
	[ -r "$f" ] && MW=$f && break; done
   if [ -n "$A" ] && [ -n "$MW" ] && have charsiu_whisper; then
	run 4
	sh "$BIN/whisper_transcribe.sh" "$BIN/charsiu_whisper" "$MW" "$A" \
	    >"$OUT/verify-whisper.txt" 2>&1
	rc=$?
	tail -2 "$OUT/verify-whisper.txt" | sed 's/^/  /'
	[ $rc = 0 ] && ok "the known clip came out as the known words" \
	           || bad "whisper's transcript changed"
	# the stage table, so the gelu row can be read against 78 ms
	CHARSIU_STAGES=1 "$BIN/charsiu_whisper" "$MW" --transcribe \
	    --audio "$A" >"$OUT/verify-whisper-stages.txt" 2>&1
	sed -n '/stages/,$p' "$OUT/verify-whisper-stages.txt" | sed 's/^/  /'
   else
	skip 4 "no ggml-tiny*.bin or jfk.wav in $OUT"
   fi ;;

5) say "5. the towers"
   if have board_modalities.sh; then
	run 5
	sh "$BIN/board_modalities.sh" >"$OUT/verify-modalities.txt" 2>&1
	rc=$?
	sed -n '/charsiu vision/,$p' "$OUT/verify-modalities.txt" | sed 's/^/  /'
	if wedged "phase 5"; then break; fi
	[ $rc = 0 ] && ok "the towers ran" || bad "board_modalities failed"
   else
	skip 5 "board_modalities.sh not installed"
   fi ;;

6) say "6. the deal: least loaded against the old index deal"
   # ⚠ THE NUMBER TO READ IS "the busier core carried N.NNx an even share".
   # 1.00 is balanced and 2.00 is one core idle. Qwen3-0.6B's n_embd is 1024,
   # so q, k, v, gate and up are single K slices and the old deal put all five
   # on device 0 -- it should be the model that moves most.
   run 6
   nd=0
   for M in "$MODELS"/Qwen3*Q4_0*.gguf "$MODELS"/gemma-3-1b*Q4_0*.gguf; do
	[ -r "$M" ] || continue
	nd=$((nd + 1))
	b=$(basename "$M")
	for arm in least index; do
		case $arm in index) E=CHARSIU_NPU_DEAL_INDEX=1 ;; *) E=CHARSIU_DEAL_DUMMY=1 ;; esac
		# shellcheck disable=SC2086
		env CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
		    CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
		    CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 $E \
		    "$BIN/charsiu_run" "$M" -p "$(seq 1 32 | tr '\n' ' ')" \
		    -n 32 --ignore-eos >"$OUT/.deal_$arm" 2>&1
		printf '  %-28s %-6s %s\n' "$b" "$arm" \
		    "$(grep -oE 'gen [0-9]+ tok in [0-9]+ ms, [0-9.]+ tok/s' "$OUT/.deal_$arm" | head -1)"
		grep -oE "the busier core carried [0-9.]+x an even share[^\"]*" \
		    "$OUT/.deal_$arm" | head -1 | sed 's/^/      /'
	done
	# ⚠ AND THE TEXT MUST MATCH BETWEEN THE ARMS. A deal that changes the
	# answer is not a scheduling change.
	sed -i 's/^\[.*//' "$OUT/.deal_least" "$OUT/.deal_index"
	cmp -s "$OUT/.deal_least" "$OUT/.deal_index" \
	    && printf '      text identical between the two deals\n' \
	    || bad "$b: the two deals disagree about the ANSWER, not just the speed"
   done
   wedged "phase 6" && break
   [ "$nd" -gt 0 ] || bad "phase 6 found no Qwen3 or gemma-3-1b under $MODELS"
   ;;

7) say "7. the scoreboard, three runs a model"
   # ⚠ REPEAT=3 IS NOT OPTIONAL HERE. TinyLLAMA has read 12.64 and 17.39 tok/s
   # on the same build minutes apart, and a single reading of this table was
   # once read as a 24% regression that did not exist.
   run 7
   CHARSIU_BENCH_PERF=1 CHARSIU_BENCH_REPEAT=3 sh "$BIN/board_vendor.sh" \
	>"$OUT/verify-vendor.txt" 2>&1
   sed -n '/^model /,$p' "$OUT/verify-vendor.txt"
   wedged "phase 7" && break
   ok "scoreboard in $OUT/verify-vendor.txt"
   ;;

8) say "8. KFIT: the runt K slice, on the tensors it is free for"
   # ⚠⚠ WHY IT IS FREE, WHICH IS NOT OBVIOUS. npuquant falls back to one
   # scale a row when k % grp is non-zero, so a tensor whose K does not
   # divide the slice is ALREADY ungrouped -- gemma-3-1b is 1152 and 6912,
   # gemma-4-E2B is 1536 -- and a tensor whose K DOES divide has no remainder
   # for KFIT to absorb. The two conditions are complementary, so KFIT costs
   # no quantisation quality on any tensor it can help. Priced offline at
   # gemma-3-1b -7.9% and gemma-4-E2B -3.5%.
   #
   # ⚠ It has never been run on hardware. The text check is the point of this
   # phase; the tok/s is the reason to look.
   run 8
   nk=0
   for M in "$MODELS"/gemma-3-1b*Q4_0*.gguf "$MODELS"/gemma-4*Q4_0*.gguf; do
	[ -r "$M" ] || continue
	nk=$((nk + 1))
	b=$(basename "$M")
	for arm in off on; do
		case $arm in on) E=CHARSIU_NPU_KFIT=1 ;; *) E=CHARSIU_KFIT_DUMMY=1 ;; esac
		# shellcheck disable=SC2086
		env CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
		    CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
		    CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 $E \
		    "$BIN/charsiu_run" "$M" -p "$(seq 1 32 | tr '\n' ' ')" \
		    -n 32 --ignore-eos >"$OUT/.kfit_$arm" 2>&1
		printf '  %-28s kfit=%-4s %s\n' "$b" "$arm" \
		    "$(grep -oE 'gen [0-9]+ tok in [0-9]+ ms, [0-9.]+ tok/s' "$OUT/.kfit_$arm" | head -1)"
	done
	sed -i 's/^\[.*//' "$OUT/.kfit_off" "$OUT/.kfit_on"
	if cmp -s "$OUT/.kfit_off" "$OUT/.kfit_on"; then
		printf '      text identical with and without KFIT\n'
	else
		bad "$b: KFIT CHANGES THE ANSWER -- it is a slicing change and must not"
		diff "$OUT/.kfit_off" "$OUT/.kfit_on" | head -4 | sed 's/^/       /'
	fi
   done
   wedged "phase 8" && break
   [ "$nk" -gt 0 ] || bad "phase 8 found no gemma model under $MODELS"
   ;;
esac
done

say "verdict"
echo "  ran:     ${RAN:- nothing}"
[ -n "$SKIP" ] && echo "  skipped:$SKIP"
echo
if [ "$FAIL" = 0 ]; then
	echo "  no failure in any phase that ran."
	echo "  ⚠ A phase that was SKIPPED verified nothing. Read the skip list"
	echo "    above before calling this a pass."
else
	echo "  ⚠⚠ $FAIL FAILURE(S). The speed numbers, if any printed, are the"
	echo "     speed of something that is wrong."
fi
echo "  logs: $OUT/verify-*.txt"
exit "$FAIL"
