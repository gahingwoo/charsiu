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
#   8  CHARSIU_NPU_KFIT         every model, repeated, to settle the default
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

# ⚠⚠ ONE PROMPT FOR THE WHOLE ROUND. `seq 1 32 | tr` leaves a TRAILING SPACE,
# which tokenises differently, and phases 3, 7 and 8 each built their own copy
# -- two with the space and one without. Inside a phase both arms saw the same
# string so the comparisons held, but a tok/s from one phase was not comparable
# to a tok/s from another, and a round has already been read as "gemma4 flipped"
# when what changed was the prompt. Built once here, stripped once, printed
# below so a log says what it measured.
P=$(seq 1 32 | tr '\n' ' '); P=${P% }

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
echo "  prompt    $(printf '%s' "$P" | wc -w) words, no trailing space"
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
		    "$BIN/charsiu_run" "$M" -p "$P" \
		    -n 32 --ignore-eos >"$OUT/.deal_$arm" \
		    2>"$OUT/.deal_$arm.err"
		printf '  %-28s %-6s %s\n' "$b" "$arm" \
		    "$(grep -hoE 'gen [0-9]+ tok in [0-9]+ ms, [0-9.]+ tok/s' "$OUT/.deal_$arm" "$OUT/.deal_$arm.err" | head -1)"
		grep -hoE "the busier core carried [0-9.]+x an even share[^\"]*" \
		    "$OUT/.deal_$arm" "$OUT/.deal_$arm.err" | head -1 \
		    | sed 's/^/      /'
	done
	# ⚠ AND THE TEXT MUST MATCH BETWEEN THE ARMS. A deal that changes the
	# answer is not a scheduling change.
	# ⚠⚠ STDOUT AND STDERR SEPARATELY, OR THE DIAGNOSTICS ARE THE DIFF.
	# The first version captured 2>&1 into the file it then compared, so the
	# index arm's own "CHARSIU_NPU_DEAL_INDEX is set" note and the staging
	# progress lines -- whose millisecond counts differ every run -- made
	# every pair "differ". Two real wins were reported as two failures.
	# charsiu_run puts the generated text on stdout and every diagnostic on
	# stderr, so the split is free.
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

8) say "8. KFIT: the runt K slice, on every model and not just the two it helps"
   # ⚠⚠ WHY IT SHOULD COST NOTHING, WHICH IS NOT OBVIOUS. npuquant falls back
   # to one scale a row when k % grp is non-zero, so a tensor whose K does not
   # divide the slice is ALREADY ungrouped, and a tensor whose K DOES divide
   # has no remainder for KFIT to absorb. The two conditions are complementary,
   # so on the argument KFIT costs no quantisation quality on any tensor.
   #
   # ⚠ THAT ARGUMENT IS STRUCTURAL, AND STRUCTURAL ARGUMENTS HAVE BEEN WRONG
   # HERE. The width law was fitted at nine points, held at all nine, and was
   # still wrong. So this runs EVERY model rather than the two the offline
   # pricing said would gain. A model that gains nothing is the cheap half of
   # the evidence: what can move the default is text identical everywhere.
   #
   # ⚠ AND ONE READING IS NOT A MEASUREMENT. The first round ran each arm once
   # and read gemma-3-1b +5.6% and gemma-4-E2B +0.4%, which is one number above
   # this board's 1% noise band and one number inside it. --repeat pays the
   # model load once and generates REP times, so the arm's figure is the best
   # of REP and the spread is on the record.
   run 8
   REP=${KFIT_REPEAT:-3}
   nk=0; nchg=0; ndead=0; nslow=0
   : >"$OUT/.kfit_rows"
   printf '  %-26s %8s %8s %8s %9s  %s\n' model kfit=off kfit=on delta narrowed text
   for M in "$MODELS"/*Q4_0*.gguf; do
	[ -r "$M" ] || continue
	nk=$((nk + 1))
	b=$(basename "$M" .gguf)
	rates_off=""; rates_on=""
	for arm in off on; do
		case $arm in on) E=CHARSIU_NPU_KFIT=1 ;; *) E=CHARSIU_KFIT_DUMMY=1 ;; esac
		# shellcheck disable=SC2086
		env CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
		    CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
		    CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 $E \
		    "$BIN/charsiu_run" "$M" -p "$P" -n 32 --ignore-eos \
		    --repeat "$REP" >"$OUT/.kfit_$arm" \
		    2>"$OUT/.kfit_$arm.err"
		# ⚠ READ THE RATES BEFORE THE STRIP BELOW EATS THEM. The summary
		# is a bracketed line on STDOUT and the sed that prepares the
		# text for comparison deletes exactly those, so a check written
		# after it greps for a line it has just removed and reports a
		# healthy run as one that never generated a token.
		# ⚠ AND ISOLATE THE GENERATION RATE. The summary is one line,
		# "[load .. | prompt N tok .. R tok/s | gen N tok .. R tok/s |
		# peak N MB]", and the halves line below it carries two more --
		# so a bare grep for "tok/s" collects FOUR rates a repeat and a
		# max over them returns the first-half figure, which is always
		# the highest and is not what this phase compares.
		rr=$(grep -hoE 'gen [0-9]+ tok in [0-9.]+ ms, [0-9.]+ tok/s' \
		    "$OUT/.kfit_$arm" "$OUT/.kfit_$arm.err" \
		    | sed 's/.*, //; s/ tok.s//' | tr '\n' ' ')
		if [ "$arm" = off ]; then rates_off=$rr; else rates_on=$rr; fi
	done
	# ⚠⚠ DID THE SWITCH DO ANYTHING AT ALL. `ks--` needs a REMAINDER, so on
	# a model whose every K is a multiple of KMAX the dispatch plan is byte
	# for byte identical and the delta measured is the MEASUREMENT, not
	# KFIT. The first eight model round scored two such models as losses.
	hits=$(grep -hoE 'KFIT narrowed [0-9]+ of [0-9]+' "$OUT/.kfit_on.err" \
	    | head -1 | awk '{print $3"/"$5}')
	[ -n "$hits" ] || hits="?"
	sed -i 's/^\[.*//' "$OUT/.kfit_off" "$OUT/.kfit_on"
	# ⚠ AN ARM THAT PRODUCED NO RATE DID NOT RUN, and that is a different
	# failure from a text difference. Calling a crash "KFIT changes the
	# answer" is a wrong diagnosis of a real problem, which is worse than
	# either alone. This is how the heap corruption was nearly mislabelled.
	non=$(printf '%s' "$rates_on"  | wc -w)
	noff=$(printf '%s' "$rates_off" | wc -w)
	# ⚠⚠ AND A SHORT ARM IS ALSO A DEAD ARM. An arm that crashed on repeat
	# two of three leaves ONE rate behind, which is not empty, so it reads
	# as healthy -- and its text is then a third the length of the other
	# arm's, so the comparison below calls it "KFIT changes the answer".
	# That is the wrong diagnosis of a real problem, which is the specific
	# failure this phase exists to not repeat. Count the rates.
	if [ "$non" -ne "$REP" ] || [ "$noff" -ne "$REP" ]; then
		if [ "$non" -ne "$REP" ]; then dead=on; n=$non; else dead=off; n=$noff; fi
		bad "$b: the kfit=$dead arm generated $n of $REP times"
		tail -6 "$OUT/.kfit_$dead.err" | sed 's/^/       /'
		ndead=$((ndead + 1))
		continue
	fi
	bo=$(printf '%s' "$rates_off" | awk '{m=0;for(i=1;i<=NF;i++)if($i>m)m=$i;printf "%.2f",m}')
	bn=$(printf '%s' "$rates_on"  | awk '{m=0;for(i=1;i<=NF;i++)if($i>m)m=$i;printf "%.2f",m}')
	d=$(awk -v a="$bo" -v b="$bn" 'BEGIN{printf "%+.1f%%",(b-a)/a*100}')
	if cmp -s "$OUT/.kfit_off" "$OUT/.kfit_on"; then
		t=same
	else
		t="⚠ DIFFERS"
		nchg=$((nchg + 1))
	fi
	printf '  %-26s %8s %8s %8s %9s  %s\n' "$b" "$bo" "$bn" "$d" "$hits" "$t"
	printf '      off: %s\n      on : %s\n' "$rates_off" "$rates_on"
	[ "$t" = same ] || diff "$OUT/.kfit_off" "$OUT/.kfit_on" | head -4 | sed 's/^/       /'
	printf '%s %s %s %s\n' "$b" "${hits%%/*}" "$bo" "$bn" >>"$OUT/.kfit_rows"
	nslow=$((nslow + $(awk -v a="$bo" -v b="$bn" 'BEGIN{print ((b-a)/a*100 < -1.0)?1:0}')))
   done
   wedged "phase 8" && break
   # ⚠⚠ ZERO MODELS IS NOT ZERO DIFFERENCES -- a loop that ran no iterations
   # leaves every counter at 0 and reads exactly like a clean sweep.
   if [ "$nk" = 0 ]; then
	bad "NO Q4_0 MODEL under $MODELS -- this phase measured nothing"
	printf '     pass CHARSIU_MODELS=/path if they live somewhere else.\n'
   elif [ "$nchg" -gt 0 ]; then
	bad "$nchg of $nk model(s) answer differently under KFIT. It is a"
	printf '     slicing change and must not move a token: keep it off.\n'
   else
	ok "text identical on $((nk - ndead)) of $((nk - ndead)) -- the switch"
	printf '     does not change an answer. Speed is a separate question:\n\n'
	# ⚠⚠ THE MODELS KFIT CANNOT TOUCH ARE THIS ROUND'"'"'S PLACEBO. `ks--`
	# needs a remainder, so a model whose every K divides KMAX runs the
	# identical dispatch plan in both arms and its delta is pure
	# measurement -- arm order, thermals, page cache. Averaging those
	# gives the round its own bias, for free, with no extra runs, and the
	# fired models are then read against it rather than against zero.
	# The first eight model round did not have this and scored two
	# untouchable models as losses, which nearly kept the switch off.
	awk '{ n=$1; h=$2+0; o=$3+0; u=$4+0; d=(u-o)/o*100
	       if (h==0) { ib+=d; ic++; iname[ic]=n; idd[ic]=d }
	       else      { fc++; fname[fc]=n; fdd[fc]=d; fh[fc]=h } }
	     END {
	       bias = ic ? ib/ic : 0
	       # ⚠ AND THE PLACEBO SETS THE BAND, not a number typed in here.
	       # The 1% figure came from repeats of ONE arm; what matters for
	       # an off-vs-on delta is how far apart two arms land when the
	       # switch provably did nothing, which is what these models are.
	       for (i=1;i<=ic;i++) { e=idd[i]-bias; if (e<0) e=-e
	                             if (e>band) band=e }
	       if (band < 0.5) band = 0.5
	       printf "     KFIT could not fire on %d model(s) -- the placebo:\n", ic
	       for (i=1;i<=ic;i++) printf "       %-28s %+6.1f%%\n", iname[i], idd[i]
	       if (ic) printf "       mean %+.1f%%, spread +/-%.1f%% <- the round%s own bias and\n       band, measured rather than assumed\n\n", bias, band, "'"'"'s"
	       printf "     it narrowed slices on %d:\n", fc
	       worst = 999; best = -999
	       for (i=1;i<=fc;i++) { c = fdd[i] - bias
	         if (c < worst) worst = c
	         if (c > best)  best  = c
	         printf "       %-28s %+6.1f%% raw   %+6.1f%% corrected   %d tensors\n", \
	                fname[i], fdd[i], c, fh[i] }
	       print ""
	       if (fc == 0) {
	         print "     ⚠ NOTHING FIRED ANYWHERE. This round says nothing about"
	         print "       KFIT at all -- every model divides KMAX evenly. Add a"
	         print "       model whose K does not, or change KMAX." }
	       else if (ic == 0) {
	         printf "     ⚠ NO PLACEBO -- every model fired, so the bias is\n"
	         printf "       unmeasured and %+.1f%% .. %+.1f%% is raw.\n", worst, best }
	       else if (worst < -band)
	         printf "     ⚠ %+.1f%% after correction is past the placebo band of\n     %.1f%%. Not free: it stays a switch.\n", worst, band
	       else if (best < band)
	         printf "     nothing loses past the %.1f%% band -- but nothing WINS past\n     it either, so this buys nothing. Leave it off.\n", band
	       else
	         printf "     nothing loses past the %.1f%% band and the best is %+.1f%%.\n     That is the evidence the default was waiting for.\n", band, best
	     }' "$OUT/.kfit_rows"
	[ "$ndead" = 0 ] || printf '\n  – %s model(s) never generated; see above.\n' "$ndead"
   fi
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
