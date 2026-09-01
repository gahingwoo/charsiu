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
#   8  CHARSIU_NPU_KFIT         three arms: off / wide buffers only / on
#   9  where TTFT goes           the five-way split, and the pooled read back
#  10  KMAX                      read is m*n*ks, so does a wider slice pay --
#                                and what does the coarser quantiser cost
#                                the share that has no name
#
# Phases 1 to 5 are correctness and any failure stops the round. 6 and 7 are
# numbers and are allowed to disappoint.
#
# `charsiu update dev` installs this at /opt/charsiu/board_verify.sh.
#
#   sh board_verify.sh [PHASES]     default: all of them
#   PHASES is a list like "1 2 3", or "fast" for 1 2 3, or "slow" for 6 7.
set -u

PHASES=${*:-1 2 3 4 5 6 7 8 9 10}
case "$PHASES" in
fast) PHASES="1 2 3" ;;
slow) PHASES="6 7 8 9 10" ;;
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

# ⚠⚠ AND THE LONG ONE TOO, FOR EXACTLY THE SAME REASON THE SHORT ONE IS
# HERE. P9 was built inside phase 9's own case arm, so `board_verify.sh 10`
# -- phase 10 alone, which is how a sweep gets run -- died on its first
# model with "P9: parameter not set". The lesson from hoisting P was that a
# prompt belongs to the ROUND and not to a phase, and P9 was written after
# that and did not get it. set -u is what turned it into an error rather
# than an empty prompt and a table of meaningless numbers.
P9=$(i=1; while [ $i -le 256 ]; do printf '%d ' "$i"; i=$((i+1)); done)
P9=${P9% }

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

8) say "8. KFIT: what the wider buffers cost, and what the slicing buys"
   # ⚠⚠ TURNING KFIT ON DOES TWO INDEPENDENT THINGS, and two board rounds
   # priced them together and could not tell them apart.
   #
   #   1. it widens five buffers to 2 * kmax -- UNCONDITIONALLY, on every
   #      model, including ones where no tensor can fire. in_stride is one of
   #      them, so every K slice sits twice as far from the next in the DMA
   #      buffer, whether or not anything needed the room.
   #   2. it makes the last slice absorb the remainder, the `ks--`, which is
   #      the part that is supposed to pay.
   #
   # The three models that CANNOT fire -- their every K divides KMAX, so the
   # dispatch plan is byte for byte identical in both arms -- came out at
   # -0.5%, +0.0% and -0.9%, and SmolLM2-1.7B read -1.0% and -0.9% in two
   # separate rounds. A reproducible loss on a model where the switch provably
   # changes no dispatch is not noise, it is (1) with none of (2).
   #
   # So there is a third arm: CHARSIU_NPU_KFIT_WIDE widens the buffers and
   # does NOT slice. Now every model is its own control rather than only the
   # three that happen to divide evenly:
   #
   #   wide - off   what the widening COSTS
   #   on   - wide  what the slicing BUYS, with the cost already paid
   #   on   - off   the net, which is what a user would see
   #
   # ⚠ ONE READING IS NOT A MEASUREMENT: --repeat pays the model load once and
   # generates REP times, and the spread is printed under every row.
   run 8
   REP=${KFIT_REPEAT:-3}
   nk=0; nchg=0; ndead=0
   : >"$OUT/.kfit_rows"
   printf '  %-30s %7s %7s %7s %9s  %s\n' model off wide on narrowed text
   for M in "$MODELS"/*Q4_0*.gguf; do
	[ -r "$M" ] || continue
	nk=$((nk + 1))
	b=$(basename "$M" .gguf)
	rates_off=""; rates_wide=""; rates_on=""
	for arm in off wide on; do
		case $arm in
		on)   E=CHARSIU_NPU_KFIT=1 ;;
		wide) E=CHARSIU_NPU_KFIT_WIDE=1 ;;
		*)    E=CHARSIU_KFIT_DUMMY=1 ;;
		esac
		# shellcheck disable=SC2086
		env CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
		    CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
		    CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 $E \
		    "$BIN/charsiu_run" "$M" -p "$P" -n 32 --ignore-eos \
		    --repeat "$REP" >"$OUT/.kfit_$arm" \
		    2>"$OUT/.kfit_$arm.err"
		# ⚠ READ THE RATES BEFORE THE STRIP BELOW EATS THEM, and isolate
		# the GENERATION one. The summary is a single line carrying the
		# prompt rate and the gen rate, the halves line below it carries
		# two more, and the sed that prepares the text for comparison
		# deletes every bracketed line -- so a check written after it
		# greps for what it just removed, and a bare "tok/s" grep
		# collects four rates a repeat and maxes to the first-half one.
		rr=$(grep -hoE 'gen [0-9]+ tok in [0-9.]+ ms, [0-9.]+ tok/s' \
		    "$OUT/.kfit_$arm" "$OUT/.kfit_$arm.err" \
		    | sed 's/.*, //; s/ tok.s//' | tr '\n' ' ')
		case $arm in
		off)  rates_off=$rr ;;
		wide) rates_wide=$rr ;;
		on)   rates_on=$rr ;;
		esac
	done
	hits=$(grep -hoE 'KFIT narrowed [0-9]+ of [0-9]+' "$OUT/.kfit_on.err" \
	    | head -1 | awk '{print $3"/"$5}')
	[ -n "$hits" ] || hits="?"
	sed -i 's/^\[.*//' "$OUT/.kfit_off" "$OUT/.kfit_wide" "$OUT/.kfit_on"
	# ⚠⚠ A SHORT ARM IS A DEAD ARM. An arm that crashed on repeat two of
	# three leaves rates behind, which is not empty, so it reads as healthy
	# -- and its text is then shorter than the others', so the comparison
	# calls it "KFIT changes the answer". That is the wrong diagnosis of a
	# real problem, which is the failure this phase was rewritten after.
	dead=""
	for arm in off wide on; do
		eval "n=\$(printf '%s' \"\$rates_$arm\" | wc -w)"
		[ "$n" -eq "$REP" ] || dead="$arm($n/$REP)"
	done
	if [ -n "$dead" ]; then
		bad "$b: the $dead arm did not generate $REP times"
		tail -6 "$OUT/.kfit_${dead%%(*}.err" | sed 's/^/       /'
		ndead=$((ndead + 1))
		continue
	fi
	bo=$(printf '%s' "$rates_off"  | awk '{m=0;for(i=1;i<=NF;i++)if($i>m)m=$i;printf "%.2f",m}')
	bw=$(printf '%s' "$rates_wide" | awk '{m=0;for(i=1;i<=NF;i++)if($i>m)m=$i;printf "%.2f",m}')
	bn=$(printf '%s' "$rates_on"   | awk '{m=0;for(i=1;i<=NF;i++)if($i>m)m=$i;printf "%.2f",m}')
	if cmp -s "$OUT/.kfit_off" "$OUT/.kfit_on" &&
	   cmp -s "$OUT/.kfit_off" "$OUT/.kfit_wide"; then
		t=same
	else
		t="⚠ DIFFERS"
		nchg=$((nchg + 1))
	fi
	printf '  %-30s %7s %7s %7s %9s  %s\n' "$b" "$bo" "$bw" "$bn" "$hits" "$t"
	printf '      off : %s\n      wide: %s\n      on  : %s\n' \
	    "$rates_off" "$rates_wide" "$rates_on"
	awk -v a="$bo" -v w="$bw" -v n="$bn" 'BEGIN{
	    printf "      buffers %+.1f%%   slicing %+.1f%%   net %+.1f%%\n",
	           (w-a)/a*100, (n-w)/w*100, (n-a)/a*100 }'
	[ "$t" = same ] || { diff "$OUT/.kfit_off" "$OUT/.kfit_on" | head -4 | sed 's/^/       /'; }
	printf '%s %s %s %s %s\n' "$b" "${hits%%/*}" "$bo" "$bw" "$bn" >>"$OUT/.kfit_rows"
   done
   wedged "phase 8" && break
   if [ "$nk" = 0 ]; then
	bad "NO Q4_0 MODEL under $MODELS -- this phase measured nothing"
	printf '     pass CHARSIU_MODELS=/path if they live somewhere else.\n'
   elif [ "$nchg" -gt 0 ]; then
	bad "$nchg of $nk model(s) answer differently. KFIT is a slicing"
	printf '     change and the wide arm changes no slicing at all, so\n'
	printf '     either one moving a token is a bug: keep it off.\n'
   else
	ok "text identical across all three arms on $((nk - ndead)) model(s)."
	printf '     Speed is a separate question:\n\n'
	awk '{ n=$1; h=$2+0; a=$3+0; w=$4+0; u=$5+0
	       cost[++c] = (w-a)/a*100; cname[c]=n; csum += cost[c]
	       if (h==0) { zc++; zg += (u-w)/w*100 }
	       else { fc++; fname[fc]=n; fgain[fc]=(u-w)/w*100
	              fnet[fc]=(u-a)/a*100; fh[fc]=h } }
	     END {
	       printf "     what the WIDER BUFFERS cost, on every model:\n"
	       for (i=1;i<=c;i++) printf "       %-30s %+6.1f%%\n", cname[i], cost[i]
	       printf "       mean %+.1f%% -- paid whether or not anything fires\n\n", csum/c
	       if (zc) printf "     consistency: the %d model(s) that cannot fire gained\n       %+.1f%% mean from the slicing arm, and should have gained 0.\n       That is this round%s measurement error.\n\n", zc, zg/zc, "'"'"'s"
	       printf "     what the SLICING buys, where it fires:\n"
	       worst=999; best=-999
	       for (i=1;i<=fc;i++) {
	         if (fgain[i]<worst) worst=fgain[i]
	         if (fgain[i]>best)  best=fgain[i]
	         printf "       %-30s %+6.1f%%  (net %+.1f%%, %d tensors)\n", \
	                fname[i], fgain[i], fnet[i], fh[i] }
	       print ""
	       if (fc == 0) { print "     ⚠ NOTHING FIRED. This round says nothing about the slicing."; exit }
	       if (csum/c < -0.4)
	         printf "     ⚠ THE WIDENING IS NOT FREE (%+.1f%% mean) and it is paid by\n     every model. Size the buffers by the widest slice actually\n     staged and the net becomes the slicing figure above.\n", csum/c
	       else
	         printf "     the widening is free, so net == slicing and the range is\n     %+.1f%% .. %+.1f%%.\n", worst, best
	     }' "$OUT/.kfit_rows"
	[ "$ndead" = 0 ] || printf '\n  – %s model(s) never generated; see above.\n' "$ndead"
   fi
   ;;

9) say "9. where TTFT goes: the five shares of a batched matmul, and the sixth"
   # ⚠⚠ WHY THIS PHASE AND NOT MORE DECODE WORK. The two gaps to the vendor
   # are not the same size. On Qwen3-0.6B decode is 19.70 tok/s against 24.85,
   # which is 1.26x; time to first token is 1588 ms against 469, which is
   # 3.39x. The distance is in prefill, npudev's own note says the NPU is idle
   # for 91% of a batched matmul, and nothing has ever printed which part of
   # this side of the ioctl that idle time is.
   #
   # ⚠ THE COUNTERS EXISTED AND NOTHING CALLED THEM. charsiu_npu_batch_split
   # and charsiu_npu_batch_prep have been in the tree with no caller anywhere;
   # vision and whisper could only have reached them through
   # charsiu_pool_report, and llama does not call that either. Same shape as
   # the switch that was "written, legal, default off" and corrupted the heap
   # the first time hardware ran it.
   #
   # ⚠⚠ AND THE HOST CANNOT CHECK THIS ONE EITHER. With no /dev/accel the pool
   # has no device, the report suppresses itself and a desk run prints nothing
   # -- which is indistinguishable from the instrument being broken. So the
   # ABSENCE of the block is a FAILURE here, not a quiet skip.
   run 9
   # A long prompt, because a 32 word one spends more time loading than
   # prefilling and the split would be reading noise. -n 4 keeps generation out
   # of the way; TTFT is the number this phase is about.
   printf '  prompt %s words, generation held to 4 tokens\n' \
       "$(printf '%s' "$P9" | wc -w)"
   # ⚠⚠ TWO ARMS, because the read back is now split across the pool and the
   # HOST CANNOT PRICE IT. With no /dev/accel read_rows never runs at all, so a
   # desk comparison of the two arms is two identical serial runs agreeing with
   # each other -- which looks exactly like a verified parallelisation.
   #
   # Rows are disjoint in Y, so the two arms must produce the SAME TEXT. A
   # difference is not a performance result, it is the parallelisation being
   # wrong, and it is checked before the timing is read.
   n9=0; nmiss=0; ntx=0
   ARMS9=serial
   [ "${READ_ARMS:-0}" = 0 ] || ARMS9="serial pool"
   for M in "$MODELS"/*Q4_0*.gguf; do
	[ -r "$M" ] || continue
	n9=$((n9 + 1))
	b=$(basename "$M" .gguf)
	# ⚠ ONE ARM BY DEFAULT. The pooled read back is a settled negative -- it
	# is correct and 5% to 18% slower on every model -- so a normal round
	# should not pay double to re-measure it. READ_ARMS=1 asks for the
	# comparison again, which is what to do after changing the granularity.
	for arm in $ARMS9; do
		case $arm in
		pool)   E=CHARSIU_NPU_POOL_READ=1 ;;
		*)      E=CHARSIU_READ_DUMMY=1 ;;
		esac
		# shellcheck disable=SC2086
		env CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
		    CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
		    CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 $E \
		    "$BIN/charsiu_run" "$M" -p "$P9" -n 4 --ignore-eos \
		    >"$OUT/.ttft_$arm" 2>"$OUT/.ttft_$arm.err"
		eval "t_$arm=\$(grep -hoE 'prompt [0-9]+ tok in [0-9.]+ ms' \
		    '$OUT/.ttft_$arm' | head -1 | grep -oE '[0-9.]+ ms')"
	done
	# ⚠⚠ READ THE ARM THAT ACTUALLY RAN. This copied .ttft_pool
	# unconditionally, and with the pool arm opt-in that file is a STALE
	# LEFTOVER from an earlier run -- so the split printed below would have
	# been another model's, or another round's, with nothing to say so.
	last=${ARMS9##* }
	cp "$OUT/.ttft_$last" "$OUT/.ttft_out"
	cp "$OUT/.ttft_$last.err" "$OUT/.ttft_err"
	tt=$(grep -hoE 'prompt [0-9]+ tok in [0-9.]+ ms, [0-9.]+ tok/s' \
	    "$OUT/.ttft_out" "$OUT/.ttft_err" | head -1)
	printf '\n  %s\n      %s\n' "$b" "${tt:-⚠ no prompt line at all}"
	if [ "$ARMS9" = serial ]; then
		:
	elif sed -i 's/^\[.*//' "$OUT/.ttft_serial" "$OUT/.ttft_pool" &&
	     cmp -s "$OUT/.ttft_serial" "$OUT/.ttft_pool"; then
		printf '      read back: serial %s   pooled %s   (text same)\n' \
		    "${t_serial:-?}" "${t_pool:-?}"
	else
		bad "$b: the pooled read back CHANGES THE TEXT"
		printf '     rows are disjoint in Y, so this is the split being\n'
		printf '     wrong, not a number to weigh against a speed-up.\n'
		diff "$OUT/.ttft_serial" "$OUT/.ttft_pool" | head -4 | sed 's/^/       /'
		ntx=$((ntx + 1))
	fi
	if grep -q "charsiu NPU batched:" "$OUT/.ttft_err"; then
		# from the header until the first line that is not part of the
		# block, so the tail line is included when it is there and
		# nothing is printed twice when it is not
		awk '/charsiu NPU batched:/ { f = 1 }
		     f { if (/charsiu NPU batched:/ || /^    /) print; else exit }' \
		    "$OUT/.ttft_err" | sed 's/^/      /'
	else
		# ⚠ NOT A SKIP. On the board this block is the whole phase, and
		# its absence means either nothing took the batched path or the
		# instrument did not fire -- and those look identical from here,
		# which is exactly why it counts as a failure rather than a
		# blank line someone reads past.
		bad "$b: NO batched breakdown was printed"
		printf '     either no prompt took the batched path, or the\n'
		printf '     counters did not fire. Both need looking at.\n'
		grep -iE "NOT on the NPU|refus|fell back" "$OUT/.ttft_err" \
		    | head -3 | sed 's/^/       /'
		nmiss=$((nmiss + 1))
	fi
   done
   wedged "phase 9" && break
   if [ "$n9" = 0 ]; then
	bad "NO Q4_0 MODEL under $MODELS -- this phase measured nothing"
   elif [ "$ntx" -gt 0 ]; then
	printf '  %s model(s) changed text between the read arms; nothing above\n' "$ntx"
	printf '  is a speed result until that is fixed.\n'
   elif [ "$nmiss" = 0 ]; then
	ok "every model printed its split. The row to act on is the last one:"
	printf '     if `other` is larger than any named share, the next thing to\n'
	printf '     do is NAME it, not optimise one of the five.\n'
   fi
   ;;

10) say "10. KMAX: read is m * n * ks, so buy ks down and see what it costs"
   # ⚠⚠ THIS IS A TRADE, NOT A FREE WIN, AND THE TREE ALREADY KNEW WHY. The
   # read back is m * n * ks and ks is ceil(K / KMAX), so a wider slice is
   # directly less read work. But npudev's own note closes the obvious version
   # of this: ONE DISPATCH CANNOT COVER K WIDER THAN ONE QUANTISATION GROUP --
   # a dispatch produces one number per output channel and nothing in the
   # register map segments the K reduction. So KMAX and CHARSIU_NPU_W4_GROUP
   # have to move together, and moving them COARSENS THE QUANTISER.
   #
   # ⚠ That is why this phase prints text. A wider group is measured at 0.1067
   # relative error per channel against group 32's 0.0666, and round 352's
   # symptom for one absmax over a long row was output that stayed "English, on
   # topic and repetitive". A differing answer here is EXPECTED and is the
   # PRICE; it is not a failure, and the phase does not call it one. What it
   # must not do is show a speed-up without showing what was paid for it.
   #
   # ⚠ AND KMAX WAS MEASURED ONCE ALREADY, at 37% SLOWER, which is why the
   # board settled on 1024. That round is not evidence any more: the reason was
   # `d = (ki * ns + ni) & 1`, slices dealt to devices by an index that
   # restarts per tensor, so a single-slice tensor put everything on device 0.
   # The deal is least-loaded now and this has not been asked since.
   run 10
   SW=${KMAX_SWEEP:-1024 2048}
   printf '  sweeping KMAX = W4_GROUP over: %s\n' "$SW"
   printf '  the first value is the baseline every later one is compared to\n'
   n10=0
   for M in "$MODELS"/*Q4_0*.gguf; do
	[ -r "$M" ] || continue
	n10=$((n10 + 1))
	b=$(basename "$M" .gguf)
	printf '\n  %s\n' "$b"
	first=1
	for K in $SW; do
		env CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
		    CHARSIU_NPU_KMAX="$K" CHARSIU_NPU_W4_GROUP="$K" \
		    CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 \
		    "$BIN/charsiu_run" "$M" -p "$P9" -n 16 --ignore-eos \
		    >"$OUT/.k_out" 2>"$OUT/.k_err"
		# ⚠ READ EVERYTHING OUT BEFORE THE STRIP, which eats the
		# bracketed line the prompt clause lives on.
		tt=$(grep -hoE 'prompt [0-9]+ tok in [0-9.]+ ms' "$OUT/.k_out" \
		     | head -1 | grep -oE '[0-9.]+ ms')
		rd=$(grep -hoE '^ *read +[0-9.]+ ms +[0-9.]+%' "$OUT/.k_err" \
		     | head -1 | tr -s ' ')
		fe=$(grep -hoE '^ *fence +[0-9.]+ ms +[0-9.]+%' "$OUT/.k_err" \
		     | head -1 | tr -s ' ')
		sed -i 's/^\[.*//' "$OUT/.k_out"
		# ⚠⚠ A RUN THAT DID NOT RUN IS NOT A TEXT DIFFERENCE. Without
		# this, a KMAX the hardware refuses leaves an empty .k_out,
		# which compares unequal to the baseline and gets reported as
		# "the quantiser changed the answer" -- a wrong diagnosis of a
		# real problem, which this round has already made once.
		if [ -z "${tt:-}" ]; then
			bad "$b at KMAX $K: no prompt line, the run did not generate"
			tail -4 "$OUT/.k_err" | sed 's/^/       /'
			continue
		fi
		if [ "$first" = 1 ]; then
			cp "$OUT/.k_out" "$OUT/.k_base"
			tx="baseline"
			first=0
		elif cmp -s "$OUT/.k_base" "$OUT/.k_out"; then
			tx="text same"
		else
			# ⚠ NOT A FAILURE. See the note above: a wider group is
			# a coarser quantiser and a different answer is what
			# was bought with the time.
			tx="⚠ TEXT CHANGED -- this is the price"
		fi
		printf '      KMAX %-5s TTFT %-11s %s\n' "$K" "${tt:-?}" "$tx"
		printf '                 %s\n' "${rd:-  read  (no split printed)}"
		printf '                 %s\n' "${fe:-  fence (no split printed)}"
	done
	# the answer at the widest KMAX, so a coarser quantiser can be judged
	# rather than inferred from the word "changed"
	printf '      last answer: %s\n' \
	    "$(tr '\n' ' ' <"$OUT/.k_out" | cut -c1-96)"
   done
   wedged "phase 10" && break
   if [ "$n10" = 0 ]; then
	bad "NO Q4_0 MODEL under $MODELS -- this phase measured nothing"
   else
	ok "$n10 model(s) swept. Read the three together: TTFT, the read row,"
	printf '     and whether the answer changed. A KMAX that halves the read\n'
	printf '     and keeps the text is free; one that changes the text has to\n'
	printf '     be judged on the answer, not on the milliseconds.\n'
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
