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
#  11  the prefill chunk         we chunk at 32, the vendor's ladder tops at 80
#  12  the quality probe         phase 10's counting prompt cannot see a
#                                coarser quantiser; this asks something that can
#  13  where the wide slice      the batched path disagrees with the token loop
#      stops being correct       above KMAX 1024; find the width it breaks at
#  14  int8 at the same widths   is the fault int4's, or every dispatch's? one
#                                shape at a time against an exact CPU reference
#  15  one int4 matmul at a time  the direct instrument, on the real emitter:
#                                is the bound K, or is it K*N?
#  16  one tensor, SLICED        the missing rung: several K slices accumulated,
#                                through npudev, against the CPU
#  17  where the CBUF window     0x1c00 is 7168 and reads like window 1's base.
#      actually ends             walk the surface across it and find out
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

PHASES=${*:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22}
case "$PHASES" in
fast) PHASES="1 2 3" ;;
slow) PHASES="6 7 8 9 10 11" ;;
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
# ⚠⚠ ONCE, AND THEN TRUE FOREVER. This is called from inside a sweep (to stop
# it at the cell that wedged) and again after it (to skip the phase's ok line),
# so it must report exactly once and keep saying yes afterwards. Without the
# WEDGED latch the second call would count a second failure for the same event.
WEDGED=0
wedged() {
	n=$(dmesg 2>/dev/null | grep -c "NPU job timed out" || echo 0)
	[ "$WEDGED" = 1 ] && return 0
	[ "$n" = "$DMESG0" ] && return 1
	WEDGED=1
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
# ⚠⚠ SAY WHICH COPY OF THIS SCRIPT IS RUNNING. A round has already been read as
# new data when it was the previous version of this file: `charsiu update dev`
# had not taken, the output was byte identical to the round before, and the only
# thing that gave it away was one word in a header nobody was checking. A stale
# probe is a wasted board trip and it looks exactly like a result.
echo "  script    $(cksum "$0" 2>/dev/null | cut -d" " -f1), $(wc -l <"$0") lines"
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
   	# ⚠ INSIDE the sweep: a wedged block makes every later cell
   	# read as garbage, and this loop printed two of them as
   	# failures before the check below ever ran.
   	wedged "phase 6" && break
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
   	# ⚠ INSIDE the sweep: a wedged block makes every later cell
   	# read as garbage, and this loop printed two of them as
   	# failures before the check below ever ran.
   	wedged "phase 8" && break
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
   	# ⚠ INSIDE the sweep: a wedged block makes every later cell
   	# read as garbage, and this loop printed two of them as
   	# failures before the check below ever ran.
   	wedged "phase 9" && break
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
   # ⚠⚠ THIS PHASE COMPARES BATCHED AGAINST BATCHED, AND SO DOES PHASE 12.
   # Both arms take the same path, so neither can see the batched path
   # disagreeing with the model's own token loop -- and at KMAX 4096 it DOES,
   # on Qwen2.5 and gemma-3-1b, which this phase and phase 12 both called
   # identical. Only phase 2 asks that question. Read a "text same" here as
   # "the quantiser did not move", never as "this width is correct".
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
   # ⚠ 4096 IS IN THE DEFAULT BECAUSE THE VENDOR USES IT. Read off its own
   # .rkllm, every one of its 3328 int4 dispatches is K = 2048 (81%) or
   # K = 4096 (19%) and none is 1024, which is what these rounds have run.
   SW=${KMAX_SWEEP:-1024 2048 4096}
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
   	# ⚠ INSIDE the sweep: a wedged block makes every later cell
   	# read as garbage, and this loop printed two of them as
   	# failures before the check below ever ran.
   	wedged "phase 10" && break
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

11) say "11. the prefill chunk: we use 32, the vendor's widest is 80"
   # The vendor's int4 M ladder, read off its own .rkllm, is 1, 16, 24, 32, 40,
   # 48, 64 and 80 -- and 80 is both the widest and the most common, 768 of
   # 3328. charsiu chunks a prompt at 32. A wider chunk does not change the
   # total read work, which is rows * n * ks, but it amortises everything
   # charged per CALL: prep, the submits, and the fence's own ramp.
   #
   # ⚠ 96 IS IN THE SWEEP ON PURPOSE, ABOVE WHAT THE VENDOR EMITS. This tree
   # has "the batch stops at m = 80" on record from the vision work, and a
   # sweep that stops where the vendor stops cannot tell a hardware ceiling
   # from a choice they made. If 96 comes back with different text, that is
   # the ceiling and it is a correctness finding, not a slow arm.
   run 11
   CH=${CHUNK_SWEEP:-32 64 80 96}
   printf '  sweeping CHARSIU_PREFILL_CHUNK over: %s\n' "$CH"
   printf '  ⚠ the chunk is a MAXIMUM and is rounded down to an expressible\n'
   printf '    width, so read the "chunks of" line, not the value asked for\n'
   n11=0; nbad11=0
   for M in "$MODELS"/*Q4_0*.gguf; do
	[ -r "$M" ] || continue
	n11=$((n11 + 1))
	b=$(basename "$M" .gguf)
	printf '\n  %s\n' "$b"
	first=1
	for C in $CH; do
		env CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
		    CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
		    CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 \
		    CHARSIU_PREFILL_CHUNK="$C" \
		    "$BIN/charsiu_run" "$M" -p "$P9" -n 8 --ignore-eos \
		    >"$OUT/.c_out" 2>"$OUT/.c_err"
		tt=$(grep -hoE 'prompt [0-9]+ tok in [0-9.]+ ms' "$OUT/.c_out" \
		     | head -1 | grep -oE '[0-9.]+ ms')
		# what it ACTUALLY batched at, not what was asked for
		wd=$(grep -hoE 'chunks of [0-9]+[^)]*' "$OUT/.c_out" "$OUT/.c_err" \
		     | head -1)
		sed -i 's/^\[.*//' "$OUT/.c_out"
		if [ -z "${tt:-}" ]; then
			bad "$b at chunk $C: no prompt line, the run did not generate"
			tail -4 "$OUT/.c_err" | sed 's/^/       /'
			nbad11=$((nbad11 + 1))
			continue
		fi
		if [ "$first" = 1 ]; then
			cp "$OUT/.c_out" "$OUT/.c_base"; tx="baseline"; first=0
		elif cmp -s "$OUT/.c_base" "$OUT/.c_out"; then
			tx="text same"
		else
			# ⚠ UNLIKE PHASE 10, A DIFFERENCE HERE IS A FAILURE. The
			# chunk width changes no arithmetic -- same weights, same
			# scales, same order -- so the answer must not move. This
			# is the m = 8 class of fault, which is what that phase
			# and the width law exist for.
			bad "$b at chunk $C: THE ANSWER MOVED. A chunk width changes"
			printf '     no arithmetic, so this is a layout fault, not a trade.\n'
			diff "$OUT/.c_base" "$OUT/.c_out" | head -4 | sed 's/^/       /'
			tx="⚠ DIFFERS"
			nbad11=$((nbad11 + 1))
		fi
		printf '      chunk %-4s TTFT %-11s %-12s %s\n' \
		    "$C" "$tt" "$tx" "${wd:-}"
	done
   	# ⚠ INSIDE the sweep: a wedged block makes every later cell
   	# read as garbage, and this loop printed two of them as
   	# failures before the check below ever ran.
   	wedged "phase 11" && break
   done
   wedged "phase 11" && break
   if [ "$n11" = 0 ]; then
	bad "NO Q4_0 MODEL under $MODELS -- this phase measured nothing"
   elif [ "$nbad11" = 0 ]; then
	ok "every chunk width gave the same answer on $n11 model(s); the TTFT"
	printf '     column is then a free choice.\n'
   fi
   ;;

12) say "12. the quality probe: something a coarser quantiser can actually break"
   # ⚠⚠ WHY THIS EXISTS. Phase 10 compares text across KMAX and reported "text
   # same" on all eight models -- and that is not evidence, because its prompt
   # is "1 2 3 ... 256" and the continuation is the model counting. Counting is
   # the least quantisation-sensitive thing a language model does. Phi-3.5 goes
   # from grouped-at-1024 to a per-row scale at KMAX 2048, which is exactly the
   # degradation that was predicted, and it counted through it.
   #
   # So: short prompts whose continuation is a real language choice, and a
   # symptom counter for the failure mode this tree has on record. Round 352's
   # description of one absmax over a long row was output that stayed "English,
   # on topic and REPETITIVE", so distinct words over total words is a direct
   # probe for it.
   #
   # ⚠ THE RATIO IS A SYMPTOM DETECTOR, NOT AN ORACLE. It cannot tell you the
   # answer is right; it can tell you the model started repeating itself, which
   # is the shape the failure takes here. Read it next to the text, never
   # instead of it.
   run 12
   SW12=${KMAX_SWEEP:-1024 2048 4096}
   n12=0
   for M in "$MODELS"/*Q4_0*.gguf; do
	[ -r "$M" ] || continue
	n12=$((n12 + 1))
	printf '\n  %s\n' "$(basename "$M" .gguf)"
	for QP in "The capital of France is" \
		  "The opposite of hot is" \
		  "Water is made of hydrogen and"; do
		printf '      "%s"\n' "$QP"
		for K in $SW12; do
			env CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
			    CHARSIU_NPU_KMAX="$K" CHARSIU_NPU_W4_GROUP="$K" \
			    CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 \
			    "$BIN/charsiu_run" "$M" -p "$QP" -n 24 --ignore-eos \
			    >"$OUT/.q_out" 2>/dev/null
			sed -i 's/^\[.*//' "$OUT/.q_out"
			r=$(tr ' ' '\n' <"$OUT/.q_out" | sed '/^$/d' \
			    | awk '{n++; if (!seen[$0]++) u++}
				   END { if (n) printf "%d/%d", u, n; else printf "0/0" }')
			printf '        KMAX %-5s distinct %-8s %s\n' "$K" "$r" \
			    "$(tr '\n' ' ' <"$OUT/.q_out" | cut -c1-70)"
		done
	done
   	# ⚠ INSIDE the sweep: a wedged block makes every later cell
   	# read as garbage, and this loop printed two of them as
   	# failures before the check below ever ran.
   	wedged "phase 12" && break
   done
   wedged "phase 12" && break
   [ "$n12" -gt 0 ] || bad "NO Q4_0 MODEL under $MODELS -- this phase measured nothing"
   ok "read the distinct ratio and the text together. A ratio that collapses as"
   printf '     KMAX widens is the recorded symptom of one scale over too long a\n'
   printf '     row; a ratio that holds while the words go wrong is a different\n'
   printf '     fault and this phase cannot name it.\n'
   ;;

13) say "13. the wide K slice: does the CBUF pair fix make it agree with m = 1"
   # ⚠⚠ CORRECTION: NOTHING WAS FIXED TO GET 2048, IT WAS NEVER BROKEN. The
   # commit that claimed credit put the split CBUF rule into regcmd.c, whose
   # only caller in this tree is tools/emit_dump.c -- the runtime submits
   # through charsiu_emit_job in job.c, which already had the rule and had it
   # right: `split = wide && surf * rows > 4096`, exact on all 3328 of the
   # vendor's int4 streams. 2048 had simply never been compared against the
   # token loop before this phase existed.
   #
   # What this phase measures, then, is not a fix. It is the first honest
   # reading of the axis, and it says:
   #
   #     slice 2816   WRONG   Qwen2.5 at KMAX 3072, surf 88
   #     slice 3072   right   gemma-3-1b at KMAX 3072, surf 96
   #     slice 4096   WRONG   both, surf 128
   #
   # ⚠⚠ THAT IS NOT A SIZE THRESHOLD. 3072 is wider than 2816 and works, so
   # whatever is wrong above 2048 is not "too big" and the next round should
   # not be a bigger sweep of the same axis. K * M does not separate them
   # either: 245760 works and 225280 does not.
   #
   # The old note, still true of the half that IS fixed: regcmd.c emitted the non split CBUF pair
   # unconditionally, and the vendor's own batched dispatches switch to the
   # split pair on K * M > 131072 -- 2048x64 and 4096x32 are its widest non
   # split, both exactly 131072. Our shipped setting is 1024 x 80 = 81920,
   # which is why every round for months was right, and 4096 x 80 = 327680,
   # which is why phase 2 was not.
   # ⚠⚠ WHAT THIS IS FOR. Phase 2 at KMAX 4096 had Qwen2.5 and gemma-3-1b
   # answering differently from their own token loop, while SmolLM2-135M was
   # fine -- and phases 10 and 12 had called all three identical at that width,
   # because both of THEIR arms were batched. A batched path that is wrong at a
   # wide slice is invisible to a batched-versus-batched comparison.
   #
   # The discriminator is the widest single dispatch, not the number of slices:
   #
   #   SmolLM2-135M   576 / 1536   at 4096 every tensor is ks = 1, widest 1536   ok
   #   Qwen2.5-1.5B  1536 / 8960   at 4096 the ffn is ks = 3, widest 4096        wrong
   #   gemma-3-1b    1152 / 6912   at 4096 the ffn is ks = 2, widest 4096        wrong
   #
   # so the bound is somewhere in (1536, 4096], and the vendor's own .rkllm
   # dispatches K = 2048 and K = 4096 -- the hardware does this, our encoder
   # does not.
   #
   # ⚠ ONLY THESE THREE MODELS, AND THAT IS THE WHOLE DESIGN. K must divide
   # none of the candidate widths, or the WEIGHTS change with the width and the
   # comparison stops being about slicing at all. 1536, 8960, 1152, 6912, 576
   # and 1536 divide none of 1024, 2048, 3072, 4096, so across this sweep the
   # quantiser emits the same bytes and only ks moves.
   # ⚠⚠ AND A SECOND ARM, BECAUSE PHASE 15 EXONERATED THE DISPATCH. One int4
   # matmul at K=4096 N=1536, m=80, through the same emitter, is EXACT -- and
   # so is every other shape these models run. So the fault is not in the
   # register stream for one dispatch; it is in something the model does and a
   # single call does not, and the shortest list is: several K slices
   # accumulated, two cores, and the buffers shared across both.
   #
   # CHARSIU_NPU_ONEDEV=1 removes one of those three outright. If 3072 and 4096
   # come back correct on one core, the fault is in the pair -- which this tree
   # has seen before, in the batched submits that corrupted each other. If they
   # stay wrong, the pair is innocent and it is the slice accumulation.
   run 13
   KW=${KMAX_WIDTHS:-1024 2048 3072 4096}
   # ⚠⚠ THE CHUNK IS THE SECOND AXIS NOW, and it is the sharper question.
   # onedev already answered its own: Qwen2.5 fails identically on one core and
   # two, so the core pair is not necessary for the fault. What has never been
   # run is a WIDE K SLICE AT A NARROW m.
   #
   # It matters because `split` -- the only thing in the emitter keyed on m at
   # all -- fires on surf * m > 4096, and surf is K/32:
   #
   #     KMAX 2048 chunk 32 -> 2048   no split      chunk 80 -> 5120   split
   #     KMAX 3072 chunk 32 -> 3072   no split      chunk 80 -> 7680   split
   #     KMAX 4096 chunk 32 -> 4096   no split      chunk 80 -> 10240  split
   #
   # so at a chunk of 32 none of these widths splits. Phase 11 already has
   # (1024, 32) and (1024, 80) both correct, and phase 13 has (3072, 80) wrong.
   # (3072, 32) is the cell nobody has run, and it separates "the width is
   # wrong" from "the width is wrong when m is wide too".
   A13=${K_ARMS:-m80 m32}
   printf '  KMAX = W4_GROUP over %s, on the three models whose weights\n' "$KW"
   printf '  do not move across it. Batched against the token loop.\n'
   printf '  arms: %s\n' "$A13"
   printf '        m80 = CHARSIU_PREFILL_CHUNK=80, m32 = 32, onedev = one core\n'
   n13=0
   for MK in Qwen2.5-1.5B gemma-3-1b SmolLM2-135M; do
	ls "$MODELS"/*"$MK"*Q4_0*.gguf >/dev/null 2>&1 || continue
	n13=$((n13 + 1))
	# ⚠⚠ gemma-3-1b IS AN INTERMITTENT REPRODUCER AND ITS CELLS ARE NOISE.
	# It read WRONG at KMAX 4096 on one round and RIGHT on the next with a
	# byte-identical runtime -- every commit between the two touched only
	# tests and comments. Its two arms also inverted against each other in
	# the round before that. A hypothesis was fitted to those cells and was
	# wrong, so REP13 runs each cell more than once and a cell that
	# disagrees with itself is reported as unstable rather than as a bound.
	# Qwen2.5-1.5B has been consistent in every round and is the reproducer.
	printf '\n  %s\n' "$MK"
	for ARM in $A13; do
		case $ARM in
		onedev) E13=CHARSIU_NPU_ONEDEV=1 ;;
		m32)    E13=CHARSIU_PREFILL_CHUNK=32 ;;
		m80)    E13=CHARSIU_PREFILL_CHUNK=80 ;;
		*)      E13=CHARSIU_DEV_DUMMY=1 ;;
		esac
		for K in $KW; do
			# shellcheck disable=SC2086
			ok13=0; no13=0; err13=0
			i13=0
			while [ "$i13" -lt "${REP13:-2}" ]; do
				i13=$((i13 + 1))
				# shellcheck disable=SC2086
				r=$(env CHARSIU_TEXT_ONLY="$MK" CHARSIU_NPU_KMAX="$K" \
				    CHARSIU_NPU_W4_GROUP="$K" $E13 \
				    sh "$BIN/board_text_all.sh" 2>&1 \
				    | grep -E "text identical|TEXT DIFFERS" | head -1)
				case "$r" in
				*identical*) ok13=$((ok13 + 1)) ;;
				*DIFFERS*)   no13=$((no13 + 1)) ;;
				*)           err13=$((err13 + 1)) ;;
				esac
			done
			if [ "$err13" != 0 ]; then
				bad "$MK at KMAX $K on $ARM: $err13 run(s) gave no verdict"
			elif [ "$no13" = 0 ]; then
				printf '      %-7s KMAX %-5s agrees, %s of %s\n' \
				    "$ARM" "$K" "$ok13" "$i13"
			elif [ "$ok13" = 0 ]; then
				printf '      %-7s KMAX %-5s ⚠ DISAGREES, %s of %s\n' \
				    "$ARM" "$K" "$no13" "$i13"
				bad "$MK breaks at a K slice of $K on $ARM"
			else
				# ⚠ NOT A BOUND. A cell that disagrees with itself
				# is the model being flaky, and fitting anything
				# to it is how the last two hypotheses died.
				printf '      %-7s KMAX %-5s ⚠⚠ UNSTABLE: %s right, %s wrong of %s\n' \
				    "$ARM" "$K" "$ok13" "$no13" "$i13"
				bad "$MK at KMAX $K on $ARM is not repeatable"
			fi
		done
	done
   	# ⚠ INSIDE the sweep: a wedged block makes every later cell
   	# read as garbage, and this loop printed two of them as
   	# failures before the check below ever ran.
   	wedged "phase 13" && break
   done
   wedged "phase 13" && break
   if [ "$n13" = 0 ]; then
	bad "none of the three models is under $MODELS -- nothing was swept"
   else
	ok "the first width that disagrees is the bound. Everything below it is"
	printf '     a slice the batched path encodes correctly; the vendor reaches\n'
	printf '     4096, so what is above the bound is ours to find, not the\n'
	printf '     hardware refusing.\n'
   fi
   ;;

14) say "14. the same widths in int8: is it int4's fault or every dispatch's"
   # ⚠⚠ EVERYTHING KNOWN ABOUT THIS FAULT IS INFERRED FROM WHOLE-MODEL TEXT.
   # Phase 13 says a model answers differently, and the slice widths were then
   # worked out from its two dimensions. That is three layers away from the
   # dispatch that is actually wrong, and it is why the shape of the fault
   # still makes no sense: 2816 wrong, 3072 RIGHT, 4096 wrong.
   #
   # npu_gemm_test is the direct instrument -- ONE shape, submitted, against an
   # exact CPU reference -- and it has never been pointed at this. It is int8,
   # which is the point: the CBUF pair, the surface fields and the geometry are
   # shared by every dispatch, so
   #
   #   int8 shows the same 2816/3072/4096 pattern  -> the fault is in the
   #       geometry every dispatch shares and int4 is a bystander
   #   int8 is exact at every width                -> it is the int4 path, and
   #       that is a far smaller place to look
   #
   # Either answer halves the search, which is more than another sweep of
   # whole-model text can do.
   #
   # ⚠ m=1 RUNS FIRST WHATEVER IS ASKED FOR, inside the tool. An m=80 failure
   # means nothing if the control is already wrong, and the tool says so itself.
   run 14
   have npu_gemm_test || { skip 14 "npu_gemm_test not installed (dev channel)"; break; }
   KG=${GEMM_WIDTHS:-2048 2816 3072 4096}
   NG=${GEMM_N:-1536}
   MG=${GEMM_M:-80}
   printf '  K over %s at N=%s, m=1 then m=%s, int8 exact against the CPU\n' \
       "$KG" "$NG" "$MG"
   printf '  (N=%s is Qwen2.5 down projection, the tensor phase 13 implicates)\n' "$NG"
   for K in $KG; do
	r=$(CHARSIU_GEMM_M="$MG" timeout 600 "$BIN/npu_gemm_test" "$K" "$NG" 2>&1)
	v=$(printf '%s' "$r" | grep -oE '[0-9]+ of [0-9]+ widths exact' | head -1)
	c=$(printf '%s' "$r" | grep -c "m=1 disagrees")
	if [ -z "$v" ]; then
		bad "K=$K: no verdict line -- the tool did not finish"
		printf '%s\n' "$r" | tail -4 | sed 's/^/       /'
	elif [ "$c" != 0 ]; then
		bad "K=$K: the m=1 CONTROL disagrees, so this width says nothing"
	else
		# ⚠ "3 of 3" is exact and "2 of 3" is the fault. Compare the two
		# numbers rather than printing the line and leaving it to a
		# reader -- a table nobody has to interpret is a table nobody
		# misreads at one in the morning.
		got=${v%% of *}
		rest=${v#* of }
		want=${rest%% *}
		if [ "$got" = "$want" ]; then
			printf '      K=%-5s %-22s exact\n' "$K" "$v"
		else
			bad "K=$K: $v -- this width is wrong in int8 too"
			printf '%s\n' "$r" | grep -E "^  *[0-9]+ +[0-9]+" \
			    | head -6 | sed 's/^/       /'
		fi
	fi
   	# ⚠ INSIDE the sweep: a wedged block makes every later cell
   	# read as garbage, and this loop printed two of them as
   	# failures before the check below ever ran.
   	wedged "phase 14" && break
   done
   wedged "phase 14" && break
   ok "a width that is not all-exact here is a fault every dispatch shares."
   printf '     All of them exact, and the fault is somewhere only int4 goes.\n'
   ;;

15) say "15. one int4 matmul at a time: is the bound K, or is it K times N"
   # ⚠⚠ THIS TOOL WAS HERE THE WHOLE TIME AND I INFERRED INSTEAD. charsiu_matmul
   # submits ONE matmul through charsiu's own emitter -- charsiu_emit_job, the
   # one npudev uses, on the width axis that int4 takes -- and checks it against
   # the CPU. Everything known about the wide slice fault so far was worked out
   # from whole-model text and two dimensions, which is how it kept looking
   # like nonsense: 2816 wrong, 3072 right, 4096 wrong.
   #
   # It stops looking like nonsense when the dispatch's WEIGHT BYTES are
   # computed instead of its K. Per slice, K * N / 2 for int4:
   #
   #   Qwen2.5 @2048  max 1536 KiB   right      gemma-3-1b @3072  max 1728 KiB  right
   #   SmolLM2 @4096  max  432 KiB   right      Qwen2.5    @3072  max 2304 KiB  WRONG
   #   Qwen2.5 @4096  max 3072 KiB   WRONG      gemma-3-1b @4096  max 2304 KiB  WRONG
   #
   # Six for six, with the boundary in (1728 KiB, 2304 KiB] -- 2 MiB sits in
   # the gap. It is the only reading that explains gemma-3-1b working at 3072
   # while Qwen2.5 does not: same K, different N.
   #
   # ⚠ AND IT IS PROBABLY OURS, NOT THE HARDWARE'S. The vendor emits 0x101c =
   # 4194304 on its 2048x4096 int4 shape, which is 4 MiB of weight bytes in one
   # dispatch, so the block does this.
   #
   # The second sweep is the one that can kill the hypothesis: hold K at 4096
   # and walk N. If the bound is K, every N fails. If it is K * N, it turns
   # over around N = 1024, where 4096 * 1024 / 2 is exactly 2 MiB.
   run 15
   have charsiu_matmul || { skip 15 "charsiu_matmul not installed (dev channel)"; break; }
   MM=${MM_M:-80}
   printf '  m=%s, int4 weights, charsiu_emit_job, against the CPU\n\n' "$MM"
   printf '  A. N fixed at 1536, K walking -- phase 13 widths, directly\n'
   for K in ${MM_K:-2048 2816 3072 4096}; do
	wb=$((K * 1536 / 2 / 1024))
	if CHARSIU_W4=1 timeout 300 "$BIN/charsiu_matmul" "$MM" "$K" 1536 \
	   >"$OUT/.mm" 2>&1; then
		printf '      K=%-5s N=1536  %5s KiB  exact\n' "$K" "$wb"
	else
		printf '      K=%-5s N=1536  %5s KiB  ⚠ WRONG\n' "$K" "$wb"
		nb15=$((${nb15:-0} + 1))
	fi
   done
   printf '\n  B. K fixed at 4096, N walking -- this is the discriminating one\n'
   for N in ${MM_N:-384 512 768 1024 1536}; do
	wb=$((4096 * N / 2 / 1024))
	if CHARSIU_W4=1 timeout 300 "$BIN/charsiu_matmul" "$MM" 4096 "$N" \
	   >"$OUT/.mm" 2>&1; then
		printf '      K=4096 N=%-5s %5s KiB  exact\n' "$N" "$wb"
	else
		printf '      K=4096 N=%-5s %5s KiB  ⚠ WRONG\n' "$N" "$wb"
		nb15=$((${nb15:-0} + 1))
	fi
   	# ⚠ INSIDE the sweep: a wedged block makes every later cell
   	# read as garbage, and this loop printed two of them as
   	# failures before the check below ever ran.
   	wedged "phase 15" && break
   done
   wedged "phase 15" && break
   printf '\n'
   if [ "${nb15:-0}" = 0 ]; then
	# ⚠ NOT A PASS. The models are wrong at these widths, so a bench that
	# says every shape is exact has failed to reproduce a fault that is
	# certainly there -- and that is a finding about the bench.
	bad "every shape here is exact, so this bench does NOT reproduce the"
	printf '     fault the models show. Something the models do and this does\n'
	printf '     not -- the K slice loop, the accumulation across slices, the\n'
	printf '     deal across two cores -- is where it actually lives.\n'
   else
	ok "sweep B is the answer: if it turns over near N=1024 the bound is on"
	printf '     K * N and 2 MiB is the number; if every N fails the bound is\n'
	printf '     on K alone and the weight bytes were a coincidence of shapes.\n'
   fi
   ;;

16) say "16. one tensor, sliced: the rung between a dispatch and a model"
   # ⚠⚠ WHY THIS EXISTS. Two rungs are measured and they disagree:
   #
   #   phase 15   ONE dispatch, int4, K=4096 N=1536 m=80    EXACT
   #   phase 13   a whole model at KMAX 3072 or 4096        WRONG
   #
   # and everything between them was inferred. Three hypotheses were fitted to
   # that inference and all three died: the CBUF pair (already in job.c and
   # right), K * N at 2 MiB (killed by phase 15), and the core pair (Qwen2.5
   # fails identically on one core).
   #
   # What is actually between the two rungs is: several K slices accumulated
   # into one Y, staged through npudev, with the batched buffers shared across
   # them. npu_slice_test is exactly that and nothing else -- one synthetic
   # tensor, charsiu_npu_add, charsiu_npu_matmul, against npu_matvec on the
   # same quantised weights, which is the comparison phase 2 makes at model
   # scale.
   #
   # ⚠ The shapes below are Qwen2.5's ffn down projection, K = 8960 N = 1536,
   # because that is the tensor phase 13 implicates and the only multi-slice
   # one it has at KMAX 3072.
   run 16
   have npu_slice_test || { skip 16 "npu_slice_test not installed (dev channel)"; break; }
   SK=${SLICE_K:-8960}
   SN=${SLICE_N:-1536}
   SM=${SLICE_M:-80}
   printf '  K=%s N=%s m=%s, KMAX walking. m=1 is the control inside the tool.\n' \
       "$SK" "$SN" "$SM"
   for K in ${SLICE_KMAX:-1024 2048 3072 4096}; do
	# ⚠ CHARSIU_NPU_ANY_SURFACE=1 LIFTS THE GUARD THIS PHASE EXISTS TO
	# INTERROGATE. Without it the first run of this phase came back with
	# 3072 and 4096 "refused by the surface guard" -- my own tripwire
	# blocking the only measurement that can explain it.
	r=$(CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
	    CHARSIU_NPU_W4_GROUP="$K" CHARSIU_NPU_MAXN=262144 \
	    CHARSIU_COEF_ELEMS=65536 CHARSIU_NPU_ANY_SURFACE=1 \
	    timeout 600 "$BIN/npu_slice_test" "$SK" "$SN" "$SM" "$K" 2>&1)
	# ⚠ the detail lines came back EMPTY last round because this grepped
	# "batched:" with a colon and the tool prints "batched vs the same
	# call". A phase that loses the diagnosis is a board round spent on one
	# bit of information.
	v=$(printf '%s' "$r" | grep -E "batched|rows wrong|channels |got/want|ks=" )
	if printf '%s' "$r" | grep -q "exact"; then
		printf '      KMAX %-5s exact\n' "$K"
	elif printf '%s' "$r" | grep -q "REFUSED"; then
		# ⚠ WITH THE HATCH SET THIS SHOULD NOT HAPPEN, so it is a
		# failure and not a note: something else refused the batch and
		# the cell was not measured either way.
		bad "KMAX $K: the batch was refused even with the guard lifted"
		printf '%s\n' "$r" | grep -iE "NOT on the NPU" | head -2 | sed 's/^/       /'
	else
		bad "KMAX $K: the sliced batch disagrees with the CPU"
		printf '%s\n' "$v" | sed 's/^/       /'
	fi
   done

   # ⚠⚠ THE FREE CONTROL: THE SAME SURFACE WITH ONE SLICE.
   #
   # phase 15 has a single dispatch at surface 10240 EXACT, but it submits a
   # raw job -- it never touches npudev's staging, slots or batched buffers.
   # phase 16 above goes through all of that AND accumulates several slices, so
   # a failure there has two possible homes and the two rungs cannot separate
   # them.
   #
   # K = KMAX = 3072 separates them for nothing: one slice, so no accumulation
   # at all, at exactly the surface that fails above. Exact here means the
   # accumulation is the fault; wrong here means it is npudev's own buffers and
   # the slice count never mattered.
   printf '\n  the same surface with ONE slice, through the same staging:\n'
   for K in ${SLICE_ONE:-3072 4096}; do
	r=$(CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
	    CHARSIU_NPU_W4_GROUP="$K" CHARSIU_NPU_MAXN=262144 \
	    CHARSIU_COEF_ELEMS=65536 CHARSIU_NPU_ANY_SURFACE=1 \
	    timeout 600 "$BIN/npu_slice_test" "$K" "$SN" "$SM" "$K" 2>&1)
	if printf '%s' "$r" | grep -q "exact"; then
		printf '      K=KMAX=%-5s one slice, surface %-6s exact\n' \
		    "$K" "$(( (K / 32) * SM ))"
	else
		bad "K=KMAX=$K: ONE slice at surface $(( (K / 32) * SM )) is already wrong"
		printf '     so it is not the accumulation -- npudev gets a single\n'
		printf '     wide slice wrong through its own staging.\n'
		printf '%s\n' "$r" | grep -E "rows wrong|channels |got/want" | sed 's/^/       /'
	fi
   	# ⚠ INSIDE the sweep: a wedged block makes every later cell
   	# read as garbage, and this loop printed two of them as
   	# failures before the check below ever ran.
   	wedged "phase 16" && break
   done
   wedged "phase 16" && break
   ok "if a KMAX is wrong HERE it is the slicing, and this is the smallest"
   printf '     thing that shows it -- one tensor, no model, no tokenizer. If\n'
   printf '     every KMAX is exact here and phase 13 still fails, then it is\n'
   printf '     not one tensor either: it is several sharing the buffers.\n'
   ;;

17) say "17. the CBUF window: 0x1c00 is 7168, and is that where it ends"
   # ⚠⚠ THE MECHANISM, AND IT WAS WRITTEN DOWN HERE BEFORE ANY OF THIS.
   # job.c on charsiu_cbuf_window: "0x1c00 is 7168 and appears in three of them
   # as what reads like a base offset", and npudev sets cbuf_window = di, the
   # device index. So the CBUF is two windows and window 1 starts at entry
   # 7168 -- which means window 0 holds 7168 entries and a surface past that
   # runs into the other core's.
   #
   # It accounts for every cell measured so far. Surfaces of 1024 through 5120
   # are exact; 7680 and 10240 are garbage, all rows, all channels, ratios in
   # the hundreds. And phase 15's raw job at surface 10240 was EXACT because
   # charsiu_cbuf_window() returns 0 by default and one job has no second
   # window to collide with -- npudev always has one.
   #
   # ⚠⚠ AND THIS PHASE ALREADY RAN AND KILLED THAT STORY. 6144 came back WRONG,
   # which is below 7168, so window 1's base is not the line either -- it is
   # the fifth explanation to fit everything known and then die. The bracket
   # this phase actually left behind is (5120, 6144], the guard sits at the
   # bottom of it, and tightening or loosening inside a 1024-wide bracket buys
   # no width that is not already reachable. Everything below is kept as the
   # instrument that walked it, NOT as an open question.
   #
   # K is held at 4096 and m walks, because that makes the surface land on
   # round numbers: (4096/32) * m = 128m.
   run 17
   have npu_slice_test || { skip 17 "npu_slice_test not installed (dev channel)"; break; }
   printf '  K=N=4096/1536, one slice, m walking. surface = 128 * m.\n'
   printf '  the guard is lifted; 7168 is the value under test.\n'
   for M in ${SURF_M:-32 40 48 56 60 64}; do
	surf=$((128 * M))
	r=$(CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
	    CHARSIU_NPU_W4_GROUP=4096 CHARSIU_NPU_MAXN=262144 \
	    CHARSIU_COEF_ELEMS=65536 CHARSIU_NPU_ANY_SURFACE=1 \
	    timeout 600 "$BIN/npu_slice_test" 4096 1536 "$M" 4096 2>&1)
	if printf '%s' "$r" | grep -q "exact"; then
		printf '      m=%-4s surface %-6s exact%s\n' "$M" "$surf" \
		    "$([ "$surf" -gt 7168 ] && echo '   ⚠ past 7168 and still right')"
	elif printf '%s' "$r" | grep -q "REFUSED"; then
		bad "m=$M: refused even with the guard lifted"
	else
		printf '      m=%-4s surface %-6s ⚠ WRONG%s\n' "$M" "$surf" \
		    "$([ "$surf" -le 7168 ] && echo '   ⚠ at or below 7168 and already wrong')"
	fi
   	# ⚠ INSIDE the sweep: a wedged block makes every later cell
   	# read as garbage, and this loop printed two of them as
   	# failures before the check below ever ran.
   	wedged "phase 17" && break
   done
   wedged "phase 17" && break
   ok "the last exact surface is the ceiling. It came back 5120, so window 1's"
   printf '     base at 7168 is a coincidence and 5120 stands on the vendor evidence.\n'
   ;;

18) say "18. the surface guard has an axis, and int8 is not on it"
   # ⚠⚠ THE SURFACE GUARD HAS AN AXIS, AND THE FIRST VERSION OF IT DID NOT.
   #
   # (slice / 32) * m is the WIDTH axis's input surface. charsiu_emit_job sets
   # inw = m only when wide, so on the HEIGHT axis the input is one column of m
   # rows and the surface is (slice / 32) * 1 -- three orders smaller and
   # nowhere near any ceiling. job.c had already written down, in the note over
   # the split window, that a rule read off int4 must not reach int8. The guard
   # shipped without the gate and reached it anyway.
   #
   # It cost the vision tower, which is the one caller that opens want_w4 = 0
   # with a wide K. SmolVLM-256M's ffn_down is K = 3072 and charsiu_pool_rows
   # batches 64 rows, so the width formula reads 6144 -- the first WRONG cell
   # in phase 17's int4 bracket -- and all twelve of them plus the idefics3
   # projector were refused. The encoder went 5.57 s to 31.0 s and the pool
   # said "73 asked and 13 fell back"; gated it reads 5.37 s and 0 fell back.
   # (15.5 s was quoted first and is the wrong baseline -- that is
   # board_modalities' number from another round, not this scoreboard's.)
   #
   # ⚠ SO THIS PHASE RUNS WITHOUT THE HATCH. Phases 16 and 17 lift the guard
   # because they exist to interrogate what is above it; this one exists to
   # check that the guard is not standing where it was never measured, and
   # lifting it would verify nothing.
   # ⚠⚠ K = 8192 WEDGED THE BLOCK the first time this phase ran, and a wedge is
   # not a result: the two cells after it were measured on a dead NPU and
   # printed as failures. It runs LAST now, and the sweep stops at it. It is
   # here at all because job.c's split-window note cites k = 8192 at m = 64 as
   # the int8 batch that has been right since the 2.94x round -- a claim in the
   # tree that nothing had ever run on its own.
   #
   # ⚠⚠ AND THE FIRST RUN'S OTHER TWO "FAILURES" WERE NOT THE BATCHING EITHER.
   # charsiu_act_q1 multiplies by 1/d and npudev's batched packer divided by d.
   # Those are not the same float, so one value in 3072 landed a code apart and
   # moved all 768 outputs a hair -- 127 of them past a 0.1% RELATIVE threshold
   # because they are near-zero channels. It read as "1 row of 64 wrong, mean
   # got/want 1.0007", which is small enough to look like noise and stable
   # enough to look like structure.
   #
   # It never needed a board. npu_slice_test now runs both quantisers on the
   # desk before it opens anything and prints the row -- it says "1 code in 1
   # of 64 rows, first such row is 3" at seed 12345, which is exactly the row
   # the board reported. npudev multiplies now, so a SINGLE-SLICE int8 cell is
   # expected EXACT and anything else is the batching.
   run 18
   have npu_slice_test || { skip 18 "npu_slice_test not installed (dev channel)"; break; }
   printf '  int8, height axis, the guard NOT lifted. width formula in brackets.\n'
   printf '  the tool prints a quantiser check first; it needs no hardware.\n'
   for C in ${INT8_CELLS:-3072/768/64/4096 3072/768/80/4096 4096/1536/80/4096 8192/1536/64/2048}; do
	IK=${C%%/*}; rest=${C#*/}; IN=${rest%%/*}; rest=${rest#*/}
	IM=${rest%%/*}; IX=${rest##*/}
	wf=$(( (IK / 32) * IM ))
	sl=$(( (IK + IX - 1) / IX ))
	r=$(CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=0 \
	    CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 \
	    CHARSIU_SLICE_SEED="${INT8_SEED:-12345}" \
	    timeout 600 "$BIN/npu_slice_test" "$IK" "$IN" "$IM" "$IX" 2>&1)
	lbl=$(printf 'K=%-5s N=%-5s m=%-4s %s slice(s)' "$IK" "$IN" "$IM" "$sl")
	if printf '%s' "$r" | grep -q "REFUSED"; then
		# the regression, exactly: refused on a formula for another axis
		bad "$lbl: refused on the height axis (width formula reads $wf)"
		printf '%s\n' "$r" | grep -iE "NOT on the NPU|surface" | head -2 | sed 's/^/       /'
	elif printf '%s' "$r" | grep -q "exact"; then
		printf '      %s exact   [width %s]%s\n' "$lbl" "$wf" \
		    "$([ "$wf" -gt 5120 ] && echo '  ⚠ the guard would have refused this')"
	else
		# ⚠⚠ THERE IS NO EXCUSE BRANCH HERE ANY MORE. This had one for
		# multi-slice cells, on the reasoning that npudev scales per K
		# slice while charsiu_act_q1 scales per row, so the two could
		# not agree. K = 8192 at 4 slices then came back EXACT, which
		# refutes it -- the reference is charsiu_npu_matvec, the NPU's
		# own row loop, and it goes through the same slicing. A branch
		# that turns a failure into prose is worse than no branch: it
		# was written to explain a result it had not seen.
		bad "$lbl: not exact -- the quantisers match now, so this is the batching"
		printf '%s\n' "$r" | grep -E "quantiser check|batched|rows wrong|channels |got/want|magnitude" | sed 's/^/       /'
	fi
	# ⚠ INSIDE the sweep: a wedged block makes every later cell read as
	# garbage, and this loop printed two of them as failures before the
	# check below ever ran.
	wedged "phase 18" && break
   done
   wedged "phase 18" && break
   ok "int8 batches past the int4 ceiling, so the gate belongs on the axis"
   printf '     ⚠ the height axis has a ceiling of its own -- phase 19 walks it.\n'
   ;;

19) say "19. the height axis has a ceiling too, and which quantity is it"
   # ⚠⚠ PHASE 18 FOUND ONE AND THIS PHASE IS WHAT IT OWES. Four int8 cells ran
   # and three were exact:
   #
   #   K=3072 m=64  1 slice   surf 96  x 64 =  6144   exact
   #   K=3072 m=80  1 slice   surf 96  x 80 =  7680   exact
   #   K=8192 m=64  4 slices  surf 64  x 64 =  4096   exact   (per dispatch)
   #   K=4096 m=80  1 slice   surf 128 x 80 = 10240   WRONG, 80/80 rows,
   #                                                  mean got/want 2.1037
   #
   # and the quantiser check printed 0 differing codes on that cell, so the
   # thing phase 18 spent a round on is excluded by the tool itself.
   #
   # ⚠ AND job.c NEVER CHECKS. Its split rule is `wide && surf * rows > 4096`,
   # and wide is 0 for int8, so an int8 dispatch never splits and nothing looks.
   # The int4 side has a guard at 5120; the height axis has nothing.
   #
   # ⚠⚠ BUT DO NOT GUARD IT YET. Three quantities fit all four cells and they
   # are not the same rule:
   #
   #   (a) surf alone, i.e. K/32     128 fails, 96 and 64 pass
   #   (b) surf * rows               10240 fails, 7680 and below pass
   #   (c) N * m, the OUTPUT         122880 fails, 98304 and below pass
   #
   # Putting a guard on the wrong one of those is exactly what shipped last
   # round on the other axis. Each cell below says which candidate it kills.
   #
   # ⚠ m STOPS AT 80. npupool's own sweep has 4..80 identical and 96 and up "a
   # different tower", so a cell above 80 would be measuring that instead.
   run 19
   have npu_slice_test || { skip 19 "npu_slice_test not installed (dev channel)"; break; }
   printf '  int8, one slice unless said. surf = K/32. each cell kills a candidate.\n'
   for C in ${H_CELLS:-4096/1536/32/a 3072/1536/80/c 4096/768/80/b 4096/1536/48/w 4096/1536/64/w 3584/1536/80/w}; do
	HK=${C%%/*}; rest=${C#*/}; HN=${rest%%/*}; rest=${rest#*/}
	HM=${rest%%/*}; HW=${rest##*/}
	sf=$(( HK / 32 )); pr=$(( sf * HM )); ou=$(( HN * HM ))
	case "$HW" in
	a) why="if WRONG: surf alone (K), and neither product matters" ;;
	b) why="if WRONG with (a) dead: the input surface, not the output" ;;
	c) why="if WRONG: the OUTPUT N*m, not the input at all" ;;
	*) why="walks the bracket" ;;
	esac
	r=$(CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=0 \
	    CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 \
	    CHARSIU_SLICE_SEED="${INT8_SEED:-12345}" \
	    timeout 600 "$BIN/npu_slice_test" "$HK" "$HN" "$HM" "$HK" 2>&1)
	lbl=$(printf 'K=%-5s N=%-5s m=%-3s surf %-4s x%-3s = %-6s out %-7s' \
	    "$HK" "$HN" "$HM" "$sf" "$HM" "$pr" "$ou")
	if printf '%s' "$r" | grep -q "REFUSED"; then
		bad "$lbl refused -- nothing was measured"
	elif printf '%s' "$r" | grep -q "exact"; then
		printf '      %s exact\n' "$lbl"
	else
		printf '      %s ⚠ WRONG\n' "$lbl"
		printf '        %s\n' "$why"
		printf '%s\n' "$r" | grep -E "rows wrong|got/want|magnitude" | sed 's/^/        /'
	fi
	# a wedge here would make every later cell read exact on a dead block
	wedged "phase 19" && break
   done
   wedged "phase 19" && break
   ok "the bracket is walked; read the three candidates off the table above"
   printf '     ⚠ EXPOSURE, unguarded either way: whisper sizes its pool at\n'
   printf '        4 * n_audio_state, so medium is K=4096 (surf 128) and large\n'
   printf '        is K=5120 (surf 160) -- both at or past the failing cell,\n'
   printf '        and the board has only ever run tiny (K=1536, surf 48).\n'
   ;;

20) say "20. speculative decoding: the price of a pass, and the text"
   # ⚠⚠ THE ONE NUMBER THE HOST CANNOT MEASURE. tests/spec_identity.sh has
   # already shown, on a machine with no NPU, that --spec commits exactly the
   # tokens greedy would have and that junk drafts are rejected every time; and
   # it has measured how many tokens a pass yields, which is a property of the
   # model and the prompt: Qwen3 2.29 a pass, gemma4 2.45, phi3 1.70, on a
   # prompt that asks for repetition. What only the board can say is what a
   # pass at m = 4 COSTS against a decode step. The argument that it is about
   # one step is an argument about bytes -- the weights are read once either
   # way -- and the fence and the accumulator read back both grow with m.
   #
   #   speed up = tokens a pass / (pass cost in decode steps)
   #
   # so at 2.29 tokens a pass the break-even cost is 2.29 steps, and anything
   # under that is a win the vendor's decode does not have.
   #
   # ⚠ THE TEXT IS CHECKED HERE TOO, because on the board the pass is a real
   # batched matmul at m = 4 and the host only ever ran that a row at a time.
   run 20
   have spec_identity.sh || { skip 20 "spec_identity.sh not installed (dev channel)"; break; }
   sh "$BIN/spec_identity.sh" "$MODELS" 48 >"$OUT/verify-spec.txt" 2>&1
   rc=$?
   sed 's/^/  /' "$OUT/verify-spec.txt"
   wedged "phase 20" && break
   [ "$rc" = 0 ] && ok "every model's speculative text is its greedy text, and junk was refused" \
                 || bad "spec_identity failed on $rc model(s)"
   # the price: the same prompt and length, plain against --spec 3, one model
   SM=""
   for f in "$MODELS"/Qwen3-0.6B*.gguf "$MODELS"/*.gguf; do [ -r "$f" ] && SM=$f && break; done
   if [ -n "$SM" ]; then
	printf '  price on %s, %s tokens:\n' "$(basename "$SM")" "${SPEC_N:-96}"
	for arm in "" "--spec 3" "--spec 5"; do
		r=$("$BIN/charsiu_run" "$SM" -p "$P9" -n "${SPEC_N:-96}" $arm 2>/dev/null)
		gen=$(printf '%s' "$r" | sed -n 's/.*| gen \([0-9]*\) tok in \([0-9]*\) ms, \([0-9.]*\) tok\/s.*/\3 tok\/s (\1 tok, \2 ms)/p')
		sl=$(printf '%s' "$r" | grep '^\[spec ' | sed 's/^\[spec k=[0-9]*: //; s/\]$//')
		printf '    %-9s %s%s\n' "${arm:-plain}" "$gen" "${sl:+   $sl}"
	done
	printf '  ⚠ tok/s over passes = the pass cost in decode steps; compare it to tok/pass.\n'
   fi
   wedged "phase 20" && break
   ;;

21) say "21. the price of a call: where 130 us goes"
   # ⚠⚠ DECODE'S BIGGEST REMAINING COST IS NOT BYTES. Phase 7's own cost model
   # on Qwen3 reads `us a call = 130 + 16 a task + 120 a MB`, and a token is
   # about 150 calls: 19.5 ms of the 51.7 ms token is the per-call floor, more
   # than the 18.6 ms the weights take at 16 GB/s. Halve the floor and decode
   # is at 96% of the vendor.
   #
   # What is in it, read off the board's own kernel (linux-next, rocket):
   #
   #   rocket_job_run     pm_runtime_resume_and_get + iommu_attach_group PER
   #                      JOB, and iommu_detach_group + put_autosuspend after
   #                      it. rk_iommu's attach writes the DTE address, resets
   #                      and polls; the detach disables paging and zeroes it.
   #   the irq            threaded: hardirq -> irq thread wakeup -> fence
   #                      signal -> waiter wakeup. Two hops where one would do.
   #   cpuidle            rk3576.dtsi: CPU_SLEEP exit-latency-us = 250,
   #                      min-residency-us = 900. A fence wait is ~300 us and
   #                      the governor guesses; a wrong guess costs 250 us on
   #                      the wake, and it costs it on the irq thread, the
   #                      scheduler thread and the waiter alike.
   #
   # This phase prices the one of those that needs no new kernel: it runs the
   # same decode with CPU_SLEEP disabled and reads the cost model line off both
   # arms. If `a call` drops, the wake latency was in it and the kernel patch
   # that keeps the IOMMU domain attached is the next arm.
   run 21
   SM=""
   for f in "$MODELS"/Qwen3-0.6B*.gguf "$MODELS"/*.gguf; do [ -r "$f" ] && SM=$f && break; done
   [ -n "$SM" ] || { skip 21 "no model"; break; }
   IDLE=/sys/devices/system/cpu/cpu0/cpuidle/state1/disable
   [ -w "$IDLE" ] || { skip 21 "no cpuidle state1 to toggle at $IDLE"; break; }
   # ⚠ THE FOURTH ARM NEEDS NO SYSFS AND NO KERNEL: CHARSIU_NPU_SPIN_US polls
   # the fence for that long before sleeping on it, so the waiter never leaves
   # C0 and never pays the exit latency. If it matches the cpu-sleep-off arm
   # the latency was the waiter's; if cpu-sleep-off wins by more, the irq
   # thread and the scheduler thread were paying it too, and only the kernel
   # can fix those.
   for arm in "cpu-sleep on" "cpu-sleep OFF" "cpu-sleep on again" "spin 400 us"; do
	case "$arm" in *OFF) v=1 ;; *) v=0 ;; esac
	case "$arm" in spin*) SPIN=400 ;; *) SPIN=0 ;; esac
	for c in /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state1/disable; do echo $v >"$c"; done
	# ⚠ THE SCOREBOARD'S OWN ENVIRONMENT, verbatim from board_vendor.sh.
	# The first run of this phase started charsiu_run bare and priced the
	# CPU: 4.60 tok/s on a model that decodes at 19.5 on the hardware, no
	# cost-model line at all, and a verdict line that read as a pass.
	r=$(env CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
	    CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 \
	    CHARSIU_STAGES=1 CHARSIU_NPU_SPIN_US=$SPIN \
	    "$BIN/charsiu_run" "$SM" -p "$P9" -n 48 -c 512 -t 4 2>&1)
	gen=$(printf '%s' "$r" | sed -n 's/.*| gen \([0-9]*\) tok in [0-9]* ms, \([0-9.]*\) tok\/s.*/\2 tok\/s/p')
	cm=$(printf '%s' "$r" | grep -o "us a call = [0-9.]* + [0-9.]* a task + [0-9.]* a MB" | head -1)
	printf '  %-20s %-12s %s\n' "$arm" "$gen" "${cm:-⚠ NO COST-MODEL LINE: this run did not take the NPU}"
	[ -n "$cm" ] || bad "phase 21 arm '$arm' ran without the NPU"
   done
   wedged "phase 21" && break
   ok "read 'a call' across the arms; the third must match the first"
   printf '     ⚠ these arms are the wake latency ALONE. The attach/detach per job\n'
   printf '        is the patched kernel (rocket: keep the domain attached), next round.\n'
   ;;

22) say "22. input reuse: which site breaks which model"
   # ⚠⚠ REUSE IS OFF BY DEFAULT BECAUSE OF THIS. Phase 2 stopped two rounds on
   # it: 6 of 9 models with one key for two devices, then Phi-3.5 and gemma4
   # with a key per device -- the two whose projections are views of a fused
   # tensor or come with per-layer embeddings. The host cannot see either
   # fault. This phase turns it on for those two models one SITE at a time
   # (k after q, v after q, up after gate) and reads phase 2's own comparison
   # off each, so the next fix is aimed at a site and not at a guess.
   run 22
   have board_text_all.sh || { skip 22 "board_text_all.sh not installed"; break; }
   for MO in ${REUSE_MODELS:-Phi-3.5 gemma-4 Qwen3}; do
	for SITES in ${REUSE_SITES:-k v up k,v k,v,up}; do
		CHARSIU_NPU_REUSE=1 CHARSIU_REUSE_SITES=$SITES CHARSIU_TEXT_ONLY=$MO \
		    sh "$BIN/board_text_all.sh" >"$OUT/verify-reuse-$MO-$SITES.txt" 2>&1
		v=$(grep -E "text identical|TEXT DIFFERS" "$OUT/verify-reuse-$MO-$SITES.txt" | head -1 | sed 's/.*prompt batched //')
		printf '      %-9s sites %-7s %s\n' "$MO" "$SITES" "${v:-no verdict line}"
		wedged "phase 22" && break 2
	done
   done
   wedged "phase 22" && break
   ok "read the table: the first site that says DIFFERS on a model is the one to look at"
   printf '     ⚠ Qwen3 is the control: it passed with all three on when the key was per device.\n'
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
