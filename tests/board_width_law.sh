#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# The safe batch widths are `m % 2 == 0 && m != 8`. This is the board round
# that VERIFIES that, densely, and separates the one fault that is still open.
#
# ⚠⚠ WHAT IS SETTLED, AND WHY THIS IS NO LONGER A SEARCH.
#
# The rule used to be `m <= 4 || m % 16 == 0`, fitted to nine measured widths.
# It was a bad fit: every multiple of 16 is even, so nine points could not tell
# the two rules apart, and the modulo was doing no work at all.
#
# charsiu_acc_index -- the read order itself, out of src/job.c -- was linked
# into a standalone checker and swept over m = 2..96 crossed with n = 512, 2048
# and 8192, asking four things of every (m, n): is every index in range, are
# there collisions, are there holes, and does the four-consecutive-slots
# property the gather relies on survive.
#
#   m EVEN   clean bijection, four in a row intact, at every n
#   m ODD    COLLISION, at every n, with no exceptions
#
# And it is STRUCTURAL rather than a bug waiting to be fixed. The roleswap2
# branch covers 64 * P slots per group where the group needs 32 * m, and
# 64P == 32m only when P == m/2. No integer P exists for odd m. The accumulator
# surface is organised in PAIRS OF ROWS, and an odd width is asking it for half
# a pair.
#
# That splits two faults that had been read as one:
#
#   m = 31   odd    0 of 6975 rows                      THE READ ORDER, proven
#   m = 8    even   871/904 on two cores, 904/904 with  THE CORE PAIR, a
#                   CHARSIU_NPU_ONEDEV=1                DIFFERENT fault
#   even, not 8     exact
#
# ⚠⚠ THE CBUF WINDOW SPLIT IS REFUTED. DO NOT REOPEN IT.
#
# The obvious story for m = 8 was that src/job.c decides
#
#     split = wide && surf * rows > 4096
#
# so the window layout is a function of M, and two cores on a shared window
# have already been measured corrupting each other three times in four. It does
# not survive the arithmetic. surf is entries_per_row * M, so the flag turns ON
# at LARGE M -- above M = 42 at k = 3072, and above M = 128 at the k = 1024
# slices the board environment here actually produces. The failures are at
# SMALL and ODD M, and 48, 64 and 80 are exact. The split flag is the opposite
# of the failure pattern, in both directions. It is not the mechanism.
#
# ⚠ SO ODD WIDTHS ARE EXPECTED TO FAIL HERE, AND THAT IS THE CONTROL. A round
# in which odd widths come back exact is not good news: it means the probe has
# stopped discriminating -- the batch fell back to a row at a time, or the
# comparison is against itself -- and nothing else in the round can be read.
# The prediction below is scored in both directions for exactly that reason.
#
# WHAT IS STILL OPEN, and it is the only thing this round can DISCOVER rather
# than confirm: m = 8. It is even, so the read order is a clean bijection at
# every n, and it is still wrong on two cores and right on one. So the round
# ends with an UNCAPPED m = 8 only pass on each arm, whose two questions are:
#
#   is it exact on one core at EVERY n, or only at the n it has been tried at
#   does it stay wrong on two cores at EVERY n, or only at n = 8192
#
# Those need every tensor in the model, which is why that pass drops the tensor
# cap the dense sweep depends on.
#
# ⚠ WHAT IT COSTS. The one row reference is m matvecs per tensor, and summed
# over 2..64 that is 2079 matvecs a tensor -- CHARSIU_PROBE_MAXT is the only
# reason a dense sweep is affordable at all. At the default cap of 8 tensors
# that is about 17k one row submits an arm plus the batched side, minutes
# rather than hours; the onedev arm has half the hardware, so budget roughly
# double for it. The uncapped m = 8 pass at the end is 8 matvecs over every
# staged tensor, about four dense widths' worth. On top of both comes the model
# load and the lazy staging of every weight, which on this board is the largest
# single term and is paid once a run. Fifteen to thirty minutes for the whole
# round is the expectation; the timeout is 1800s an arm so a wedged arm cannot
# eat the others.
#
# ⚠ AND THE TENSOR CAP IS A BIAS WITH A SHAPE. CHARSIU_PROBE_MAXT takes the
# FIRST N staged tensors, which are layer 0's projections and the start of
# layer 1 -- every distinct (k, n) an ordinary layer has, including the
# n = 8192 gate/up pair that m = 8 misses. What it does NOT cover is the output
# head, staged last and the one shape nothing else in the model resembles. A
# width that is exact in the dense sweep is exact ON A LAYER. The probe prints
# the cap on every table row so that argument can never be made from a number
# that has forgotten it, and the m = 8 pass at the end is uncapped precisely
# because it is the one asking about all the n.
#
# `charsiu update dev` installs this at /opt/charsiu/board_width_law.sh.
#
#   sh board_width_law.sh [MODEL.gguf | substring]
#
#   CHARSIU_LAW_WIDTHS=2,3,4,...    the width list (default 2..64 dense)
#   CHARSIU_LAW_MAXT=8              staged tensors checked per width
#   CHARSIU_LAW_ARMS="both onedev"  which arms to run
#   CHARSIU_LAW_M8=0                skip the uncapped m = 8 pass
#   CHARSIU_LAW_TIMEOUT=1800        seconds an arm may take before it is killed
set -u

MODEL=${1:-}

# --- the binary ------------------------------------------------------------
RUN=${CHARSIU_RUN_BIN:-}
[ -n "$RUN" ] || RUN=$(command -v charsiu_run 2>/dev/null || true)
[ -n "$RUN" ] || for d in /opt/charsiu /usr/bin "$HOME/charsiu" ./build .; do
	[ -x "$d/charsiu_run" ] && { RUN="$d/charsiu_run"; break; }
done
[ -n "${RUN:-}" ] || { echo "board_width_law: charsiu_run not found" >&2; exit 1; }

# --- the NPU ---------------------------------------------------------------
# ⚠ WITHOUT THE NPU THERE IS NO ROUND AT ALL. The batched matmul never reaches
# hardware, the probe says "no NPU staged" and returns -- and charsiu_run still
# EXITS 0, so an exit status check does not catch it. That is the exact shape
# of the failure this tree keeps shipping: a run that exits clean having
# measured nothing. It is caught below by looking for the table itself, and
# this guard is here so the common case says so before spending the time.
# ⚠ ANY accel NODE, NOT accel0. A rebind of rocket takes the next free
# minor, so the NPU can sit at accel1 or accel2 and a test that looks only
# for accel0 refuses on a board that has one.
if [ -z "$(ls /dev/accel/accel* 2>/dev/null)" ] && [ -z "${CHARSIU_ALLOW_NO_NPU:-}" ]; then
	echo "NO /dev/accel/accel* -- every width would fall back to the CPU," >&2
	echo "every arm would agree, odd widths would come back exact, and a" >&2
	echo "prediction scored against that says nothing about the silicon." >&2
	echo "Run it on the board.  CHARSIU_ALLOW_NO_NPU=1 overrides." >&2
	exit 1
fi

# --- the model -------------------------------------------------------------
# ⚠ A SUBSTRING IS ENOUGH, AND IT FOLDS CASE AND PUNCTUATION. Two rounds have
# been spent on a path typed by hand, one of which ran the wrong model entirely
# and answered its question perfectly. The file is `Phi-3.5-mini-...` and the
# thing anyone types is `phi3`; both sides go to lowercase letters and digits
# before they are compared.
DIRS="$HOME/.charsiu/models $HOME/models /opt/charsiu/models"
norm() { echo "$1" | tr 'A-Z' 'a-z' | tr -cd 'a-z0-9'; }
if [ -n "$MODEL" ] && [ ! -r "$MODEL" ]; then
	want=$(norm "$MODEL")
	MODEL=
	for d in $DIRS; do
		for f in $d/*.gguf; do
			[ -r "$f" ] || continue
			case $(norm "${f##*/}") in
			*"$want"*) MODEL="$f"; break 2 ;;
			esac
		done
	done
	[ -n "$MODEL" ] || {
		echo "board_width_law: no model matching '$1' in $DIRS" >&2
		for d in $DIRS; do
			for f in $d/*.gguf; do
				[ -r "$f" ] && echo "  $f" >&2
			done
		done
		exit 1
	}
	echo "board_width_law: '$1' -> $MODEL"
fi
# ⚠ THE SAME FILE THE WIDTH RECORD WAS MADE ON. Every number this round is read
# against -- 871 of 904 at m = 8, 0 of 6975 at m = 31, exact at 2, 4, 16, 32,
# 48, 64, 80 -- came off Llama-3.2-1B in Q4_0. Defaulting to some other gguf
# would produce a table that looks like the record and is not comparable to it.
if [ -z "$MODEL" ]; then
	for pat in '*Llama-3.2*Q4_0*.gguf' '*llama*Q4_0*.gguf' '*Q4_0*.gguf'; do
		for d in $DIRS; do
			for f in $d/$pat; do
				[ -r "$f" ] && { MODEL="$f"; break 3; }
			done
		done
	done
fi
[ -n "${MODEL:-}" ] && [ -r "$MODEL" ] || {
	echo "board_width_law: no int4 gguf found in $DIRS" >&2
	echo "  pass one:  board_width_law.sh /path/to/model-Q4_0.gguf" >&2
	exit 1
}

OUTDIR=${CHARSIU_BOARD_DIR:-$HOME/charsiu-board}
mkdir -p "$OUTDIR"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

# --- the widths ------------------------------------------------------------
# ⚠ DENSE, AND STILL DENSE EVEN THOUGH THE ANSWER IS KNOWN. The prediction is
# about a parity, and a parity is the one kind of rule a sparse list is worst
# at testing: the old rule agreed with this one on all nine widths anybody had
# ever run. Sixty three widths asked once is what tells `even` from
# `multiple of 16` and from every other rule that happens to fit.
WIDTHS=${CHARSIU_LAW_WIDTHS:-$(awk 'BEGIN{for(i=2;i<=64;i++)printf "%s%d",(i>2?",":""),i}')}
# ⚠ THE PROBE'S LIST BUFFER IS 64 ENTRIES AND IT STOPS THERE SILENTLY. A longer
# list would be truncated, the table would be short, and the missing widths
# would read as "not measured" rather than "never asked".
NW=$(echo "$WIDTHS" | tr ',' ' ' | wc -w)
if [ "$NW" -gt 64 ]; then
	echo "board_width_law: $NW widths, and the probe takes at most 64." >&2
	echo "  It would drop the tail without saying so. Split the round." >&2
	exit 1
fi
MMAX=$(echo "$WIDTHS" | tr ',' '\n' | sort -n | tail -1)
MAXT=${CHARSIU_LAW_MAXT:-8}
ARMS=${CHARSIU_LAW_ARMS:-"both onedev"}
DO_M8=${CHARSIU_LAW_M8:-1}
TMO=${CHARSIU_LAW_TIMEOUT:-1800}
command -v timeout >/dev/null 2>&1 || TMO=

# ⚠⚠ THE WHOLE int4 ENVIRONMENT, NOT JUST THE AXIS. This is the set
# board_vendor.sh runs and it is what puts the run on the int4 path at all;
# CHARSIU_NPU_W4V=1 with no CHARSIU_NPU=1 stages nothing, and the probe then
# says "no NPU staged" and measures the CPU in silence.
W4_ENV="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"

# ⚠⚠ THE ESCAPE HATCHES, AND A CONTROL THAT CANNOT RUN IS NOT A CONTROL.
#
# npudev.c REFUSES the widths it believes are unsafe: it returns -1 before the
# job is built, the probe's matmul fails, the tensor is skipped, and a width
# with no tensors prints NO TABLE ROW AT ALL. Without these two variables every
# width the runtime calls unsafe would be refused in software, this round would
# measure only the widths already believed good, and the prediction would score
# perfectly because the only evidence against it had been removed by it. That
# is not a round, it is a mirror. Odd widths MUST reach the hardware and fail
# there, or the control is vacuous.
#
# board_w4_axis.sh paid for this once already: its height control failed by
# hitting a software gate and said nothing whatever about the silicon.
#
#   CHARSIU_NPU_W4_M8=1     lifts the refusal as it stands today, m = 8 only
#   CHARSIU_NPU_W4_ANYM=1   lifts the generalised refusal on every unsafe width
#
# BOTH are set on purpose. The first is for the binary in the field, the second
# for the one being built; whichever refusal this charsiu_run carries, it is
# lifted. If a future binary renames them again, the NO ROW column below is
# what says so, and it says so per width rather than as a total.
GATES="CHARSIU_NPU_W4_M8=1 CHARSIU_NPU_W4_ANYM=1"

echo "model    $MODEL"
echo "binary   $RUN"
echo "widths   $NW of them, 2..$MMAX: $WIDTHS"
echo "tensors  first $MAXT staged per width (CHARSIU_PROBE_MAXT)"
echo "arms     $ARMS${TMO:+   (${TMO}s each, then killed)}"
echo "gates    $GATES"
echo "predict  safe iff  m % 2 == 0 && m != 8"
echo "         even and not 8   exact on BOTH arms"
echo "         m = 8            FAILS on two cores, EXACT on one"
echo "         odd              FAILS on both arms -- and that is the control"
echo

# probe(OUTFILE, WIDTHLIST, MAXT-or-empty, EXTRA-ENV...) -> 0 if a table came
# back, 1 otherwise, with the reason printed. Every caller reads the return
# value; none of them reads the exit status of charsiu_run, which is 0 whatever
# the probe did.
probe() {
	_out=$1; _w=$2; _cap=$3; shift 3
	_top=$(echo "$_w" | tr ',' '\n' | sort -n | tail -1)
	# shellcheck disable=SC2086
	${TMO:+timeout $TMO} \
	env $W4_ENV $GATES CHARSIU_M_AXIS=w \
	    CHARSIU_PROBE_WIDTHS="$_w" ${_cap:+CHARSIU_PROBE_MAXT=$_cap} \
	    "$@" "$RUN" "$MODEL" --batch-probe "$_top" >"$_out" 2>&1
	_rc=$?
	if [ $_rc -eq 124 ] && [ -n "$TMO" ]; then
		echo "  THIS PASS RAN OUT OF TIME (${TMO}s) and was killed. The"
		echo "  rest of the round still runs, which is the point of the"
		echo "  clock. Its last lines:"
		tail -12 "$_out" | sed 's/^/    /'
		return 1
	fi
	if [ $_rc -ne 0 ]; then
		echo "  THE RUN FAILED (exit $_rc), last lines:"
		tail -12 "$_out" | sed 's/^/    /'
		return 1
	fi
	# ⚠⚠ EXIT 0 IS NOT EVIDENCE OF A MEASUREMENT. charsiu_run returns 0
	# from the --batch-probe path whatever the probe did, including the
	# case where it found nothing staged and returned immediately. The
	# table is the only thing that proves a round happened.
	if grep -q "no NPU staged" "$_out"; then
		echo "  NOTHING WAS STAGED ON THE NPU, so this pass measured the"
		echo "  CPU and nothing else. Exit status was 0 and there is no"
		echo "  table: that is the silent failure, caught. Last lines:"
		tail -12 "$_out" | sed 's/^/    /'
		return 1
	fi
	if ! grep -q "batching .* layers" "$_out"; then
		echo "  THE PROBE NEVER RAN -- no batching table in the output,"
		echo "  and exit status 0. Nothing here is a measurement."
		tail -12 "$_out" | sed 's/^/    /'
		return 1
	fi
	return 0
}

# --- the dense sweep, two arms ---------------------------------------------
: >"$T/both.tsv"
: >"$T/onedev.tsv"
RAN=0
for ARM in $ARMS; do
	case $ARM in
	both)   EXTRA=""
		label="both cores, as shipped -- the configuration the record was made in" ;;
	onedev) EXTRA="CHARSIU_NPU_ONEDEV=1"
		label="one core: the two K slices stop running concurrently" ;;
	*) echo "board_width_law: unknown arm '$ARM'" >&2; continue ;;
	esac
	echo "===== dense sweep, $ARM -- $label ====="
	out="$OUTDIR/width-law-$ARM.txt"
	# shellcheck disable=SC2086
	if ! probe "$out" "$WIDTHS" "$MAXT" $EXTRA; then
		echo
		continue
	fi
	# ⚠ NO head CAP, EVER. This tree has lost two rounds to one: eighteen
	# lines of head over an eighteen line landing table took the whole
	# timing table with it, and a head -80 in board_acc_map.sh lost m = 8.
	# The deciding lines of this tool -- the MISS lines and the
	# where-did-it-go scan -- are at the BOTTOM of what it prints.
	sed -n '/batching .* layers/,$p' "$out" | sed 's/^/  /'
	echo
	# ⚠ THE TABLE ROW IS RECOGNISED BY ITS TENSOR FRACTION AND ITS "of".
	# `8/225` in the second column and `of` in the fifth is a shape no MISS
	# line, no landing table row and no prose line has. Matching on the
	# leading number alone would eat half the landing map.
	awk '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+\/[0-9]+$/ && $5 == "of" \
	     { print $1, $4, $6 }' "$out" >"$T/$ARM.tsv"
	n=$(wc -l <"$T/$ARM.tsv")
	echo "  ($n of $NW widths produced a table row in this arm)"
	echo
	[ "$n" -gt 0 ] && RAN=$((RAN + 1))
done

# --- did anything happen at all? -------------------------------------------
if [ "$RAN" -eq 0 ]; then
	echo "======================================================================"
	echo "⚠⚠ THIS ROUND MEASURED NOTHING. Not one arm produced a table row,"
	echo "   so there is no verdict below and there is nothing to read into"
	echo "   the absence of one. A clean looking empty table is exactly how"
	echo "   this tree has shipped four silent fallbacks in two days."
	echo
	echo "   The usual causes, in the order they are worth checking:"
	echo "     - no NPU: the probe says 'no NPU staged' and charsiu_run"
	echo "       still exits 0"
	echo "     - the int4 environment is incomplete, so nothing was routed"
	echo "     - every width was refused at the npudev gate, which means"
	echo "       $GATES did not reach this binary"
	echo
	for ARM in $ARMS; do
		f="$OUTDIR/width-law-$ARM.txt"
		[ -r "$f" ] || continue
		echo "   last lines of the $ARM arm ($f):"
		tail -12 "$f" | sed 's/^/     /'
		echo
	done
	echo "======================================================================"
	exit 1
fi

# --- the uncapped m = 8 pass, which is the one open question ---------------
# ⚠ UNCAPPED ON PURPOSE, and it is the only pass in this script that is. The
# dense sweep can afford eight tensors because it is asking about a parity, and
# a parity shows on one shape. m = 8 is asking whether the fault follows n, and
# that question is exactly the one the cap cannot answer: the first eight
# staged tensors carry two or three distinct n and the model carries five,
# including the output head, which is the widest thing in it and is staged
# last. One width over every tensor costs about what four dense widths cost.
M8_OK=0
if [ "$DO_M8" != 0 ]; then
	for ARM in $ARMS; do
		case $ARM in
		both)   EXTRA="" ;;
		onedev) EXTRA="CHARSIU_NPU_ONEDEV=1" ;;
		*) continue ;;
		esac
		echo "===== m = 8 alone, UNCAPPED, $ARM ====="
		out="$OUTDIR/width-law-m8-$ARM.txt"
		# shellcheck disable=SC2086
		if ! probe "$out" 8 "" $EXTRA; then
			echo
			continue
		fi
		sed -n '/batching .* layers/,$p' "$out" | sed 's/^/  /'
		echo
		awk '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+\/[0-9]+$/ && $5 == "of" \
		     { print $1, $4, $6 }' "$out" >"$T/m8-$ARM.tsv"
		[ -s "$T/m8-$ARM.tsv" ] && M8_OK=1
	done
fi

# --- the verdict -----------------------------------------------------------
# ⚠ ONE awk OVER BOTH ARMS, because the blocks below are the same join and
# splitting them into three passes is how two of them end up computed over
# different width sets.
awk -v widths="$WIDTHS" -v arms="$ARMS" -v maxt="$MAXT" '
function safe(m) { return (m % 2 == 0 && m != 8) }
FILENAME ~ /both\.tsv$/   { bok[$1] = $2; btot[$1] = $3; next }
FILENAME ~ /onedev\.tsv$/ { ook[$1] = $2; otot[$1] = $3; next }
END {
	n = split(widths, W, ",")
	has_both = (arms ~ /both/); has_one = (arms ~ /onedev/)

	print "======================================================================"
	print "  DENSE SWEEP, SIDE BY SIDE. `pred` is what m % 2 == 0 && m != 8"
	print "  predicts for each arm; `!!` marks a width where the board and the"
	print "  prediction disagree, and those are the only interesting rows."
	print ""
	printf "  %-5s %-13s %-13s %-16s %s\n", \
	       "m", "both cores", "one core", "pred both/one", "verdict"
	printf "  %-5s %-13s %-13s %-16s %s\n", \
	       "-----", "-----------", "-----------", "-------------", "-------"
	for (i = 1; i <= n; i++) {
		m = W[i] + 0
		bs = (m in btot) ? bok[m] "/" btot[m] : (has_both ? "NO ROW" : "-")
		os = (m in otot) ? ook[m] "/" otot[m] : (has_one ? "NO ROW" : "-")
		# what is predicted: odd fails everywhere (the read order
		# collides), m = 8 fails on two cores and is exact on one, every
		# other even width is exact everywhere.
		pb = safe(m); po = (m % 2 == 0)
		ps = (pb ? "PASS" : "FAIL") "/" (po ? "PASS" : "FAIL")
		v = ""
		if (m in btot) {
			bp = (bok[m] == btot[m] && btot[m] > 0)
			if (bp != pb) v = v " !! both:" (bp ? "PASS" : "FAIL")
		}
		if (m in otot) {
			op = (ook[m] == otot[m] && otot[m] > 0)
			if (op != po) v = v " !! one:" (op ? "PASS" : "FAIL")
		}
		if (v == "") v = "as predicted"
		printf "  %-5d %-13s %-13s %-16s %s\n", m, bs, os, ps, v
	}
	print ""

	# --- the prediction, scored, in both directions ---------------------
	print "  ---- the prediction `m % 2 == 0 && m != 8`, scored against the board ----"
	nm = 0; nml = ""
	for (a = 0; a < 2; a++) {
		arm = a ? "one core " : "two cores"
		if (a == 0 && !has_both) continue
		if (a == 1 && !has_one) continue
		tp = tn = fp = fn = 0; fpl = ""; fnl = ""
		for (i = 1; i <= n; i++) {
			m = W[i] + 0
			if (a == 0) {
				if (!(m in btot)) { continue }
				mp = (bok[m] == btot[m] && btot[m] > 0)
				pp = safe(m)
			} else {
				if (!(m in otot)) { continue }
				mp = (ook[m] == otot[m] && otot[m] > 0)
				pp = (m % 2 == 0)
			}
			if (pp && mp) tp++
			else if (!pp && !mp) tn++
			else if (pp && !mp) { fp++; fpl = fpl " " m }
			else { fn++; fnl = fnl " " m }
		}
		tested = tp + tn + fp + fn
		printf "  %s: %d widths measured, %d right, %d wrong\n", \
		       arm, tested, tp + tn, fp + fn
		printf "      predicted PASS and passed   %d\n", tp
		printf "      predicted FAIL and failed   %d   <- the control: odd widths MUST be here\n", tn
		printf "      predicted PASS, FAILED      %d%s\n", fp, (fp ? " :" fpl : "")
		printf "      predicted FAIL, PASSED      %d%s\n", fn, (fn ? " :" fnl : "")
		if (tn == 0 && tested > 0) {
			print  "      ⚠⚠ NOTHING FAILED IN THIS ARM AT ALL. Odd widths"
			print  "         collide in the read order by construction, so an"
			print  "         arm in which they pass is an arm where the probe"
			print  "         is not discriminating -- a fallback to a row at a"
			print  "         time, or a comparison against itself. Nothing else"
			print  "         in this arm can be read."
		}
		print ""
	}
	for (i = 1; i <= n; i++) {
		m = W[i] + 0
		if (has_both && !(m in btot)) { nm++; nml = nml " " m "(two cores)" }
		if (has_one && !(m in otot)) { nm++; nml = nml " " m "(one core)" }
	}
	if (nm) {
		printf "  ⚠ %d arm-widths produced NO TABLE ROW and are in no count above:%s\n", nm, nml
		print  "    A width with no row is a width whose every matmul was refused"
		print  "    or failed. It is not a PASS and it is not a FAIL; it is a width"
		print  "    this round did not ask. If it is most of the odd list, the gate"
		print  "    variables did not reach this binary and the round is VOID."
		print ""
	}

	# --- the round verdict ----------------------------------------------
	print "  ---- does the round PASS? ----"
	print "  It passes only if all three hold:"
	print "    (a) every even width except 8 is exact on BOTH arms"
	print "    (b) every odd width FAILS on BOTH arms"
	print "    (c) m = 8 FAILS with two cores and is EXACT with one"
	print ""
	fail_a = ""; fail_b = ""; fail_c = ""; na = nb = 0
	for (i = 1; i <= n; i++) {
		m = W[i] + 0
		if (m == 8) continue
		if (m % 2 == 0) {
			if ((m in btot) && !(bok[m] == btot[m] && btot[m] > 0))
				{ fail_a = fail_a " " m "(two cores)"; na++ }
			if ((m in otot) && !(ook[m] == otot[m] && otot[m] > 0))
				{ fail_a = fail_a " " m "(one core)"; na++ }
		} else {
			if ((m in btot) && (bok[m] == btot[m] && btot[m] > 0))
				{ fail_b = fail_b " " m "(two cores)"; nb++ }
			if ((m in otot) && (ook[m] == otot[m] && otot[m] > 0))
				{ fail_b = fail_b " " m "(one core)"; nb++ }
		}
	}
	nc = 0
	if (8 in btot) {
		if (bok[8] == btot[8] && btot[8] > 0)
			{ fail_c = fail_c " m=8 was EXACT on two cores, and the record says 871/904"; nc++ }
	}
	if (8 in otot) {
		if (!(ook[8] == otot[8] && otot[8] > 0))
			{ fail_c = fail_c " m=8 was WRONG on one core (" ook[8] "/" otot[8] "), and the record says 904/904"; nc++ }
	}
	if (na) printf "  (a) FAILS, %d deviations:%s\n", na, fail_a
	else    print  "  (a) holds"
	if (nb) printf "  (b) FAILS, %d deviations:%s\n", nb, fail_b
	else if (has_both || has_one) print  "  (b) holds -- every odd width failed, so the probe is discriminating"
	if (nc) printf "  (c) FAILS:%s\n", fail_c
	else if ((8 in btot) || (8 in otot)) print "  (c) holds"
	else    print  "  (c) NOT TESTED -- m = 8 produced no row in either arm"
	print ""
	if (na + nb + nc == 0 && ((8 in btot) || (8 in otot)))
		print "  → THE ROUND PASSES. The safe set is m % 2 == 0 && m != 8, verified"
	else if (na + nb + nc == 0)
		print "  → the three checks that ran all hold, but m = 8 was never measured"
	else {
		print "  → ⚠⚠ THE ROUND DOES NOT PASS, and the deviations above ARE the"
		print "    headline. Every one of them is named; none of them is a"
		print "    rounding story. Read them before anything else here, because"
		print "    a prediction contradicted at one width is not a prediction"
		print "    with an exception, it is a prediction that is wrong."
	}
	print ""

	# --- what a chunker could be built from -----------------------------
	print "  ---- what a chunker could be built from (two cores, as shipped) ----"
	sfl = ""; nsafe = 0; big32 = 0; bigall = 0
	for (i = 1; i <= n; i++) {
		m = W[i] + 0
		if (!(m in btot)) continue
		if (bok[m] != btot[m] || btot[m] == 0) continue
		nsafe++; sfl = sfl " " m
		if (m > bigall) bigall = m
		if (m <= 32 && m > big32) big32 = m
	}
	if (!nsafe) {
		print "  NO WIDTH PASSED, so there is nothing to build a chunker from."
	} else {
		printf "  safe widths measured   %d:%s\n", nsafe, sfl
		printf "  largest safe <= 32     %s\n", (big32 ? big32 "" : "NONE -- and 32 is the shipped chunk")
		printf "  largest safe overall   %d\n", bigall
		# ⚠⚠ A CHUNK SIZE IS NOT ENOUGH: EVERY PROMPT HAS A TAIL.
		# phi3 is 87 tokens, which at a chunk of 32 is 32, 32 and
		# TWENTY THREE -- a width no sweep before this one ever asked
		# about, on a model whose text is wrong on the board. So the
		# question a chunker actually has to answer is not "which
		# chunk" but "which prompt lengths can be tiled by safe widths
		# at all", and the answer is computed here rather than assumed.
		top = 2 * bigall + 64
		reach[0] = 1
		for (m = 1; m <= top; m++) {
			reach[m] = 0
			for (i = 1; i <= n; i++) {
				w = W[i] + 0
				if (!(w in btot)) continue
				if (bok[w] != btot[w] || btot[w] == 0) continue
				if (w <= m && reach[m - w]) { reach[m] = 1; break }
			}
		}
		nbad = 0; badl = ""; firstok = -1; run = 0
		ev_ok = ev_bad = od_ok = od_bad = 0
		for (m = 2; m <= top; m++) {
			if (reach[m]) {
				run++
				if (run >= 32 && firstok < 0) firstok = m - run + 1
				if (m % 2) od_ok++; else ev_ok++
			} else {
				run = 0; firstok = -1
				if (nbad < 24) badl = badl " " m
				nbad++
				if (m % 2) od_bad++; else ev_bad++
			}
		}
		print ""
		print "  prompt lengths a chunker could tile using ONLY safe widths:"
		# ⚠ THE PARITY CASE IS CALLED BY NAME, because it is the one that
		# will actually happen and "no unbroken run of tileable lengths"
		# reads like a wall when it is arithmetic. Every safe width even
		# means every reachable length even, and nothing else.
		if (!ev_bad && !od_ok) {
			print "    EVERY EVEN length from 2 up, and NO odd one."
			print "    That is arithmetic, not a wall: every safe width is even,"
			print "    so an odd prompt is an even one plus ONE row, and a single"
			print "    row is the m = 1 path that has always been correct. The"
			print "    chunker needs a one-token tail, not a new width."
		} else {
			if (firstok >= 0)
				printf "    every length from %d up\n", firstok
			else
				print  "    no unbroken run of tileable lengths -- see the list below"
			if (nbad) printf "    NOT tileable:%s%s\n", badl, (nbad > 24 ? " ... (" nbad " in all)" : "")
			else      print  "    every length from 2 up is tileable"
			if (nbad) print  "    ⚠ a length that cannot be tiled is one the chunker has to"
			if (nbad) print  "      finish a row at a time, which is correct and merely slower."
		}
		print ""
		print "  ⚠ every number in this block is measured over the first"
		printf "  %s tensors only. It is a statement about a layer, not about\n", maxt
		print "  the model: the output head is staged last and is not in it."
	}
	print "======================================================================"
}
' "$T/both.tsv" "$T/onedev.tsv"

# --- m = 8, on its own, over every tensor ----------------------------------
# ⚠ THIS IS THE ONLY BLOCK IN THE ROUND THAT CAN DISCOVER SOMETHING. Everything
# above confirms a rule that was settled by sweeping the read order itself. The
# m = 8 fault is even -- the read order is a clean bijection there -- so it is
# not the read order at all, and the two questions it raises can only be
# answered by every n in the model.
if [ "$DO_M8" != 0 ]; then
	echo
	echo "======================================================================"
	echo "  m = 8, UNCAPPED, THE ONE OPEN QUESTION"
	echo
	if [ "$M8_OK" -eq 0 ]; then
		echo "  ⚠⚠ NO m = 8 PASS PRODUCED A TABLE. The question this round"
		echo "     exists to move is unanswered, and the blocks above are"
		echo "     confirmation only. Do not read the absence as a result."
	else
		for ARM in $ARMS; do
			f="$OUTDIR/width-law-m8-$ARM.txt"
			[ -r "$f" ] || continue
			line=$(awk '$1 == "8" && $2 ~ /^[0-9]+\/[0-9]+$/ && $5 == "of" \
				    { print $4 " of " $6 " rows, worst rel " $3 }' "$f")
			printf '  %-8s %s\n' "$ARM" "${line:-no table row}"
			# ⚠ THE MISS LINES CARRY THE n, AND THE n IS THE WHOLE
			# QUESTION. `row R of 8` is how a MISS line names its
			# width, so this filters on the width and then counts
			# the distinct n. "every ffn_gate and ffn_up" was
			# written down from a count of 33 that nobody had
			# broken down by shape, and 33 is not 32.
			grep 'MISS ' "$f" 2>/dev/null | grep 'of 8,' \
			  | awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^n=/) { sub(/^n=/, "", $i); c[$i]++ } }
				 END { s = ""; t = 0
				       for (k in c) { s = s "  n=" k " x" c[k]; t += c[k] }
				       if (t) printf "           %d missed rows, by width n:%s\n", t, s
				       else   print  "           no MISS lines at m = 8" }'
		done
		echo
		echo "  what to read:"
		echo "    ONE CORE EXACT AT EVERY n, TWO CORES WRONG AT SOME n"
		echo "      the fault is the core PAIR and the n it shows at is a"
		echo "      symptom of which tensors are big enough to overlap."
		echo "      m = 8 is even, so the read order is a clean bijection"
		echo "      there and is excluded by construction."
		echo "    ONE CORE WRONG AT SOME n TOO"
		echo "      then it is not the pair, and every story built on"
		echo "      904/904 with CHARSIU_NPU_ONEDEV was built on one"
		echo "      configuration of one model. Name the n and start there."
		echo "    TWO CORES WRONG AT EVERY n"
		echo "      then n = 8192 was never the discriminator and the"
		echo "      capped sweeps that only ever saw two or three n have"
		echo "      been reporting a shape that is not there."
		echo
		echo "  ⚠ the split flag in src/job.c is NOT the mechanism and was"
		echo "    checked: surf * rows > 4096 turns ON above M = 42 at"
		echo "    k = 3072 and above M = 128 at the k = 1024 slices this"
		echo "    environment produces, so it is off at m = 8 and on at"
		echo "    48, 64 and 80, which are exact. It is the opposite of"
		echo "    the failure pattern in both directions."
	fi
	echo "======================================================================"
fi

echo
echo "  full logs: $OUTDIR/width-law-{both,onedev}.txt"
echo "             $OUTDIR/width-law-m8-{both,onedev}.txt"
echo "  ⚠ they are uncapped on purpose. The MISS lines and the where-did-it-go"
echo "    scan sit UNDER the timing table, and every summary above is a"
echo "    summary -- the reason a width failed is in the log and nowhere else."
