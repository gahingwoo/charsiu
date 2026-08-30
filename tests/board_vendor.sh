#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# charsiu against Rockchip's own published numbers, on their protocol.
#
# The table below is copied from
#   https://raw.githubusercontent.com/airockchip/rknn-llm/main/benchmark.md
# fetched 2026-08-28, RK3576 section, and it is the vendor's runtime on the
# vendor's driver. Their protocol is a 128 token prompt and 64 new tokens, so
# that is what this runs.
#
# ⚠⚠ THEIR NUMBERS ARE AT MAXIMUM CPU AND NPU FREQUENCY. Their own header says
# so: "collected based on the maximum CPU and NPU frequencies of each platform",
# with a script in their repo to set them. This runs at whatever the governor is
# doing, which on an idle board is ondemand. That is not a small difference and
# it is not one to quietly leave out of a comparison -- CHARSIU_BENCH_PERF=1
# sets the performance governor first and says it did.
#
# ⚠ AND w4a16 IS THEIR int4, WHICH IS OURS. w4a16_g128 is a finer group size
# and w8a8 is int8; the rows compared here are w4a16 against CHARSIU_NPU_W4V=1.
#
#   sh tests/board_vendor.sh
set -eu

REPEAT=${CHARSIU_BENCH_REPEAT:-1}
DIR=${CHARSIU_BOARD_DIR:-$HOME/charsiu-board}
mkdir -p "$DIR"
# ⚠⚠ PRICING THE CORRECTNESS FIX, AND THIS ARM RETURNS WRONG TEXT.
#
# The two NPU cores corrupt each other when their batched submits overlap --
# phi3 at width 24 came back wrong 13 runs of 16 overlapped and 0 of 16
# serialised -- so serialised is the default and it costs a batched
# projection's parallelism. How much that costs is a fair question and it has
# exactly one honest answer: run the broken configuration and read the clock.
#
# CHARSIU_VENDOR_PRICE_SERIAL=1 does that. Every number it produces is the
# speed of an answer that may be wrong, it is not a configuration to ship, and
# the banner says so on every round that sets it.
PRICE_ENV=""
if [ -n "${CHARSIU_VENDOR_PRICE_SERIAL:-}" ]; then
	PRICE_ENV="CHARSIU_NPU_BATCH_PARALLEL=1"
	echo "======================================================================"
	echo "⚠⚠ CHARSIU_VENDOR_PRICE_SERIAL: the two cores' submits OVERLAP in"
	echo "   this round. That is the configuration measured wrong 13 runs of"
	echo "   16 on phi3. EVERY NUMBER BELOW IS THE SPEED OF A POSSIBLY WRONG"
	echo "   ANSWER, and exists only to price what serialising costs."
	echo "   Compare its TTFT column against a normal round; ship neither"
	echo "   this build nor this conclusion without board_text_all.sh."
	echo "======================================================================"
	echo
fi

MODELS=${CHARSIU_MODELS:-$HOME/.charsiu/models}
[ -d "$MODELS" ] || MODELS=/opt/charsiu/models

# ⚠ CHARSIU_RUN_BIN, so the shell around the table can be exercised without a
# board. The repeat loop below went in untested because a missing model takes a
# different branch and a present-but-empty one takes the failure branch, so
# neither reaches it; a stand-in binary that prints one [load ...] line does.
RUN=${CHARSIU_RUN_BIN:-}
[ -n "$RUN" ] || for d in /usr/bin /opt/charsiu "$PWD/build"; do
	[ -x "$d/charsiu_run" ] && { RUN="$d/charsiu_run"; break; }
done
[ -n "$RUN" ] || { echo "charsiu_run not found" >&2; exit 1; }

# ⚠ THE PULL NAME IS IN THE ROW, because "not here" without it is a dead end:
# the zoo's name and the file's name are not the same string and nobody should
# have to grep charsiu-get to find out which to type.
#
# pull name | file | vendor tok/s (w4a16) | vendor TTFT ms | vendor MB | label
rows() {
cat <<'ROWS'
qwen3-0.6b-q4|Qwen3-0.6B-Q4_0.gguf|24.85|468.61|512.71|Qwen3 0.6B
tinyllama-1.1b-q4|tinyllama-1.1b-chat-v1.0.Q4_0.gguf|19.71|543.68|601.09|TinyLLAMA 1.1B
phi3.5-mini-q4|Phi-3.5-mini-instruct-Q4_0.gguf|6.58|1829.12|1995.78|Phi3 3.8B
gemma4-e2b-q4|gemma-4-E2B-it-Q4_0.gguf|9.23|1219.25|1463.42|Gemma4 E2B
ROWS
}

# ⚠ 128 TOKENS IS THEIR SEQLEN, so the prompt has to be about that and the
# actual count is printed rather than assumed -- a comparison whose left column
# ran 40 tokens against their 128 is not a comparison.
PROMPT="The history of computing begins long before the first electronic machine.
Mechanical calculators, punched cards and tabulating engines each solved a part
of the problem, and each was built by somebody who wanted an answer faster than
a room full of clerks could produce one. What changed with the stored program
computer was not arithmetic but instruction: the machine could be told what to
do next by the same medium that held its data. Explain in your own words why
that distinction matters, and what it made possible that had not been possible."

if [ "${CHARSIU_BENCH_PERF:-0}" = 1 ]; then
	echo "setting the performance governor, as their protocol does"
	for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
		[ -w "$g" ] && echo performance > "$g" || true
	done
	echo "  governors now: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
else
	echo "⚠ the governor is left alone; their numbers are at MAXIMUM frequency."
	echo "  CHARSIU_BENCH_PERF=1 sets it, and this line changes when you do."
fi
echo "  governor: $(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
echo

printf '%-16s %10s %10s   %10s %10s   %8s %8s\n' \
	model "ours t/s" "theirs" "ours TTFT" "theirs" "ours MB" "theirs"

rows | while IFS='|' read -r name file vt vttft vmb label; do
	M=""
	for d in "$MODELS" "$DIR"; do
		[ -f "$d/$file" ] && { M="$d/$file"; break; }
	done
	if [ -z "$M" ]; then
		printf '%-16s %10s %10s   %10s %10s   %8s %8s   charsiu pull %s\n' \
			"$label" "-" "$vt" "-" "$vttft" "-" "$vmb" "$name"
		continue
	fi
	# ⚠ STDERR IS KEPT. The refusal that decides whether the batched prefill
	# does anything at all -- "int4 computes one row" -- is on it, and a
	# comparison that throws it away cannot tell a batched run from a run
	# that fell back to exactly what it was being compared against.
	# ⚠⚠ A FAILING RUN USED TO END THE WHOLE SCRIPT, IN SILENCE. `set -e`
	# plus OUT=$(cmd) means one model that will not load takes every model
	# after it with it -- three rounds of this table printed exactly one row
	# and nobody, me included, asked where the other four went. A row that
	# fails is a row, and it says so.
	# ⚠⚠ ONE RUN IS A READING AND THIS COLUMN IS THE HEADLINE. Qwen3's TTFT
	# came back 2055, 1867 and 2191 on three consecutive rounds of the same
	# build -- a spread of 9% -- because the governor is ondemand and each
	# model runs once. A change worth less than that cannot be seen here at
	# all, and the round that added the four-at-a-time gather read WORSE on
	# this table while the probe's own read column, which compares inside
	# one run, said 28% better.
	#
	# CHARSIU_BENCH_REPEAT=3 runs each model three times and prints the
	# best with the spread beside it, so a number and its noise arrive
	# together. The default is still 1, because three times four models is
	# a long round.
	BEST_TT=; BEST_TS=; BEST_MB=; BEST_NP=; WORST_TT=; nrun=0
	while [ $nrun -lt "$REPEAT" ]; do
		nrun=$((nrun + 1))
		# ⚠⚠ env, NOT A BARE ASSIGNMENT PREFIX. A shell only treats
		# NAME=VALUE as an assignment when it is written literally in
		# the command position; a VARIABLE that expands to NAME=VALUE is
		# looked up as a COMMAND. Adding $PRICE_ENV to the prefix form
		# turned four rows into
		#
		#   board_vendor.sh: 142: CHARSIU_NPU_BATCH_PARALLEL=1: not found
		#
		# `env` takes them as arguments, so an expanded one works.
		# shellcheck disable=SC2086
		OUT=$(env CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
		      CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
		      CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536 \
		      $PRICE_ENV \
		      "$RUN" "$M" -p "$PROMPT" -n 64 --ignore-eos -c 512 -t 4 \
		      2>"$DIR/.v.err" | grep '^\[load') || OUT=""
		[ -n "$OUT" ] || continue
		NP=$(echo "$OUT" | sed 's/.*| *prompt \([0-9]*\) tok.*/\1/')
		TT=$(echo "$OUT" | sed 's/.*prompt [0-9]* tok in \([0-9]*\) ms.*/\1/')
		TS=$(echo "$OUT" | sed 's/.*| *gen [0-9]* tok in [0-9]* ms, \([0-9.]*\) tok.*/\1/')
		MB=$(echo "$OUT" | sed 's/.*peak \([0-9]*\) MB.*/\1/')
		if [ -z "$BEST_TT" ] || [ "$TT" -lt "$BEST_TT" ] 2>/dev/null; then
			BEST_TT=$TT; BEST_TS=$TS; BEST_MB=$MB; BEST_NP=$NP
		fi
		if [ -z "$WORST_TT" ] || [ "$TT" -gt "$WORST_TT" ] 2>/dev/null; then
			WORST_TT=$TT
		fi
	done
	# ⚠⚠ A FAILING RUN USED TO END THE WHOLE SCRIPT, IN SILENCE. `set -e`
	# plus OUT=$(cmd) means one model that will not load takes every model
	# after it with it -- three rounds of this table printed exactly one row
	# and nobody, me included, asked where the other four went. A row that
	# fails is a row, and it says so.
	if [ -z "$BEST_TT" ]; then
		printf '%-16s %10s %10s   %10s %10s   %8s %8s   THE RUN FAILED\n' \
			"$label" "-" "$vt" "-" "$vttft" "-" "$vmb"
		tail -3 "$DIR/.v.err" | sed 's/^/     /'
		continue
	fi
	TS=$BEST_TS; TT=$BEST_TT; MB=$BEST_MB; NP=$BEST_NP
	spread=""
	[ "$REPEAT" -gt 1 ] && spread=" ttft $BEST_TT..$WORST_TT over $REPEAT runs"
	printf '%-16s %10s %10s   %10s %10s   %8s %8s   (%s tok prompt)%s\n' \
		"$label" "$TS" "$vt" "$TT" "$vttft" "$MB" "$vmb" "$NP" "$spread"
	# ⚠ THE REFUSAL STRING MOVED ON 2026-08-29 and this grep did not: it
	# looked for "int4 computes one row", which no longer exists anywhere,
	# so it matched nothing and said nothing for a round. It looks for the
	# m = 8 fallback and the batch_why_not list now.
	#
	# ⚠ AND THE DEVICE'S OWN REPORT. "how many submits" is the difference
	# between a prefill that runs on the hardware one row at a time -- a
	# fence each, and the fence has been measured at 94% of the hardware
	# path -- and one that fell back to the CPU. Nothing else in this table
	# can tell those two apart.
	# ⚠ MATCH THE REFUSAL BY ITS PREFIX, NOT BY ONE OF ITS REASONS. This
	# read "int4 at m=", which is the m = 8 refusal's own words, so the
	# odd-width refusal added later would have gone through this filter
	# unseen -- a prefill silently falling back a row at a time, in the
	# table that exists to tell exactly those two cases apart.
	grep -E "prompt batched|prompt a token|NOT on the NPU|not batched" \
		"$DIR/.v.err" | sed 's/^/     /' | sort -u || true
	grep -E "^charsiu NPU: .*(submits|hardware path)" "$DIR/.v.err" \
		| sed 's/^/     /' || true
done

echo
# ⚠⚠ THE MEMORY COLUMN UNDERSTATES US, AND BY MORE THAN THE GAP IT SHOWS.
#
# charsiu_run's `peak N MB` is VmHWM, and every DRM buffer object is invisible
# to it: rocket's objects come from drm_gem_shmem, whose mmap sets VM_PFNMAP
# and faults with vmf_insert_pfn, and VM_PFNMAP pages are never added to the
# RSS counters. So the weights on the hardware, the coefficient buffers and
# the batched output buffers are all absent from this column.
#
# Computed from the real shapes at the settings above: Qwen3 holds another
# 488 MiB of device buffers and Phi-3.5 another 2618 MiB. True DRAM is nearer
# 1556 and 8528 MiB against the 1068 and 5910 printed here.
#
# Three board logs say the same thing from the other side. r382 measured 2526
# MiB where the host-anon model predicts 2472; adding the device buffers would
# put the prediction 27% ABOVE a measurement. r364 and r366 change the slice
# count enough to move the device buffers by 60 MB and the peak does not move.
# r371 sweeps CPU_FRAC, whose buffer is host anon, and the peak tracks it
# exactly.
#
# ⚠ AND THE UNITS DIFFER. This column is MiB (hwm/1024) and Rockchip's is
# labelled MB, so 1068 here is 1120 of theirs and the honest ratio is 2.18x
# rather than 2.08x.
echo
echo "⚠ THE MEMORY COLUMN IS VmHWM AND DOES NOT SEE THE DEVICE BUFFERS."
echo "  Weights on the hardware, coefficient buffers and batched output"
echo "  buffers are all VM_PFNMAP and never counted. Computed at these"
echo "  settings that is another 488 MiB on Qwen3 and 2618 MiB on Phi-3.5."
echo "  It is not the number that decides which models fit on the board."
echo "  Our column is MiB and theirs is MB, which is a further 5%."
echo
echo "⚠ TTFT AND PROMPT TIME ARE NOT THE SAME THING. Theirs is time to the first"
echo "  token, ours is the prompt's forward passes; the first token's own step is"
echo "  in theirs and not in ours, which is one token's worth in our favour."
echo
echo "SmolVLM-256M, the one multimodal model both sides publish, RK3576 w4a16:"
echo "  theirs   img-encoder 512x512   768 ms   prefill(128) 180 ms   decode 57.73 t/s"
if [ -f "$DIR/mmproj.gguf" ] && [ -f "$DIR/smolvlm.gguf" ]; then
	VIS=""
	for d in /usr/bin /opt/charsiu "$PWD/build"; do
		[ -x "$d/charsiu_vision" ] && { VIS="$d/charsiu_vision"; break; }
	done
	if [ -n "$VIS" ]; then
		t0=$(awk '{printf "%d", $1 * 1000}' /proc/uptime)
		CHARSIU_NPU=1 CHARSIU_STAGES=1 "$VIS" "$DIR/mmproj.gguf" \
			--encode >/dev/null 2>"$DIR/.vis2.err"
		t1=$(awk '{printf "%d", $1 * 1000}' /proc/uptime)
		echo "  ours     img-encoder 512x512   $((t1 - t0)) ms   (staging included)"
		echo
		# ⚠ WHERE, not just how much. 768 ms against ours is a number to
		# chase and the table is the only thing that says which part of
		# it to chase.
		sed -n '/charsiu vision:/,/pixel shuffle/p' "$DIR/.vis2.err" \
			| sed 's/^/  /'
		grep -E "NPU pool|ON THE HARDWARE" "$DIR/.vis2.err" | sed 's/^/  /' || true
	fi
fi
