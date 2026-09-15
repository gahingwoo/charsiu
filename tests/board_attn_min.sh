#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# HOW LONG A PROMPT BEFORE fp16 ATTENTION ON THE NPU PAYS -- ASKED OF EVERY
# MODEL ON THE CARD, BECAUSE THE THRESHOLD IS ONE NUMBER FOR ALL OF THEM.
#
# attn_npu_min_tokens() is 448 and that number was chosen in r411 against an
# arm that was 478 ms slow (the reservation default was still on when the
# sweep ran, section 6 of the r411 log). Re-measured on the shipping arm,
# Llama-3.2-1B says the crossover is between 302 and 352, not at 448:
#
#     252   CPU 1766 (1744..1786)   NPU 1759 (1754..1766)   0.996  level
#     302   CPU 2138 (2100..2146)   NPU 2088 (2082..2092)   0.977  margin 2.3%, spread 2.2%
#     352   CPU 2534 (2510..2553)   NPU 2390 (2386..2396)   0.943  margin 5.7%, spread 1.7%
#
# A MARGIN HAS TO CLEAR TWO THINGS AND THE SECOND ONE IS THIS ARM'S OWN
# SPREAD. 302 above is 2.3% ahead with the CPU arm's three readings spanning
# 2.2%, which is not a lead; 352 is 5.7% ahead of a 1.7% spread, which is. The
# verdict column below computes that rather than leaving it to whoever reads
# the table, because r411 got it wrong in prose with the ranges printed on the
# same line.
#
# AND ONE MODEL CANNOT SET A THRESHOLD THAT EVERY MODEL OBEYS. The mirror
# costs a fixed build per layer per kv head, so a model with many kv heads and
# a narrow head pays more for it: the note in npufp16.c has Qwen3 and TinyLLAMA
# LOSING 35% of TTFT at 110 tokens. The crossover is per model and the knob is
# not, so the threshold that ships has to be the LAST model's crossover, not
# the first one's.
#
#   CHARSIU_MIN_LENS="252 302 352 452"   token targets (clauses of about 25)
#   CHARSIU_MIN_N=3                      repeats an arm a length
#   CHARSIU_MIN_MODELS=9                 refuse below this many models
. "$(dirname "$0")/board_clk.sh"
NPUCLK=$(npu_clk) || exit 1
set -u

LENS=${CHARSIU_MIN_LENS:-302 352}
N=${CHARSIU_MIN_N:-3}
RUN=${CHARSIU_RUN:-/opt/charsiu/charsiu_run}
MINM=${CHARSIU_MIN_MODELS:-9}
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"
CLAUSE="A written record can be copied, checked against other records, and read by someone who never met the person who wrote it. "
DIRS="/opt/charsiu/models /opt/vendor/models ${CHARSIU_BOARD_DIR:-$HOME/charsiu-board}"

[ -n "$(ls /dev/accel/accel* 2>/dev/null)" ] || { echo "no /dev/accel"; exit 1; }
[ -x "$RUN" ] || { echo "no $RUN"; exit 1; }

# SAY WHAT WAS SEARCHED AND WHAT WAS FOUND. An unmounted /opt/vendor looks
# exactly like an empty one, and it came unmounted twice on 2026-09-14.
MODELS=""
for d in $DIRS; do
	n=$(ls "$d"/*.gguf 2>/dev/null | wc -l)
	printf '  %-40s %s gguf\n' "$d" "$n"
	[ "$n" -gt 0 ] && MODELS="$MODELS $(ls "$d"/*.gguf 2>/dev/null)"
done
# ONE COPY OF EACH MODEL. Llama-3.2-1B is in both directories on this board,
# and measuring it twice would put the same model in the table under two paths
# and make a nine model card read as ten.
MODELS=$(for f in $MODELS; do echo "$(basename "$f") $f"; done \
	 | sort -u -k1,1 | awk '{print $2}')
# A SUBSET IS A SMALLER QUESTION AND HAS TO SAY SO. Ten models at two
# lengths is two hours of board; CHARSIU_MIN_ONLY narrows it to the models
# whose crossover is actually in doubt -- a SPACE SEPARATED list of
# substrings, so three models can be named. The refusal below still counts what
# was FOUND, so narrowing cannot be mistaken for a missing mount.
if [ -n "${CHARSIU_MIN_ONLY:-}" ]; then
	SEL=""
	for f in $MODELS; do
		b=$(basename "$f")
		for pat in $CHARSIU_MIN_ONLY; do
			case "$b" in *"$pat"*) SEL="$SEL $f"; break ;; esac
		done
	done
	echo "  CHARSIU_MIN_ONLY=$CHARSIU_MIN_ONLY: $(printf '%s\n' $SEL | grep -c .) of $(printf '%s\n' $MODELS | grep -c .) models"
fi
NM=$(printf '%s\n' $MODELS | grep -c .)
if [ "$NM" -lt "$MINM" ]; then
	echo "found $NM models, expected $MINM -- mount /opt/vendor or lower"
	echo "CHARSIU_MIN_MODELS. Reporting on a subset is how a round answers a"
	echo "smaller question than the one asked."
	exit 1
fi

for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo userspace > "$p/scaling_governor" 2>/dev/null || true
	tr ' ' '\n' < "$p/scaling_available_frequencies" | grep -v '^$' | sort -n \
		| tail -1 > "$p/scaling_setspeed" 2>/dev/null || true
done
sleep 1

echo
echo "== where fp16 attention starts paying, by model"
echo "   boot      $(cat /proc/sys/kernel/random/boot_id)"
echo "   npu clk   $NPUCLK Hz"
echo "   binary    $RUN"
echo "   build     $(charsiu_build "$RUN")"
echo "   arms      CHARSIU_ATTN_NPU=0 against =1; the default (auto) is NEITHER"
echo "   $N repeats an arm a length, arms alternating, one warm-up discarded"
echo "   verdict   'wins' only when the margin clears the CPU arm's own spread"
echo

mid() { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$0} END{if(NR%2)printf "%s\n",a[(NR+1)/2]; else printf "%.0f\n",(a[NR/2]+a[NR/2+1])/2}'; }
lo()  { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | head -1; }
hi()  { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1; }

run1() { # $1 = arm env, $2 = model, $3 = prompt
	env $E $1 "$RUN" "$2" -p "$3" -n 1 --ignore-eos -c 1024 -t 4 2>/dev/null \
		| grep '^\[load' \
		| sed 's/.*prompt \([0-9]*\) tok in \([0-9]*\) ms.*/\1 \2/'
}

[ -z "${SEL:-}" ] || MODELS="$SEL"

for L in $LENS; do
	R=$(( (L + 12) / 25 ))
	P=""; i=0
	while [ $i -lt "$R" ]; do P="$P$CLAUSE"; i=$((i+1)); done
	echo "   -- about $L tokens ($R clauses) --"
	printf '   %-34s %6s  %7s %13s  %7s %13s  %7s %7s  %s\n' \
		model tok 'CPU ms' range 'NPU ms' range margin spread verdict
	for M in $MODELS; do
		run1 "CHARSIU_ATTN_NPU=0" "$M" "$P" >/dev/null 2>&1
		run1 "CHARSIU_ATTN_NPU=1" "$M" "$P" >/dev/null 2>&1
		A=; B=; TOK=; n=0
		while [ $n -lt "$N" ]; do
			n=$((n+1))
			r=$(run1 "CHARSIU_ATTN_NPU=0" "$M" "$P"); TOK=${r%% *}; A="$A ${r##* }"
			r=$(run1 "CHARSIU_ATTN_NPU=1" "$M" "$P"); B="$B ${r##* }"
		done
		[ -n "$A" ] && [ -n "$B" ] || { printf '   %-34s NO OUTPUT\n' "$(basename "$M")"; continue; }
		MA=$(mid "$A"); MB=$(mid "$B")
		# THE TEST IS ON THE SIZE OF THE MARGIN, NOT ITS SIGN. Comparing
		# a signed margin against a spread reads every LOSS as "level",
		# which is the friendly direction and therefore the wrong one.
		#
		# AND THE TERNARY IS ON ONE LINE BECAUSE BUSYBOX AWK ENDS A
		# STATEMENT AT A NEWLINE: split across the `:` it is "Unexpected
		# token" on the board and nothing else -- the numbers still
		# print, only the verdict is missing, which is the column the
		# script exists for.
		V=$(awk "BEGIN{
			ma=$MA; mb=$MB; l=$(lo "$A"); h=$(hi "$A");
			marg=100*(ma-mb)/ma; spr=100*(h-l)/ma;
			am=(marg<0)?-marg:marg;
			v=\"level\";
			if (am > spr) v=(marg>0)?\"NPU wins\":\"NPU LOSES\";
			printf \"%+6.1f%% %6.1f%%  %s\", marg, spr, v;
		}")
		printf '   %-34s %6s  %7s %5s..%-5s  %7s %5s..%-5s  %s\n' \
			"$(basename "$M" .gguf)" "$TOK" \
			"$MA" "$(lo "$A")" "$(hi "$A")" \
			"$MB" "$(lo "$B")" "$(hi "$B")" "$V"
	done
	echo
done

echo "restoring schedutil"
for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo schedutil > "$p/scaling_governor" 2>/dev/null || true
done
