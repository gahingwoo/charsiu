#!/bin/sh
# The deal's per-task weight: the withdrawn 36.8 against the measured 4.81.
# One binary, two arms, alternating.  $1 binary  $2 repeats
#
# npudev's own note says "swapping 36.8 for 4.81 without measuring would replace
# a refuted number with an unmeasured one.  It is on the board list."  This is
# that round.  Both arms are the same md5; only the environment moves.
set -u
BIN="${1:-/opt/charsiu/charsiu_run_minmac}"
N="${2:-4}"
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_COEF_ELEMS=65536"
MODELS="/opt/vendor/models/gemma-4-E2B-it-Q4_0.gguf /opt/vendor/models/gemma-3-1b-it-Q4_0.gguf /opt/vendor/models/Qwen3-0.6B-Q4_0.gguf"

for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo performance > "$p/scaling_governor" 2>/dev/null || true
done
sleep 1

mid() { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$0} END{if(NR%2)printf "%s\n",a[(NR+1)/2]; else printf "%.2f\n",(a[NR/2]+a[NR/2+1])/2}'; }
lo() { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | head -1; }
hi() { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1; }
rate() { grep '^\[load' "$1" | sed 's/.*gen [0-9]* tok in [0-9]* ms, \([0-9.]*\) tok.s.*/\1/'; }
# ⚠ charsiu_run prints TWO timing lines on stdout: [load ...] and [first 8 tok
# ...].  Hashing with only the first filtered gave five different hashes for
# five identical runs of one arm, in the min_mac round -- the check said
# "DIFFERS" and was measuring the clock.  Filter every line that starts '['.
text() { grep -v '^\[' "$1" | grep -v '^charsiu' | md5sum | cut -c1-12; }

echo "boot $(cat /proc/sys/kernel/random/boot_id)"
echo "bin  $BIN  $(md5sum "$BIN" | cut -c1-12)"
echo "clk  $(cat /sys/kernel/debug/clk/clk_rknn_dsu0/clk_rate 2>/dev/null) governor performance"
echo "arms A DEAL_US_TASK=36.8 (the shipping, withdrawn value)   B DEAL_US_TASK=4.81 (round 152's measurement)"
echo "     $N alternating pairs, 64 tokens, 128 token prompt is NOT used here -- short prompt, decode only"

for M in $MODELS; do
	[ -f "$M" ] || { echo "########## $(basename "$M") NOT ON THIS BOARD -- skipped"; continue; }
	echo ""
	echo "########## $(basename "$M")"
	for V in 36.8 4.81; do
		echo "   balance at $V: $(env $E CHARSIU_NPU_DEAL_US_TASK=$V "$BIN" "$M" -p "hello" -n 8 --ignore-eos -c 1024 -t 4 2>&1 >/dev/null | grep 'even share' | sed 's/charsiu NPU: //')"
	done
	env $E CHARSIU_NPU_DEAL_US_TASK=36.8 "$BIN" "$M" -p "hello" -n 16 --ignore-eos -c 1024 -t 4 >/dev/null 2>&1
	env $E CHARSIU_NPU_DEAL_US_TASK=4.81 "$BIN" "$M" -p "hello" -n 16 --ignore-eos -c 1024 -t 4 >/dev/null 2>&1
	TA=; TB=; HA=; HB=; n=0
	while [ $n -lt "$N" ]; do
		n=$((n+1))
		env $E CHARSIU_NPU_DEAL_US_TASK=36.8 "$BIN" "$M" -p "hello" -n 64 --ignore-eos -c 1024 -t 4 > /tmp/a.$$ 2>/dev/null
		env $E CHARSIU_NPU_DEAL_US_TASK=4.81 "$BIN" "$M" -p "hello" -n 64 --ignore-eos -c 1024 -t 4 > /tmp/b.$$ 2>/dev/null
		TA="$TA $(rate /tmp/a.$$)"; TB="$TB $(rate /tmp/b.$$)"
		HA="$HA $(text /tmp/a.$$)"; HB="$HB $(text /tmp/b.$$)"
	done
	rm -f /tmp/a.$$ /tmp/b.$$
	MA=$(mid "$TA"); MB=$(mid "$TB")
	printf '   A %7s tok/s  %s..%s\n' "$MA" "$(lo "$TA")" "$(hi "$TA")"
	printf '   B %7s tok/s  %s..%s\n' "$MB" "$(lo "$TB")" "$(hi "$TB")"
	U=$(printf '%s\n' $HA $HB | sort -u | wc -l)
	if [ "$U" = 1 ]; then echo "   text IDENTICAL across both arms and all $N reps ($(printf '%s\n' $HA | head -1))"
	else echo "   text DIFFERS -- $U distinct over both arms:"; echo "     A$HA"; echo "     B$HB"; fi
	awk "BEGIN{ma=$MA;mb=$MB;l=$(lo "$TA");h=$(hi "$TA");lb=$(lo "$TB");hb=$(hi "$TB");
	 marg=100*(mb-ma)/ma;spr=100*(h-l)/ma;sprb=100*(hb-lb)/mb;am=(marg<0)?-marg:marg;
	 s=(spr>sprb)?spr:sprb; v=\"level (margin inside the wider arm spread)\";
	 if(am>s)v=(marg>0)?\"4.81 WINS\":\"4.81 LOSES\";
	 printf \"   B/A %.4f   margin %+.2f%%   A spread %.2f%%  B spread %.2f%%   %s\n\",mb/ma,marg,spr,sprb,v}"
done

for p in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil > "$p/scaling_governor" 2>/dev/null || true; done
echo ""
echo "boot $(cat /proc/sys/kernel/random/boot_id)"
