#!/bin/sh
# One binary, two arms, alternating: does refusing a tiny NPU dispatch pay?
#   $1 binary   $2 threshold in MACs   $3 repeats
# Arm A is CHARSIU_NPU_MIN_MAC=0 (the shipping default, named), arm B is the
# threshold.  gemma-4-E2B is the model under test; gemma-3-1b is the NULL
# CONTROL -- its smallest dispatch is 1.77 MMAC, so the gate cannot fire, and
# the refusal line must not print for it.
set -u
BIN="${1:-/opt/charsiu/charsiu_run_minmac}"
T="${2:-1000000}"
N="${3:-5}"
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_COEF_ELEMS=65536"
M4=/opt/vendor/models/gemma-4-E2B-it-Q4_0.gguf
M3=/opt/vendor/models/gemma-3-1b-it-Q4_0.gguf

for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo userspace > "$p/scaling_governor" 2>/dev/null || true
	tr ' ' '\n' < "$p/scaling_available_frequencies" | grep -v '^$' | sort -n | tail -1 > "$p/scaling_setspeed" 2>/dev/null || true
done
sleep 1

mid() { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$0} END{if(NR%2)printf "%s\n",a[(NR+1)/2]; else printf "%.2f\n",(a[NR/2]+a[NR/2+1])/2}'; }
lo() { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | head -1; }
hi() { printf '%s\n' "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1; }

# rate, and the md5 of the TEXT (not of the timings).
# charsiu_run prints TWO timing lines on stdout: [load ...] and [first 8 tok
# ...].  The first version of this filtered only '^\[load' and so hashed the
# second one -- five identical runs of one arm gave five different hashes and
# the check reported "text DIFFERS" while measuring the clock.  Filter every
# line that starts '['.
rate() { grep '^\[load' "$1" | sed 's/.*gen [0-9]* tok in [0-9]* ms, \([0-9.]*\) tok.s.*/\1/'; }
text() { grep -v '^\[' "$1" | grep -v '^charsiu' | md5sum | cut -c1-12; }

echo "boot $(cat /proc/sys/kernel/random/boot_id)"
echo "bin  $BIN  $(md5sum "$BIN" | cut -c1-12)"
echo "clk  $(cat /sys/kernel/debug/clk/clk_rknn_dsu0/clk_rate 2>/dev/null) governor userspace at max"
echo "arms A CHARSIU_NPU_MIN_MAC=0   B CHARSIU_NPU_MIN_MAC=$T   $N alternating pairs, 64 tokens"

for M in "$M4" "$M3"; do
	echo ""
	echo "########## $(basename "$M")"
	# the refusal, said once, from arm B -- this is the tell that the gate fired
	env $E CHARSIU_NPU_MIN_MAC=$T "$BIN" "$M" -p "hello" -n 8 --ignore-eos -c 1024 -t 4 2>&1 >/dev/null \
		| grep 'under CHARSIU_NPU_MIN_MAC' | sed 's/^/   refused: /'
	# warm up BOTH arms: the first point in a round is cold
	env $E CHARSIU_NPU_MIN_MAC=0  "$BIN" "$M" -p "hello" -n 16 --ignore-eos -c 1024 -t 4 >/dev/null 2>&1
	env $E CHARSIU_NPU_MIN_MAC=$T "$BIN" "$M" -p "hello" -n 16 --ignore-eos -c 1024 -t 4 >/dev/null 2>&1
	TA=; TB=; HA=; HB=; n=0
	while [ $n -lt "$N" ]; do
		n=$((n+1))
		env $E CHARSIU_NPU_MIN_MAC=0  "$BIN" "$M" -p "hello" -n 64 --ignore-eos -c 1024 -t 4 > /tmp/a.$$ 2>/dev/null
		env $E CHARSIU_NPU_MIN_MAC=$T "$BIN" "$M" -p "hello" -n 64 --ignore-eos -c 1024 -t 4 > /tmp/b.$$ 2>/dev/null
		TA="$TA $(rate /tmp/a.$$)"; TB="$TB $(rate /tmp/b.$$)"
		HA="$HA $(text /tmp/a.$$)"; HB="$HB $(text /tmp/b.$$)"
	done
	rm -f /tmp/a.$$ /tmp/b.$$
	MA=$(mid "$TA"); MB=$(mid "$TB")
	printf '   A %7s tok/s  %s..%s\n' "$MA" "$(lo "$TA")" "$(hi "$TA")"
	printf '   B %7s tok/s  %s..%s\n' "$MB" "$(lo "$TB")" "$(hi "$TB")"
	echo "   A text$HA"
	echo "   B text$HB"
	UA=$(printf '%s\n' $HA | sort -u | wc -l); UB=$(printf '%s\n' $HB | sort -u | wc -l)
	if [ "$UA" = 1 ] && [ "$UB" = 1 ] && [ "$(printf '%s\n' $HA | sort -u)" = "$(printf '%s\n' $HB | sort -u)" ]; then
		echo "   text IDENTICAL across both arms and all $N reps"
	else
		echo "   text DIFFERS -- A unique $UA, B unique $UB"
	fi
	awk "BEGIN{ma=$MA;mb=$MB;l=$(lo "$TA");h=$(hi "$TA");lb=$(lo "$TB");hb=$(hi "$TB");
	 marg=100*(mb-ma)/ma;spr=100*(h-l)/ma;sprb=100*(hb-lb)/mb;am=(marg<0)?-marg:marg;
	 s=(spr>sprb)?spr:sprb; v=\"level (margin inside the wider arm spread)\";
	 if(am>s)v=(marg>0)?\"B WINS\":\"B LOSES\";
	 printf \"   B/A %.4f   margin %+.2f%%   A spread %.2f%%  B spread %.2f%%   %s\n\",mb/ma,marg,spr,sprb,v}"
done

for p in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil > "$p/scaling_governor" 2>/dev/null || true; done
echo ""
echo "boot $(cat /proc/sys/kernel/random/boot_id)"
