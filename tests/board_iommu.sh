#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# THE 19.5 ms WAS NEVER MEASURED, AND THIS DOES NOT MEASURE IT EITHER.
#
# What this project has been writing down is "the per-job IOMMU attach/detach
# costs about 19.5 ms of a 51.7 ms decode token". That number is a per-call
# floor of 130 us -- itself the runtime's cost model, not a reading of those
# two calls -- multiplied by about 150 jobs a core a token. The multiplication
# is the unmeasured step. This tree's own rule says a floor may be reported as
# a floor and MUST NOT be multiplied into a share of a token, and the 19.5 ms
# is that rule being broken in the project's own writing.
#
# SO THIS SCRIPT DOES NOT PRINT A PERCENTAGE AND WILL NOT BE MADE TO. It
# prints microseconds a job, jobs a token, and milliseconds a token, in three
# separate columns, and it leaves the multiplication undone on purpose. If a
# later edit adds a "% of a token" column, that edit has re-introduced the
# thing this round was written to retire. The reasons the product is not the
# answer are not stylistic:
#
#   - the counter is summed over BOTH cores. "150 jobs a core" and "jobs a
#     token" here are different quantities, off by a factor of two before
#     anybody multiplies anything.
#   - the two cores attach CONCURRENTLY. Wall clock for N jobs is not N times
#     the cost of one, and nothing here can say by how much.
#   - time inside iommu_attach_group() is not time the token loses. Some of it
#     overlaps nothing; some of it overlaps the other core. A floor is a lower
#     bound on the call, not a lower bound on the token.
#
# THE NUMBER THAT WOULD SETTLE IT IS AN A/B, NOT THIS. An arm whose only
# difference is whether the attach/detach runs per job or once, measured as
# tok/s, is the whole answer and needs no arithmetic at all. That arm is
# rfc-send-v12/attach-once/ in linux-rk3576-npu and it needs a reflash. This
# script is the weak measurement you can take without one: a floor, honestly
# labelled, and a count to go with it.
#
# AND IT SAYS WHETHER THAT A/B IS WORTH THE TWO BOOTS, BEFORE THEY ARE
# SPENT. An IOMMU domain belongs to an open DRM FILE; rocket_job_open() gives
# every file ONE drm_sched entity spanning EVERY core; and charsiu_npu_open_mode
# calls charsiu_open(NULL) twice, which open_any_accel() answers with the same
# /dev/accel/accel0 both times. Two files, two domains, either one landing on
# either core. "Keep the domain attached across jobs" only saves the jobs whose
# domain did not change, so the last table below counts exactly that, with one
# pointer compare and no behaviour change. If it reads near 0%, the A/B has
# nothing to find and the 19.5 ms is not merely unmeasured, it is a saving that
# was never available.
#
# IT NEEDS A KERNEL BUILT WITH THE PROBE. CONFIG_DRM_ACCEL_ROCKET_IOMMU_PROBE
# (DEBUG patch 0001-DEBUG-accel-rocket-time-the-per-job-IOMMU-attach-and*.patch)
# creates /sys/kernel/debug/rocket_iommu_probe. ftrace cannot stand in for it:
# the board's kernel config has "# CONFIG_FTRACE is not set" and no
# CONFIG_TRACING, so fs/tracefs is not compiled in and there is no
# /sys/kernel/debug/tracing on that board at all.
#
# AND A COUNT OF ZERO READS EXACTLY LIKE A CLOCK THAT IS NOT MOUNTED. A
# stats file that exists but never counted -- probe armed on the wrong kernel,
# NPU arm not taken, every job served from the other path -- divides to a clean
# "0.0 us a job" and looks like an answer. The round refuses on a zero count.
#
# AND IF A KERNEL EVER SHIPS WITH ftrace ON, THE PATCH IS THE FALLBACK AND
# NOT THE PLAN. Nothing about these symbols resists tracing: iommu_attach_group
# and iommu_detach_group are exported globals in drivers/iommu/iommu.c with no
# notrace and no CFLAGS_REMOVE on the file, and rocket_job_run has its address
# taken by .run_job so it cannot be inlined away. The only thing missing is the
# tracer. On a kernel built with CONFIG_FUNCTION_GRAPH_TRACER the whole of this
# script's first table is:
#
#   T=/sys/kernel/tracing;  [ -d $T ] || { mount -t tracefs none /sys/kernel/tracing; }
#   echo 0 > $T/tracing_on
#   echo function_graph > $T/current_tracer
#   # EXACT NAMES, NOT A GLOB: iommu_attach_group* also matches
#   #   iommu_attach_group_handle, which is a different function
#   printf 'iommu_attach_group\niommu_detach_group\n' > $T/set_ftrace_filter
#   echo > $T/set_ftrace_notrace
#   # DEPTH 1, so each call is a LEAF. At any greater depth the entry line
#   #   ends in "{" with no duration and the duration lands on a closing "}"
#   #   that carries no function name -- unparseable by name, which is how a
#   #   recipe like this quietly counts nothing.
#   echo 1 > $T/max_graph_depth
#   echo 16384 > $T/buffer_size_kb               # ~150 calls a token fills fast
#   echo > $T/trace
#   echo 1 > $T/tracing_on
#   charsiu_run "$M" -p "$PROMPT" -n 64 --ignore-eos -c 1024 -t 4
#   echo 0 > $T/tracing_on
#   cat $T/trace > /tmp/fg.txt
#
# and microseconds a call come straight out of the DURATION column, which
# function_graph already prints in us. NF-3 rather than a fixed field because
# the marker column ("+" over 100 us, "!" over 1000) and funcgraph-abstime both
# shift the line, and NF-3 is the duration under all four combinations:
#
#   awk '/iommu_attach_group\(\);/ { d = $(NF-3) + 0; n++; s += d
#                                     if (m == "" || d < m) m = d }
#        END { printf "attach n=%d  floor %.1f us  mean %.1f us\n", n, m, s/n }' /tmp/fg.txt
#
# BOUNDING IT TO ONE DECODE IS THE SAME TWO-RUN SUBTRACTION THIS SCRIPT DOES.
# tracing_on around the whole charsiu_run catches the prefill too, so run it
# once at -n 1 and once at -n 64 and subtract the counts; there is no way to
# start the tracer between the prefill and the first token from outside the
# process. Pinning the trace to the pid (set_event_pid / function-fork) does
# not help: the attach runs on the drm_sched kthread, not on charsiu_run.
#
# AND CHECK $T/lost_events BEFORE BELIEVING A COUNT. 150 calls a token times
# 64 tokens times two entries a call overruns a small ring, and a ring that
# dropped records reports a smaller count, not an error.
#
#   CHARSIU_IOMMU_PROBE=...  point at another directory, which is how the
#                            refusals below are exercised with no board at
#                            all: a check nobody has seen fire is not a check
#   CHARSIU_IOMMU_N=3        repeats
#   CHARSIU_IOMMU_NTOK=64    tokens a repeat
#   CHARSIU_IOMMU_FORCE=1    run even though the probe is already armed
. "$(dirname "$0")/board_clk.sh"
set -u

PROBE=${CHARSIU_IOMMU_PROBE:-/sys/kernel/debug/rocket_iommu_probe}
N=${CHARSIU_IOMMU_N:-3}
NTOK=${CHARSIU_IOMMU_NTOK:-64}
RUN=${CHARSIU_RUN:-/opt/charsiu/charsiu_run}
M=${CHARSIU_MODEL:-/opt/charsiu/models/Llama-3.2-1B-Instruct-Q4_0.gguf}
PROMPT=${CHARSIU_IOMMU_PROMPT:-"Explain in plain words why a written record outlasts a memory."}
E="CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_NPU_MAXN=262144 CHARSIU_COEF_ELEMS=65536"

# ---- the refusals, before anything is armed or any governor is touched ----
[ -x "$RUN" ] || { echo "no $RUN" >&2; exit 1; }
[ -r "$M" ]  || { echo "no model $M" >&2; exit 1; }

[ -r "$PROBE/stats" ] || mount -t debugfs none /sys/kernel/debug 2>/dev/null
if [ ! -r "$PROBE/stats" ]; then
	echo "" >&2
	echo "THERE IS NO IOMMU PROBE ON THIS KERNEL, and without it this" >&2
	echo "   round would have nothing to read and every column would be" >&2
	echo "   blank or zero." >&2
	echo "" >&2
	echo "   $PROBE/stats is not readable. Either debugfs is not" >&2
	echo "   mounted (this script already tried" >&2
	echo "   'mount -t debugfs none /sys/kernel/debug') or the running" >&2
	echo "   kernel was not built with" >&2
	echo "   CONFIG_DRM_ACCEL_ROCKET_IOMMU_PROBE=y." >&2
	echo "" >&2
	echo "   ftrace is NOT a fallback here: the board kernel's config" >&2
	echo "   has '# CONFIG_FTRACE is not set' and no CONFIG_TRACING, so" >&2
	echo "   fs/tracefs is not compiled in and /sys/kernel/debug/tracing" >&2
	echo "   does not exist. Getting this number needs the reflash." >&2
	exit 1
fi
[ -w "$PROBE/enabled" ] || { echo "$PROBE/enabled is not writable (run as root)" >&2; exit 1; }

if [ ! -e /dev/accel/accel0 ]; then
	echo "no /dev/accel/accel0: this measures the rocket arm and the" >&2
	echo "   rocket arm is not present. A CPU round would arm the probe," >&2
	echo "   dispatch nothing, and print a floor of zero." >&2
	exit 1
fi

# ONE JOB AT A TIME, AND THE COUNTERS ARE WHY. They are global to the
# driver, so a second charsiu_run does not slow this round down, it ADDS ITS
# JOBS TO THIS ROUND'S COUNT -- a wrong number rather than a slow one.
# matching on comm, not on the command line: a `ps | grep charsiu_run`
# matches its own command line and either locks the round out or, worse, is
# quietly always true.
OTHERS=$(ps -eo pid=,comm= 2>/dev/null | awk -v me="$$" '$1 != me && $2 ~ /^charsiu_run/ { print $1 }')
if [ -n "$OTHERS" ]; then
	echo "charsiu_run is already running (pid $(echo "$OTHERS" | tr '\n' ' ')):" >&2
	echo "   its jobs would land in this round's counters. Wait for it." >&2
	exit 1
fi

ARMED=$(awk '$1 == "armed" { print $2 }' "$PROBE/stats" 2>/dev/null)
if [ "${ARMED:-0}" = 1 ] && [ -z "${CHARSIU_IOMMU_FORCE:-}" ]; then
	echo "the probe is ALREADY ARMED, so another round is either" >&2
	echo "   running or died without disarming. Arming again zeroes its" >&2
	echo "   counters under it. CHARSIU_IOMMU_FORCE=1 takes it anyway." >&2
	exit 1
fi

NPUCLK=$(npu_clk) || exit 1

# ---- conditions, printed, because a reading without them is not comparable ----
echo "== the IOMMU attach/detach floor, measured instead of derived"
echo "   boot id   $(npu_boot)"
echo "   kernel    $(uname -r)"
echo "   npu clk   $NPUCLK Hz"
echo "   probe     $PROBE"
echo "   binary    $RUN"
echo "   build     $(charsiu_build "$RUN")"
echo "   model     $(basename "$M")"
echo "   arms      $N repeats of $NTOK tokens, plus one -n 1 run to split"
echo "             prefill off the decode; one job at a time"
echo

for p in /sys/devices/system/cpu/cpufreq/policy*; do
	echo userspace > "$p/scaling_governor" 2>/dev/null || true
	tr ' ' '\n' < "$p/scaling_available_frequencies" 2>/dev/null | grep -v '^$' \
		| sort -n | tail -1 > "$p/scaling_setspeed" 2>/dev/null || true
done
echo "   cpu       $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq 2>/dev/null)/$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq 2>/dev/null) kHz"
echo

restore() {
	echo 0 > "$PROBE/enabled" 2>/dev/null || true
	echo
	echo "restoring schedutil"
	for p in /sys/devices/system/cpu/cpufreq/policy*; do
		echo schedutil > "$p/scaling_governor" 2>/dev/null || true
	done
}
trap 'restore; exit 130' INT TERM

# A FIELD THAT COMES BACK EMPTY DOES NOT STAY A FIELD. `set -- $R` collapses
# runs of spaces, so one unreadable counter silently shifts every column after
# it one place left and the table reads fine. Every read has a floor of 0 and
# the zero refusal further down is what catches it.
st() {
	_v=$(awk -v k="$1" '$1 == k { print $2 }' "$PROBE/stats" 2>/dev/null)
	case "$_v" in
	"" | *[!0-9]* ) echo 0 ;;
	* ) echo "$_v" ;;
	esac
}

# one armed run. $1 = tokens.  prints 12 fields:
#   a_count a_sum_ns a_min_ns a_max_ns d_count d_sum_ns d_min_ns d_max_ns
#   dom_same dom_switch gen_tok gen_ms
one() {
	echo 1 > "$PROBE/enabled" || return 1
	O=$(env $E "$RUN" "$M" -p "$PROMPT" -n "$1" --ignore-eos -c 1024 -t 4 \
	    2>/dev/null | grep '^\[load' | head -1)
	echo 0 > "$PROBE/enabled" 2>/dev/null || true
	[ -n "$O" ] || return 1
	# and a sed that does not match returns the WHOLE line, which would
	# arrive as a dozen extra fields rather than as a failure
	case "$O" in
	*"gen "*" tok in "*" ms"*) ;;
	*) return 1 ;;
	esac
	G=$(printf '%s\n' "$O" | sed 's/.*gen \([0-9]*\) tok in \([0-9]*\) ms.*/\1 \2/')
	case "$G" in
	[0-9]*" "[0-9]*) ;;
	*) return 1 ;;
	esac
	printf '%s %s %s %s %s %s %s %s %s %s %s\n' \
		"$(st attach_count)" "$(st attach_sum_ns)" "$(st attach_min_ns)" "$(st attach_max_ns)" \
		"$(st detach_count)" "$(st detach_sum_ns)" "$(st detach_min_ns)" "$(st detach_max_ns)" \
		"$(st domain_same)" "$(st domain_switch)" \
		"$G"
}

# one cold run thrown away: the first reading of a round is cold and this one
# also faults in the model
one 8 >/dev/null 2>&1

echo "-- one token, so the prefill jobs can be subtracted off"
P1=$(one 1) || { echo "   NO OUTPUT from the -n 1 run"; restore; exit 1; }
P1_A=$(echo "$P1" | cut -d' ' -f1)
if [ "${P1_A:-0}" -eq 0 ] 2>/dev/null; then
	echo "   the -n 1 run counted 0 attaches. Either the probe is on a"
	echo "      kernel whose rocket never runs, or the NPU arm was not"
	echo "      taken. Zero is not a floor, it is a round with no jobs."
	restore; exit 1
fi
echo "   $P1_A attach calls for the prompt and one token"
echo

echo "-- $N repeats of $NTOK tokens"
printf '   %3s %9s %9s %9s %9s %9s %9s %9s\n' \
	run 'attach n' 'min us' 'mean us' 'max us' 'detach n' 'min us' 'mean us'
printf '   %3s %9s %9s %9s %9s %9s %9s %9s\n' \
	--- --------- --------- --------- --------- --------- --------- ---------
JPT=""; MSPT=""; AMIN=""; AMEAN=""; DMIN=""; DMEAN=""; SAME=""; SW=""
i=0
while [ "$i" -lt "$N" ]; do
	i=$((i + 1))
	R=$(one "$NTOK") || { echo "   $i: NO OUTPUT"; continue; }
	set -- $R
	[ $# -eq 12 ] || { echo "   $i: $# fields, expected 12 -- not a table row"; continue; }
	ac=$1; asum=$2; amin=$3; amax=$4; dc=$5; dsum=$6; dmin=$7
	dsame=$9; dsw=${10}; gtok=${11}; gms=${12}

	# THE ZERO. A stats file that exists but never counted divides to a
	# clean answer, and it is the shape a missing mount has every time.
	if [ "${ac:-0}" -eq 0 ] 2>/dev/null || [ -z "${ac:-}" ]; then
		echo "   run $i: attach_count is 0. The probe is armed and the"
		echo "      driver never reached the attach -- this is not a floor"
		echo "      of zero, it is a round that did not measure anything."
		restore; exit 1
	fi

	amean=$(awk "BEGIN{printf \"%.1f\", $asum/$ac/1000}")
	dmean=$(awk "BEGIN{printf \"%.1f\", ${dsum:-0}/${dc:-1}/1000}")
	printf '   %3s %9s %9s %9s %9s %9s %9s %9s\n' "$i" "$ac" \
		"$(awk "BEGIN{printf \"%.1f\", $amin/1000}")" "$amean" \
		"$(awk "BEGIN{printf \"%.1f\", $amax/1000}")" \
		"${dc:-?}" "$(awk "BEGIN{printf \"%.1f\", ${dmin:-0}/1000}")" "$dmean"

	AMIN="$AMIN $(awk "BEGIN{printf \"%.1f\", $amin/1000}")"
	AMEAN="$AMEAN $amean"
	DMIN="$DMIN $(awk "BEGIN{printf \"%.1f\", ${dmin:-0}/1000}")"
	DMEAN="$DMEAN $dmean"
	# decode jobs only: this run's attaches minus the prompt-and-one-token run
	JPT="$JPT $(awk "BEGIN{printf \"%.1f\", ($ac-$P1_A)/($gtok-1)}")"
	MSPT="$MSPT $(awk "BEGIN{printf \"%.2f\", $gms/$gtok}")"
	SAME="$SAME $(awk "BEGIN{t=${dsame:-0}+${dsw:-0}; printf \"%.1f\", t?100*${dsame:-0}/t:0}")"
	SW="$SW ${dsw:-0}"

	# a detach missing against an attach is a job still in flight when the
	# file was read, not a free detach. It is a tell, not a result.
	[ "${dc:-0}" -eq "$ac" ] 2>/dev/null || \
		echo "      (detach $dc against attach $ac -- jobs in flight at the read)"
done

mid() { printf '%s\n' $1 | sort -g | awk '{a[NR]=$0} END{ if (NR==0) exit;
	if (NR%2) printf "%s\n", a[(NR+1)/2]; else printf "%.2f\n", (a[NR/2]+a[NR/2+1])/2 }'; }
lo() { printf '%s\n' $1 | sort -g | head -1; }
hi() { printf '%s\n' $1 | sort -g | tail -1; }

echo
echo "== THE THREE COLUMNS, AND THEY ARE NOT MULTIPLIED TOGETHER"
printf '   %-34s %10s   %s\n' "attach, floor (min over all jobs)" "$(lo "$AMIN") us" "range $(lo "$AMIN")..$(hi "$AMIN")"
printf '   %-34s %10s   %s\n' "attach, mean a job"                "$(mid "$AMEAN") us" "range $(lo "$AMEAN")..$(hi "$AMEAN")"
printf '   %-34s %10s   %s\n' "detach, floor (min over all jobs)" "$(lo "$DMIN") us" "range $(lo "$DMIN")..$(hi "$DMIN")"
printf '   %-34s %10s   %s\n' "detach, mean a job"                "$(mid "$DMEAN") us" "range $(lo "$DMEAN")..$(hi "$DMEAN")"
printf '   %-34s %10s   %s\n' "jobs a token, BOTH CORES SUMMED"   "$(mid "$JPT")" "range $(lo "$JPT")..$(hi "$JPT")"
printf '   %-34s %10s   %s\n' "token, end to end"                 "$(mid "$MSPT") ms" "range $(lo "$MSPT")..$(hi "$MSPT")"
echo
echo "== AND WHETHER THE STRONG A/B HAS ANYTHING TO FIND, counted not guessed"
printf '   %-34s %10s   %s\n' "jobs whose domain was unchanged"   "$(mid "$SAME") %" "range $(lo "$SAME")..$(hi "$SAME")"
printf '   %-34s %10s   %s\n' "domain switches a run"             "$(mid "$SW")" "range $(lo "$SW")..$(hi "$SW")"
echo
echo "   An IOMMU domain is per open DRM FILE, and rocket_job_open() gives"
echo "   every file one sched entity spanning EVERY core -- so drm_sched can"
echo "   put either of charsiu's two open files on either core. Every switch"
echo "   above is an attach that keeping the domain attached across jobs"
echo "   would STILL have had to do. The first row is the whole of what that"
echo "   change can remove: near 100% and the A/B is worth two boots, near"
echo "   0% and it is worth none, and no reflash was needed to tell."
echo
echo "   The product of those columns is NOT printed and must not be"
echo "      written down. It is the 19.5 ms all over again: the cores"
echo "      attach concurrently, so N jobs do not cost N times one job,"
echo "      and time inside the call is not time the token loses. The"
echo "      floor above is a lower bound on the CALL. Only the attach-once"
echo "      A/B (rfc-send-v12/attach-once/) bounds the TOKEN, and it"
echo "      measures tok/s directly with no arithmetic in between."
echo
echo "   and the count is summed over both cores. The sentence this"
echo "     round retires said '150 jobs a core'; halve the column above"
echo "     before comparing it to that, or do not compare it at all."

restore
