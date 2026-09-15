#!/bin/sh
#
# Every binary the installer ships answers --version with its commit.
#
# Round 413 added the build stamp and recorded that all eighteen probe tools
# answered. Round 418 asked them: eleven did. The other seven ran instead --
# charsiu_matmul started a matmul at K=64 N=32, out16_bound tried to open
# "--version" as a gguf, npu_qpack_test compared 56000 rows. A binary that
# cannot say its commit cannot be an arm, and round 415 lost a comparison to
# exactly that: an installed scalar control that predated the stamp.
#
# This is the check that keeps the claim honest. It runs on the desk against
# build/, and on a board against /opt/charsiu, so a deployment can be asked
# the same question.
#
#   tests/version_all.sh [DIR]
set -u
DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/build}"
[ -d "$DIR" ] || { echo "version_all.sh: $DIR is not a directory" >&2; exit 2; }

BINS="charsiu_run charsiu_ppl charsiu_check charsiu_probe charsiu_serve
charsiu_bench charsiu_vision charsiu_clip charsiu_whisper charsiu_wide
charsiu_int4 bench_batch npu_gemm_test npu_slice_test npu_fp16_test
npu_fence_scan charsiu_matmul vattn_bench acc_index_check fp16_plan
charsiu_membw npu_qpack_test npu_prep_cost npu_job_cost charsiu_shapes
npu_out_fmt npu_mixed_test out16_bound bench_gather charsiu_run_scalar
tokenizer_roundtrip emit_dump"

TMP="${TMPDIR:-/tmp}/version_all.$$"
WAIT="${CHARSIU_VERSION_WAIT:-5}"
# THE POLL IS A SECOND AND THE ANSWER TAKES FIVE MILLISECONDS, so every
# binary that passes still costs a whole tick and the desk run took 28 s for
# 28 processes that had all exited. A fractional sleep is not POSIX, so ask
# this shell for one rather than assume it: the board's busybox may refuse,
# and then this is exactly the loop it was. The BOUND stays in seconds either
# way -- ticks per second is the only thing that moves.
if sleep 0.05 2>/dev/null; then
	TICK="0.05"; HZ=20
else
	TICK="1"; HZ=1
fi
trap 'rm -f "$TMP"' EXIT INT TERM

# A TRAIL THAT SURVIVES THE MACHINE, because twice now the board has gone
# silent during this script and neither time could say which binary it was on.
# Everything this prints goes to a pipe that is lost when the far end dies, and
# the loop only prints AFTER a binary returns -- so the one that does not
# return is exactly the one with no line. The trail is written and synced
# BEFORE each child starts, so what is on the disk after a power cycle is the
# name of the binary that was running.
#
# sync per binary is 32 syncs. This script measures nothing timed.
TRAIL="${CHARSIU_VERSION_TRAIL:-$DIR/version_all.last}"
: > "$TRAIL" 2>/dev/null || TRAIL=/dev/null

ok=0; bad=0; missing=0
for b in $BINS; do
	if [ ! -x "$DIR/$b" ]; then
		missing=$((missing + 1))
		continue
	fi
	printf 'starting %s\n' "$b" >> "$TRAIL" 2>/dev/null
	sync 2>/dev/null || true
	# A TIMEOUT, because the failure mode IS running: a tool that does not
	# recognise --version starts its real work, and several of these take
	# minutes or drive the NPU. Without one the check hangs on exactly the
	# binaries it exists to find.
	#
	# THE FIRST VERSION OF THIS SAID ALL OF THAT IN A COMMENT AND DID NOT
	# DO IT. It ran the binary straight, on a board that had just received
	# sixteen probe tools which had never executed on that hardware, and
	# the board went silent. A comment describing a check that is not there
	# is the oldest fault in this tree and I wrote a fresh one.
	#
	# `timeout` is not on the board's busybox, so this is a background
	# child and a bounded wait. VTIME cannot help here: the child is a
	# process, not a tty.
	rm -f "$TMP"
	"$DIR/$b" --version > "$TMP" 2>&1 < /dev/null &
	pid=$!
	i=0
	while kill -0 "$pid" 2>/dev/null; do
		i=$((i + 1))
		if [ "$i" -gt $((WAIT * HZ)) ]; then
			kill "$pid" 2>/dev/null
			sleep 1
			kill -9 "$pid" 2>/dev/null
			wait "$pid" 2>/dev/null
			echo "BAD   $b: --version did not return within ${WAIT}s -- it is RUNNING"
			bad=$((bad + 1))
			pid=""
			break
		fi
		sleep "$TICK"
	done
	[ -n "$pid" ] || continue
	wait "$pid" 2>/dev/null
	v=$(head -1 "$TMP" 2>/dev/null)
	case "$v" in
	*[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*)
		case "$v" in
		*" "*) echo "BAD   $b: --version printed a sentence, not a commit: $v"
		       bad=$((bad + 1)) ;;
		*)     ok=$((ok + 1)) ;;
		esac ;;
	*)  echo "BAD   $b: --version did not answer with a commit: $(echo "$v" | cut -c1-60)"
	    bad=$((bad + 1)) ;;
	esac
done
# A FINISHED RUN SAYS SO IN THE TRAIL, or a stale "starting X" from a clean
# run reads exactly like the binary that killed the machine.
printf 'finished cleanly\n' >> "$TRAIL" 2>/dev/null
sync 2>/dev/null || true
echo "$ok answered, $bad did not, $missing not present in $DIR"
exit "$bad"
