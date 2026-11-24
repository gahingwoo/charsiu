#!/bin/bash
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# HOW TO RUN IT (on the board, as the user who owns the models):
#
#   1. Copy each candidate binary to the board as ~/charsiu_run_NAME
#      (and, for --ppl, the charsiu_ppl built at the same commit as
#      ~/charsiu_ppl_NAME: `make build/charsiu_ppl.aarch64` on the desk).
#   2. Binary A/B:      bash board_arms.sh --a old --b new
#   3. Switch off/on:   bash board_arms.sh --a new --b new:CHARSIU_SPEC=4
#   4. Quality A/B:     bash board_arms.sh --ppl --a new --b new:CHARSIU_X=1
#   5. Pick models/prompts: --models '*Qwen3-0.6B*,*Llama-3.2-1B*' --prompts chat,copy,long
#   6. Every run's stdout/stderr and the report land in ~/charsiu-board/arms-*/
#   7. Nothing else may use the NPU while it runs; it refuses if it sees one.
#   8. Read the verdict lines last, and the FAILED / UNSTABLE lines first.
#   9. --ppl also needs the corpus: copy tests/corpus/long.txt to ~/corpus/
#      (the apt package ships neither it nor charsiu_ppl), or --corpus FILE.
#  10. Not tests/board_ab.sh: that one A/Bs one env var on --batch-probe.
#
# ---------------------------------------------------------------------------
#
# Two arms, one boot, interleaved: is B faster than A, by more than either
# arm's own run-to-run spread, and does it say the same thing?
#
# usage: board_arms.sh --a NAME_A[:ENV=V,...] --b NAME_B[:ENV=V,...]
#                      [--models GLOB,...] [--prompts SET[,SET...]]
#                      [--reps N] [-n N] [-c N] [--timeout S]
#                      [--model-dir DIR] [--out DIR] [--ignore-eos]
#        board_arms.sh --ppl --a ... --b ... [--models GLOB,...]
#                      [--corpus FILE] [--ppl-n N] [--ppl-batch]
#                      [--ppl-a PATH] [--ppl-b PATH]
#
#   NAME       ~/charsiu_run_NAME; "installed" is the apt package's
#              charsiu_run; anything with a "/" is a path.
#   :ENV=V,..  set for that arm only, after the runner's environment, so
#              "new" against "new:CHARSIU_X=1" is the same binary with the
#              switch off and on. A value may contain commas (CPU lists): a
#              comma starts a new pair only when NAME= follows it.
#   SET        chat (4 prompts), copy (3, copy heavy), long (~400 and ~1400
#              tokens, at -c 2048), all.  Default: chat.
#   --reps     measured runs per arm per prompt (default 3; under 3 no
#              difference is ever called real, there is no spread to beat).
#   -n / -c    tokens to generate (default 128) / context (default 1024;
#              the long set uses at least 2048).
#
# THE RULES, EACH ONE PAID FOR (see the feedback notes in this tree):
#
#   - Only the generated text is hashed: stdout with every line that starts
#     with "[" removed. stderr carries the MHz, the counters and the timings,
#     and hashing it makes every pair of runs differ.
#   - Arms are compared inside ONE boot, INTERLEAVED A B A B. This board has
#     drifted ~25% between minutes and ~3% between boots; two blocks of runs
#     taken at different times measure the drift, not the change.
#   - The first run after a model is loaded is cold (file cache, staging):
#     each arm gets one discarded warm-up per model.
#   - A run that timed out, crashed, or printed no summary line is FAILED with
#     its reason. It never contributes a number, and a prompt with no completed
#     run for an arm prints FAILED where its median would be. A missing thing
#     must not read as a zero.
#   - Strictly sequential. Two processes on the NPU at once made a correctness
#     check report three models wrong that were not; this refuses to start, and
#     stops, if another charsiu process is running.
#   - A difference is "real" only if it exceeds BOTH arms' own min-max spread.
#     The median alone has picked the fast mode of a bimodal decode before.
#
# The user is not root on the CM5, so the governor is not touched and the NPU
# clock is not read from debugfs: the header prints what the runtime itself
# says on stderr (it reads the rate from the device tree) and the governors.
set -u

usage() { sed -n '/^# usage:/,/^#   -n \/ -c/p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

ARM_A=""; ARM_B=""; MODELS="*Qwen3-0.6B*Q4*,*Llama-3.2-1B*Q4*,*Qwen3-1.7B*Q4*"
SETS="chat"; REPS=3; NGEN=128; NCTX=1024; TMO=600
MDIR="$HOME/.charsiu/models"; OUT=""; IGNORE_EOS=""
PPL=0; CORPUS=""; PPLN=300; PPLBATCH=""; PPL_A=""; PPL_B=""
while [ $# -gt 0 ]; do
	case "$1" in
	--a) ARM_A=$2; shift 2 ;;
	--b) ARM_B=$2; shift 2 ;;
	--models) MODELS=$2; shift 2 ;;
	--prompts) SETS=$2; shift 2 ;;
	--reps) REPS=$2; shift 2 ;;
	-n) NGEN=$2; shift 2 ;;
	-c) NCTX=$2; shift 2 ;;
	--timeout) TMO=$2; shift 2 ;;
	--model-dir) MDIR=$2; shift 2 ;;
	--out) OUT=$2; shift 2 ;;
	--ignore-eos) IGNORE_EOS="--ignore-eos"; shift ;;
	--ppl) PPL=1; shift ;;
	--corpus) CORPUS=$2; shift 2 ;;
	--ppl-n) PPLN=$2; shift 2 ;;
	--ppl-batch) PPLBATCH="--batch"; shift ;;
	--ppl-a) PPL_A=$2; shift 2 ;;
	--ppl-b) PPL_B=$2; shift 2 ;;
	-h|--help) usage ;;
	*) echo "unknown argument: $1" >&2; usage ;;
	esac
done
[ -n "$ARM_A" ] && [ -n "$ARM_B" ] || { echo "both --a and --b are required" >&2; usage; }
for v in REPS NGEN NCTX TMO PPLN; do
	case "${!v}" in ''|*[!0-9]*) echo "$v must be a number, got '${!v}'" >&2; exit 2 ;; esac
done
[ "$SETS" = all ] && SETS="chat,copy,long"

# THE RUNNER'S ENVIRONMENT, the one charsiu-runner gives the product. An arm's
# own pairs come after these, so an arm can override any of them.
BASE_ENV=(CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 CHARSIU_CPUS=4-7)
THREADS=4

die() { echo "" >&2; echo "board_arms: $*" >&2; exit 1; }

# ---------------------------------------------------------------- the arms
# NAME[:K=V,K=V]. Split the pairs on a comma only where a NAME= follows, so
# CHARSIU_CPUS=4,5,6,7 stays one pair.
arm_name() { case "$1" in *:*) echo "${1%%:*}" ;; *) echo "$1" ;; esac; }
arm_pairs() {
	case "$1" in *:*) ;; *) return 0 ;; esac
	printf '%s\n' "${1#*:}" | sed 's/,\([A-Za-z_][A-Za-z0-9_]*=\)/\n\1/g' | grep -v '^$'
}
resolve_run() {
	case "$1" in
	""|installed)
		for c in "$(command -v charsiu_run 2>/dev/null)" /opt/charsiu/charsiu_run /usr/bin/charsiu_run; do
			[ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
		done
		return 1 ;;
	*/*) [ -x "$1" ] && { echo "$1"; return 0; }; return 1 ;;
	*)   [ -x "$HOME/charsiu_run_$1" ] && { echo "$HOME/charsiu_run_$1"; return 0; }; return 1 ;;
	esac
}
for s in "$ARM_A" "$ARM_B"; do
	case "$s" in
	*:*) ;;
	*=*) die "arm '$s' has a '=' and no ':'. A switch on the installed binary is
   'installed:$s'; on a candidate it is 'NAME:$s'." ;;
	esac
	while IFS= read -r p; do
		case "$p" in [A-Za-z_]*=*) ;; *) die "arm '$s': '$p' is not ENV=VALUE" ;; esac
	done < <(arm_pairs "$s")
done
NAME_A=$(arm_name "$ARM_A"); NAME_B=$(arm_name "$ARM_B")
mapfile -t ENV_A < <(arm_pairs "$ARM_A")
mapfile -t ENV_B < <(arm_pairs "$ARM_B")
RUN_A=$(resolve_run "$NAME_A") || die "arm A: no executable for '$NAME_A' (looked for ~/charsiu_run_$NAME_A, or the path given, or the installed charsiu_run for 'installed')"
RUN_B=$(resolve_run "$NAME_B") || die "arm B: no executable for '$NAME_B' (looked for ~/charsiu_run_$NAME_B, or the path given, or the installed charsiu_run for 'installed')"

command -v timeout >/dev/null 2>&1 || die "'timeout' (coreutils) is not installed, and without it a hung run would hang the round"
command -v md5sum >/dev/null 2>&1 || die "md5sum is not installed"

# AND THIS MACHINE MUST HAVE THE NPU, or every arm runs the CPU fallback and
# agrees with itself. ANY accel node: a rebind of rocket takes the next minor.
if [ -z "$(ls /dev/accel/accel* 2>/dev/null)" ] && [ -z "${CHARSIU_ALLOW_NO_NPU:-}" ]; then
	die "NO /dev/accel/accel* -- this machine cannot answer the question.
   Without the NPU both arms run the CPU path and read as a tie.
   CHARSIU_ALLOW_NO_NPU=1 runs it anyway, to test the script itself."
fi

# ---------------------------------------------------- one job on the NPU
other_jobs() {
	# BY PROCESS NAME, NOT BY COMMAND LINE: a compile line or a shell that
	# merely mentions charsiu_run is not on the NPU. comm is the executable's
	# own name (15 characters), so ~/charsiu_run_x shows as charsiu_run_x.
	ps -eo pid=,comm=,args= 2>/dev/null | awk -v me="$$" '
		$1 != me && $2 ~ /^(charsiu_(run|serve|ppl|vision|whisper|clip)|rkllm)/ {
			print substr($0, 1, 160) }'
}
J=$(other_jobs)
[ -z "$J" ] || die "another NPU job is running, and two on the NPU at once
   corrupt each other's answers:
$(printf '%s\n' "$J" | sed 's/^/     /')
   Stop it (charsiu serve runs as a service: 'charsiu ps') and start again."
ps -eo pid= >/dev/null 2>&1 || PGREP_NOTE="ps -eo does not work here: this round CANNOT see a second NPU job"

# THE NPU DOES NOT COME BACK FROM A TIMEOUT, and after it wedges every run
# falls back to something that still answers. Count the kernel's complaint
# before and after each run; a user who cannot read dmesg is told so.
if dmesg >/dev/null 2>&1; then
	DMESG_OK=1; DMESG0=$(dmesg 2>/dev/null | grep -c "NPU job timed out")
else
	DMESG_OK=0; DMESG0=0
fi
wedged() {
	[ "$DMESG_OK" = 1 ] || return 1
	n=$(dmesg 2>/dev/null | grep -c "NPU job timed out")
	[ "$n" = "$DMESG0" ] && return 1
	return 0
}

# -------------------------------------------------------------- output
if [ -z "$OUT" ]; then
	OUT="$HOME/charsiu-board/arms-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUT" || die "cannot create $OUT"
exec > >(tee -a "$OUT/report.txt") 2>&1
TEE_PID=$!
# let tee drain before the script is gone, or the last lines of the report
# (the verdict) can be missing from the terminal
trap 'exec >&- 2>&-; wait "$TEE_PID" 2>/dev/null' EXIT

BOOT=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)
gov() { cat "/sys/devices/system/cpu/cpufreq/$1/scaling_governor" 2>/dev/null || echo "?"; }
cur() { cat "/sys/devices/system/cpu/cpufreq/$1/scaling_cur_freq" 2>/dev/null || echo "?"; }
binfo() {
	v=$("$1" --version 2>/dev/null | head -1)
	case "$v" in ""|*[!0-9a-zA-Z.+~-]*) v="no --version (predates the stamp)" ;; esac
	printf '%s  [%s, md5 %s]' "$1" "$v" "$(md5sum "$1" | cut -c1-12)"
}
envstr() { local e; for e in "$@"; do printf '%s ' "$e"; done; }

echo "board_arms  $(date '+%Y-%m-%d %H:%M:%S')  $(uname -n)  $(uname -r)  user $(id -un)"
echo "  boot      $BOOT"
echo "  governor  policy0 $(gov policy0), policy4 $(gov policy4)  (not changed: not root)"
echo "  cpu now   $(cur policy0) / $(cur policy4) kHz"
echo "  arm A     $(binfo "$RUN_A")"
echo "            env: $(envstr "${ENV_A[@]}")"
echo "  arm B     $(binfo "$RUN_B")"
echo "            env: $(envstr "${ENV_B[@]}")"
echo "  runner    $(envstr "${BASE_ENV[@]}")-t $THREADS --temp 0"
INH=$(env | grep '^CHARSIU_' | sort)
[ -z "$INH" ] || { echo "  INHERITED by BOTH arms from this shell:"; printf '%s\n' "$INH" | sed 's/^/            /'; }
if [ "$RUN_A" = "$RUN_B" ] && [ "$(envstr "${ENV_A[@]}")" = "$(envstr "${ENV_B[@]}")" ]; then
	echo "  NOTE      A and B are the SAME arm: this is a null control, it measures"
	echo "            the spread and nothing else."
fi
[ "$DMESG_OK" = 1 ] || echo "  WARNING   dmesg is not readable as $(id -un): an NPU timeout will NOT be noticed by this round"
[ -z "${PGREP_NOTE:-}" ] || echo "  WARNING   $PGREP_NOTE"
echo "  output    $OUT"
echo

# ---------------------------------------------------------------- models
# A glob per model, matched in MDIR. A file still being written is skipped and
# SAID to be skipped: a half-downloaded gguf loads its header fine.
SKIPPED=()
MODEL_FILES=()
IFS=',' read -r -a GLOBS <<< "$MODELS"
for g in "${GLOBS[@]}"; do
	shopt -s nullglob
	m=( "$MDIR"/$g )
	shopt -u nullglob
	ok=()
	for f in "${m[@]}"; do
		case "$f" in *.gguf) [ -f "$f" ] && ok+=("$f") ;; esac
	done
	if [ "${#ok[@]}" -eq 0 ]; then
		SKIPPED+=("$g: no *.gguf in $MDIR matches")
		continue
	fi
	f=${ok[0]}
	if [ "${#ok[@]}" -gt 1 ]; then
		echo "  glob $g matched ${#ok[@]} files; running $(basename "$f") only:"
		for x in "${ok[@]:1}"; do echo "      not run: $(basename "$x")"; done
	fi
	if ls "$f".part "$f".tmp "$f".download "$f".aria2 >/dev/null 2>&1; then
		SKIPPED+=("$(basename "$f"): a .part/.tmp next to it, still downloading")
		continue
	fi
	s1=$(stat -c %s "$f"); sleep 1; s2=$(stat -c %s "$f")
	if [ "$s1" != "$s2" ]; then
		SKIPPED+=("$(basename "$f"): size changing ($s1 -> $s2 bytes), still downloading")
		continue
	fi
	case "$(head -c 4 "$f")" in GGUF) ;; *)
		SKIPPED+=("$(basename "$f"): does not start with GGUF"); continue ;; esac
	dup=0; for x in "${MODEL_FILES[@]}"; do [ "$x" = "$f" ] && dup=1; done
	[ "$dup" = 1 ] || MODEL_FILES+=("$f")
done
for s in "${SKIPPED[@]}"; do echo "  SKIPPED   $s"; done
[ "${#MODEL_FILES[@]}" -gt 0 ] || die "no model to run (see SKIPPED above). Nothing was measured, and that is not a result."
for f in "${MODEL_FILES[@]}"; do
	echo "  model     $(basename "$f")  $(stat -c %s "$f") bytes"
done
echo

# ===================================================================== ppl
if [ "$PPL" = 1 ]; then
	resolve_ppl() { # arm-name run-path override
		[ -n "$3" ] && { [ -x "$3" ] && echo "$3"; return; }
		case "$1" in
		""|installed)
			for c in "$(command -v charsiu_ppl 2>/dev/null)" /opt/charsiu/charsiu_ppl; do
				[ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return; }
			done ;;
		*/*) c="$(dirname "$2")/$(basename "$2" | sed 's/charsiu_run/charsiu_ppl/')"
		     [ "$c" != "$2" ] && [ -x "$c" ] && echo "$c" ;;
		*)   [ -x "$HOME/charsiu_ppl_$1" ] && echo "$HOME/charsiu_ppl_$1" ;;
		esac
	}
	missing_ppl() {
		echo "MISSING: charsiu_ppl for arm $1 ('$2')."
		case "$2" in
		""|installed) echo "   looked for charsiu_ppl on PATH and /opt/charsiu/charsiu_ppl." ;;
		*/*) echo "   looked for $(dirname "$3")/$(basename "$3" | sed 's/charsiu_run/charsiu_ppl/')." ;;
		*)   echo "   looked for ~/charsiu_ppl_$2." ;;
		esac
		echo "   The apt package (stable channel) does not ship charsiu_ppl; it is a"
		echo "   dev-channel probe. charsiu_run has no perplexity mode of its own."
		echo "   Build it at the SAME commit as that arm's charsiu_run:"
		echo "       make build/charsiu_ppl.aarch64"
		case "$2" in
		*/*) echo "   put it there (or pass --ppl-$(echo "$1" | tr AB ab) PATH)." ;;
		*)   echo "   copy it to the board as ~/charsiu_ppl_$2 (or pass --ppl-$(echo "$1" | tr AB ab) PATH)." ;;
		esac
	}
	P_A=$(resolve_ppl "$NAME_A" "$RUN_A" "$PPL_A")
	P_B=$(resolve_ppl "$NAME_B" "$RUN_B" "$PPL_B")
	miss=0
	[ -n "$P_A" ] || { missing_ppl A "$NAME_A" "$RUN_A"; miss=1; }
	[ -n "$P_B" ] || { missing_ppl B "$NAME_B" "$RUN_B"; miss=1; }
	# THE CORPUS IS PART OF THE NUMBER. The tree's recorded figures are on
	# tests/corpus/long.txt at 300 tokens, md5 below; anything else is a
	# different number, not a worse or better one.
	LONG_MD5=4237c8fc3163a359fc21bde60c7b1d8b
	if [ -z "$CORPUS" ]; then
		for c in "$(dirname "$0")/corpus/long.txt" /opt/charsiu/corpus/long.txt \
			 "$HOME/corpus/long.txt" "$HOME/long.txt"; do
			[ -r "$c" ] && { CORPUS=$c; break; }
		done
	fi
	if [ -z "$CORPUS" ] || [ ! -r "$CORPUS" ]; then
		echo "MISSING: the corpus. Looked for long.txt in $(dirname "$0")/corpus,"
		echo "   /opt/charsiu/corpus, ~/corpus and ~. The apt package does not ship"
		echo "   tests/corpus; copy tests/corpus/long.txt (md5 $LONG_MD5) to"
		echo "   ~/corpus/long.txt or pass --corpus FILE."
		miss=1
	fi
	[ "$miss" = 0 ] || die "--ppl cannot run: see MISSING above. No number was invented."
	CMD5=$(md5sum "$CORPUS" | cut -d' ' -f1)
	PCTX=$(( PPLN > 512 ? PPLN : 512 ))
	echo "perplexity, both arms, same model file, same corpus, same length"
	echo "  ppl A     $(binfo "$P_A")"
	echo "  ppl B     $(binfo "$P_B")"
	echo "  corpus    $CORPUS  md5 $CMD5"
	if [ "$CMD5" = "$LONG_MD5" ]; then
		echo "            (the tree's long.txt, the corpus every recorded ppl is on)"
	else
		echo "            (NOT the tree's long.txt, $LONG_MD5: not comparable with recorded figures)"
	fi
	echo "  length    -n $PPLN, -c $PCTX, $([ -n "$PPLBATCH" ] && echo "--batch (the batched prefill arithmetic)" || echo "token loop (llama_forward, one position at a time)")"
	echo "  order     A, B, A again (the repeat checks the instrument is deterministic here)"
	echo

	ppl_run() { # arm-letter bin outfile model ; prints "ppl positions tokens" or FAILED reason
		local L=$1 B=$2 F=$3 M=$4 rc env_arm
		if [ "$L" = A ]; then env_arm=("${ENV_A[@]}"); else env_arm=("${ENV_B[@]}"); fi
		J=$(other_jobs); [ -z "$J" ] || { echo "FAILED another NPU job appeared: $J"; return; }
		( timeout -k 10 "$TMO" env "${BASE_ENV[@]}" CHARSIU_THREADS=$THREADS "${env_arm[@]}" \
			"$B" "$M" "$CORPUS" -n "$PPLN" -c "$PCTX" $PPLBATCH >"$F.out" 2>"$F.err" ) 2>/dev/null
		rc=$?
		if wedged; then echo "FAILED THE NPU WEDGED (dmesg: NPU job timed out)"; return; fi
		[ "$rc" = 124 ] && { echo "FAILED timed out after $TMO s"; return; }
		[ "$rc" -gt 128 ] && { echo "FAILED killed by signal $((rc - 128))"; return; }
		[ "$rc" = 0 ] || { echo "FAILED exit $rc: $(tail -1 "$F.err" | tr -d '\r' | cut -c1-120)"; return; }
		sed -n 's/^ppl \([0-9.]*\)  over \([0-9]*\) scored positions of \([0-9]*\) tokens.*/\1 \2 \3/p' "$F.out" | head -1 | grep . \
			|| echo "FAILED no 'ppl ... over N scored positions' line on stdout"
	}
	VERD=()
	for M in "${MODEL_FILES[@]}"; do
		b=$(basename "$M"); d="$OUT/ppl-${b%.gguf}"; mkdir -p "$d"
		echo "== $b"
		echo "  model md5 $(md5sum "$M" | cut -d' ' -f1)  $(stat -c %s "$M") bytes"
		rA=$(ppl_run A "$P_A" "$d/A1" "$M"); echo "  A        $rA"
		rB=$(ppl_run B "$P_B" "$d/B1" "$M"); echo "  B        $rB"
		rA2=$(ppl_run A "$P_A" "$d/A2" "$M"); echo "  A again  $rA2"
		case "$rA$rB" in *FAILED*)
			VERD+=("$b: ppl FAILED (see above); no comparison"); echo; continue ;; esac
		set -- $rA; pa=$1; sa=$2; ta=$3
		set -- $rB; pb=$1; sb=$2
		pa2=$(printf '%s' "$rA2" | awk '{print $1}')
		if [ "$sa" != "$sb" ]; then
			VERD+=("$b: NOT COMPARABLE, A scored $sa positions and B $sb")
			echo; continue
		fi
		det="deterministic"
		case "$rA2" in *FAILED*) det="repeat FAILED, determinism unchecked" ;; *)
			[ "$pa" = "$pa2" ] || det="NOT DETERMINISTIC: A gave $pa then $pa2, so a gap smaller than that means nothing" ;; esac
		rel=$(awk -v a="$pa" -v b="$pb" 'BEGIN { printf "%+.3f%%", (b - a) / a * 100 }')
		echo "  B vs A   $rel  ($det)"
		if [ "$pa" = "$pb" ]; then
			v="identical to the last digit. A null arm is not a null result:"
			v="$v either the switch leaves the numerics alone, or charsiu_ppl never"
			v="$v reads it (a switch only tools/charsiu_run.c reads, e.g. the"
			v="$v speculative or sampling loop, is invisible to this instrument)"
		else
			v="$(awk -v a="$pa" -v b="$pb" 'BEGIN { print (b < a) ? "B lower (better)" : "B higher (worse)" }') by $rel"
		fi
		VERD+=("$b: ppl A $pa, B $pb on $(basename "$CORPUS") (md5 ${CMD5:0:12}), -n $PPLN ($sa positions of $ta tokens): $v")
		echo
	done
	echo "== verdict"
	for s in "${SKIPPED[@]}"; do echo "  SKIPPED $s"; done
	for v in "${VERD[@]}"; do echo "  $v"; done
	exit 0
fi

# ================================================================ prompts
P_ID=(); P_TEXT=(); P_CTX=()
add() { P_ID+=("$1"); P_CTX+=("$2"); P_TEXT+=("$3"); }

PASSAGE_COPY=$(cat <<'EOF'
The old bridge over the river was built of grey stone in five low arches, and for a long time nobody thought much about it. Carts crossed it in the morning on their way to the market, and in the evening the same carts came back lighter, with the drivers half asleep and the horses finding the way themselves. In the spring the water rose until it nearly touched the keystones, and children stood on the parapet to watch branches and whole trees go past underneath. One year a crack appeared in the middle arch. The town council met three times to argue about it. Some wanted to pull the bridge down and build a new one of iron; others said the stone had stood for two hundred years and would stand for two hundred more if it were left alone. In the end a mason from the next valley was called. He spent a week looking at the crack, tapping the stones with a small hammer and listening, and then he filled it with lime mortar and went home. The bridge is still there.
EOF
)
CODE_COPY=$(cat <<'EOF'
static int sum_positive(const int *v, int n)
{
	int total = 0;
	for (int i = 0; i < n; i++) {
		if (v[i] > 0)
			total += v[i];
	}
	return total;
}
EOF
)
LONG1=$(cat <<'EOF'
The canal reached the town in the spring of a dry year, and the first boat to come up the new locks carried nothing more remarkable than coal and a crate of oranges. People walked out along the towpath to see it anyway. The lock keeper, a careful man who had been a carpenter before the canal company hired him, opened the gates slowly so that the water would not rush, and the boat rose in the chamber like something being lifted on a hand. When it came level with the upper pound the horse leaned into its collar and the rope tightened, and the boat slid out of the lock with hardly a sound. Children ran beside it for half a mile. An old woman who had been born before the canal was dug said that she had never expected to see a boat climb a hill, and that she was not sure she liked it.

Within ten years the canal had changed the shape of the town. Warehouses went up along the basin, built of red brick with wide doors on the water side and narrow ones on the street. A boatyard opened where the old tannery had been, and the smell of tar replaced the smell of hides. The price of coal fell by a third, and families who had burned peat and wood all their lives began to burn coal in iron grates. The miller on the river, who had always set the price of flour, found that flour could now come in by boat from mills sixty miles away, and he lowered his price and grumbled about it for the rest of his life. His son, who was more practical, sold the mill to the canal company, and the old wheel was taken out and the building turned into a store for grain that came and went by water and was never ground there at all.
EOF
)
LONG2=$(cat <<'EOF'
The boats themselves were long and narrow, seventy feet from stem to stern and barely seven feet across, because the locks had been built to that size and no wider. A family could live on one. The cabin at the back was small enough to cross in two steps, but it held a stove, a bed that folded into the wall, a cupboard for plates painted with roses and castles, and a brass lamp that was polished every Sunday. The men steered and worked the locks; the women steered too, and cooked, and kept accounts, and often knew the cargo rates better than the men did. Children learned to lead the horse before they learned to read, and many of them never learned to read at all, because the boat was never in one place long enough for school.

The lock keeper kept a ledger, and the ledger survives. Every boat that passed is in it: its name, its owner, its cargo, the hour it entered the lock and the hour it left, and the toll it paid. In the first year there were nine hundred entries. In the busiest year, twenty years later, there were more than six thousand, and the keeper had taken on two assistants and a boy to run messages to the company office. The cargoes tell their own story. Coal comes first, always, then lime, timber, bricks, salt, grain and stone. Later there are crates of pottery from the north, barrels of beer, sacks of sugar, and once, written in a careful hand with a line under it, an elephant belonging to a travelling circus, which would not walk onto the boat and had to be led along the towpath instead. The toll for the elephant is not recorded. Near the back of the ledger the entries grow shorter and further apart, and the hand that writes them changes twice, and the last page is half empty.

Winter was the hard season. When the canal froze, the boats could not move, and a boat that does not move earns nothing. The company kept an icebreaker, a short heavy boat with an iron-clad bow, and when the ice was thin a team of horses would drag it along the cut while a dozen men stood on deck and rocked it from side to side to break the ice on both banks. When the ice was thick even that failed, and the boats waited in the basin, tied three and four abreast, while the families burned the coal they were supposed to be carrying and hoped the company would not count it too carefully.

The railway came forty years after the canal. At first the canal company laughed at it. A train, they said, could carry a few passengers quickly, but nobody would pay to send coal by steam when a horse could pull thirty tons along still water for the cost of its hay. They were right for about fifteen years. Then the railway companies began to buy the canals, sometimes to run them and sometimes to let them decay, and the rates on the water rose while the rates on the rails fell. The boatyard closed first. Then the warehouses along the basin were turned into a brewery, a furniture store and, for a while, a roller skating rink.

By the time the last working boat came down through the locks, the lock keeper's grandson was keeping them. He had never been a carpenter. He had learned the job from his father, who had learned it from the careful man who opened the gates so slowly on the first day, and he still opened them slowly, out of habit, even though the boat was empty and nobody had come out to watch it. After that the gates stayed shut. Reeds grew in the upper pound, and a heron took to standing on the sill of the bottom gate every morning, as if it were waiting for something to come up.

For a long time afterwards the canal was simply a strip of still water and mud behind the houses. Boys fished in it for small grey fish that nobody ate. In hard winters people skated on the upper pound, and once a cow fell through the ice and was pulled out by six men with a rope and a great deal of shouting. The towpath became a short cut to the station, and then a place where courting couples walked on Sunday afternoons, and few of them knew or cared what it had been built for.

Many years later a group of volunteers began to clear the canal. They were mostly retired, and they worked at weekends with shovels, wheelbarrows and a borrowed pump. It took them six years to restore a single mile. They found a great deal of mud, two bicycles, a surprising number of bottles, and in the bottom of the lock chamber a small brass lamp, green with age, which one of them cleaned and hung in the window of the lock cottage. The lamp is lit now on summer evenings, when the pleasure boats come up through the locks and their crews stop to ask what it is.
EOF
)

IFS=',' read -r -a SETLIST <<< "$SETS"
for s in "${SETLIST[@]}"; do
	case "$s" in
	chat)
		add chat-explain "$NCTX" "Explain in plain words how a refrigerator keeps food cold, for someone who has never studied physics."
		add chat-story "$NCTX" "Write a short story about a night-shift baker who finds an unexpected note tucked inside a sack of flour."
		add chat-code "$NCTX" "In Python, what is the difference between a list and a tuple? Show a short example of when you would use each one."
		add chat-list "$NCTX" "List seven practical tips for keeping a small vegetable garden healthy through a hot, dry summer, one line each." ;;
	copy)
		add copy-repeat "$NCTX" "$PASSAGE_COPY

Repeat the passage above exactly, word for word, with no comment before or after it."
		add copy-summary "$NCTX" "$PASSAGE_COPY

Summarise the passage above in three sentences, keeping its own wording where you can."
		add copy-code "$NCTX" "$CODE_COPY

Rewrite this C function with the variable total renamed to sum. Change nothing else, and print the whole function." ;;
	long)
		LC=$(( NCTX > 2048 ? NCTX : 2048 ))
		add long-400 "$LC" "$LONG1

Using only the text above, explain how the canal changed the town and who lost out from it."
		add long-1400 "$LC" "$LONG1

$LONG2

Using only the text above, tell the history of the canal from its opening to its restoration, and say what happened to the brass lamp." ;;
	*) die "unknown prompt set '$s' (chat, copy, long, all)" ;;
	esac
done

# ================================================================ one run
# Sets R_OK (1/0), R_WHY, R_DEC, R_PMS, R_PTOK, R_GTOK, R_HASH, R_TXT.
run_one() { # arm-letter prompt-index model outfile-stem ngen
	local L=$1 i=$2 M=$3 F=$4 N=$5 B rc line env_arm
	if [ "$L" = A ]; then B=$RUN_A; env_arm=("${ENV_A[@]}"); else B=$RUN_B; env_arm=("${ENV_B[@]}"); fi
	R_OK=0; R_WHY=""; R_DEC=""; R_PMS=""; R_PTOK=""; R_GTOK=""; R_HASH=""; R_TXT="$F.txt"
	J=$(other_jobs)
	if [ -n "$J" ]; then
		R_WHY="another NPU job appeared: $(printf '%s' "$J" | head -1 | cut -c1-80)"
		return
	fi
	# in a subshell so bash's own "Segmentation fault" notice does not land
	# in the report; the run's stderr is in $F.err either way
	( timeout -k 10 "$TMO" env "${BASE_ENV[@]}" "${env_arm[@]}" "$B" "$M" \
		-t "$THREADS" -p "${P_TEXT[$i]}" -n "$N" -c "${P_CTX[$i]}" --chat --temp 0 \
		$IGNORE_EOS >"$F.out" 2>"$F.err" ) 2>/dev/null
	rc=$?
	if wedged; then
		R_WHY="THE NPU WEDGED (dmesg: NPU job timed out)"
		echo "  STOP: the NPU wedged. Nothing measured after a timeout counts; reboot and start again."
		dmesg 2>/dev/null | grep "NPU job timed out" | tail -2 | sed 's/^/     /'
		print_tables; exit 3
	fi
	if [ "$rc" = 124 ]; then R_WHY="timed out after $TMO s"; return; fi
	if [ "$rc" = 137 ]; then R_WHY="killed by SIGKILL (the timeout's own kill, or the OOM killer)"; return; fi
	if [ "$rc" -gt 128 ]; then R_WHY="crashed, signal $((rc - 128))"; return; fi
	if [ "$rc" != 0 ]; then
		R_WHY="exit $rc: $(grep -v '^ *$' "$F.err" | tail -1 | cut -c1-100)"; return
	fi
	# THE RUN DID NOT RUN THE CONFIGURATION, and still printed a summary.
	if grep -q "no NPU; staying on the CPU" "$F.err"; then
		R_WHY="the runtime found no NPU and ran on the CPU"; return; fi
	if grep -q "CHARSIU_CPUS=.* did not apply" "$F.err"; then
		R_WHY="CHARSIU_CPUS did not apply (the scheduling lottery)"; return; fi
	line=$(grep -m1 '^\[load ' "$F.out")
	[ -n "$line" ] || { R_WHY="no [load ... | prompt ... | gen ...] summary line on stdout"; return; }
	R_PTOK=$(printf '%s' "$line" | sed -n 's/.*| prompt \([0-9]*\) tok in \([0-9.]*\) ms.*/\1/p')
	R_PMS=$(printf '%s' "$line" | sed -n 's/.*| prompt \([0-9]*\) tok in \([0-9.]*\) ms.*/\2/p')
	R_GTOK=$(printf '%s' "$line" | sed -n 's/.*| gen \([0-9]*\) tok in [0-9.]* ms, \([0-9.]*\) tok\/s.*/\1/p')
	R_DEC=$(printf '%s' "$line" | sed -n 's/.*| gen \([0-9]*\) tok in [0-9.]* ms, \([0-9.]*\) tok\/s.*/\2/p')
	if [ -z "$R_PMS" ] || [ -z "$R_DEC" ]; then
		R_WHY="summary line did not parse: $(printf '%s' "$line" | cut -c1-100)"; return; fi
	if [ "$R_GTOK" = 0 ]; then R_WHY="generated 0 tokens, so there is no decode rate"; return; fi
	# THE TEXT, AND ONLY THE TEXT: stdout without the bracketed report lines.
	grep -v '^\[' "$F.out" > "$R_TXT"
	R_HASH=$(md5sum < "$R_TXT" | cut -c1-12)
	R_OK=1
}

# stats over a space separated list: "median min max n"
stats() {
	printf '%s\n' $1 | grep . | sort -g | awk '
		{ v[NR] = $1 }
		END {
			if (NR == 0) { print "- - - 0"; exit }
			m = (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2
			print m, v[1], v[NR], NR
		}'
}
# real if |mb - ma| exceeds both spreads, is at least MIN_EFFECT of A (3% by
# default), and both arms have >= 3 runs. The floor is there because a null
# control on the CM5 (the same binary as A and B, 3 reps) flagged a 1% prompt
# difference whose two spreads happened to be 2 ms wide and not to overlap.
MIN_EFFECT=${MIN_EFFECT:-0.03}
judge() { # ma mina maxa na  mb minb maxb nb  higher-is-better(1/0)
	awk -v ma="$1" -v la="$2" -v ha="$3" -v na="$4" \
	    -v mb="$5" -v lb="$6" -v hb="$7" -v nb="$8" -v up="$9" \
	    -v fl="$MIN_EFFECT" 'BEGIN {
		if (na < 1 || nb < 1) { print "FAILED"; exit }
		r = up ? mb / ma : ma / mb
		d = mb - ma; if (d < 0) d = -d
		sa = ha - la; sb = hb - lb
		if (na < 3 || nb < 3) tag = "n<3"
		else if (d > sa && d > sb && d > fl * ma) tag = (r > 1) ? "B FASTER" : "B SLOWER"
		else tag = "within spread"
		printf "%.3fx %s\n", r, tag
	}'
}

declare -A DEC PMS HSH NFAIL WHY PTOK GTOK FIRSTTXT
MODEL_DONE=()

print_model_table() { # model-basename
	local b=$1 i a s ja jb tA tB dline v nP=0 nSame=0 nDiff=0 nUnst=0
	local dF=0 dS=0 dW=0 dX=0 pF=0 pS=0 pW=0 pX=0 fails=0
	echo
	echo "== $b   (decode tok/s higher is better, prompt ms lower is better;"
	echo "   [min-max] is the arm's own spread, Nx is B relative to A, >1 = B faster)"
	printf '%-13s %5s  %-26s %-26s %-21s %-29s %-21s %s\n' prompt ptok \
		"decode A med [min-max]" "decode B med [min-max]" "decode B/A" \
		"prompt ms A / B" "prompt B/A" text
	for i in "${!P_ID[@]}"; do
		local p=${P_ID[$i]} k="$b:${P_ID[$i]}"
		nP=$((nP + 1))
		read -r ma la ha na <<< "$(stats "${DEC[$k:A]:-}")"
		read -r mb lb hb nb <<< "$(stats "${DEC[$k:B]:-}")"
		read -r qa qla qha qna <<< "$(stats "${PMS[$k:A]:-}")"
		read -r qb qlb qhb qnb <<< "$(stats "${PMS[$k:B]:-}")"
		fmt() { [ "$4" = 0 ] && { echo "FAILED (no run)"; return; }; printf '%.2f [%.2f-%.2f] n%d' "$1" "$2" "$3" "$4"; }
		fmq() { [ "$4" = 0 ] && { echo "FAILED"; return; }; printf '%.0f[%.0f-%.0f]' "$1" "$2" "$3"; }
		if [ "$na" = 0 ] || [ "$nb" = 0 ]; then ja="FAILED"; jb="FAILED"
		else ja=$(judge "$ma" "$la" "$ha" "$na" "$mb" "$lb" "$hb" "$nb" 1)
		     jb=$(judge "$qa" "$qla" "$qha" "$qna" "$qb" "$qlb" "$qhb" "$qnb" 0); fi
		case "$ja" in FAILED) dX=$((dX + 1)) ;; *FASTER*) dF=$((dF + 1)) ;; *SLOWER*) dS=$((dS + 1)) ;; *) dW=$((dW + 1)) ;; esac
		case "$jb" in FAILED) pX=$((pX + 1)) ;; *FASTER*) pF=$((pF + 1)) ;; *SLOWER*) pS=$((pS + 1)) ;; *) pW=$((pW + 1)) ;; esac
		# text: stable within each arm, then A against B
		hA=$(printf '%s\n' ${HSH[$k:A]:-} | grep . | sort -u | wc -l)
		hB=$(printf '%s\n' ${HSH[$k:B]:-} | grep . | sort -u | wc -l)
		dline=""
		if [ "$hA" = 0 ] || [ "$hB" = 0 ]; then v="n/a (an arm has no completed run)"
		elif [ "$hA" -gt 1 ] || [ "$hB" -gt 1 ]; then
			v="UNSTABLE (A $hA variants, B $hB) at --temp 0"; nUnst=$((nUnst + 1))
		elif [ "$(printf '%s' "${HSH[$k:A]}" | awk '{print $1}')" = "$(printf '%s' "${HSH[$k:B]}" | awk '{print $1}')" ]; then
			v="TEXT SAME"; nSame=$((nSame + 1))
		else
			v="TEXT DIFFERENT"; nDiff=$((nDiff + 1))
			tA=${FIRSTTXT[$k:A]}; tB=${FIRSTTXT[$k:B]}
			dline=$(awk 'NR == FNR { a[FNR] = $0; na = FNR; next }
				{ b[FNR] = $0; nb = FNR }
				END {
					n = na > nb ? na : nb
					for (i = 1; i <= n; i++) {
						x = (i <= na) ? a[i] : "<end of text>"
						y = (i <= nb) ? b[i] : "<end of text>"
						if (x != y) {
							printf "      first difference, line %d:\n", i
							printf "        A: %s\n", substr(x, 1, 150)
							printf "        B: %s\n", substr(y, 1, 150)
							exit
						}
					}
				}' "$tA" "$tB")
		fi
		printf '%-13s %5s  %-26s %-26s %-21s %-29s %-21s %s\n' "$p" "${PTOK[$k]:-?}" \
			"$(fmt "$ma" "$la" "$ha" "$na")" "$(fmt "$mb" "$lb" "$hb" "$nb")" "$ja" \
			"$(fmq "$qa" "$qla" "$qha" "$qna") / $(fmq "$qb" "$qlb" "$qhb" "$qnb")" "$jb" "$v"
		[ -z "${GTOK[$k]:-}" ] || {
			g=$(printf '%s\n' ${GTOK[$k]} | sort -u | tr '\n' ' ')
			case "$g" in *" "*" "*) echo "      generated token counts differ between runs: $g(rates at different lengths)" ;; esac; }
		[ -z "$dline" ] || printf '%s\n' "$dline"
		for a in A B; do
			[ "${NFAIL[$k:$a]:-0}" = 0 ] || {
				nf=${NFAIL[$k:$a]}; fails=$((fails + nf))
				printf '%s\n' "${WHY[$k:$a]}" | grep . | sort | uniq -c | \
					sed "s/^ *\([0-9]*\) /      arm $a FAILED x\1: /"; }
		done
	done
	# THE VERDICT, ONE LINE. The exception leads: a slower prompt or a
	# different text is said before the summary that would hide it.
	local dv pv tv
	# A FAILED PROMPT IS NOT A TIE. It is counted on its own and said, so a
	# round where half the prompts failed cannot read "no difference".
	side() { # label F S W X n
		local L=$1 F=$2 S=$3 W=$4 X=$5 n=$6 c=$(( $2 + $3 + $4 )) v
		if [ "$c" = 0 ]; then echo "$L FAILED on every prompt"; return; fi
		if [ "$S" -gt 0 ] && [ "$F" = 0 ] && [ "$W" = 0 ]; then v="$L B SLOWER on all $c compared"
		elif [ "$F" -gt 0 ] && [ "$S" = 0 ] && [ "$W" = 0 ]; then v="$L B faster on all $c compared"
		elif [ "$F" = 0 ] && [ "$S" = 0 ]; then v="$L: no measurable difference on $c compared"
		else v="$L MIXED: $S slower, $F faster, $W within spread or n<3"; fi
		[ "$X" = 0 ] || v="$v, $X of $n FAILED and not compared"
		echo "$v"
	}
	dv=$(side decode "$dF" "$dS" "$dW" "$dX" "$nP")
	pv=$(side prompt "$pF" "$pS" "$pW" "$pX" "$nP")
	if [ "$nDiff" -gt 0 ] || [ "$nUnst" -gt 0 ]; then
		tv="TEXT NOT IDENTICAL: $nDiff of $nP different, $nUnst unstable"
	elif [ "$nSame" = "$nP" ]; then tv="text identical on all $nP prompts"
	elif [ "$nSame" = 0 ]; then tv="text NOT COMPARED: no prompt completed in both arms"
	else tv="text identical on $nSame of $nP (the rest had no completed run)"; fi
	local fv=""; [ "$fails" = 0 ] || fv="; $fails runs FAILED, medians are over the runs that completed"
	MODEL_DONE+=("$b: $tv; $dv; $pv$fv")
}
print_tables() {
	local b
	for b in "${MODELS_STARTED[@]}"; do print_model_table "$b"; done
	echo
	echo "== verdict  (A = $ARM_A, B = $ARM_B; boot ${BOOT:0:8}, $REPS reps interleaved A B)"
	for s in "${SKIPPED[@]}"; do echo "  SKIPPED $s"; done
	for v in "${MODEL_DONE[@]}"; do echo "  $v"; done
	[ "$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)" = "$BOOT" ] \
		|| echo "  WARNING the boot id changed during the round: the arms are not one boot"
	echo "  per-run files: $OUT"
}

# ================================================================ the round
MODELS_STARTED=()
echo "prompts: ${P_ID[*]}"
echo "reps $REPS per arm per prompt, -n $NGEN, interleaved A B per prompt, timeout $TMO s"
for M in "${MODEL_FILES[@]}"; do
	b=$(basename "$M"); d="$OUT/${b%.gguf}"; mkdir -p "$d"
	MODELS_STARTED+=("$b")
	echo
	echo "== $b"
	# THE WARM-UP: one discarded run per arm, so neither arm pays the cold
	# file cache and the first staging inside its measured runs.
	for a in A B; do
		run_one "$a" 0 "$M" "$d/warmup.$a" 16
		if [ "$R_OK" = 1 ]; then
			echo "  warm-up $a  (discarded) ok, $(grep -h -m3 -i 'MHz\|pinned' "$d/warmup.$a.err" | tr '\n' ';' | cut -c1-150)"
		else
			echo "  warm-up $a  (discarded) FAILED: $R_WHY"
		fi
	done
	r=1
	while [ "$r" -le "$REPS" ]; do
		for i in "${!P_ID[@]}"; do
			p=${P_ID[$i]}; k="$b:$p"
			for a in A B; do
				run_one "$a" "$i" "$M" "$d/$p.$a.$r" "$NGEN"
				if [ "$R_OK" = 1 ]; then
					DEC[$k:$a]="${DEC[$k:$a]:-} $R_DEC"
					PMS[$k:$a]="${PMS[$k:$a]:-} $R_PMS"
					HSH[$k:$a]="${HSH[$k:$a]:-} $R_HASH"
					GTOK[$k]="${GTOK[$k]:-} $R_GTOK"
					PTOK[$k]=$R_PTOK
					[ -n "${FIRSTTXT[$k:$a]:-}" ] || FIRSTTXT[$k:$a]=$R_TXT
					printf '  rep %d  %-13s %s  decode %6s tok/s  prompt %6s ms (%s tok)  gen %s  text %s\n' \
						"$r" "$p" "$a" "$R_DEC" "$R_PMS" "$R_PTOK" "$R_GTOK" "$R_HASH"
				else
					NFAIL[$k:$a]=$(( ${NFAIL[$k:$a]:-0} + 1 ))
					WHY[$k:$a]="${WHY[$k:$a]:-}$R_WHY
"
					printf '  rep %d  %-13s %s  FAILED: %s\n' "$r" "$p" "$a" "$R_WHY"
				fi
			done
		done
		r=$((r + 1))
	done
done
print_tables
