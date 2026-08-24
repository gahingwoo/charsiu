#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# charsiu-install.sh -- build and install charsiu onto a board that already has
# a kernel able to drive the NPU.
#
#   sh scripts/charsiu-install.sh              build, install, fetch a model
#   sh scripts/charsiu-install.sh --no-model   skip the download
#   sh scripts/charsiu-install.sh --prefix DIR stage into DIR instead of /
#   sh scripts/charsiu-install.sh --uninstall  remove what this installed
#
# ⚠⚠ IT CANNOT INSTALL A KERNEL, AND RK3576 SUPPORT IS NOT UPSTREAM YET.
#
# The rocket driver is in mainline for RK3588. RK3576 support is an unmerged
# series -- as of 2026-08-25 it is v9 on the linux-media / dri-devel lists, and
# the commit that adds `rockchip,rk3576-rknn-core` is not reachable from any
# tag. So this script's FIRST job is to find out whether the running kernel can
# do the work at all, and to stop and say what is missing if it cannot. Building
# a userspace onto a kernel that will never bind the NPU wastes everyone's time.
#
#   the patches:  https://github.com/gahingwoo/linux-rk3576-npu  (kernel/*.patch)
#   the series:   lore.kernel.org, "accel/rocket: RK3576 NPU (RKNN) enablement"
#
# Env: CHARSIU_ACCEL_DEV names the accel node (default /dev/accel/accel0).
set -eu

PREFIX="/"
DOMODEL=1
DOBUILD=1
UNINSTALL=0
ASSUME=0

while [ $# -gt 0 ]; do
	case "$1" in
	--prefix)    PREFIX="$2"; shift 2 ;;
	--no-model)  DOMODEL=0; shift ;;
	--no-build)  DOBUILD=0; shift ;;
	--uninstall) UNINSTALL=1; shift ;;
	-y|--yes)    ASSUME=1; shift ;;
	-h|--help)   sed -n '4,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
	*)           echo "unknown option: $1" >&2; exit 2 ;;
	esac
done

if [ -t 1 ]; then
	C_C=$(printf '\033[1;36m'); C_R=$(printf '\033[1;31m')
	C_Y=$(printf '\033[1;33m'); C_G=$(printf '\033[1;32m')
	C_D=$(printf '\033[2m');    C_0=$(printf '\033[0m')
else C_C=; C_R=; C_Y=; C_G=; C_D=; C_0=; fi
say()  { printf '\n%s[charsiu-install]%s %s\n' "$C_C" "$C_0" "$*"; }
warn() { printf '%s[charsiu-install] WARN:%s %s\n' "$C_Y" "$C_0" "$*" >&2; }
die()  { printf '\n%s[charsiu-install] ERROR:%s %s\n' "$C_R" "$C_0" "$*" >&2; exit 1; }

SRC=$(cd "$(dirname "$0")/.." && pwd)
BIN="$PREFIX/opt/charsiu"
SBIN="$PREFIX/usr/bin"
ETC="$PREFIX/etc/charsiu"
MODELS="$BIN/models"

# ⚠ Escalate only when the DESTINATION needs it. Testing `id -u` instead made
# --prefix into a directory the user already owns prompt for a password and then
# fail, which defeats the whole point of staging.
writable() {
	d=$1
	while [ ! -e "$d" ] && [ "$d" != "/" ] && [ "$d" != "." ]; do d=$(dirname "$d"); done
	[ -w "$d" ]
}
NEEDROOT=0
writable "$BIN" && writable "$SBIN" && writable "$ETC" || NEEDROOT=1
SUDO=""
if [ "$NEEDROOT" = 1 ] && [ "$(id -u)" -ne 0 ]; then
	SUDO=$(command -v sudo || true)
	[ -n "$SUDO" ] || die "$PREFIX needs root and sudo is not installed"
fi
as_root() { if [ -n "$SUDO" ]; then $SUDO "$@"; else "$@"; fi; }

if [ "$UNINSTALL" = 1 ]; then
	say "removing charsiu (models in $MODELS are LEFT ALONE)"
	for f in charsiu charsiu-get charsiu-config charsiu-doctor; do
		as_root rm -f "$SBIN/$f"
	done
	for f in charsiu_run charsiu_check; do as_root rm -f "$BIN/$f"; done
	printf '  %sconfig kept at %s -- delete it by hand if you mean to%s\n' \
		"$C_D" "$ETC/config.ini" "$C_0"
	exit 0
fi

# ---------------------------------------------------------------------------
say "preflight -- can this kernel drive the NPU at all?"
# ---------------------------------------------------------------------------
[ "$(uname -m)" = "aarch64" ] || die "charsiu's NPU path is aarch64 only (this is $(uname -m))"

# A board with more than one accel device, or a test rig, can name it.
ACCEL="${CHARSIU_ACCEL_DEV:-/dev/accel/accel0}"
if [ -e "$ACCEL" ]; then
	printf '  %s[ OK ]%s %s -- the rocket driver is bound\n' "$C_G" "$C_0" "$ACCEL"
else
	printf '  %s[FAIL]%s %s is missing\n' "$C_R" "$C_0" "$ACCEL"
	# Say WHICH half is missing: no node in DT, or a node with no driver.
	if [ -d /proc/device-tree ] && \
	   grep -rlq 'rknn-core' /proc/device-tree 2>/dev/null; then
		printf '         the device tree HAS an rknn-core node, so the DTS is fine\n'
		printf '         and the kernel has no driver bound to it.\n'
		printf '         CONFIG_DRM_ACCEL_ROCKET, plus the RK3576 patches:\n'
	else
		printf '         no rknn-core node in the device tree either, so the DTB\n'
		printf '         does not describe the NPU. Both halves are needed:\n'
	fi
	printf '         %shttps://github.com/gahingwoo/linux-rk3576-npu%s (kernel/*.patch)\n' \
		"$C_D" "$C_0"
	die "stopping: a userspace built now would have nothing to talk to"
fi

n=0
for d in /sys/bus/platform/devices/*.npu; do [ -d "$d" ] && n=$((n + 1)); done
case "$n" in
2) printf '  %s[ OK ]%s two NPU cores\n' "$C_G" "$C_0" ;;
1) warn "only ONE NPU core is enabled; charsiu's int4 decode wants both and one core is about 1.4x slower" ;;
0) warn "no *.npu platform device, though /dev/accel/accel0 exists" ;;
esac

if [ "$DOBUILD" = 1 ]; then
	command -v make >/dev/null 2>&1 || die "make is not installed"
	command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 \
		|| die "no C compiler (cc or gcc)"
	printf '  %s[ OK ]%s a toolchain\n' "$C_G" "$C_0"
fi

# ---------------------------------------------------------------------------
if [ "$DOBUILD" = 1 ]; then
	say "building in $SRC"
	# `all` is the native set: charsiu_run and charsiu_check. The .aarch64
	# targets are for cross building from a host and are not wanted here.
	( cd "$SRC" && make all ) || die "the build failed"
	printf '  %s[ OK ]%s built\n' "$C_G" "$C_0"
	RUNBIN="$SRC/build/charsiu_run"; CHKBIN="$SRC/build/charsiu_check"
else
	RUNBIN="$SRC/build/charsiu_run"; CHKBIN="$SRC/build/charsiu_check"
	[ -x "$RUNBIN" ] || die "--no-build was given but $RUNBIN does not exist"
fi

say "installing"
as_root mkdir -p "$BIN" "$SBIN" "$ETC" "$MODELS"
as_root cp "$RUNBIN" "$BIN/charsiu_run"
as_root cp "$CHKBIN" "$BIN/charsiu_check"
for f in charsiu charsiu-get charsiu-config charsiu-doctor; do
	as_root cp "$SRC/scripts/$f" "$SBIN/$f"
	as_root chmod 0755 "$SBIN/$f"
done
printf '  %s[ OK ]%s %s and %s\n' "$C_G" "$C_0" "$BIN" "$SBIN"

# ⚠ NEVER overwrite a config someone has edited. This is the one file with
# their choices in it, and an installer that clobbers it is a bug.
if [ -f "$ETC/config.ini" ]; then
	printf '  %s[keep]%s %s already exists and was left alone\n' "$C_Y" "$C_0" "$ETC/config.ini"
	as_root cp "$SRC/etc/config.ini" "$ETC/config.ini.default"
else
	as_root cp "$SRC/etc/config.ini" "$ETC/config.ini"
	printf '  %s[ OK ]%s %s\n' "$C_G" "$C_0" "$ETC/config.ini"
fi

# ---------------------------------------------------------------------------
if [ "$DOMODEL" = 1 ]; then
	say "a model"
	have=$(ls "$MODELS"/*.gguf 2>/dev/null | head -1 || true)
	if [ -n "$have" ]; then
		printf '  %s[ OK ]%s %s is already here\n' "$C_G" "$C_0" "$(basename "$have")"
	else
		printf '  nothing in %s yet.\n' "$MODELS"
		printf '  %sthe smallest that talks is 92 MB; the one every board round\n' "$C_D"
		printf '  uses is 1321 MB. charsiu-get lists the rest.%s\n\n' "$C_0"
		if [ "$ASSUME" = 1 ]; then a=y; else
			printf '  fetch SmolLM2-135M-Q4_0 (92 MB) now? [y/N] '
			read -r a || a=n
		fi
		case "$a" in
		y|Y) CHARSIU_MODELS_DIR="$MODELS" CHARSIU_CHECK="$BIN/charsiu_check" \
			"$SBIN/charsiu-get" smollm2-135m-q4 || warn "the fetch did not finish; charsiu-get resumes" ;;
		*)   printf '  skipped. %scharsiu-get%s when you want one.\n' "$C_D" "$C_0" ;;
		esac
	fi
fi

# ---------------------------------------------------------------------------
say "checking the install"
CHARSIU_CONFIG="$ETC/config.ini" "$SBIN/charsiu-doctor" || \
	warn "charsiu-doctor reported something critical -- read it above"

cat <<EOF

  ${C_G}installed${C_0}

    charsiu-config     pick a model, set threads and context
    charsiu -p "..."   run it
    charsiu-doctor     what works and what does not
    charsiu-get        more models

  ${C_D}config: $ETC/config.ini   models: $MODELS${C_0}
EOF
