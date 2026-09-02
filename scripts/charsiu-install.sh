#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# charsiu-install.sh: set up charsiu on a Rockchip RK3576 board.
#
#   sh charsiu-install.sh              the wizard
#   sh charsiu-install.sh --kernel     only offer the kernel step
#   sh charsiu-install.sh --no-kernel  skip it
#   sh charsiu-install.sh --dry-run    say what it would do, change nothing
#   sh charsiu-install.sh --prefix DIR stage instead of installing
#   sh charsiu-install.sh --uninstall  remove what this installed
#   sh charsiu-install.sh --dev        the probes too, and track dev
#   CHARSIU_PLAIN=1 ...                no full-screen dialogs
#
# ⚠ TWO CHANNELS. stable is the runtime and is what a fresh install gets. dev
# adds npu_gemm_test, charsiu_matmul and bench_batch, which exist to ask the
# hardware questions and have wedged the block doing it, and tracks the branch
# the work happens on. `charsiu update dev` switches later; nothing here needs
# reinstalling to change channel.
#
# ⚠⚠ RK3576 NPU SUPPORT IS NOT UPSTREAM, SO NO STOCK KERNEL CAN RUN THIS.
#
# The rocket driver is mainline for RK3588. The commit adding
# `rockchip,rk3576-rknn-core` is ours, from 2026-08-06, and is not reachable
# from any tag. An earlier version of this script checked for a working NPU and
# stopped when it did not find one. That was every machine on earth, so the
# check was a dead end wearing a helpful expression. It now OFFERS A KERNEL:
# CI builds linux-next plus the v9 series and publishes it, and this fetches it.
#
# ⚠ AND IT KEEPS THE ONE ALREADY THERE. The new kernel becomes the default boot
# entry and the previous one stays on the card as a second entry, because a
# kernel that does not boot is not a thing to discover with no way back.
# ⚠⚠ THE WHOLE SCRIPT IS ONE BRACE GROUP, ON PURPOSE.
#
# `sh` reads a piped script in chunks and runs each one as it arrives. This
# script reattaches the terminal with `exec < /dev/tty`, which CLOSES the pipe
# curl is still writing into, and curl then dies with
#
#     curl: (23) Failure writing output to destination, passed 1378 returned 1311
#
# Measured, not theorised: that is what the first `curl ... | sh` run printed.
# A brace group has to be parsed to its closing brace before any of it runs, so
# the shell drains the whole stream first and curl finishes cleanly. It also
# means a truncated download runs nothing at all instead of half of something.
{
set -eu

# ---------------------------------------------------------------------------
# BOOTSTRAP: the `curl ... | sh` case, before anything else can need it.
# ---------------------------------------------------------------------------
#
# ⚠ Read before the terminal check, which branches on it.
_BOOT_DRY=0; _BOOT_NOTTY=0
for _a in "$@"; do case "$_a" in --dry-run|-n) _BOOT_DRY=1 ;; esac; done
# ⚠ THE CHANNEL HAS TO BE KNOWN BEFORE THE SOURCE IS FETCHED, because it
# decides which ref to fetch. The full argument parsing happens much later, in
# the tree this is about to download.
_BOOT_REF=stable
for _a in "$@"; do case "$_a" in
	--dev) _BOOT_REF=${CHARSIU_DEV_REF:-dev} ;;
	--stable) _BOOT_REF=stable ;;
esac; done

CHARSIU_SRC_REPO="${CHARSIU_SRC_REPO:-https://github.com/gahingwoo/charsiu}"
CHARSIU_SELF_URL="https://raw.githubusercontent.com/gahingwoo/charsiu/stable/scripts/charsiu-install.sh"

# ⚠⚠ PIPED IN, STDIN IS THE SCRIPT ITSELF. Every `read` would eat the rest of
# this file, and a wizard that asks questions cannot run that way. Reattach the
# terminal first, and if there is not one, say so rather than silently
# consuming ourselves.
if [ ! -t 0 ]; then
	# ⚠ A FAILED REDIRECTION ON `exec` KILLS THE SHELL. Testing with
	# `exec < /dev/tty` meant that when there was no controlling terminal the
	# script exited silently instead of saying why. And `[ -r /dev/tty ]`
	# is not the test either: the device node can be readable while opening
	# it returns ENXIO. Probe in a subshell, where a failure costs nothing.
	if ( : < /dev/tty ) 2>/dev/null; then
		exec < /dev/tty
	elif [ "$_BOOT_DRY" = 1 ]; then
		# ⚠ A REHEARSAL NEEDS NO CONSENT. It writes nothing, so refusing to
		# run for want of a terminal would be refusing the one thing asked
		# for, and piping this into a container is exactly how it gets
		# rehearsed. Answer every question with yes and carry on.
		CTUI_ASSUME=yes; export CTUI_ASSUME
		_BOOT_NOTTY=1
	else
		echo "charsiu-install: there is no terminal to ask questions on." >&2
		echo "" >&2
		echo "  curl -fsSL $CHARSIU_SELF_URL -o install.sh && sh install.sh" >&2
		echo "" >&2
		echo "  or rehearse it without one:" >&2
		echo "  curl -fsSL $CHARSIU_SELF_URL | sh -s -- --dry-run" >&2
		echo "" >&2
		exit 1
	fi
fi

# ⚠ THE TUI LAYER DOES NOT EXIST YET. It lives in the source, which is the very
# thing this stage is here to fetch, so the bootstrap cannot source it. But
# whiptail may well be installed already, and a wizard that opens with a bare
# shell prompt and only becomes a dialog later is worse than one that is a
# dialog throughout. These two use whiptail when it is there.
_boot_ui() { command -v whiptail >/dev/null 2>&1 && [ -n "${TERM:-}" ] && \
             [ "${TERM:-}" != dumb ] && [ -z "${CHARSIU_PLAIN:-}" ]; }
_boot_msg() {
	if _boot_ui; then whiptail --title "charsiu setup" --msgbox "$1" 16 72
	else printf '\n%s\n' "$1"; fi
}
_boot_yesno() {
	if _boot_ui; then whiptail --title "charsiu setup" --yesno "$1" 16 72
	else
		printf '\n%s\n\n  [Y/n] ' "$1"
		read -r _a || _a=n
		case "${_a:-y}" in n|N|no|NO) return 1 ;; *) return 0 ;; esac
	fi
}

# ⚠ Piped in, "$(dirname "$0")" is meaningless: there is no tree beside us.
# Anything that needs the source (the build, the scripts, the TUI layer) has to
# come from somewhere, so fetch it and hand over to the copy that has neighbours.
_boot_src=$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || echo "")
if [ -z "$_boot_src" ] || [ ! -f "$_boot_src/Makefile" ] || \
   [ ! -f "$_boot_src/scripts/charsiu-tui.sh" ]; then
	DIR="${CHARSIU_DIR:-}"
	if [ -z "$DIR" ]; then
		if [ "$_BOOT_DRY" = 1 ]; then
			# ⚠ A REHEARSAL THAT WRITES 900 KB INTO /opt IS NOT A
			# REHEARSAL. The dialog says nothing is written or
			# downloaded, and on a real Debian this stage was quietly
			# doing both, into a root-owned directory, before the
			# dry-run flag was ever looked at. Fetch somewhere
			# disposable instead, and throw it away on the way out.
			DIR="${TMPDIR:-/tmp}/charsiu-dryrun.$$"
			CHARSIU_DRY_SRC="$DIR"; export CHARSIU_DRY_SRC
		elif [ "$(id -u)" -eq 0 ]; then
			DIR=/opt/charsiu/src
		else
			# ⚠ HAVING sudo IS NOT A REASON TO PUT THE SOURCE WHERE YOU
			# CANNOT BUILD. This used to fetch into /opt/charsiu/src with
			# sudo, leaving a root-owned tree, and then ran `make` as the
			# user, which could not even create build/. On the board that
			# came out as "the build failed" with the reason discarded.
			# The source is not a system asset; only the install is.
			DIR="${XDG_CACHE_HOME:-$HOME/.cache}/charsiu/src"
		fi
	fi
	if [ "$_BOOT_DRY" = 1 ]; then
		printf '\n  dry run: fetching the source into %s (deleted on exit)\n' "$DIR"
	elif [ "$_BOOT_NOTTY" = 1 ]; then
		printf '\n  fetching the source into %s\n' "$DIR"
	else
		_boot_yesno "charsiu needs its source to build from.

It will be fetched into
  $DIR

Continue?" || { echo "  stopped."; exit 1; }
	fi

	# ⚠ `-w` ON A PATH THAT DOES NOT EXIST YET IS ALWAYS FALSE. This asked for
	# a sudo password to create a directory inside the user's own home, which
	# on the board arrived as a bare "[sudo] password for" right after a
	# dialog had cleared the screen. Try to make it first; only a real failure
	# is a reason to escalate.
	_sudo=""
	if [ "$(id -u)" -ne 0 ] && ! mkdir -p "$(dirname "$DIR")" 2>/dev/null; then
		_sudo=$(command -v sudo || true)
	fi
	if [ -d "$DIR/.git" ]; then
		printf '  updating %s (%s)\n' "$DIR" "$_BOOT_REF"
		# ⚠ fetch and move to the ref rather than pull, or a tree that
		# is on one branch stays there however stable was asked for.
		#
		# ⚠⚠ AND A FAILED FETCH USED TO BE SWALLOWED BY `|| true`, so a
		# ref that does not exist meant building whatever was already in
		# the directory and saying "updating" while doing it. The
		# development branch was renamed from main to dev on 2026-08-28
		# and every existing checkout hit exactly that: a silent build of
		# a stale tree. An install that cannot reach the code it was
		# asked for has to say so.
		if ! $_sudo git -C "$DIR" fetch --quiet origin "$_BOOT_REF"; then
			echo "  cannot fetch $_BOOT_REF from $CHARSIU_SRC_REPO." >&2
			echo "  If this tree predates 2026-08-28 it is on the old" >&2
			echo "  branch name; the development branch is 'dev' now." >&2
			echo "  Delete $DIR and install again." >&2
			exit 1
		fi
		$_sudo git -C "$DIR" checkout --quiet -B "$_BOOT_REF" FETCH_HEAD \
			|| { echo "  $DIR will not move to $_BOOT_REF." >&2; exit 1; }
	elif command -v git >/dev/null 2>&1; then
		$_sudo mkdir -p "$(dirname "$DIR")"
		$_sudo git clone --depth 1 --quiet --branch "$_BOOT_REF" \
			"$CHARSIU_SRC_REPO" "$DIR" \
			|| { echo "  clone failed: $CHARSIU_SRC_REPO ($_BOOT_REF)" >&2; exit 1; }
	else
		# ⚠ no git is a normal state on a minimal rootfs, and a tarball
		# needs neither git nor a key.
		printf '  git is not installed; taking a tarball instead\n'
		$_sudo mkdir -p "$DIR"
		curl -fsSL "$CHARSIU_SRC_REPO/archive/refs/heads/$_BOOT_REF.tar.gz" \
		 | $_sudo tar -xz -C "$DIR" --strip-components=1 \
			|| { echo "  download failed" >&2; exit 1; }
	fi
	[ -f "$DIR/scripts/charsiu-install.sh" ] \
		|| { echo "  $DIR does not look like charsiu" >&2; exit 1; }
	printf '  handing over to %s\n\n' "$DIR/scripts/charsiu-install.sh"
	exec sh "$DIR/scripts/charsiu-install.sh" "$@"
fi
unset _boot_src

# ---------------------------------------------------------------------------
# THE DIALOG ITSELF
# ---------------------------------------------------------------------------
#
# ⚠ WITHOUT whiptail EVERY PAGE SILENTLY BECOMES A SHELL PROMPT. charsiu-tui.sh
# falls back on purpose, because a serial console with no TERM has to work, but
# a fresh Debian or Ubuntu has no whiptail and falling back there is not a
# feature, it is the wizard quietly not being one. So ask, once, and install it.
#
# The package is named differently everywhere: whiptail on Debian and Ubuntu,
# newt on Fedora and Alpine, libnewt on Arch.
if ! command -v whiptail >/dev/null 2>&1 && [ -z "${CHARSIU_PLAIN:-}" ]; then
	_pm=""
	if   command -v apt-get >/dev/null 2>&1; then _pm="apt-get install -y --no-install-recommends whiptail"
	elif command -v dnf     >/dev/null 2>&1; then _pm="dnf install -y newt"
	elif command -v apk     >/dev/null 2>&1; then _pm="apk add newt"
	elif command -v pacman  >/dev/null 2>&1; then _pm="pacman -S --noconfirm libnewt"
	fi
	if [ -n "$_pm" ]; then
		# ⚠ THE ONE THING A DRY RUN DOES FOR REAL, AND WHY. whiptail is the
		# MEDIUM, not the content. Deferring it made the rehearsal run
		# entirely in text, so it rehearsed everything except the interface
		# it exists to show. A dry run that cannot draw the wizard is not
		# rehearsing the wizard.
		_why="This wizard draws dialogs with whiptail and it is not installed,
so every page would be a plain shell prompt instead.

  $_pm

Answering no is fine. Everything still works, in text."
		[ "$_BOOT_DRY" = 1 ] && _why="$_why

⚠ This is the ONE thing this dry run would actually do. Without it
there is no dialog to show you, and the rehearsal would be text."
		if [ "$_BOOT_NOTTY" = 1 ]; then
			printf '\n  no terminal, so whiptail is not installed and this runs in text\n'
			printf '  (%s, if you want the dialogs)\n' "$_pm"
		elif _boot_yesno "$_why" ; then
			_bs=""
			[ "$(id -u)" -eq 0 ] || _bs=$(command -v sudo || true)
			command -v apt-get >/dev/null 2>&1 && $_bs apt-get update -qq >/dev/null 2>&1
			# shellcheck disable=SC2086
			$_bs $_pm >/dev/null 2>&1 || printf '  could not install it; carrying on in text\n'
		fi
	fi
fi

REPO="${CHARSIU_REPO:-gahingwoo/linux-rk3576-npu}"
PREFIX="/"
DRY=0
DOKERNEL=ask
DOMODEL=1
DOBUILD=1
UNINSTALL=0

CHANNEL="${CHARSIU_CHANNEL:-stable}"
while [ $# -gt 0 ]; do
	case "$1" in
	--prefix)    PREFIX="$2"; shift 2 ;;
	--dry-run|-n) DRY=1; shift ;;
	--kernel)    DOKERNEL=only; shift ;;
	--no-kernel) DOKERNEL=no; shift ;;
	--no-model)  DOMODEL=0; shift ;;
	--no-build)  DOBUILD=0; shift ;;
	# stable installs the runtime; dev adds the probes. See INSTALL_BINS.
	--channel)   CHANNEL="$2"; shift 2 ;;
	--dev)       CHANNEL=dev; shift ;;
	--stable)    CHANNEL=stable; shift ;;
	--uninstall) UNINSTALL=1; shift ;;
	-h|--help)   sed -n '4,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
	*)           echo "unknown option: $1" >&2; exit 2 ;;
	esac
done

SRC=$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || echo /opt/charsiu)
# ⚠ The TUI layer has to be findable from every layout this ships in: a source
# tree, a real install under /opt/charsiu, and a staged --prefix install where
# /opt is not at the root. CHARSIU_LIB names it outright; the rest are guesses
# in the order they are likely to be right.
for t in ${CHARSIU_LIB:+"$CHARSIU_LIB/charsiu-tui.sh"} \
	 "$(dirname "$0")/charsiu-tui.sh" \
	 "$SRC/scripts/charsiu-tui.sh" \
	 "$(dirname "$0")/../opt/charsiu/charsiu-tui.sh" \
	 /opt/charsiu/charsiu-tui.sh; do
	[ -r "$t" ] && { . "$t"; break; }
done
command -v ui_msg >/dev/null 2>&1 || { echo "charsiu-tui.sh not found" >&2; exit 1; }
CTUI_TITLE="charsiu setup"

# ⚠ --prefix / gives //opt/charsiu without this. Harmless to the kernel and
# ugly in a dry run's summary, which is the one place people read these paths.
# ⚠ Two different needs. For BUILDING paths the trailing slash has to go, and
# "/" trimmed to "" is exactly right, because "$PREFIX/opt/..." then gives /opt/...
# rather than //opt/... . For SHOWING it, the empty string reads as a blank.
PREFIX=$(printf '%s' "$PREFIX" | sed 's|/*$||')
PDISP="${PREFIX:-/}"
BIN="$PREFIX/opt/charsiu"
SBIN="$PREFIX/usr/bin"
ETC="$PREFIX/etc/charsiu"
# ⚠ THE MODELS DIRECTORY MUST NOT NEED A CHOWN AT ALL. Under /opt it lands
# root-owned, and then charsiu-get, which nobody should have to run as root to
# download a file, fails at the last step, after the download. A user's models
# belong in the user's own directory; only a root install puts them in /opt.
if [ -n "$PREFIX" ] || [ "$(id -u)" -eq 0 ]; then
	MODELS="$BIN/models"
else
	MODELS="${CHARSIU_MODELS:-$HOME/.charsiu/models}"
fi

die() { ui_msg "$1"; exit 1; }

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
	# ⚠ A DRY RUN WRITES NOTHING, so it has no business demanding root. This
	# refused to even rehearse as an ordinary user, which is the one case a
	# rehearsal is most wanted.
	if [ -z "$SUDO" ] && [ "$DRY" = 0 ]; then
		die "$PDISP needs root and sudo is not installed."
	fi
	[ -z "$SUDO" ] && ui_warn "not root and no sudo: a real run would need one"
fi
# ⚠ A BARE "[sudo] password for ..." APPEARING AFTER A DIALOG HAS CLEARED THE
# SCREEN LOOKS LIKE THE INSTALLER BROKE. Ask once, deliberately, with the
# reason still on screen, and let sudo's own timestamp cover everything after.
if [ -n "$SUDO" ] && [ "$DRY" = 0 ] && ! $SUDO -n true 2>/dev/null; then
	ui_msg "Installing into
  $BIN
  $SBIN
needs root, so sudo will ask for your password on the next screen.

Nothing else in this install asks for it."
	$SUDO -v || die "sudo was declined, so nothing was installed."
fi
# ⚠ EVERY MUTATION GOES THROUGH THIS. --dry-run prints the command instead of
# running it, so the difference between a rehearsal and the real thing is one
# branch in one place rather than a flag threaded through twenty call sites --
# which is how a dry run ends up writing something anyway.
DRYLOG=""
as_root() {
	if [ "$DRY" = 1 ]; then
		printf '  %swould%s  %s\n' "$T_Y" "$T_0" "$*" >&2
		DRYLOG="$DRYLOG
  $*"
		return 0
	fi
	if [ -n "$SUDO" ]; then $SUDO "$@"; else "$@"; fi
}
# for things that are not a single command: a network fetch, a build, a child tool
would() {
	printf '  %swould%s  %s\n' "$T_Y" "$T_0" "$*" >&2
	DRYLOG="$DRYLOG
  $*"
}

fetch() {  # fetch URL OUT
	if command -v curl >/dev/null 2>&1; then
		curl -fL --retry 3 --connect-timeout 15 -C - -o "$2" "$1"
	elif command -v wget >/dev/null 2>&1; then
		wget -q -c -O "$2" "$1"
	else
		return 1
	fi
}
api() {
	if command -v curl >/dev/null 2>&1; then curl -fsL --connect-timeout 15 "$1"
	elif command -v wget >/dev/null 2>&1; then wget -qO- "$1"
	else return 1; fi
}

# ---------------------------------------------------------------------------
if [ "$UNINSTALL" = 1 ]; then
	ui_yesno "Remove charsiu?

The models in $MODELS and your $ETC/config.ini are LEFT ALONE.
A kernel this installed is NOT removed either. Pick the previous
entry in the boot menu instead, then remove it by hand." defaultno || exit 0
	# ⚠⚠ REMOVE WHAT IS THERE, NOT WHAT THE LIST SAID WHEN IT WAS WRITTEN --
	# which is what the sentence that used to be here promised and the six
	# hardcoded names underneath it did not do. It missed npu_gemm_test,
	# charsiu_matmul, prefill_control.sh, charsiu_vision, charsiu_clip and
	# charsiu_whisper, so an uninstall left most of a dev install behind and
	# said "Removed."
	#
	# ⚠ FILES ONLY, AND NOT RECURSIVELY. $BIN is /opt/charsiu and the
	# default models directory is /opt/charsiu/models, which the message
	# above promises to leave alone.
	as_root rm -f "$SBIN/charsiu"
	for f in "$SBIN"/charsiu-* "$SBIN"/charsiu_*; do
		[ -f "$f" ] && as_root rm -f "$f"
	done
	for f in "$BIN"/*; do
		[ -f "$f" ] && as_root rm -f "$f"
	done
	ui_msg "Removed. Models and config kept."
	exit 0
fi

# ---------------------------------------------------------------------------
# PREFLIGHT
# ---------------------------------------------------------------------------
ACCEL="${CHARSIU_ACCEL_DEV:-/dev/accel/accel0}"
NPU_OK=0
[ -e "$ACCEL" ] && NPU_OK=1

[ "$(uname -m)" = "aarch64" ] || die "charsiu's NPU path is aarch64 only; this is $(uname -m)."

if [ "$DRY" = 1 ]; then
	ui_msg "DRY RUN

Nothing is written, downloaded, built or chowned. Every action is
printed as \"would ...\" and a summary follows at the end.

Read-only checks still run for real. That is the point: this is
what the installer SEES on this machine."
fi

if [ "$DOKERNEL" != only ]; then
	ui_msg "charsiu, an open LLM runtime for the RK3576 NPU

This will:
  * check whether this kernel can drive the NPU, and offer one if not
  * build and install charsiu
  * fetch a model you can actually run
  * check the result

Nothing is written until each step is agreed to."
fi

# ---------------------------------------------------------------------------
# THE KERNEL
# ---------------------------------------------------------------------------
install_kernel() {
	# ⚠ THE ONE THING NOT TO GUESS. If this board does not boot through
	# extlinux, writing an extlinux.conf achieves nothing at best and
	# confuses the next person at worst. Say so and leave the board alone.
	# CHARSIU_BOOTDIR names it outright, for a boot partition mounted
	# somewhere else, and for rehearsing this on a machine that has none.
	BOOTDIR=""; BOOTKIND=""
	for b in ${CHARSIU_BOOTDIR:+"$CHARSIU_BOOTDIR"} /boot /boot/firmware /mnt/boot; do
		if [ -f "$b/extlinux/extlinux.conf" ]; then
			BOOTDIR="$b"; BOOTKIND=extlinux; break
		fi
		# ⚠ ARMBIAN IS THE OTHER LAYOUT WORTH KNOWING, and it is what most
		# of these boards actually run: boot.scr reads armbianEnv.txt and
		# loads ${prefix}Image and dtb/${fdtfile}. Since no distro kernel
		# binds this NPU, refusing here left the one thing charsiu needs
		# switched off on the commonest install there is.
		if [ -f "$b/armbianEnv.txt" ]; then
			BOOTDIR="$b"; BOOTKIND=armbian; break
		fi
	done
	if [ -z "$BOOTDIR" ]; then
		ui_msg "No extlinux.conf and no armbianEnv.txt under /boot.

This board boots some other way (its own boot.scr, a distro kernel
package, or U-Boot environment variables), and guessing at it would
be worse than doing nothing.

Install a kernel built from the series by hand instead:
  https://github.com/$REPO  (rfc-send-v9/, kernel/base.config)"
		return 1
	fi

	# ⚠ A NAMED RELEASE, FOR A TEST KERNEL. CHARSIU_KERNEL_TAG=<tag> asks for
	# that release instead of the latest one, which is how a pre-release --
	# something `releases/latest` never returns -- gets installed on purpose
	# and only on purpose. The board scripts use it to put a kernel with
	# patches under test on the board; nobody else should see it.
	if [ -n "${CHARSIU_KERNEL_TAG:-}" ]; then
		ui_note "asking $REPO for the kernel tagged $CHARSIU_KERNEL_TAG..."
		J=$(api "https://api.github.com/repos/$REPO/releases/tags/$CHARSIU_KERNEL_TAG" 2>/dev/null || true)
	else
		ui_note "asking $REPO for the latest kernel..."
		J=$(api "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null || true)
	fi
	if [ -z "$J" ]; then
		ui_msg "Could not reach the GitHub API.

Check the network (charsiu-doctor reports it), or build a kernel
from rfc-send-v9/ by hand."
		return 1
	fi
	URLS=$(echo "$J" | tr ',' '\n' | grep -o '"browser_download_url":[ ]*"[^"]*"' \
		| sed 's/.*"\(https[^"]*\)"/\1/')
	TAG=$(echo "$J" | tr ',' '\n' | grep -m1 -o '"tag_name":[ ]*"[^"]*"' \
		| sed 's/.*"\([^"]*\)"$/\1/')
	IMG=$(echo "$URLS" | grep -m1 '/Image$'   || true)
	DTB=$(echo "$URLS" | grep -m1 '\.dtb$'    || true)
	MODS=$(echo "$URLS" | grep -m1 'modules-.*\.tar\.gz$' || true)
	SUMS=$(echo "$URLS" | grep -m1 'SHA256SUMS$' || true)
	if [ -z "$IMG" ] || [ -z "$DTB" ]; then
		ui_msg "The latest release of $REPO has no kernel in it.

Nothing was changed."
		return 1
	fi

	if [ "$BOOTKIND" = extlinux ]; then
		ui_yesno "Install this kernel?

  release   $TAG
  into      $BOOTDIR  (extlinux)

The kernel now on this board is kept as a SECOND boot entry. The new
one becomes the default; if it misbehaves, interrupt the boot and
pick the old one.

⚠ This rewrites $BOOTDIR/extlinux/extlinux.conf. The kernel command
line already in it is carried over unchanged. root=, console= and
the rest are board-specific and are not re-invented here." || return 1
	else
		# ⚠ SAY THE PART THAT IS WORSE HERE. boot.scr loads one Image and
		# there is no menu, so unlike extlinux the old kernel cannot be
		# offered at boot. It is kept as a file, and putting it back needs
		# either a system that still boots or the card in another machine.
		# Promising a fallback that does not exist would be the worst thing
		# this screen could do.
		ui_yesno "Install this kernel?

  release   $TAG
  into      $BOOTDIR  (armbian: armbianEnv.txt + boot.scr)

The kernel now on this board is copied aside as Image.previous, and
the dtb likewise.

⚠ THIS LAYOUT HAS NO BOOT MENU. boot.scr loads one Image, so the old
kernel cannot be offered at boot the way extlinux can. If the new one
does not boot, put the old file back:

  cp $BOOTDIR/Image.previous $BOOTDIR/Image

which needs a system that still boots, or this card in another
machine. Have a way to do that before saying yes.

⚠ armbianEnv.txt is NOT touched. root=, console= and the rest stay
exactly as Armbian set them." || return 1
	fi

	if [ "$DRY" = 1 ]; then
		for u in "$IMG" "$DTB" ${MODS:+"$MODS"} ${SUMS:+"$SUMS"}; do
			would "fetch $u"
		done
		would "verify SHA256SUMS before touching $BOOTDIR"
		would "cp $BOOTDIR/Image $BOOTDIR/Image.previous   (only if it does not exist)"
		would "cp <new> $BOOTDIR/Image  and the dtb"
		would "tar -C / -xzf modules-*.tar.gz"
		if [ "$BOOTKIND" = extlinux ]; then
			would "rewrite $BOOTDIR/extlinux/extlinux.conf with two entries, default the new one"
		else
			would "leave $BOOTDIR/armbianEnv.txt alone; boot.scr picks up the new Image"
		fi
		would "depmod -a for the installed module tree"
		ui_msg "DRY RUN. The kernel step stops here.

  release  $TAG
  boot     $BOOTDIR
  append   carried over from the file already there"
		return 0
	fi
	TMP=$(mktemp -d)
	trap 'rm -rf "$TMP"' EXIT
	for u in "$IMG" "$DTB" ${MODS:+"$MODS"} ${SUMS:+"$SUMS"}; do
		ui_note "fetching $(basename "$u")"
		fetch "$u" "$TMP/$(basename "$u")" || { ui_msg "download failed: $u"; return 1; }
	done

	# ⚠ Verify before touching /boot. A truncated Image that overwrites a
	# working one is the exact failure this whole step is meant to avoid.
	if [ -n "$SUMS" ] && command -v sha256sum >/dev/null 2>&1; then
		( cd "$TMP" && sha256sum -c SHA256SUMS >/dev/null 2>&1 ) \
			|| { ui_msg "The download does not match SHA256SUMS. Nothing was written."; return 1; }
		ui_ok "checksums match"
	else
		ui_warn "no SHA256SUMS to check against"
	fi

	DTBNAME=$(basename "$DTB")
	if [ "$BOOTKIND" = extlinux ]; then
		DTBDEST="$BOOTDIR/$DTBNAME"
		APPEND=$(awk '/^[ \t]*append /{sub(/^[ \t]*append[ \t]*/,""); print; exit}' \
			"$BOOTDIR/extlinux/extlinux.conf")
		[ -n "$APPEND" ] || { ui_msg "Could not read the current kernel command line. Nothing was written."; return 1; }
	else
		# ⚠ WRITE THE DTB WHERE fdtfile ALREADY POINTS, and do not edit
		# armbianEnv.txt. boot.scr loads dtb/${fdtfile}; putting ours at a
		# name of our own choosing would load the OLD one and look like the
		# new kernel had simply failed.
		FDT=$(awk -F= '/^[ \t]*fdtfile[ \t]*=/{sub(/^[ \t]+|[ \t]+$/,"",$2); print $2; exit}' \
			"$BOOTDIR/armbianEnv.txt")
		if [ -n "$FDT" ]; then
			DTBDEST="$BOOTDIR/dtb/$FDT"
		elif [ -d "$BOOTDIR/dtb/rockchip" ]; then
			DTBDEST="$BOOTDIR/dtb/rockchip/$DTBNAME"
		else
			DTBDEST="$BOOTDIR/dtb/$DTBNAME"
		fi
		[ -d "$(dirname "$DTBDEST")" ] || {
			ui_msg "$(dirname "$DTBDEST") does not exist, so this is not the
Armbian layout after all. Nothing was written."
			return 1
		}
		APPEND=""
	fi

	# ⚠ Do not clobber a good backup with a bad one. If .previous already
	# exists, the kernel currently in place may itself be one of ours from a
	# previous run, so keep the ORIGINAL as the fallback.
	if [ ! -f "$BOOTDIR/Image.previous" ] && [ -e "$BOOTDIR/Image" ]; then
		# ⚠ `cp` FOLLOWS THE SYMLINK ON PURPOSE. On Armbian /boot/Image is
		# usually a link to vmlinuz-<version>, and copying the link itself
		# would leave a fallback that points at whatever replaces the
		# target later. -L takes the bytes.
		as_root cp -L "$BOOTDIR/Image" "$BOOTDIR/Image.previous"
		[ -e "$DTBDEST" ] && as_root cp -L "$DTBDEST" "$DTBDEST.previous"
		ui_ok "kept the current kernel as Image.previous"
	else
		ui_info "Image.previous already exists and was left as it is"
	fi

	# ⚠ REMOVE BEFORE COPYING. Writing through a symlink would overwrite
	# whatever it points at, which on Armbian is the distro's own
	# vmlinuz-<version> and is not ours to replace.
	TAGREL=$(strings "$TMP/Image" 2>/dev/null | sed -n 's/^Linux version \([^ ]*\).*/\1/p' | head -1)
	as_root rm -f "$BOOTDIR/Image"
	as_root cp "$TMP/Image" "$BOOTDIR/Image"
	as_root rm -f "$DTBDEST"
	as_root cp "$TMP/$DTBNAME" "$DTBDEST"
	if [ -n "$MODS" ]; then
		# ⚠⚠ NEVER `tar -C /`. Armbian is merged-usr: /lib is a SYMLINK to
		# usr/lib, and GNU tar, extracting a `lib/` directory member over
		# it, deletes the symlink and makes a real directory. Every
		# dynamically linked program then fails to exec -- the loader is
		# reached as /lib/ld-linux-aarch64.so.1 -- init included, and the
		# next boot panics in run-init. That is what happened on 2026-09-02
		# with a modules tarball rooted at lib/. Extract aside, find the
		# release directory wherever the tarball put it, and copy it
		# THROUGH the existing /lib/modules path, whatever /lib is.
		as_root rm -rf "$TMP/mods" && as_root mkdir -p "$TMP/mods"
		as_root tar -C "$TMP/mods" -xzf "$TMP/$(basename "$MODS")"
		MODDIR=$(find "$TMP/mods" -maxdepth 4 -type d -path "*modules/$TAGREL" 2>/dev/null | head -1)
		[ -n "$MODDIR" ] || MODDIR=$(find "$TMP/mods" -maxdepth 4 -type d -path "*modules/*" 2>/dev/null | head -1)
		if [ -z "$MODDIR" ]; then
			ui_msg "The modules tarball has no modules/<release> directory in it. Image and dtb are installed; modules were not."
		else
			as_root mkdir -p /lib/modules
			as_root cp -a "$MODDIR" "/lib/modules/$(basename "$MODDIR")"
		fi
		# ⚠ MODULES WITHOUT depmod ARE MODULES NOBODY CAN LOAD, and neither
		# layout was running it. The version is whatever the tarball says,
		# not `uname -r`: the running kernel is still the old one.
		KVER=$(tar tzf "$TMP/$(basename "$MODS")" \
			| awk -F/ '/^lib\/modules\//{print $3; exit}')
		[ -n "$KVER" ] && as_root depmod -a "$KVER" 2>/dev/null || true
		ui_ok "modules installed${KVER:+ for $KVER}"
	fi

	if [ "$BOOTKIND" = extlinux ]; then
		CONF=$(mktemp)
		{
			echo "default charsiu"
			echo "prompt 1"
			# tenths of a second. Long enough to catch it on a serial
			# console, which is how these boards are usually watched.
			echo "timeout 50"
			echo "menu title  which kernel?"
			echo ""
			echo "label charsiu"
			echo "    menu label charsiu NPU kernel ($TAG)"
			echo "    kernel /Image"
			echo "    fdt /$DTBNAME"
			echo "    append $APPEND"
			if [ -f "$BOOTDIR/Image.previous" ]; then
				echo ""
				echo "label previous"
				echo "    menu label the kernel that was here before"
				echo "    kernel /Image.previous"
				if [ -f "$BOOTDIR/$DTBNAME.previous" ]; then
					echo "    fdt /$DTBNAME.previous"
				else
					echo "    fdt /$DTBNAME"
				fi
				echo "    append $APPEND"
			fi
		} > "$CONF"
		as_root cp "$BOOTDIR/extlinux/extlinux.conf" "$BOOTDIR/extlinux/extlinux.conf.bak" 2>/dev/null || true
		as_root cp "$CONF" "$BOOTDIR/extlinux/extlinux.conf"
		rm -f "$CONF"
		sync
	fi

	if [ "$BOOTKIND" = extlinux ]; then
		ui_msg "Kernel installed.

  default    charsiu NPU kernel ($TAG)
  fallback   the kernel that was here before
  menu       5 seconds at boot, on the serial console

Reboot, then run this again to finish the userspace."
	else
		ui_msg "Kernel installed.

  kernel     $BOOTDIR/Image        ($TAG)
  dtb        $DTBDEST
  fallback   $BOOTDIR/Image.previous

⚠ There is no boot menu on this layout. If it does not come back,
put the old kernel back with

  cp $BOOTDIR/Image.previous $BOOTDIR/Image

Reboot, then run this again to finish the userspace."
	fi
	return 0
}

if [ "$NPU_OK" = 1 ] && [ "$DOKERNEL" = only ] && [ -n "${CHARSIU_KERNEL_TAG:-}" ]; then
	# a kernel that already works is being REPLACED on purpose, by tag
	ui_ok "$ACCEL is here; CHARSIU_KERNEL_TAG=$CHARSIU_KERNEL_TAG asks for a specific kernel anyway"
	install_kernel && exit 0
	exit 1
elif [ "$NPU_OK" = 1 ]; then
	ui_ok "$ACCEL is here, so this kernel already drives the NPU"
	[ "$DOKERNEL" = only ] && { ui_msg "Nothing to do: the NPU already works."; exit 0; }
elif [ "$DOKERNEL" = no ]; then
	ui_warn "no NPU and --no-kernel was given; charsiu will fall back to the CPU"
else
	HAVE_DT=no
	[ -d /proc/device-tree ] && grep -rlq 'rknn-core' /proc/device-tree 2>/dev/null && HAVE_DT=yes
	if [ "$HAVE_DT" = yes ]; then
		WHY="The device tree HAS an rknn-core node, so the dtb is fine and
the running kernel simply has no driver bound to it."
	else
		WHY="There is no rknn-core node in the device tree either, so this
dtb does not describe the NPU at all."
	fi
	ui_yesno "$ACCEL is missing, so this kernel cannot drive the NPU.

$WHY

RK3576 NPU support is not upstream yet: the patches are ours and are
still under review on the kernel lists. There is no distribution
kernel anywhere that will bind this hardware.

Fetch a prebuilt one from $REPO?
The kernel now on this board is kept either way; whether it stays
SELECTABLE at boot depends on how this board boots, and the next
screen says which it is." \
	&& { install_kernel && exit 0; }
	[ "$DOKERNEL" = only ] && exit 0
	ui_warn "continuing without the NPU; charsiu will run on the CPU"
fi

# ---------------------------------------------------------------------------
# USERSPACE
# ---------------------------------------------------------------------------
if [ "$DOBUILD" = 1 ]; then
	# ⚠ A DRY RUN MUST NOT STOP AT A MISSING TOOL. Finding out what is absent
	# is most of the reason to rehearse. Dying on the first gap shows one
	# problem where the run could have shown all of them.
	miss=""
	command -v make >/dev/null 2>&1 || miss="$miss make"
	{ command -v cc || command -v gcc; } >/dev/null 2>&1 || miss="$miss a-C-compiler"
	[ -f "$SRC/Makefile" ] || miss="$miss the-charsiu-source"
	if [ -n "$miss" ]; then
		# ⚠ A STOCK DEBIAN HAS NO COMPILER. This is not an edge case, it is
		# what every `curl ... | sh` into a fresh install hits, and stopping
		# here left the reader to work out the package name themselves.
		# Offer it the same way whiptail is offered at the bootstrap.
		_pm=""; _pmname=""
		case "$miss" in
		*make*|*C-compiler*)
			if   command -v apt-get >/dev/null 2>&1; then
				_pm="apt-get install -y --no-install-recommends build-essential"; _pmname="build-essential"
			elif command -v dnf >/dev/null 2>&1; then
				_pm="dnf install -y gcc make"; _pmname="gcc and make"
			elif command -v apk >/dev/null 2>&1; then
				_pm="apk add build-base"; _pmname="build-base"
			elif command -v pacman >/dev/null 2>&1; then
				_pm="pacman -S --noconfirm base-devel"; _pmname="base-devel"
			fi ;;
		esac
		if [ "$DRY" = 1 ]; then
			ui_bad "missing:$miss"
			if [ -n "$_pm" ]; then would "$_pm   (to supply$miss)"
			else ui_bad "  and no package manager here knows how to supply it"; fi
			DOBUILD=0
		elif [ -n "$_pm" ] && ui_yesno "charsiu is built from source, and this machine has no compiler.

Missing:$miss

Install it now?
  $_pm"; then
			ui_note "installing $_pmname..."
			command -v apt-get >/dev/null 2>&1 && \
				as_root apt-get update -qq >/dev/null 2>&1 || true
			# ⚠ SWALLOWING APT'S OUTPUT LEAVES "it failed" AND NOTHING TO ACT
			# ON. It fails for ordinary reasons (no network, a held
			# package, another apt holding the lock) and each one has a
			# different fix, so keep the last lines and show them.
			_aptlog="${TMPDIR:-/tmp}/charsiu-apt.$$"
			if ! as_root $_pm >"$_aptlog" 2>&1; then
				printf '\n%s\n' "$(tail -n 6 "$_aptlog")" >&2
				rm -f "$_aptlog"
				die "installing $_pmname failed (see above). Install it by hand and run this again."
			fi
			rm -f "$_aptlog"
			miss=""
			command -v make >/dev/null 2>&1 || miss="$miss make"
			{ command -v cc || command -v gcc; } >/dev/null 2>&1 || miss="$miss a-C-compiler"
			[ -n "$miss" ] && die "still missing:$miss"
			ui_ok "$_pmname installed"
		else
			die "missing:$miss"
		fi
	fi
fi
if [ "$DOBUILD" = 1 ]; then
	if [ "$DRY" = 1 ]; then
		would "make all   (in $SRC)"
	else
		# ⚠ A BUILD IS THE LONGEST SILENT STRETCH OF THE WHOLE INSTALL, and
		# it used to print one line and then nothing for minutes. Drive a
		# gauge off the binaries as they actually appear in build/, which is
		# a real measure rather than an animation.
		BLOG="${TMPDIR:-/tmp}/charsiu-build.$$"
		TARGETS="emit_dump emit_job charsiu_run charsiu_check charsiu_serve"
		ntot=0; for t in $TARGETS; do ntot=$((ntot + 1)); done
		( cd "$SRC" && make -j"$(nproc 2>/dev/null || echo 2)" all ) \
			> "$BLOG" 2>&1 &
		_mpid=$!
		{
			while kill -0 "$_mpid" 2>/dev/null; do
				n=0
				for t in $TARGETS; do
					[ -x "$SRC/build/$t" ] && n=$((n + 1))
				done
				echo $(( n * 100 / ntot ))
				sleep 1
			done
			echo 100
		} | ui_progress "Building charsiu in $SRC

This is C, not a download: a minute or two on a board."
		if ! wait "$_mpid"; then
			# ⚠ `>/dev/null 2>&1` ON THE BUILD MEANT THE ONE MESSAGE THAT
			# COULD HAVE EXPLAINED IT WAS THROWN AWAY, and the board only
			# ever said "the build failed".
			printf '\n%s\n' "$(tail -n 12 "$BLOG")" >&2
			ui_msg "The build failed. The last lines were printed above, and
the whole log is in

  $BLOG

Run 'make all' in $SRC to see it live."
			exit 1
		fi
		rm -f "$BLOG"
		ui_ok "built"
	fi
fi
RUNBIN="$SRC/build/charsiu_run"; CHKBIN="$SRC/build/charsiu_check"
if [ ! -x "$RUNBIN" ]; then
	# ⚠ In a dry run the build did not happen, so the binary legitimately is
	# not there yet. Saying so is useful; dying is not.
	[ "$DRY" = 1 ] && ui_info "$RUNBIN is not built yet (the build was skipped)" \
		|| die "$RUNBIN does not exist."
fi

as_root mkdir -p "$BIN" "$SBIN" "$ETC" "$MODELS"

# ⚠ SPELLING THIS LIST OUT IS HOW `charsiu list` SHIPPED BROKEN. The front door
# execs one helper per subcommand, and six of them (list, ps, rm, show, runner,
# serve) arrived after the list was written. A fresh Debian install got a
# charsiu that printed --help and then died with "exec: : Permission denied" on
# every subcommand. Install whatever the source actually has.
# ⚠ bench_batch too: [debug] enable makes `charsiu bench` reach for it, and a
# setting that points at a binary nobody installed is a setting that lies.
# ⚠ npu_gemm_test as well: it is the only thing that can answer whether the
# hardware does a matmul with more than one row, which is the whole of prefill,
# and asking somebody to go find it under ~/.cache is how a board round does
# not happen.
# ⚠⚠ THE PROBES ARE A DEV THING, and installing them on somebody who asked for
# a way to run a model is how a tool stops being trusted. npu_gemm_test,
# charsiu_matmul and bench_batch exist to ask the hardware questions -- what a
# register does at a width nobody has run, whether batching pays -- and they
# have wedged the block, timed out and printed the opposite of their own data
# on the way to the answers. `charsiu update dev` asks for them.
# ⚠⚠ charsiu_vision, charsiu_clip AND charsiu_whisper ARE RUNTIME, NOT PROBES.
# `charsiu pull` offers whisper-tiny.en and clip-b32 and both are useless
# without their binary -- and the paragraph above is the record of what leaving
# a name off this list costs. They are read only, they do not touch the NPU's
# registers, and none of them has ever wedged the block; the probes above have
# done all three.
RUNTIME_BINS="charsiu_run charsiu_check charsiu_serve \
	      charsiu_vision charsiu_clip charsiu_whisper"
# ⚠ A SCRIPT WITHOUT ITS BINARY IS A ROUND THAT DOES NOT HAPPEN, and this has
# now happened twice. vattn_sweep.sh went to the board without vattn_bench and
# without itself, and came back "cannot open /opt/charsiu/vattn_sweep.sh" -- so
# six attention defaults chosen on a compute bound desktop are still the
# defaults on a bandwidth bound board. Anything added to PROBE_SCRIPTS that
# runs a binary has to add the binary here in the same edit.
PROBE_BINS="bench_batch npu_gemm_test npu_slice_test charsiu_matmul vattn_bench acc_index_check"
# ⚠ EVERY BOARD SCRIPT, NOT JUST THE FIRST ONE WRITTEN. The paragraph further
# down says a probe that lives only in the source tree under ~/.cache is a
# board round that does not happen -- and then only prefill_control.sh was
# listed, so every board_*.sh written since has been exactly that: reachable
# by a path nobody types.
PROBE_SCRIPTS="board_next.sh spec_identity.sh prefill_control.sh board_w4_axis.sh board_rows_sweep.sh \
board_acc_map.sh board_width_short.sh board_vendor.sh board_modalities.sh \
board_threads.sh board_w4_m8.sh vattn_sweep.sh vattn_edges.sh \
board_text_all.sh board_refused_onedev.sh board_chunk_sweep.sh board_intermittent.sh \
board_width_law.sh board_verify.sh verify_selftest.sh whisper_transcribe.sh"
case "$CHANNEL" in
dev) INSTALL_BINS="$RUNTIME_BINS $PROBE_BINS" ;;
*)   INSTALL_BINS="$RUNTIME_BINS" ;;
esac
for f in $INSTALL_BINS; do
	[ "$DRY" = 1 ] || [ -x "$SRC/build/$f" ] || continue
	as_root cp "$SRC/build/$f" "$BIN/$f"
done
# ⚠⚠ AND THE ONES A PERSON TYPES GO WHERE A PERSON CAN TYPE THEM. $BIN is
# /opt/charsiu and is NOT on anybody's PATH -- charsiu_run does not need to be,
# because the front door execs it by path, but charsiu_whisper and charsiu_clip
# are commands in their own right and the README tells people to run them by
# name. Installed only under /opt they are a documented command that does not
# exist.
TYPED_BINS="charsiu_check charsiu_vision charsiu_clip charsiu_whisper"
for f in $TYPED_BINS; do
	[ "$DRY" = 1 ] || [ -x "$SRC/build/$f" ] || continue
	as_root cp "$SRC/build/$f" "$SBIN/$f"
	as_root chmod 0755 "$SBIN/$f"
done
# ⚠ AND THE PROBES THAT ARE SHELL RATHER THAN C. prefill_control.sh is a probe
# by everything that matters -- it asks the hardware a question the runtime
# cannot answer about itself, and it is dev only for the same reason the other
# three are. It goes next to them, because the paragraph above is right: a
# probe that lives only in the source tree under ~/.cache is a board round that
# does not happen, and `charsiu update dev` is how this board tests.
case "$CHANNEL" in
dev)	for f in $PROBE_SCRIPTS; do
		[ "$DRY" = 1 ] || [ -r "$SRC/tests/$f" ] || continue
		as_root cp "$SRC/tests/$f" "$BIN/$f"
		as_root chmod 0755 "$BIN/$f"
	done ;;
esac
# ⚠ BOTH LIBRARIES, OR EVERY COMMAND EXITS ON THE FIRST LINE. charsiu-lib.sh
# is where ini_get and find_bin live now; a script that cannot source it
# says so and stops rather than guessing.
for f in charsiu-tui.sh charsiu-lib.sh; do
	as_root cp "$SRC/scripts/$f" "$BIN/$f"
done
for p in "$SRC"/scripts/charsiu "$SRC"/scripts/charsiu-*; do
	f=${p##*/}
	# the installer is not a subcommand, and the TUI layer is a library
	case "$f" in charsiu-install.sh|charsiu-tui.sh|charsiu-lib.sh) continue ;; esac
	as_root cp "$p" "$SBIN/$f"
	as_root chmod 0755 "$SBIN/$f"
done

# ⚠ THE MODELS DIRECTORY MUST BELONG TO WHOEVER WILL FILL IT. Installed under
# sudo it lands root-owned, and then charsiu-get, which nobody should have to
# run as root to download a file, fails at the last step, after the download.
OWNER="${SUDO_USER:-$(id -un)}"
if [ "$OWNER" != root ] && id "$OWNER" >/dev/null 2>&1; then
	# ⚠ as_root RETURNS 0 IN A DRY RUN, so a `&& ui_ok "..."` here announced
	# a chown that never happened. A rehearsal that claims work it did not do
	# is worse than no rehearsal.
	# ⚠ `2>/dev/null` on the as_root call SWALLOWS the dry run's own notice,
	# which goes to stderr. The action then appeared in the final summary
	# but not in the live output. Split the two cases.
	if [ "$DRY" = 1 ]; then
		as_root chown -R "$OWNER" "$MODELS"
	elif as_root chown -R "$OWNER" "$MODELS" 2>/dev/null; then
		ui_ok "$MODELS belongs to $OWNER, so charsiu-get needs no sudo"
	fi
fi

if [ -f "$ETC/config.ini" ]; then
	as_root cp "$SRC/etc/config.ini" "$ETC/config.ini.default"
	[ "$DRY" = 0 ] && ui_info "your $ETC/config.ini was left alone (template: config.ini.default)"
else
	as_root cp "$SRC/etc/config.ini" "$ETC/config.ini"
	# ⚠ so that a later plain `charsiu update` stays on the channel that
	# was installed rather than quietly going back to stable.
	[ "$CHANNEL" = stable ] || as_root sh -c \
		"sed -i 's/^channel = .*/channel = $CHANNEL/' '$ETC/config.ini'"
fi
# ⚠ THE CONFIG A USER OWNS HAS TO BE WRITABLE BY THAT USER. /etc/charsiu is
# root's, so charsiu-get could not record the model it had just downloaded and
# every later run went looking for the placeholder in the shipped template. The
# front door already prefers ~/.charsiu/config.ini, so put one there, pointing
# at the models directory that user will actually fill.
if [ "$DRY" = 0 ] && [ -z "$PREFIX" ] && [ "$(id -u)" -ne 0 ]; then
	UCONF="${CHARSIU_HOME:-$HOME/.charsiu}/config.ini"
	if [ ! -f "$UCONF" ]; then
		mkdir -p "$(dirname "$UCONF")" "$MODELS"
		sed "s|^path *=.*|path = $MODELS/$(basename "$(sed -n 's|^path *= *||p' "$SRC/etc/config.ini" | head -1)")|" \
			"$SRC/etc/config.ini" > "$UCONF"
		ui_ok "your config is $UCONF, and your models are $MODELS"
	fi
	CONFDISP="$UCONF"
else
	CONFDISP="$ETC/config.ini"
fi
[ "$DRY" = 0 ] && ui_ok "installed into $BIN and $SBIN"

# ---------------------------------------------------------------------------
if [ "$DOMODEL" = 1 ] && [ -z "$(ls "$MODELS"/*.gguf 2>/dev/null || true)" ]; then
	if [ "$DRY" = 1 ]; then
		would "charsiu-get --wizard   (pick and download a model into $MODELS)"
	else
		# ⚠ charsiu-get RECORDS WHAT IT FETCHED, so it has to be told which
		# config is in effect. Left to guess it looked at ~/.charsiu and then
		# /etc, found neither in a staged install, and the download went
		# unrecorded, which is the bug that shipped a working install unable
		# to name its own model.
		CHARSIU_MODELS_DIR="$MODELS" CHARSIU_CHECK="$BIN/charsiu_check" \
			CHARSIU_LIB="$BIN" \
			CHARSIU_CONFIG="${CONFDISP:-$ETC/config.ini}" \
			"$SBIN/charsiu-get" --wizard || true
	fi
fi

ui_hdr "checking"
if [ "$DRY" = 1 ]; then
	# ⚠ the doctor is READ-ONLY, so a dry run should still run it. What it
	# reports is the most useful thing this rehearsal produces. It is pointed
	# at the SOURCE tree's tools, since nothing was installed.
	CHARSIU_CONFIG="$SRC/etc/config.ini" CHARSIU_LIB="$SRC/scripts" \
		"$SRC/scripts/charsiu-doctor" || true
else
	# ⚠ THE LONGEST STRETCH OF LOOSE TEXT IN THE WHOLE INSTALL. On a serial
	# console it scrolls past between two dialogs and reads as the wizard
	# having given up. Keep printing it, so it is in the log and in a pipe,
	# and also put it on screen as something you can read and scroll.
	DLOG="${TMPDIR:-/tmp}/charsiu-doctor.$$"
	CHARSIU_CONFIG="${CONFDISP:-$ETC/config.ini}" CHARSIU_LIB="$BIN" \
		"$SBIN/charsiu-doctor" 2>&1 | tee "$DLOG" >&2 || true
	ui_pane "$DLOG"
	rm -f "$DLOG"
fi

# ⚠ A REPORT IS NOT A DEMONSTRATION. Ending on a list of ticks leaves someone
# who has waited through a build and a download with no evidence the thing
# talks. One sentence is cheap and it is the whole point of installing it.
if [ "$DRY" = 1 ]; then
	would "charsiu -p 'The capital of France is' -n 32   (one sentence, to prove it works)"
elif [ -n "$(ls "$MODELS"/*.gguf 2>/dev/null || true)" ] && [ "$NPU_OK" = 1 ]; then
	if ui_yesno "Ask it something, to see it work?

The first run stages the NPU tensors, which takes about twenty
seconds before the first word." ; then
		ui_hdr "asking: the capital of France is"
		CHARSIU_CONFIG="${CONFDISP:-$ETC/config.ini}" CHARSIU_LIB="$BIN" \
			"$SBIN/charsiu" -p "The capital of France is" -n 32 -q || true
		printf '\n'
	fi
fi

if [ "$DRY" = 1 ]; then
	printf '\n%sDRY RUN. Nothing above was done. In order, it would have:%s\n%s\n\n' \
	       "$T_B" "$T_0" "$DRYLOG"
	ui_msg "Dry run finished. Nothing was written.

Run it again without --dry-run to do it for real."
	# Deleting the tree we are running from is fine: the shell already has
	# the file open, and this is the only way a rehearsal leaves nothing.
	[ -n "${CHARSIU_DRY_SRC:-}" ] && rm -rf "$CHARSIU_DRY_SRC"
	exit 0
fi

# ⚠ The closing screen still named the old scripts one by one, from before the
# front door existed. Say the commands a user will actually type.
ui_msg "Done.

  charsiu             a conversation
  charsiu \"...\"       ask once and exit
  charsiu list        what is here
  charsiu pull        fetch another model
  charsiu doctor      what works and what does not
  charsiu serve       an OpenAI compatible endpoint on :11434

  This build can also SEE. `charsiu pull` lists the models that take a
  picture, and they are two files -- it fetches both:

  charsiu --image photo.jpg \"what is in this picture?\" 

  channel $CHANNEL$([ "$CHANNEL" = stable ] && echo "      charsiu update dev  adds the hardware probes" || echo "         charsiu update stable  goes back to the runtime alone")$([ "$CHANNEL" = dev ] && echo "
  probes  $BIN: $PROBE_BINS $PROBE_SCRIPTS")

  config  ${CONFDISP:-$ETC/config.ini}
  models  $MODELS"
}
