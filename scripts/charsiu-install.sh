#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# charsiu-install.sh -- set up charsiu on a Rockchip RK3576 board.
#
#   sh charsiu-install.sh              the wizard
#   sh charsiu-install.sh --kernel     only offer the kernel step
#   sh charsiu-install.sh --no-kernel  skip it
#   sh charsiu-install.sh --dry-run    say what it would do, change nothing
#   sh charsiu-install.sh --prefix DIR stage instead of installing
#   sh charsiu-install.sh --uninstall  remove what this installed
#   CHARSIU_PLAIN=1 ...                no full-screen dialogs
#
# ⚠⚠ RK3576 NPU SUPPORT IS NOT UPSTREAM, SO NO STOCK KERNEL CAN RUN THIS.
#
# The rocket driver is mainline for RK3588. The commit adding
# `rockchip,rk3576-rknn-core` is ours, from 2026-08-06, and is not reachable
# from any tag. An earlier version of this script checked for a working NPU and
# stopped when it did not find one -- which was every machine on earth, so the
# check was a dead end wearing a helpful expression. It now OFFERS A KERNEL:
# CI builds linux-next plus the v9 series and publishes it, and this fetches it.
#
# ⚠ AND IT KEEPS THE ONE ALREADY THERE. The new kernel becomes the default boot
# entry and the previous one stays on the card as a second entry, because a
# kernel that does not boot is not a thing to discover with no way back.
set -eu

REPO="${CHARSIU_REPO:-gahingwoo/linux-rk3576-npu}"
PREFIX="/"
DRY=0
DOKERNEL=ask
DOMODEL=1
DOBUILD=1
UNINSTALL=0

while [ $# -gt 0 ]; do
	case "$1" in
	--prefix)    PREFIX="$2"; shift 2 ;;
	--dry-run|-n) DRY=1; shift ;;
	--kernel)    DOKERNEL=only; shift ;;
	--no-kernel) DOKERNEL=no; shift ;;
	--no-model)  DOMODEL=0; shift ;;
	--no-build)  DOBUILD=0; shift ;;
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
# "/" trimmed to "" is exactly right -- "$PREFIX/opt/..." then gives /opt/...
# rather than //opt/... . For SHOWING it, the empty string reads as a blank.
PREFIX=$(printf '%s' "$PREFIX" | sed 's|/*$||')
PDISP="${PREFIX:-/}"
BIN="$PREFIX/opt/charsiu"
SBIN="$PREFIX/usr/bin"
ETC="$PREFIX/etc/charsiu"
MODELS="$BIN/models"

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
A kernel this installed is NOT removed either -- pick the previous
entry in the boot menu instead, then remove it by hand." defaultno || exit 0
	for f in charsiu charsiu-get charsiu-config charsiu-doctor; do
		as_root rm -f "$SBIN/$f"
	done
	for f in charsiu_run charsiu_check charsiu-tui.sh; do as_root rm -f "$BIN/$f"; done
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

Read-only checks still run for real -- that is the point: this is
what the installer SEES on this machine."
fi

if [ "$DOKERNEL" != only ]; then
	ui_msg "charsiu -- an open LLM runtime for the RK3576 NPU

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
	# somewhere else -- and for rehearsing this on a machine that has none.
	BOOTDIR=""
	for b in ${CHARSIU_BOOTDIR:+"$CHARSIU_BOOTDIR"} /boot /boot/firmware /mnt/boot; do
		[ -f "$b/extlinux/extlinux.conf" ] && { BOOTDIR="$b"; break; }
	done
	if [ -z "$BOOTDIR" ]; then
		ui_msg "No extlinux.conf found under /boot.

This board boots some other way (a boot.scr, a distro's own kernel
package, or U-Boot environment variables), and guessing at it would
be worse than doing nothing.

Install a kernel built from the series by hand instead:
  https://github.com/$REPO  (rfc-send-v9/, kernel/base.config)"
		return 1
	fi

	ui_note "asking $REPO for the latest kernel..."
	J=$(api "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null || true)
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

	ui_yesno "Install this kernel?

  release   $TAG
  into      $BOOTDIR

The kernel now on this board is kept as a SECOND boot entry. The new
one becomes the default; if it misbehaves, interrupt the boot and
pick the old one.

⚠ This rewrites $BOOTDIR/extlinux/extlinux.conf. The kernel command
line already in it is carried over unchanged -- root=, console= and
the rest are board-specific and are not re-invented here." || return 1

	if [ "$DRY" = 1 ]; then
		for u in "$IMG" "$DTB" ${MODS:+"$MODS"} ${SUMS:+"$SUMS"}; do
			would "fetch $u"
		done
		would "verify SHA256SUMS before touching $BOOTDIR"
		would "cp $BOOTDIR/Image $BOOTDIR/Image.previous   (only if it does not exist)"
		would "cp <new> $BOOTDIR/Image  and the dtb"
		would "tar -C / -xzf modules-*.tar.gz"
		would "rewrite $BOOTDIR/extlinux/extlinux.conf with two entries, default the new one"
		ui_msg "DRY RUN -- the kernel step stops here.

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
	APPEND=$(awk '/^[ \t]*append /{sub(/^[ \t]*append[ \t]*/,""); print; exit}' \
		"$BOOTDIR/extlinux/extlinux.conf")
	[ -n "$APPEND" ] || { ui_msg "Could not read the current kernel command line. Nothing was written."; return 1; }

	# ⚠ Do not clobber a good backup with a bad one. If .previous already
	# exists, the kernel currently in place may itself be one of ours from a
	# previous run -- keep the ORIGINAL as the fallback.
	if [ ! -f "$BOOTDIR/Image.previous" ] && [ -f "$BOOTDIR/Image" ]; then
		as_root cp "$BOOTDIR/Image" "$BOOTDIR/Image.previous"
		[ -f "$BOOTDIR/$DTBNAME" ] && \
			as_root cp "$BOOTDIR/$DTBNAME" "$BOOTDIR/$DTBNAME.previous"
		ui_ok "kept the current kernel as Image.previous"
	else
		ui_info "Image.previous already exists and was left as it is"
	fi

	as_root cp "$TMP/Image" "$BOOTDIR/Image"
	as_root cp "$TMP/$DTBNAME" "$BOOTDIR/$DTBNAME"
	if [ -n "$MODS" ]; then
		as_root tar -C / -xzf "$TMP/$(basename "$MODS")"
		ui_ok "modules installed"
	fi

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

	ui_msg "Kernel installed.

  default    charsiu NPU kernel ($TAG)
  fallback   the kernel that was here before
  menu       5 seconds at boot, on the serial console

Reboot, then run this again to finish the userspace."
	return 0
}

if [ "$NPU_OK" = 1 ]; then
	ui_ok "$ACCEL is here -- this kernel already drives the NPU"
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
	ui_yesno "$ACCEL is missing -- this kernel cannot drive the NPU.

$WHY

RK3576 NPU support is not upstream yet: the patches are ours and are
still under review on the kernel lists. There is no distribution
kernel anywhere that will bind this hardware.

Fetch a prebuilt one from $REPO?
The kernel now on this board is kept and stays selectable at boot." \
	&& { install_kernel && exit 0; }
	[ "$DOKERNEL" = only ] && exit 0
	ui_warn "continuing without the NPU; charsiu will run on the CPU"
fi

# ---------------------------------------------------------------------------
# USERSPACE
# ---------------------------------------------------------------------------
if [ "$DOBUILD" = 1 ]; then
	# ⚠ A DRY RUN MUST NOT STOP AT A MISSING TOOL. Finding out what is absent
	# is most of the reason to rehearse -- dying on the first gap shows one
	# problem where the run could have shown all of them.
	miss=""
	command -v make >/dev/null 2>&1 || miss="$miss make"
	{ command -v cc || command -v gcc; } >/dev/null 2>&1 || miss="$miss a-C-compiler"
	[ -f "$SRC/Makefile" ] || miss="$miss the-charsiu-source"
	if [ -n "$miss" ]; then
		if [ "$DRY" = 1 ]; then
			ui_bad "missing:$miss  -- a real run would stop here"
			DOBUILD=0
		else
			die "missing:$miss"
		fi
	fi
fi
if [ "$DOBUILD" = 1 ]; then
	if [ "$DRY" = 1 ]; then
		would "make all   (in $SRC)"
	else
		ui_note "building charsiu..."
		( cd "$SRC" && make all ) >/dev/null 2>&1 \
			|| die "the build failed. Run 'make all' in $SRC to see why."
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
as_root cp "$RUNBIN" "$BIN/charsiu_run"
as_root cp "$CHKBIN" "$BIN/charsiu_check"
as_root cp "$SRC/scripts/charsiu-tui.sh" "$BIN/charsiu-tui.sh"
for f in charsiu charsiu-get charsiu-config charsiu-doctor; do
	as_root cp "$SRC/scripts/$f" "$SBIN/$f"
	as_root chmod 0755 "$SBIN/$f"
done

# ⚠ THE MODELS DIRECTORY MUST BELONG TO WHOEVER WILL FILL IT. Installed under
# sudo it lands root-owned, and then charsiu-get -- which nobody should have to
# run as root to download a file -- fails at the last step, after the download.
OWNER="${SUDO_USER:-$(id -un)}"
if [ "$OWNER" != root ] && id "$OWNER" >/dev/null 2>&1; then
	# ⚠ as_root RETURNS 0 IN A DRY RUN, so a `&& ui_ok "..."` here announced
	# a chown that never happened. A rehearsal that claims work it did not do
	# is worse than no rehearsal.
	# ⚠ `2>/dev/null` on the as_root call SWALLOWS the dry run's own notice,
	# which goes to stderr -- the action then appeared in the final summary
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
fi
[ "$DRY" = 0 ] && ui_ok "installed into $BIN and $SBIN"

# ---------------------------------------------------------------------------
if [ "$DOMODEL" = 1 ] && [ -z "$(ls "$MODELS"/*.gguf 2>/dev/null || true)" ]; then
	if [ "$DRY" = 1 ]; then
		would "charsiu-get --wizard   (pick and download a model into $MODELS)"
	else
		CHARSIU_MODELS_DIR="$MODELS" CHARSIU_CHECK="$BIN/charsiu_check" \
			CHARSIU_LIB="$BIN" "$SBIN/charsiu-get" --wizard || true
	fi
fi

ui_hdr "checking"
if [ "$DRY" = 1 ]; then
	# ⚠ the doctor is READ-ONLY, so a dry run should still run it -- what it
	# reports is the most useful thing this rehearsal produces. It is pointed
	# at the SOURCE tree's tools, since nothing was installed.
	CHARSIU_CONFIG="$SRC/etc/config.ini" CHARSIU_LIB="$SRC/scripts" \
		"$SRC/scripts/charsiu-doctor" || true
else
	CHARSIU_CONFIG="$ETC/config.ini" "$SBIN/charsiu-doctor" || true
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
		CHARSIU_CONFIG="$ETC/config.ini" CHARSIU_LIB="$BIN" \
			"$SBIN/charsiu" -p "The capital of France is" -n 32 -q || true
		printf '\n'
	fi
fi

if [ "$DRY" = 1 ]; then
	printf '\n%sDRY RUN -- nothing above was done. In order, it would have:%s\n%s\n\n' \
	       "$T_B" "$T_0" "$DRYLOG"
	ui_msg "Dry run finished. Nothing was written.

Run it again without --dry-run to do it for real."
	exit 0
fi

ui_msg "Done.

  charsiu-config     pick a model, threads, context
  charsiu -p \"...\"    run it
  charsiu-doctor     what works and what does not
  charsiu-get        more models

  config  $ETC/config.ini
  models  $MODELS"
