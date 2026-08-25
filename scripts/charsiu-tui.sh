# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# charsiu-tui.sh: the dialog layer. Sourced, not run.
#
# whiptail when it is there, plain prompts when it is not. Both paths are real:
# the board image ships whiptail (newt, 1.8 MB), Debian and Armbian have it, and
# the plain path is what runs when someone pipes this into a script or runs it
# over a link too dumb for a full-screen redraw.
#
# ⚠ EVERY FUNCTION HERE WRITES ITS PROMPT TO STDERR AND ITS ANSWER TO STDOUT,
# so `x=$(ui_input ...)` captures the answer and not the question. whiptail
# needs the 3>&1 1>&2 2>&3 dance for the same reason and gets it here once.
#
# ⚠ A CANCELLED DIALOG IS NOT AN EMPTY ANSWER. ui_input and ui_menu return
# non-zero when the user backs out, so a caller can tell "they chose nothing"
# from "they left". Check the exit status, not the string.

CTUI=plain
command -v whiptail >/dev/null 2>&1 && CTUI=whiptail
[ -t 0 ] && [ -t 2 ] || CTUI=plain          # no terminal, no full screen
# ⚠ whiptail REFUSES to run without TERM and prints "TERM environment variable
# needs set.", which a serial console often is. Measured: without this guard
# every dialog fails and, worse, the error text arrives where the answer should
# be (see the fd note below). Fall back rather than fail.
case "${TERM:-}" in ""|dumb|unknown) CTUI=plain ;; esac
[ -n "${CHARSIU_PLAIN:-}" ] && CTUI=plain

CTUI_TITLE="charsiu"

# ⚠ CTUI_ASSUME makes every question answer itself, without a terminal. That is
# what a rehearsal piped into a container needs: a dry run writes nothing, so
# there is nothing to consent to, and refusing to run for want of a tty would
# be refusing to do the one thing that was asked.
#   CTUI_ASSUME=yes   take the affirmative
#   CTUI_ASSUME=no    take the negative
CTUI_ASSUME="${CTUI_ASSUME:-}"

if [ -t 2 ]; then
	T_B=$(printf '\033[1m'); T_G=$(printf '\033[1;32m'); T_R=$(printf '\033[1;31m')
	T_Y=$(printf '\033[1;33m'); T_D=$(printf '\033[2m'); T_0=$(printf '\033[0m')
else T_B=; T_G=; T_R=; T_Y=; T_D=; T_0=; fi

# ui_msg TEXT           say something and wait for acknowledgement
ui_msg() {
	if [ -n "$CTUI_ASSUME" ]; then printf '\n%s\n' "$1" >&2; return 0; fi
	if [ "$CTUI" = whiptail ]; then
		whiptail --title "$CTUI_TITLE" --msgbox "$1" 20 74
	else
		printf '\n%s\n\n%spress enter%s ' "$1" "$T_D" "$T_0" >&2
		read -r _ || true
	fi
}

# ui_note TEXT          say something and keep going
ui_note() {
	if [ "$CTUI" = whiptail ]; then
		whiptail --title "$CTUI_TITLE" --infobox "$1" 12 74
		sleep 1
	else
		printf '\n%s\n' "$1" >&2
	fi
}

# ui_yesno TEXT [defaultno]  returns 0 for yes
ui_yesno() {
	if [ -n "$CTUI_ASSUME" ]; then
		printf '\n%s\n  [assuming %s]\n' "$1" "$CTUI_ASSUME" >&2
		[ "$CTUI_ASSUME" = yes ] && return 0 || return 1
	fi
	if [ "$CTUI" = whiptail ]; then
		if [ "${2:-}" = defaultno ]; then
			whiptail --title "$CTUI_TITLE" --defaultno --yesno "$1" 20 74
		else
			whiptail --title "$CTUI_TITLE" --yesno "$1" 20 74
		fi
	else
		if [ "${2:-}" = defaultno ]; then d="y/N"; else d="Y/n"; fi
		printf '\n%s\n\n  [%s] ' "$1" "$d" >&2
		read -r a || a=""
		case "$a" in
		y|Y|yes|YES) return 0 ;;
		n|N|no|NO)   return 1 ;;
		"")          [ "${2:-}" = defaultno ] && return 1 || return 0 ;;
		*)           return 1 ;;
		esac
	fi
}

# ui_input PROMPT DEFAULT   the answer on stdout; non-zero if cancelled
#
# ⚠⚠ THE fd DANCE PUTS whiptail's ERRORS WHERE THE ANSWER GOES. 3>&1 1>&2 2>&3
# swaps stdout and stderr so the selection comes back on stdout, and so does
# any diagnostic whiptail decides to print. Measured: with TERM unset the caller
# received the string "TERM environment variable needs set." as the user's
# answer. Capture first, echo only if whiptail actually succeeded.
ui_input() {
	if [ -n "$CTUI_ASSUME" ]; then echo "$2"; return 0; fi
	if [ "$CTUI" = whiptail ]; then
		_o=$(whiptail --title "$CTUI_TITLE" --inputbox "$1" 12 74 "$2" 3>&1 1>&2 2>&3) \
			|| return 1
		echo "$_o"
	else
		printf '\n%s\n  [%s]: ' "$1" "$2" >&2
		read -r a || return 1
		[ -n "$a" ] && echo "$a" || echo "$2"
	fi
}

# ui_menu TEXT  tag1 desc1  tag2 desc2 ...   the chosen tag on stdout
ui_menu() {
	text="$1"; shift
	# ⚠ assuming an ANSWER to a menu is not possible, so it declines instead
	# of guessing which entry someone meant.
	if [ -n "$CTUI_ASSUME" ]; then printf '\n%s\n  [skipped]\n' "$text" >&2; return 1; fi
	if [ "$CTUI" = whiptail ]; then
		n=$(( $# / 2 ))
		[ "$n" -gt 12 ] && n=12
		# same trap as ui_input: capture, then decide.
		_o=$(whiptail --title "$CTUI_TITLE" --menu "$text" $((n + 9)) 76 "$n" \
			"$@" 3>&1 1>&2 2>&3) || return 1
		echo "$_o"
	else
		printf '\n%s\n\n' "$text" >&2
		i=0
		# ⚠ "$@" is consumed as we walk it, so the tags are stashed in
		# positional slots that survive the loop.
		set -- "$@"
		saved=""
		while [ $# -ge 2 ]; do
			i=$((i + 1))
			printf '  %2d) %-22s %s\n' "$i" "$1" "$2" >&2
			saved="$saved $1"
			shift 2
		done
		printf '\n  > ' >&2
		read -r c || return 1
		case "$c" in
		''|*[!0-9]*) return 1 ;;
		esac
		[ "$c" -ge 1 ] && [ "$c" -le "$i" ] || return 1
		# shellcheck disable=SC2086
		set -- $saved
		eval "echo \${$c}"
	fi
}

# ui_progress TEXT  run the rest of a pipeline under a gauge. Reads
# percentages on stdin, one integer per line.
ui_progress() {
	if [ "$CTUI" = whiptail ]; then
		whiptail --title "$CTUI_TITLE" --gauge "$1" 8 74 0
	else
		printf '\n%s\n' "$1" >&2
		while read -r pct; do printf '\r  %3s%%' "$pct" >&2; done
		printf '\n' >&2
	fi
}

# ui_ok / ui_bad / ui_warn: one line of report, for the plain path and for
# anything a dialog would be too heavy for.
ui_ok()   { printf '  %s[ OK ]%s %s\n' "$T_G" "$T_0" "$*" >&2; }
ui_bad()  { printf '  %s[FAIL]%s %s\n' "$T_R" "$T_0" "$*" >&2; }
ui_warn() { printf '  %s[WARN]%s %s\n' "$T_Y" "$T_0" "$*" >&2; }
ui_info() { printf '  %s[INFO]%s %s\n' "$T_D" "$T_0" "$*" >&2; }
ui_hdr()  { printf '\n%s%s%s\n' "$T_B" "$*" "$T_0" >&2; }
