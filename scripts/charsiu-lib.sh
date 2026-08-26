# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# charsiu-lib.sh: the two things every command needs, written once.
#
# ⚠⚠ WHY THIS FILE EXISTS. Reading a key out of the ini and finding a binary
# were written out again in nearly every script: eight copies of the first and
# seven of the second. Three of the bugs found on the board were a fix applied
# to one copy and not the others -- `charsiu list` reporting every model as "?"
# because its own search predated --prefix, `charsiu serve` dying on a config
# that named a model that was gone because the fallback only landed in the
# runner. A helper with seven copies is not a helper.
#
# Sourced, never run. Nothing here prints or exits.

CHARSIU_HOME="${CHARSIU_HOME:-$HOME/.charsiu}"
CHARSIU_SYS_LIB=/opt/charsiu
CHARSIU_SYS_MODELS=/opt/charsiu/models
CHARSIU_SYS_CONF=/etc/charsiu/config.ini

# --- the config -----------------------------------------------------------
#
# ⚠ WHERE TO READ AND WHERE TO WRITE ARE NOT THE SAME QUESTION, and collapsing
# them is what made the setup screen say "now: (unset)" on a board whose config
# was fine: /etc is not writable by an ordinary user, so the NAME was replaced
# by one in $HOME that did not exist yet.

# charsiu_conf: the config to read. The first that exists, user before system.
charsiu_conf() {
	if [ -n "${CHARSIU_CONFIG:-}" ]; then echo "$CHARSIU_CONFIG"; return; fi
	for c in "$CHARSIU_HOME/config.ini" "$CHARSIU_SYS_CONF"; do
		[ -f "$c" ] && { echo "$c"; return; }
	done
	echo "$CHARSIU_HOME/config.ini"      # where one would be created
}

# charsiu_conf_w: the config to write, seeded from the readable one when the
# readable one cannot be written. Echoes the path; empty if nothing is possible.
charsiu_conf_w() {
	_r=$(charsiu_conf)
	if [ -w "$_r" ] || { [ ! -e "$_r" ] && [ -w "$(dirname "$_r")" ]; }; then
		echo "$_r"; return
	fi
	_u="$CHARSIU_HOME/config.ini"
	if [ ! -f "$_u" ] && [ -f "$_r" ]; then
		mkdir -p "$CHARSIU_HOME" 2>/dev/null && cp "$_r" "$_u" 2>/dev/null || true
	fi
	[ -f "$_u" ] || mkdir -p "$CHARSIU_HOME" 2>/dev/null || true
	echo "$_u"
}

# ini_get SECTION KEY [FILE]. Strips inline comments, tolerates a hand-edited
# file, and says nothing when the file is not there.
ini_get() {
	_f="${3:-${CONF:-$(charsiu_conf)}}"
	[ -f "$_f" ] || return 0
	awk -v s="$1" -v k="$2" '
		/^[ \t]*\[.*\][ \t]*$/ { h=$0; sub(/^[ \t]*\[/,"",h); sub(/\][ \t]*$/,"",h); t=(h==s); next }
		t { if (match($0,/^[ \t]*[^=]*=/)) { n=substr($0,1,RLENGTH-1); gsub(/^[ \t]+|[ \t]+$/,"",n);
		    if (n==k) { v=substr($0,RLENGTH+1); sub(/[ \t]#.*$/,"",v); gsub(/^[ \t]+|[ \t]+$/,"",v); print v; exit } } }
	' "$_f" 2>/dev/null || true
}

# --- finding what was installed -------------------------------------------
#
# ⚠ FIVE PLACES, IN THIS ORDER, AND EVERY COMMAND USES THE SAME FIVE. From
# $PREFIX/usr/bin the staged /opt is TWO levels up, not one; getting that wrong
# is why a --prefix install reported its own binaries missing.
find_bin() {
	for _c in ${CHARSIU_LIB:+"$CHARSIU_LIB/$1"} \
		  "${CHARSIU_HERE:-.}/$1" \
		  "${CHARSIU_HERE:-.}/../build/$1" \
		  "${CHARSIU_HERE:-.}/../../opt/charsiu/$1" \
		  "$CHARSIU_SYS_LIB/$1"; do
		[ -x "$_c" ] && { echo "$_c"; return 0; }
	done
	command -v "$1" 2>/dev/null || true
}

# file_bytes FILE: the size, without reading the file.
#
# ⚠⚠ NOT `wc -c`. Two of these scripts chose wc over stat on purpose, with the
# comment "busybox is often built without the stat applet and `stat -c%s ||
# echo 0` then reports every model as 0 MB". The reasoning was right and the
# conclusion was backwards: BUSYBOX wc -c READS THE WHOLE FILE. GNU coreutils
# fstats it and answers in three syscalls; busybox counts bytes, and measured
# here that is 532831 read() calls and 1.55 seconds for one 2.2 GB model with
# the page cache already warm. On a card it is tens of seconds, per model, and
# it is why charsiu-config's model picker got slow after a `charsiu pull`.
#
# So try stat, and fall back to ls -ln rather than to wc. ls is in every shell
# environment there is, its fifth field is the size whatever the file is
# called, and it never opens the file.
file_bytes() {
	[ -f "$1" ] || { echo 0; return; }
	_sz=$(stat -c %s "$1" 2>/dev/null) || _sz=""
	case "$_sz" in
	''|*[!0-9]*) _sz=$(ls -ln "$1" 2>/dev/null | awk 'NR==1 {print $5}') ;;
	esac
	case "$_sz" in
	''|*[!0-9]*) _sz=0 ;;
	esac
	echo "$_sz"
}

# --- models ---------------------------------------------------------------

# model_dirs: every directory to look in, user first, each at most once.
#
# ⚠ THE SAME DIRECTORY TWICE LISTS EVERY MODEL TWICE, and that is the normal
# case rather than a corner one: the installer and charsiu-serve both set
# CHARSIU_MODELS to /opt/charsiu/models, which is exactly the system directory.
model_dirs() {
	_seen=""
	for _d in "${CHARSIU_MODELS:-$CHARSIU_HOME/models}" "$CHARSIU_SYS_MODELS"; do
		[ -d "$_d" ] || continue
		case " $_seen " in *" $_d "*) continue ;; esac
		_seen="$_seen $_d"
		echo "$_d"
	done
}

# all_models: every gguf that can be seen, user directory first.
all_models() {
	model_dirs | while IFS= read -r _d; do
		for _f in "$_d"/*.gguf; do [ -f "$_f" ] && echo "$_f"; done
	done
	return 0
}

# resolve_model NAME: a path, a bare name, or a name without .gguf.
# ⚠ ALWAYS RETURNS 0. Under set -e an assignment from a failing command
# substitution kills the caller, so a lookup that finds nothing must not be a
# failure -- it is an empty answer.
resolve_model() {
	if [ -f "$1" ]; then echo "$1"; return 0; fi
	model_dirs | while IFS= read -r _d; do
		for _c in "$_d/$1" "$_d/$1.gguf"; do
			if [ -f "$_c" ]; then echo "$_c"; break 2; fi
		done
	done | head -1
	return 0
}

# current_model: what the config names, or empty.
current_model() { ini_get model path; }
