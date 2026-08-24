# Suggest the setup once, on the first interactive login, and never again.
#
# ⚠ It does NOT auto-run a full-screen wizard at every login -- that is hostile
# on a board people ssh into to do other things. It runs once, and only on a
# real terminal, and only when charsiu has never been set up.
case "$-" in *i*) ;; *) return 2>/dev/null || true ;; esac
[ -t 0 ] && [ -t 1 ] || return 2>/dev/null || true
[ -e /etc/charsiu/.setup-done ] && return 2>/dev/null || true
command -v charsiu-config >/dev/null 2>&1 || return 2>/dev/null || true
printf '\n  charsiu is not set up yet. Run %scharsiu-config --setup%s\n\n' \
       "$(printf '\033[1m')" "$(printf '\033[0m')"
