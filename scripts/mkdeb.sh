#!/bin/sh
# Build charsiu_<version>_arm64.deb from this tree.
#
#   sh scripts/mkdeb.sh [stable|dev]      (default stable; output in build/deb/)
#
# THE PACKAGE IS WHAT THE INSTALLER INSTALLS. The file list lives in one place,
# charsiu-install.sh, and drifted twice when it lived in two (see the probe
# list there). So this builds the tree, lets the installer lay it out under a
# staging --prefix, and wraps that. A .deb and a `curl | sh` install of the
# same commit hold the same files.
#
# THE glibc IT NEEDS IS THE ONE IT WAS BUILT AGAINST. Built on Ubuntu 24.04 the
# binaries ask for GLIBC_2.38 and will not start on Debian bookworm (2.36).
# The release workflow builds in debian:bookworm for that reason; the
# Depends line below is computed from the binaries, so a package built
# somewhere newer says so instead of failing at run time.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CHANNEL=${1:-stable}
case "$CHANNEL" in stable|dev) ;; *) echo "usage: $0 [stable|dev]" >&2; exit 2 ;; esac
command -v dpkg-deb >/dev/null || { echo "mkdeb: dpkg-deb is needed" >&2; exit 1; }

cd "$ROOT"

# vX.Y.Z on this commit is the version; otherwise a snapshot that sorts below
# the first release and in date order among snapshots.
commit=$(git describe --always --dirty --abbrev=12 2>/dev/null \
	|| cat .charsiu-commit 2>/dev/null || echo unknown)
tag=$(git describe --tags --exact-match --match 'v[0-9]*' 2>/dev/null || true)
if [ -n "$tag" ]; then
	VER=${tag#v}
else
	# The commit's own time, so two snapshots of one day still order, and the
	# same commit gives the same version wherever it is packaged.
	when=$(TZ=UTC git log -1 --date=format-local:%Y%m%d%H%M%S --format=%cd 2>/dev/null \
		|| date -u +%Y%m%d%H%M%S)
	VER="0.0~git$when.$(printf '%s' "$commit" | tr -c '0-9a-z\n' '.')"
fi
[ "$CHANNEL" = dev ] && VER="$VER+dev"

make -j"$(nproc 2>/dev/null || echo 2)" all
[ "$CHANNEL" = dev ] && make -j"$(nproc 2>/dev/null || echo 2)" build/charsiu_run_scalar

OUT="$ROOT/build/deb"
STAGE="$OUT/stage"
rm -rf "${STAGE:?}"
mkdir -p "$STAGE/root" "$OUT"

# The installer's own checks run at the end of an install and want an NPU;
# on a build machine they report CRITICAL and that is expected. Its exit
# status is what matters here.
if ! sh scripts/charsiu-install.sh --prefix "$STAGE/root" --no-kernel \
	--no-model --no-demo --no-build --yes "--$CHANNEL" \
	> "$OUT/install.log" 2>&1 </dev/null; then
	echo "mkdeb: the installer failed; see $OUT/install.log" >&2
	exit 1
fi

PKG="$STAGE/root"
mkdir -p "$PKG/opt/charsiu/models" "$PKG/DEBIAN"
# The builder's umask is not the package's. Installed as root:root, a group
# write bit would let any member of group root edit the runtime.
chmod -R u+rwX,go+rX,go-w "$PKG"

# Highest GLIBC_x.y any shipped binary asks for.
glibc=$(find "$PKG" -type f -perm -u+x -exec sh -c \
	'objdump -T "$1" 2>/dev/null | grep -o "GLIBC_[0-9.]*"' _ {} \; \
	| sed 's/GLIBC_//' | sort -t. -k1,1n -k2,2n -k3,3n -u | tail -1)
[ -n "$glibc" ] || { echo "mkdeb: no GLIBC version found in the binaries" >&2; exit 1; }

size=$(du -sk "$PKG" | cut -f1)
cat > "$PKG/DEBIAN/control" <<EOF
Package: charsiu
Version: $VER
Architecture: arm64
Maintainer: Jiaxing Hu <gahing@gahingwoo.com>
Installed-Size: $size
Depends: libc6 (>= $glibc)
Recommends: curl | wget, whiptail
Section: utils
Priority: optional
Homepage: https://github.com/gahingwoo/charsiu
Description: open LLM runtime for the Rockchip RK3576 NPU
 Runs GGUF language models on the RK3576 NPU through the mainline rocket
 driver (/dev/accel), with no vendor runtime in the path. It needs a kernel
 with RK3576 NPU support. Models are not included; fetch one with
 "charsiu pull".
 .
 Built from commit $commit, channel $CHANNEL.
EOF
printf '/etc/charsiu/config.ini\n' > "$PKG/DEBIAN/conffiles"
(cd "$PKG" && find . -path ./DEBIAN -prune -o -type f -print \
	| sed 's|^\./||' | sort | xargs md5sum) > "$PKG/DEBIAN/md5sums"

DEB="$OUT/charsiu_${VER}_arm64.deb"
dpkg-deb --root-owner-group -Zxz --build "$PKG" "$DEB" >/dev/null
echo "$DEB"
echo "  commit $commit, channel $CHANNEL, needs glibc >= $glibc, $(ls "$PKG/usr/bin" | wc -l) commands"
