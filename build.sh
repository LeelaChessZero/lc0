#!/usr/bin/env bash

set -e

# Move to this script's directory.
CDPATH= cd -- "$(dirname -- "$0")"

case $1 in
  plain|debug|debugoptimized|release|minsize)
    BUILDTYPE=$1
    shift
    ;;
  *)
    BUILDTYPE=release
    ;;
esac

BUILDDIR=build/${BUILDTYPE}

MESON=$(PATH="${PATH}:${HOME}/.local/bin" command -v meson || :)
MESON=${MESON:?"Could not find meson. Is it installed and in PATH?"}

if [ -f "${BUILDDIR}/build.ninja" ]
then
  "${MESON}" configure "${BUILDDIR}" -Dbuildtype="${BUILDTYPE}" -Dprefix="${INSTALL_PREFIX:-/usr/local}" "$@"
else
  "${MESON}" "${BUILDDIR}" --buildtype "${BUILDTYPE}" --prefix "${INSTALL_PREFIX:-/usr/local}" "$@"
fi

"${MESON}" compile -C "${BUILDDIR}"

if [ -n "${INSTALL_PREFIX}" ]
then
  "${MESON}" install -C "${BUILDDIR}"
fi
