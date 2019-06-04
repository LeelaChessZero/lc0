#!/usr/bin/env bash

pushd "$(dirname "$0")"

set -e

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

if [ -f ${BUILDDIR}/build.ninja ]
then
  meson configure ${BUILDDIR} -Dbuildtype=${BUILDTYPE} -Dprefix=${INSTALL_PREFIX:-/usr/local} "$@"
else
  meson ${BUILDDIR} --buildtype ${BUILDTYPE} --prefix ${INSTALL_PREFIX:-/usr/local} "$@"
fi

cd ${BUILDDIR}

NINJA=$(awk '/ninja/ {ninja=$4} END {print ninja}' meson-logs/meson-log.txt)

if [ -n "${INSTALL_PREFIX}" ]
then
  ${NINJA} install
else
  ${NINJA}
fi

popd
