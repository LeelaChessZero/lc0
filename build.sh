#!/usr/bin/env bash

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

if [ -d ${BUILDDIR} ]
then
  meson configure ${BUILDDIR} --buildtype ${BUILDTYPE} --prefix ${INSTALL_PREFIX:-/usr/local} "$@"
else
  meson ${BUILDDIR} --buildtype ${BUILDTYPE} --prefix ${INSTALL_PREFIX:-/usr/local} "$@"
fi

pushd ${BUILDDIR}

if [ -n "${INSTALL_PREFIX}" ]
then
  ninja install
else
  ninja
fi

popd
