#!/usr/bin/env bash

set -e

BUILDTYPE=$1

if [ -z "${BUILDTYPE}" ]
then
  BUILDTYPE=release
fi

BUILDDIR=build/${BUILDTYPE}

if [ -d ${BUILDDIR} ]
then
  meson configure ${BUILDDIR} --buildtype ${BUILDTYPE} --prefix ${INSTALL_PREFIX:-/usr/local}
else
  meson ${BUILDDIR} --buildtype ${BUILDTYPE} --prefix ${INSTALL_PREFIX:-/usr/local}
fi

pushd ${BUILDDIR}

if [ -n "${INSTALL_PREFIX}" ]
then
  ninja install
else
  ninja
fi

popd
