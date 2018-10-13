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

# Temporary fix for Tensorflow on Darwin until meson find_library() is fixed
if [ `uname` == "Darwin" ] ; then
  if [ -f /usr/local/lib/tensorflow_cc/libtensorflow_cc.so ]; then
    if [ ! -f /usr/local/lib/tensorflow_cc/libtensorflow_cc.dylib ]; then
      ln -s /usr/local/lib/tensorflow_cc/libtensorflow_cc.so /usr/local/lib/tensorflow_cc/libtensorflow_cc.dylib
    fi
  fi
fi


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
