#!/usr/bin/env bash

pushd "$(dirname "$0")"

set -e

EXTRA="-Db_pgo=off -Db_lto=false"
NET=

case $1 in
  plain|debug|debugoptimized|release|minsize)
    BUILDTYPE=$1
    shift
    ;;
  pgo)
  # Search for the latest network file to use for the pgo run.
    file -C -m scripts/magic
    for f in * networks/*
    do
      if [ -f "$f" ] && [ "$NET" -ot "$f" ]
      then
        zcat -f "$f" | file -m scripts/magic - | grep -q LeelaChessZero && NET="$f"
      fi
    done
    if [ ! -f "$NET" ]
    then
      echo No network file found!
      exit 0
    fi
    BUILDTYPE=release
    EXTRA="-Db_pgo=generate -Db_lto=true"
    shift
    ;;
  lto)
    BUILDTYPE=release
    EXTRA="-Db_pgo=off -Db_lto=true"
    shift
    ;;
  *)
    BUILDTYPE=release
    ;;
esac

BUILDDIR=build/${BUILDTYPE}

if ! hash meson 2>/dev/null && [ -x ${HOME}/.local/bin/meson ]
then
  export PATH=${PATH}:${HOME}/.local/bin
fi

if [ -f ${BUILDDIR}/build.ninja ]
then
  meson configure ${BUILDDIR} -Dbuildtype=${BUILDTYPE} -Dprefix=${INSTALL_PREFIX:-/usr/local} ${EXTRA} "$@"
else
  meson ${BUILDDIR} --buildtype ${BUILDTYPE} --prefix ${INSTALL_PREFIX:-/usr/local} ${EXTRA} "$@"
fi

cd ${BUILDDIR}

NINJA=$(awk '/ninja/ {ninja=$4} END {print ninja}' meson-logs/meson-log.txt)

# For a pgo run generate the profile data.
if [ -f ../../"$NET" ]
then
  ${NINJA}
  ./lc0 benchmark --weights=../../"$NET" --movetime=30000
  meson configure -Db_pgo=use
fi

if [ -n "${INSTALL_PREFIX}" ]
then
  ${NINJA} install
else
  ${NINJA}
fi

popd
