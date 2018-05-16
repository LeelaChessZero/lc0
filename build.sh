#!/bin/bash

CC=${CC:=clang}
CXX=${CXX:=clang++}

rm -fr build
CC=${CC} CXX=${CXX} meson build --buildtype release # -Db_ndebug=true
# CC=clang CXX=clang++ meson build --buildtype debugoptimized -Db_asneeded=false
# CC=clang CXX=clang++ meson build --buildtype debug
cp testdata/* build
cd build
ninja
