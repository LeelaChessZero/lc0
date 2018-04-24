#!/bin/bash

rm -fr build
CC=clang CXX=clang++ meson build --buildtype release # -Db_ndebug=true
# CC=clang CXX=clang++ meson build --buildtype debugoptimized -Db_asneeded=false
# CC=clang CXX=clang++ meson build --buildtype debug
cp testdata/* build
cd build
ninja
