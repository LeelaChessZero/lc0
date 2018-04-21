#!/usr/bin/bash

rm -fr build
env CC=clang CXX=clang++ meson build --buildtype release # -Db_ndebug=true
# CC=clang CXX=clang++ meson build --buildtype debugoptimized -Db_asneeded=false
# CC=clang CXX=clang++ meson build --buildtype debug
cd build
ninja
