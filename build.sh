#!/usr/bin/bash

rm -fr build
CC=clang CXX=clang++ meson build --buildtype release # -Db_ndebug=true
# CC=clang CXX=clang++ meson build --buildtype debugoptimized
cd build
ninja
