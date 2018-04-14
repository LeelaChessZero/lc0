#!/usr/bin/bash

rm -fr build
CC=clang CXX=clang++ meson build --buildtype release
# CC=clang CXX=clang++ meson build --buildtype debug
# CC=clang CXX=clang++ meson build --buildtype debugoptimized
cd build
ninja
