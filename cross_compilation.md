
# Cross compilation for Android, Linux ARM, etc.

## Pre-requisites

You'll need a toolchain like NDK or GNU ARM. A set of cross-files is provided, you can choose one of them or create your own. These are the platforms currently tested:

* Android aarch64
* Android armv7-a
* Linux ARM

## Build

Once the toolchain is installed and a Meson cross file is chosen, to build simply run:

`CC=clang CXX=clang++ ./build.sh --cross-file cross-files/arm-linux-androideabi`

You might have to remove the build/ directory if it exists.

