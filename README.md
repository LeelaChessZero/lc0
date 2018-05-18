# Building and running Lc0

Building right now is still a bit rough, we are still improving and finding ways to simplifying the build process.

## Linux

1. (if you want version with tensorflow) Install `tensorflow_cc` by following steps described [here](https://github.com/FloopCZ/tensorflow_cc).
2. (if you want cuDNN version) Install CUDA and cuDNN.
3. Install ninja build (`ninja-build`), meson, and (optionally) gtest (`libgtest-dev`).
4. Go to lc0/
5. If you decided not to install tensorflow or CUDA, comment out building `network_tf.cc` and/or `network_cudnn.cu` from meson.build.
6. Run ./build.sh

If you want to build with a different compiler, pass the `CC` and `CXX` environment variables:

    CC=clang-6.0 CXX=clang++-6.0 ./build.sh

### Ubuntu 16.04

For Ubuntu 16.04 you need the latest version of meson and clang-6.0 before performing the steps above:

    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install clang-6.0 ninja-build
    pip3 install meson --user
    CC=clang-6.0 CXX=clang++-6.0 INSTALL_PREFIX=~/.local ./build.sh

Make sure that `~/.local/bin` is in your `PATH` environment variable. You can now type `lc0 --help` and start.


## Windows

Building for windows is currently complicated, you can try executeing steps listed [here](https://github.com/glinscott/leela-chess/issues/334#issuecomment-382848569).

Alternatively, you can use pre-built binary:

1. Install CUDA v9.0 (not 9.1)
2. Install cuDNN for CUDA v9.0
3. Take latest binary from [here](https://crem.xyz/lc0/)

For the cudnn (non-TF) version, building is much simpler. Install latest CUDA toolkit and compatible cuDNN, and build using the visual studio project.
