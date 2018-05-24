# Building and running Lc0

Building right now may be a bit rough, we are improving and finding ways to simplifying the build process.

## Linux (generic)

1. (if you want version with tensorflow) Install `tensorflow_cc` by following steps described [here](https://github.com/FloopCZ/tensorflow_cc).
2. (if you want cuDNN version) Install CUDA and cuDNN.
3. Install ninja build (`ninja-build`), meson, and (optionally) gtest (`libgtest-dev`).
4. Go to lc0/
5. Run ./build.sh
6. `lc0` will be in `lc0/build/release/` directory

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

0. Install Microsoft Visual Studio
1. Install CUDA (v9.2 is fine)
2. Install cuDNN.
3. Install Python3
4. Install Meson: `pip3 install --upgrade meson`
5. Edit `build-cuda.cmd`:

* If you use MSVS other than 2015 (or if it's installed into non-standard location):
    * `C:\Program Files (x86)\Microsoft Visual Studio 14.0\` replace 14.0 with your version
    * `--backend 2015` replace 2015 with your version
* `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\lib\x64` replace with your CUDA path
* `C:\dev\cuDNN\` replace with your cuDNN directory

6. Run `build-cuda.cmd`. It will generate MSVS project and pause.

Then either:

7. Hit <Enter> to build it.
8. Resulting binary will be `build/lc0.exe`

Or.

7. Opten generated solution `build/lc0.sln` in Visual Studio and build yourself.