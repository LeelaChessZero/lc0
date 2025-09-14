[![CircleCI](https://circleci.com/gh/LeelaChessZero/lc0.svg?style=shield)](https://circleci.com/gh/LeelaChessZero/lc0)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/3245b83otdee7oj7?svg=true)](https://ci.appveyor.com/project/leelachesszero/lc0)

# Lc0

Lc0 is a UCI-compliant chess engine designed to play chess via neural network, specifically those of the [LeelaChessZero project](https://lczero.org).

## Downloading source

Lc0 can be acquired either via a git clone or an archive download from GitHub.

For essentially all purposes, including selfplay game generation and match play, we highly recommend using the latest `release/version` branch (for example `release/0.32`), which is equivalent to using the latest version tag.

Versioning follows the Semantic Versioning guidelines, with major, minor and patch sections. The training server enforces game quality using the versions output by the client and engine.

Download using git:

```shell
git clone -b release/0.32 https://github.com/LeelaChessZero/lc0.git
```

If you have cloned already an old version, fetch, view and checkout a new branch:
```shell
git fetch --all
git branch --all
git checkout -t remotes/origin/release/0.32
```

If you prefer to download an archive:
 * Download the [.zip](https://api.github.com/repos/LeelaChessZero/lc0/zipball/release/0.32) file ([.tar.gz](https://api.github.com/repos/LeelaChessZero/lc0/tarball/release/0.32) archive is also available)
 * Extract

Having successfully acquired Lc0 via either of these methods, proceed to the build section below and follow the instructions for your OS.


## Building and running Lc0

Building should be easier now than it was in the past. Please report any problems you have.

Building lc0 requires the Meson build system and at least one backend library for evaluating the neural network, as well as a few libraries. If your system already has these libraries installed, they will be used; otherwise Meson will generate its own copy (a "subproject"), which in turn requires that git is installed (yes, separately from cloning the actual lc0 repository). Meson also requires python and Ninja.

Backend support includes (in theory) any CBLAS-compatible library for CPU usage, but OpenBLAS or Intel's DNNL are the main ones. For GPUs, the following are supported: CUDA (with optional cuDNN), various flavors of onnxruntime, and Apple's Metal Performance Shaders. There is also experimental SYCL support for AMD and Intel GPUs.

Finally, lc0 requires a compiler supporting C++20. Minimal versions tested are g++ v10.0, clang v12.0 and Visual Studio 2019 version 16.11.

Given those basics, the OS and backend specific instructions are below.

### Linux

#### Generic

1. Install backend (also read the detailed instructions in later sections):
    - If you want to use NVidia graphics cards Install [CUDA](https://developer.nvidia.com/cuda-zone) (and optionally [cuDNN](https://developer.nvidia.com/cudnn)).
    - If you want to use AMD or Intel graphics cards you can try SYCL.
    - if you want BLAS install either OpenBLAS or DNNL.
2. Install ninja build (`ninja-build`), meson, and (optionally) gtest (`libgtest-dev`).
3. Go to `lc0/`
4. Run `./build.sh`
5. `lc0` will be in `lc0/build/release/` directory
6. Download a [neural network](https://lczero.org/play/networks/bestnets/) in the same directory as the binary (no need to unpack it).

If you want to build with a different compiler, pass the `CC` and `CXX` environment variables:
```shell
CC=clang CXX=clang++ ./build.sh
```

#### Ubuntu 20.04

For Ubuntu 20.04 you need meson, ninja and gcc-10 before performing the steps above. The following should work:
```shell
apt-get update
apt-get -y install git python3-pip gcc-10 g++-10 zlib1g zlib1g-dev
pip3 install meson
pip3 install ninja
CC=gcc-10 CXX=g++-10 INSTALL_PREFIX=~/.local ./build.sh
```

Make sure that `~/.local/bin` is in your `PATH` environment variable. You can now type `lc0 --help` and start.

### Windows

Here are the brief instructions for CUDA/cuDNN, for details and other options see `windows-build.md` and the instructions in the following sections.

1. Install Microsoft Visual Studio (2019 version 16.11 or later)
2. Install [CUDA](https://developer.nvidia.com/cuda-zone)
3. (Optionally install [cuDNN](https://developer.nvidia.com/cudnn)).
4. Install Python3 if you didn't install it with Visual Studio.
5. Install Meson: `pip3 install --upgrade meson`
6. If `CUDA_PATH` is not set (run the `set` command to see the full list of variables), edit `build.cmd` and set the `CUDA_PATH` with your CUDA directory
* If you also want cuDNN, set `CUDNN_PATH` with your cuDNN directory (not needed if it is the same with `CUDA_PATH`).

7. Run `build.cmd`. It will ask permission to delete the build directory, then generate MSVS project and pause.

Then either:

8. Hit `Enter` to build it.
9. Resulting binary will be `build/lc0.exe`

Or.

8. Open generated solution `build/lc0.sln` in Visual Studio and build it yourself.

### Mac

You will need xcode and python3 installed. Then you need to install some required packages through Terminal:

1. Install meson: `pip3 install meson`
2. Install ninja: `pip3 install ninja`

Now download the lc0 source, if you haven't already done so, following the instructions earlier in the page.

3. Go to the lc0 directory.
4. Run `./build.sh -Dgtest=false`

The compiled Lc0 will be in `build/release` 

Starting with v0.32.0, we are also offering a pre-compiled version that can be downloaded from the [release page](https://github.com/LeelaChessZero/lc0/releases).

### CUDA

CUDA can be downloaded and installed following the instructions in from <https://developer.nvidia.com/cuda-downloads>. The build in most cases will pick it up with no further action. However if the cuda compiler (`nvcc`) is not found you can call the build like this: `PATH=/usr/local/cuda/bin:$PATH ./build.sh`, replacing the path with the correct one for `nvcc`.

*Note* that CUDA uses the system compiler and stops if it doesn't recognize the version, even if newer. This is more of an issue with new Linux versions, but you can get around with the `nvcc_ccbin` build option to specify a different compiler just for cuda. As an example, adding `-Dnvcc_ccbin=g++-11` to the build command line will use g++-11 with cuda instead of the system compiler.

### ONNX

Lc0 offers several ONNX based backends, namely onnx-cpu, onnx-cuda, onnx-trt, onnx-rocm and on Windows onnx-dml, utilizing the execution providers offered by onnxruntime.

Some Linux systems are starting to offer onnxruntime packages, so after installing this there is a good chance the Lc0 build will pick it up with no further action required. Otherwise you can set the `onnx_libdir` and `onnx_include` build options to point to the onnxruntime libraries and include directories respectively. The same options are used if you unpack a package downloaded from <https://github.com/microsoft/onnxruntime/releases>.

For Windows, we offer pre-compiled packages for onnx-dml and onnx-trt, see the included README for installation instructions.

### SYCL

*Note* that SYCL support is new in v0.32.0 and as such is still considered experimental.

You will need the Intel "oneAPI DPC++/C++ Compiler", "DPC++ Compatibility Tool" and (for an Intel GPU) "oneAPI Math Kernel Library (oneMKL)" or (for an AMD GPU) hipBLAS.

The Intel tools can be found in either the "oneAPI Base Toolkit" or "C++ Essentials" packages that can be downloaded from
<https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>, while hipBLAS can be downloaded from
<https://rocm.docs.amd.com/projects/hipBLAS/en/latest/>

The compiler for C code is icx and for C++ code is icx on Windows but icpx on Linux.

To build Lc0 with SYCL you need to set the `sycl` build option using `-Dsycl=l0` (that is el zero) for an Intel GPU or `-Dsycl=amd` for (you guessed it) an AMD GPU.

You may also have to set the `dpct_include` option to point to the DPC++ Compatibility Tool includes, the `onemkl_include` similarly for the oneMKL includes, or `hip_libdirs` and `hip_include` to the AMD HIP libraries and includes respectively.

On Linux, a typical session would go like this:
```shell
. /opt/intel/oneapi/setvars.sh --include-intel-llvm
CC=icx CXX=icpx AR=llvm-ar ./build.sh release -Dgtest=false -Dsycl=l0
```
The first line is to initialize the build environment and is only needed once per session, while the build line may need modification as described above.

On windows you will have to build using `ninja`, this is provided by Visual Studio if you install the CMake component. We provide a `build-sycl.cmd` script that should build just fine for an Intel GPU. This script has not yet been tested with and AMD GPU, some editing will be required.

You can also install the [oneAPI DPC++/C++ Compiler Runtime](https://www.intel.com/content/www/us/en/developer/articles/tool/compilers-redistributable-libraries-by-version.html) so you can run Lc0 without needing to initialize the build environment every time.

### BLAS

Lc0 can also run (a bit slow) on CPU, using matrix multiplication functions from a BLAS library. By default OpenBLAS is used if available as it seems to offer good performance on a wide range of processors. If your system doesn't offer an OpenBLAS package (e.g. `libopenblas-dev`), or you have a recent processor you can get DNNL from [here](<https://github.com/uxlfoundation/oneDNN/releases/v2.2>). To use DNNL you have to pass `-Ddnnl=true` to the build and specify the directory where it was installed using the `-Ddnnl_dir=` option. For macs, the Accelerate library will be used.

If the "Intel Implicit SPMD Program Compiler" (`ispc`) is [installed](<https://ispc.github.io/downloads.html>), some performance critical functions will use vectorized code for faster execution. 

*Note* that Lc0 is not able to control the number of threads with all BLAS libraries. Some libraries try to exploit cores aggressively, in which case it may be best to leave the threads set to the default (i.e. automatic) setting.

## Getting help

If there is an issue or the above instructions were not clear, you can always ask for help. The fastest way is to ask in the help channel of our [discord chat](http://lc0.org/chat), but you can also open a [github issue](https://github.com/LeelaChessZero/lc0/issues) (after checking the issue hasn't already been reported).

## Python bindings

Python bindings can be built and installed as follows.

```shell
pip install --user git+https://github.com/LeelaChessZero/lc0.git
```

This will build the package `lczero-bindings` and install it to your Python user install directory.
All the `lc0` functionality related to position evaluation is now available in the module `lczero.backends`.
An example interactive session can be found [here](https://github.com/LeelaChessZero/lc0/pull/1261#issuecomment-622951248).

## License

Leela Chess is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Leela Chess is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

### Additional permission under GNU GPL version 3 section 7

_The source files of Lc0 with the exception of the BLAS, OpenCL and SYCL
backends (all files in the `blas`, `opencl` and `sycl` sub-directories) have
the following additional permission, as allowed under GNU GPL version 3
section 7:_

If you modify this Program, or any covered work, by linking or
combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
modified version of those libraries), containing parts covered by the
terms of the respective license agreement, the licensors of this
Program grant you additional permission to convey the resulting work.

