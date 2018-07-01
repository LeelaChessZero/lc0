## Windows BLAS/OpenCL

0. [Install Microsoft Visual Studio](https://visualstudio.microsoft.com/). For VS2017 make sure the
   option "Desktop development with C++" is installed (you can add it later if not).

1. [Install git for windows](https://git-scm.com/download/win) - this can be used to get lc0 but is also
   needed for meson.

2. Install a BLAS library. This can be either OpenBLAS or Intel MKL.
*  For [OpenBLAS go here](http://www.openblas.net/), you need a binary package with a filename of the
   form `OpenBLAS-version-Win64-int32.zip`, they are not available for all versions, which you just unpack
   at a location of your choise (but not inside the lc0 directory).
*  For [Intel MKL go here](https://software.intel.com/en-us/mkl), where you need to register. After
   installation don't forget to run `mklvars.bat intel64` to set up the paths to the dlls.

3. For OpenCL you also need to install OpenCL developer libraries.
*  For AMD cards the AMD APP SDK 3.0 seems to be the appropriate one, to be installed after the card drivers.
   This is not currently available on the AMD website, but links to a signed installer are available in the
   [AMD community forum](https://community.amd.com/thread/222855).
*  For nVIDIA cards you probably need the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads).

4. [Install Python3](https://www.python.org/) - be sure to check the box to add python to the path.

5. Install Meson: `pip3 install --upgrade meson`

6. Edit `build-cl.cmd`:
*  If you use MSVS other than 2017 community edition (or if it's installed into non-standard location)
   replace the path to vcvarsall.bat and MSBuild.exe. If you can't find vcvarsall.bat on VS2017, you
   need to install option "Desktop development with C++". Some example paths are in comments.
*  In `--backend 2017` replace 2017 with the correct MSVS version.
*  Set the BLAS (and optionally OpenCL) library `include` and `lib` directories in the appropriate
   variables.
    - For OpenBLAS, they are `openblas_include` and `openblas_libdirs`.
    - For Intel MKL, they are `mkl_include` and `mkl_libdirs`. The `lib` directory typically ends in
      `\lib\intel64`.
    - For OpenCL, they are `opencl_libdirs` and `opencl_include`. The include directory is the one with
      the `CL` directory containing `opencl.h`, not directly the one containing `opencl.h`.

7. Run `build-cl.cmd`. It will generate MSVS project and pause.

8. Hit `Enter` to build it.

9. Resulting binary will be `build/lc0.exe`

Alternatively you can

8. open generated solution `build/lc0.sln` in Visual Studio and build yourself.


