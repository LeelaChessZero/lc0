## Windows build

0. [Install Microsoft Visual Studio](https://visualstudio.microsoft.com/) (2017 or later). Make sure
   the option "Desktop development with C++" is selected (you can add it later if not).

1. [Install git for windows](https://git-scm.com/download/win) - this can be used to get lc0 but is also
   needed for meson. If you haven't downloaded lc0, you can do it now following the instructions in
   the `README`(https://github.com/LeelaChessZero/lc0/blob/master/README.md).

2. GPU users with nVIDIA cards (and "compute capability" 3.0 or higher) can build with CUDA/CuDNN.
*  Install [CUDA](https://developer.nvidia.com/cuda-zone) (v10.0 is fine for Visual Studio 2017, newer is
   needed for Visual Studio 2019) and then
*  install the appropriate [cuDNN](https://developer.nvidia.com/cudnn).

3. GPU users with recent Windows 10 installations can build with DirectX 12, this only requires updated
   SDK headers (that may already be available in Visual Studio).

4. CPU users may want to install a BLAS library. This can be either OpenBLAS, Intel MKL or Intel DNNL.
   This is optional since the Eigen library can be used without installing anything, but probably with
   worse performance.
*  For [OpenBLAS go here](http://www.openblas.net/), you need a binary package with a filename of the
   form `OpenBLAS-version-Win64-int32.zip`, they are not available for all versions, which you just unpack
   at a location of your choise (but not inside the lc0 directory).
*  For [Intel MKL go here](https://software.intel.com/en-us/mkl), where you need to register. After
   installation don't forget to run `mklvars.bat intel64` to set up the paths to the dlls.
*  For [Intel DNNL go here](https://github.com/intel/mkl-dnn/releases). Note that not all releases have
   binaries available, you want `dnnl_win_*_cpu_vcomp.zip`.

5. For OpenCL you also need to install OpenCL developer libraries.
*  For AMD cards the AMD APP SDK 3.0 seems to be the appropriate one, to be installed after the card drivers.
   This is not currently available on the AMD website, but links to a signed installer are available in the
   [AMD community forum](https://community.amd.com/thread/222855).
*  For nVIDIA cards it is included in the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads).

6. [Install Python3](https://www.python.org/) - be sure to check the box to add python to the path.

7. Install Meson: `pip3 install --upgrade meson`

8. Edit `build.cmd`:
*  At the top, set to `true` and `false` the variables for the backends you want to build.
*  Then set the paths for the build dependencies.
    - Note: for `OPENCL_INCLUDE_PATH` you don't want the directory containing `opencl.h`, but one level higher
    (the one containing `CL`).

9. Run `build.cmd`. It will ask permission to delete the build directory, then generate MSVS project and
   pause.

10. Hit `Enter` to build it.

11. Resulting binary will be `build/lc0.exe`

Alternatively you can

10. open generated solution `build/lc0.sln` in Visual Studio and build yourself.


### Troubleshooting

If you get something like

   Downloading zlib patch from https://wrapdb.mesonbuild.com/v1/projects/zlib/1.2.11/4/get_zip
   A fallback URL could be specified using patch_fallback_url key in the wrap file

   meson.build:604:4: ERROR: WrapDB connection failed to https://wrapdb.mesonbuild.com/v1/projects/zlib/1.2.11/4/get_zip with error <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1108)>

when you run build.cmd, then download the

   https://wrapdb.mesonbuild.com/v1/projects/zlib/1.2.11/4/get_zip

file in your browser, and place the file in the lc0\subprojects\packagecache folder. then Remove the

   lc0\subprojects\zlib-1.2.11

folder, and run build again.
